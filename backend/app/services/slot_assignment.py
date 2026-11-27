"""Assigning a spool to an AMS slot, for both inventories (#3756).

The routes ``POST /inventory/assignments`` (built-in inventory) and
``POST /spoolman/inventory/slot-assignments`` (Spoolman) call these, and so
can code that has no request of its own. Behaviour is the routes' own, moved
here unchanged: the same checks, the same HTTP errors, the same lock order
and the same MQTT push.

The printer manager, the websocket manager and the route helpers the tests
replace by name are passed in by the routes, so a test that patches them in
the route module still reaches the code here. A caller without a request
leaves them out and gets the real ones.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import HTTPException
from sqlalchemy import delete, select, text, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.api.routes._spoolman_helpers import MappedSpoolFields, _map_spoolman_spool
from backend.app.core.printer_scope import PrinterScope
from backend.app.models.printer import Printer
from backend.app.models.spool import Spool
from backend.app.models.spool_assignment import SpoolAssignment
from backend.app.models.spoolman_k_profile import SpoolmanKProfile
from backend.app.models.spoolman_slot_assignment import SpoolmanSlotAssignment
from backend.app.models.user import User
from backend.app.services import slot_unlink_grace, spool_swaps
from backend.app.services.ams_assignment_lock import get_ams_assignment_lock, get_spool_move_lock
from backend.app.services.ams_slot_presence import spool_present
from backend.app.services.slicer_filament_resolver import resolve_slicer_filament, reusable_slot_filament_id
from backend.app.services.slot_nozzle import resolve_slot_nozzle
from backend.app.services.spool_filament_preset import resolve_spoolman_preset
from backend.app.services.spoolman import (
    SpoolmanClient,
    SpoolmanClientError,
    SpoolmanNotFoundError,
    SpoolmanUnavailableError,
)
from backend.app.utils.filament_finish import finish_generic_id
from backend.app.utils.filament_ids import (
    GENERIC_FILAMENT_IDS,
    filament_id_to_setting_id,
    normalize_slicer_filament,
)
from backend.app.utils.filament_types import nozzle_temp_range, printer_filament_type, support_filament_type
from backend.app.utils.tray_ids import external_tray_id

if TYPE_CHECKING:
    from backend.app.core.websocket import ConnectionManager
    from backend.app.services.printer_manager import PrinterManager

logger = logging.getLogger(__name__)


@dataclass
class LocalSlotAssignment:
    """What assigning a built-in inventory spool did.

    ``assignment`` is loaded with its spool and printer, ready for
    ``SpoolAssignmentResponse``. ``configured`` is True when the MQTT push went
    out; ``pending_config`` when it waits for the AMS to report the spool.
    """

    assignment: SpoolAssignment
    configured: bool
    pending_config: bool


async def _announce_slots(ws_manager: ConnectionManager, slots: list[tuple[int, int, int]]) -> None:
    for printer_id, ams_id, tray_id in slots:
        await ws_manager.broadcast(
            {"type": "spool_assignment_changed", "printer_id": printer_id, "ams_id": ams_id, "tray_id": tray_id}
        )


async def _printers_holding_spool(db: AsyncSession, spool_id: int, printer_scope: PrinterScope) -> set[int]:
    """The printers in the caller's scope with a slot assigned to the spool."""
    held = await db.execute(select(SpoolAssignment.printer_id).where(SpoolAssignment.spool_id == spool_id).distinct())
    return {pid for pid in held.scalars() if printer_scope.allows(pid)}


async def assign_local_spool(
    db: AsyncSession,
    *,
    spool_id: int,
    printer_id: int,
    ams_id: int,
    tray_id: int,
    move: bool,
    printer_scope: PrinterScope,
    preset_user: User | None,
    ws_manager: ConnectionManager | None = None,
    apply_mqtt: Callable[..., Awaitable[bool]] | None = None,
) -> LocalSlotAssignment:
    """Assign a built-in inventory spool to an AMS slot and auto-configure it via MQTT.

    The caller has checked that ``printer_id`` is in ``printer_scope``.
    ``preset_user`` is the user whose cloud sign-in resolves the spool's
    preset (#3519), and is recorded as the one who assigned it.
    """
    from backend.app.api.routes.inventory import _find_tray_in_ams_data, spool_response_loads  # noqa: PLC0415
    from backend.app.services.printer_manager import printer_manager  # noqa: PLC0415

    if ws_manager is None:
        from backend.app.core.websocket import ws_manager  # noqa: PLC0415
    if apply_mqtt is None:
        from backend.app.api.routes.inventory import apply_spool_to_slot_via_mqtt as apply_mqtt  # noqa: PLC0415

    # 1. Validate spool exists and is not archived
    result = await db.execute(select(Spool).options(*spool_response_loads()).where(Spool.id == spool_id))
    spool = result.scalar_one_or_none()
    if not spool:
        raise HTTPException(404, "Spool not found")
    if spool.archived_at:
        raise HTTPException(400, "Cannot assign an archived spool")

    # 2. Get current AMS tray state for fingerprint + existing filament ID.
    # tray_state: Bambu firmware reports 11=loaded, 9=empty, 10=spool present
    # but filament not in feeder. Captured here so the empty-slot heuristic
    # below can prefer it over tray_type — a manual "Reset slot" clears
    # tray_type to "" while leaving state at 11 (filament still physically
    # present), which would otherwise mislead the heuristic into the
    # pending-config branch and skip MQTT forever (#1228 follow-up).
    fingerprint_color = None
    fingerprint_type = None
    current_tray_info_idx = ""
    current_tray_sub_brands = ""
    tray_state: int | None = None
    # Firmware's tray_exist_bits answer for this slot, when the payload carries
    # one. Outranks tray_state below — see services/ams_slot_presence.py.
    tray_has_spool: bool | None = None
    state = printer_manager.get_status(printer_id)
    if state and state.raw_data:
        if ams_id == 255:
            # External slot: look up tray from vt_tray by global ID
            vt_tray = state.raw_data.get("vt_tray") or []
            ext_id = external_tray_id(tray_id)  # 0→254, 1→255, rack 2-8→247-253
            for vt in vt_tray:
                if isinstance(vt, dict) and int(vt.get("id", 254)) == ext_id:
                    fingerprint_color = vt.get("tray_color", "")
                    fingerprint_type = vt.get("tray_type", "")
                    current_tray_info_idx = vt.get("tray_info_idx", "")
                    current_tray_sub_brands = vt.get("tray_sub_brands", "")
                    raw_state = vt.get("state")
                    if isinstance(raw_state, int):
                        tray_state = raw_state
                    tray_has_spool = spool_present(vt)
                    break
        else:
            ams_data = state.raw_data.get("ams", {})
            ams_list = (
                ams_data.get("ams", [])
                if isinstance(ams_data, dict)
                else ams_data
                if isinstance(ams_data, list)
                else []
            )
            tray = _find_tray_in_ams_data(
                ams_list,
                ams_id,
                tray_id,
            )
            if tray:
                fingerprint_color = tray.get("tray_color", "")
                fingerprint_type = tray.get("tray_type", "")
                current_tray_info_idx = tray.get("tray_info_idx", "")
                current_tray_sub_brands = tray.get("tray_sub_brands", "")
                raw_state = tray.get("state")
                if isinstance(raw_state, int):
                    tray_state = raw_state
                tray_has_spool = spool_present(tray)

    # 3. Upsert assignment (replace if same printer+ams+tray), under the lock
    # the AMS change handler writes the printer's slots with.
    #
    # A move (#1574) also takes the spool out of every other slot it holds, in
    # the same commit, so a failed move leaves it where it was instead of in no
    # slot. Moves of one spool take turns (get_spool_move_lock): two at once
    # would each read where the spool is before the other committed, and both
    # assign it. The slots' printers are locked as well, in id order so two
    # moves can't take the same two locks the other way round. Which printers
    # those are is read again once the locks are held: a move served first, a
    # plain assign or an AMS change may have put the spool on one more, and
    # then this one lets go and locks the larger set. A slot on a printer
    # outside the caller's scope stays assigned, as unassigning leaves it
    # (#1727).
    locked_printer_ids = {printer_id}
    if move:
        locked_printer_ids |= await _printers_holding_spool(db, spool_id, printer_scope)
    vacated: list[tuple[int, int, int]] = []
    while True:
        async with AsyncExitStack() as locks:
            if move:
                await locks.enter_async_context(get_spool_move_lock("inventory", spool_id))
            for pid in sorted(locked_printer_ids):
                await locks.enter_async_context(get_ams_assignment_lock(pid))
            if move:
                holding = await _printers_holding_spool(db, spool_id, printer_scope)
                if not holding <= locked_printer_ids:
                    locked_printer_ids |= holding
                    continue
                others = await db.execute(
                    select(SpoolAssignment).where(
                        SpoolAssignment.spool_id == spool_id,
                        SpoolAssignment.printer_id.in_(locked_printer_ids),
                    )
                )
                for row in others.scalars().all():
                    if (row.printer_id, row.ams_id, row.tray_id) != (printer_id, ams_id, tray_id):
                        vacated.append((row.printer_id, row.ams_id, row.tray_id))
                        await db.delete(row)
            existing = await db.execute(
                select(SpoolAssignment).where(
                    SpoolAssignment.printer_id == printer_id,
                    SpoolAssignment.ams_id == ams_id,
                    SpoolAssignment.tray_id == tray_id,
                )
            )
            old = existing.scalar_one_or_none()
            if old:
                await db.delete(old)
                await db.flush()

            assignment = SpoolAssignment(
                spool_id=spool_id,
                printer_id=printer_id,
                ams_id=ams_id,
                tray_id=tray_id,
                fingerprint_color=fingerprint_color,
                fingerprint_type=fingerprint_type,
                assigned_by_user_id=preset_user.id if preset_user is not None else None,
            )
            db.add(assignment)
            await spool_swaps.record_assignment(db, printer_id, ams_id, tray_id, spool_id, spool_swaps.INVENTORY)
            await db.commit()
        break
    await db.refresh(assignment)
    slot_unlink_grace.forget_slot(printer_id, ams_id, tray_id)
    if vacated:
        logger.info(
            "Moved spool %d to printer %d AMS%d-T%d from %s",
            spool.id,
            printer_id,
            ams_id,
            tray_id,
            ", ".join(f"printer {p} AMS{a}-T{t}" for p, a, t in vacated),
        )

    # 4. Auto-configure AMS slot via MQTT.
    #
    # Only suppress the publish when the firmware's *explicit* empty signal
    # (state ∈ {9, 10}) is set — "no spool" / "spool present but no feed".
    # Every other state, including state=3 (the default idle on A1 Mini BMCU /
    # P1S Standard AMS for both loaded and unconfigured slots) and missing
    # state (older firmwares), is treated as the user's assertion that a
    # spool is in the slot and we attempt the MQTT push.
    #
    # The pre-existing "skip when slot looks empty" guard read state=3 +
    # tray_type="" as "empty" and skipped MQTT. On these firmwares that
    # combination is the post-"Reset Slot" state with the spool still
    # physically inserted — there is NO AMS signal that distinguishes it
    # from a truly-empty slot, so the guard created a deadlock: MQTT never
    # fired, the AMS never reported any change (because nothing changed
    # physically), and on_ams_change replay therefore never re-fired the
    # config either. Reporter (#1322 follow-up by @RosdasHH) verified
    # empirically that removing the guard makes the slot configure
    # correctly because Bambu firmware DOES accept the push for a
    # physically-loaded slot, even when tray_type is "" and state is 3.
    #
    # Trade-off for the truly-empty slot case: firmware drops the push
    # silently (per Bambu's documented behavior), the SpoolAssignment row
    # still has empty fingerprint_type because nothing in the assign path
    # updates that column, and on_ams_change at main.py:1031-1054 still
    # fires the deferred config when a spool eventually appears. So the
    # SpoolBuddy weigh-then-assign-before-insert workflow continues to
    # work — just without the optimization of skipping a no-op MQTT call.
    #
    # state ∈ {9, 10} stays as an explicit short-circuit so we don't churn
    # a doomed MQTT push when the firmware has positively confirmed "no
    # spool" — and to keep the on_ams_change replay path as the single
    # source of truth for those slots.
    #
    # ...except that `state` cannot carry that meaning. Two independent ways
    # a loaded slot reads 9 here:
    #
    #   - an AMS-HT reports its LOADED tray as 9, not 11, because it does not
    #     feed into a shared buffer the way a 4-slot AMS does (#2594, and the
    #     merge above skips its own state heuristic for HT units for exactly
    #     this reason). So this branch called every HT slot empty on sight.
    #   - apply_tray_exist_bits stamps state=9 on any slot whose tray_exist_bits
    #     bit is 0 and never takes it back when the bit returns, so a slot that
    #     was briefly emptied keeps the 9 until something configures it.
    #
    # Either way the slot sits at exists=True, state=9, this branch took the
    # pending path, nothing was published, and the printer kept showing "?"
    # (#3084 — reported against an H2C's AMS-HT, where both apply). Firmware's
    # presence bit is what actually answers "is a spool in this slot", and the
    # printer card has read it ahead of `state` since #2527.
    #
    # It is allowed to overrule the 9 and nothing else. A bit reading *empty*
    # deliberately does NOT start suppressing pushes that go out today: the
    # cost of being wrong there is a slot that silently stops configuring, on
    # whichever AMS variant we compute the bit position wrong for, against a
    # saving of one MQTT message the firmware would have dropped anyway.
    slot_is_definitely_empty = tray_has_spool is not True and (tray_state == 9 or tray_state == 10)
    configured = False
    if not slot_is_definitely_empty:
        try:
            configured = await apply_mqtt(
                db=db,
                current_user=preset_user,
                spool=spool,
                printer_id=printer_id,
                ams_id=ams_id,
                tray_id=tray_id,
                current_tray_info_idx=current_tray_info_idx,
                current_tray_type=fingerprint_type or "",
                current_tray_sub_brands=current_tray_sub_brands,
            )
        except Exception as e:
            logger.warning("MQTT auto-configure failed for spool %d: %s", spool.id, e)
        else:
            # Nudge a fresh pushall so the read-back verification registered in
            # apply_spool_to_slot_via_mqtt (#2582) has current tray telemetry to
            # compare against within its window, instead of waiting for the next
            # idle push. Best-effort — the periodic push is the fallback.
            if configured:
                try:
                    client = printer_manager.get_client(printer_id)
                    if client:
                        client.request_status_update()
                except Exception:
                    pass
    # pending_config is the "config not landed yet" UI marker. True when the
    # firmware said empty, OR when MQTT couldn't actually publish (printer
    # offline, no client, transient failure). on_ams_change replay re-fires
    # the config in either case once the AMS reports a non-empty fingerprint.
    pending_config = slot_is_definitely_empty or not configured

    result = await db.execute(
        select(SpoolAssignment)
        .options(
            selectinload(SpoolAssignment.spool).options(*spool_response_loads()),
            selectinload(SpoolAssignment.printer),
        )
        .where(SpoolAssignment.id == assignment.id)
    )
    loaded = result.scalar_one()

    if pending_config:
        logger.info(
            "Pre-configured assignment: spool %d → printer %d AMS%d-T%d (slot empty, will configure on insert)",
            spool.id,
            printer_id,
            ams_id,
            tray_id,
        )

    await _announce_slots(ws_manager, [*vacated, (printer_id, ams_id, tray_id)])

    return LocalSlotAssignment(assignment=loaded, configured=configured, pending_config=pending_config)


def _live_slot_tray(state, ams_id: int, tray_id: int) -> dict:
    """The tray the printer reports for a slot right now, or {} (external: vt_tray 254/255)."""
    raw = getattr(state, "raw_data", None) if state is not None else None
    if not isinstance(raw, dict):
        return {}
    if ams_id == 255:
        vt_trays = raw.get("vt_tray") or []
        for vt in [vt_trays] if isinstance(vt_trays, dict) else vt_trays:
            if isinstance(vt, dict) and str(vt.get("id", 254)) == str(external_tray_id(tray_id)):
                return vt
        return {}
    units = raw.get("ams") or []
    if isinstance(units, dict):
        units = units.get("ams") or []
    for unit in units if isinstance(units, list) else []:
        if isinstance(unit, dict) and str(unit.get("id")) == str(ams_id):
            for tray in unit.get("tray") or []:
                if isinstance(tray, dict) and str(tray.get("id")) == str(tray_id):
                    return tray
    return {}


async def assign_spoolman_spool(
    db: AsyncSession,
    client: SpoolmanClient,
    *,
    spoolman_spool_id: int,
    printer_id: int,
    ams_id: int,
    tray_id: int,
    move: bool,
    printer_scope: PrinterScope,
    current_user: User | None,
    printer_manager: PrinterManager | None = None,
    ws_manager: ConnectionManager | None = None,
    clear_vacated_fallback_tag: Callable[..., Awaitable[str | None]] | None = None,
) -> MappedSpoolFields:
    """Assign a Spoolman spool to an AMS slot (stored in the local DB only) and auto-configure it via MQTT.

    Raises 404 if the printer does not exist or the spool is not found in
    Spoolman. Spoolman's own ``spool.location`` field is NOT touched; it is
    user-managed. The caller has checked that ``printer_id`` is in
    ``printer_scope``. Returns the spool's mapped fields.
    """
    from backend.app.api.routes.inventory import _find_tray_in_ams_data  # noqa: PLC0415
    from backend.app.api.routes.spoolman_inventory import (  # noqa: PLC0415
        _clear_stale_slot_fallback_tag_links,
        _clear_vacated_slot_fallback_tag,
        _translate_spoolman_errors,
    )

    if printer_manager is None:
        from backend.app.services.printer_manager import printer_manager  # noqa: PLC0415
    if ws_manager is None:
        from backend.app.core.websocket import ws_manager  # noqa: PLC0415
    if clear_vacated_fallback_tag is None:
        clear_vacated_fallback_tag = _clear_vacated_slot_fallback_tag

    result = await db.execute(select(Printer).where(Printer.id == printer_id))
    printer = result.scalar_one_or_none()
    if not printer:
        raise HTTPException(status_code=404, detail="Printer not found")

    # Verify the Spoolman spool exists before committing to local DB.
    # This prevents ghost rows pointing at non-existent spool IDs.
    async with _translate_spoolman_errors():
        spool = await client.get_spool(spoolman_spool_id)
    # As the built-in inventory refuses it: an archived spool on a slot would
    # be charged the next print's usage (SSA-8).
    if spool.get("archived"):
        raise HTTPException(status_code=400, detail="Cannot assign an archived spool")

    # A move (#1574) takes the spool out of every other slot it holds, in the
    # same commit as the upsert below, so a failed move leaves it where it was
    # instead of in no slot. Its fallback tag for a slot it leaves is cleared
    # first. Moves of one spool take turns (get_spool_move_lock): on PostgreSQL
    # a move's DELETE doesn't see the slot another move inserts and commits
    # after it began, and both left the spool assigned. A slot on a printer
    # outside the caller's scope stays assigned, as unassigning leaves it
    # (#1727).
    target = (printer_id, ams_id, tray_id)
    slot_columns = (SpoolmanSlotAssignment.printer_id, SpoolmanSlotAssignment.ams_id, SpoolmanSlotAssignment.tray_id)
    others = [SpoolmanSlotAssignment.spoolman_spool_id == spoolman_spool_id, tuple_(*slot_columns) != target]
    if (in_scope := printer_scope.where_strict(SpoolmanSlotAssignment.printer_id)) is not None:
        others.append(in_scope)
    vacated: list[tuple[int, int, int]] = []
    cleared_tag: str | None = None
    async with get_spool_move_lock("spoolman", spoolman_spool_id) if move else nullcontext():
        if move:
            vacated = list((await db.execute(select(*slot_columns).where(*others))).tuples())
            cleared_tag = await clear_vacated_fallback_tag(
                client, db, spool_id=spoolman_spool_id, spool=spool, vacated=vacated
            )

        # Spool confirmed in Spoolman — upsert into local slot-assignment table
        # assigned_at is intentionally not refreshed on re-assign (original timestamp preserved)
        try:
            if move:
                # Every other slot of the spool, not only those read above: a
                # link or the AMS sync may have given it one more meanwhile.
                result = await db.execute(delete(SpoolmanSlotAssignment).where(*others).returning(*slot_columns))
                vacated = list(result.tuples())
            await db.execute(
                text(
                    "INSERT INTO spoolman_slot_assignments"
                    " (printer_id, ams_id, tray_id, spoolman_spool_id)"
                    " VALUES (:printer_id, :ams_id, :tray_id, :spool_id)"
                    " ON CONFLICT(printer_id, ams_id, tray_id)"
                    " DO UPDATE SET spoolman_spool_id = excluded.spoolman_spool_id"
                ),
                {
                    "printer_id": printer_id,
                    "ams_id": ams_id,
                    "tray_id": tray_id,
                    "spool_id": spoolman_spool_id,
                },
            )
            await spool_swaps.record_assignment(
                db, printer_id, ams_id, tray_id, spoolman_spool_id, spool_swaps.SPOOLMAN
            )
            await db.commit()
            slot_unlink_grace.forget_slot(printer_id, ams_id, tray_id)
        except Exception as exc:
            await db.rollback()
            logger.error("Failed to persist slot assignment: %s", exc)
            if cleared_tag:
                # The spool stays in its old slot, so it gets that slot's tag back.
                try:
                    await client.merge_spool_extra(spoolman_spool_id, {"tag": json.dumps(cleared_tag)})
                except (SpoolmanClientError, SpoolmanUnavailableError, SpoolmanNotFoundError) as restore_exc:
                    logger.warning(
                        "Could not put back the fallback tag of Spoolman spool %s: %s",
                        spoolman_spool_id,
                        restore_exc,
                    )
            raise HTTPException(status_code=500, detail="Failed to save slot assignment") from exc
    if vacated:
        logger.info(
            "Moved Spoolman spool %d to printer %d AMS%d-T%d from %s",
            spoolman_spool_id,
            printer_id,
            ams_id,
            tray_id,
            ", ".join(f"printer {p} AMS{a}-T{t}" for p, a, t in vacated),
        )
    # #1457: clear stale fallback-tag links on OTHER spools still bound to this
    # slot. Without this, a non-RFID slot's deterministic fallback tag stays
    # attached to the previous spool in Spoolman's extra.tag and re-surfaces in
    # the hover card whenever the local slot assignment is removed.
    if printer.serial_number:
        await _clear_stale_slot_fallback_tag_links(
            client,
            printer_serial=printer.serial_number,
            ams_id=ams_id,
            tray_id=tray_id,
            keep_spool_id=spoolman_spool_id,
        )

    # Other tabs and the SpoolBuddy kiosk follow slot changes over the
    # websocket, as the built-in assign announces them: the slots the spool
    # left, and the one it went to. After the tag clean-up above, which the
    # pages read back with the slot rows (#3402).
    await _announce_slots(ws_manager, [*vacated, target])

    mapped = _map_spoolman_spool(spool)

    # Fetch K-profiles before the MQTT try block so we can use async DB access.
    kp_rows_result = await db.execute(
        select(SpoolmanKProfile).where(
            SpoolmanKProfile.spoolman_spool_id == spoolman_spool_id,
            SpoolmanKProfile.printer_id == printer_id,
        )
    )
    kp_rows = kp_rows_result.scalars().all()

    # Auto-configure AMS slot via MQTT (best-effort; slot assignment is already persisted)
    try:
        mqtt_client = printer_manager.get_client(printer_id)
        if mqtt_client:
            # Spoolman's material is free text, so it arrives as whatever the
            # user typed there -- "PLA+", "PolyTerra PLA". The sub-brand keeps
            # that wording; the slot's type has to be one the printer and the
            # slicer know (issue #2902).
            material = mapped.get("material") or ""
            tray_type = printer_filament_type(material)
            brand = mapped.get("brand") or ""
            subtype = mapped.get("subtype") or ""
            if brand:
                tray_sub_brands = f"{brand} {material} {subtype}".strip()
            elif subtype:
                tray_sub_brands = f"{material} {subtype}".strip()
            else:
                tray_sub_brands = material

            tray_color = (mapped.get("rgba") or "808080FF").upper()
            if len(tray_color) == 6:
                tray_color = tray_color + "FF"

            # Printer state, read here rather than further down because the
            # per-model preset override below needs the slot's nozzle
            # diameter and the K-profile cascade further down needs the same
            # value -- one read, so they cannot disagree. (The previous
            # `mqtt_client.printer_state` access via hasattr always returned
            # None -- the attribute is `state`, not `printer_state` -- so the
            # K-profile cascade silently skipped state.kprofiles, defaulted
            # nozzle_diameter to 0.4, and left slot_extruder unset.)
            state = printer_manager.get_status(printer_id)
            slot_nozzle = resolve_slot_nozzle(state, ams_id, tray_id, printer_manager.get_model(printer_id))
            nozzle_diameter = slot_nozzle.diameter

            # Per-printer-model preset override, same cascade as internal
            # mode: a cloud/Orca preset is bound to a model, so one stored
            # preset per spool is wrong across two models. Returns Spoolman's
            # own value when no override is set.
            slot_slicer_filament, slot_slicer_filament_name = await resolve_spoolman_preset(
                db,
                spoolman_spool_id=spoolman_spool_id,
                printer_model=printer_manager.get_model(printer_id),
                nozzle_diameter=nozzle_diameter,
                fallback_filament=mapped.get("slicer_filament"),
                fallback_name=mapped.get("slicer_filament_name"),
            )

            # #1713: resolve the spool's stored slicer_filament reference
            # (cloud preset, local preset, GF-prefix builtin, or numeric
            # LocalPreset id) to the printer-side tray_info_idx + setting_id.
            # Previously the Spoolman path dropped slicer_filament on the
            # floor and only the generic-material fallback fired; the user-
            # configured profile never reached the printer. Shared with the
            # internal-mode route via the same helper so the two flows can't
            # drift again.
            tray_info_idx, setting_id, sub_brand_override, type_override = await resolve_slicer_filament(
                db=db,
                current_user=current_user,
                slicer_filament=slot_slicer_filament,
                slicer_filament_name=slot_slicer_filament_name,
                material=material,
            )
            if sub_brand_override:
                tray_sub_brands = sub_brand_override
            # A preset carries its own type; the reduction above only infers
            # one from Spoolman's free-text material. The preset wins when the
            # spool has one (issue #2902, @doncaruana).
            if type_override:
                tray_type = printer_filament_type(type_override)
            # A support material keeps the slicer's suffix ("PLA-S") (#3559).
            tray_type = support_filament_type(
                tray_type, tray_info_idx, f"{slot_slicer_filament_name or ''} {tray_sub_brands}"
            )

            material_upper = material.upper().strip()
            # Without a preset, the slot's own filament id where it fits, as in
            # internal mode, before any generic one (#3433).
            if not tray_info_idx:
                current_tray = _live_slot_tray(state, ams_id, tray_id)
                tray_info_idx = reusable_slot_filament_id(
                    current_tray.get("tray_info_idx"),
                    current_tray.get("tray_type"),
                    current_tray.get("tray_sub_brands"),
                    tray_type,
                    f"{material} {subtype}",
                )
            # Fall back to generic-material id when slicer_filament is empty
            # or the resolver discarded an unresolvable value. Matches the
            # internal-mode tail in inventory.py:_apply_spool_to_slot_inner,
            # including the order: the spool's own wording first and the
            # reduced type only after it, so "PETG HF" keeps its own generic
            # preset (GFG96) rather than trading it for "PETG"'s GFG99.
            if not tray_info_idx:
                tray_info_idx = (
                    # The finish first, as in internal mode: a Silk spool
                    # gets Generic PLA Silk.
                    finish_generic_id(tray_type, f"{material} {subtype}")
                    or GENERIC_FILAMENT_IDS.get(material_upper)
                    or GENERIC_FILAMENT_IDS.get(material_upper.split("-")[0].split(" ")[0])
                    or GENERIC_FILAMENT_IDS.get(tray_type.upper())
                    or ""
                )

            # Ensure setting_id is always derivable from tray_info_idx. The
            # local-preset path can leave it empty when the LP's setting JSON
            # has no filament_id and falls through to the generic material id;
            # without this fallback the slicer gets a half-configured slot
            # (filament id without setting id) and the slot detail modal
            # renders empty fields. Same pattern as the internal-mode tail.
            if tray_info_idx and not setting_id:
                setting_id = filament_id_to_setting_id(tray_info_idx)

            temp_defaults = nozzle_temp_range(material, tray_type)
            temp_min = mapped.get("nozzle_temp_min") or temp_defaults[0]
            temp_max = temp_defaults[1]

            # Pull printer state from printer_manager. The previous
            # `mqtt_client.printer_state` access via hasattr always returned
            # None (the attribute is `state`, not `printer_state`), so the
            # K-profile cascade silently skipped state.kprofiles, defaulted
            # nozzle_diameter to 0.4, and left slot_extruder unset.
            slot_extruder = slot_nozzle.extruder

            # Prefer exact extruder match, fall back to extruder-agnostic kp
            # for the same nozzle. Hard-skipping on mismatch silently dropped
            # valid stored profiles when the AMS-extruder mapping had shifted.
            exact_kp = None
            fallback_kp = None
            for kp in kp_rows:
                if kp.nozzle_diameter != nozzle_diameter or kp.cali_idx is None:
                    continue
                if not slot_nozzle.flow_matches(kp.nozzle_type):
                    continue
                if slot_extruder is not None and kp.extruder is not None and kp.extruder == slot_extruder:
                    exact_kp = kp
                    break
                if fallback_kp is None:
                    fallback_kp = kp
            matching_kp = exact_kp or fallback_kp

            # Resolve the printer-side calibration entry by cali_idx so we
            # know the authoritative filament_id (the printer indexes its
            # calibration table by filament_id, not setting_id).
            printer_kp = None
            if matching_kp and state and state.kprofiles:
                for pkp in state.kprofiles:
                    if pkp.slot_id == matching_kp.cali_idx and pkp.nozzle_diameter == nozzle_diameter:
                        printer_kp = pkp
                        break
                if printer_kp is None:
                    logger.warning(
                        "Spoolman assign: cali_idx=%d not present in printer's "
                        "calibration table — stored kp may be stale.",
                        matching_kp.cali_idx,
                    )

            # Realign the slot's filament context (tray_info_idx + setting_id)
            # to the kp's calibration context. Without this, ams_filament_setting
            # declares the slot under generic PLA while extrusion_cali_sel points
            # the cali_idx at a different preset — the printer can't link them
            # and falls back to the default profile. P-prefix local presets are
            # valid for tray_info_idx; PFUS-prefix cloud-user presets are not
            # (the slicer rejects them).
            effective_tray_info_idx = tray_info_idx
            effective_setting_id = setting_id
            if printer_kp and printer_kp.filament_id:
                if not printer_kp.filament_id.startswith("PFUS"):
                    effective_tray_info_idx = printer_kp.filament_id
                if printer_kp.setting_id:
                    effective_setting_id = printer_kp.setting_id
            elif matching_kp and matching_kp.setting_id:
                derived = normalize_slicer_filament(matching_kp.setting_id)[0]
                if derived and not derived.startswith("PFUS"):
                    effective_tray_info_idx = derived
                effective_setting_id = matching_kp.setting_id
            if effective_tray_info_idx != tray_info_idx or effective_setting_id != setting_id:
                logger.info(
                    "Spoolman assign: realigning tray_info_idx %r → %r, setting_id %r → %r (kp_id=%s, source=%s)",
                    tray_info_idx,
                    effective_tray_info_idx,
                    setting_id,
                    effective_setting_id,
                    matching_kp.id if matching_kp else None,
                    "printer" if printer_kp else "stored",
                )

            mqtt_client.ams_set_filament_setting(
                ams_id=ams_id,
                tray_id=tray_id,
                tray_info_idx=effective_tray_info_idx,
                tray_type=tray_type,
                tray_sub_brands=tray_sub_brands,
                tray_color=tray_color,
                nozzle_temp_min=temp_min,
                nozzle_temp_max=temp_max,
                setting_id=effective_setting_id,
            )

            if matching_kp and matching_kp.cali_idx is not None:
                # Use printer-reported filament_id when available, otherwise
                # fall back to the realigned tray_info_idx so both commands
                # reference the same filament context.
                cali_filament_id = (
                    printer_kp.filament_id if printer_kp and printer_kp.filament_id else None
                ) or effective_tray_info_idx
                mqtt_client.extrusion_cali_sel(
                    ams_id=ams_id,
                    tray_id=tray_id,
                    cali_idx=matching_kp.cali_idx,
                    filament_id=cali_filament_id,
                    nozzle_diameter=nozzle_diameter,
                )
                logger.info(
                    "Spoolman assign: applied K-profile cali_idx=%d "
                    "(kp_id=%d, filament_id=%s) for spool %d on printer %d AMS%d-T%d",
                    matching_kp.cali_idx,
                    matching_kp.id,
                    cali_filament_id,
                    spoolman_spool_id,
                    printer_id,
                    ams_id,
                    tray_id,
                )
            else:
                # No stored K-profile for this spool — always reset the slot to
                # Default K (cali_idx=-1). The live cali_idx belongs to whatever
                # filament was there before, so preserving it would apply the
                # wrong filament's calibration to the new spool.
                mqtt_client.extrusion_cali_sel(
                    ams_id=ams_id,
                    tray_id=tray_id,
                    cali_idx=-1,
                    filament_id=effective_tray_info_idx,
                    nozzle_diameter=nozzle_diameter,
                )
                logger.info(
                    "No stored K-profile for Spoolman spool %d — reset slot to Default K (cali_idx=-1)",
                    spoolman_spool_id,
                )

            # Read-back verification, as the built-in assign registers it (#2582):
            # the AMS pushes that follow confirm the tray took what was sent.
            # Not for a slot the firmware calls empty, which echoes nothing
            # back until a spool goes in (the built-in check, assign_local_spool).
            live_tray = None
            if state and state.raw_data:
                ams_raw = state.raw_data.get("ams", [])
                if isinstance(ams_raw, dict):
                    ams_raw = ams_raw.get("ams", [])
                live_tray = _find_tray_in_ams_data(ams_raw, ams_id, tray_id)
            tray_state = (live_tray or {}).get("state")
            if not (spool_present(live_tray) is not True and tray_state in (9, 10)):
                mqtt_client.register_assignment_verification(
                    ams_id=ams_id,
                    tray_id=tray_id,
                    tray_info_idx=effective_tray_info_idx,
                    tray_color=tray_color,
                    cali_idx=matching_kp.cali_idx if matching_kp and matching_kp.cali_idx is not None else -1,
                )
                # Fresh telemetry for the check to compare against.
                mqtt_client.request_status_update()

            logger.info(
                "Auto-configured AMS slot ams=%d tray=%d for Spoolman spool %d on printer %d",
                ams_id,
                tray_id,
                spoolman_spool_id,
                printer_id,
            )
    except Exception:
        logger.exception(
            "Failed to auto-configure AMS slot for Spoolman spool %d (printer=%d, ams=%d, tray=%d)",
            spoolman_spool_id,
            printer_id,
            ams_id,
            tray_id,
        )

    return mapped
