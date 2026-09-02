"""Dagster schedules — the project's only schedule (US-060 drift monitor).

Restored from the US-060 handoff spec (``docs/us-handoff/us-060.md``): the
original module never reached git because the generic ``schedules/`` pattern
in ``.gitignore`` (meant for Dagster runtime directories) swallowed this
package. ``definitions.py`` has imported it since US-060.

Training jobs stay on-demand (no schedule) so GPU is never spent by accident;
``drift_check_weekly_schedule`` runs Mondays 06:00 UTC and ships STOPPED by
default so a fresh deployment never fires it without an operator decision.
"""

from __future__ import annotations

from dagster import (
    AssetSelection,
    DefaultScheduleStatus,
    ScheduleDefinition,
    define_asset_job,
)

#: Materializes only the ``drift_check`` asset (weekly Evidently drift monitor).
drift_check_job = define_asset_job(
    name="drift_check_job",
    selection=AssetSelection.assets("drift_check"),
    tags={"us": "US-060", "epic": "E10"},
)

#: Weekly cron (Mondays 06:00 UTC). Default STOPPED: must be enabled in the UI.
drift_check_weekly_schedule = ScheduleDefinition(
    name="drift_check_weekly_schedule",
    job=drift_check_job,
    cron_schedule="0 6 * * 1",
    execution_timezone="UTC",
    default_status=DefaultScheduleStatus.STOPPED,
)

__all__ = ["drift_check_job", "drift_check_weekly_schedule"]
