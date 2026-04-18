"""Emit a JSON matrix of today's games for a dispatcher window.

Reads the GitHub Actions cron expression from env var GITHUB_SCHEDULE (format like "0 14 * * *")
or falls back to current UTC hour. Emits games whose first pitch falls in a
3-hour window starting ~1h after the dispatcher fires (so trigger_ts = first_pitch - 60min
is always in the future).

Output:
  games=[{"id": "823965", "title": "NYM @ LAD", "trigger_ts": 1745019900, "first_pitch": "..."}, ...]
  written to $GITHUB_OUTPUT
"""

import json
import os
import sys
from datetime import date, datetime, timezone, timedelta

import statsapi


# Each dispatcher covers a 3-hour window of first-pitch times (UTC).
# The cron fires 1 hour before the window start.
# Dispatcher at 14 UTC → games starting 15:00–18:00 UTC (11am–2pm ET).
DISPATCHER_WINDOWS = {
    "0 14 * * *": (15, 18),   # D1: 15:00–18:00 UTC (11am–2pm ET)
    "0 17 * * *": (18, 21),   # D2: 18:00–21:00 UTC (2pm–5pm ET)
    "0 20 * * *": (21, 24),   # D3: 21:00–00:00 UTC (5pm–8pm ET)
    "0 23 * * *": (24, 27),   # D4: 00:00–03:00+1 UTC (8pm–11pm ET)
}


def get_window():
    """Determine the dispatcher window [start_hour, end_hour] in UTC.

    Looks at GITHUB_SCHEDULE env var set by GHA; falls back to "current hour +1".
    """
    sched = os.environ.get("GITHUB_SCHEDULE", "")
    if sched in DISPATCHER_WINDOWS:
        return DISPATCHER_WINDOWS[sched]
    # Fallback — useful for workflow_dispatch triggers: use next 3 hours
    now = datetime.now(timezone.utc)
    start_h = now.hour + 1
    return (start_h, start_h + 3)


def main():
    now = datetime.now(timezone.utc)
    today = date.today().isoformat()

    start_h, end_h = get_window()

    # Compute window start/end. Handle hour overflow (24+) for D4.
    base_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    window_start = base_midnight + timedelta(hours=start_h)
    window_end = base_midnight + timedelta(hours=end_h)

    print(f"Dispatcher window (UTC): {window_start.isoformat()} → {window_end.isoformat()}")
    print(f"Current time (UTC):      {now.isoformat()}")

    games = statsapi.schedule(start_date=today, end_date=today)
    matrix = []
    skipped = []

    for g in games:
        status = g.get("status", "")
        if status in {"Final", "Completed Early", "Postponed", "Cancelled"}:
            skipped.append((g.get("game_id"), f"status={status}"))
            continue

        fp_str = g.get("game_datetime")
        if not fp_str:
            skipped.append((g.get("game_id"), "no game_datetime"))
            continue

        first_pitch = datetime.fromisoformat(fp_str.replace("Z", "+00:00"))
        if first_pitch < window_start or first_pitch >= window_end:
            continue

        # Trigger 60 min before first pitch, but at least 2 min from now
        trigger = first_pitch - timedelta(minutes=60)
        min_trigger = now + timedelta(minutes=2)
        if trigger < min_trigger:
            trigger = min_trigger

        matrix.append({
            "id": str(g["game_id"]),
            "title": f"{g.get('away_name', '')[:12]} @ {g.get('home_name', '')[:12]}",
            "trigger_ts": int(trigger.timestamp()),
            "first_pitch": fp_str,
        })

    # Write to GHA output if running in Actions
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a") as f:
            f.write(f"games={json.dumps(matrix)}\n")

    print(f"\nEmitting {len(matrix)} games:")
    for m in matrix:
        trig_iso = datetime.fromtimestamp(m["trigger_ts"], timezone.utc).isoformat()
        print(f"  {m['title']:<30} first_pitch={m['first_pitch']}  trigger={trig_iso}")

    if skipped:
        print(f"\nSkipped {len(skipped)} games:")
        for gid, reason in skipped[:5]:
            print(f"  {gid}: {reason}")

    # Emit just the JSON as stdout for local testing
    if not output_path:
        print(f"\nJSON matrix:\n{json.dumps(matrix, indent=2)}")


if __name__ == "__main__":
    main()
