"""Display formatting helpers."""

from datetime import datetime, timezone


def format_relative_date(dt: datetime) -> str:
    """Format datetime as human-readable relative string for dashboard."""
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    diff = now - dt
    seconds = int(diff.total_seconds())

    if seconds < 60:
        return "Just now"
    if seconds < 3600:
        mins = seconds // 60
        return f"{mins} min ago" if mins > 1 else "1 min ago"
    if seconds < 86400:
        hours = seconds // 3600
        return f"{hours} h ago" if hours > 1 else "1 h ago"
    if seconds < 172800:
        return f"Yesterday, {dt.strftime('%H:%M')}"
    if seconds < 604800:
        days = seconds // 86400
        return f"{days} days ago"
    return dt.strftime("%d %b, %H:%M")
