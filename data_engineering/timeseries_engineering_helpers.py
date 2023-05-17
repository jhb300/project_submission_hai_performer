from datetime import datetime

def normalize_datetime(dt: datetime) -> datetime:
    "Normalize datetime object to be in UTC time and at midnight."

    if isinstance(dt, datetime):
        dt = dt.tz_convert('UTC')
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return dt
