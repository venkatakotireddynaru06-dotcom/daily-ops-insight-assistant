import jpholiday  # pip install jpholiday


def is_holiday(d) -> bool:
    """Japanese public holiday check. Accepts date or datetime/Timestamp."""
    d = d.date() if hasattr(d, "date") else d
    return bool(jpholiday.is_holiday(d))
