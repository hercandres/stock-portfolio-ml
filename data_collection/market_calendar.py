"""US stock-market closures the charts skip.

Nights, weekends, and full-day holidays are not plotted and no price is
invented for them. A missing minute on a day the market is open is different:
the chart leaves a break there because the source has no bar.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import lru_cache

from .alpaca_bars import NEW_YORK, REGULAR_CLOSE_MINUTES, REGULAR_OPEN_MINUTES


def is_nyse_holiday(day: date) -> bool:
    """True when the NYSE is closed all day. Weekends are not listed here."""
    return day in holiday_dates()


def holiday_dates() -> frozenset[date]:
    """Full-day NYSE closures. Weekends are handled separately."""
    return _holidays()


def next_session_mark(after: datetime, step: str) -> datetime:
    """The next bar time after ``after``, skipping closed market time.

    ``step`` is ``1m``, ``5m``, ``30m``, or ``1d``. The result is New York
    wall time with a timezone. Closed nights, weekends, and holidays are
    stepped over. No price is created.
    """
    if after.tzinfo is None:
        raise ValueError("after must be timezone-aware")
    local = after.astimezone(NEW_YORK).replace(second=0, microsecond=0)
    if step == "1d":
        probe = local + timedelta(days=1)
        while probe.weekday() >= 5 or is_nyse_holiday(probe.date()):
            probe += timedelta(days=1)
        return probe
    minutes = {"1m": 1, "5m": 5, "30m": 30}.get(step)
    if minutes is None:
        raise ValueError(f"unknown bar step {step}")
    probe = local + timedelta(minutes=minutes)
    for _ in range(10 * 24 * 60):
        if _session_bucket(probe, step):
            return probe
        probe += timedelta(minutes=minutes)
    raise RuntimeError("no open session within 10 days")


def session_has_a_hole(previous: datetime, current: datetime, step: str) -> bool:
    """True when a bar is missing while the market was open.

    The close, the night, the weekend, and a holiday are not a hole.
    """
    return next_session_mark(previous, step) != current.astimezone(NEW_YORK).replace(
        second=0, microsecond=0
    )


def _session_bucket(local: datetime, step: str) -> bool:
    if local.weekday() >= 5 or is_nyse_holiday(local.date()):
        return False
    minute_of_day = local.hour * 60 + local.minute
    if not (REGULAR_OPEN_MINUTES <= minute_of_day < REGULAR_CLOSE_MINUTES):
        return False
    if step == "5m" and local.minute % 5 != 0:
        return False
    if step == "30m" and local.minute % 30 != 0:
        return False
    return True


@lru_cache(maxsize=1)
def _holidays() -> frozenset[date]:
    """Full-day NYSE closures from 2016 through 2035."""
    found: set[date] = set()
    for year in range(2016, 2036):
        found.add(_observed(date(year, 1, 1)))
        found.add(_nth_weekday(year, 1, 0, 3))  # Martin Luther King Jr. Day
        found.add(_nth_weekday(year, 2, 0, 3))  # Washington's Birthday
        found.add(_easter(year) - timedelta(days=2))  # Good Friday
        found.add(_last_weekday(year, 5, 0))  # Memorial Day
        if year >= 2022:
            found.add(_observed(date(year, 6, 19)))  # Juneteenth
        found.add(_observed(date(year, 7, 4)))
        found.add(_nth_weekday(year, 9, 0, 1))  # Labor Day
        found.add(_nth_weekday(year, 11, 3, 4))  # Thanksgiving
        found.add(_observed(date(year, 12, 25)))
    return frozenset(found)


def _observed(day: date) -> date:
    """Saturday closes Friday. Sunday closes Monday."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    shift = (weekday - first.weekday()) % 7
    return first + timedelta(days=shift + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        cursor = date(year, 12, 31)
    else:
        cursor = date(year, month + 1, 1) - timedelta(days=1)
    return cursor - timedelta(days=(cursor.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Gregorian Easter Sunday. Good Friday is two days earlier."""
    century, year_in_century = divmod(year, 100)
    golden = year % 19
    century_leap = century // 4
    moon_shift = (century + 8) // 25
    century_correction = (century - moon_shift + 1) // 3
    lunar = (19 * golden + century - century_leap - century_correction + 15) % 30
    weekday_leap = year_in_century // 4
    weekday = (32 + 2 * (century % 4) + 2 * weekday_leap - lunar - (year_in_century % 4)) % 7
    extra = (golden + 11 * lunar + 22 * weekday) // 451
    month, day_index = divmod(lunar + weekday - 7 * extra + 114, 31)
    return date(year, month, day_index + 1)
