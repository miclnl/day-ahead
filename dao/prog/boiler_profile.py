"""Pure, complete boiler energy allocation over actual UTC intervals."""

import math
from fractions import Fraction


def boiler_profile(
    *, intervals: tuple[tuple[int, int], ...], start_index: int,
    required_kwh: float, power_w: float,
) -> tuple[float, ...] | None:
    """Return a full horizon vector, or None when the complete cycle cannot fit."""
    if (not isinstance(intervals, tuple) or not intervals or type(start_index) is not int
        or not 0 <= start_index < len(intervals)):
        raise ValueError("invalid boiler intervals or start index")
    previous_end = None
    for interval in intervals:
        if not isinstance(interval, tuple) or len(interval) != 2:
            raise ValueError("invalid boiler interval")
        start, end = interval
        if (type(start) is not int or type(end) is not int or start >= end
            or (previous_end is not None and start != previous_end)):
            raise ValueError("invalid boiler interval")
        previous_end = end
    try:
        if not math.isfinite(required_kwh) or not math.isfinite(power_w):
            raise ValueError("non-finite boiler energy or power")
        remaining = Fraction(str(required_kwh))
        power = Fraction(str(power_w))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid boiler energy or power") from exc
    if remaining < 0 or power <= 0:
        raise ValueError("invalid boiler energy or power")

    profile = [0.0] * len(intervals)
    for u in range(start_index, len(intervals)):
        start, end = intervals[u]
        energy = min(remaining, power * (end - start) / 3_600_000)
        profile[u] = float(energy)
        remaining -= energy
        if remaining == 0:
            return tuple(profile)
    return None
