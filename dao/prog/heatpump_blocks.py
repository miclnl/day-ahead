"""Pure duration distribution and interval spans for heat-pump power/curve blocks."""


def block_durations(
    *, required_seconds: int, minimum_seconds: int, first_seconds: int, count: int,
) -> tuple[int, ...]:
    """Fit the heuristic count to the required duration and real block minima."""
    if count == 0:
        return ()
    remaining = required_seconds - first_seconds
    while count > 1 and remaining < (count - 1) * minimum_seconds:
        count -= 1
    if count == 1:
        return (max(required_seconds, first_seconds),)
    # #896 divides the remaining hours across whole-hour middle blocks.
    # Keep every remaining second in the last block, without modulo loss.
    middle_seconds = max(minimum_seconds, remaining // ((count - 1) * 3600) * 3600)
    last_seconds = remaining - (count - 2) * middle_seconds
    return (first_seconds, *(middle_seconds for _ in range(count - 2)), last_seconds)


def block_spans(
    intervals: tuple[tuple[int, int], ...], duration_seconds: int,
) -> tuple[tuple[int, int], ...]:
    """Return the shortest contiguous half-open index span at each feasible start."""
    if duration_seconds <= 0:
        return ()
    spans = []
    for start, (start_ts, _) in enumerate(intervals):
        for end in range(start, len(intervals)):
            if end > start and intervals[end][0] != intervals[end - 1][1]:
                break
            if intervals[end][1] - start_ts >= duration_seconds:
                spans.append((start, end + 1))
                break
    return tuple(spans)
