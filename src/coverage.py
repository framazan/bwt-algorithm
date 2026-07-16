"""Boolean coverage masks over a sequence.

Four call sites open-coded the same "mark these intervals True over n positions"
loop: the satellite gap-fill and the catch-all pass in `finder`, the Tier-1 mask
in `tier2`, and the previous-tier mask in `tier3`. They differ only in where the
intervals come from.

Intervals are half-open `[start, end)` and are clipped to `n`; an interval that
starts at or past `n` contributes nothing.
"""
from typing import Iterable, Tuple

import numpy as np


def intervals_to_mask(intervals: Iterable[Tuple[int, int]], n: int) -> np.ndarray:
    """Boolean mask of length `n`, True wherever some interval covers a position."""
    mask = np.zeros(n, dtype=bool)
    for start, end in intervals:
        if start < n:
            mask[start:min(end, n)] = True
    return mask


def mask_from_repeats(repeats, n: int) -> np.ndarray:
    """`intervals_to_mask` over the spans of a list of `TandemRepeat`s."""
    return intervals_to_mask(((r.start, r.end) for r in repeats), n)
