"""Shared autocorrelation primitives for periodicity detection.

Three detection paths independently re-derived the same "how self-similar is the
sequence at offset ``period``" computation:

  - ``finder._fill_satellite_gaps``   — a scalar identity per (window, period)
  - ``finder._catchall_periodicity_fill`` — a cumsum-based sliding-window count
  - ``tier2._autocorr_seed``          — the identical cumsum sliding-window count

This module factors those out so the period-detection math lives in one place.
Every function reproduces the exact numpy operations the inline code used (same
``int64`` cumsum, same ``int8``-view run detection), so substituting them is
behavior-preserving.

All functions take a 1-D ``uint8`` view of the sequence (e.g. ``BWTCore.text_arr``).
"""
from typing import Optional, Tuple

import numpy as np


def autocorr_identity(arr: np.ndarray, period: int) -> float:
    """Fraction of bases that match their copy ``period`` positions ahead.

    ``identity = mean(arr[:-period] == arr[period:])`` over the whole array.
    Returns ``0.0`` when ``period`` is not shorter than ``arr`` (no overlap).
    This is the scalar primitive used by the satellite gap-fill's per-window
    best-period search.
    """
    total = arr.size - period
    if total <= 0:
        return 0.0
    return float(np.sum(arr[:total] == arr[period:]) / total)


def windowed_match_counts(
    arr: np.ndarray, period: int, window: int
) -> Optional[np.ndarray]:
    """Sliding-window count of period-``period`` matches, in O(n) via a cumsum.

    Returns an array ``winsum`` where ``winsum[i]`` is the number of matching
    bases between ``arr[i:i+window]`` and ``arr[i+period:i+period+window]`` — i.e.
    the local autocorrelation strength at every window start. Its length is
    ``(arr.size - period) - window``.

    Returns ``None`` when the array is too short to contain a full window
    (``arr.size - period <= window``). Used by the catch-all and Tier-2
    autocorrelation seeders to find locally periodic stretches.
    """
    n = arr.size
    eq = (arr[:n - period] == arr[period:n])
    cs = np.empty(eq.size + 1, dtype=np.int64)
    cs[0] = 0
    np.cumsum(eq, out=cs[1:])
    m = eq.size - window
    if m <= 0:
        return None
    return cs[window:window + m] - cs[:m]


def contiguous_true_runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Start/end indices of maximal contiguous ``True`` runs in a boolean mask.

    Returns ``(run_starts, run_ends)`` with exclusive ends, so run ``k`` spans
    ``mask[run_starts[k]:run_ends[k]]``. Vectorized (no per-index Python) — the
    same ``np.diff`` over an ``int8`` view both seeders used inline. Empty arrays
    when there are no ``True`` positions.
    """
    d = np.diff(mask.view(np.int8))
    run_s = np.nonzero(d == 1)[0] + 1
    run_e = np.nonzero(d == -1)[0] + 1
    if mask[0]:
        run_s = np.concatenate(([0], run_s))
    if mask[-1]:
        run_e = np.concatenate((run_e, [mask.size]))
    return run_s, run_e
