"""Accelerated hot paths, with pure-Python fallbacks that give identical answers.

Five symbols are consumed by the tiers: ``extend_with_mismatches``,
``find_periodic_runs``, ``lcp_tandem_candidates``, ``align_unit_to_window`` and
``anchor_scan_boundaries``.  Each is aliased to the compiled Cython extension
``_accelerators`` when it is importable, and otherwise to a pure-Python
implementation in this module.

Every fallback here is a **faithful fallback**: it transcribes the corresponding
``_accelerators.pyx`` routine and returns the same value for the same input, so
a build without the compiled extension detects exactly the same repeats — only
more slowly.  ``tests/test_accel_parity.py`` pins that property by asserting the
two paths emit byte-identical BED.  If you add a fallback that cannot honour it,
say so in its docstring and make it raise rather than return a degenerate value:
a tandem-repeat caller that silently finds nothing is worse than one that stops.

Set ``BWT_DISABLE_NATIVE=1`` to force the pure-Python path even when the
extension is available (used by the parity test, and useful for isolating a
suspected accelerator bug).
"""
from __future__ import annotations

import math
import os
import sys
import warnings
from typing import List, Optional, Tuple

import numpy as np

_DISABLED = os.environ.get("BWT_DISABLE_NATIVE") == "1"

if _DISABLED:
    _native = None
else:
    try:
        from . import _accelerators as _native  # type: ignore
    except Exception as exc:  # ImportError, or a numpy/Cython ABI mismatch
        _native = None
        _message = (
            f"Compiled extension src/_accelerators could not be imported ({exc}); "
            "falling back to pure Python. Results are identical but detection will "
            "be much slower. See CLAUDE.md 'Building Cython Extensions'."
        )
        try:
            warnings.warn(_message, RuntimeWarning, stacklevel=2)
        except RuntimeWarning:
            # `-W error` / PYTHONWARNINGS=error escalates this advisory into an
            # exception, which would make the package unimportable without the
            # extension. The fallback is correct, so say so and keep going.
            print(_message, file=sys.stderr)

NATIVE_AVAILABLE = _native is not None


def _rolling_extender_requested() -> bool:
    """Parse TIER1_EXT_ROLLING exactly as `_accelerators.pyx` does at its import.

    The .pyx uses `bool(int(...))`, so "00" is falsey and "" or "false" raise.
    Diverging here would mean the same environment selects a different extender
    depending on whether the extension happens to be built.
    """
    raw = os.environ.get("TIER1_EXT_ROLLING", "0")
    try:
        return bool(int(raw))
    except ValueError:
        raise ValueError(f"TIER1_EXT_ROLLING must be an integer, got {raw!r}") from None


# Read once at import, like the .pyx, so the check stays out of the per-seed hot path.
_EXT_ROLLING_REQUESTED = _rolling_extender_requested()

AcceleratorResult = Optional[Tuple[int, int, int, int, int]]
"""(array_start, array_end, copies, full_start, full_end) or None."""


# --------------------------------------------------------------------------
# Pure-Python implementations (used when the extension is unavailable)
# --------------------------------------------------------------------------

def _max_mismatch_threshold(period: int, copies: int, allowed_rate: float) -> int:
    """Mismatch budget for `copies` copies of a `period`-mer. Mirrors the .pyx.

    The .pyx deliberately dropped the old ``period == 1 -> 0`` special case, so
    homopolymers get the same allowed rate as every other period. Do not
    reintroduce it.
    """
    if period <= 0 or copies <= 0:
        return 0
    allowed_rate = min(max(allowed_rate, 0.01), 0.5)
    return max(1, math.ceil(allowed_rate * (period * copies)))


def _total_mismatches(text_arr: np.ndarray, start_pos: int, end_pos: int,
                      consensus: np.ndarray, period: int, n: int) -> int:
    """Sum of per-copy Hamming distance to `consensus` over [start_pos, end_pos).

    Copies that would run past `n` are dropped, matching the .pyx `break`.
    """
    copies = (end_pos - start_pos) // period
    copies = min(copies, (n - start_pos) // period)
    if copies <= 0:
        return 0
    span = text_arr[start_pos:start_pos + copies * period].reshape(copies, period)
    return int(np.count_nonzero(span != consensus))


def _extend_with_mismatches_py(s_arr: np.ndarray, start_pos: int, period: int,
                               n: int, allowed_mismatch_rate: float) -> AcceleratorResult:
    """FAITHFUL FALLBACK for `_accelerators.extend_with_mismatches` (default branch).

    Fixes the consensus at the seed copy, then grows the array one whole period
    at a time — right first, then left — while the cumulative mismatch count
    stays within budget. A perfect copy never consumes budget, so runs of exact
    copies always extend. Finally both flanks grow by a partial, exactly-matching
    prefix/suffix of the consensus.
    """
    if _EXT_ROLLING_REQUESTED:
        raise NotImplementedError(
            "TIER1_EXT_ROLLING=1 selects the rolling-consensus extender, which "
            "exists only in the compiled _accelerators extension. Build it "
            "(see CLAUDE.md) or unset TIER1_EXT_ROLLING."
        )

    if period <= 0:
        return None
    arr_len = s_arr.shape[0]
    if n <= 0 or n > arr_len:
        n = arr_len
    if start_pos < 0 or start_pos + period > n:
        return None

    consensus = s_arr[start_pos:start_pos + period]
    start = start_pos
    end = start_pos + period
    copies = 1

    while end + period <= n:
        temp_copies = copies + 1
        temp_end = end + period
        if not np.array_equal(s_arr[end:end + period], consensus):
            max_mm = _max_mismatch_threshold(period, temp_copies, allowed_mismatch_rate)
            if _total_mismatches(s_arr, start, temp_end, consensus, period, n) > max_mm:
                break
        copies = temp_copies
        end = temp_end

    while start - period >= 0:
        temp_copies = copies + 1
        temp_start = start - period
        if not np.array_equal(s_arr[temp_start:temp_start + period], consensus):
            max_mm = _max_mismatch_threshold(period, temp_copies, allowed_mismatch_rate)
            if _total_mismatches(s_arr, temp_start, end, consensus, period, n) > max_mm:
                break
        copies = temp_copies
        start = temp_start

    full_start, full_end = start, end

    partial_right = 0
    while partial_right < period and full_end + partial_right < n:
        if s_arr[full_end + partial_right] != consensus[partial_right]:
            break
        partial_right += 1

    partial_left = 0
    while partial_left < period and full_start - partial_left - 1 >= 0:
        if s_arr[full_start - partial_left - 1] != consensus[period - 1 - partial_left]:
            break
        partial_left += 1

    return (full_start - partial_left, full_end + partial_right,
            copies, full_start, full_end)


def _find_periodic_runs_py(positions: np.ndarray, min_period: int, max_period: int,
                           min_copies: int, tolerance_ratio: float = 0.01) -> List[Tuple[int, int, int]]:
    """FAITHFUL FALLBACK for `_accelerators.find_periodic_runs`.

    One pass over the adjacent gaps of a sorted position array; a run is a
    maximal stretch of gaps that agree with the first gap of the run to within
    `max(1.0, gap * tolerance_ratio)` and lie in [min_period, max_period].
    Returns (run_start, run_end, period) per run of at least `min_copies`
    positions.
    """
    n = positions.shape[0]
    results: List[Tuple[int, int, int]] = []
    if n < min_copies:
        return results

    prev_pos = int(positions[0])
    last_diff = -1.0
    run_start_idx = 0
    gap_count = 0

    def _flush(end_idx: int) -> None:
        if gap_count + 1 >= min_copies:
            results.append((int(positions[run_start_idx]), int(positions[end_idx]),
                            int(last_diff + 0.5)))

    for i in range(1, n):
        cur_pos = int(positions[i])
        diff = float(cur_pos - prev_pos)
        prev_pos = cur_pos

        if diff < min_period or diff > max_period:
            _flush(i - 1)
            run_start_idx = i
            gap_count = 0
            last_diff = -1.0
            continue

        if last_diff < 0:
            last_diff = diff
            gap_count = 1
            run_start_idx = i - 1
        else:
            tol = max(1.0, last_diff * tolerance_ratio)
            if last_diff - tol <= diff <= last_diff + tol:
                gap_count += 1
            else:
                _flush(i - 1)
                last_diff = diff
                gap_count = 1
                run_start_idx = i - 1

    _flush(n - 1)
    return results


def _lcp_tandem_candidates_py(sa: np.ndarray, lcp: np.ndarray, n: int,
                              min_period: int, max_period: int,
                              min_lcp_threshold: int = 10) -> List[Tuple[int, int]]:
    """FAITHFUL FALLBACK for `_accelerators.lcp_tandem_candidates`.

    Adjacent suffixes sharing a long prefix and separated by a plausible period
    are tandem-repeat candidates; returns (period, start) pairs.
    """
    results: List[Tuple[int, int]] = []
    limit = min(len(sa), len(lcp))
    for i in range(1, limit):
        if int(lcp[i]) < min_lcp_threshold:
            continue
        pos_a, pos_b = int(sa[i - 1]), int(sa[i])
        if pos_a >= n or pos_b >= n:
            continue  # past the '$' sentinel
        diff = abs(pos_b - pos_a)
        if min_period <= diff <= max_period:
            results.append((diff, min(pos_a, pos_b)))
    return results


def _align_unit_to_window_py(motif: bytes, window: bytes, max_indel: int,
                             mismatch_tolerance: int) -> Optional[Tuple]:
    """FAITHFUL FALLBACK: defers to `MotifUtils._align_unit_to_window`.

    Returning None here is not a degenerate stub — it is the documented signal
    that makes the sole caller run its own banded-DP Python body, which produces
    the same alignment.
    """
    return None


def _anchor_scan_boundaries_py(text_arr: np.ndarray, seed_pos: int, period: int, n: int,
                               match_threshold: float, max_backward_periods: int,
                               max_forward_periods: int) -> Tuple[int, int]:
    """FAITHFUL FALLBACK for `_accelerators.anchor_scan_boundaries`.

    Walks whole periods out from the seed in both directions, stopping when a
    window matches the seed motif on fewer than `match_threshold` of its bases.
    """
    if seed_pos + period > n:
        return (seed_pos, seed_pos + period)

    motif_arr = text_arr[seed_pos:seed_pos + period]
    true_start = seed_pos
    true_end = seed_pos + period

    scan_start = max(0, seed_pos - period * max_backward_periods)
    pos = seed_pos - period
    while pos >= scan_start:
        window = text_arr[pos:pos + period]
        if window.size != period:
            break
        if int(np.count_nonzero(window == motif_arr)) / period < match_threshold:
            break
        true_start = pos
        pos -= period

    scan_end = min(n, seed_pos + period * max_forward_periods)
    pos = seed_pos + period
    while pos + period <= scan_end:
        window = text_arr[pos:pos + period]
        if window.size != period:
            break
        if int(np.count_nonzero(window == motif_arr)) / period < match_threshold:
            break
        true_end = pos + period
        pos += period

    return (true_start, true_end)


# --------------------------------------------------------------------------
# Public bindings
# --------------------------------------------------------------------------

if NATIVE_AVAILABLE:
    extend_with_mismatches = _native.extend_with_mismatches
    find_periodic_runs = _native.find_periodic_runs
    lcp_tandem_candidates = _native.lcp_tandem_candidates
    align_unit_to_window = _native.align_unit_to_window
    anchor_scan_boundaries = _native.anchor_scan_boundaries
else:
    extend_with_mismatches = _extend_with_mismatches_py
    find_periodic_runs = _find_periodic_runs_py
    lcp_tandem_candidates = _lcp_tandem_candidates_py
    align_unit_to_window = _align_unit_to_window_py
    anchor_scan_boundaries = _anchor_scan_boundaries_py
