"""Randomised differential test: each Python fallback vs its Cython original.

`test_accel_parity.py` compares the two accelerator paths end-to-end, which is
the property users care about but only exercises the inputs the fixtures happen
to produce. This module calls the four ported functions directly on thousands of
random inputs — including the degenerate ones (period longer than the array, `n`
past the array end, mismatch rates outside the clamp, runs that never close) —
and requires exact agreement.

Skipped when the compiled extension is absent, since there is nothing to compare.
"""
import numpy as np
import pytest

from src import accelerators as acc

pytestmark = pytest.mark.skipif(
    not acc.NATIVE_AVAILABLE,
    reason="compiled src/_accelerators is absent; nothing to differentially test against",
)

if acc.NATIVE_AVAILABLE:
    from src import _accelerators as native

ACGT = np.array([65, 67, 71, 84], dtype=np.uint8)  # A, C, G, T


def random_sequence(rng, size):
    return ACGT[rng.integers(0, 4, size=size)].astype(np.uint8)


def tandem_sequence(rng, period, copies, mismatches):
    """A repeat array with a controlled number of substitutions."""
    unit = random_sequence(rng, period)
    arr = np.tile(unit, copies)
    if mismatches and arr.size:
        idx = rng.choice(arr.size, size=min(mismatches, arr.size), replace=False)
        arr[idx] = ACGT[rng.integers(0, 4, size=idx.size)]
    return arr


def test_extend_with_mismatches_matches_cython():
    rng = np.random.default_rng(20260709)
    for trial in range(400):
        period = int(rng.integers(1, 12))
        copies = int(rng.integers(1, 14))
        pad = int(rng.integers(0, 25))
        arr = np.concatenate([
            random_sequence(rng, pad),
            tandem_sequence(rng, period, copies, int(rng.integers(0, 5))),
            random_sequence(rng, pad),
        ])
        start_pos = int(rng.integers(0, max(1, arr.size)))
        n = int(rng.integers(0, arr.size + 3))  # deliberately allow n > arr.size and n == 0
        rate = float(rng.choice([0.0, 0.005, 0.05, 0.2, 0.5, 0.9]))

        expected = native.extend_with_mismatches(arr, start_pos, period, n, rate)
        got = acc._extend_with_mismatches_py(arr, start_pos, period, n, rate)
        assert got == expected, (
            f"trial {trial}: start={start_pos} period={period} n={n} rate={rate} "
            f"size={arr.size}\n  cython={expected}\n  python={got}"
        )


def test_extend_with_mismatches_rejects_bad_periods():
    arr = np.zeros(20, dtype=np.uint8)
    for period in (0, -3):
        assert native.extend_with_mismatches(arr, 0, period, 20, 0.1) is None
        assert acc._extend_with_mismatches_py(arr, 0, period, 20, 0.1) is None


def test_find_periodic_runs_matches_cython():
    rng = np.random.default_rng(775533)
    for trial in range(400):
        k = int(rng.integers(0, 40))
        if k and rng.random() < 0.6:
            # a near-arithmetic progression, so real runs actually form
            step = int(rng.integers(1, 30))
            jitter = rng.integers(-1, 2, size=k)
            positions = np.cumsum(np.full(k, step) + jitter).astype(np.int64)
            positions = np.abs(positions)
        else:
            positions = np.sort(rng.integers(0, 500, size=k)).astype(np.int64)

        min_period = int(rng.integers(1, 10))
        max_period = min_period + int(rng.integers(0, 30))
        min_copies = int(rng.integers(1, 5))
        tol = float(rng.choice([0.0, 0.01, 0.1, 0.5]))

        expected = native.find_periodic_runs(positions, min_period, max_period, min_copies, tol)
        got = acc._find_periodic_runs_py(positions, min_period, max_period, min_copies, tol)
        assert got == expected, (
            f"trial {trial}: min_p={min_period} max_p={max_period} "
            f"min_copies={min_copies} tol={tol}\n  positions={positions.tolist()}\n"
            f"  cython={expected}\n  python={got}"
        )


def test_lcp_tandem_candidates_matches_cython():
    """This fallback predates the port; it had never been checked against the C path.

    int32 because that is the contract: the .pyx takes `const int[:]`, and tier2
    passes `suffix_array.astype(np.int32)` alongside a Kasai LCP built as int32.
    """
    rng = np.random.default_rng(4242)
    for _ in range(200):
        size = int(rng.integers(1, 60))
        sa = rng.permutation(size).astype(np.int32)
        lcp = rng.integers(0, 40, size=size).astype(np.int32)
        n = int(rng.integers(1, size + 2))
        min_period = int(rng.integers(1, 12))
        max_period = min_period + int(rng.integers(0, 40))
        thresh = int(rng.integers(0, 20))

        expected = native.lcp_tandem_candidates(sa, lcp, n, min_period, max_period, thresh)
        got = acc._lcp_tandem_candidates_py(sa, lcp, n, min_period, max_period, thresh)
        assert got == expected


def test_anchor_scan_boundaries_matches_cython():
    """Also a pre-existing fallback that no test covered."""
    rng = np.random.default_rng(31337)
    for _ in range(200):
        period = int(rng.integers(1, 15))
        copies = int(rng.integers(1, 12))
        arr = np.concatenate([
            random_sequence(rng, int(rng.integers(0, 20))),
            tandem_sequence(rng, period, copies, int(rng.integers(0, 4))),
            random_sequence(rng, int(rng.integers(0, 20))),
        ])
        seed_pos = int(rng.integers(0, arr.size))
        n = int(rng.integers(1, arr.size + 1))
        thresh = float(rng.choice([0.0, 0.5, 0.8, 1.0]))
        back = int(rng.integers(0, 6))
        fwd = int(rng.integers(0, 6))

        expected = native.anchor_scan_boundaries(arr, seed_pos, period, n, thresh, back, fwd)
        got = acc._anchor_scan_boundaries_py(arr, seed_pos, period, n, thresh, back, fwd)
        assert tuple(got) == tuple(expected), (
            f"seed={seed_pos} period={period} n={n} thresh={thresh} back={back} fwd={fwd}\n"
            f"  cython={expected}\n  python={got}"
        )
