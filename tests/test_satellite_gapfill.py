"""Regression test for the satellite interior-gap fill.

Divergent alpha-satellite arrays leave interior gaps that the 3-tier pipeline
cannot seed; those gaps sit between large-HOR-motif calls. `_fill_satellite_gaps`
is the backstop that recovers them. Two historical bugs silently dropped these
gaps and cost ~5-9pp of bp-recall on CHM13 centromeric HOR arrays:

  1. Satellite anchors were gated to motif length 100-300bp, so the large
     HOR-unit calls (motif >> 300bp) flanking a gap did not mark it as
     near-satellite and it was never scanned.
  2. The acceptance threshold (identity >= 0.55, period <= 300) rejected
     divergent arrays whose strongest periodicity is ~0.46 at the dimeric
     period (2x171=342).

This test reproduces both conditions on a synthetic diverged 171bp array and
asserts the interior gap is filled, plus a negative control (random sequence
must NOT be filled).
"""
import os
import numpy as np
import pytest

from src.finder import TandemRepeatFinder
from src.models import TandemRepeat

RNG = np.random.RandomState(1234)
BASES = np.frombuffer(b"ACGT", dtype=np.uint8)


def _rand_seq(n):
    return BASES[RNG.randint(0, 4, n)].tobytes().decode()


def _diverged_array(monomer, n_copies, p_mut):
    """Tile `monomer` n_copies times, mutating each base with prob p_mut."""
    m = np.frombuffer(monomer.encode(), dtype=np.uint8).copy()
    out = []
    for _ in range(n_copies):
        c = m.copy()
        mask = RNG.rand(len(c)) < p_mut
        c[mask] = BASES[RNG.randint(0, 4, int(mask.sum()))]
        out.append(c)
    return np.concatenate(out).tobytes().decode()


def _finder(seq):
    # tier set is irrelevant; we call the gap-filler directly
    return TandemRepeatFinder(seq, chromosome="syn", min_period=1,
                              max_period=8000, enabled_tiers={"tier1"})


def _coverage(calls, n):
    cov = bytearray(n)
    for r in calls:
        for i in range(r.start, min(r.end, n)):
            cov[i] = 1
    return cov


def test_interior_gap_between_large_motif_calls_is_filled():
    monomer = _rand_seq(171)
    clean = _diverged_array(monomer, 30, 0.02)      # ~pure flank blocks
    gap = _diverged_array(monomer, 80, 0.42)        # divergent interior (~0.46 id)
    seq = clean + gap + clean
    L1 = len(clean)
    L2 = L1 + len(gap)
    n = len(seq)

    finder = _finder(seq)
    # Flanking calls carry LARGE HOR-unit motifs (> 300bp) -> reproduce bug #1.
    big_motif_left = seq[:600]
    big_motif_right = seq[L2:L2 + 600]
    anchors = [
        TandemRepeat(chrom="syn", start=0, end=L1, motif=big_motif_left,
                     copies=L1 / 600, length=L1, tier=2,
                     consensus_motif=big_motif_left, mismatch_rate=0.02),
        TandemRepeat(chrom="syn", start=L2, end=n, motif=big_motif_right,
                     copies=(n - L2) / 600, length=n - L2, tier=2,
                     consensus_motif=big_motif_right, mismatch_rate=0.02),
    ]

    filled = finder._fill_satellite_gaps(anchors)
    cov = _coverage(filled, n)
    gap_cov = sum(cov[L1:L2])
    gap_len = L2 - L1
    assert gap_cov >= 0.8 * gap_len, (
        f"interior satellite gap not filled: {gap_cov}/{gap_len} bp covered")


def test_random_gap_is_not_filled():
    # Same anchor geometry but a NON-periodic interior must stay empty.
    monomer = _rand_seq(171)
    clean = _diverged_array(monomer, 30, 0.02)
    gap = _rand_seq(80 * 171)                       # pure random, no periodicity
    seq = clean + gap + clean
    L1 = len(clean)
    L2 = L1 + len(gap)
    n = len(seq)

    finder = _finder(seq)
    anchors = [
        TandemRepeat(chrom="syn", start=0, end=L1, motif=seq[:600],
                     copies=L1 / 600, length=L1, tier=2,
                     consensus_motif=seq[:600], mismatch_rate=0.02),
        TandemRepeat(chrom="syn", start=L2, end=n, motif=seq[L2:L2 + 600],
                     copies=(n - L2) / 600, length=n - L2, tier=2,
                     consensus_motif=seq[L2:L2 + 600], mismatch_rate=0.02),
    ]
    filled = finder._fill_satellite_gaps(anchors)
    cov = _coverage(filled, n)
    gap_cov = sum(cov[L1:L2])
    gap_len = L2 - L1
    assert gap_cov <= 0.2 * gap_len, (
        f"random non-periodic gap wrongly filled: {gap_cov}/{gap_len} bp")
