# tests/test_maxrecall_seeding.py
"""Behavioral tests: diverged short STRs must be detected by Tier1.
Run: python tests/test_maxrecall_seeding.py  (exits nonzero on failure)
Mirrors the 'recoverable' class: period 1-3, ~82-92% purity, 7-15 copies.

Expected on baseline (comboA: TIER1_MIN_ARRAY_LEN=20 TIER1_MIN_SCORE=20):
  FAIL  mono_A_15cop_08mm   — 15 bp array, always below required_threshold=20
  FAIL  dinuc_AC_10cop_15mm — 20 bp, expected score 17.0 < MIN_SCORE=20
  FAIL  tri_CAG_7cop_15mm   — 21 bp, expected score 17.85 < MIN_SCORE=20
  (dinuc_AT_12cop_18mm may PASS or FAIL depending on random draws)

These cases are NOT expected to pass on the baseline; they encode the miss class
that later tasks (C1 mismatch-tolerant seeder, C2 rolling extender) will fix.

NOTE: test_ground_truth.py imports pytest (not installed in this env), so we
inline the three helpers we need directly rather than importing from that module.
The implementations are identical.

Random seed is fixed (SEED=42) so results are deterministic.
"""
import os
import sys
import random
import tempfile
import shutil

# Project root must be on sys.path so src/ is importable.
# tests/ must be on sys.path so fixtures/ is importable.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_tests_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _repo_root)
sys.path.insert(0, _tests_dir)

from fixtures.generate_synthetic import random_dna, make_repeat, write_fasta


# ── Inlined helpers (same logic as test_ground_truth.py) ─────────────────────

def _parse_fasta_simple(path: str) -> list:
    seqs = []
    name = None
    parts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    seqs.append((name, "".join(parts)))
                name = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
        if name:
            seqs.append((name, "".join(parts)))
    return seqs


def run_finder(fasta_path: str, enabled_tiers: set,
               min_period: int = 1, max_period: int = 100000) -> list:
    """Run TandemRepeatFinder; returns list of TandemRepeat objects."""
    from src.finder import TandemRepeatFinder
    all_repeats = []
    for name, seq in _parse_fasta_simple(fasta_path):
        seq = seq.upper()
        finder = TandemRepeatFinder(
            seq,
            chromosome=name,
            min_period=min_period,
            max_period=max_period,
            enabled_tiers=enabled_tiers,
        )
        all_repeats.extend(finder.find_all())
        finder.cleanup()
    return all_repeats


def overlap_ratio(s1: int, e1: int, s2: int, e2: int) -> float:
    """Fraction of the larger interval covered by the overlap."""
    overlap = max(0, min(e1, e2) - max(s1, s2))
    if overlap == 0:
        return 0.0
    span = max(e1 - s1, e2 - s2)
    return overlap / span if span > 0 else 0.0


def periods_compatible(period_a: int, period_b: int) -> bool:
    """True if one period divides the other, or they differ by ≤20%."""
    if period_a == 0 or period_b == 0:
        return False
    lo, hi = min(period_a, period_b), max(period_a, period_b)
    if hi % lo == 0:
        return True
    return (hi - lo) / lo <= 0.2


# ── Test cases ───────────────────────────────────────────────────────────────

# (label, motif, copies, mismatch_rate)
#
# Design: arrays that the chr21 comboA baseline misses.
# With SEED=42, the following FAIL on the baseline:
#   mono_A_15cop_08mm   — structural: 15 bp < required_threshold of 20 bp
#   dinuc_AC_10cop_15mm — score gate: 20 bp * 0.85 = 17.0 < MIN_SCORE=20
#   tri_CAG_7cop_15mm   — score gate: 21 bp * 0.85 = 17.85 < MIN_SCORE=20
# dinuc_AT_12cop_18mm (24 bp, score≈19.7) may PASS or FAIL depending on
# random draw; it is included to anchor the recoverable-at-higher-mismatch class.
#
# After C1+C2 fixes these should ALL pass.
CASES = [
    ("mono_A_15cop_08mm",    "A",   15, 0.08),   # 15 bp array,  92% purity
    ("dinuc_AC_10cop_15mm",  "AC",  10, 0.15),   # 20 bp array,  85% purity
    ("tri_CAG_7cop_15mm",    "CAG",  7, 0.15),   # 21 bp array,  85% purity
    ("dinuc_AT_12cop_18mm",  "AT",  12, 0.18),   # 24 bp array,  82% purity
]

# Fixed seed so FAIL/PASS results are deterministic across runs.
SEED = 42


def build_case(motif: str, copies: int, mm: float):
    """300 bp flank + imperfect array + 300 bp flank.
    Returns (seq, array_start, array_end).
    Random state is consumed from the global stream (set once via SEED).
    """
    left  = random_dna(300, gc=0.45)
    array = make_repeat(motif, copies, mismatch_rate=mm)
    right = random_dna(300, gc=0.45)
    return left + array + right, len(left), len(left) + len(array)


def detected(seq: str, a_start: int, a_end: int, motif: str) -> bool:
    """Return True if Tier1 reports an overlapping, period-compatible repeat.

    TandemRepeat objects (returned by run_finder) expose:
      .start, .end   — 0-based half-open coordinates
      .consensus_motif, .motif — motif strings (consensus preferred)
    Period is len(consensus_motif or motif); no dedicated .period field exists.
    """
    tmp = tempfile.mkdtemp()
    try:
        fa = os.path.join(tmp, "case.fa")
        write_fasta(fa, "case", seq)
        preds = run_finder(fa, enabled_tiers={"tier1"}, min_period=1, max_period=9)
        for p in preds:
            if overlap_ratio(a_start, a_end, p.start, p.end) >= 0.5:
                p_motif_str = p.consensus_motif or p.motif or ""
                p_period = len(p_motif_str) if p_motif_str else 0
                if periods_compatible(len(motif), p_period):
                    return True
        return False
    finally:
        shutil.rmtree(tmp)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Fix random seed so results are reproducible.
    random.seed(SEED)

    fails = []
    for label, motif, copies, mm in CASES:
        seq, s, e = build_case(motif, copies, mm)
        ok = detected(seq, s, e, motif)
        status = "PASS" if ok else "FAIL"
        array_len = e - s         # actual length (may differ from copies*len due to indels)
        purity = 1.0 - mm
        print(f"{status}  {label}  "
              f"(array≈{copies * len(motif)}bp, purity={purity:.0%}, copies={copies})")
        if not ok:
            fails.append(label)

    print()
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        sys.exit(1)
    print("ALL PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
