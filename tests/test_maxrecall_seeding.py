# tests/test_maxrecall_seeding.py
"""Regression test for the Tier 1 period-stratified length/score gate.

The short-STR recall gap (Exp1 / chr21 adotto) is dominated by short perfect
cores (a 7-copy dinucleotide = 14 bp, an 8-copy mononucleotide = 16 bp, ...)
that sit inside a much larger adotto region but are REJECTED by the global
acceptance gate in tier1.py:

    required_threshold = max(min_array_length, motif_len * dynamic_min_copies)
    ... reject if ext_length < required_threshold or rep_score < min_score

At the comboA operating point min_array_length=20 / min_score=20, so a short
core below ~20 bp never produces a call, even though it overlaps a real region.

The fix is a *period-stratified* relaxation: when motif_len <= SHORT_PERIOD_MAX,
the gate uses SHORT_MIN_ARRAY_LEN / SHORT_MIN_SCORE instead of the global
floors. Longer motifs keep the strict global gate, so precision on long repeats
is unaffected. When SHORT_PERIOD_MAX is unset (== 0) the behaviour is identical
to the global gate (baseline reproduced exactly).

This test pins that mechanism deterministically using a short mononucleotide
core (15 bp at 92% purity) that the run scanner seeds but the global gate
rejects:

  BASELINE (no lever)            -> REJECTED (15 bp < global 20)
  lever, SHORT_PERIOD_MAX >= 1   -> ACCEPTED (period 1 is "short"; floor 15)
  lever, SHORT_PERIOD_MAX == 0   -> REJECTED (stratification disabled, so the
                                    relaxed floors never apply -> baseline)

The last two cases share identical SHORT_MIN_ARRAY_LEN / SHORT_MIN_SCORE and
differ ONLY in SHORT_PERIOD_MAX, so they prove the period stratification is what
gates the relaxation (not merely lower thresholds).

Run: python tests/test_maxrecall_seeding.py   (exits nonzero on failure)
Random seed is fixed (SEED=42) so the synthetic array is deterministic.

NOTE: test_ground_truth.py imports pytest (not installed in this env), so we
inline the helpers we need rather than importing from that module.
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
    """True if one period divides the other, or they differ by <=20%."""
    if period_a == 0 or period_b == 0:
        return False
    lo, hi = min(period_a, period_b), max(period_a, period_b)
    if hi % lo == 0:
        return True
    return (hi - lo) / lo <= 0.2


# ── Probe construction ───────────────────────────────────────────────────────

# A short mononucleotide core that the run scanner seeds but the global gate
# rejects: 15 copies of "A" at 92% purity ~= 15 bp << global threshold of 20.
# Mononucleotides do not extend into random flanks, so this is a clean,
# deterministic probe of the length/score gate (it does not rely on the
# mismatch extender growing the array past the gate).
PROBE_MOTIF = "A"
PROBE_COPIES = 15
PROBE_MM = 0.08
SEED = 42


def build_probe():
    """300 bp flank + 15 bp mono-A core + 300 bp flank.

    Returns (seq, core_start, core_end). Random state is consumed from the
    global stream (set once via SEED) so the result is deterministic.
    """
    left = random_dna(300, gc=0.45)
    array = make_repeat(PROBE_MOTIF, PROBE_COPIES, mismatch_rate=PROBE_MM)
    right = random_dna(300, gc=0.45)
    return left + array + right, len(left), len(left) + len(array)


def detected(seq: str, a_start: int, a_end: int, motif: str) -> bool:
    """True if Tier1 reports an overlapping, period-compatible repeat."""
    tmp = tempfile.mkdtemp()
    try:
        fa = os.path.join(tmp, "probe.fa")
        write_fasta(fa, "probe", seq)
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


def _run_under_env(overrides: dict) -> bool:
    """Build the probe and report detection under the given env overrides.

    The comboA operating point is fixed here so the test is self-contained and
    not sensitive to whatever the caller exported. The probe is rebuilt from a
    fixed seed inside each call for determinism.
    """
    base = {
        "TIER1_MIN_ARRAY_LEN": "20",
        "TIER1_MIN_SCORE": "20",
        "TIER1_MIN_COPIES": "2",
        "TIER1_COPYBASE": "6",
        "TIER1_COPYADD": "2",
        "TIER1_EXT_COPIES": "2",
    }
    lever_keys = ("TIER1_SHORT_PERIOD_MAX", "TIER1_SHORT_MIN_ARRAY_LEN",
                  "TIER1_SHORT_MIN_SCORE")
    saved = {k: os.environ.get(k) for k in list(base) + list(lever_keys)}
    try:
        # Clear any inherited lever vars, then apply comboA + the overrides.
        for k in lever_keys:
            os.environ.pop(k, None)
        os.environ.update(base)
        os.environ.update(overrides)
        random.seed(SEED)
        seq, s, e = build_probe()
        return detected(seq, s, e, PROBE_MOTIF)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Probe: {PROBE_MOTIF}x{PROBE_COPIES} (~{PROBE_COPIES} bp core, "
          f"purity={1 - PROBE_MM:.0%}); global gate = 20 bp / score 20")
    print()

    # Three configurations differing only by the lever:
    #   baseline   - no lever set                     -> must REJECT (15 < 20)
    #   lever_p4   - SHORT_PERIOD_MAX=4, floors 15/14  -> must ACCEPT
    #   strat_off  - SHORT_PERIOD_MAX=0, floors 15/14  -> must REJECT (period 1
    #                is not "short" when max=0, so the global gate still applies)
    baseline = _run_under_env({})
    lever_p4 = _run_under_env({
        "TIER1_SHORT_PERIOD_MAX": "4",
        "TIER1_SHORT_MIN_ARRAY_LEN": "15",
        "TIER1_SHORT_MIN_SCORE": "14",
    })
    strat_off = _run_under_env({
        "TIER1_SHORT_PERIOD_MAX": "0",
        "TIER1_SHORT_MIN_ARRAY_LEN": "15",
        "TIER1_SHORT_MIN_SCORE": "14",
    })

    print(f"baseline (no lever)            detected={baseline}  (expect False)")
    print(f"lever SHORT_PERIOD_MAX=4 15/14 detected={lever_p4}  (expect True)")
    print(f"lever SHORT_PERIOD_MAX=0 15/14 detected={strat_off}  (expect False)")
    print()

    failures = []
    if baseline:
        failures.append("baseline should REJECT the 15 bp core (global gate=20)")
    if not lever_p4:
        failures.append("lever (SHORT_PERIOD_MAX>=1) should ACCEPT the 15 bp core")
    if strat_off:
        failures.append(
            "SHORT_PERIOD_MAX=0 must keep the global gate (no relaxation) -> REJECT")

    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL PASS: period-stratified gate rescues the short core only when "
          "SHORT_PERIOD_MAX covers its period.")
    sys.exit(0)


if __name__ == "__main__":
    main()
