# tests/test_maxrecall_extender.py
"""Regression test for the rolling-consensus mismatch extender (Task C2).

The original `extend_with_mismatches` (in src/_accelerators.pyx) compares every
copy to a FIXED first/seed copy and breaks on CUMULATIVE mismatch. On a diverged
STR array this truncates the call early: as drift accumulates, the cumulative
budget is exhausted long before the true array boundary, so the reported span
covers only part of the implanted repeat.

The new behaviour (opt-in via TIER1_EXT_ROLLING=1) maintains a per-position
MAJORITY consensus that is updated as copies are accepted, and breaks only after
K consecutive "bad" copies (per-copy mismatch fraction over the allowed rate),
extending in BOTH directions. This lets diverged arrays extend to their true
boundaries.

This test pins the mechanism deterministically:

  ROLLING ON  (TIER1_EXT_ROLLING=1) -> Tier1's best call must cover >= 90% of the
                                       implanted diverged array span.
  ROLLING OFF (env unset)           -> must still run without crashing (the old
                                       extender is the inert default; lower
                                       coverage is acceptable, a crash is not).

The default-inert guarantee (unset == original behaviour) is what makes the
env-gate safe to ship; the binding precision win is measured on real chr21 data
by the harness, not here.

Run: python tests/test_maxrecall_extender.py   (exits nonzero on failure)
Random seed is fixed (SEED=1234) so the synthetic array is deterministic.

NOTE: test_ground_truth.py imports pytest (not installed in this env), so we
inline the helpers we need rather than importing from that module (same style as
tests/test_maxrecall_seeding.py).
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
    """Run TandemRepeatFinder; returns list of TandemRepeat objects.

    Re-imports src.* so that the module-level env constants in the compiled
    extender (TIER1_EXT_ROLLING / TIER1_EXT_BAD_RUN, read once at import) pick up
    whatever the caller has set in os.environ for this invocation.
    """
    for m in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
        del sys.modules[m]
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


# ── Probe construction ───────────────────────────────────────────────────────

# A long, diverged dinucleotide array embedded in random flanks. 10% per-copy
# mismatch is enough that the FIXED-consensus cumulative extender drifts off and
# truncates, while the rolling consensus tracks the array to its true ends.
PROBE_MOTIF = "AC"
PROBE_COPIES = 40           # 80 bp implanted array
PROBE_MM = 0.10
FLANK = 250
SEED = 1234

# comboA + Task1 recommended flags (period-stratified short-STR gate). Fixed here
# so the test is self-contained regardless of the caller's environment.
TASK1_FLAGS = {
    "TIER1_MIN_ARRAY_LEN": "20",
    "TIER1_MIN_SCORE": "20",
    "TIER1_MIN_COPIES": "2",
    "TIER1_COPYBASE": "6",
    "TIER1_COPYADD": "2",
    "TIER1_EXT_COPIES": "2",
    "TIER1_SHORT_PERIOD_MAX": "4",
    "TIER1_SHORT_MIN_ARRAY_LEN": "18",
    "TIER1_SHORT_MIN_SCORE": "18",
}


def build_probe():
    """FLANK bp random + diverged (AC)x40 core + FLANK bp random.

    Returns (seq, core_start, core_end). Random state is consumed from the global
    stream (set once via SEED) so the result is deterministic.
    """
    left = random_dna(FLANK, gc=0.45)
    core = make_repeat(PROBE_MOTIF, PROBE_COPIES, mismatch_rate=PROBE_MM)
    right = random_dna(FLANK, gc=0.45)
    return left + core + right, len(left), len(left) + len(core)


def best_coverage(seq: str, cs: int, ce: int) -> float:
    """Fraction of the implanted [cs, ce) span covered by Tier1's best call."""
    tmp = tempfile.mkdtemp()
    try:
        fa = os.path.join(tmp, "probe.fa")
        write_fasta(fa, "probe", seq)
        preds = run_finder(fa, enabled_tiers={"tier1"}, min_period=1, max_period=9)
        span = ce - cs
        best = 0.0
        for p in preds:
            overlap = max(0, min(ce, p.end) - max(cs, p.start))
            cov = overlap / span if span > 0 else 0.0
            if cov > best:
                best = cov
        return best
    finally:
        shutil.rmtree(tmp)


def _run_under_env(overrides: dict):
    """Build the probe and report (best_coverage, crashed) under env overrides."""
    ext_keys = ("TIER1_EXT_ROLLING", "TIER1_EXT_BAD_RUN")
    saved = {k: os.environ.get(k)
             for k in list(TASK1_FLAGS) + list(ext_keys)}
    try:
        for k in ext_keys:
            os.environ.pop(k, None)
        os.environ.update(TASK1_FLAGS)
        os.environ.update(overrides)
        random.seed(SEED)
        seq, cs, ce = build_probe()
        try:
            return best_coverage(seq, cs, ce), False
        except Exception as exc:  # noqa: BLE001 - we want to report any crash
            print(f"  !! extender raised: {exc!r}")
            return 0.0, True
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Probe: ({PROBE_MOTIF})x{PROBE_COPIES} = {PROBE_COPIES * len(PROBE_MOTIF)} bp "
          f"core at {1 - PROBE_MM:.0%} purity, in {FLANK} bp random flanks")
    print()

    # OLD extender (default, env unset) - must run without crashing.
    old_cov, old_crash = _run_under_env({})
    # NEW rolling extender - must cover >= 90% of the implanted span.
    roll_cov, roll_crash = _run_under_env({
        "TIER1_EXT_ROLLING": "1",
        "TIER1_EXT_BAD_RUN": "2",
    })

    print(f"OLD  (TIER1_EXT_ROLLING unset) best coverage = {old_cov:.1%}  "
          f"(must just run; crashed={old_crash})")
    print(f"ROLL (TIER1_EXT_ROLLING=1)     best coverage = {roll_cov:.1%}  "
          f"(must be >= 90%; crashed={roll_crash})")
    print()

    failures = []
    if old_crash:
        failures.append("OLD extender path crashed (default must remain stable)")
    if roll_crash:
        failures.append("ROLLING extender path crashed")
    if roll_cov < 0.90:
        failures.append(
            f"ROLLING coverage {roll_cov:.1%} < 90% of implanted array span")

    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL PASS: rolling extender covers >= 90% of the diverged array; "
          "old extender still runs.")
    sys.exit(0)


if __name__ == "__main__":
    main()
