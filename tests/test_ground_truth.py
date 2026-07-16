"""Ground truth regression tests for tandem repeat detection.

Runs TandemRepeatFinder on synthetic sequences with known repeats,
compares results against ground truth BED files, and asserts
sensitivity/precision thresholds per tier.
"""
import os
import pytest
import numpy as np
from typing import List, Tuple, Set, Optional

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

# Tier 2/3 used to be skipped here when the Cython _accelerators extension was
# absent, because the pure-Python stubs returned nothing and every case failed.
# The fallbacks are faithful now (see src/accelerators.py), so both builds must
# clear the same thresholds. tests/test_accel_parity.py pins them together.


# ── Ground truth BED parsing ──────────────────────────────────

def parse_truth_bed(path: str) -> list:
    """Parse ground truth BED: chrom, start, end, motif, copies, tier."""
    records = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            records.append({
                "chrom": parts[0],
                "start": int(parts[1]),
                "end": int(parts[2]),
                "motif": parts[3],
                "copies": float(parts[4]),
                "tier": int(parts[5]),
            })
    return records


def parse_fasta_simple(path: str) -> list:
    """Parse FASTA, return list of (name, sequence)."""
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


# ── Motif canonicalization (standalone, no MotifUtils dependency) ──

def canonical_motif(motif: str) -> str:
    """Get canonical rotation of motif (smallest rotation, both strands)."""
    motif = motif.upper()
    if not motif:
        return motif

    # Reduce to primitive period first
    motif = primitive_motif(motif)

    # All rotations of forward strand
    n = len(motif)
    rotations = [motif[i:] + motif[:i] for i in range(n)]

    # Reverse complement
    comp = str.maketrans("ACGT", "TGCA")
    rc = motif[::-1].translate(comp)
    rc_rotations = [rc[i:] + rc[:i] for i in range(n)]

    all_rotations = rotations + rc_rotations
    return min(all_rotations)


def primitive_motif(motif: str) -> str:
    """Reduce motif to its primitive (shortest repeating) unit.
    E.g., AGCAGC -> AGC, ATAT -> AT, AAAA -> A."""
    n = len(motif)
    for p in range(1, n + 1):
        if n % p == 0:
            unit = motif[:p]
            if unit * (n // p) == motif:
                return unit
    return motif


# ── Matching logic ────────────────────────────────────────────

def overlap_ratio(s1: int, e1: int, s2: int, e2: int) -> float:
    """Compute overlap ratio between two intervals."""
    overlap = max(0, min(e1, e2) - max(s1, s2))
    if overlap == 0:
        return 0.0
    span = max(e1 - s1, e2 - s2)
    return overlap / span if span > 0 else 0.0


def periods_compatible(period_a: int, period_b: int) -> bool:
    """Check if two motif periods are compatible.

    Compatible means one divides the other, or they are within 20% of
    each other (to handle imperfect motif extraction).
    """
    if period_a == 0 or period_b == 0:
        return False
    lo, hi = min(period_a, period_b), max(period_a, period_b)
    # One divides the other
    if hi % lo == 0:
        return True
    # Within 20% of each other
    return (hi - lo) / lo <= 0.2


def match_repeats(truth: list, predictions: list, min_overlap: float = 0.5
                  ) -> Tuple[int, int, int, list, list]:
    """Match truth records to predictions.

    A prediction matches a truth record if:
    1. overlap_ratio >= min_overlap
    2. canonical motif matches OR motif periods are compatible

    A single prediction may match multiple adjacent truth records
    (e.g., when the tool merges nearby repeats with the same motif).
    Each prediction is counted as one for FP purposes regardless of
    how many truth records it covers.

    Returns: (TP, FP, FN, missed_truth, extra_preds)
    """
    # Track which predictions matched at least one truth
    pred_matched = set()
    matched_truth = set()

    for ti, t in enumerate(truth):
        best_overlap = 0.0
        best_pi = -1

        for pi, p in enumerate(predictions):
            ov = overlap_ratio(t["start"], t["end"], p.start, p.end)
            if ov < min_overlap:
                continue

            # Compare canonical motifs (strict match)
            t_canon = canonical_motif(t["motif"])
            p_motif = p.consensus_motif or p.motif or ""
            p_canon = canonical_motif(p_motif)

            motif_ok = (t_canon == p_canon)

            # Fallback: period-compatible match (for imperfect repeats)
            if not motif_ok:
                t_period = len(primitive_motif(t["motif"].upper()))
                p_period = len(primitive_motif(p_motif.upper())) if p_motif else 0
                motif_ok = periods_compatible(t_period, p_period)

            if motif_ok and ov > best_overlap:
                best_overlap = ov
                best_pi = pi

        if best_pi >= 0:
            matched_truth.add(ti)
            pred_matched.add(best_pi)

    tp = len(matched_truth)
    fn = len(truth) - tp
    fp = len(predictions) - len(pred_matched)

    missed = [truth[i] for i in range(len(truth)) if i not in matched_truth]
    extra = [predictions[i] for i in range(len(predictions)) if i not in pred_matched]

    return tp, fp, fn, missed, extra


def compute_metrics(tp: int, fp: int, fn: int) -> dict:
    """Compute sensitivity, precision, F1."""
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * sensitivity * precision / (sensitivity + precision)
          if (sensitivity + precision) > 0 else 0.0)
    return {"sensitivity": sensitivity, "precision": precision, "f1": f1}


# ── Runner helper ─────────────────────────────────────────────

def run_finder(fasta_path: str, enabled_tiers: set,
               min_period: int = 1, max_period: int = 100000) -> list:
    """Run TandemRepeatFinder on a FASTA file, return list of TandemRepeat."""
    from src.finder import TandemRepeatFinder

    all_repeats = []
    for name, seq in parse_fasta_simple(fasta_path):
        seq = seq.upper()
        finder = TandemRepeatFinder(
            seq,
            chromosome=name,
            min_period=min_period,
            max_period=max_period,
            enabled_tiers=enabled_tiers,
        )
        repeats = finder.find_all()
        all_repeats.extend(repeats)
        finder.cleanup()

    return all_repeats


def filter_by_tier(truth: list, tier: int) -> list:
    """Filter truth records by tier number."""
    return [t for t in truth if t["tier"] == tier]


def report_results(label: str, tp: int, fp: int, fn: int,
                   missed: list, extra: list):
    """Print detailed results to stdout."""
    metrics = compute_metrics(tp, fp, fn)
    print(f"\n=== {label} Ground Truth Results ===")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Sensitivity: {metrics['sensitivity']:.1%}, "
          f"Precision: {metrics['precision']:.1%}, "
          f"F1: {metrics['f1']:.1%}")
    if missed:
        print("Missed:")
        for m in missed:
            print(f"  {m['chrom']}:{m['start']}-{m['end']} "
                  f"({m['motif']} x{m['copies']})")
    if extra:
        print("Extra:")
        for e in extra[:10]:  # limit output
            motif = e.consensus_motif or e.motif or "?"
            print(f"  {e.chrom}:{e.start}-{e.end} ({motif} x{e.copies:.1f})")


# ── Tier 1 Tests ──────────────────────────────────────────────

class TestTier1GroundTruth:
    """Tier 1 ground truth: sensitivity >= 95%, precision >= 90%."""

    @pytest.fixture(scope="class")
    def tier1_results(self):
        fasta = os.path.join(FIXTURES, "synth_tier1.fa")
        bed = os.path.join(FIXTURES, "synth_tier1_truth.bed")
        truth = parse_truth_bed(bed)
        preds = run_finder(fasta, {"tier1"}, min_period=1, max_period=9)
        tp, fp, fn, missed, extra = match_repeats(truth, preds)
        report_results("Tier 1", tp, fp, fn, missed, extra)
        return tp, fp, fn, missed, extra

    def test_sensitivity(self, tier1_results):
        tp, fp, fn, missed, extra = tier1_results
        metrics = compute_metrics(tp, fp, fn)
        assert metrics["sensitivity"] >= 0.95, (
            f"Tier 1 sensitivity {metrics['sensitivity']:.1%} < 95%. "
            f"Missed: {[(m['motif'], m['start']) for m in missed]}"
        )

    def test_precision(self, tier1_results):
        tp, fp, fn, missed, extra = tier1_results
        metrics = compute_metrics(tp, fp, fn)
        assert metrics["precision"] >= 0.90, (
            f"Tier 1 precision {metrics['precision']:.1%} < 90%. "
            f"FP={fp}"
        )


# ── Tier 2 Tests ──────────────────────────────────────────────

class TestTier2GroundTruth:
    """Tier 2 ground truth: sensitivity >= 90%, precision >= 90%."""

    @pytest.fixture(scope="class")
    def tier2_results(self):
        fasta = os.path.join(FIXTURES, "synth_tier2.fa")
        bed = os.path.join(FIXTURES, "synth_tier2_truth.bed")
        truth = parse_truth_bed(bed)
        preds = run_finder(fasta, {"tier2"}, min_period=10, max_period=100)
        tp, fp, fn, missed, extra = match_repeats(truth, preds)
        report_results("Tier 2", tp, fp, fn, missed, extra)
        return tp, fp, fn, missed, extra

    def test_sensitivity(self, tier2_results):
        tp, fp, fn, missed, extra = tier2_results
        metrics = compute_metrics(tp, fp, fn)
        assert metrics["sensitivity"] >= 0.90, (
            f"Tier 2 sensitivity {metrics['sensitivity']:.1%} < 90%. "
            f"Missed: {[(m['motif'][:20], m['start']) for m in missed]}"
        )

    def test_precision(self, tier2_results):
        tp, fp, fn, missed, extra = tier2_results
        metrics = compute_metrics(tp, fp, fn)
        assert metrics["precision"] >= 0.90, (
            f"Tier 2 precision {metrics['precision']:.1%} < 90%. "
            f"FP={fp}"
        )


# ── Tier 3 Tests ──────────────────────────────────────────────

class TestTier3GroundTruth:
    """Tier 3 ground truth: sensitivity >= 90%, precision >= 90%."""

    @pytest.fixture(scope="class")
    def tier3_results(self):
        fasta = os.path.join(FIXTURES, "synth_tier3.fa")
        bed = os.path.join(FIXTURES, "synth_tier3_truth.bed")
        truth = parse_truth_bed(bed)
        preds = run_finder(fasta, {"tier3"}, min_period=100, max_period=100000)
        tp, fp, fn, missed, extra = match_repeats(truth, preds)
        report_results("Tier 3", tp, fp, fn, missed, extra)
        return tp, fp, fn, missed, extra

    def test_sensitivity(self, tier3_results):
        tp, fp, fn, missed, extra = tier3_results
        metrics = compute_metrics(tp, fp, fn)
        assert metrics["sensitivity"] >= 0.90, (
            f"Tier 3 sensitivity {metrics['sensitivity']:.1%} < 90%. "
            f"Missed: {[(m['motif'][:20], m['start']) for m in missed]}"
        )

    def test_precision(self, tier3_results):
        tp, fp, fn, missed, extra = tier3_results
        metrics = compute_metrics(tp, fp, fn)
        assert metrics["precision"] >= 0.90, (
            f"Tier 3 precision {metrics['precision']:.1%} < 90%. "
            f"FP={fp}"
        )


# ── Mixed Tests ───────────────────────────────────────────────

class TestMixedGroundTruth:
    """Mixed tiers ground truth: per-tier thresholds on a single sequence."""

    @pytest.fixture(scope="class")
    def mixed_results(self):
        fasta = os.path.join(FIXTURES, "synth_mixed.fa")
        bed = os.path.join(FIXTURES, "synth_mixed_truth.bed")
        truth = parse_truth_bed(bed)
        preds = run_finder(fasta, {"tier1", "tier2", "tier3"})
        return truth, preds

    def test_tier1_sensitivity(self, mixed_results):
        truth, preds = mixed_results
        t1_truth = filter_by_tier(truth, 1)
        tp, fp, fn, missed, extra = match_repeats(t1_truth, preds)
        report_results("Mixed-Tier1", tp, fp, fn, missed, extra)
        metrics = compute_metrics(tp, fp, fn)
        assert metrics["sensitivity"] >= 0.95, (
            f"Mixed Tier 1 sensitivity {metrics['sensitivity']:.1%} < 95%"
        )

    def test_tier2_sensitivity(self, mixed_results):
        truth, preds = mixed_results
        t2_truth = filter_by_tier(truth, 2)
        tp, fp, fn, missed, extra = match_repeats(t2_truth, preds)
        report_results("Mixed-Tier2", tp, fp, fn, missed, extra)
        metrics = compute_metrics(tp, fp, fn)
        assert metrics["sensitivity"] >= 0.70, (
            f"Mixed Tier 2 sensitivity {metrics['sensitivity']:.1%} < 70%"
        )

    def test_tier3_sensitivity(self, mixed_results):
        truth, preds = mixed_results
        t3_truth = filter_by_tier(truth, 3)
        tp, fp, fn, missed, extra = match_repeats(t3_truth, preds)
        report_results("Mixed-Tier3", tp, fp, fn, missed, extra)
        metrics = compute_metrics(tp, fp, fn)
        assert metrics["sensitivity"] >= 0.70, (
            f"Mixed Tier 3 sensitivity {metrics['sensitivity']:.1%} < 70%"
        )


# ── Adjacent/Edge Case Tests ─────────────────────────────────

class TestAdjacentGroundTruth:
    """Adjacent repeat edge cases: overall sensitivity >= 95%, precision >= 90%."""

    @pytest.fixture(scope="class")
    def adjacent_results(self):
        fasta = os.path.join(FIXTURES, "synth_adjacent.fa")
        bed = os.path.join(FIXTURES, "synth_adjacent_truth.bed")
        truth = parse_truth_bed(bed)
        preds = run_finder(fasta, {"tier1", "tier2", "tier3"})
        tp, fp, fn, missed, extra = match_repeats(truth, preds)
        report_results("Adjacent", tp, fp, fn, missed, extra)
        return tp, fp, fn, missed, extra

    def test_sensitivity(self, adjacent_results):
        tp, fp, fn, missed, extra = adjacent_results
        metrics = compute_metrics(tp, fp, fn)
        assert metrics["sensitivity"] >= 0.95, (
            f"Adjacent sensitivity {metrics['sensitivity']:.1%} < 95%. "
            f"Missed: {[(m['motif'][:20], m['start']) for m in missed]}"
        )

    def test_precision(self, adjacent_results):
        tp, fp, fn, missed, extra = adjacent_results
        metrics = compute_metrics(tp, fp, fn)
        assert metrics["precision"] >= 0.90, (
            f"Adjacent precision {metrics['precision']:.1%} < 90%. "
            f"FP={fp}"
        )
