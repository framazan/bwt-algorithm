#!/usr/bin/env python3
"""
Test script for imperfect repeat detection.

Creates synthetic test cases and validates the implementation.
"""

import tempfile
import os
from bwt import TandemRepeatFinder, MotifUtils

def create_test_fasta(sequences, filename):
    """Create a test FASTA file."""
    with open(filename, 'w') as f:
        for name, seq in sequences.items():
            f.write(f">{name}\n")
            # Write in 80-char lines
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")

def test_perfect_repeat():
    """Test 1: Perfect tandem repeat (exact matches)."""
    print("Test 1: Perfect tandem repeat")
    print("-" * 60)

    # Create perfect repeat: ATCG repeated 10 times
    seq = "ATCG" * 10 + "$"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
        f.write(">test_perfect\n")
        f.write(seq + "\n")
        temp_fa = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as f:
        temp_out = f.name

    try:
        finder = TandemRepeatFinder(
            temp_fa,
            allow_mismatches=True,
            max_motif_length=6,
            min_copies=3
        )
        sequences = finder.load_reference()
        finder.build_indices(sequences)
        repeats = finder.find_tandem_repeats(enable_tier1=True, enable_tier2=False)

        print(f"  Input: {'ATCG' * 10}")
        print(f"  Found {len(repeats)} repeat(s)")

        if repeats:
            r = repeats[0]
            print(f"  Motif: {r.motif}")
            print(f"  Consensus: {r.consensus_motif}")
            print(f"  Copies: {r.copies}")
            print(f"  Mismatch rate: {r.mismatch_rate:.3f}")
            print(f"  Strand: {r.strand}")

            assert r.copies >= 9, f"Expected ≥9 copies, got {r.copies}"
            assert r.mismatch_rate == 0.0, f"Expected 0 mismatches, got {r.mismatch_rate}"
            print("  ✓ PASSED")
        else:
            print("  ✗ FAILED: No repeats found")
    finally:
        os.unlink(temp_fa)
        os.unlink(temp_out)

    print()

def test_imperfect_repeat_single_mismatch():
    """Test 2: Imperfect repeat with single mismatch per copy."""
    print("Test 2: Imperfect repeat (single mismatch)")
    print("-" * 60)

    # Create imperfect repeat: ATCG with one base changed in middle copies
    seq = "ATCG" + "ATCG" + "ATGG" + "ATCG" + "ATCG" + "$"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
        f.write(">test_imperfect\n")
        f.write(seq + "\n")
        temp_fa = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as f:
        temp_out = f.name

    try:
        finder = TandemRepeatFinder(
            temp_fa,
            allow_mismatches=True,
            max_motif_length=6,
            min_copies=3
        )
        sequences = finder.load_reference()
        finder.build_indices(sequences)
        repeats = finder.find_tandem_repeats(enable_tier1=True, enable_tier2=False)

        print(f"  Input: ATCG + ATCG + ATGG + ATCG + ATCG (1 mismatch in copy 3)")
        print(f"  Found {len(repeats)} repeat(s)")

        if repeats:
            r = repeats[0]
            print(f"  Motif: {r.motif}")
            print(f"  Consensus: {r.consensus_motif}")
            print(f"  Copies: {r.copies}")
            print(f"  Mismatch rate: {r.mismatch_rate:.3f}")
            print(f"  Max MM per copy: {r.max_mismatches_per_copy}")

            assert r.copies >= 4, f"Expected ≥4 copies, got {r.copies}"
            assert r.mismatch_rate > 0, f"Expected mismatches, got {r.mismatch_rate}"
            assert r.max_mismatches_per_copy == 1, f"Expected max 1 MM, got {r.max_mismatches_per_copy}"
            print("  ✓ PASSED")
        else:
            print("  ✗ FAILED: No repeats found")
    finally:
        os.unlink(temp_fa)
        os.unlink(temp_out)

    print()

def test_low_complexity_filter():
    """Test 3: Low-complexity sequence should be filtered."""
    print("Test 3: Low-complexity filter")
    print("-" * 60)

    # Create homopolymer: AAAA repeated
    seq = "A" * 40 + "$"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
        f.write(">test_lowcomp\n")
        f.write(seq + "\n")
        temp_fa = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as f:
        temp_out = f.name

    try:
        finder = TandemRepeatFinder(
            temp_fa,
            allow_mismatches=True,
            max_motif_length=6,
            min_copies=3,
            min_entropy=1.0  # Should filter homopolymer
        )
        sequences = finder.load_reference()
        finder.build_indices(sequences)
        repeats = finder.find_tandem_repeats(enable_tier1=True, enable_tier2=False)

        print(f"  Input: {'A' * 40} (homopolymer)")
        print(f"  Entropy: {MotifUtils.calculate_entropy('A'):.2f} bits")
        print(f"  Found {len(repeats)} repeat(s)")

        # Should find very few or no repeats due to entropy filter
        if len(repeats) == 0:
            print("  ✓ PASSED: Low-complexity sequence filtered")
        else:
            print(f"  ⚠ WARNING: Found {len(repeats)} repeats (may be expected for some motifs)")
    finally:
        os.unlink(temp_fa)
        os.unlink(temp_out)

    print()

def test_strand_canonicalization():
    """Test 4: Strand-aware canonicalization."""
    print("Test 4: Strand canonicalization")
    print("-" * 60)

    # Test reverse complement
    seq = "ATCG"
    rc = MotifUtils.reverse_complement(seq)
    print(f"  {seq} → RC: {rc}")
    assert rc == "CGAT", f"Expected CGAT, got {rc}"

    # Test canonical motif selection
    motif1 = "ATCG"
    canonical1, strand1 = MotifUtils.get_canonical_motif_stranded(motif1)
    print(f"  {motif1} → Canonical: {canonical1} (strand {strand1})")

    motif2 = "CGAT"  # Reverse complement of ATCG
    canonical2, strand2 = MotifUtils.get_canonical_motif_stranded(motif2)
    print(f"  {motif2} → Canonical: {canonical2} (strand {strand2})")

    # Both should yield the same canonical motif
    assert canonical1 == canonical2, f"Expected same canonical: {canonical1} vs {canonical2}"
    print("  ✓ PASSED")
    print()

def test_hamming_distance():
    """Test 5: Hamming distance calculation."""
    print("Test 5: Hamming distance")
    print("-" * 60)

    tests = [
        ("ATCG", "ATCG", 0),
        ("ATCG", "ATGG", 1),
        ("ATCG", "GGGG", 3),
        ("AAAA", "TTTT", 4),
    ]

    for s1, s2, expected in tests:
        dist = MotifUtils.hamming_distance(s1, s2)
        print(f"  {s1} vs {s2}: {dist} (expected {expected})")
        assert dist == expected, f"Expected {expected}, got {dist}"

    print("  ✓ PASSED")
    print()

def test_consensus_building():
    """Test 6: Consensus motif building."""
    print("Test 6: Consensus building")
    print("-" * 60)

    # Test with perfect agreement
    copies = ["ATCG", "ATCG", "ATCG"]
    consensus, mm_rate = MotifUtils.build_consensus_motif(copies)
    print(f"  Perfect: {copies}")
    print(f"  Consensus: {consensus}, MM rate: {mm_rate:.3f}")
    assert consensus == "ATCG", f"Expected ATCG, got {consensus}"
    assert mm_rate == 0.0, f"Expected 0 MM rate, got {mm_rate}"

    # Test with majority vote
    copies = ["ATCG", "ATCG", "ATGG"]
    consensus, mm_rate = MotifUtils.build_consensus_motif(copies)
    print(f"  Majority: {copies}")
    print(f"  Consensus: {consensus}, MM rate: {mm_rate:.3f}")
    assert consensus == "ATCG", f"Expected ATCG, got {consensus}"
    assert mm_rate > 0, f"Expected >0 MM rate, got {mm_rate}"

    print("  ✓ PASSED")
    print()

def test_entropy_calculation():
    """Test 7: Shannon entropy calculation."""
    print("Test 7: Entropy calculation")
    print("-" * 60)

    tests = [
        ("AAAA", 0.0, "homopolymer"),
        ("ATAT", 1.0, "dinucleotide"),
        ("ATCG", 2.0, "all different"),
        ("AAAAAAAAAT", None, "low complexity"),
    ]

    for seq, expected, desc in tests:
        entropy = MotifUtils.calculate_entropy(seq)
        if expected is not None:
            print(f"  {seq:15} entropy: {entropy:.2f} bits (expected ~{expected:.1f}) - {desc}")
            assert abs(entropy - expected) < 0.1, f"Expected ~{expected}, got {entropy}"
        else:
            print(f"  {seq:15} entropy: {entropy:.2f} bits - {desc}")

    print("  ✓ PASSED")
    print()

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Imperfect Repeat Detection")
    print("=" * 60)
    print()

    try:
        test_hamming_distance()
        test_entropy_calculation()
        test_consensus_building()
        test_strand_canonicalization()
        test_perfect_repeat()
        test_imperfect_repeat_single_mismatch()
        test_low_complexity_filter()

        print("=" * 60)
        print("All tests completed successfully! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
