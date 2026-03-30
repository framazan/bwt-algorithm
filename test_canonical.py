#!/usr/bin/env python3
"""Test script to diagnose why TCATCGG repeat is not being detected."""

def reverse_complement(seq):
    """Return reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement.get(base, base) for base in reversed(seq))

def get_canonical_motif(motif):
    """Get lexicographically smallest rotation across both strands."""
    if not motif:
        return motif

    # Generate all rotations of forward strand
    rotations = [motif[i:] + motif[:i] for i in range(len(motif))]

    # Generate all rotations of reverse complement
    rc = reverse_complement(motif)
    rc_rotations = [rc[i:] + rc[:i] for i in range(len(rc))]

    # Return lexicographically smallest
    all_forms = rotations + rc_rotations
    return min(all_forms)

def is_primitive_motif(motif):
    """Check if motif is primitive (not composed of smaller repeating unit)."""
    n = len(motif)
    # Try all possible period lengths
    for period in range(1, n):
        if n % period == 0:
            # Check if motif is period repeated
            unit = motif[:period]
            if unit * (n // period) == motif:
                return False  # Not primitive
    return True

# Test the sequence
sequence = "TCATCGGTCATCGGTCATCGGTCATCGGTCATCGG"
motif = "TCATCGG"

print(f"Testing sequence: {sequence}")
print(f"Motif: {motif} (length {len(motif)})")
print()

# Check if motif is primitive
is_prim = is_primitive_motif(motif)
print(f"Is '{motif}' primitive? {is_prim}")
print()

# Get canonical form
canonical = get_canonical_motif(motif)
print(f"Canonical form of '{motif}': {canonical}")
print()

# Generate all rotations to see which one is canonical
print("All rotations of forward strand:")
for i in range(len(motif)):
    rotation = motif[i:] + motif[:i]
    print(f"  {rotation}")

print()
print("Reverse complement:", reverse_complement(motif))
print("All rotations of reverse complement:")
rc = reverse_complement(motif)
for i in range(len(rc)):
    rotation = rc[i:] + rc[:i]
    print(f"  {rotation}")

print()
print(f"Lexicographically smallest (canonical): {canonical}")
print()

# Check if canonical form appears in sequence
print(f"Does canonical '{canonical}' appear in sequence?")
for i in range(len(sequence) - len(canonical) + 1):
    substr = sequence[i:i+len(canonical)]
    if substr == canonical:
        print(f"  YES at position {i}: {substr}")

# Check all 7-mers in the sequence
print()
print("All 7-mers in the sequence:")
for i in range(len(sequence) - 7 + 1):
    kmer = sequence[i:i+7]
    canonical_kmer = get_canonical_motif(kmer)
    print(f"  Position {i}: {kmer} -> canonical: {canonical_kmer}")
