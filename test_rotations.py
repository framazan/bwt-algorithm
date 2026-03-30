#!/usr/bin/env python3
"""Test what rotations should be searched."""

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

# Test motif
observed_motif = "TCATCGG"
canonical = get_canonical_motif(observed_motif)

print(f"Observed motif: {observed_motif}")
print(f"Canonical motif: {canonical}")
print()

# What rotations does the current code search?
print("Current code searches (forward rotations of canonical only):")
for i in range(len(canonical)):
    rotation = canonical[i:] + canonical[:i]
    print(f"  {rotation}")
print()

# What SHOULD be searched?
print("Should search (all rotations of both canonical AND its RC):")
rotations_fwd = [canonical[i:] + canonical[:i] for i in range(len(canonical))]
rc_canonical = reverse_complement(canonical)
rotations_rc = [rc_canonical[i:] + rc_canonical[:i] for i in range(len(rc_canonical))]

print("Forward rotations of canonical:")
for rotation in rotations_fwd:
    if rotation == observed_motif:
        print(f"  {rotation} ← OBSERVED MOTIF!")
    else:
        print(f"  {rotation}")

print()
print("RC rotations of canonical:")
for rotation in rotations_rc:
    if rotation == observed_motif:
        print(f"  {rotation} ← OBSERVED MOTIF!")
    else:
        print(f"  {rotation}")

print()
print("Finding observed motif:")
if observed_motif in rotations_fwd:
    print(f"✗ Found in forward rotations")
elif observed_motif in rotations_rc:
    print(f"✓ Found in RC rotations")
else:
    print(f"✗ NOT FOUND in either set!")
