#!/usr/bin/env python3
"""Verify that the rotation fix is complete."""

def reverse_complement(seq):
    """Return reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement.get(base, base) for base in reversed(seq))

def get_all_search_rotations(canonical_motif):
    """Get all rotations that should be searched (forward + RC)."""
    # Forward rotations
    forward_rotations = [canonical_motif[i:] + canonical_motif[:i]
                        for i in range(len(canonical_motif))]

    # RC rotations
    rc_motif = reverse_complement(canonical_motif)
    rc_rotations = [rc_motif[i:] + rc_motif[:i]
                   for i in range(len(rc_motif))]

    # Combine and deduplicate
    all_rotations = list(set(forward_rotations + rc_rotations))
    return sorted(all_rotations)

# Test case
canonical = "ACCGATG"
observed = "TCATCGG"
sequence = "TCATCGGTCATCGGTCATCGGTCATCGGTCATCGG"

print(f"Canonical motif: {canonical}")
print(f"Observed motif in sequence: {observed}")
print(f"Test sequence: {sequence}")
print()

# Get all rotations that will be searched
all_rotations = get_all_search_rotations(canonical)

print(f"All rotations to be searched ({len(all_rotations)} total):")
for i, rot in enumerate(all_rotations, 1):
    marker = " ← OBSERVED!" if rot == observed else ""
    print(f"  {i:2d}. {rot}{marker}")

print()

# Verify observed motif will be found
if observed in all_rotations:
    print(f"✓ SUCCESS: Observed motif '{observed}' WILL be found in search!")

    # Find where it appears in sequence
    positions = []
    for i in range(len(sequence) - len(observed) + 1):
        if sequence[i:i+len(observed)] == observed:
            positions.append(i)

    print(f"✓ Appears at positions: {positions}")
    print(f"✓ That's {len(positions)} copies (expected: 5)")

    if len(positions) == 5:
        print("✓ Correct number of copies!")
    else:
        print(f"✗ Wrong number! Expected 5, found {len(positions)}")
else:
    print(f"✗ FAIL: Observed motif '{observed}' will NOT be found!")

print()
print("Summary of fix:")
print(f"- Forward rotations of '{canonical}': {len([canonical[i:] + canonical[:i] for i in range(len(canonical))])} rotations")
print(f"- RC rotations of '{canonical}': {len([reverse_complement(canonical)[i:] + reverse_complement(canonical)[:i] for i in range(len(canonical))])} rotations")
print(f"- Total unique rotations: {len(all_rotations)}")
print(f"- Includes observed motif: {'YES' if observed in all_rotations else 'NO'}")
