#!/usr/bin/env python3
"""Debug script to test motif search"""

import sys
sys.path.insert(0, '.')

from bwt import BWTCore, Tier1STRFinder, MotifUtils

# Test sequence
seq = "AAAAAAAAAAATATATATATATATATGGGGGGGGGGCACACACACACACACACATTTTTTTTTTAGCAGCAGCAGCAGCAGCCCCCCCCCCCAGTCAGTCAGTCAGTCAGTCNNNNNNNNNNATGCATGCATGCATGCATGC"

print(f"Sequence length: {len(seq)}")
print(f"Sequence: {seq[:50]}...")
print()

# Build BWT
seq_with_sentinel = seq + '$'
bwt = BWTCore(seq_with_sentinel, 32)
print(f"BWT built successfully")
print()

# Test motif search
test_motifs = ["AT", "CA", "AGC", "AGTC", "ATGC"]

for motif in test_motifs:
    print(f"Testing motif: {motif}")

    # Check entropy
    entropy = MotifUtils.calculate_entropy(motif)
    print(f"  Entropy: {entropy:.3f} bits (min=1.0)")

    # Check if primitive
    is_prim = MotifUtils.is_primitive_motif(motif)
    print(f"  Primitive: {is_prim}")

    # Search for motif
    positions = bwt.get_kmer_positions(motif)
    print(f"  Positions found: {len(positions)}")
    print(f"  Positions: {positions}")

    # Check rotations
    rotations = [motif[i:] + motif[:i] for i in range(len(motif))]
    rc_motif = MotifUtils.reverse_complement(motif)
    rc_rotations = [rc_motif[i:] + rc_motif[:i] for i in range(len(rc_motif))]
    all_rots = list(set(rotations + rc_rotations))
    print(f"  All rotations: {all_rots}")

    all_pos = []
    for rot in all_rots:
        rot_pos = bwt.get_kmer_positions(rot)
        print(f"    {rot}: {len(rot_pos)} positions")
        all_pos.extend(rot_pos)

    unique_pos = sorted(set(all_pos))
    print(f"  Total unique positions: {len(unique_pos)}")
    print()

# Now run Tier1 finder
print("="*60)
print("Running Tier1STRFinder...")
print("="*60)
tier1 = Tier1STRFinder(bwt, max_motif_length=9, allow_mismatches=True)
tier1.min_copies = 3
tier1.min_entropy = 1.0

repeats = tier1.find_strs("synthetic_chromosome")
print(f"\nFound {len(repeats)} repeats:")
for r in repeats:
    print(f"  {r.start}-{r.end}: {r.motif} x {r.copies:.1f} ({r.length}bp)")
