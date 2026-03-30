#!/usr/bin/env python3
"""Test both Tier1 and Tier2 together"""

import numpy as np
from bwt import Tier1STRFinder, Tier2STRFinder

# Test sequence with known repeats
seq = "AAAAAAAAAAATATATATATATATATGGGGGGGGGGCACACACACACACACACATTTTTTTTTTAGCAGCAGCAGCAGCAGCCCCCCCCCCCAGTCAGTCAGTCAGTCAGTCNNNNNNNNNNATGCATGCATGCATGCATGC"

print(f"Test sequence length: {len(seq)}")
print(f"Sequence: {seq}")
print()

# Convert to numpy array
text_arr = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)

# Run Tier1 first (perfect repeats using sliding window)
print("="*60)
print("Running Tier1 (perfect repeats)...")
print("="*60)
tier1 = Tier1STRFinder(text_arr, max_motif_length=9, show_progress=False)
tier1.min_copies = 3
tier1.min_entropy = 1.0
tier1.min_array_length = 6

tier1_repeats = tier1.find_strs("synthetic_chromosome")
tier1_seen = set((r.start, r.end) for r in tier1_repeats)

print(f"Tier1 found {len(tier1_repeats)} repeats:")
for r in tier1_repeats:
    actual = seq[r.start:r.end]
    print(f"  Tier{r.tier}: {r.start:3d}-{r.end:3d}: [{r.motif}]n x {r.copies:.1f} = {actual}")
print()

# Run Tier2 (imperfect repeats using BWT/FM-index)
print("="*60)
print("Running Tier2 (imperfect repeats)...")
print("="*60)

# Import BWTCore for Tier2
from bwt import BWTCore

# Build BWT
seq_with_sentinel = seq + '$'
bwt = BWTCore(seq_with_sentinel, 32)

tier2 = Tier2STRFinder(bwt, max_short_motif=9, allow_mismatches=True, show_progress=False)
tier2.min_copies = 3

tier2_repeats = tier2.find_short_imperfect_repeats("synthetic_chromosome", tier1_seen)

print(f"Tier2 found {len(tier2_repeats)} additional repeats:")
for r in tier2_repeats:
    actual = seq[r.start:r.end]
    print(f"  Tier{r.tier}: {r.start:3d}-{r.end:3d}: [{r.motif}]n x {r.copies:.1f} ({r.percent_matches:.1f}% match) = {actual}")
print()

# Combine and show all
all_repeats = tier1_repeats + tier2_repeats
all_repeats.sort(key=lambda r: (r.start, r.end))

print("="*60)
print(f"Total: {len(all_repeats)} repeats")
print("="*60)
for r in all_repeats:
    actual = seq[r.start:r.end]
    match_info = f" ({r.percent_matches:.1f}%)" if hasattr(r, 'percent_matches') and r.percent_matches else ""
    print(f"  Tier{r.tier}: {r.start:3d}-{r.end:3d}: [{r.motif}]n x {r.copies:.1f}{match_info} = {actual}")
