#!/usr/bin/env python3
"""Test the sliding window method directly"""

import numpy as np
from bwt import Tier1STRFinder

# Simple test sequence with AT repeat
seq = "AAAAAAAAAAA" + "ATATATATATAT" + "GGGGGGGGGG"
print(f"Test sequence: {seq}")
print(f"Length: {len(seq)}")
print()

# Convert to numpy array
text_arr = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)

# Create Tier1 finder
tier1 = Tier1STRFinder(text_arr, max_motif_length=9, show_progress=False)
tier1.min_copies = 3
tier1.min_entropy = 1.0
tier1.min_array_length = 6

# Find repeats
repeats = tier1.find_strs("test_chrom")

print(f"Found {len(repeats)} repeats:")
for r in repeats:
    actual = seq[r.start:r.end]
    print(f"  {r.start}-{r.end}: {r.motif} x {r.copies:.1f} = {actual}")
