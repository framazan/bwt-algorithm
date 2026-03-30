#!/usr/bin/env python3
"""Debug script to understand consensus building for the test sequence."""

# Test sequence
sequence = "TCATCGGTCATCGGTCATCGGTCATCGGTCATCGG"
print(f"Sequence: {sequence}")
print(f"Length: {len(sequence)} bp")
print()

# Expected: 5 copies of 7bp motif TCATCGG
motif_len = 7
expected_motif = "TCATCGG"
print(f"Expected motif: {expected_motif}")
print()

# Let me simulate what the consensus building would do
# Assuming seed position is 2 (where ATCGGTC starts)
seed_pos = 2
print(f"If seed is at position {seed_pos}: {sequence[seed_pos:seed_pos+motif_len]}")
print()

# Extract all 7-bp windows starting at position 2
print("If we start from position 2 and collect 7bp copies:")
for i in range(5):
    start = seed_pos + i * motif_len
    end = start + motif_len
    if end <= len(sequence):
        copy = sequence[start:end]
        print(f"Copy {i+1} (pos {start}-{end}): {copy}")
print()

# Now let's start from position 0 (the correct starting point)
print("If we start from position 0 (correct) and collect 7bp copies:")
for i in range(5):
    start = i * motif_len
    end = start + motif_len
    if end <= len(sequence):
        copy = sequence[start:end]
        print(f"Copy {i+1} (pos {start}-{end}): {copy}")
print()

# The issue: if seed is at position 2, and extension doesn't go left far enough
# The tool finds seed at position 2 (ATCGGTC)
# Then extends right: position 9 (ATCGGTC), position 16 (ATCGGTC), position 23 (ATCGGTC)
# But doesn't extend left to position 0!

print("Problem diagnosis:")
print("1. Seed found at position 2: 'ATCGGTC' (a rotation of canonical 'ACCGATG')")
print("2. Extension right finds copies at positions 9, 16, 23")
print("3. Extension left should find position -5, but that's negative, so stops")
print("4. Result: 4 copies starting at position 2, motif 'ATCGGTC'")
print()
print("Why doesn't it find position 0?")
print(f"Position 0: {sequence[0:7]} = 'TCATCGG'")
print(f"Position 2: {sequence[2:9]} = 'ATCGGTC'")
print("Hamming distance between them:", sum(1 for a, b in zip(sequence[0:7], sequence[2:9]) if a != b))
print()

# Let's check if there's a seed at position 0
# We're searching for all rotations, so should find 'TCATCGG'
print("Rotations being searched:")
canonical = "ACCGATG"
for i in range(7):
    rotation = canonical[i:] + canonical[:i]
    print(f"  {rotation}")
print()
print("Does 'TCATCGG' appear in rotations? Let's check reverse complement too...")
def reverse_complement(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement.get(base, base) for base in reversed(seq))

rc = reverse_complement(canonical)
print(f"RC of canonical: {rc}")
print("RC rotations:")
for i in range(7):
    rotation = rc[i:] + rc[:i]
    print(f"  {rotation}")
    if rotation == "TCATCGG":
        print(f"  ^^^ FOUND! 'TCATCGG' is a rotation of the RC!")
