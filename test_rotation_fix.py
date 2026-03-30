#!/usr/bin/env python3
"""Test script to verify the rotation-aware seeding fix."""

# Create test FASTA file
test_sequence = """>test_rotation
TCATCGGTCATCGGTCATCGGTCATCGGTCATCGG
"""

with open("test_rotation_input.fa", "w") as f:
    f.write(test_sequence)

print("Created test file: test_rotation_input.fa")
print()
print("Test sequence: TCATCGGTCATCGGTCATCGGTCATCGGTCATCGG")
print("Expected: 5 copies of 7bp motif")
print()
print("To test, run:")
print("  python bwt.py test_rotation_input.fa --tier1 --format bed -o test_rotation_output.bed")
print()
print("Expected output in test_rotation_output.bed:")
print("  test_rotation  0  35  ACCGATG  5.0  1  0.000  +")
print()
print("Note: The canonical motif is ACCGATG, even though the sequence uses rotation TCATCGG")
