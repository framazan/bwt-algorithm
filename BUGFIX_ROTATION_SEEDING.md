# Bug Fix: Rotation-Aware Seeding and Consensus Reporting

## Issues

### Issue 1: Missing Repeats (Rotation-Blind Seeding)

The tool was failing to detect tandem repeats when the array started with a non-canonical rotation of the motif.

### Issue 2: Wrong Motif Reported (Canonical Instead of Observed)

The tool was reporting the canonical form of the motif instead of the actual observed sequence.

### Example Case

**Sequence:**
```
>0
TCATCGGTCATCGGTCATCGGTCATCGGTCATCGG
```

This is 5 perfect copies of the 7bp motif "TCATCGG" (35bp total).

**Problems:**
1. Tool was **not detecting it** at all (Issue #1)
2. When fixed, tool reported `7[ACCGATG]4` instead of `7[TCATCGG]5` (Issue #2)

## Root Cause

The tool uses canonical motifs to avoid redundancy. For "TCATCGG":

**Forward rotations:**
- TCATCGG
- CATCGGT
- ATCGGTC
- TCGGTCA
- CGGTCAT
- GGTCATC
- GTCATCG

**Reverse complement:** CCGATGA

**RC rotations:**
- CCGATGA
- CGATGAC
- GATGACC
- ATGACCG
- TGACCGA
- GACCGAT
- **ACCGATG** ← Lexicographically smallest (canonical)

The tool only enumerated **"ACCGATG"** as the canonical form, and the seeding phase searched for exact occurrences of **"ACCGATG"** only.

**Problem:** "ACCGATG" does not appear in the sequence `TCATCGGTCATCGGTCATCGGTCATCGGTCATCGG`, so no seeds were found, and the repeat was missed.

## Solutions

### Solution 1: Rotation-Aware Seeding (Including RC Rotations)

Modified the seeding phase in `Tier1STRFinder.find_strs()` to search for **all rotations of both the canonical motif AND its reverse complement**, not just rotations of the canonical form itself.

**Key insight**: A canonical motif is the lexicographically smallest among ALL rotations of BOTH strands. But the original fix only searched forward rotations, missing reverse complement rotations.

### Solution 2: Report Observed Consensus

Modified `Tier1STRFinder._find_tandems_with_mismatches()` to store the **observed consensus** built from the actual sequence, not the canonical form.

### Code Changes

#### Change 1: Rotation-Aware Seeding (lines 867-896)

**Before:**
```python
# Fast k-mer lookup for short motifs (performance optimization)
if k <= 8:
    # Use hash table for O(1) lookup
    positions = self.bwt.get_kmer_positions(motif)  # Only canonical!
    count = len(positions)
else:
    # Use FM-index for longer motifs
    count = self.bwt.count_occurrences(motif)  # Only canonical!
    positions = self.bwt.locate_positions(motif) if count >= self.min_copies else []
```

**After:**
```python
# Search for all rotations of the canonical motif AND its reverse complement
# This is necessary because a tandem repeat can start at any rotation
# and can appear on either strand
all_positions = []

# Generate rotations of forward canonical motif
forward_rotations = [motif[i:] + motif[:i] for i in range(len(motif))]

# Generate rotations of reverse complement
rc_motif = MotifUtils.reverse_complement(motif)
rc_rotations = [rc_motif[i:] + rc_motif[:i] for i in range(len(rc_motif))]

# Combine and deduplicate (some motifs are palindromic)
rotations = list(set(forward_rotations + rc_rotations))

for rotation in rotations:
    # Fast k-mer lookup for short motifs (performance optimization)
    if k <= 8:
        # Use hash table for O(1) lookup
        rot_positions = self.bwt.get_kmer_positions(rotation)
    else:
        # Use FM-index for longer motifs
        rot_count = self.bwt.count_occurrences(rotation)
        rot_positions = self.bwt.locate_positions(rotation) if rot_count > 0 else []

    all_positions.extend(rot_positions)

# Remove duplicates and sort
positions = sorted(set(all_positions))
count = len(positions)
```

#### Change 2: Report Observed Consensus (lines 934-959)

**Before:**
```python
consensus_str = consensus_arr.tobytes().decode('ascii', errors='replace')

# Get canonical motif considering both strands
canonical, strand = MotifUtils.get_canonical_motif_stranded(consensus_str)

# ... maximality check ...

repeat = TandemRepeat(
    chrom=chromosome,
    start=start_pos,
    end=end_pos,
    motif=motif,  # Wrong: uses canonical search motif!
    copies=copies,
    ...
    consensus_motif=canonical,  # Wrong: uses canonicalized form!
```

**After:**
```python
consensus_str = consensus_arr.tobytes().decode('ascii', errors='replace')

# Get canonical motif considering both strands (for deduplication only)
canonical, strand = MotifUtils.get_canonical_motif_stranded(consensus_str)

# ... maximality check ...

repeat = TandemRepeat(
    chrom=chromosome,
    start=start_pos,
    end=end_pos,
    motif=consensus_str,  # Fixed: uses observed consensus!
    copies=copies,
    ...
    consensus_motif=consensus_str,  # Fixed: uses observed consensus!
```

## Impact

### Correctness
- **Fixed Issue #1:** Tandem repeats starting with any rotation are now correctly detected
- **Fixed Issue #2:** Reports observed motif instead of canonical form
- **Example:** `TCATCGGTCATCGGTCATCGGTCATCGGTCATCGG` will now be detected and reported as:
  - Period: 7bp
  - Motif: `TCATCGG` (observed form, not canonical `ACCGATG`)
  - Copies: 5
  - Mismatches: 0

### Performance
- **Overhead:** For a k-mer of length k, searches k rotations instead of 1
  - k=1: 1× (no change for mononucleotides)
  - k=4: 4× searches
  - k=7: 7× searches
  - k=10: 10× searches

- **Mitigation:**
  - Hash table lookup is O(1), so overhead is linear in k
  - FM-index search is O(k), so overhead is O(k²)
  - Most time is spent in extension phase, not seeding
  - De-duplication ensures each position is only extended once

### Expected Performance Impact
- **Tier 1 runtime:** ~4-6× slower due to searching all rotations of BOTH forward and RC
  - Each canonical motif now searches up to 2k rotations (k forward + k RC, minus duplicates for palindromes)
  - k=7: up to 14 searches (was 7 in first fix, was 1 originally)
- **Trade-off:** Correctness vs speed
- **Mitigation:** Most motifs are not palindromic, so we get ~2k distinct rotations
- **Benchmark needed:** Test on chr22 to measure actual impact

## Testing

### Test Case 1: Non-canonical rotation
```
>test_case_1
TCATCGGTCATCGGTCATCGGTCATCGGTCATCGG
```
**Expected Output:**
- Detects: Yes (Issue #1 fixed)
- Period: 7bp
- Motif reported: `TCATCGG` (Issue #2 fixed - observed, not canonical)
- Copies: 5
- Mismatch rate: 0.000

### Test Case 2: Canonical rotation
```
>test_case_2
ACCGATGACCGATGACCGATGACCGATGACCGATG
```
**Expected:** Detects 5 copies of 7bp motif (canonical: ACCGATG, observed: ACCGATG)

### Test Case 3: Mixed rotations (should not happen naturally but tests robustness)
```
>test_case_3
TCATCGGACCGATGCGGTCAT
```
**Expected:** May detect partial repeats or no repeats (depending on extension logic)

## Alternative Solutions Considered

### Option 1: Keep canonical-only seeding, fix extension
**Rejected:** Extension logic already handles mismatches, but won't help if no seed is found

### Option 2: Enumerate all rotations in `enumerate_motifs()`
**Rejected:** Would break canonical motif deduplication and increase redundancy

### Option 3: Current solution (rotation-aware seeding)
**Accepted:** Maintains canonical enumeration, fixes seeding phase

## First Fix Attempt (Incomplete)

**Initial fix**: Only searched forward rotations of canonical motif
```python
rotations = [motif[i:] + motif[:i] for i in range(len(motif))]
```

**Problem**: This found rotations like "ATCGGTC" but missed "TCATCGG" because:
- "TCATCGG" is a rotation of the **reverse complement**, not the forward strand
- Canonical "ACCGATG" → RC "CATCGGT" → rotation "TCATCGG"

**Result**: Partial fix that found some repeats but reported wrong motifs

## Related Issues

This bug would affect:
- Any tandem repeat where the array starts at a non-canonical rotation
- Especially affects repeats where the observed form is a rotation of the RC strand
- Estimated impact: ~13/14 of tandem repeats for 7-mers (only 1 rotation is canonical, and need both strands)
- More severe for longer motifs

## Follow-up Work

1. **Benchmark performance impact** on chr22
2. **Optimize rotation search** (only search forward rotations, not RC, since RC has different canonical form)
3. **Add test cases** to ensure all rotations are detected correctly
4. **Document in README** that seeding is now rotation-aware

## Version

- **Fixed in:** Current version (post-fix)
- **Affects:** All previous versions using canonical motif enumeration
- **Breaking change:** No (only fixes missing results)
- **Output change:** More repeats will be detected (previously missed ones)
