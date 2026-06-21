# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Cython accelerators for tandem repeat extension loops."""

import os
import numpy as np
cimport numpy as cnp
from libc.math cimport ceil

ctypedef cnp.uint8_t UINT8

from libc.stdint cimport uint64_t

cdef extern from *:
    int __builtin_popcountll(unsigned long long) nogil

# ── Mismatch-extender behaviour knobs (read once at module load) ──────────────
# TIER1_EXT_ROLLING (default 0): when 0 the extender uses the ORIGINAL fixed-seed
# consensus + cumulative-mismatch break (baseline preserved exactly). When 1 it
# uses a per-position rolling-majority consensus with a windowed "K consecutive
# bad copies" break, extending in both directions from the seed.
# TIER1_EXT_BAD_RUN (default 2, only used when ROLLING=1): number of consecutive
# bad copies (per-copy mismatch fraction over the allowed rate) tolerated before
# the extension in that direction stops.
cdef bint _EXT_ROLLING = bool(int(os.environ.get("TIER1_EXT_ROLLING", "0")))
cdef int _EXT_BAD_RUN = int(os.environ.get("TIER1_EXT_BAD_RUN", "2"))
if _EXT_BAD_RUN < 1:
    _EXT_BAD_RUN = 1

cdef unsigned char[256] _base_map
cdef bint _base_map_initialized = False

cdef void _init_base_map() noexcept nogil:
    global _base_map_initialized
    if _base_map_initialized:
        return
    cdef int i
    for i in range(256):
        _base_map[i] = 0 # Default to A (00)
    
    # C: 01
    _base_map[67] = 1 
    _base_map[99] = 1
    # G: 10
    _base_map[71] = 2
    _base_map[103] = 2
    # T: 11
    _base_map[84] = 3
    _base_map[116] = 3
    # A is 0, so 65/97 are already 0
    
    _base_map_initialized = True

cpdef cnp.ndarray[UINT8, ndim=1] pack_sequence(const unsigned char[:] text_arr):
    if not _base_map_initialized:
        _init_base_map()
        
    cdef int n = text_arr.shape[0]
    # Pad with 8 bytes (64 bits) to allow safe over-reading
    cdef int packed_len = (n + 3) // 4 + 8
    cdef cnp.ndarray[UINT8, ndim=1] packed = np.zeros(packed_len, dtype=np.uint8)
    cdef unsigned char[:] packed_view = packed
    cdef int i
    cdef unsigned char val
    cdef int byte_idx, bit_shift
    
    for i in range(n):
        val = _base_map[text_arr[i]]
        byte_idx = i >> 2
        bit_shift = (i & 3) << 1
        # Little endian packing in byte: Base 0 at bits 0-1
        packed_view[byte_idx] |= (val << bit_shift)
        
    return packed

cdef inline int _hamming_distance_2bit(const unsigned char[:] packed, int start1, int start2, int length) nogil:
    cdef int mismatches = 0
    cdef int i
    cdef int chunks = length >> 5  # 32 bases per 64-bit chunk
    cdef int rem = length & 31
    
    cdef uint64_t* ptr = <uint64_t*> &packed[0]
    cdef uint64_t diff, z
    
    cdef int bit_off1 = start1 << 1
    cdef int bit_off2 = start2 << 1
    
    cdef int byte_idx1 = bit_off1 >> 3
    cdef int shift1 = bit_off1 & 7
    
    cdef int byte_idx2 = bit_off2 >> 3
    cdef int shift2 = bit_off2 & 7
    
    # Pointers to the start of the words
    cdef uint64_t* p1 = <uint64_t*> (<unsigned char*>ptr + byte_idx1)
    cdef uint64_t* p2 = <uint64_t*> (<unsigned char*>ptr + byte_idx2)
    
    cdef uint64_t v1, v1_next
    cdef uint64_t v2, v2_next
    
    for i in range(chunks):
        # Load 64 bits (32 bases) for seq1
        v1 = p1[0]
        if shift1:
            v1_next = (<uint64_t*>((<unsigned char*>p1) + 8))[0]
            v1 = (v1 >> shift1) | (v1_next << (64 - shift1))
            
        # Load 64 bits (32 bases) for seq2
        v2 = p2[0]
        if shift2:
            v2_next = (<uint64_t*>((<unsigned char*>p2) + 8))[0]
            v2 = (v2 >> shift2) | (v2_next << (64 - shift2))
            
        diff = v1 ^ v2
        # Count mismatches
        # Combine pairs: (d | d>>1) & 0x55...
        z = (diff | (diff >> 1)) & <uint64_t>0x5555555555555555
        mismatches += __builtin_popcountll(z)
        
        # Advance pointers by 8 bytes
        p1 = <uint64_t*> (<unsigned char*>p1 + 8)
        p2 = <uint64_t*> (<unsigned char*>p2 + 8)
        
    # Handle remaining bases
    if rem > 0:
        # Load remaining bits
        v1 = p1[0]
        if shift1:
            v1_next = (<uint64_t*>((<unsigned char*>p1) + 8))[0]
            v1 = (v1 >> shift1) | (v1_next << (64 - shift1))
            
        v2 = p2[0]
        if shift2:
            v2_next = (<uint64_t*>((<unsigned char*>p2) + 8))[0]
            v2 = (v2 >> shift2) | (v2_next << (64 - shift2))
            
        diff = v1 ^ v2
        z = (diff | (diff >> 1)) & <uint64_t>0x5555555555555555
        
        # Mask out bits beyond rem
        z &= ((1ULL << (2 * rem)) - 1)
        mismatches += __builtin_popcountll(z)
        
    return mismatches

cdef inline int _max_mismatch_threshold(int period, int copies, double allowed_rate):
    """Reproduce Python mismatch threshold logic in Cython.

    Note: the original implementation special-cased ``period == 1`` to return 0
    (zero tolerance for homopolymers). That special case has been removed so
    homopolymers get the normal allowed rate like every other period.
    """
    if period <= 0 or copies <= 0:
        return 0

    if allowed_rate < 0.01:
        allowed_rate = 0.01
    elif allowed_rate > 0.5:
        allowed_rate = 0.5

    cdef double total_len = period * copies
    cdef int threshold = <int>ceil(allowed_rate * total_len)
    if threshold < 1:
        threshold = 1
    return threshold

cdef inline int _total_mismatches(const unsigned char[:] text_arr, int start_pos, int end_pos,
                                  const unsigned char[:] consensus, int period, int n) nogil:
    cdef int copies = (end_pos - start_pos) // period
    cdef int total = 0
    cdef int copy_start, copy_end, idx, pos

    for idx in range(copies):
        copy_start = start_pos + idx * period
        copy_end = copy_start + period
        if copy_end > n:
            break
        for pos in range(period):
            if text_arr[copy_start + pos] != consensus[pos]:
                total += 1
    return total

cpdef int hamming_distance(const unsigned char[:] arr1, const unsigned char[:] arr2):
    """Calculate Hamming distance between two arrays."""
    cdef int n1 = arr1.shape[0]
    cdef int n2 = arr2.shape[0]
    if n1 != n2:
        raise ValueError("Arrays must have same length")
    return _hamming_distance(arr1, arr2, n1)

cdef inline int _hamming_distance(const unsigned char[:] arr1, const unsigned char[:] arr2, int length) nogil:
    cdef int i = 0
    cdef int mismatches = 0
    cdef uint64_t* p1
    cdef uint64_t* p2
    cdef uint64_t v, z
    cdef uint64_t LOW_MASK = 0x0101010101010101
    cdef uint64_t HIGH_MASK = 0x8080808080808080
    # Process 64-bit chunks
    cdef int chunks = length >> 3
    if chunks > 0:
        p1 = <uint64_t*> &arr1[0]
        p2 = <uint64_t*> &arr2[0]
        for i in range(chunks):
            v = p1[i] ^ p2[i]
            if v != 0:
                # SWAR: count zero bytes in v
                z = ((v - LOW_MASK) & ~v & HIGH_MASK)
                mismatches += (8 - __builtin_popcountll(z))
        i = chunks << 3
    else:
        i = 0
    # Handle remaining bytes
    for i in range(i, length):
        if arr1[i] != arr2[i]:
            mismatches += 1
    return mismatches

cdef tuple _extend_rolling(const unsigned char[:] text_arr,
                           int start_pos, int period, int n,
                           double allowed_mismatch_rate,
                           int bad_run_limit):
    """Rolling-consensus, windowed-break bidirectional extension.

    Compares each new copy to a per-position MAJORITY consensus that is updated
    as copies are accepted (instead of a fixed seed copy compared on cumulative
    mismatch). Extension in each direction stops only after ``bad_run_limit``
    *consecutive* bad copies, where a copy is "bad" if its per-copy mismatch
    fraction vs the current consensus exceeds ``allowed_mismatch_rate``. Trailing
    bad copies are trimmed so the reported core ends on a good copy.

    Returns (array_start, array_end, copies, full_start, full_end).
    """
    # Per-position vote table over raw bytes; period is small so 256 cols is fine.
    cdef cnp.ndarray[cnp.int32_t, ndim=2] votes_arr = np.zeros(
        (period, 256), dtype=np.int32
    )
    cdef int[:, :] votes = votes_arr
    # Current majority byte per position (the rolling consensus).
    cdef cnp.ndarray[UINT8, ndim=1] consensus_arr = np.empty(period, dtype=np.uint8)
    cdef unsigned char[:] consensus = consensus_arr

    cdef int pos, b, best_b, best_cnt
    cdef unsigned char ch

    # Seed the consensus from the first copy.
    for pos in range(period):
        ch = text_arr[start_pos + pos]
        votes[pos, ch] += 1
        consensus[pos] = ch

    cdef int start = start_pos
    cdef int end = start_pos + period
    cdef int copies = 1

    # Per-copy mismatch budget: at least 1 so a single SNP per copy never trips.
    cdef double rate = allowed_mismatch_rate
    if rate < 0.0:
        rate = 0.0
    elif rate > 0.5:
        rate = 0.5
    cdef int per_copy_max = <int>(rate * period)
    if per_copy_max < 1:
        per_copy_max = 1

    cdef int copy_start, mm
    cdef int bad_run, last_good_end, last_good_start

    # ── Extend right ─────────────────────────────────────────────────────────
    bad_run = 0
    last_good_end = end
    while end + period <= n:
        copy_start = end
        # Count mismatches of the candidate copy vs the current consensus.
        mm = 0
        for pos in range(period):
            if text_arr[copy_start + pos] != consensus[pos]:
                mm += 1
        if mm > per_copy_max:
            bad_run += 1
            if bad_run >= bad_run_limit:
                break
        else:
            bad_run = 0
            # Accept: fold this copy into the consensus votes and refresh majority.
            for pos in range(period):
                ch = text_arr[copy_start + pos]
                votes[pos, ch] += 1
                if votes[pos, ch] > votes[pos, consensus[pos]]:
                    consensus[pos] = ch
            last_good_end = copy_start + period
        copies += 1
        end += period
    end = last_good_end

    # ── Extend left ──────────────────────────────────────────────────────────
    bad_run = 0
    last_good_start = start
    while start - period >= 0:
        copy_start = start - period
        mm = 0
        for pos in range(period):
            if text_arr[copy_start + pos] != consensus[pos]:
                mm += 1
        if mm > per_copy_max:
            bad_run += 1
            if bad_run >= bad_run_limit:
                break
        else:
            bad_run = 0
            for pos in range(period):
                ch = text_arr[copy_start + pos]
                votes[pos, ch] += 1
                if votes[pos, ch] > votes[pos, consensus[pos]]:
                    consensus[pos] = ch
            last_good_start = copy_start
        start -= period
    start = last_good_start

    # Recompute the accepted copy count from the trimmed [start, end) span.
    copies = (end - start) // period

    cdef int full_start = start
    cdef int full_end = end
    cdef int array_start, array_end
    cdef int partial_left, partial_right

    # Partial right extension (exact matching against the final consensus).
    partial_right = 0
    while partial_right < period and full_end + partial_right < n:
        if text_arr[full_end + partial_right] != consensus[partial_right]:
            break
        partial_right += 1
    array_end = full_end + partial_right

    # Partial left extension (exact matching against the final consensus).
    partial_left = 0
    while partial_left < period and full_start - partial_left - 1 >= 0:
        if text_arr[full_start - partial_left - 1] != consensus[period - 1 - partial_left]:
            break
        partial_left += 1
    array_start = full_start - partial_left

    return array_start, array_end, copies, full_start, full_end


cpdef tuple extend_with_mismatches(const unsigned char[:] s_arr,
                                   int start_pos, int period, int n,
                                   double allowed_mismatch_rate):
    """Accelerated version of Tier2/Tier1 bidirectional extension.

    Returns (array_start, array_end, copies, full_start, full_end) or None on failure.

    Behaviour is selected by the module-level ``TIER1_EXT_ROLLING`` env flag
    (read once at import). Default (0) keeps the original fixed-seed-consensus +
    cumulative-mismatch break; (1) uses the rolling-consensus windowed-break
    extender (``_extend_rolling``) which extends diverged arrays to their true
    boundaries.
    """
    if period <= 0:
        return None

    cdef Py_ssize_t arr_len = s_arr.shape[0]
    if n <= 0 or n > arr_len:
        n = arr_len

    if start_pos < 0 or start_pos + period > n:
        return None

    if _EXT_ROLLING:
        return _extend_rolling(s_arr, start_pos, period, n,
                               allowed_mismatch_rate, _EXT_BAD_RUN)

    cdef cnp.ndarray[UINT8, ndim=1] consensus_arr = np.array(
        s_arr[start_pos:start_pos + period], copy=True
    )
    cdef const unsigned char[:] consensus = consensus_arr
    cdef const unsigned char[:] text_arr = s_arr

    cdef int start = start_pos
    cdef int end = start_pos + period
    cdef int copies = 1
    cdef int temp_copies, temp_end, temp_start
    cdef int new_mm, max_mm
    cdef int full_start, full_end
    cdef int array_start, array_end
    cdef int partial_left, partial_right

    # Extend right
    while end + period <= n:
        temp_copies = copies + 1
        temp_end = end + period
        new_mm = _hamming_distance(text_arr[end:end + period], consensus, period)
        max_mm = _max_mismatch_threshold(period, temp_copies, allowed_mismatch_rate)

        if new_mm > 0:
            if _total_mismatches(text_arr, start, temp_end, consensus, period, n) > max_mm:
                break

        copies = temp_copies
        end = temp_end

    # Extend left
    while start - period >= 0:
        temp_copies = copies + 1
        temp_start = start - period
        new_mm = _hamming_distance(text_arr[temp_start:temp_start + period], consensus, period)
        max_mm = _max_mismatch_threshold(period, temp_copies, allowed_mismatch_rate)

        if new_mm > 0:
            if _total_mismatches(text_arr, temp_start, end, consensus, period, n) > max_mm:
                break

        copies = temp_copies
        start = temp_start

    full_start = start
    full_end = end

    # Partial right extension (exact matching)
    partial_right = 0
    while partial_right < period and full_end + partial_right < n:
        if text_arr[full_end + partial_right] != consensus[partial_right]:
            break
        partial_right += 1
    array_end = full_end + partial_right

    # Partial left extension (exact matching)
    partial_left = 0
    while partial_left < period and full_start - partial_left - 1 >= 0:
        if text_arr[full_start - partial_left - 1] != consensus[period - 1 - partial_left]:
            break
        partial_left += 1
    array_start = full_start - partial_left

    return array_start, array_end, copies, full_start, full_end

cpdef list scan_unit_repeats(const unsigned char[:] text_arr, int n, int unit_len, int min_copies, int max_mismatch, const unsigned char[:] packed_arr=None):
    """Scan for repeats of a specific unit length."""
    cdef int i = 0
    cdef list results = []
    cdef int count, start_pos, end_pos
    cdef int a_start, a_end, b_start, b_end
    cdef int allowed_errors
    cdef int dist
    cdef bint found_indel
    cdef bint use_packed = (packed_arr is not None)
    
    # Calculate dynamic error threshold (15% of unit length or max_mismatch, whichever is higher)
    allowed_errors = max_mismatch
    cdef int dynamic_errors = <int>(unit_len * 0.15)
    if dynamic_errors > allowed_errors:
        allowed_errors = dynamic_errors

    while i + unit_len * min_copies <= n:
        count = 1
        start_pos = i
        
        # Extend right while adjacency holds
        while True:
            a_start = i + (count - 1) * unit_len
            a_end = i + count * unit_len
            b_start = i + count * unit_len
            b_end = b_start + unit_len
            
            if b_end > n:
                break
                
            # 1. Check direct Hamming distance
            if use_packed:
                dist = _hamming_distance_2bit(packed_arr, a_start, b_start, unit_len)
            else:
                dist = _hamming_distance(text_arr[a_start:a_end], text_arr[b_start:b_end], unit_len)
                
            if dist <= allowed_errors:
                count += 1
                continue
                
            # 2. Check for 1bp Indels
            found_indel = False
            
            # Check shift -1
            if b_start > 0:
                if use_packed:
                    dist = _hamming_distance_2bit(packed_arr, a_start, b_start-1, unit_len)
                else:
                    dist = _hamming_distance(text_arr[a_start:a_end], text_arr[b_start-1:b_end-1], unit_len)
                    
                if dist <= allowed_errors:
                    count += 1
                    found_indel = True
            
            if not found_indel and b_end + 1 <= n:
                # Check shift +1
                if use_packed:
                    dist = _hamming_distance_2bit(packed_arr, a_start, b_start+1, unit_len)
                else:
                    dist = _hamming_distance(text_arr[a_start:a_end], text_arr[b_start+1:b_end+1], unit_len)
                    
                if dist <= allowed_errors:
                    count += 1
                    found_indel = True
            
            if not found_indel:
                break
        
        if count >= min_copies:
            end_pos = i + count * unit_len
            results.append((i, end_pos))
            i = end_pos
        else:
            i += 1
            
    return results

cpdef list scan_simple_repeats(
    const unsigned char[:] text_arr,
    const unsigned char[:] tier1_mask,
    int n,
    int min_p,
    int max_p,
    int period_step,
    int position_step,
    double allowed_mismatch_rate
):
    """Scan for simple repeats using accelerated logic."""
    cdef int p, i
    cdef int check_len
    cdef int start_pos, end_pos, copies, full_start, full_end
    cdef int array_start, array_end
    cdef list results = []
    cdef int j
    cdef bint match
    
    # Iterate periods
    for p in range(min_p, max_p + 1, period_step):
        i = 0
        check_len = 4
        if p < 4:
            check_len = p
            
        while i < n - p:
            # Skip if masked
            if tier1_mask[i]:
                i += position_step
                continue
                
            # Quick check for periodicity (array_equal replacement)
            if i + p + check_len <= n:
                match = True
                for j in range(check_len):
                    if text_arr[i + j] != text_arr[i + p + j]:
                        match = False
                        break
                
                if match:
                    # Found match, extend
                    # We call the C implementation of extend_with_mismatches directly
                    # Note: extend_with_mismatches returns tuple or None
                    # But we can't call cpdef from cdef easily if we want C speed?
                    # Actually we can call it.
                    # Or better, inline the logic or call a cdef version.
                    # But extend_with_mismatches is cpdef, so it's callable.
                    # However, it returns a Python tuple.
                    # To avoid tuple overhead, we should probably refactor extend_with_mismatches to have a cdef core.
                    # But for now, let's just call it. The extension happens rarely compared to the scan.
                    
                    res = extend_with_mismatches(text_arr, i, p, n, allowed_mismatch_rate)
                    if res is not None:
                        array_start, array_end, copies, full_start, full_end = res
                        
                        # Dynamic copy threshold
                        if (p >= 20 and copies >= 2) or copies >= 3:
                            results.append((full_start, full_end, p))
                            
                            # Skip past this repeat
                            i = full_end
                            continue
            
            i += position_step
            
    return results

cpdef list find_periodic_patterns(long[:] positions, int min_period, int max_period, int min_copies, double tolerance_ratio=0.01):
    """Find periodic patterns in a sorted list of positions."""
    cdef int n = positions.shape[0]
    cdef list results = []
    cdef int i, j, k
    cdef long p1, p2, diff, next_val, last_val, target
    cdef int count
    cdef int tolerance
    
    if n < min_copies:
        return results

    # Limit N to avoid O(N^2) explosion on highly repetitive k-mers
    if n > 500:
        n = 500

    for i in range(n):
        p1 = positions[i]
        for j in range(i + 1, n):
            p2 = positions[j]
            diff = p2 - p1
            
            if diff < min_period:
                continue
            if diff > max_period:
                break 
            
            count = 2
            last_val = p2
            
            for k in range(j + 1, n):
                next_val = positions[k]
                target = last_val + diff
                tolerance = <int>(diff * tolerance_ratio) + 1
                
                if next_val < target - tolerance:
                    continue
                elif next_val > target + tolerance:
                    break
                else:
                    count += 1
                    last_val = next_val
            
            if count >= min_copies:
                results.append((p1, last_val, diff))
                
    return results

cpdef list find_periodic_runs(long[:] positions, int min_period, int max_period, int min_copies, double tolerance_ratio=0.01):
    """Detect periodic runs using adjacent gaps only (O(k)).

    Returns list of (start_pos, end_pos, period).
    A run requires at least `min_copies` positions, i.e., at least `min_copies-1` consecutive gaps
    within tolerance and within [min_period, max_period].
    """
    cdef int n = positions.shape[0]
    cdef list results = []
    if n < min_copies:
        return results

    cdef long prev_pos = positions[0]
    cdef double last_diff = -1.0
    cdef int run_start_idx = 0
    cdef int gap_count = 0  # number of consecutive gaps consistent with last_diff
    cdef long cur_pos
    cdef double diff
    cdef double tol
    cdef int i
    cdef long run_start_pos
    cdef long run_end_pos
    cdef int period_int

    for i in range(1, n):
        cur_pos = positions[i]
        diff = cur_pos - prev_pos
        prev_pos = cur_pos

        if diff < min_period or diff > max_period:
            # finish any existing run
            if gap_count + 1 >= min_copies:
                run_start_pos = positions[run_start_idx]
                run_end_pos = positions[i - 1]
                period_int = <int>(last_diff + 0.5)
                results.append((run_start_pos, run_end_pos, period_int))
            # reset
            run_start_idx = i
            gap_count = 0
            last_diff = -1.0
            continue

        if last_diff < 0:
            last_diff = diff
            gap_count = 1
            run_start_idx = i - 1
        else:
            tol = last_diff * tolerance_ratio
            if tol < 1.0:
                tol = 1.0
            if diff >= last_diff - tol and diff <= last_diff + tol:
                gap_count += 1
            else:
                # end current run
                if gap_count + 1 >= min_copies:
                    run_start_pos = positions[run_start_idx]
                    run_end_pos = positions[i - 1]
                    period_int = <int>(last_diff + 0.5)
                    results.append((run_start_pos, run_end_pos, period_int))
                # start new run with this gap
                last_diff = diff
                gap_count = 1
                run_start_idx = i - 1

    # flush at end
    if gap_count + 1 >= min_copies:
        run_start_pos = positions[run_start_idx]
        run_end_pos = positions[n - 1]
        period_int = <int>(last_diff + 0.5)
        results.append((run_start_pos, run_end_pos, period_int))

    return results

from libc.stdlib cimport malloc, free

cpdef tuple align_unit_to_window(
    const unsigned char[:] motif, 
    const unsigned char[:] window, 
    int max_indel, 
    int mismatch_tolerance
):
    """Cython implementation of Needleman-Wunsch alignment for repeat units."""
    cdef int m = motif.shape[0]
    cdef int n = window.shape[0]
    
    if m == 0 or n == 0:
        return None

    if max_indel < 0: max_indel = 0
    if mismatch_tolerance < 0: mismatch_tolerance = 0

    cdef int lower = m - max_indel
    if lower < 0: lower = 0
    cdef int upper = m + max_indel
    if upper > n: upper = n
    
    if lower > upper:
        return None

    cdef int inf = m + n + 10
    cdef int rows = m + 1
    cdef int cols = n + 1
    
    # Allocate flattened arrays
    cdef int* dp = <int*> malloc(rows * cols * sizeof(int))
    cdef char* ptr = <char*> malloc(rows * cols * sizeof(char))
    
    if not dp or not ptr:
        if dp: free(dp)
        if ptr: free(ptr)
        raise MemoryError()

    cdef int i, j, idx
    cdef int sub_cost, del_cost, ins_cost, best_cost
    cdef char best_ptr
    cdef int j_min, j_max
    cdef int band_extra = max_indel + 2
    cdef int best_j
    cdef int min_final_cost
    cdef char op
    cdef int mismatch_count
    cdef int insertion_len
    cdef int deletion_len
    cdef int ref_pos
    cdef int pending_ins_pos
    cdef int pending_del_len
    cdef int pending_del_pos
    cdef int r_code, q_code
    cdef str r_char, q_char
    
    # 0=Stop, 1=Match(M), 2=Sub(S), 3=Del(D), 4=Ins(I)
    
    try:
        # Initialize
        for i in range(rows):
            for j in range(cols):
                dp[i * cols + j] = inf
                ptr[i * cols + j] = 0
        
        dp[0] = 0
        for j in range(1, cols):
            dp[j] = j
            ptr[j] = 4 # I
            
        for i in range(1, rows):
            dp[i * cols] = i
            ptr[i * cols] = 3 # D
            
        # Fill DP
        for i in range(1, rows):
            j_min = i - band_extra
            if j_min < 1: j_min = 1
            j_max = i + band_extra
            if j_max > n: j_max = n
            
            for j in range(j_min, j_max + 1):
                # Match/Sub
                sub_cost = dp[(i - 1) * cols + (j - 1)] + (1 if motif[i - 1] != window[j - 1] else 0)
                # Del (gap in window, consume motif)
                del_cost = dp[(i - 1) * cols + j] + 1
                # Ins (gap in motif, consume window)
                ins_cost = dp[i * cols + (j - 1)] + 1
                
                best_cost = sub_cost
                best_ptr = 1 if motif[i - 1] == window[j - 1] else 2 # M or S
                
                if del_cost < best_cost:
                    best_cost = del_cost
                    best_ptr = 3 # D
                if ins_cost < best_cost:
                    best_cost = ins_cost
                    best_ptr = 4 # I
                    
                dp[i * cols + j] = best_cost
                ptr[i * cols + j] = best_ptr
                
        # Find best end
        best_j = -1
        min_final_cost = inf
        
        for j in range(lower, upper + 1):
            cost = dp[m * cols + j]
            if cost < min_final_cost:
                min_final_cost = cost
                best_j = j
                
        if best_j <= 0 or min_final_cost >= inf:
            return None
            
        # Backtrack
        # We need to reconstruct the alignment
        # Since we can't easily append to lists in reverse in C, we'll use a temporary buffer or just Python lists
        
        aligned_ref_codes = []
        aligned_query_codes = []
        
        i = m
        j = best_j
        
        while i > 0 or j > 0:
            op = ptr[i * cols + j]
            if op == 1 or op == 2: # M or S
                aligned_ref_codes.append(motif[i - 1])
                aligned_query_codes.append(window[j - 1])
                i -= 1
                j -= 1
            elif op == 3: # D
                aligned_ref_codes.append(motif[i - 1])
                aligned_query_codes.append(45) # '-' is 45
                i -= 1
            elif op == 4: # I
                aligned_ref_codes.append(45) # '-'
                aligned_query_codes.append(window[j - 1])
                j -= 1
            else:
                break
                
        # Reverse
        aligned_ref_codes.reverse()
        aligned_query_codes.reverse()
        
        # Process alignment to generate operations
        operations = []
        observed_bases = []
        mismatch_count = 0
        insertion_len = 0
        deletion_len = 0
        
        ref_pos = 0
        pending_ins = []
        pending_ins_pos = 0
        pending_del_len = 0
        pending_del_pos = 0
        
        for k in range(len(aligned_ref_codes)):
            r_code = aligned_ref_codes[k]
            q_code = aligned_query_codes[k]
            
            r_char = chr(r_code)
            q_char = chr(q_code)
            
            if r_char == '-':
                if not pending_ins:
                    pending_ins_pos = ref_pos
                pending_ins.append(q_char)
                continue
                
            if pending_ins:
                ins_seq = "".join(pending_ins)
                operations.append(('ins', pending_ins_pos, ins_seq))
                insertion_len += len(ins_seq)
                pending_ins = []
                pending_ins_pos = 0
                
            ref_pos += 1
            
            if q_char == '-':
                if pending_del_len == 0:
                    pending_del_pos = ref_pos
                pending_del_len += 1
                continue
                
            if pending_del_len > 0:
                operations.append(('del', pending_del_pos, pending_del_len))
                deletion_len += pending_del_len
                pending_del_len = 0
                
            observed_bases.append((ref_pos - 1, q_char))
            
            if r_code != q_code:
                operations.append(('sub', ref_pos, r_char, q_char))
                mismatch_count += 1
                
        if pending_ins:
            ins_seq = "".join(pending_ins)
            operations.append(('ins', pending_ins_pos, ins_seq))
            insertion_len += len(ins_seq)
            
        if pending_del_len > 0:
            operations.append(('del', pending_del_pos, pending_del_len))
            deletion_len += pending_del_len
            
        if mismatch_count > mismatch_tolerance:
            return None
            
        if insertion_len > max_indel or deletion_len > max_indel:
            return None
            
        # Construct unit_sequence from window
        # window is bytes/memoryview, convert to string
        unit_sequence = bytes(window[:best_j]).decode('ascii', 'replace')
        
        return (
            best_j,
            unit_sequence,
            mismatch_count,
            insertion_len,
            deletion_len,
            operations,
            observed_bases,
            min_final_cost
        )
        
    finally:
        free(dp)
        free(ptr)

cpdef list lcp_tandem_candidates(
    const int[:] sa,
    const int[:] lcp,
    int n,
    int min_period,
    int max_period,
    int min_lcp_threshold=10
):
    """Scan LCP array for tandem repeat candidate positions.

    A tandem repeat with period p produces neighboring SA entries where:
    - LCP[i] >= min_lcp_threshold (suffixes share enough common prefix)
    - |SA[i] - SA[i-1]| is in [min_period, max_period] (SA difference = candidate period)

    For perfect repeats, LCP >= period. For imperfect repeats with divergence d,
    LCP ≈ 1/d (expected position of first mismatch). Using a low threshold
    (e.g., 10) catches repeats with up to ~10% divergence per copy.

    Returns list of (period, text_position) tuples.
    """
    cdef int sa_len = sa.shape[0]
    cdef list results = []
    cdef int i, L, pos_a, pos_b, diff, start
    cdef int lcp_len = lcp.shape[0]
    cdef int limit = sa_len
    if lcp_len < limit:
        limit = lcp_len

    for i in range(1, limit):
        L = lcp[i]
        if L < min_lcp_threshold:
            continue

        pos_a = sa[i - 1]
        pos_b = sa[i]

        # Skip sentinel positions
        if pos_a >= n or pos_b >= n:
            continue

        diff = pos_b - pos_a
        if diff < 0:
            diff = -diff

        if diff < min_period or diff > max_period:
            continue

        start = pos_a if pos_a < pos_b else pos_b
        results.append((diff, start))

    return results


cpdef list find_tandem_runs(
    long[:] positions,
    int period,
    int min_copies
):
    """Find maximal runs of positions with exact spacing = period.

    Given sorted positions where a motif occurs, find maximal runs
    of consecutive tandem copies (positions differing by exactly period).

    Returns list of (run_start, run_end) tuples where run_end is
    the end of the last copy (last_position + period).
    """
    cdef int n_pos = positions.shape[0]
    if n_pos < min_copies:
        return []

    cdef list results = []
    cdef long run_start = positions[0]
    cdef long expected_next = positions[0] + period
    cdef int count = 1
    cdef int i

    for i in range(1, n_pos):
        if positions[i] == expected_next:
            count += 1
            expected_next = positions[i] + period
        else:
            if count >= min_copies:
                # run_end = last confirmed position + period
                results.append((run_start, expected_next))
            run_start = positions[i]
            expected_next = positions[i] + period
            count = 1

    # Flush final run
    if count >= min_copies:
        results.append((run_start, expected_next))

    return results


cpdef tuple anchor_scan_boundaries(
    const unsigned char[:] text_arr,
    int seed_pos,
    int period,
    int n,
    double match_threshold,
    int max_backward_periods,
    int max_forward_periods,
):
    """Scan backward and forward from seed_pos to find repeat boundaries.

    Compares windows of size `period` against the motif at seed_pos.
    Extends as long as match ratio >= match_threshold.

    Returns (true_start, true_end).
    """
    cdef int true_start = seed_pos
    cdef int true_end = seed_pos + period
    cdef int pos, i, matches
    cdef int scan_start, scan_end

    # Motif reference is text_arr[seed_pos : seed_pos + period]
    # If seed_pos + period > n, clamp
    if seed_pos + period > n:
        return (true_start, true_end)

    # --- Scan backward ---
    scan_start = seed_pos - period * max_backward_periods
    if scan_start < 0:
        scan_start = 0

    pos = seed_pos - period
    while pos >= scan_start:
        if pos + period > n:
            break
        matches = 0
        for i in range(period):
            if text_arr[pos + i] == text_arr[seed_pos + i]:
                matches += 1
        if <double>matches / <double>period >= match_threshold:
            true_start = pos
            pos -= period
        else:
            break

    # --- Scan forward ---
    scan_end = seed_pos + period * max_forward_periods
    if scan_end > n:
        scan_end = n

    pos = seed_pos + period
    while pos + period <= scan_end:
        matches = 0
        for i in range(period):
            if text_arr[pos + i] == text_arr[seed_pos + i]:
                matches += 1
        if <double>matches / <double>period >= match_threshold:
            true_end = pos + period
            pos += period
        else:
            break

    return (true_start, true_end)
