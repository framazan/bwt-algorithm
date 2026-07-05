import numpy as np
import ctypes
from typing import List, Tuple, Dict, Iterator, Optional
from collections import Counter
import math
from .models import AlignmentResult, RepeatAlignmentSummary, RefinedRepeat, TandemRepeat
from .accelerators import align_unit_to_window

# Load C acceleration libraries
try:
    from .c_extensions.build import load_align as _load_align
    _c_align_lib = _load_align()
except Exception:
    _c_align_lib = None

try:
    from .c_extensions.build import load_tier2 as _load_tier2
    _c_tier2_lib = _load_tier2()
except Exception:
    _c_tier2_lib = None

class MotifUtils:
    """Utilities for canonical motif handling.

    Uses integer encoding for fast k-mer comparison (inspired by STRling,
    Genome Biology 2022, DOI:10.1186/s13059-022-02826-4).
    Encodes bases as 2-bit integers for O(1) rotation comparison
    instead of O(k) string comparison.
    """

    _BASE_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    _INT_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    _COMP_INT = {0: 3, 1: 2, 2: 1, 3: 0}  # A<->T, C<->G

    @staticmethod
    def _motif_to_int(motif: str) -> int:
        """Encode motif as integer (2 bits per base) for fast comparison."""
        val = 0
        for ch in motif:
            val = (val << 2) | MotifUtils._BASE_TO_INT.get(ch, 0)
        return val

    @staticmethod
    def _int_to_motif(val: int, length: int) -> str:
        """Decode integer back to motif string."""
        chars = []
        for _ in range(length):
            chars.append(MotifUtils._INT_TO_BASE[val & 3])
            val >>= 2
        return ''.join(reversed(chars))

    @staticmethod
    def _rotate_int(val: int, length: int) -> int:
        """Rotate 2-bit encoded motif left by 1 position."""
        top = (val >> (2 * (length - 1))) & 3
        mask = (1 << (2 * length)) - 1
        return ((val << 2) | top) & mask

    @staticmethod
    def _revcomp_int(val: int, length: int) -> int:
        """Reverse complement of 2-bit encoded motif."""
        result = 0
        for _ in range(length):
            base = val & 3
            result = (result << 2) | MotifUtils._COMP_INT[base]
            val >>= 2
        return result

    @staticmethod
    def _min_rotation_int(val: int, length: int) -> int:
        """Return the smallest (as integer) of all rotations of a 2-bit encoded motif."""
        best = val
        current = val
        for _ in range(length - 1):
            current = MotifUtils._rotate_int(current, length)
            if current < best:
                best = current
        return best

    @staticmethod
    def get_canonical_motif(motif: str) -> str:
        """Get lexicographically smallest rotation of motif.

        Uses integer encoding for motifs <= 32bp for fast comparison.
        """
        if not motif:
            return motif
        n = len(motif)

        # Use integer encoding for short motifs (up to 32bp fits in 64-bit int)
        if n <= 32 and all(c in 'ACGT' for c in motif):
            val = MotifUtils._motif_to_int(motif)
            best = MotifUtils._min_rotation_int(val, n)
            return MotifUtils._int_to_motif(best, n)

        # Fallback for long or non-standard motifs
        rotations = [motif[i:] + motif[:i] for i in range(len(motif))]
        return min(rotations)

    @staticmethod
    def reverse_complement(seq: str) -> str:
        """Get reverse complement of DNA sequence."""
        complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        return ''.join(complement_map.get(b, b) for b in reversed(seq))

    @staticmethod
    def get_canonical_motif_stranded(motif: str) -> Tuple[str, str]:
        """Get canonical motif considering both strands (rotation + revcomp).

        Uses integer encoding (inspired by STRling) for motifs <= 32bp.

        Returns:
            (canonical_motif, strand) where strand is '+' or '-'
        """
        if not motif:
            return motif, '+'

        n = len(motif)

        # Fast path: integer encoding for short ACGT-only motifs
        if n <= 32 and all(c in 'ACGT' for c in motif):
            val = MotifUtils._motif_to_int(motif)

            # Min rotation of the forward strand and of the reverse complement
            fwd_best = MotifUtils._min_rotation_int(val, n)
            rc_best = MotifUtils._min_rotation_int(MotifUtils._revcomp_int(val, n), n)

            if fwd_best <= rc_best:
                return MotifUtils._int_to_motif(fwd_best, n), '+'
            else:
                return MotifUtils._int_to_motif(rc_best, n), '-'

        # Fallback for long or non-standard motifs
        forward_rotations = [motif[i:] + motif[:i] for i in range(n)]
        forward_canonical = min(forward_rotations)

        rc = MotifUtils.reverse_complement(motif)
        rc_rotations = [rc[i:] + rc[:i] for i in range(n)]
        rc_canonical = min(rc_rotations)

        if forward_canonical <= rc_canonical:
            return forward_canonical, '+'
        else:
            return rc_canonical, '-'

    @staticmethod
    def is_primitive_motif(motif: str) -> bool:
        """Check if motif is not a repetition of a shorter motif."""
        n = len(motif)
        for i in range(1, n):
            if n % i == 0:
                period = motif[:i]
                if period * (n // i) == motif:
                    return False
        return True

    @staticmethod
    def calculate_entropy(seq: str) -> float:
        """Calculate Shannon entropy of sequence (bits per base)."""
        if not seq:
            return 0.0

        counts = Counter(seq)
        n = len(seq)
        entropy = 0.0

        for count in counts.values():
            if count > 0:
                p = count / n
                entropy -= p * np.log2(p)

        return entropy

    @staticmethod
    def is_transition(base1: str, base2: str) -> bool:
        """Check if a base change is a transition (A↔G or C↔T).

        Transitions are more common than transversions in biology.
        Purines: A, G (transitions within purines: A↔G)
        Pyrimidines: C, T (transitions within pyrimidines: C↔T)
        """
        if base1 == base2:
            return True  # No change

        transitions = {
            ('A', 'G'), ('G', 'A'),  # Purine transitions
            ('C', 'T'), ('T', 'C')   # Pyrimidine transitions
        }
        return (base1, base2) in transitions

    @staticmethod
    def hamming_distance(s1: str, s2: str) -> int:
        """Calculate Hamming distance between two strings of equal length."""
        if len(s1) != len(s2):
            return max(len(s1), len(s2))

        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    @staticmethod
    def hamming_distance_array(arr1: np.ndarray, arr2: np.ndarray) -> int:
        """Calculate Hamming distance between two uint8 arrays."""
        if arr1.size != arr2.size:
            return max(arr1.size, arr2.size)

        return int(np.count_nonzero(arr1 != arr2))

    @staticmethod
    def count_transversions_array(arr1: np.ndarray, arr2: np.ndarray) -> int:
        """Count transversions (non-transition mismatches) between two uint8 arrays.

        Returns number of transversion changes (A↔C, A↔T, G↔C, G↔T).
        """
        if arr1.size != arr2.size:
            return max(arr1.size, arr2.size)

        # ASCII codes: A=65, C=67, G=71, T=84
        transversions = 0
        for i in range(arr1.size):
            b1, b2 = arr1[i], arr2[i]
            if b1 != b2:
                # Convert to characters for transition check
                c1 = chr(b1) if 65 <= b1 <= 84 else 'N'
                c2 = chr(b2) if 65 <= b2 <= 84 else 'N'
                if not MotifUtils.is_transition(c1, c2):
                    transversions += 1

        return transversions

    @staticmethod
    def edit_distance(a: str, b: str) -> int:
        """Compute Levenshtein edit distance between two short strings."""
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la

        prev = list(range(lb + 1))
        curr = [0] * (lb + 1)

        for i in range(1, la + 1):
            curr[0] = i
            ai = a[i - 1]
            for j in range(1, lb + 1):
                cost = 0 if ai == b[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,      # deletion
                    curr[j - 1] + 1,  # insertion
                    prev[j - 1] + cost  # substitution
                )
            prev, curr = curr, prev

        return prev[lb]

    @staticmethod
    def _align_unit_to_window(motif: str, window: str, max_indel: int,
                              mismatch_tolerance: int) -> Optional[AlignmentResult]:
        """Align motif to a window allowing mismatches and small indels."""
        # Try accelerated version first
        try:
            # Convert to bytes for Cython
            motif_bytes = motif.encode('ascii')
            window_bytes = window.encode('ascii')
            res = align_unit_to_window(motif_bytes, window_bytes, max_indel, mismatch_tolerance)
            if res is not None:
                (consumed, unit_sequence, mismatch_count, insertion_len, 
                 deletion_len, operations, observed_bases, edit_distance) = res
                 
                return AlignmentResult(
                    consumed=consumed,
                    unit_sequence=unit_sequence,
                    mismatch_count=mismatch_count,
                    insertion_length=insertion_len,
                    deletion_length=deletion_len,
                    operations=operations,
                    observed_bases=observed_bases,
                    edit_distance=edit_distance
                )
        except Exception:
            pass # Fallback to Python implementation

        m = len(motif)
        n = len(window)

        if m == 0 or n == 0:
            return None

        max_indel = max(0, max_indel)
        mismatch_tolerance = max(0, mismatch_tolerance)

        lower = max(0, m - max_indel)
        upper = min(n, m + max_indel)
        if lower > upper:
            return None

        inf = m + n + 10
        dp = [[inf] * (n + 1) for _ in range(m + 1)]
        ptr = [[''] * (n + 1) for _ in range(m + 1)]

        dp[0][0] = 0
        for j in range(1, n + 1):
            dp[0][j] = j
            ptr[0][j] = 'I'
        for i in range(1, m + 1):
            dp[i][0] = i
            ptr[i][0] = 'D'

        band_extra = max_indel + 2

        for i in range(1, m + 1):
            j_min = max(1, i - band_extra)
            j_max = min(n, i + band_extra)
            for j in range(j_min, j_max + 1):
                sub_cost = dp[i - 1][j - 1] + (motif[i - 1] != window[j - 1])
                del_cost = dp[i - 1][j] + 1
                ins_cost = dp[i][j - 1] + 1

                best_cost = sub_cost
                best_ptr = 'M' if motif[i - 1] == window[j - 1] else 'S'

                if del_cost < best_cost:
                    best_cost = del_cost
                    best_ptr = 'D'
                if ins_cost < best_cost:
                    best_cost = ins_cost
                    best_ptr = 'I'

                dp[i][j] = best_cost
                ptr[i][j] = best_ptr

        best_j = -1
        best_cost = inf
        for j in range(lower, upper + 1):
            cost = dp[m][j]
            if cost < best_cost:
                best_cost = cost
                best_j = j

        if best_j <= 0 or best_cost >= inf:
            return None

        aligned_ref = []
        aligned_query = []
        i, j = m, best_j
        while i > 0 or j > 0:
            op = ptr[i][j]
            if op in ('M', 'S'):
                aligned_ref.append(motif[i - 1])
                aligned_query.append(window[j - 1])
                i -= 1
                j -= 1
            elif op == 'D':
                aligned_ref.append(motif[i - 1])
                aligned_query.append('-')
                i -= 1
            elif op == 'I':
                aligned_ref.append('-')
                aligned_query.append(window[j - 1])
                j -= 1
            else:  # Should only occur at origin
                break

        aligned_ref.reverse()
        aligned_query.reverse()

        operations: List[Tuple] = []
        observed_bases: List[Tuple[int, str]] = []
        mismatch_count = 0
        insertion_len = 0
        deletion_len = 0

        ref_pos = 0
        pending_ins: List[str] = []
        pending_ins_pos = 0
        pending_del_len = 0
        pending_del_pos = 0

        for r, q in zip(aligned_ref, aligned_query):
            if r == '-':
                if not pending_ins:
                    pending_ins_pos = ref_pos
                pending_ins.append(q)
                continue

            if pending_ins:
                ins_seq = ''.join(pending_ins)
                operations.append(('ins', pending_ins_pos, ins_seq))
                insertion_len += len(ins_seq)
                pending_ins = []
                pending_ins_pos = 0

            ref_pos += 1

            if q == '-':
                if pending_del_len == 0:
                    pending_del_pos = ref_pos
                pending_del_len += 1
                continue

            if pending_del_len:
                operations.append(('del', pending_del_pos, pending_del_len))
                deletion_len += pending_del_len
                pending_del_len = 0

            observed_bases.append((ref_pos - 1, q))
            if r != q:
                operations.append(('sub', ref_pos, r, q))
                mismatch_count += 1

        if pending_ins:
            ins_seq = ''.join(pending_ins)
            operations.append(('ins', pending_ins_pos, ins_seq))
            insertion_len += len(ins_seq)

        if pending_del_len:
            operations.append(('del', pending_del_pos, pending_del_len))
            deletion_len += pending_del_len

        if mismatch_count > mismatch_tolerance:
            return None
        if insertion_len > max_indel or deletion_len > max_indel:
            return None

        return AlignmentResult(
            consumed=best_j,
            unit_sequence=window[:best_j],
            mismatch_count=mismatch_count,
            insertion_length=insertion_len,
            deletion_length=deletion_len,
            operations=operations,
            observed_bases=observed_bases,
            edit_distance=best_cost
        )

    @staticmethod
    def _consensus_from_counts(counts: List[Counter], fallback: str) -> str:
        """Build consensus string from per-position base counts."""
        consensus = []
        for idx, counter in enumerate(counts):
            if counter:
                base, _ = counter.most_common(1)[0]
                consensus.append(base)
            else:
                consensus.append(fallback[idx] if idx < len(fallback) else 'N')
        return ''.join(consensus)

    # Class-level cache for text_arr pointer to avoid repeated from_buffer_copy
    _c_text_cache_id = None
    _c_text_ptr = None
    _c_text_len = 0

    @staticmethod
    def _get_text_ptr(sequence: str):
        """Get or cache ctypes pointer to sequence bytes."""
        seq_id = id(sequence)
        if MotifUtils._c_text_cache_id == seq_id:
            return MotifUtils._c_text_ptr, MotifUtils._c_text_len

        seq_len = len(sequence)
        # Use numpy for zero-copy pointer access
        text_arr = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
        MotifUtils._c_text_arr_ref = text_arr  # prevent GC
        MotifUtils._c_text_ptr = text_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        MotifUtils._c_text_len = seq_len
        MotifUtils._c_text_cache_id = seq_id
        return MotifUtils._c_text_ptr, seq_len

    @staticmethod
    def _align_repeat_region_c(sequence: str, start: int, end: int, motif_template: str,
                               mismatch_fraction: float, max_indel: int,
                               min_copies: int) -> Optional[RepeatAlignmentSummary]:
        """Fast C implementation of align_repeat_region loop."""
        if _c_align_lib is None:
            return None

        motif_len = len(motif_template)

        try:
            text_ptr, seq_len = MotifUtils._get_text_ptr(sequence)
            motif_bytes = motif_template.encode('ascii')
        except (UnicodeEncodeError, AttributeError):
            return None

        motif_arr = (ctypes.c_ubyte * motif_len).from_buffer_copy(motif_bytes)
        result = _c_align_lib.AlignRegionResult()
        consensus_buf = (ctypes.c_ubyte * motif_len)()

        ok = _c_align_lib.align_repeat_region_c(
            text_ptr, seq_len,
            start, end,
            motif_arr, motif_len,
            mismatch_fraction, max_indel, min_copies,
            ctypes.byref(result), consensus_buf
        )

        if not ok:
            return None

        consensus = bytes(consensus_buf).decode('ascii')

        return RepeatAlignmentSummary(
            consensus=consensus,
            motif_len=motif_len,
            copies=result.copies,
            consumed_length=result.consumed_length,
            mismatch_rate=result.mismatch_rate,
            max_errors_per_copy=result.max_errors_per_copy,
            variations=[],
            copy_sequences=[],
            total_insertions=result.total_insertions,
            total_deletions=result.total_deletions,
            error_counts=[],
            total_mismatches=result.total_mismatches
        )

    @staticmethod
    def align_repeat_region(sequence: str, start: int, end: int, motif_template: str,
                            mismatch_fraction: float = 0.1,
                            max_indel: Optional[int] = None,
                            min_copies: int = 3) -> Optional[RepeatAlignmentSummary]:
        """Align sequential copies of a motif within a sequence region."""
        if not motif_template:
            return None

        seq_len = len(sequence)
        if seq_len == 0:
            return None

        start = max(0, start)
        if end <= start:
            return None
        end = min(seq_len, end)

        motif_len = len(motif_template)
        if motif_len == 0:
            return None

        if max_indel is None:
            computed_indel = max(1, min(10, motif_len // 2 if motif_len >= 4 else 1))
        else:
            computed_indel = max(0, max_indel)

        # Try C-accelerated path first
        c_result = MotifUtils._align_repeat_region_c(
            sequence, start, end, motif_template,
            mismatch_fraction, computed_indel, min_copies
        )
        if c_result is not None:
            return c_result

        tolerance = max(1, int(math.floor(motif_len * mismatch_fraction)))
        # Python fallback reuses the indel bound already resolved for the C path.
        max_indel = computed_indel

        position_counts: List[Counter] = [Counter() for _ in range(motif_len)]
        copy_sequences: List[str] = []
        operations_by_copy: List[List[Tuple]] = []
        error_counts: List[int] = []

        total_insertions = 0
        total_deletions = 0
        total_mismatches = 0

        current_motif = motif_template
        pos = start
        safety_limit = min(seq_len, max(end, start + motif_len * min_copies) + max(motif_len * 3, max_indel * 4))

        while pos < safety_limit:
            window_end = min(seq_len, pos + motif_len + max_indel)
            window = sequence[pos:window_end]
            if len(window) < motif_len - max_indel:
                break

            result = MotifUtils._align_unit_to_window(current_motif, window, max_indel, tolerance)
            if result is None or result.consumed == 0:
                # Alignment failed - stop extending
                break

            copy_sequences.append(result.unit_sequence)
            operations_by_copy.append(result.operations)
            error_counts.append(result.error_count)
            total_mismatches += result.mismatch_count
            total_insertions += result.insertion_length
            total_deletions += result.deletion_length

            for motif_idx, base in result.observed_bases:
                if 0 <= motif_idx < motif_len:
                    position_counts[motif_idx][base] += 1

            pos += result.consumed
            current_motif = MotifUtils._consensus_from_counts(position_counts, current_motif)

        copies = len(copy_sequences)
        if copies < min_copies:
            return None

        consumed_len = pos - start
        if consumed_len <= 0:
            return None

        consensus = MotifUtils._consensus_from_counts(position_counts, current_motif)
        denom = copies * motif_len
        mismatch_rate = total_mismatches / denom if denom > 0 else 0.0
        max_errors_per_copy = max(error_counts) if error_counts else 0

        variations: List[str] = []
        for idx, ops in enumerate(operations_by_copy, 1):
            for op in ops:
                if not op:
                    continue
                kind = op[0]
                if kind == 'sub':
                    _, pos_idx, ref_base, alt_base = op
                    variations.append(f"{idx}:{pos_idx}:{ref_base}>{alt_base}")
                elif kind == 'ins':
                    _, pos_idx, inserted = op
                    if inserted:
                        variations.append(f"{idx}:{pos_idx}:ins({inserted})")
                elif kind == 'del':
                    _, pos_idx, length = op
                    if length > 0:
                        variations.append(f"{idx}:{pos_idx}:del({length})")

        return RepeatAlignmentSummary(
            consensus=consensus,
            motif_len=motif_len,
            copies=copies,
            consumed_length=consumed_len,
            mismatch_rate=mismatch_rate,
            max_errors_per_copy=max_errors_per_copy,
            variations=variations,
            copy_sequences=copy_sequences,
            total_insertions=total_insertions,
            total_deletions=total_deletions,
            error_counts=error_counts,
            total_mismatches=total_mismatches
        )

    @staticmethod
    def refine_repeat(sequence: str, start: int, end: int, motif_template: str,
                      mismatch_fraction: float, indel_fraction: float,
                      min_copies: int) -> Optional[RefinedRepeat]:
        """Run DP alignment and reduce motif to primitive representation."""
        if not motif_template:
            return None

        motif_len = len(motif_template)
        if motif_len == 0:
            return None

        if indel_fraction <= 0:
            max_indel = 0
        else:
            max_indel = max(1, int(math.ceil(motif_len * indel_fraction)))

        summary = MotifUtils.align_repeat_region(
            sequence,
            start,
            end,
            motif_template,
            mismatch_fraction=mismatch_fraction,
            max_indel=max_indel,
            min_copies=min_copies
        )
        if not summary:
            return None

        primitive_len = MotifUtils.smallest_period_str(summary.consensus)
        if primitive_len == len(summary.consensus):
            # Conservative fallback: allow tiny noise when collapsing composite motifs.
            approx_primitive_len = MotifUtils.smallest_period_str_approx(
                summary.consensus,
                max_error_rate=0.02,
            )
            if 0 < approx_primitive_len < primitive_len:
                primitive_len = approx_primitive_len
        # If consensus period equals its length, check actual repeat region
        # for a shorter period via autocorrelation (handles off-by-one motifs
        # like AAACCCTA that should be period 7 AAACCCT).
        if primitive_len == len(summary.consensus) and primitive_len <= 20:
            region = sequence[start:end]
            region_len = len(region)
            if region_len >= primitive_len * 2:
                best_sub_p = primitive_len
                best_score = 0.0
                for p in range(max(1, primitive_len - 2), primitive_len):
                    matches = sum(1 for i in range(region_len - p)
                                  if region[i] == region[i + p])
                    score = matches / (region_len - p)
                    if score > 0.85 and score > best_score:
                        best_score = score
                        best_sub_p = p
                if best_sub_p < primitive_len:
                    primitive_len = best_sub_p
        # For long motifs (satellite DNA), try higher mismatch tolerance
        # CEN180-like satellites can have 20-30% inter-copy divergence
        if primitive_len == len(summary.consensus) and primitive_len >= 200:
            sat_primitive = MotifUtils._detect_satellite_period(summary.consensus)
            if 0 < sat_primitive < primitive_len:
                primitive_len = sat_primitive
        if primitive_len == 0:
            return None

        primitive = summary.consensus[:primitive_len]
        consumed_len = summary.consumed_length
        refined_end = start + consumed_len
        copies = consumed_len / primitive_len if primitive_len > 0 else float(summary.copies)

        return RefinedRepeat(
            start=start,
            end=refined_end,
            consensus=summary.consensus,
            primitive_motif=primitive,
            motif_len=primitive_len,
            copies=copies,
            summary=summary
        )

    @staticmethod
    def is_insertion_variant(candidate: str, consensus: str) -> bool:
        """Return True if removing a single base from candidate yields consensus."""
        if len(candidate) != len(consensus) + 1:
            return False
        for i in range(len(candidate)):
            if candidate[:i] + candidate[i + 1:] == consensus:
                return True
        return False

    @staticmethod
    def is_deletion_variant(candidate: str, consensus: str) -> bool:
        """Return True if inserting a single base into candidate yields consensus."""
        if len(candidate) + 1 != len(consensus):
            return False
        for i in range(len(consensus)):
            if consensus[:i] + consensus[i + 1:] == candidate:
                return True
        return False

    @staticmethod
    def smallest_period_str(s: str) -> int:
        """Return length of the smallest period of string s."""
        if not s:
            return 0
        n = len(s)
        # Use C acceleration when available
        if _c_tier2_lib is not None and n <= 10000:
            try:
                s_bytes = s.encode('ascii')
                arr = (ctypes.c_ubyte * n).from_buffer_copy(s_bytes)
                return _c_tier2_lib.smallest_period_str(arr, n)
            except Exception:
                pass
        for p in range(1, n + 1):
            if n % p == 0 and s == s[:p] * (n // p):
                return p
        return n

    @staticmethod
    def _detect_satellite_period(s: str) -> int:
        """Detect the primitive period of highly divergent satellite DNA.

        Uses autocorrelation: for each candidate period p, count how many
        positions i have s[i] == s[i+p]. The period with the highest
        autocorrelation peak (above a threshold) is the primitive period.

        Designed for satellite DNA like CEN180 with 20-30% inter-copy divergence.
        Only called for motifs >= 200bp.
        """
        n = len(s)
        if n < 200:
            return n

        best_p = n
        best_score = 0.0
        min_identity = 0.60  # 60% identity threshold for satellite DNA

        # Search periods from small to large, looking for the smallest
        # period with high autocorrelation
        for p in range(50, min(n // 2 + 1, 500)):
            total = n - p
            if total <= 0:
                continue
            matches = sum(1 for i in range(total) if s[i] == s[i + p])
            identity = matches / total

            if identity >= min_identity and identity > best_score:
                best_score = identity
                best_p = p
                # Found a good period; check if there's an even smaller
                # sub-period that also works
                break

        if best_p == n:
            return n

        # Verify that this period divides the motif reasonably well
        # (at least 2 copies)
        if n / best_p < 2.0:
            return n

        # Check for sub-periods of best_p
        for sub_p in range(max(10, best_p // 20), best_p):
            if best_p % sub_p != 0:
                continue
            total = n - sub_p
            if total <= 0:
                continue
            matches = sum(1 for i in range(total) if s[i] == s[i + sub_p])
            identity = matches / total
            if identity >= min_identity:
                return sub_p

        return best_p

    @staticmethod
    def smallest_period_str_approx(s: str, max_error_rate: float = 0.02) -> int:
        """Return smallest approximate period length for a noisy string."""
        if not s:
            return 0

        n = len(s)
        if n == 1:
            return 1

        # Use C acceleration when available
        if _c_tier2_lib is not None and n <= 10000:
            try:
                s_bytes = s.encode('ascii')
                arr = (ctypes.c_ubyte * n).from_buffer_copy(s_bytes)
                result = _c_tier2_lib.smallest_period_str_approx(
                    arr, n, ctypes.c_double(max(0.0, min(0.05, max_error_rate)))
                )
                return result
            except Exception:
                pass

        max_error_rate = max(0.0, min(0.05, max_error_rate))

        for p in range(1, (n // 2) + 1):
            template = s[:p]
            mismatches = 0
            for i, ch in enumerate(s):
                if ch != template[i % p]:
                    mismatches += 1
            if (mismatches / n) <= max_error_rate:
                return p

        return n

    @staticmethod
    def normalize_variant(candidate: str, consensus: str) -> str:
        """Rotate variant to best align with consensus (minimal edit distance)."""
        if not candidate:
            return candidate

        best = candidate
        best_cost = MotifUtils.edit_distance(candidate, consensus)

        if len(candidate) == 1:
            return candidate

        for shift in range(1, len(candidate)):
            rotated = candidate[shift:] + candidate[:shift]
            cost = MotifUtils.edit_distance(rotated, consensus)
            if cost < best_cost or (cost == best_cost and rotated < best):
                best = rotated
                best_cost = cost

        return best

    @staticmethod
    def rotate_deletion_variant(candidate: str, consensus: str) -> str:
        """Rotate shorter variant so it no longer begins with the consensus prefix."""
        if not candidate:
            return candidate

        rotated = candidate
        for _ in range(len(candidate)):
            if rotated[0] != consensus[0]:
                return rotated
            rotated = rotated[1:] + rotated[:1]
        return rotated

    @staticmethod
    def build_consensus_motif(sequences: List[str]) -> Tuple[str, float]:
        """Build consensus motif from multiple aligned sequences using majority vote.

        Returns:
            (consensus, avg_mismatch_rate) - consensus sequence and average mismatch rate
        """
        if not sequences:
            return "", 0.0

        if len(sequences) == 1:
            return sequences[0], 0.0

        motif_len = len(sequences[0])
        consensus = []
        total_mismatches = 0

        for pos in range(motif_len):
            bases = [seq[pos] for seq in sequences if pos < len(seq)]
            if not bases:
                consensus.append('N')
                continue

            # Majority vote
            counts = Counter(bases)
            most_common = counts.most_common(1)[0][0]
            consensus.append(most_common)

            # Count mismatches at this position
            mismatches = len(bases) - counts[most_common]
            total_mismatches += mismatches

        total_bases = len(sequences) * motif_len
        avg_mismatch_rate = total_mismatches / total_bases if total_bases > 0 else 0.0

        return ''.join(consensus), avg_mismatch_rate

    @staticmethod
    def build_consensus_motif_array(text_arr: np.ndarray, start: int, motif_len: int,
                                   n_copies: int) -> Tuple[np.ndarray, float, int]:
        """Build consensus motif from array copies using majority vote.

        Returns:
            (consensus_array, mismatch_rate, max_mismatches_per_copy)
        """
        if n_copies == 0 or motif_len == 0:
            return np.array([], dtype=np.uint8), 0.0, 0

        consensus = np.zeros(motif_len, dtype=np.uint8)
        total_mismatches = 0
        max_mismatches_per_copy = 0

        # Collect all copies
        copies = []
        for i in range(n_copies):
            copy_start = start + i * motif_len
            copy_end = copy_start + motif_len
            if copy_end > text_arr.size:
                break
            copy_arr = text_arr[copy_start:copy_end]
            copies.append(copy_arr)

        if not copies:
            return np.array([], dtype=np.uint8), 0.0, 0

        # Build consensus by majority vote at each position
        for pos in range(motif_len):
            bases = [copy[pos] for copy in copies if pos < len(copy)]
            if not bases:
                consensus[pos] = ord('N')
                continue

            # Find most common base
            unique, counts = np.unique(bases, return_counts=True)
            most_common_idx = np.argmax(counts)
            consensus[pos] = unique[most_common_idx]

        # Calculate mismatch statistics
        for copy in copies:
            mismatches = MotifUtils.hamming_distance_array(copy, consensus)
            total_mismatches += mismatches
            max_mismatches_per_copy = max(max_mismatches_per_copy, mismatches)

        total_bases = len(copies) * motif_len
        mismatch_rate = total_mismatches / total_bases if total_bases > 0 else 0.0

        return consensus, mismatch_rate, max_mismatches_per_copy

    @staticmethod
    def summarize_variations_array(text_arr: np.ndarray, start: int, end: int, motif_len: int,
                                   consensus_arr: np.ndarray) -> List[str]:
        """Summarize per-copy variations relative to consensus, allowing small indels."""
        if text_arr.size == 0 or motif_len <= 0:
            return []

        sequence = text_arr.tobytes().decode('ascii', errors='replace')
        start = max(0, start)
        end = min(len(sequence), end if end > start else len(sequence))

        if end <= start:
            return []

        if consensus_arr.size > 0:
            motif_template = consensus_arr.tobytes().decode('ascii', errors='replace')
        else:
            motif_template = sequence[start:start + motif_len]

        summary = MotifUtils.align_repeat_region(
            sequence,
            start,
            end,
            motif_template,
            mismatch_fraction=0.3,  # Allow ~30% mismatches to capture variations
            min_copies=1
        )
        if not summary:
            return []
        return summary.variations

    @staticmethod
    def calculate_composition(sequence: str) -> Dict[str, float]:
        """Calculate nucleotide composition as percentages.

        Returns:
            Dictionary with A, C, G, T percentages
        """
        if not sequence:
            return {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}

        counts = Counter(sequence.upper())
        total = len(sequence)

        composition = {
            'A': (counts.get('A', 0) / total) * 100.0,
            'C': (counts.get('C', 0) / total) * 100.0,
            'G': (counts.get('G', 0) / total) * 100.0,
            'T': (counts.get('T', 0) / total) * 100.0,
        }

        return composition

    @staticmethod
    def calculate_trf_score(consensus: str, copies: int, mismatch_rate: float, length: int,
                             indel_rate: float = 0.0) -> int:
        """Calculate TRF-style alignment score.

        TRF uses match/mismatch/indel scoring. We approximate:
        - Match: +2 points
        - Mismatch: -7 points
        - Indel: -7 points (we use 0 indels for Hamming distance)

        Returns:
            Alignment score (integer)
        """
        total_bases = length
        matches = total_bases * max(0.0, 1.0 - mismatch_rate - indel_rate)
        mismatches = total_bases * mismatch_rate
        indels = total_bases * indel_rate

        # TRF scoring parameters (approximately)
        match_score = 2
        mismatch_penalty = 7

        score = int((matches * match_score) - ((mismatches + indels) * mismatch_penalty))
        return max(0, score)  # Don't allow negative scores

    @staticmethod
    def calculate_trf_statistics(text_arr: np.ndarray, start: int, end: int,
                                 consensus_motif: str, copies: int,
                                 mismatch_rate: float, indel_rate: float = 0.0) -> Tuple[float, float, int, Dict[str, float], float, str]:
        """Calculate TRF-compatible statistics for a repeat.

        Returns:
            (percent_matches, percent_indels, score, composition, entropy, actual_sequence)
        """
        # Extract actual sequence
        if end <= text_arr.size:
            actual_sequence = text_arr[start:end].tobytes().decode('ascii', errors='replace')
        else:
            actual_sequence = consensus_motif * int(copies)

        # Percent matches (inverse of mismatch rate)
        percent_matches = max(0.0, (1.0 - mismatch_rate - indel_rate) * 100.0)

        percent_indels = max(0.0, indel_rate * 100.0)

        # Composition
        composition = MotifUtils.calculate_composition(consensus_motif)

        # Entropy (already calculated, but recalculate for consistency)
        entropy = MotifUtils.calculate_entropy(consensus_motif)

        # Score
        length = end - start
        score = MotifUtils.calculate_trf_score(consensus_motif, copies, mismatch_rate, length, indel_rate)

        return percent_matches, percent_indels, score, composition, entropy, actual_sequence

    @staticmethod
    def compute_trf_stats_from_summary(text_arr: np.ndarray, refined: RefinedRepeat) -> Tuple[float, float, int, Dict[str, float], float, str]:
        """Reuse the full sequence to compute TRF stats based on alignment summary."""
        summary = refined.summary
        total_len = refined.end - refined.start
        if total_len <= 0:
            total_len = summary.consumed_length

        mismatch_rate = summary.total_mismatches / (summary.copies * summary.motif_len) if summary.copies > 0 else 0.0
        indel_rate = (summary.total_insertions + summary.total_deletions) / max(1, summary.copies * summary.motif_len)

        return MotifUtils.calculate_trf_statistics(
            text_arr,
            refined.start,
            refined.end,
            refined.primitive_motif,
            int(round(refined.copies)),
            mismatch_rate,
            indel_rate
        )

    @staticmethod
    def refined_to_repeat(chromosome: str, refined: RefinedRepeat, tier: int,
                          text_arr: np.ndarray, strand: str = '+') -> TandemRepeat:
        percent_matches, percent_indels, score, composition, entropy_val, actual_sequence = (
            MotifUtils.compute_trf_stats_from_summary(text_arr, refined)
        )

        return TandemRepeat(
            chrom=chromosome,
            start=refined.start,
            end=refined.end,
            motif=refined.primitive_motif,
            copies=float(refined.copies),
            length=refined.end - refined.start,
            tier=tier,
            confidence=max(0.5, 1.0 - refined.summary.mismatch_rate),
            consensus_motif=refined.summary.consensus,
            mismatch_rate=refined.summary.mismatch_rate,
            max_mismatches_per_copy=refined.summary.max_errors_per_copy,
            n_copies_evaluated=int(refined.summary.copies),
            strand=strand,
            percent_matches=percent_matches,
            percent_indels=percent_indels,
            score=score,
            composition=composition,
            entropy=entropy_val,
            actual_sequence=actual_sequence,
            variations=refined.summary.variations if refined.summary.variations else None
        )

    @staticmethod
    def enumerate_motifs(k: int, alphabet: str = "ACGT") -> Iterator[str]:
        """Generate all canonical primitive motifs of length k."""
        def generate_strings(length, current=""):
            if length == 0:
                canonical = MotifUtils.get_canonical_motif(current)
                if canonical == current and MotifUtils.is_primitive_motif(current):
                    yield current
                return

            for char in alphabet:
                yield from generate_strings(length - 1, current + char)

        yield from generate_strings(k)
