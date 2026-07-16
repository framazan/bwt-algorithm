import numpy as np
import ctypes
from typing import List, Tuple, Dict, Union

SENTINEL = ord('$')  # appended before BWT construction; never a repeat base


def effective_length(text_arr: np.ndarray) -> int:
    """Length of `text_arr` with any trailing '$' sentinel excluded.

    Note Tier 1 and Tier 3 deliberately do *not* strip: they keep the sentinel in
    `n` and rely on `min(end, n)` clamping plus downstream '$' checks. Only call
    this where the caller means the sentinel-free length.
    """
    n = int(text_arr.size)
    if n > 0 and text_arr[n - 1] == SENTINEL:
        n -= 1
    return n


# Optional: JIT acceleration with numba when available
HAVE_NUMBA = False
try:
    import numba as _nb  # type: ignore
    HAVE_NUMBA = True
except Exception:
    _nb = None  # type: ignore

# C acceleration for rank queries
_c_bwt_lib = None
try:
    from .c_extensions.build import load_bwt as _load_bwt
    _c_bwt_lib = _load_bwt()
except Exception:
    pass

if HAVE_NUMBA:
    @_nb.njit(cache=True)
    def _count_equal_range(arr: np.ndarray, start: int, end: int, code: int) -> int:  # type: ignore
        c = 0
        for i in range(start, end):
            if arr[i] == code:
                c += 1
        return c

    @_nb.njit(cache=True)
    def _kasai_lcp_uint8(text_codes: np.ndarray, sa: np.ndarray) -> np.ndarray:  # type: ignore
        n = text_codes.size
        lcp = np.zeros(n, dtype=np.int32)
        rank = np.zeros(n, dtype=np.int32)
        for i in range(n):
            rank[sa[i]] = i
        h = 0
        for i in range(n):
            r = rank[i]
            if r > 0:
                j = sa[r - 1]
                while i + h < n and j + h < n and text_codes[i + h] == text_codes[j + h]:
                    h += 1
                lcp[r] = h
                if h > 0:
                    h -= 1
        return lcp
else:
    # Dummy decorator to avoid syntax errors if someone tries to use it
    def _dummy_jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def _count_equal_range(arr: np.ndarray, start: int, end: int, code: int) -> int:
        # Pure-python/numpy fallback
        return int(np.count_nonzero(arr[start:end] == code))

    def _kasai_lcp_uint8(text_codes: np.ndarray, sa: np.ndarray) -> np.ndarray:
        n = text_codes.size
        # Use C acceleration when available
        if _c_bwt_lib is not None:
            sa32 = np.ascontiguousarray(sa, dtype=np.int32)
            text_u8 = np.ascontiguousarray(text_codes, dtype=np.uint8)
            lcp = np.zeros(n, dtype=np.int32)
            _c_bwt_lib.kasai_lcp(
                text_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                sa32.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                n,
                lcp.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            )
            return lcp
        # Fallback non-jitted Kasai
        lcp = np.zeros(n, dtype=np.int32)
        rank = np.zeros(n, dtype=np.int32)
        for i in range(n):
            rank[sa[i]] = i
        h = 0
        for i in range(n):
            r = rank[i]
            if r > 0:
                j = sa[r - 1]
                while i + h < n and j + h < n and text_codes[i + h] == text_codes[j + h]:
                    h += 1
                lcp[r] = h
                if h > 0:
                    h -= 1
        return lcp


class BWTCore:
    """Core BWT construction and FM-index operations.
    """

    def __init__(self, text: str, sa_sample_rate: int = 32, occ_sample_rate: int = 128):
        """
        Initialize BWT with FM-index.

        Args:
            text: Input text (should end with a single '$' sentinel not present elsewhere)
            sa_sample_rate: Inert. Suffix-array sampling was removed — `locate_positions`
                reads `suffix_array` directly, and at the rate of 1 that `finder` used, the
                sampled dict duplicated the whole array as Python ints (tens of GB per
                large chromosome). Accepted for call-site compatibility.
            occ_sample_rate: Occurrence checkpoints every nth position to reduce memory
        """
        if '$' in text[:-1]:
            raise ValueError("Input text must not contain internal '$' sentinel characters")

        self.text: str = text
        self.n = len(text)
        self.sa_sample_rate = sa_sample_rate
        self.occ_sample_rate = occ_sample_rate

        self.text_arr = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)

        # Build suffix array and BWT (memory-efficient)
        self.suffix_array = self._build_suffix_array()
        self.bwt_arr = self._build_bwt_array()
        self.alphabet = sorted(set(text))
        self.char_to_code = {c: ord(c) for c in self.alphabet}
        self.code_to_char = {ord(c): c for c in self.alphabet}
        self.char_counts, self.char_totals = self._build_char_counts()
        self.char_counts_code = {ord(k): v for k, v in self.char_counts.items()}
        self.char_totals_code = {ord(k): v for k, v in self.char_totals.items()}
        self.occ_checkpoints = self._build_occurrence_checkpoints()

        # Prepare C-accelerated data structures for backward search
        self._c_bwt_ptr = None
        self._c_char_counts = None
        self._c_char_totals = None
        self._c_cp_flat = None
        self._c_cp_offsets = None
        self._c_cp_lengths = None
        if _c_bwt_lib is not None:
            self._prepare_c_bwt_data()

    def clear(self):
        """Release heavy memory structures to let GC reclaim memory."""
        # Replace large attributes with minimal stubs
        self.text = ""
        self.text_arr = np.array([], dtype=np.uint8)
        self.bwt_arr = np.array([], dtype=np.uint8)
        self.suffix_array = np.array([], dtype=np.int32)
        self.occ_checkpoints = {}
        self.char_counts = {}
        self.char_totals = {}
        self.alphabet = []
        self.char_to_code = {}
        self.code_to_char = {}
        self.char_counts_code = {}
        self.char_totals_code = {}
    
    def _build_suffix_array(self) -> np.ndarray:
        """Build suffix array, preferring pydivsufsort (C backend) with a NumPy fallback.

        Fallback uses prefix-doubling with NumPy lexsort (significantly faster than
        Python list.sort + lambdas). Complexity ~O(n log n) sorts.
        """
        # Prefer fast C implementation when available
        try:
            import pydivsufsort  # type: ignore
            sa_list = pydivsufsort.divsufsort(self.text_arr.tobytes())
            return np.array(sa_list, dtype=np.int32)
        except ImportError:
            pass  # pydivsufsort not installed; use NumPy fallback
        except Exception:
            import sys
            print("  Warning: pydivsufsort failed, using NumPy fallback", file=sys.stderr)

        n = self.n
        if n == 0:
            return np.array([], dtype=np.int32)

        # Initial rank from character codes
        codes = self.text_arr.astype(np.int32, copy=False)
        # Compress codes to 0..sigma-1 for stability
        uniq_codes, inv = np.unique(codes, return_inverse=True)
        rank = inv.astype(np.int32, copy=False)
        sa = np.arange(n, dtype=np.int32)

        k = 1
        tmp_rank = np.empty(n, dtype=np.int32)
        idx = np.arange(n, dtype=np.int32)
        while k < n:
            # secondary key is rank[i+k] else -1
            ipk = idx + k
            # Use safe indexing: clip ipk to valid range, then apply condition
            ipk_safe = np.clip(ipk, 0, n - 1)
            key2 = np.where(ipk < n, rank[ipk_safe], -1)
            # Sort by (rank[i], key2[i]) using lexsort with primary last
            sa = np.lexsort((key2, rank))
            # Compute new ranks
            r_sa = rank[sa]
            k2_sa = key2[sa]
            # mark changes
            change = np.empty(n, dtype=np.int32)
            change[0] = 0
            change[1:] = (r_sa[1:] != r_sa[:-1]) | (k2_sa[1:] != k2_sa[:-1])
            new_rank_ordered = np.cumsum(change, dtype=np.int32)
            # remap to original index order
            tmp_rank[sa] = new_rank_ordered
            rank, tmp_rank = tmp_rank, rank
            if rank[sa[-1]] == n - 1:
                break
            k <<= 1
        return sa.astype(np.int32, copy=False)
    
    def _build_bwt_array(self) -> np.ndarray:
        """Build BWT from suffix array as uint8 NumPy array (ASCII codes)."""
        if self.n == 0:
            return np.array([], dtype=np.uint8)
        sa = self.suffix_array.astype(np.int64, copy=False)
        # previous index (sa-1) % n
        prev_idx = (sa - 1) % self.n
        # Gather from numeric text array
        return self.text_arr[prev_idx]
    
    def _build_char_counts(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Count character frequencies and compute cumulative counts C[char]."""
        totals: Dict[str, int] = {c: 0 for c in self.alphabet}
        for ch in self.text:
            totals[ch] += 1
        counts: Dict[str, int] = {}
        cumulative = 0
        for char in self.alphabet:
            counts[char] = cumulative
            cumulative += totals[char]
        return counts, totals
    
    def _build_occurrence_checkpoints(self) -> Dict[int, np.ndarray]:
        """Build checkpointed occurrence counts for efficient rank queries with low memory.

        Returns a mapping from ASCII code -> np.ndarray of counts at positions m*k
        (prefix length), with cp[0] = 0. If the last block is partial, a final
        checkpoint with the total count at n is appended to mirror previous behavior.
        """
        bwt = self.bwt_arr
        n = bwt.size
        k = int(self.occ_sample_rate)
        if n == 0:
            return {}

        checkpoints: Dict[int, np.ndarray] = {}
        # Precompute full cumsum once per distinct code as we have small alphabets
        distinct_codes = np.unique(bwt)
        # indices where boundaries end (1-based length m*k corresponds to index m*k-1)
        block_ends = np.arange(k - 1, n, k, dtype=np.int64)

        for code in distinct_codes.tolist():
            mask = (bwt == code)
            csum = np.cumsum(mask, dtype=np.int32)
            # cp[0]=0, then take counts at each block end
            cp_list = [0]
            if block_ends.size:
                cp_list.extend(csum[block_ends].tolist())
            # Optionally append final count for partial block remainder
            if n % k != 0:
                cp_list.append(int(csum[-1]))
            checkpoints[int(code)] = np.asarray(cp_list, dtype=np.int32)
        # Ensure every alphabet character has a checkpoint array (even if absent)
        for c in self.alphabet:
            code = ord(c)
            if code not in checkpoints:
                # Build an all-zeros checkpoint array of same length as others
                # Determine representative length from any existing array
                any_cp = next(iter(checkpoints.values())) if checkpoints else np.array([0], dtype=np.int32)
                checkpoints[code] = np.zeros_like(any_cp)
        return checkpoints

    def _prepare_c_bwt_data(self):
        """Flatten checkpoint data for C backward_search."""
        # BWT array pointer
        self._c_bwt_ptr = self.bwt_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

        # char_counts and char_totals as 256-element arrays
        cc = (ctypes.c_int * 256)()
        ct = (ctypes.c_int * 256)()
        for ch, val in self.char_counts.items():
            cc[ord(ch)] = val
        for ch, val in self.char_totals.items():
            ct[ord(ch)] = val
        self._c_char_counts = cc
        self._c_char_totals = ct

        # Flatten checkpoints: concatenate all per-code checkpoint arrays
        cp_offsets = (ctypes.c_int * 256)()
        cp_lengths = (ctypes.c_int * 256)()
        flat_parts = []
        offset = 0
        for code in range(256):
            cp = self.occ_checkpoints.get(code)
            if cp is not None and len(cp) > 0:
                cp_offsets[code] = offset
                cp_lengths[code] = len(cp)
                flat_parts.append(np.asarray(cp, dtype=np.int32))
                offset += len(cp)
            else:
                cp_offsets[code] = 0
                cp_lengths[code] = 0

        if flat_parts:
            flat = np.concatenate(flat_parts).astype(np.int32)
        else:
            flat = np.zeros(1, dtype=np.int32)

        self._c_cp_flat_arr = flat  # prevent GC
        self._c_cp_flat = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self._c_cp_offsets = cp_offsets
        self._c_cp_lengths = cp_lengths

        # Pre-allocate output buffers for backward_search
        self._c_out_buf = np.zeros(2, dtype=np.int32)
        self._c_out_sp = self._c_out_buf[0:1].ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self._c_out_ep = self._c_out_buf[1:2].ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    def rank(self, char: Union[str, int], pos: int) -> int:
        """Count occurrences of `char` in bwt[0:pos]. Vectorized with checkpoints.

        Args:
            char: character (str) or ASCII code (int) to count
            pos: count occurrences in bwt[0:pos] (pos can be 0..n)
        """
        if pos <= 0:
            return 0
        if pos > self.n:
            pos = self.n
        code = ord(char) if isinstance(char, str) else int(char)

        # Use C count_equal_range when available (faster than numpy fallback)
        if _c_bwt_lib is not None and self._c_bwt_ptr is not None:
            cp_len = self._c_cp_lengths[code]
            if cp_len == 0:
                return 0
            k = int(self.occ_sample_rate)
            cp_idx = pos // k
            cp_flat_offset = self._c_cp_offsets[code]
            base = self._c_cp_flat_arr[cp_flat_offset + cp_idx]
            cp_pos = cp_idx * k
            if pos > cp_pos:
                base += _c_bwt_lib.count_equal_range(self._c_bwt_ptr, cp_pos, pos, code)
            return int(base)

        cp = self.occ_checkpoints.get(code)
        if cp is None:
            return 0
        k = int(self.occ_sample_rate)
        cp_idx = pos // k
        cp_pos = cp_idx * k
        base = int(cp[cp_idx])
        # Fast remainder scan (Numba-accelerated if available)
        if pos > cp_pos:
            base += int(_count_equal_range(self.bwt_arr, cp_pos, pos, code))
        return base
    
    def backward_search(self, pattern: str) -> Tuple[int, int]:
        """
        Find suffix array interval for pattern using backward search.

        Returns:
            (start, end) interval in suffix array, or (-1, -1) if not found
        """
        if not pattern:
            return (0, self.n - 1)

        # Use C-accelerated backward search when available
        if self._c_bwt_ptr is not None and _c_bwt_lib is not None:
            pat_bytes = pattern.encode('ascii')
            pat_arr = (ctypes.c_ubyte * len(pat_bytes)).from_buffer_copy(pat_bytes)
            _c_bwt_lib.backward_search(
                self._c_bwt_ptr, self.n,
                pat_arr, len(pat_bytes),
                self._c_char_counts, self._c_char_totals,
                self._c_cp_flat, self._c_cp_offsets, self._c_cp_lengths,
                int(self.occ_sample_rate),
                self._c_out_sp, self._c_out_ep
            )
            return (self._c_out_sp[0], self._c_out_ep[0])

        # Initialize with character range
        char = pattern[-1]
        if char not in self.char_counts:
            return (-1, -1)
        # sp inclusive, ep inclusive
        sp = self.char_counts[char]
        ep = sp + self.char_totals[char] - 1
        
        # Process pattern right to left
        for i in range(len(pattern) - 2, -1, -1):
            char = pattern[i]
            if char not in self.char_counts:
                return (-1, -1)

            sp = self.char_counts[char] + self.rank(char, sp)
            ep = self.char_counts[char] + self.rank(char, ep + 1) - 1

            if sp > ep:
                return (-1, -1)
        
        return (sp, ep)
    
    def count_occurrences(self, pattern: str) -> int:
        """Count pattern occurrences in text."""
        sp, ep = self.backward_search(pattern)
        if sp == -1:
            return 0
        return ep - sp + 1
    
    def locate_positions(self, pattern: str) -> List[int]:
        """
        Locate all positions of pattern in text.
        Uses sampled suffix array for efficiency.
        """
        sp, ep = self.backward_search(pattern)
        if sp == -1:
            return []

        # Directly read positions from the suffix array (much faster than LF walking)
        positions = self.suffix_array[sp:ep + 1].tolist()
        positions.sort()
        return positions
    
