Tandem Repeat Finder
Tier 1: Short tandem repeats with FM-index counting and locating

Great for 1 to 10 bp motifs.
Idea
 Enumerate motifs, do fast occurrence counting with backward search, then inspect the positions to see if hits sit back to back with spacing equal to the motif length.
Sketch

1. Build an FM-index on the reference with a unique sentinel per contig.
2. For k in 1..K (K about 10 for STRs), enumerate canonical motifs of length k. Skip motifs that are themselves periodic substrings, so "ATAT" reduces to "AT".
3. Backward-search each motif m to get the suffix array interval [sp, ep].
4. Locate positions for i in [sp..ep].
5. Sort positions by contig then coordinate. Scan the list and collapse runs where consecutive hits differ by exactly k.
6. For each run, compute start, end, number of copies, and check maximality by testing the base immediately left and right. Report as BED or VCF-style record.

Why FM helps
 Counting is O(k) per motif and very fast. You only pay the heavier locate cost for intervals that look promising. For a 2 to 6 bp scan on a plant genome this is very feasible, especially with a larger SA sample rate for faster locating.

Tier 2: Medium to long tandem repeats with SA + LCP on top of BWT

Good for tens to thousands of bp.
Idea
 Tandem repeats show up as plateaus in the LCP array. You can compute or stream the LCP from the BWT and then detect periodic structure.
Sketch

1. Build BWT or FM-index.
2. Compute the LCP array in compressed space from BWT using a Phi or LF based method.
3. Scan LCP and maintain a stack of intervals with LCP ≥ threshold p. For each interval, inspect the suffix array values inside. Differences between adjacent SA values that equal p indicate tandem structure.
4. Validate periodicity by checking that the substring of length L has minimal period p and that L ≥ 2p. Extend left and right to ensure maximality.

Why this works
 All tandem repeats are repeats, and repeats create high LCP. Using LCP avoids enumerating motifs. This finds unknown motifs and long imperfect repeats if you allow a small Hamming or edit slack during validation.

Tier 3: Very long or imperfect tandem arrays with read evidence

Kilobase to megabase arrays, or repeats with many mutations or indels.
Idea
 Map long reads and genotype the repeat length directly across the array. BWT based mappers give you the seeds and anchors, then you infer copy counts from the alignments.
Sketch

1. Map ONT or PacBio reads with a long read aligner.
2. Extract reads that span the candidate locus.
3. Count copies by aligning the motif against the read segment or by dot plot periodicity.
4. Summarize per sample copy number and confidence.

Why this matters
 FM-index on the reference alone cannot resolve very long arrays if reads do not bridge them. Long reads break the ambiguity.
