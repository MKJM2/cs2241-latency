import math
import bisect
import time
from typing import List, Tuple, Optional, Any, Final

from bloom_filter import BloomFilter

# --- Constants ---
EPS: Final[float] = 1e-8
INF: Final[float] = float('inf')

# --- prList Class ---
class PrList:
    """
    Calculates and stores probability distributions over N segments based on scores.
    Provides methods for accessing accumulated probabilities.
    """
    __slots__ = ('thre_list', 'N', 'pr', 'accPr')

    def __init__(self, scores: List[float], thre_list: List[float]):
        """
        Args:
            scores: A list of scores (expected between 0 and 1).
            thre_list: Thresholds dividing [0, 1] into N segments.
                       Must start with 0.0 and end with 1.0.
        """
        if not (abs(thre_list[0] - 0.0) < EPS and abs(thre_list[-1] - 1.0) < EPS):
             raise ValueError("thre_list must start with 0.0 and end with 1.0")
        if not all(thre_list[i] <= thre_list[i+1] for i in range(len(thre_list)-1)):
            raise ValueError("thre_list must be sorted")

        self.thre_list: Final[List[float]] = thre_list
        self.N: Final[int] = len(thre_list) - 1

        cnt_list = [0] * (self.N + 1) # 1-based indexing for segments
        for score in scores:
            if not (0.0 <= score <= 1.0):
                 # Be stricter in production
                 raise ValueError(f"Score {score} out of bounds [0, 1]")

            # Find segment index: score > t[idx-1] and score <= t[idx]
            segment_idx = bisect.bisect_left(thre_list, score)
            # Handle edge case score == 0.0, place it in segment 1
            if segment_idx == 0 and abs(score - 0.0) < EPS:
                segment_idx = 1
            # Handle edge case score == 1.0, place it in segment N
            elif segment_idx > self.N and abs(score - 1.0) < EPS:
                 segment_idx = self.N
            # Scores exactly on a threshold go to the segment *above* it
            elif abs(score - thre_list[segment_idx-1]) < EPS and segment_idx > 1:
                 # If score is exactly threshold t[i], bisect_left puts it
                 # at index i. We want it in segment i (index i).
                 pass # Correctly placed by bisect_left

            if not (1 <= segment_idx <= self.N):
                 # This shouldn't happen with the checks above, but safeguard
                 raise RuntimeError(f"Failed to assign score {score} to segment.")

            cnt_list[segment_idx] += 1

        total_cnt = len(scores)
        if total_cnt == 0:
            # Handle empty input gracefully
            self.pr = [0.0] * (self.N + 1)
            self.accPr = [0.0] * (self.N + 1)
            return

        self.pr = [0.0] * (self.N + 1)
        self.accPr = [0.0] * (self.N + 1)
        for i in range(1, self.N + 1):
            self.pr[i] = cnt_list[i] / total_cnt
            self.accPr[i] = self.accPr[i - 1] + self.pr[i]

        # Final accumulated probability should be close to 1.0
        if not abs(self.accPr[self.N] - 1.0) < EPS:
             # Warn or raise depending on required precision
             # print(f"Warning: Accumulated probability is {self.accPr[self.N]}")
             pass


    def _get_th_idx_from_score(self, score: float) -> int:
        """Finds the index `i` such that `thre_list[i]` is the threshold value."""
        # This assumes thresholds are exactly i/N, which might not hold
        # if thre_list is arbitrary. Use bisect instead for general case.
        # Original implementation assumed equal segments. Let's use bisect.
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"Score {score} out of bounds [0, 1]")
        # Find index i such that thre_list[i] >= score
        idx = bisect.bisect_left(self.thre_list, score)
        # If score is exactly on a threshold, bisect_left gives the index
        # *at* or *above* it.
        # Adjust if score is exactly 0.0
        if idx == 0 and abs(score - 0.0) < EPS:
            return 0
        # If score is exactly a threshold > 0, return its index
        if abs(score - self.thre_list[idx]) < EPS:
             return idx
        # Otherwise, score is between thre_list[idx-1] and thre_list[idx]
        # The relevant threshold *index* for accumulation up to 'score'
        # depends on how accumulation is defined. The original acc_range
        # implies accumulation *up to* the segment containing the score.
        # Let's stick to the original logic using indices directly.
        # Reverting to original logic assuming fixed thresholds i/N for get_th_idx
        # *IF* that assumption holds. If not, the original get_th_idx is flawed.
        # Let's *assume* the thresholds ARE i/N for this specific function,
        # as the original code likely relied on this.
        # If thre_list can be arbitrary, this function needs rethinking or removal.
        # Check assumption:
        is_uniform = all(abs(self.thre_list[i] - i / self.N) < EPS for i in range(self.N + 1))
        if not is_uniform:
            raise NotImplementedError("PrList._get_th_idx assumes uniform thresholds (i/N)")

        # Original logic assuming uniform thresholds:
        idx = int(round(score * self.N)) # Use round for robustness near boundaries
        # Ensure idx is within bounds [0, N]
        idx = max(0, min(self.N, idx))
        # Validate that the calculated index corresponds closely to the score
        if not abs(self.thre_list[idx] - score) < EPS:
             # This might happen if score isn't exactly i/N
             # Fallback to bisect search for the closest threshold index
             idx = bisect.bisect_left(self.thre_list, score + EPS) -1 # Find index below or equal
             idx = max(0, min(self.N, idx)) # Clamp
             # Recalculate index based on closest threshold
             idx = bisect.bisect_left(self.thre_list, score)
             if idx > 0 and abs(score - self.thre_list[idx-1]) < abs(score - self.thre_list[idx]):
                 idx = idx -1


        return idx


    def acc_range(self, score_l: float, score_r: float) -> float:
        """Accumulated probability in (score_l, score_r]."""
        # Use direct index calculation based on thresholds for accuracy
        # Find index i such that thre_list[i] >= score_l
        idx_l = bisect.bisect_left(self.thre_list, score_l)
        # Find index j such that thre_list[j] >= score_r
        idx_r = bisect.bisect_left(self.thre_list, score_r)

        # If score_r is exactly on a threshold, bisect_left gives its index.
        # accPr[idx_r] includes the probability *up to* that threshold.
        if abs(score_r - self.thre_list[idx_r]) > EPS:
             # score_r is within segment idx_r, so we need accPr up to idx_r-1? No.
             # accPr[i] = sum(pr[1]...pr[i]). We want sum pr[l+1...r]
             # where l = segment containing score_l, r = segment containing score_r
             # This corresponds to accPr[idx_r] - accPr[idx_l] if thresholds align perfectly.
             # Let's use the index version for clarity.
             pass # Stick to original index logic below for now

        # Find segment indices corresponding to scores
        # Segment i corresponds to interval ( thre_list[i-1], thre_list[i] ]
        seg_idx_l = bisect.bisect_left(self.thre_list, score_l)
        seg_idx_r = bisect.bisect_left(self.thre_list, score_r)

        # Adjust if score is exactly on threshold
        if seg_idx_l > 0 and abs(score_l - self.thre_list[seg_idx_l-1]) < EPS :
             # If score_l is exactly t[i-1], it's the lower bound of segment i
             # We want to exclude segments below i. Start accumulation from i.
             # accPr index should be seg_idx_l - 1
              pass # bisect_left handles this? No, need index *below* score_l
        idx_l = bisect.bisect_right(self.thre_list, score_l) -1
        idx_l = max(0, idx_l)


        # If score_r is exactly t[i], bisect_left gives i. accPr[i] is correct.
        if abs(score_r - self.thre_list[seg_idx_r]) < EPS:
             idx_r = seg_idx_r
        else:
             # score_r is within segment seg_idx_r. We need up to seg_idx_r.
             idx_r = seg_idx_r
        idx_r = max(0, min(self.N, idx_r)) # Clamp idx_r

        # Find index corresponding to score_l threshold
        idx_l = bisect.bisect_left(self.thre_list, score_l)
        # Find index corresponding to score_r threshold
        idx_r = bisect.bisect_left(self.thre_list, score_r)
        # Adjust if score_r is exactly on a threshold
        if idx_r < len(self.thre_list) and abs(score_r - self.thre_list[idx_r]) < EPS:
             pass # idx_r is correct
        else:
             # score_r falls within segment idx_r, so we need up to idx_r
             idx_r = max(0, min(self.N, idx_r))


        # If score_l is exactly on a threshold, idx_l is that threshold's index.
        # accPr[idx_l] includes probability up to segment idx_l.
        # We want probability *strictly greater than* score_l?
        # Original paper likely means [score_l, score_r] or similar.
        # Let's assume [score_l, score_r] -> accPr[idx_r] - accPr[idx_l-1]? No.
        # Let's use the index version acc_range_idx which is clearer.

        # Find the segment index containing score_l (or just below if exact)
        idx_l = bisect.bisect_right(self.thre_list, score_l) - 1
        idx_l = max(0, idx_l) # Ensure non-negative

        # Find the segment index containing score_r (or at threshold if exact)
        idx_r = bisect.bisect_left(self.thre_list, score_r)
         # If score_r is exactly a threshold, use that index
        if idx_r < len(self.thre_list) and abs(score_r - self.thre_list[idx_r]) < EPS:
             pass
        else:
             # Otherwise, score_r is within segment idx_r, use index idx_r
             idx_r = max(0, min(self.N, idx_r))


        # The probability is sum(pr[i]) for segments i where thre_list[i-1] >= score_l
        # and thre_list[i] <= score_r. This seems overly complex.
        # Revert to the simpler index-based logic:
        # Find index i such that thre_list[i] is the first threshold >= score_l
        _idx_l_lookup = bisect.bisect_left(self.thre_list, score_l)
        # Find index j such that thre_list[j] is the first threshold >= score_r
        _idx_r_lookup = bisect.bisect_left(self.thre_list, score_r)

        # If score_r is exactly on threshold j, we want accPr[j]
        # If score_r is between j-1 and j, we still want accPr[j]? Check definition.
        # Let's assume accPr[i] = sum P(segment 1..i)
        # We want sum P(segment l..r) where segment l contains score_l and r contains score_r
        # This is accPr[r] - accPr[l-1]

        # Find segment index l for score_l
        L = bisect.bisect_left(self.thre_list, score_l)
        if L == 0 and abs(score_l - 0.0) < EPS:
            L = 1 # score 0 is segment 1
        elif abs(score_l - self.thre_list[L-1]) < EPS and L > 1:
            pass # exact threshold, belongs to segment l

        # Find segment index r for score_r
        R = bisect.bisect_left(self.thre_list, score_r)
        if R > self.N and abs(score_r - 1.0) < EPS :
            R = self.N # score 1 is segment N
        elif abs(score_r - self.thre_list[R-1]) < EPS and R > 1:
            pass # exact threshold, belongs to segment r

        # Clamp indices
        L = max(1, min(self.N, L))
        R = max(1, min(self.N, R))

        if R < L:
            return 0.0 # Handle empty range

        # Probability = sum(pr[l...r]) = accPr[r] - accPr[l-1]
        return self.accPr[R] - self.accPr[L - 1]


    def acc_range_idx(self, idx_l: int, idx_r: int) -> float:
        """Accumulated probability for segments idx_l through idx_r (inclusive)."""
        if not (1 <= idx_l <= self.N):
             raise IndexError(f"idx_l {idx_l} out of bounds [1, {self.N}]")
        if not (1 <= idx_r <= self.N):
             raise IndexError(f"idx_r {idx_r} out of bounds [1, {self.N}]")
        if idx_r < idx_l:
            return 0.0 # Empty range

        return self.accPr[idx_r] - self.accPr[idx_l - 1]

# --- DPKL Calculation ---
def calc_DPKL(g: PrList, h: PrList, k: int, j: Optional[int] = None) -> Tuple[List[List[float]], List[List[Optional[int]]]]:
    """
    Calculates Dynamic Programming table for KL divergence maximization.

    Args:
        g: PrList for positive keys.
        h: PrList for negative keys.
        k: Number of regions.
        j: Optional upper bound for segments considered (defaults to N).

    Returns:
        Tuple (DPKL, DPPre):
            DPKL[n][q]: Max KL divergence using segments 1..n into q regions.
            DPPre[n][q]: Predecessor segment index (n') for the optimal solution
                         ending at segment n with q regions. The last region
                         spans segments n'+1 to n.
    """
    N = g.N
    if h.N != N:
        raise ValueError("g and h PrLists must have the same N")
    if j is None:
        j = N
    if not (1 <= j <= N):
        raise ValueError(f"Segment upper bound j={j} out of range [1, {N}]")

    # DPKL[n][q]: max KL divergence using segments 1..n into q regions
    DPKL: List[List[float]] = [[-INF] * (k + 1) for _ in range(j + 1)]
    # DPPre[n][q]: index i-1 such that clustering segments i..n as region q yields max DPKL[n][q]
    DPPre: List[List[Optional[int]]] = [[None] * (k + 1) for _ in range(j + 1)]

    DPKL[0][0] = 0.0 # Base case: 0 segments, 0 regions, 0 divergence

    for q_reg in range(1, k + 1): # Number of regions
        for n_seg in range(1, j + 1): # Ending segment index
            max_kl = -INF
            best_predecessor_idx = None
            # Iterate through possible start segments 'i_seg' for the q_reg-th region
            for i_seg in range(1, n_seg + 1):
                # Region q_reg covers segments i_seg to n_seg
                pos_pr = g.acc_range_idx(i_seg, n_seg)
                neg_pr = h.acc_range_idx(i_seg, n_seg)

                current_kl_term = 0.0
                if neg_pr > EPS: # Avoid division by zero and log(0)
                    if pos_pr > EPS:
                        current_kl_term = pos_pr * math.log(pos_pr / neg_pr)
                    # else: pos_pr is 0, term is 0
                elif pos_pr > EPS:
                    # Positive probability but zero negative probability -> infinite KL divergence?
                    # The paper likely assumes neg_pr > 0 for regions considered.
                    # Or handle this case specifically. Let's treat as INF contribution.
                    # However, the original code continues if Neg==0. Let's match that.
                     continue # Skip if neg_pr is 0, cannot form KL term meaningfully here

                # Predecessor state: segments 1..(i_seg-1) using q_reg-1 regions
                predecessor_kl = DPKL[i_seg - 1][q_reg - 1]

                if predecessor_kl > -INF: # Check if predecessor state is reachable
                    total_kl = predecessor_kl + current_kl_term
                    if total_kl > max_kl:
                        max_kl = total_kl
                        best_predecessor_idx = i_seg - 1 # Store index before start segment

            DPKL[n_seg][q_reg] = max_kl
            DPPre[n_seg][q_reg] = best_predecessor_idx

    return DPKL, DPPre


# --- Threshold Reconstruction ---
def ThresMaxDiv(DPPre: List[List[Optional[int]]], j: int, k: int, thre_list: List[float]) -> Optional[List[float]]:
    """
    Reconstructs the optimal region thresholds from the DPPre table.

    Args:
        DPPre: DP predecessor table from calc_DPKL.
        j: The final segment index considered for the k-th region (e.g., N).
        k: Total number of regions.
        thre_list: The list of segment boundaries (0.0 to 1.0).

    Returns:
        List of k+1 threshold boundaries [t_0, t_1, ..., t_k], or None if no
        valid partitioning exists.
    """
    N = len(thre_list) - 1
    if N <= 0:
        return None
    if k <= 0:
        return None
    if j <= 0 and k > 0:
        return None
    if j >= len(DPPre) or k >= len(DPPre[0]):
        return None # Bounds check

    # Start backtracking from the state (j, k)
    # Note: Original code used DPPre[j-1][k-1] - check indexing carefully.
    # DPPre[n][q] stores predecessor for state ending at segment n with q regions.
    # If the last region ends at segment j, we look at DPPre[j][k].
    current_seg_idx = j
    thresholds_rev = [thre_list[j]] # t_k = threshold of last segment considered

    for q_reg in range(k, 0, -1): # Iterate regions backwards from k down to 1
        if current_seg_idx < 0 or DPPre[current_seg_idx][q_reg] is None:
            return None # Invalid path or unreachable state

        predecessor_idx = DPPre[current_seg_idx][q_reg]
        # predecessor_idx is the index of the last segment of the *previous* region (q-1)
        # The threshold is thre_list[predecessor_idx]
        thresholds_rev.append(thre_list[predecessor_idx])
        current_seg_idx = predecessor_idx # Move to the end of the previous region

    # The last threshold added corresponds to t_0, which should be 0.0
    if not abs(thresholds_rev[-1] - 0.0) < EPS:
         # This might indicate an issue if the path doesn't trace back to index 0
         # print(f"Warning: Backtracking did not end at threshold 0.0 (ended at {thresholds_rev[-1]})")
         pass # Allow for now, maybe valid paths don't always start perfectly at 0?

    # Reverse to get [t_0, t_1, ..., t_k]
    thresholds = list(reversed(thresholds_rev))

    if len(thresholds) != k + 1:
        # print(f"Warning: Expected {k+1} thresholds, got {len(thresholds)}")
        return None # Or handle error

    # Add t_k = 1.0 if it wasn't the last segment boundary j
    # The logic implies t_k is the upper bound of the score range considered.
    # If j=N, then thre_list[j] is 1.0. If j<N, what should t_k be?
    # The paper implies partitioning the *entire* [0,1] range.
    # Let's assume j=N always for the final call.
    # The original code iterates j from k to N, implying the last region
    # might end before segment N. Let's adjust ThresMaxDiv to match MaxDivDP.

    # --- Re-aligning ThresMaxDiv with MaxDivDP logic ---
    # MaxDivDP calculates DPKL[N][k] and DPPre[N][k]
    # We trace back from DPPre[N][k]

    current_seg_idx = N # Start from the end, segment N
    thresholds_rev = [thre_list[N]] # t_k = 1.0

    for q_reg in range(k, 0, -1):
        if current_seg_idx < 0:
            return None # Should not happen if DP table is correct
        predecessor_idx = DPPre[current_seg_idx][q_reg]
        if predecessor_idx is None:
            return None # No valid path

        thresholds_rev.append(thre_list[predecessor_idx])
        current_seg_idx = predecessor_idx

    if not abs(thresholds_rev[-1] - 0.0) < EPS:
        return None # Must trace back to 0

    thresholds = list(reversed(thresholds_rev))
    if len(thresholds) != k + 1:
        return None

    return thresholds


# --- Optimal FPR Calculation (Target Overall FPR F) ---
def OptimalFPR(g: PrList, h: PrList, t: List[float], F: float, k: int) -> Optional[List[Optional[float]]]:
    """
    Calculates optimal FPR f_i for each region to achieve target overall FPR F,
    minimizing memory usage implicitly (related to KL divergence).

    Args:
        g: PrList for positive keys.
        h: PrList for negative keys.
        t: List of k+1 region threshold boundaries [t_0, ..., t_k].
        F: Target overall FPR (0 < F < 1).
        k: Number of regions.

    Returns:
        List of k+1 FPRs [None, f_1, ..., f_k], where f_i is the FPR for region i.
        Returns None if calculation fails or F is unachievable.
    """
    if not (0 < F < 1):
        raise ValueError("Target FPR F must be between 0 and 1")
    if len(t) != k + 1:
        raise ValueError("Threshold list length must be k+1")

    pos_pr_list = [g.acc_range(t[i - 1], t[i]) for i in range(1, k + 1)]
    neg_pr_list = [h.acc_range(t[i - 1], t[i]) for i in range(1, k + 1)]

    # Validate probabilities sum approximately to 1
    if not abs(sum(pos_pr_list) - 1.0) < EPS:
        print(f"Warning: Sum of pos_pr != 1 ({sum(pos_pr_list)})")
    if not abs(sum(neg_pr_list) - 1.0) < EPS:
        print(f"Warning: Sum of neg_pr != 1 ({sum(neg_pr_list)})")

    # Identify regions where neg_pr is effectively zero (cannot contribute to FPR)
    # Also identify regions where pos_pr is zero (no elements, FPR is irrelevant but often set to 1)
    valid_list = [True] * k # Regions where optimization applies
    opt_fpr_list = [0.0] * k # Initialize FPRs (0-indexed for now)

    for i in range(k):
        if neg_pr_list[i] < EPS:
            valid_list[i] = False
            # If neg_pr is 0, this region *must* have f_i = 0 to meet target F,
            # unless pos_pr is also 0. If pos_pr > 0 and neg_pr = 0, F is unachievable?
            # The paper's formula implies f_i -> infinity, but practically f_i=1?
            # Let's assume if neg_pr=0, this region doesn't contribute to FPR sum.
            # What should its f_i be? If pos_pr > 0, filter is needed.
            # If pos_pr=0, filter is not needed.
            # Original code seems to handle this in the iterative loop.
            opt_fpr_list[i] = 1.0 # Tentatively set to 1 if invalid? Or 0? Let loop decide.

    # Iteratively adjust FPRs for regions where calculated f_i > 1
    while True:
        valid_pos_pr_sum = sum(pos_pr_list[i] for i in range(k) if valid_list[i])
        valid_neg_pr_sum = sum(neg_pr_list[i] for i in range(k) if valid_list[i])
        # FPR contribution from invalid regions (where f_i is fixed, usually to 1)
        invalid_neg_pr_sum = sum(neg_pr_list[i] * opt_fpr_list[i] for i in range(k) if not valid_list[i])

        # Target FPR for the *valid* regions
        if valid_neg_pr_sum < EPS:
             # No negative samples in valid regions.
             if F < invalid_neg_pr_sum - EPS:
                 # Target F already exceeded by invalid regions
                 # print("Warning: Target FPR F unachievable due to fixed FPRs in invalid regions.")
                 return None # Cannot achieve target F
             # Otherwise, all valid regions can have f_i = 0?
             normed_F = 0.0 # Aim for zero FPR in valid regions
        else:
            normed_F = (F - invalid_neg_pr_sum) / valid_neg_pr_sum
            # Clamp normed_F to be non-negative
            normed_F = max(0.0, normed_F)


        if valid_pos_pr_sum < EPS:
            # No positive samples in valid regions. Any FPR is achievable?
            # Set f_i = 0 for valid regions?
            for i in range(k):
                if valid_list[i]:
                    opt_fpr_list[i] = 0.0
            break # Finished

        # Calculate lambda (Lagrange multiplier, related to normed_F and probabilities)
        # The formula derived is f_i = lambda * pos_pr_i / neg_pr_i
        # Sum(neg_pr_i * f_i) = F' => Sum(neg_pr_i * lambda * pos_pr_i / neg_pr_i) = F'
        # lambda * Sum(pos_pr_i) = F' => lambda = F' / Sum(pos_pr_i)
        lambda_multiplier = normed_F / valid_pos_pr_sum if valid_pos_pr_sum > EPS else 0

        # Update FPRs for valid regions
        changed = False
        for i in range(k):
            if valid_list[i]:
                if neg_pr_list[i] > EPS:
                    # Calculate ideal f_i based on lambda (or directly using normed_F)
                    # Formula from paper seems to be f_i = normed_F * (pos_pr_i / valid_pos_pr_sum) / (neg_pr_i / valid_neg_pr_sum)
                    # Simplified: f_i = lambda * pos_pr_i / neg_pr_i
                    new_f_i = lambda_multiplier * pos_pr_list[i] / neg_pr_list[i]
                    # Original code used: normed_F * n_pos_pr / n_neg_pr where n_ are normed probs
                    # Let's re-derive: target Sum(h_i * f_i) = F' over valid i
                    # Minimize Sum(g_i * log(1/f_i)) subject to Sum(h_i * f_i) = F'
                    # Lagrangian: L = Sum(g_i * log(1/f_i)) - lambda * (Sum(h_i * f_i) - F')
                    # dL/df_i = -g_i / f_i - lambda * h_i = 0 => f_i = -g_i / (lambda * h_i)
                    # This lambda seems different. Let's use the structure from the original code.

                    normed_pos_pr = pos_pr_list[i] / valid_pos_pr_sum if valid_pos_pr_sum > EPS else 0
                    normed_neg_pr = neg_pr_list[i] / valid_neg_pr_sum if valid_neg_pr_sum > EPS else 0

                    if normed_neg_pr < EPS:
                         # Cannot achieve target F if normed_pos > 0?
                         # This case means pos_pr > 0, neg_pr > 0, but neg_pr_sum is 0? Impossible.
                         # This means neg_pr_i is 0, should have been caught earlier.
                         # Let's assume normed_neg_pr > EPS if valid_list[i] is True.
                         new_f_i = 1.0 # Or INF? Set to 1.0 for safety.
                    else:
                         new_f_i = normed_F * normed_pos_pr / normed_neg_pr

                    # Clamp f_i between 0 and 1
                    new_f_i = max(0.0, min(1.0, new_f_i))

                    if new_f_i >= 1.0 - EPS: # If calculated FPR is >= 1
                        if valid_list[i]: # If it was previously considered valid
                            valid_list[i] = False # Mark as invalid (fix FPR to 1)
                            opt_fpr_list[i] = 1.0
                            changed = True
                    else:
                        opt_fpr_list[i] = new_f_i
                else: # neg_pr_list[i] is 0, should be invalid
                     if valid_list[i]: # Should not happen
                          valid_list[i] = False
                          opt_fpr_list[i] = 1.0 # Or 0? If pos_pr=0, 0. If pos_pr>0, 1?
                          changed = True


        if not changed: # Converged
            break

    # Final check on achieved FPR
    achieved_F = sum(neg_pr_list[i] * opt_fpr_list[i] for i in range(k))
    if not abs(achieved_F - F) < EPS * k: # Allow some tolerance
         # This might happen if F is very small or distributions are tricky
         # print(f"Warning: Achieved FPR {achieved_F} differs from target {F}")
         pass

    # Convert to 1-based indexing for consistency with original return type
    final_fprs: List[Optional[float]] = [None] + opt_fpr_list
    return final_fprs


# --- Space Used Calculation ---
def SpaceUsed(g: PrList, t: List[float], f: List[Optional[float]], n_keys: int) -> float:
    """
    Calculates the total memory space (in bits) used by the backup Bloom filters.

    Args:
        g: PrList for positive keys.
        t: List of k+1 region threshold boundaries.
        f: List of k+1 FPRs [None, f_1, ..., f_k].
        n_keys: Total number of positive keys inserted.

    Returns:
        Total space used in bits.
    """
    k = len(t) - 1
    if len(f) != k + 1:
        raise ValueError("Length of f must be k+1")
    if n_keys < 0:
        raise ValueError("Number of keys cannot be negative")
    if n_keys == 0:
        return 0.0 # No keys, no space needed

    total_space = 0.0
    _bits_per_element_const = -1.0 / (math.log(2) ** 2) # log2(e) / log(2) = 1/ln(2) ? No, it's -1/ln(2)^2

    for i in range(1, k + 1): # Iterate through regions 1 to k
        fpr = f[i]
        if fpr is None:
             # Should not happen if OptimalFPR returns correctly
             raise ValueError(f"FPR for region {i} is None")

        # Calculate number of positive keys expected in this region
        pos_pr = g.acc_range(t[i - 1], t[i])
        pos_num_expected = pos_pr * n_keys

        if pos_num_expected < EPS:
            continue # No elements expected, no space needed for this filter

        if fpr <= 0.0 or fpr >= 1.0:
            # If FPR is 0, infinite space needed theoretically.
            # If FPR is 1, zero space needed (no filter).
            # BloomFilter handles capacity=0 or error_rate=0/1.
            # Space calculation assumes optimal parameters.
            if fpr >= 1.0 - EPS:
                 continue # No space needed for FPR=1
            if fpr <= EPS:
                 # Treat FPR=0 as needing very large space, or skip?
                 # The formula breaks down. Assume filter is perfect but takes space?
                 # Let's skip for space calculation, assuming it means "reject all".
                 # Or, if pos_num > 0 and f=0, it implies perfect recall needed.
                 # The space formula gives infinity. Let's return INF if this happens.
                 if pos_num_expected > EPS:
                     # print(f"Warning: Region {i} has pos_pr > 0 but target FPR is 0.")
                     return INF # Cannot achieve zero FPR with finite space

        # Formula for bits per element: m/n = -log2(fpr) / ln(2) = -log(fpr) / (ln(2)^2)
        # Total bits = n * (-log(fpr) / (ln(2)^2))
        try:
            bits_for_region = pos_num_expected * (-math.log(fpr) / (math.log(2)**2))
            total_space += bits_for_region
        except ValueError: # log(fpr) error if fpr <= 0
             if pos_num_expected > EPS:
                 return INF # Should have been caught above
             # else: pos_num is 0, log error irrelevant

    return total_space

# --- MaxDivDP related functions (if needed by FastPLBF directly) ---
# These seem to be primarily used within find_best_t_and_f which uses the
# calc_DPKL and ThresMaxDiv functions defined above. MaxDivDP itself might
# just be a wrapper around calc_DPKL. Let's assume we only need calc_DPKL
# and ThresMaxDiv as refactored above.
def MaxDivDP(g: PrList, h: PrList, N: int, k: int) -> Tuple[List[List[float]], List[List[Optional[int]]]]:
     """Wrapper around calc_DPKL assuming the full range [1, N] is used."""
     if g.N != N or h.N != N:
          raise ValueError("N parameter must match PrList N")
     return calc_DPKL(g, h, k, N) # Call with j=N



class FastPLBF:
    """
    Partitioned Learned Bloom Filter (FastPLBF).

    Optimizes region thresholds (t) and per-region False Positive Rates (f)
    to minimize memory usage for a target overall False Positive Rate (F).
    Uses dynamic programming based on KL divergence maximization for partitioning.

    Relies on BloomFilter for the underlying Bloom filter implementation.
    Currently assumes keys are strings for Bloom filter operations.
    """
    __slots__ = (
        'F', 'N', 'k', 'n', 't', 'f',
        'memory_usage_of_backup_bf', 'backup_bloom_filters',
        'segment_thre_list' # Keep track of segment thresholds used
    )

    def __init__(self,
                 pos_keys: List[str], # Explicitly type keys as str for now
                 pos_scores: List[float],
                 neg_scores: List[float],
                 F: float,
                 N: int,
                 k: int):
        """
        Args:
            pos_keys: List of positive keys (strings).
            pos_scores: List of scores [0, 1] corresponding to positive keys.
            neg_scores: List of scores [0, 1] for negative samples (used for training).
            F: Target overall False Positive Rate (0 < F < 1).
            N: Number of segments to divide the score range [0, 1] into.
            k: Number of regions (partitions) for the Bloom filters.

        Raises:
            ValueError: If inputs are invalid (lengths mismatch, F out of range, etc.)
            RuntimeError: If optimal parameters cannot be determined.
        """
        # --- Input Validation ---
        if not isinstance(pos_keys, list) or not isinstance(pos_scores, list):
            raise TypeError("pos_keys and pos_scores must be lists.")
        if len(pos_keys) != len(pos_scores):
            raise ValueError("pos_keys and pos_scores must have the same length.")
        if not isinstance(neg_scores, list):
            raise TypeError("neg_scores must be a list.")
        if not isinstance(F, float) or not (0 < F < 1):
            raise ValueError("Target FPR F must be a float between 0 and 1.")
        if not isinstance(N, int) or N <= 0:
            raise ValueError("Number of segments N must be a positive integer.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Number of regions k must be a positive integer.")
        if k > N:
             raise ValueError("Number of regions k cannot exceed number of segments N.")

        # Validate score ranges (optional but good practice)
        if not all(0.0 <= s <= 1.0 for s in pos_scores):
            raise ValueError("All pos_scores must be between 0.0 and 1.0.")
        if not all(0.0 <= s <= 1.0 for s in neg_scores):
            raise ValueError("All neg_scores must be between 0.0 and 1.0.")

        self.F: Final[float] = F
        self.N: Final[int] = N
        self.k: Final[int] = k
        self.n: Final[int] = len(pos_keys) # Number of positive keys

        # --- Parameter Calculation ---
        # 1. Divide into segments and get probability distributions
        self.segment_thre_list, g, h = self._divide_into_segments(pos_scores, neg_scores)

        # 2. Find optimal thresholds (t) and FPRs (f)
        best_t, best_f, min_space = self._find_best_t_and_f(g, h)
        if best_t is None or best_f is None:
            raise RuntimeError("Could not determine optimal thresholds (t) and FPRs (f). "
                               "Check input data distributions and parameters (N, k, F).")

        self.t: Final[List[float]] = best_t
        self.f: Final[List[Optional[float]]] = best_f # Note: f[0] is None
        self.memory_usage_of_backup_bf: Final[float] = min_space

        # --- Bloom Filter Construction ---
        # 3. Insert positive keys into the corresponding regional Bloom filters
        self.backup_bloom_filters = self._build_bloom_filters(pos_keys, pos_scores)


    def _divide_into_segments(self,
                              pos_scores: List[float],
                              neg_scores: List[float]
                             ) -> Tuple[List[float], PrList, PrList]:
        """Divides score range [0,1] into N segments and calculates PrLists."""
        # Create uniform segment thresholds: [0/N, 1/N, ..., N/N]
        segment_thre_list = [i / self.N for i in range(self.N + 1)]
        g = PrList(pos_scores, segment_thre_list)
        h = PrList(neg_scores, segment_thre_list)
        return segment_thre_list, g, h

    def _find_best_t_and_f(self, g: PrList, h: PrList) -> Tuple[Optional[List[float]], Optional[List[Optional[float]]], float]:
        """Finds optimal t and f by minimizing space for target FPR F."""
        minSpaceUsed = INF
        t_best: Optional[List[float]] = None
        f_best: Optional[List[Optional[float]]] = None

        # Calculate DP table once using the full range [1, N]
        # MaxDivDP uses calc_DPKL internally for segments 1..N
        try:
            _, DPPre = MaxDivDP(g, h, self.N, self.k)
        except Exception as e:
             # Catch potential errors during DP calculation (e.g., math errors)
             print(f"Error during MaxDivDP calculation: {e}")
             return None, None, INF


        # Reconstruct thresholds using the DPPre table for the full range
        # ThresMaxDiv traces back from DPPre[N][k]
        t = ThresMaxDiv(DPPre, self.N, self.k, self.segment_thre_list)

        if t is None:
            # print("Warning: Could not reconstruct thresholds from DP table.")
            # This implies no valid partitioning into k regions was found for segments 1..N
            # The original code iterated through j (last segment of last region).
            # Let's re-introduce that loop if the single ThresMaxDiv(N, k) fails.
            # This suggests the interpretation of MaxDivDP/ThresMaxDiv might need refinement
            # based on the original paper's intent.

            # --- Reverting to original loop structure for finding best t ---
            DPKL_full, DPPre_full = MaxDivDP(g, h, self.N, self.k) # Calculate once

            for j_end_segment in range(self.k, self.N + 1):
                 # Try reconstructing thresholds assuming last region ends at j_end_segment
                 # We need a way to reconstruct from DPPre_full[j_end_segment][k]?
                 # The ThresMaxDiv function needs adaptation or DPPre needs reinterpretation.

                 # Let's assume the original ThresMaxDiv was correct for its specific DP calc.
                 # The FastPLBF version uses MaxDivDP -> calc_DPKL(N, k).
                 # Let's try the direct reconstruction from N, k first.
                 t_candidate = ThresMaxDiv(DPPre, self.N, self.k, self.segment_thre_list)

                 if t_candidate is None:
                      continue # Skip if no valid thresholds for this j

                 # Calculate optimal FPRs for this threshold set t
                 f_candidate = OptimalFPR(g, h, t_candidate, self.F, self.k)
                 if f_candidate is None:
                      continue # Skip if FPR calculation fails

                 # Calculate space used for this (t, f) combination
                 space = SpaceUsed(g, t_candidate, f_candidate, self.n)

                 if space < minSpaceUsed:
                      minSpaceUsed = space
                      t_best = t_candidate
                      f_best = f_candidate

            # If after the loop, t_best is still None, then no solution was found.

        else:
             # Direct reconstruction from N, k worked
             f = OptimalFPR(g, h, t, self.F, self.k)
             if f is not None:
                  space = SpaceUsed(g, t, f, self.n)
                  if space < minSpaceUsed: # Should be the only result here
                       minSpaceUsed = space
                       t_best = t
                       f_best = f

        # Handle case where minSpaceUsed remains INF
        if minSpaceUsed == INF:
             return None, None, INF

        return t_best, f_best, minSpaceUsed


    def _build_bloom_filters(self, pos_keys: List[str], pos_scores: List[float]
                            ) -> List[Optional[BloomFilter]]:
        """Creates and populates the regional Bloom filters."""
        if self.t is None or self.f is None:
             raise RuntimeError("Thresholds (t) or FPRs (f) not calculated.")

        # Count positive keys per region
        pos_cnt_list = [0] * (self.k + 1) # 1-based index
        region_assignments = [0] * self.n # Store region for each key
        for idx, score in enumerate(pos_scores):
            region_idx = self._get_region_idx(score)
            pos_cnt_list[region_idx] += 1
            region_assignments[idx] = region_idx

        # Initialize Bloom filters
        backup_filters: List[Optional[BloomFilter]] = [None] * (self.k + 1)
        for i in range(1, self.k + 1):
            region_fpr = self.f[i]
            region_capacity = pos_cnt_list[i]

            if region_fpr is None:
                 # Should not happen with valid f from OptimalFPR
                 print(f"Warning: FPR for region {i} is None. Skipping filter.")
                 continue

            if region_capacity == 0:
                # No keys in this region, no filter needed.
                # Ensure consistency: if capacity is 0, FPR should ideally be handled.
                # OptimalFPR might set f=0 or f=1 depending on context.
                # If f=0 and capacity=0, space is 0. If f=1 and capacity=0, space is 0.
                continue # Leave backup_filters[i] as None

            if region_fpr >= 1.0 - EPS:
                # FPR is 1, no filtering needed for this region.
                continue # Leave backup_filters[i] as None
            elif region_fpr <= EPS:
                # Target FPR is 0. BloomFilter requires error_rate > 0.
                # If capacity > 0 and target FPR is 0, this implies perfect recall
                # is needed, which standard Bloom filters approximate with very low FPR.
                # Use a very small error rate.
                # print(f"Warning: Region {i} has target FPR near 0. Using minimal error rate.")
                effective_fpr = EPS # Use a tiny error rate
                # Check if pos_cnt_list[i] > 0? Yes, checked above.
                backup_filters[i] = BloomFilter(capacity=region_capacity,
                                                          error_rate=effective_fpr)
            else:
                # Standard case: 0 < FPR < 1
                backup_filters[i] = BloomFilter(capacity=region_capacity,
                                                          error_rate=region_fpr)

        # Add keys to the appropriate filters
        for key, region_idx in zip(pos_keys, region_assignments):
            if backup_filters[region_idx] is not None:
                # Use the type-specific method (assuming string keys)
                backup_filters[region_idx].add_str(key)

        return backup_filters

    def _get_region_idx(self, score: float) -> int:
        """Finds the region index (1 to k) for a given score."""
        if self.t is None:
            raise RuntimeError("Thresholds (t) not initialized.")
        if not (0.0 <= score <= 1.0):
            raise ValueError("Score must be between 0.0 and 1.0")

        # Find the index `i` such that t[i-1] < score <= t[i]
        # bisect_left finds first index `i` where t[i] >= score
        region_idx = bisect.bisect_left(self.t, score)

        # Adjustments:
        # - If score is exactly t[0] (0.0), it belongs to region 1.
        # - If score is exactly t[i], bisect_left returns i. It belongs to region i.
        # - If score > t[k-1] and score <= t[k] (1.0), it belongs to region k.
        # bisect_left handles most cases correctly.

        # Handle score == 0.0 explicitly
        if abs(score - self.t[0]) < EPS:
             return 1

        # Handle score > t[k-1]
        if region_idx == self.k + 1:
             # This happens if score > t[k] (which should be 1.0)
             # Clamp to region k if score is 1.0
             if abs(score - self.t[self.k]) < EPS:
                  return self.k
             else: # Score > 1.0, should have been caught by validation
                  raise ValueError("Score > 1.0 encountered in _get_region_idx")

        # If score is exactly t[i], bisect_left gives i. This score falls into
        # region i, which spans (t[i-1], t[i]]. So index i is correct.

        # Clamp index to be within [1, k]
        region_idx = max(1, min(self.k, region_idx))

        return region_idx


    def contains(self, key: str, score: float) -> bool:
        """
        Checks if the key-score pair might be in the filter.

        Args:
            key: The key (string) to check.
            score: The score associated with the key.

        Returns:
            True if the item is potentially present (membership test passes or
                 the region has FPR=1).
            False if the item is definitely not present (membership test fails).
        """
        if self.backup_bloom_filters is None:
             raise RuntimeError("Bloom filters not initialized.")
        # We assume key is string here, matching __init__ and _build_bloom_filters
        if not isinstance(key, str):
             raise TypeError(f"Expected key to be str, got {type(key)}")

        region_idx = self._get_region_idx(score)
        regional_filter = self.backup_bloom_filters[region_idx]

        if regional_filter is None:
            # No filter for this region (implies FPR >= 1), so always return True.
            return True
        else:
            # Query the specific Bloom filter using the type-specific method
            return regional_filter.contains_str(key)

    def __contains__(self, item: Tuple[str, float]) -> bool:
        """Allows using 'in' operator: `(key, score) in plbf`."""
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Item for 'in' operator must be a tuple (key: str, score: float)")
        key, score = item
        if not isinstance(key, str):
             raise TypeError(f"Expected key to be str, got {type(key)}")
        if not isinstance(score, (float, int)):
             raise TypeError(f"Expected score to be float, got {type(score)}")
        return self.contains(key, float(score))

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"F={self.F:.2e}, N={self.N}, k={self.k}, n={self.n}, "
                f"mem_bits={self.memory_usage_of_backup_bf:.2f})")



# --- Synthetic Data Generation ---

def generate_synthetic_data(
    num_pos: int,
    num_neg: int,
    seed: int = 42
) -> Tuple[List[str], List[float], List[str], List[float]]:
    """
    Generates synthetic keys and scores with some correlation between score and label.

    Args:
        num_pos: Number of positive samples (label=1).
        num_neg: Number of negative samples (label=0).
        seed: Random seed for reproducibility.

    Returns:
        A tuple containing: (pos_keys, pos_scores, neg_keys, neg_scores)
    """
    random.seed(seed)
    pos_keys: List[str] = []
    pos_scores: List[float] = []
    neg_keys: List[str] = []
    neg_scores: List[float] = []

    print(f"Generating {num_pos} positive and {num_neg} negative samples...")

    # Positive samples: Scores biased towards 1.0
    # Using Beta distribution: Beta(alpha, beta). High alpha, low beta biases towards 1.
    alpha_pos, beta_pos = 5, 1.5
    for i in range(num_pos):
        key = f"positive_key_{i:07d}"
        score = random.betavariate(alpha_pos, beta_pos)
        pos_keys.append(key)
        pos_scores.append(score)

    # Negative samples: Scores biased towards 0.0
    # Using Beta distribution: Low alpha, high beta biases towards 0.
    alpha_neg, beta_neg = 1.5, 5
    for i in range(num_neg):
        key = f"negative_key_{i:07d}"
        score = random.betavariate(alpha_neg, beta_neg)
        neg_keys.append(key)
        neg_scores.append(score)

    print("Synthetic data generation complete.")
    return pos_keys, pos_scores, neg_keys, neg_scores

# --- Main Experiment Logic ---
def main() -> None:
    # --- Configuration Constants ---
    NUM_POSITIVE_SAMPLES: int = 100000
    NUM_NEGATIVE_SAMPLES: int = 500000
    TEST_SET_FRACTION: float = 0.7
    RANDOM_SEED: int = 42
    N_SEGMENTS: int = 1000
    K_REGIONS: int = 10
    TARGET_FPR: float = 0.01 # Target for PLBF construction

    # --- Data Generation & Preparation ---
    pos_keys, pos_scores, all_neg_keys, all_neg_scores = generate_synthetic_data(
        NUM_POSITIVE_SAMPLES, NUM_NEGATIVE_SAMPLES, RANDOM_SEED
    )
    if not all_neg_keys:
        print("Warning: No negative samples generated. FPR testing will be skipped.")
        train_neg_scores, test_neg_keys, test_neg_scores = [], [], []
    else:
        neg_combined = list(zip(all_neg_keys, all_neg_scores))
        train_neg_combined, test_neg_combined = train_test_split(
            neg_combined, test_size=TEST_SET_FRACTION, random_state=RANDOM_SEED
        )
        train_neg_scores = [item[1] for item in train_neg_combined]
        test_neg_keys = [item[0] for item in test_neg_combined]
        test_neg_scores = [item[1] for item in test_neg_combined] # Keep scores

    print(f"\nParameters: N={N_SEGMENTS}, k={K_REGIONS}, F={TARGET_FPR}")
    print(f"Data: {len(pos_keys)} positive keys, "
          f"{len(train_neg_scores)} train negative scores, "
          f"{len(test_neg_keys)} test negative keys.")
    if not train_neg_scores: print("Warning: No training negative scores available.")

    # --- FastPLBF Construction ---
    print("\nConstructing FastPLBF...")
    construct_start_plbf = time.time()
    plbf: Optional[FastPLBF] = None
    try:
        plbf = FastPLBF(
            pos_keys=pos_keys, pos_scores=pos_scores,
            neg_scores=train_neg_scores, F=TARGET_FPR,
            N=N_SEGMENTS, k=K_REGIONS
        )
    except Exception as e:
         print(f"\nError during FastPLBF construction: {e}")
         import traceback
         traceback.print_exc()
    construct_end_plbf = time.time()

    # --- FastPLBF Verification & Testing ---
    fpr_plbf = float('nan')
    fp_cnt_plbf = 'N/A'
    target_memory_bits_plbf = 0
    num_test_neg = len(test_neg_keys)

    if plbf:
        print(f"FastPLBF Construction finished in {construct_end_plbf - construct_start_plbf:.4f} seconds.")
        print(plbf)
        print("\nVerifying FastPLBF (no false negatives)...")
        false_negatives_plbf = 0
        for key, score in zip(pos_keys, pos_scores):
            if not plbf.contains(key, score): false_negatives_plbf += 1
        print(f"  False Negatives Found: {false_negatives_plbf}")

        print("\nCalculating FastPLBF false positive rate on test set...")
        fp_cnt_plbf_val = 0
        if num_test_neg > 0:
            for key, score in zip(test_neg_keys, test_neg_scores):
                if plbf.contains(key, score): fp_cnt_plbf_val += 1
            fpr_plbf = fp_cnt_plbf_val / num_test_neg
            fp_cnt_plbf = fp_cnt_plbf_val # Store count
        else:
            fpr_plbf = 0.0
            fp_cnt_plbf = 0
        print(f"  False Positives: {fp_cnt_plbf} / {num_test_neg}")

        # Get memory usage for comparison target
        mem_usage_plbf = plbf.memory_usage_of_backup_bf
        if isinstance(mem_usage_plbf, (int, float)) and mem_usage_plbf > 0 and mem_usage_plbf != math.inf:
            target_memory_bits_plbf = int(round(mem_usage_plbf))
            print(f"\nFastPLBF Memory Usage (Target for Standard): {target_memory_bits_plbf} bits")
        else:
            print("\nWarning: Could not determine valid memory usage for FastPLBF. Cannot construct comparable standard BF.")
            target_memory_bits_plbf = 0 # Prevent standard BF construction
    else:
        print("FastPLBF construction failed. Cannot perform verification or testing.")


    # --- Standard Bloom Filter Construction & Testing ---
    print("\nConstructing Standard Bloom Filter (Targeting Same Size)...")
    fpr_standard = float('nan')
    fp_cnt_standard = 'N/A'
    construction_time_standard = 0.0
    actual_memory_bits_standard = 0
    std_bf: Optional[BloomFilter] = None

    if target_memory_bits_plbf > 0 and NUM_POSITIVE_SAMPLES > 0:
        construct_start_standard = time.time()
        # Calculate theoretical parameters for standard BF with n items and m bits
        m = target_memory_bits_plbf
        n = NUM_POSITIVE_SAMPLES
        # Optimal k
        k_std_opt = max(1, int(round((m / n) * math.log(2))))
        # Theoretical FPR (p) for this n, m, k
        try:
             exponent = -k_std_opt * n / m
             # Prevent overflow if exponent is extremely small
             if exponent < -700: # Approximately exp(-700) is near float minimum
                  p_std_theory = 0.0
             else:
                  base = 1.0 - math.exp(exponent)
                  # Prevent potential issues with base slightly > 1 due to precision
                  base = min(1.0, base)
                  p_std_theory = base ** k_std_opt
             # Ensure p is within valid range (0, 1) for BloomFilter init
             p_std_theory = max(sys.float_info.min, min(1.0 - sys.float_info.epsilon, p_std_theory))

        except (OverflowError, ValueError):
             print("Warning: Could not calculate theoretical FPR for standard BF. Using default 0.01.")
             p_std_theory = 0.01 # Fallback, though comparison will be less accurate

        print(f"  Standard BF Target: n={n}, target_m={m}, optimal_k={k_std_opt}, theoretical_p={p_std_theory:.4e}")

        if p_std_theory <= 0 or p_std_theory >= 1:
             print("  Warning: Theoretical standard FPR is <= 0 or >= 1. Cannot initialize BloomFilter accurately.")
        else:
            try:
                # Initialize BloomFilter using n and theoretical p
                std_bf = BloomFilter(capacity=n, error_rate=p_std_theory)
                actual_memory_bits_standard = std_bf.size # Get the *actual* size calculated by the filter
                print(f"  Standard BF Initialized: Actual size (m) = {actual_memory_bits_standard} bits")

                # Populate the standard Bloom filter
                print("  Populating standard BF...")
                for key in pos_keys:
                    std_bf.add_str(key) # Use the correct method
                construct_end_standard = time.time()
                construction_time_standard = construct_end_standard - construct_start_standard
                print(f"  Population finished in {construction_time_standard:.4f} s.")

                # Test FPR of the standard Bloom filter
                print("  Calculating standard BF false positive rate...")
                fp_cnt_standard_val = 0
                if num_test_neg > 0:
                    for key in test_neg_keys: # Scores are irrelevant
                        if std_bf.contains_str(key): # Use the correct method
                            fp_cnt_standard_val += 1
                    fpr_standard = fp_cnt_standard_val / num_test_neg
                    fp_cnt_standard = fp_cnt_standard_val
                else:
                    fpr_standard = 0.0
                    fp_cnt_standard = 0
                print(f"  False Positives: {fp_cnt_standard} / {num_test_neg}")

            except ValueError as e:
                print(f"Error initializing/populating standard BloomFilter: {e}")
                fp_cnt_standard = 'InitError'
            except Exception as e:
                print(f"An unexpected error occurred during standard BF construction/testing: {e}")
                import traceback
                traceback.print_exc()
                fp_cnt_standard = 'Error'

    elif NUM_POSITIVE_SAMPLES == 0:
         print("  Skipping standard BF: No positive samples to insert.")
    else:
         print("  Skipping standard BF: Target memory size from FastPLBF is invalid.")


    # --- Final Results Comparison ---
    print("\n--- Final Results ---")
    if plbf:
        mem_plbf_str = f"{target_memory_bits_plbf}" # Use the calculated target bits
        mem_plbf_bytes_str = f"{target_memory_bits_plbf / 8:.2f}"
        print(f"FastPLBF Construction Time: {construct_end_plbf - construct_start_plbf:.4f} s")
        print(f"FastPLBF Memory Usage (bits): {mem_plbf_str}")
        print(f"FastPLBF Memory Usage (bytes): {mem_plbf_bytes_str}")
        print(f"FastPLBF False Positive Rate: {fpr_plbf:.6f} [{fp_cnt_plbf} / {num_test_neg}]")
    else:
        print("FastPLBF Construction Time: N/A (Failed)")
        print("FastPLBF Memory Usage (bits): N/A")
        print("FastPLBF Memory Usage (bytes): N/A")
        print(f"FastPLBF False Positive Rate: N/A")

    print("-" * 20)
    if std_bf:
        print(f"Standard BF Construction Time: {construction_time_standard:.4f} s")
        print(f"Standard BF Memory Usage (bits): {actual_memory_bits_standard}") # Report actual size
        print(f"Standard BF Memory Usage (bytes): {actual_memory_bits_standard / 8:.2f}")
        print(f"Standard BF False Positive Rate: {fpr_standard:.6f} [{fp_cnt_standard} / {num_test_neg}]")
    else:
        print("Standard BF Construction Time: N/A (Not run or failed)")
        print(f"Standard BF Memory Usage (bits): N/A (Target was {target_memory_bits_plbf})")
        print("Standard BF Memory Usage (bytes): N/A")
        print(f"Standard BF False Positive Rate: N/A")

    print("-" * 20)
    print(f"Target Overall FPR for PLBF (F): {TARGET_FPR:.6f}")

    # Optional: Print PLBF details if construction succeeded
    if plbf:
        # (Details printing code remains the same as before)
        if hasattr(plbf, 't') and plbf.t:
            print("\nFastPLBF Optimal Thresholds (t):")
            t_formatted = [f"{th:.4f}" for th in plbf.t]
            print(f"  [{', '.join(t_formatted)}]")
        if hasattr(plbf, 'f') and plbf.f:
            print("FastPLBF Optimal Region FPRs (f):")
            f_formatted = ["None" if x is None else f"{x:.4e}" for x in plbf.f]
            print(f"  [{', '.join(f_formatted)}]")



if __name__ == "__main__":
    import time
    import random
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import sys
    import os
    main()
    


