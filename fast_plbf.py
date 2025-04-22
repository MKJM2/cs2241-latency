import argparse
import math
import time
import bisect
from typing import (
    Any,
    Callable,
    Final,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Dict,
    Sequence,
)
import pandas as pd
from sklearn.model_selection import train_test_split
from bloom_filter import BloomFilter


# Type variables and aliases
KeyType = TypeVar("KeyType")
# Define a specific type for the predictor function
Predictor = Callable[[KeyType], float]

# Constants
EPS: Final[float] = 1e-8
INF: Final[float] = float("inf")


class prList:
    """
    Represents a discrete probability distribution over score segments
    defined by thresholds. Calculates segment probabilities and
    cumulative probabilities from a list of scores.
    """

    def __init__(
        self, scores: Sequence[float], thre_list: Sequence[float]
    ) -> None:
        """
        Initializes the probability list.

        Args:
            scores (Sequence[float]): A sequence of scores (between 0 and 1).
            thre_list (Sequence[float]): Thresholds dividing scores into segments.
                                         Must start with 0 and end with 1, strictly increasing.
        """
        assert thre_list[0] == 0.0, "Threshold list must start with 0.0"
        assert thre_list[-1] == 1.0, "Threshold list must end with 1.0"
        assert all(
            thre_list[i] < thre_list[i + 1] for i in range(len(thre_list) - 1)
        ), "Thresholds must be strictly increasing"

        self.thre_list: Final[List[float]] = list(thre_list)
        self.N: Final[int] = len(thre_list) - 1  # Number of segments

        cnt: List[int] = [0] * (self.N + 1)  # 1-based index for segments
        for sc in scores:
            assert 0.0 <= sc <= 1.0, f"Score out of bounds [0, 1]: {sc}"

            # Find segment index: segment `i` corresponds to (t_{i-1}, t_i]
            # Scores equal to a threshold fall into the segment *starting* at that threshold.
            idx: int = bisect.bisect_right(self.thre_list, sc)

            # Clamp index to be within [1, N]
            # bisect_right ensures score > t[idx-1] and score <= t[idx]
            # If score == 0.0, bisect_right gives 1 (correct).
            # If score == 1.0, bisect_right gives N+1, needs clamping.
            idx = max(1, min(idx, self.N))
            cnt[idx] += 1

        total: int = len(scores)
        self.pr: List[float] = [0.0] * (self.N + 1)
        self.accPr: List[float] = [0.0] * (self.N + 1)
        if total > 0:
            for i in range(1, self.N + 1):
                self.pr[i] = cnt[i] / total
                self.accPr[i] = self.accPr[i - 1] + self.pr[i]
            # Allow for small floating point inaccuracies in sum
            assert abs(self.accPr[self.N] - 1.0) < EPS, (
                f"Accumulated probability is {self.accPr[self.N]}, expected ~1.0"
            )

    def acc_range_idx(self, l_idx: int, r_idx: int) -> float:
        """
        Calculates the sum of probabilities for segments from index l_idx to r_idx (inclusive).

        Args:
            l_idx (int): Starting segment index \in {1 ... N}.
            r_idx (int): Ending segment index \in {1 ... N}.

        Returns:
            float: Sum of self.pr[l_idx...r_idx].
        """
        assert 1 <= l_idx <= self.N, f"l_idx out of bounds: {l_idx}"
        assert 1 <= r_idx <= self.N, f"r_idx out of bounds: {r_idx}"
        assert l_idx <= r_idx, f"l_idx must be <= r_idx ({l_idx}, {r_idx})"
        # accPr[r_idx] = sum(pr[1]...pr[r_idx])
        # accPr[l_idx - 1] = sum(pr[1]...pr[l_idx - 1])
        return self.accPr[r_idx] - self.accPr[l_idx - 1]

    def acc_range(self, score_l: float, score_r: float) -> float:
        """
        Calculates accumulated probability in the score range (score_l, score_r].

        Args:
            score_l (float): Left boundary (exclusive) \in [0, 1].
            score_r (float): Right boundary (inclusive) \in [0, 1].

        Returns:
            float: Accumulated probability Pr(score_l < score <= score_r).
        """
        assert 0.0 <= score_l <= 1.0
        assert 0.0 <= score_r <= 1.0
        assert score_l <= score_r, "score_l must be <= score_r"

        # Find segment indices corresponding to scores
        # We want segments *strictly greater* than score_l up to score_r
        # Segment `i` corresponds to (t_{i-1}, t_i]
        # Index `l` such that t_{l-1} < score_l <= t_l
        # Index `r` such that t_{r-1} < score_r <= t_r
        idx_l: int = bisect.bisect_right(self.thre_list, score_l)
        idx_r: int = bisect.bisect_right(self.thre_list, score_r)

        # We need sum from segment idx_l up to segment idx_r
        # Clamp indices to valid segment range [1, N]
        start_seg_idx = max(1, idx_l)
        end_seg_idx = max(0, min(idx_r -1, self.N)) # -1 because bisect_right gives insertion point

        if start_seg_idx > end_seg_idx:
            return 0.0  # Range contains no full segments or is invalid

        # Sum probabilities from segment start_seg_idx to end_seg_idx
        return self.accPr[end_seg_idx] - self.accPr[start_seg_idx - 1]


def matrix_problem_on_monotone_matrix(
    f: Callable[[int, int], float], n: int, m: int
) -> List[Optional[int]]:
    """
    Solves the rows maxima problem for a totally monotone n x m matrix B.
    Finds the smallest column index j for each row i such that B[i, j] is maximal.
    Uses the SMAWK algorithm principle (recursive divide and conquer). Assumes
    1-based indexing for the matrix B via function f.

    Args:
        f (Callable[[int, int], float]): Function returning B[i, j] (1-based i, j).
        n (int): Number of rows.
        m (int): Number of columns.

    Returns:
        List[Optional[int]]: 1-indexed list `a` where `a[i]` is the smallest
                             column index `j` maximizing `f(i, j)`. `a[0]` is None.
    """
    a: List[Optional[int]] = [None] * (n + 1)  # 1-based result array

    def calc_j(row_idx: int, col_start: int, col_end: int) -> int:
        """Finds the best column index for a given row in a specified range."""
        max_val: float = -INF
        argmax_col: int = col_start
        for col_idx in range(col_start, col_end + 1):
            val: float = f(row_idx, col_idx)
            # Use > for smallest index in case of ties
            if val > max_val:
                max_val = val
                argmax_col = col_idx
        return argmax_col

    def rec_solve(
        row_start: int, row_end: int, col_start: int, col_end: int
    ) -> None:
        """Recursive step of the algorithm."""
        if row_start > row_end:
            return

        mid_row: int = (row_start + row_end) // 2
        best_col_for_mid: int = calc_j(mid_row, col_start, col_end)
        a[mid_row] = best_col_for_mid

        # Recurse on the top-left quadrant (rows < mid, cols <= best_col)
        rec_solve(row_start, mid_row - 1, col_start, best_col_for_mid)
        # Recurse on the bottom-right quadrant (rows > mid, cols >= best_col)
        rec_solve(mid_row + 1, row_end, best_col_for_mid, col_end)

    rec_solve(1, n, 1, m)
    return a


def calc_DPKL(
    g: prList, h: prList, k: int, j_max: Optional[int] = None
) -> Tuple[List[List[float]], List[List[Optional[int]]]]:
    """
    Calculates DPKL table using standard dynamic programming O(N^2 * k).
    DPKL[n][q] = max_{1 <= i <= n} ( DPKL[i-1][q-1] + dkl(i, n) )
    where dkl(i, n) is the KL divergence contribution for segment [i, n].

    Args:
        g (prList): Key probability distribution over segments.
        h (prList): Non-key probability distribution over segments.
        k (int): Number of regions (partitions).
        j_max (Optional[int]): Optional upper bound for segments considered (default: N).

    Returns:
        Tuple[List[List[float]], List[List[Optional[int]]]]:
            DPKL table (max DKL sum), DPPredecessor table (optimal previous index).
            Both tables are (N+1)x(k+1) or (j_max+1)x(k+1), 0-indexed.
    """
    N: int = g.N
    if j_max is None:
        j_max = N
    assert h.N == N, "g and h must have the same number of segments (N)"
    assert 1 <= k <= N, "Number of regions k must be between 1 and N"
    assert 1 <= j_max <= N, "j_max must be between 1 and N"

    # DPKL[n][q]: max DKL sum using q regions up to segment n
    # DPPre[n][q]: the end index (i-1) of the (q-1)-th region for the optimal solution ending at n with q regions.
    DPKL: List[List[float]] = [
        [-INF] * (k + 1) for _ in range(j_max + 1)
    ]
    DPPre: List[List[Optional[int]]] = [
        [None] * (k + 1) for _ in range(j_max + 1)
    ]
    DPKL[0][0] = 0.0  # Base case: 0 regions, 0 segments -> 0 DKL

    for q in range(1, k + 1):  # Iterate through number of regions
        for n in range(1, j_max + 1):  # Iterate through ending segment index
            max_dpkl_val: float = -INF
            best_predecessor: Optional[int] = None
            # Iterate through possible start segments `i` for the q-th region [i, n]
            for i in range(1, n + 1):
                # Check if the previous state DPKL[i-1][q-1] is reachable
                if DPKL[i - 1][q - 1] == -INF:
                    continue

                # Calculate KL divergence contribution for segments i to n
                Pos: float = g.acc_range_idx(i, n)
                Neg: float = h.acc_range_idx(i, n)

                dkl_term: float
                if Neg == 0.0:
                    # If no non-keys, KL div is infinite if keys exist, 0 otherwise
                    if Pos > EPS:
                        continue  # Effectively -INF DPKL, skip this split
                    else:
                        dkl_term = 0.0
                elif Pos == 0.0:
                    # If no keys, KL div contribution is 0
                    dkl_term = 0.0
                else:
                    # Standard KL divergence term: Pos * log(Pos / Neg)
                    # Use natural log as in the reference code
                    dkl_term = Pos * math.log(Pos / Neg)

                current_sum: float = DPKL[i - 1][q - 1] + dkl_term

                if current_sum > max_dpkl_val:
                    max_dpkl_val = current_sum
                    # Store the end index of the previous region (q-1)
                    best_predecessor = i - 1

            DPKL[n][q] = max_dpkl_val
            DPPre[n][q] = best_predecessor

    return DPKL, DPPre


def fast_calc_DPKL(
    g: prList, h: prList, k: int
) -> Tuple[List[List[float]], List[List[Optional[int]]]]:
    """
    Calculates DPKL table using the faster O(N * k * logN) approach
    leveraging the total monotonicity property of the underlying matrix problem.

    Args:
        g (prList): Key probability distribution.
        h (prList): Non-key probability distribution.
        k (int): Number of regions.

    Returns:
        Tuple[List[List[float]], List[List[Optional[int]]]]: DPKL table, DPPredecessor table.
    """
    N: int = g.N
    assert h.N == N, "g and h must have the same number of segments (N)"
    assert 1 <= k <= N

    DPKL: List[List[float]] = [[-INF] * (k + 1) for _ in range(N + 1)]
    DPPre: List[List[Optional[int]]] = [
        [None] * (k + 1) for _ in range(N + 1)
    ]
    DPKL[0][0] = 0.0  # Base case

    for q in range(1, k + 1):  # Iterate through number of regions
        # Define the function A(p, i) for the monotone matrix problem
        # A[p, i] corresponds to the DPKL value if the q-th region starts at i and ends at p
        # A[p, i] = DPKL[i-1][q-1] + dkl(i, p)
        # Note: matrix_problem_on_monotone_matrix uses 1-based indexing for p and i
        def func_A(p: int, i: int) -> float:
            """
            Calculates the potential DPKL value for row p, column i (1-based).
            p: end segment index (1 to N)
            i: start segment index (1 to N)
            """
            # The q-th region must start at or before it ends (i <= p)
            if i > p:
                return -INF
            # The previous state DPKL[i-1][q-1] must be valid (reachable)
            if DPKL[i - 1][q - 1] == -INF:
                return -INF

            Pos: float = g.acc_range_idx(i, p)
            Neg: float = h.acc_range_idx(i, p)

            dkl_term: float
            if Neg == 0.0:
                if Pos > EPS:
                    return -INF  # Infinite KL divergence
                else:
                    dkl_term = 0.0
            elif Pos == 0.0:
                dkl_term = 0.0
            else:
                dkl_term = Pos * math.log(Pos / Neg)

            return DPKL[i - 1][q - 1] + dkl_term

        # Solve the row maxima problem for the implicitly defined N x N matrix A
        # max_args[p] will give the optimal starting segment index `i` (1-based)
        # for the q-th region ending at segment p (1-based).
        max_args: List[Optional[int]] = matrix_problem_on_monotone_matrix(
            func_A, N, N
        )

        # Update DPKL and DPPre tables using the optimal starting points found
        for n in range(1, N + 1):  # n is the ending segment index (1-based)
            optimal_start_idx: Optional[int] = max_args[n]  # Optimal 'i'
            if optimal_start_idx is None:
                # This row (ending segment n) might be unreachable
                continue

            # Recalculate the value using the optimal start index to store in DPKL
            dpkl_value: float = func_A(n, optimal_start_idx)
            DPKL[n][q] = dpkl_value
            # Store the end index of the *previous* region (optimal_start_idx - 1)
            DPPre[n][q] = optimal_start_idx - 1

    return DPKL, DPPre


def MaxDivDP(
    g: prList, h: prList, N: int, k: int
) -> Tuple[List[List[float]], List[List[Optional[int]]]]:
    """
    Wrapper function to calculate DPKL using the standard DP method.
    (Provided for compatibility with the original PLBF class structure).

    Args:
        g (prList): Key density.
        h (prList): Non-key density.
        N (int): Number of segments (should match g.N).
        k (int): Number of regions.

    Returns:
        Tuple[list[list[float]], list[list[int]]]: DPKL, DPPre tables.
    """
    assert g.N == N, "N parameter does not match g.N"
    return calc_DPKL(g, h, k)


def fastMaxDivDP(
    g: prList, h: prList, N: int, k: int
) -> Tuple[List[List[float]], List[List[Optional[int]]]]:
    """
    Wrapper function to calculate DPKL using the fast (monotone matrix) method.

    Args:
        g (prList): Key density.
        h (prList): Non-key density.
        N (int): Number of segments (should match g.N).
        k (int): Number of regions.

    Returns:
        Tuple[list[list[float]], list[list[int]]]: DPKL, DPPre tables.
    """
    assert g.N == N, "N parameter does not match g.N"
    return fast_calc_DPKL(g, h, k)


def ThresMaxDiv(
    DPPre: List[List[Optional[int]]],
    end_segment_plus1: int,
    k: int,
    segment_thre_list: Sequence[float],
) -> Optional[List[float]]:
    """
    Reconstructs the optimal threshold boundaries `t` by backtracking through
    the DPPre table. Finds thresholds for `k` regions ending at segment `end_segment_plus1 - 1`.

    Args:
        DPPre (List[List[Optional[int]]]): DP Predecessor table (0-indexed).
        end_segment_plus1 (int): The index *after* the last segment included
                                 in the k regions (e.g., if regions cover 1..N, this is N+1).
        k (int): Number of regions.
        segment_thre_list (Sequence[float]): The list of segment thresholds (0..N).

    Returns:
        Optional[List[float]]: Optimal threshold boundaries `t` (length k+1),
                               or None if no valid path exists. `t` includes 0.0 and 1.0.
    """
    end_segment_idx: int = end_segment_plus1 - 1  # Last segment index included

    # Check if the starting state for backtracking is valid
    if (
        end_segment_idx < 0
        or k <= 0
        or end_segment_idx >= len(DPPre)
        or k >= len(DPPre[0])
        or DPPre[end_segment_idx][k] is None
    ):
        return None  # Invalid starting state or no path found

    # Backtrack to find the boundaries
    # `t` will store [t_0, t_1, ..., t_k] where t_0=0, t_k=1
    reversed_t: List[float] = [1.0]  # Last boundary is always 1.0

    current_end_idx: int = end_segment_idx
    # Add boundary t_{k-1} which is the threshold at the end of segment current_end_idx
    # Note: segment_thre_list[i] is the threshold *after* segment i.
    # So, segment_thre_list[current_end_idx] is the correct boundary.
    reversed_t.append(segment_thre_list[current_end_idx])

    # Trace back from region k down to region 1
    for reg_idx in reversed(range(1, k)):  # reg_idx goes from k-1 down to 1
        # Find where the previous region (reg_idx) ended.
        # This is stored in DPPre[current_end_idx][reg_idx + 1]
        prev_end_idx: Optional[int] = DPPre[current_end_idx][reg_idx + 1]
        if prev_end_idx is None:
            return None  # Invalid path during backtracking

        # Add the threshold at the end of the previous region
        reversed_t.append(segment_thre_list[prev_end_idx])
        current_end_idx = prev_end_idx

    # After the loop, current_end_idx is the end index of region 0 (which is index 0).
    # The first boundary t_0 should be 0.0.
    # Check if the path correctly leads back to the start DPKL[0][0]
    # The predecessor for the first region ending at current_end_idx should be 0.
    if DPPre[current_end_idx][1] != 0:
         # This might indicate an issue if the first region didn't start correctly.
         # However, the DP logic should handle this. If we reached here, path is likely valid.
         pass

    reversed_t.append(0.0)  # Add the first boundary t_0 = 0.0
    t: List[float] = list(reversed(reversed_t))

    # Expected length is k+1 boundaries for k regions.
    if len(t) != k + 1:
        # This could happen if the optimal solution effectively uses fewer than k regions,
        # or if backtracking logic has an issue. Return None for safety.
        print(
            f"Warning: Unexpected threshold list length {len(t)} for k={k}."
            f" Path: {t}"
        )
        return None

    # Final check for monotonicity (should hold if DPPre is correct)
    assert all(
        t[i] <= t[i + 1] for i in range(len(t) - 1)
    ), "Thresholds not monotonic after backtracking"

    return t


def OptimalFPR(
    g: prList, h: prList, t: List[float], F: float, k: int
) -> List[Optional[float]]:
    """
    Calculates the optimal False Positive Rates (FPRs) `f` for each region
    defined by thresholds `t`, given a target overall FPR `F`.
    Uses the method from the paper (Equation 7 / Algorithm 3).

    Args:
        g (prList): Key probability distribution.
        h (prList): Non-key probability distribution.
        t (List[float]): Threshold boundaries of each region (length k+1).
                           t = [t_0, t_1, ..., t_k] where t_0=0, t_k=1.
        F (float): Target overall FPR (must be > 0 and < 1).
        k (int): Number of regions.

    Returns:
        List[Optional[float]]: Optimal FPRs `f` for each region (1-indexed, f[0]=None).
                               Length k+1. Returns None for regions where FPR is undefined
                               or calculation fails.
    """
    assert len(t) == k + 1, "Threshold list length must be k+1"
    assert 0 < F < 1, "Target FPR F must be between 0 and 1"

    # Calculate positive (key) and negative (non-key) probabilities per region
    # Region i corresponds to interval (t[i-1], t[i]]
    pos_pr_list: List[float] = [
        g.acc_range(t[i - 1], t[i]) for i in range(1, k + 1)
    ]
    neg_pr_list: List[float] = [
        h.acc_range(t[i - 1], t[i]) for i in range(1, k + 1)
    ]

    # Check if probabilities sum correctly (within tolerance)
    assert abs(sum(pos_pr_list) - 1.0) < EPS, (
        f"Sum of pos_pr != 1: {sum(pos_pr_list)}"
    )
    # Sum of neg_pr might not be 1 if h represents a subset (e.g., training negatives)

    # Initialize validity: a region is invalid if it has zero non-key probability
    # (cannot contribute to FPR calculation meaningfully) or if forced to f=1.
    # Use 0-based indexing internally for easier list manipulation.
    valid_list: List[bool] = [True] * k
    for i in range(k):
        if neg_pr_list[i] < EPS:  # Treat near-zero as zero
            valid_list[i] = False

    opt_fpr_list: List[float] = [0.0] * k  # 0-indexed temporary list

    # Iteratively determine optimal FPRs, handling cases where calculated FPR > 1
    while True:
        valid_pos_pr_sum: float = 0.0
        valid_neg_pr_sum: float = 0.0
        # Sum of neg_pr for regions forced to FPR=1 (or originally neg_pr=0)
        invalid_neg_pr_sum: float = 0.0

        for i in range(k):
            if valid_list[i]:
                valid_pos_pr_sum += pos_pr_list[i]
                valid_neg_pr_sum += neg_pr_list[i]
            else:
                # If invalid (neg_pr=0 or forced f=1), its contribution to overall F
                # is neg_pr[i] * 1.0 (if f=1) or 0 (if neg_pr=0).
                # This sum is used to adjust the target F for remaining valid regions.
                invalid_neg_pr_sum += neg_pr_list[i] # Adds 0 if neg_pr was 0

        # Check for edge cases where calculation is impossible or trivial
        if valid_neg_pr_sum < EPS:
            # All remaining regions have neg_pr=0. Cannot satisfy target F unless
            # F is already met by the invalid regions. Set remaining FPRs to 0.
            for i in range(k):
                if valid_list[i]:
                    opt_fpr_list[i] = 0.0
                else:
                    # Keep forced/original invalid as 1.0
                    opt_fpr_list[i] = 1.0
            break  # Exit loop

        if valid_pos_pr_sum < EPS:
            # All remaining valid regions have pos_pr=0. Set their FPR to 0.
            # This might happen if F is very low.
            for i in range(k):
                if valid_list[i]:
                    opt_fpr_list[i] = 0.0
                else:
                    opt_fpr_list[i] = 1.0
            break # Exit loop

        # Calculate the proportionality constant C based on adjusted target FPR
        # F = sum(f_i * neg_pr_i) = sum_{valid} f_i*neg_pr_i + sum_{invalid} 1*neg_pr_i
        # F - invalid_neg_pr_sum = sum_{valid} f_i*neg_pr_i
        # Let f_i = C * pos_pr_i / neg_pr_i for valid i
        # F - invalid_neg_pr_sum = sum_{valid} (C * pos_pr_i / neg_pr_i) * neg_pr_i
        # F - invalid_neg_pr_sum = C * sum_{valid} pos_pr_i
        # C = (F - invalid_neg_pr_sum) / sum_{valid} pos_pr_i
        adjusted_F: float = F - invalid_neg_pr_sum
        if adjusted_F < 0:
             # Target F is unachievable even with f=0 for all valid regions.
             # Set all valid region FPRs to 0 and proceed.
             print(f"Warning: Target FPR F={F} is too low to be achieved. "
                   f"Setting remaining valid FPRs to 0.")
             for i in range(k):
                 if valid_list[i]:
                     opt_fpr_list[i] = 0.0
                 else:
                     opt_fpr_list[i] = 1.0
             break # Exit loop

        constant_C: float = adjusted_F / valid_pos_pr_sum

        # Calculate optimal FPRs for currently valid regions
        changed_in_iteration: bool = False
        for i in range(k):
            if valid_list[i]:
                # Check for division by zero (should be caught by valid_neg_pr_sum check)
                if neg_pr_list[i] < EPS:
                    # This case should technically be handled by initial valid_list setting
                    # If reached, force to 1.0 and invalidate.
                    opt_fpr_list[i] = 1.0
                    valid_list[i] = False
                    changed_in_iteration = True
                    continue

                fpr_i: float = constant_C * pos_pr_list[i] / neg_pr_list[i]

                # Check if calculated FPR needs clamping
                if fpr_i >= 1.0 - EPS:  # Clamp FPR >= 1 to 1
                    opt_fpr_list[i] = 1.0
                    valid_list[i] = False  # Mark as invalid for next iteration
                    changed_in_iteration = True
                elif fpr_i < 0.0: # Should not happen if adjusted_F >= 0
                    # Clamp negative FPR to 0
                    opt_fpr_list[i] = 0.0
                    # Keep it valid for now, maybe other regions compensate.
                else:
                    opt_fpr_list[i] = fpr_i
            else:
                # Keep previously invalidated regions at FPR = 1.0
                opt_fpr_list[i] = 1.0

        # If no FPRs were capped at 1.0 in this iteration, the solution is stable
        if not changed_in_iteration:
            break

    # Final check: Calculate the achieved FPR with the computed opt_fpr_list
    achieved_F: float = sum(
        opt_fpr_list[i] * neg_pr_list[i] for i in range(k)
    )
    if abs(achieved_F - F) > max(1e-5, F * 0.1): # Allow some tolerance
        print(
            f"Warning: OptimalFPR calculation result deviates significantly "
            f"from target F={F}. Achieved F={achieved_F:.6f}. "
            f"This might happen if F is unachievable or due to edge cases."
        )

    # Convert f to 1-index with None at index 0 for consistency with API
    final_fpr_list: List[Optional[float]] = [None] + opt_fpr_list
    assert len(final_fpr_list) == k + 1
    return final_fpr_list


def SpaceUsed(
    g: prList,
    h: prList, # h is unused but kept for consistent signature
    t: List[float],
    f: List[Optional[float]],
    n: int,
) -> float:
    """
    Calculates the estimated total space usage (in bits) for the backup Bloom filters,
    based on the standard Bloom filter space formula.

    Formula: Space = sum_i (n * pos_pr_i * log2(1/f_i)) / ln(2)
    where n * pos_pr_i is the number of keys in region i, and f_i is the FPR.
    This corresponds to m/n_i = -log2(f_i) / ln(2) bits per element.

    Args:
        g (prList): Key probability distribution.
        h (prList): Non-key probability distribution (unused).
        t (List[float]): Threshold boundaries (length k+1).
        f (List[Optional[float]]): FPRs for each region (1-indexed, f[0]=None).
        n (int): Total number of positive keys inserted.

    Returns:
        float: Estimated total space usage in bits. Returns INF if any region
               with keys requires f=0.
    """
    k: int = len(t) - 1
    assert len(f) == k + 1, "FPR list length must be k+1"

    total_space_bits: float = 0.0
    ln2: float = math.log(2)  # Constant ln(2) approx 0.693

    for i in range(1, k + 1):  # Iterate through regions 1 to k
        pos_pr: float = g.acc_range(t[i - 1], t[i])
        num_keys_in_region: float = pos_pr * n

        if num_keys_in_region < EPS:  # Skip regions with effectively zero keys
            continue

        fpr_i: Optional[float] = f[i]

        # Handle edge cases for FPR
        if fpr_i is None:
            # This shouldn't happen for regions with keys unless OptimalFPR failed.
            # Assume no filter / zero space contribution? Or raise error?
            print(f"Warning: FPR f[{i}] is None for region with keys.")
            continue
        elif fpr_i <= EPS:
            # FPR = 0 implies infinite space (perfect filter).
            # If keys exist here, space is theoretically infinite.
            # In practice, means storing keys directly or using near-zero FPR.
            # Return INF to signal this theoretical requirement.
            return INF
        elif fpr_i >= 1.0 - EPS:
            # FPR = 1 implies no filter needed (accepts everything). Space = 0.
            space_i: float = 0.0
        else:
            # Standard space formula for a Bloom filter:
            # bits_per_element = -log2(fpr_i) / ln(2)  (approx 1.44 * log2(1/fpr_i))
            # total_bits = num_elements * bits_per_element
            bits_per_key: float = -math.log2(fpr_i) / ln2
            space_i = num_keys_in_region * bits_per_key

        total_space_bits += space_i

    return total_space_bits


# --- PLBF Base Class ---
class PLBF(Generic[KeyType]):
    """
    Partitioned Learned Bloom Filter (Base Implementation).
    Uses standard DP for threshold calculation. Accepts a predictor function.
    """

    def __init__(
        self,
        predictor: Predictor[KeyType],
        pos_keys: Sequence[KeyType],
        neg_keys: Sequence[KeyType], # Used for training h distribution
        F: float,
        N: int,
        k: int,
    ) -> None:
        """
        Initializes the PLBF.

        Args:
            predictor (Predictor[KeyType]): Function mapping a key to a score [0, 1].
            pos_keys (Sequence[KeyType]): Positive keys to insert into the filter.
            neg_keys (Sequence[KeyType]): Negative keys used for learning the
                                          non-key score distribution `h`.
            F (float): Target overall FPR (0 < F < 1).
            N (int): Number of initial segments for score discretization.
            k (int): Number of final regions (partitions) (1 <= k <= N).
        """
        # --- Input Validation ---
        assert isinstance(pos_keys, Sequence)
        assert isinstance(neg_keys, Sequence)
        assert callable(predictor)
        assert isinstance(F, float) and 0 < F < 1
        assert isinstance(N, int) and N > 0
        assert isinstance(k, int) and 1 <= k <= N

        # --- Store Core Parameters ---
        self._predictor: Final[Predictor[KeyType]] = predictor
        self.F: Final[float] = F
        self.N: Final[int] = N
        self.k: Final[int] = k
        self.n: Final[int] = len(pos_keys)  # Total number of positive keys

        # --- Step 1: Compute Scores using Predictor ---
        # Compute scores only once for efficiency
        print("Computing scores using predictor...")
        start_score_time = time.time()
        pos_scores: List[float] = [predictor(key) for key in pos_keys]
        neg_scores: List[float] = [predictor(key) for key in neg_keys]
        end_score_time = time.time()
        print(f"Score computation took {end_score_time - start_score_time:.2f}s")

        # --- Step 2: Divide scores into segments & build prLists ---
        print("Building segment distributions (prList)...")
        self.segment_thre_list: Final[List[float]] = [
            i / N for i in range(N + 1)
        ]
        self.g: prList = prList(pos_scores, self.segment_thre_list)
        self.h: prList = prList(neg_scores, self.segment_thre_list)

        # --- Step 3: Find optimal thresholds (t) and FPRs (f) ---
        # These will be set by _find_best_t_and_f
        self.t: List[float] = []
        self.f: List[Optional[float]] = []
        self.memory_usage_of_backup_bf: float = 0.0
        print("Finding optimal thresholds and FPRs...")
        start_fit_time = time.time()
        self._find_best_t_and_f() # Calls the appropriate DP method
        end_fit_time = time.time()
        print(f"Threshold/FPR optimization took {end_fit_time - start_fit_time:.2f}s")


        # --- Step 4: Build Bloom filters and insert keys ---
        # Filters are stored in self.bfs
        self.bfs: List[Optional[BloomFilter]] = []
        print("Building backup Bloom filters and inserting keys...")
        start_build_time = time.time()
        self._build_filters(pos_keys, pos_scores)
        end_build_time = time.time()
        print(f"Filter construction took {end_build_time - start_build_time:.2f}s")


    def _find_best_t_and_f(self) -> None:
        """Finds the best region thresholds (t) and FPRs (f) using standard DP."""
        # Calculate DPKL using standard O(N^2*k) DP
        DPKL, DPPre = MaxDivDP(self.g, self.h, self.N, self.k)

        minSpaceUsed: float = INF
        t_best: Optional[List[float]] = None
        f_best: Optional[List[Optional[float]]] = None

        # Iterate through possible ending segments (j-1) for the k-th region
        for j in range(self.k, self.N + 1):
            # Reconstruct thresholds `t` for k regions ending at segment j-1
            t_current: Optional[List[float]] = ThresMaxDiv(
                DPPre, j, self.k, self.segment_thre_list
            )
            if t_current is None:  # No valid partition found ending at j-1
                continue

            # Calculate optimal FPRs `f` for these thresholds `t`
            f_current: List[Optional[float]] = OptimalFPR(
                self.g, self.h, t_current, self.F, self.k
            )
            # OptimalFPR should ideally always return a valid list if t is valid.

            # Calculate space usage for this (t, f) combination
            currentSpaceUsed: float = SpaceUsed(
                self.g, self.h, t_current, f_current, self.n
            )

            # Update best if current space is lower
            if currentSpaceUsed < minSpaceUsed:
                minSpaceUsed = currentSpaceUsed
                t_best = t_current
                f_best = f_current

        if t_best is None or f_best is None:
            raise RuntimeError(
                "PLBF: Could not find a valid partition (t, f). "
                "Check input data or parameters (N, k, F)."
            )

        self.t = t_best
        self.f = f_best
        self.memory_usage_of_backup_bf = (
            minSpaceUsed if minSpaceUsed != INF else 0.0
        )

    def _get_region_idx(self, score: float) -> int:
        """Finds the region index (1 to k) for a given score based on thresholds t."""
        assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"
        assert hasattr(self, 't') and self.t, "Thresholds 't' not calculated yet."

        # Find i such that t[i-1] < score <= t[i]. Region index is i (1-based).
        # bisect_right finds insertion point `idx` where all elements <= score are to the left.
        # So, t[idx-1] < score <= t[idx].
        region_idx: int = bisect.bisect_right(self.t, score)

        # Clamp index to be within [1, k]
        region_idx = max(1, min(region_idx, self.k))
        return region_idx

    def _build_filters(
        self, pos_keys: Sequence[KeyType], pos_scores: List[float]
    ) -> None:
        """Creates Bloom filters and inserts positive keys."""
        assert hasattr(self, 't') and self.t, "Thresholds 't' not set."
        assert hasattr(self, 'f') and self.f, "FPRs 'f' not set."
        assert len(pos_keys) == len(pos_scores)

        # Count keys and collect keys falling into each region
        pos_cnt_list: List[int] = [0] * (self.k + 1)  # 1-based index
        keys_per_region: List[List[KeyType]] = [
            [] for _ in range(self.k + 1)
        ]

        for key, score in zip(pos_keys, pos_scores):
            region_idx: int = self._get_region_idx(score)
            pos_cnt_list[region_idx] += 1
            keys_per_region[region_idx].append(key)

        # Create backup Bloom filters for regions where f_i is between 0 and 1
        self.bfs = [None] * (self.k + 1)  # 1-based list of filters
        for i in range(1, self.k + 1):
            fpr_i: Optional[float] = self.f[i]
            count_i: int = pos_cnt_list[i]

            if count_i == 0:
                # No keys in this region, no filter needed.
                self.bfs[i] = None
                continue

            if fpr_i is None:
                print(f"Warning: FPR f[{i}] is None. Cannot create filter.")
                self.bfs[i] = None
            elif fpr_i <= EPS:
                # FPR is effectively 0. Requires perfect filtering.
                # A standard BF cannot achieve f=0. This implies either:
                # 1) No non-keys land here (neg_pr=0), so any key is positive.
                # 2) Storing keys directly is needed.
                # For simplicity, we won't create a BF. `contains` must handle f=0.
                # If neg_pr=0, `contains` should be accurate without BF.
                # If neg_pr>0, f=0 is theoretically impossible with BF.
                self.bfs[i] = None # Mark as needing special handling in contains
            elif fpr_i >= 1.0 - EPS:
                # FPR is effectively 1. No filter needed, always return True.
                self.bfs[i] = None
            else:
                # Standard case: Create and populate a Bloom filter
                try:
                    bf = BloomFilter(capacity=count_i, error_rate=fpr_i)
                    for key in keys_per_region[i]:
                        bf.add(key)
                    self.bfs[i] = bf
                except Exception as e:
                    print(f"Error creating BloomFilter for region {i}: {e}")
                    self.bfs[i] = None # Failed to create filter

    def contains(self, key: KeyType) -> bool:
        """
        Checks if a key might be present in the set represented by the PLBF.

        Args:
            key (KeyType): The key to check.

        Returns:
            bool: True if the key might be present (potential positive or false positive),
                  False if the key is definitely not present (true negative).
        """
        assert hasattr(self, 'bfs'), "Filters not initialized."
        score: float = self._predictor(key)
        assert 0.0 <= score <= 1.0, f"Predictor score out of bounds: {score}"

        region_idx: int = self._get_region_idx(score)
        fpr_i: Optional[float] = self.f[region_idx]
        bf: Optional[BloomFilter] = self.bfs[region_idx]

        if fpr_i is None:
            # Should not happen if initialization succeeded. Assume positive?
            print(f"Warning: FPR f[{region_idx}] is None during contains().")
            return True # Fail safe?
        elif fpr_i >= 1.0 - EPS:
            # Region has FPR=1. Always return True.
            return True
        elif fpr_i <= EPS:
            # Region has FPR=0. Should only contain positive keys.
            # If a BF exists (e.g., placeholder for testing), query it.
            # If no BF was created (because f=0 implies perfect separation or
            # direct storage), the result depends on whether *any* positives landed here.
            # A simple check: if bf exists, query it. Otherwise, assume False?
            # This assumes f=0 only occurs when neg_pr=0 for the region.
            # A more robust f=0 handling might involve storing keys if needed.
            # Let's return False if no BF exists for f=0 region.
            return bf is not None and (key in bf)
        else:
            # Standard case: Query the Bloom filter for the region.
            # If bf is None (e.g., creation failed or count was 0), key cannot be present.
            return bf is not None and (key in bf)


# --- FastPLBF Class ---
class FastPLBF(PLBF[KeyType]):
    """
    Partitioned Learned Bloom Filter (Fast Implementation).
    Uses faster DP (monotone matrix property) for threshold calculation.
    Inherits most methods from PLBF, overrides DP calculation.
    """

    # Override _find_best_t_and_f to use the faster DP calculation
    def _find_best_t_and_f(self) -> None:
        """Finds the best region thresholds (t) and FPRs (f) using fast DP."""
        # Calculate DPKL using fast O(N*k*logN) DP
        DPKL, DPPre = fastMaxDivDP(self.g, self.h, self.N, self.k)

        minSpaceUsed: float = INF
        t_best: Optional[List[float]] = None
        f_best: Optional[List[Optional[float]]] = None

        # Iterate through possible ending segments (j-1) for the k-th region.
        # This allows finding the optimal partition which might not use all N segments.
        for j in range(self.k, self.N + 1):
            t_current: Optional[List[float]] = ThresMaxDiv(
                DPPre, j, self.k, self.segment_thre_list
            )
            if t_current is None:
                continue

            f_current: List[Optional[float]] = OptimalFPR(
                self.g, self.h, t_current, self.F, self.k
            )
            currentSpaceUsed: float = SpaceUsed(
                self.g, self.h, t_current, f_current, self.n
            )

            if currentSpaceUsed < minSpaceUsed:
                minSpaceUsed = currentSpaceUsed
                t_best = t_current
                f_best = f_current

        if t_best is None or f_best is None:
            raise RuntimeError(
                "FastPLBF: Could not find a valid partition (t, f). "
                "Check input data or parameters (N, k, F)."
            )

        self.t = t_best
        self.f = f_best
        self.memory_usage_of_backup_bf = (
            minSpaceUsed if minSpaceUsed != INF else 0.0
        )


# --- Main Execution Block ---
def main() -> None:
    """Main function to parse arguments, load data, build filter, and test."""
    parser = argparse.ArgumentParser(
        description="Construct and test a Fast Partitioned Learned Bloom Filter."
    )
    parser.add_argument(
        "--data_path",
        action="store",
        dest="data_path",
        type=str,
        required=True,
        help="Path of the dataset CSV file (needs 'key', 'score', 'label' columns)",
    )
    parser.add_argument(
        "--N",
        action="store",
        dest="N",
        type=int,
        required=True,
        help="N: The number of initial segments for discretization",
    )
    parser.add_argument(
        "--k",
        action="store",
        dest="k",
        type=int,
        required=True,
        help="k: The number of final regions (partitions)",
    )
    parser.add_argument(
        "--F",
        action="store",
        dest="F",
        type=float,
        required=True,
        help="F: The target overall false positive rate",
    )
    parser.add_argument(
        "--test_split",
        action="store",
        dest="test_split",
        type=float,
        default=0.7,
        help="Fraction of negative samples to use for testing (default: 0.7)",
    )
    parser.add_argument(
        "--seed",
        action="store",
        dest="seed",
        type=int,
        default=0,
        help="Random seed for train/test split (default: 0)",
    )
    parser.add_argument(
        "--use_fast_dp",
        action="store_true",
        default=True,
        help="Use the FastPLBF (O(N^2k) DP) implementation as opposed to standard PLBF (O(NklogN) DP). Default is FastPLBF.",
    )

    results: argparse.Namespace = parser.parse_args()

    DATA_PATH: str = results.data_path
    N_param: int = results.N
    k_param: int = results.k
    F_param: float = results.F
    TEST_SPLIT: float = results.test_split
    SEED: int = results.seed
    USE_FAST: bool = results.use_fast_dp

    # --- Data Loading and Preparation ---
    print(f"Loading data from: {DATA_PATH}")
    try:
        data: pd.DataFrame = pd.read_csv(DATA_PATH)
        # Ensure required columns exist
        required_cols: set[str] = {'key', 'score', 'label'}
        if not required_cols.issubset(data.columns):
            raise ValueError(
                f"CSV must contain columns: {', '.join(required_cols)}"
            )
        # Ensure scores are numeric and within [0, 1]
        data['score'] = pd.to_numeric(data['score'], errors='coerce')
        if data['score'].isnull().any():
            raise ValueError("Non-numeric values found in 'score' column.")
        if not ((data['score'] >= 0) & (data['score'] <= 1)).all():
            raise ValueError("Scores must be between 0 and 1.")
        # Ensure labels are numeric (typically 1 for positive, others negative)
        data['label'] = pd.to_numeric(data['label'], errors='coerce')
        if data['label'].isnull().any():
            raise ValueError("Non-numeric values found in 'label' column.")

    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        exit(1)
    except ValueError as e:
        print(f"Error loading or validating data: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        exit(1)

    # Separate positive and negative samples
    positive_sample: pd.DataFrame = data.loc[(data["label"] == 1)]
    negative_sample: pd.DataFrame = data.loc[(data["label"] != 1)]

    if len(positive_sample) == 0:
        print("Error: No positive samples (label=1) found in the data.")
        exit(1)
    if len(negative_sample) == 0:
        print(
            "Warning: No negative samples (label!=1) found."
            " FPR calculation might be trivial or inaccurate."
        )
        # Create dummy negative sample for code structure if needed
        train_negative: pd.DataFrame = negative_sample
        test_negative: pd.DataFrame = negative_sample
    else:
        # Split negatives into train (for learning h) and test (for evaluation)
        train_negative, test_negative = train_test_split(
            negative_sample, test_size=TEST_SPLIT, random_state=SEED
        )

    # Prepare key lists
    pos_keys: List[Any] = list(positive_sample["key"])
    # Use training negatives for building the filter's h distribution
    train_neg_keys: List[Any] = list(train_negative["key"])
    # Use testing negatives for evaluating the actual FPR
    test_neg_keys: List[Any] = list(test_negative["key"])

    # --- Create Predictor Function ---
    # In a real scenario, this would be your trained model or scoring function.
    # Here, we simulate it by looking up pre-computed scores from the CSV.
    score_lookup: Dict[Any, float] = dict(zip(data['key'], data['score']))
    def csv_predictor(key: Any) -> float:
        """Simulated predictor looking up scores from the loaded CSV."""
        return score_lookup.get(key, 0.0) # Default score if key not found?

    predictor_func: Predictor[Any] = csv_predictor

    print(f"Positive samples: {len(pos_keys)}")
    print(f"Training negative samples: {len(train_neg_keys)}")
    print(f"Testing negative samples: {len(test_neg_keys)}")
    print(f"Parameters: N={N_param}, k={k_param}, F={F_param}")
    print(f"Using {'Fast DP' if USE_FAST else 'Standard DP'} implementation.")

    # --- Construct PLBF ---
    print("Constructing PLBF...")
    total_construct_start: float = time.time()
    plbf_instance: PLBF[Any]
    try:
        if USE_FAST:
            plbf_instance = FastPLBF(
                predictor=predictor_func,
                pos_keys=pos_keys,
                neg_keys=train_neg_keys, # Use train negatives to learn h
                F=F_param,
                N=N_param,
                k=k_param,
            )
        else:
            plbf_instance = PLBF(
                predictor=predictor_func,
                pos_keys=pos_keys,
                neg_keys=train_neg_keys,
                F=F_param,
                N=N_param,
                k=k_param,
            )
    except RuntimeError as e:
        print(f"Error during PLBF construction: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during construction: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    total_construct_end: float = time.time()
    print(
        "Construction finished in "
        f"{total_construct_end - total_construct_start:.4f} seconds (total)."
    )

    # --- Verification: No False Negatives ---
    print("Verifying no false negatives...")
    fn_cnt: int = 0
    for key in pos_keys:
        if not plbf_instance.contains(key):
            # print(f"False Negative detected: Key={key}, Score={predictor_func(key)}")
            fn_cnt += 1
    if fn_cnt == 0:
        print("Verification successful: No false negatives found.")
    else:
        print(
            f"Error: {fn_cnt} false negatives detected out of {len(pos_keys)}."
        )

    # --- Testing: False Positive Rate ---
    print("Calculating false positive rate on test set...")
    fp_cnt: int = 0
    if len(test_neg_keys) > 0:
        for key in test_neg_keys:
            if plbf_instance.contains(key):
                fp_cnt += 1
        measured_fpr: float = fp_cnt / len(test_neg_keys)
    else:
        measured_fpr = 0.0  # No test negatives to measure FPR

    # --- Output Results ---
    print("\n--- Results ---")
    print(f"Target Overall FPR (F): {F_param:.6f}")
    print(
        f"Measured FPR on test set: {measured_fpr:.6f} "
        f"[{fp_cnt} / {len(test_neg_keys)}]"
    )
    print(
        "Total Construction Time: "
        f"{total_construct_end - total_construct_start:.4f} seconds"
    )
    # Convert bits to KiB or MiB for readability
    mem_bits: float = plbf_instance.memory_usage_of_backup_bf
    mem_kib: float = mem_bits / 8.0 / 1024.0
    mem_mib: float = mem_kib / 1024.0
    if mem_mib >= 1.0:
        print(
            f"Memory Usage of Backup BFs: {mem_mib:.2f} MiB "
            f"({mem_bits:.0f} bits)"
        )
    else:
        print(
            f"Memory Usage of Backup BFs: {mem_kib:.2f} KiB "
            f"({mem_bits:.0f} bits)"
        )

    print("\nOptimal Thresholds (t):")
    print([f"{th:.4f}" for th in plbf_instance.t])
    print("\nOptimal FPRs per Region (f):")
    # Format FPRs nicely (handle None)
    f_formatted: List[str] = ["None"] + [
        f"{fpr:.4e}" if fpr is not None else "None"
        for fpr in plbf_instance.f[1:]
    ]
    print(f_formatted)
    print("----------------")


if __name__ == "__main__":
    main()
