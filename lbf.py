import argparse
import math
import time
import sys
from typing import (
    Callable,
    Final,
    Generic,
    Any,
    List,
    Optional,
    Tuple,
    TypeVar,
    Sequence,
    Iterable,
)
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

from bloom_filter import BloomFilter, Serializer
from utils import deep_sizeof, sizeof

# Type variables and aliases
KeyType = TypeVar("KeyType")
# Define a specific type for the predictor function
Predictor = Callable[[Iterable[KeyType]], List[float]]

# Constants
EPS: Final[float] = 1e-8
INF: Final[float] = float("inf")


class LearnedBloomFilter(Generic[KeyType]):
    """
    Learned Bloom Filter Implementation.

    Uses a predictor model and a single threshold (`tau`) to classify keys.
    Keys predicted as positive (score >= tau) are accepted.
    Keys predicted as negative (score < tau) are checked against a backup
    Bloom filter containing the true positive keys that were missed by the model
    (false negatives).
    """

    def __init__(
        self,
        predictor: Predictor[KeyType],
        pos_keys: Sequence[KeyType],
        neg_keys: Sequence[KeyType],  # Used for threshold tuning & testing
        target_overall_fpr: float,
        neg_val_split: float = 0.5,  # Fraction of neg_keys for validation
        random_seed: int = 42,
        serializer: Optional[Serializer[KeyType]] = None,
    ) -> None:
        """
        Initializes the LearnedBloomFilter.

        Args:
            predictor (Predictor[KeyType]): Function mapping keys to scores [0, 1].
            pos_keys (Sequence[KeyType]): Positive keys to insert.
            neg_keys (Sequence[KeyType]): Negative keys used for threshold tuning
                                          and potentially final FPR testing.
            target_overall_fpr (float): The desired overall FPR (0 < F < 1).
            neg_val_split (float): Fraction of `neg_keys` used for determining
                                   the threshold (validation set). The rest can
                                   be used for testing the final filter's FPR.
            random_seed (int): Seed for splitting negative keys.
            serializer (Optional[Serializer[KeyType]]): Serializer for keys in
                                                        the backup Bloom filter.
        """
        # --- Input Validation ---
        assert isinstance(pos_keys, Sequence)
        assert isinstance(neg_keys, Sequence)
        assert callable(predictor)
        assert isinstance(target_overall_fpr, float) and 0 < target_overall_fpr < 1
        assert isinstance(neg_val_split, float) and 0 < neg_val_split < 1
        assert isinstance(random_seed, int)

        # --- Store Core Parameters ---
        self._predictor: Final[Predictor[KeyType]] = predictor
        self._serializer: Final[Optional[Serializer[KeyType]]] = serializer
        self.target_overall_fpr: Final[float] = target_overall_fpr
        self.n_pos: Final[int] = len(pos_keys)

        # --- Split Negative Keys ---
        self.neg_keys_val: List[KeyType] = []
        self.neg_keys_test: List[KeyType] = []
        if neg_keys:
            if len(neg_keys) == 1:  # Cannot split a single key
                self.neg_keys_val = list(neg_keys)
                self.neg_keys_test = []
                print(
                    "Warning: Only one negative key provided. Using it for validation, none for testing."
                )
            else:
                self.neg_keys_val, self.neg_keys_test = train_test_split(
                    neg_keys,
                    test_size=(1.0 - neg_val_split),
                    random_state=random_seed,
                )
        print(
            f"Split negative keys: {len(self.neg_keys_val)} for validation (threshold tuning), "
            f"{len(self.neg_keys_test)} for testing."
        )

        # --- Step 1: Compute Scores using Predictor ---
        print("Computing scores using predictor...")
        start_score_time = time.time()
        # Scores needed for positives (to find false negatives) and validation negatives (for threshold)
        pos_scores: List[float] = self._predictor(pos_keys)
        neg_scores_val: List[float] = []
        if self.neg_keys_val:
            neg_scores_val = self._predictor(self.neg_keys_val)
        end_score_time = time.time()
        print(f"Score computation took {end_score_time - start_score_time:.2f}s")

        # --- Step 2: Determine the Threshold (tau) ---
        # Aim for model FPR = target_overall_fpr / 2 on validation negatives
        self.target_model_fpr: Final[float] = self.target_overall_fpr / 2.0
        print(
            f"Determining threshold for target model FPR: {self.target_model_fpr:.6f}"
        )
        start_thresh_time = time.time()
        self.threshold: float = self._determine_threshold(
            neg_scores_val, self.target_model_fpr
        )
        end_thresh_time = time.time()
        print(
            f"Determined threshold: {self.threshold:.6f} "
            f"({end_thresh_time - start_thresh_time:.2f}s)"
        )

        # --- Step 3: Identify False Negatives ---
        false_negatives: List[KeyType] = [
            key for key, score in zip(pos_keys, pos_scores) if score < self.threshold
        ]
        self.n_false_negatives: Final[int] = len(false_negatives)
        print(
            f"Identified {self.n_false_negatives} false negatives "
            f"(positives with score < {self.threshold:.6f})."
        )

        # --- Step 4: Build Backup Bloom Filter ---
        # Aim for backup filter FPR = target_overall_fpr / 2
        self.target_backup_fpr: Final[float] = self.target_overall_fpr / 2.0
        self.backup_bf: Optional[BloomFilter[KeyType]] = None
        self.memory_usage_backup_bf_bytes: int = 0  # Store calculated size

        print(
            f"Building backup Bloom filter for {self.n_false_negatives} keys "
            f"with target FPR: {self.target_backup_fpr:.6f}"
        )
        start_build_time = time.time()
        if self.n_false_negatives > 0:
            try:
                # Ensure error rate is strictly between 0 and 1
                effective_backup_fpr = max(EPS, min(1.0 - EPS, self.target_backup_fpr))

                self.backup_bf = BloomFilter[KeyType](
                    capacity=self.n_false_negatives,
                    error_rate=effective_backup_fpr,
                    serializer=self._serializer,
                )
                for key in false_negatives:
                    self.backup_bf.add(key)

                # Estimate memory usage based on the BF's own calculation if available
                if hasattr(self.backup_bf, "memory_usage_bytes"):
                    self.memory_usage_backup_bf_bytes = (
                        self.backup_bf.memory_usage_bytes()
                    )
                elif hasattr(self.backup_bf, "bit_array_size"):
                    self.memory_usage_backup_bf_bytes = math.ceil(
                        self.backup_bf.bit_array_size() / 8
                    )
                else:
                    # Fallback: estimate using formula if capacity/error_rate are accessible
                    try:
                        n = self.backup_bf.capacity  # type: ignore
                        p = self.backup_bf.error_rate  # type: ignore
                        if n > 0 and 0 < p < 1:
                            m_bits = -(n * math.log(p)) / (math.log(2) ** 2)
                            self.memory_usage_backup_bf_bytes = int(
                                math.ceil(m_bits / 8)
                            )
                        else:
                            self.memory_usage_backup_bf_bytes = 0
                    except AttributeError:
                        self.memory_usage_backup_bf_bytes = 0  # Cannot estimate

            except Exception as e:
                print(f"Error creating or populating backup BloomFilter: {e}")
                self.backup_bf = None
                self.memory_usage_backup_bf_bytes = 0
        else:
            # No false negatives, no backup filter needed.
            self.backup_bf = None
            self.memory_usage_backup_bf_bytes = 0

        end_build_time = time.time()
        print(
            f"Backup filter construction took {end_build_time - start_build_time:.2f}s. "
            f"Estimated size: {self.memory_usage_backup_bf_bytes} bytes."
        )

    def _determine_threshold(
        self, neg_scores_val: List[float], target_model_fpr: float
    ) -> float:
        """
        Determines the score threshold `tau` such that approximately
        `target_model_fpr` of the validation negative scores are >= `tau`.

        Args:
            neg_scores_val (List[float]): Scores of the validation negative keys.
            target_model_fpr (float): The desired FPR for the model alone.

        Returns:
            float: The calculated threshold `tau`.
        """
        if not neg_scores_val:
            print(
                "Warning: No validation negative scores provided. "
                "Cannot determine threshold dynamically. Returning default 0.5"
            )
            return 0.5  # Default threshold if no negatives to tune on

        # Sort scores in ascending order
        sorted_neg_scores = sorted(neg_scores_val)
        n_neg = len(sorted_neg_scores)

        # Find the index corresponding to the (1 - target_model_fpr) quantile
        # If target_model_fpr = 0.01, we want the score at the 99th percentile
        # (such that 1% of scores are >= this value)
        quantile_index = int(math.ceil(n_neg * (1.0 - target_model_fpr)))

        # Clamp index to be within valid bounds [0, n_neg - 1]
        quantile_index = max(0, min(quantile_index, n_neg - 1))

        # The threshold is the score at this index
        threshold = sorted_neg_scores[quantile_index]

        # Handle edge case: if multiple scores are identical at the threshold,
        # picking this value might result in slightly different FPR.
        # A small adjustment might be needed in practice, but this is standard.

        # Ensure threshold is within [0, 1]
        threshold = max(0.0, min(threshold, 1.0))

        # Optional: Verify the achieved FPR on the validation set (for debugging)
        # actual_fpr = sum(s >= threshold for s in neg_scores_val) / n_neg
        # print(f"  Threshold {threshold:.6f} yields actual model FPR {actual_fpr:.6f} on validation set.")

        return threshold

    def contains(self, key: KeyType) -> bool:
        """
        Checks if a key might be present in the set.

        Args:
            key (KeyType): The key to check.

        Returns:
            bool: True if the key might be present (potential positive or false positive),
                  False if the key is definitely not present (true negative).
        """
        # Get score from predictor (handle potential batching if predictor expects iterable)
        try:
            score: float = self._predictor([key])[0]
        except Exception as e:
            print(f"Error getting prediction for key {key}: {e}. Assuming positive.")
            return True  # Fail-safe assumption

        assert 0.0 <= score <= 1.0, f"Predictor score out of bounds: {score}"

        # Step 1: Check against threshold
        if score >= self.threshold:
            return True  # Predicted positive by model

        # Step 2: If predicted negative, check backup Bloom filter
        if self.backup_bf is not None:
            try:
                return key in self.backup_bf
            except Exception as e:
                print(
                    f"Error checking backup Bloom filter for key {key}: {e}. Assuming positive."
                )
                return True  # Fail-safe assumption
        else:
            # No backup filter exists (either no FNs or creation failed)
            return False  # Definitely not present if score < threshold and no backup

    def contains_batch(self, keys: Iterable[KeyType]) -> List[bool]:
        """
        Batched version of the `contains` method.

        Args:
            keys (Iterable[KeyType]): The keys to check.

        Returns:
            List[bool]: For each key, True if it might be present, False otherwise.
        """
        key_list = list(keys)
        if not key_list:
            return []

        results: List[bool] = [False] * len(key_list)
        try:
            scores: List[float] = self._predictor(key_list)
        except Exception as e:
            print(f"Error getting batch prediction: {e}. Assuming all positive.")
            return [True] * len(key_list)  # Fail-safe assumption

        assert len(scores) == len(key_list), "Predictor returned wrong number of scores"

        for i, (key, score) in enumerate(zip(key_list, scores)):
            assert 0.0 <= score <= 1.0, f"Predictor score out of bounds: {score}"

            if score >= self.threshold:
                results[i] = True
            elif self.backup_bf is not None:
                try:
                    results[i] = key in self.backup_bf
                except Exception as e:
                    print(
                        f"Error checking backup Bloom filter for key {key} in batch: {e}. Assuming positive."
                    )
                    results[i] = True  # Fail-safe assumption
            else:
                results[i] = False  # Score < threshold and no backup filter

        return results

    def get_actual_size_bytes(
        self, with_overhead: bool = False, verbose: bool = False
    ) -> int:
        """
        Calculates the actual memory footprint of the core data structures
        in bytes, including the threshold, backup Bloom filter, and predictor.

        Args:
            with_overhead (bool): Whether to include Python's base object overhead.
            verbose (bool): If True, prints detailed size breakdown.

        Returns:
            int: The estimated size in bytes.
        """
        total_bytes: int = 0
        print_prefix = "Calculating actual size:" if verbose else None

        # 1. Size of the threshold (float)
        try:
            threshold_bytes: int = sizeof(self.threshold, with_overhead=with_overhead)
            if verbose and print_prefix:
                print(f"{print_prefix} Threshold (tau): {threshold_bytes} bytes")
            total_bytes += threshold_bytes
        except Exception as e:
            if verbose:
                print(f"{print_prefix} Error calculating size of threshold: {e}")

        # 2. Size of the backup Bloom filter (self.backup_bf)
        if self.backup_bf is not None:
            try:
                filter_bytes: int = deep_sizeof(
                    self.backup_bf, with_overhead=with_overhead, verbose=False
                )

                if verbose and print_prefix:
                    print(
                        f"{print_prefix} Backup Bloom Filter (backup_bf): {filter_bytes} bytes"
                    )
                total_bytes += filter_bytes
            except Exception as e:
                if verbose:
                    print(
                        f"{print_prefix} Error calculating size of backup filter: {e}"
                    )
        elif verbose and print_prefix:
            print(f"{print_prefix} Backup Bloom Filter (backup_bf): 0 bytes (None)")

        # 3. Size of the predictor function/object (self._predictor)
        if hasattr(self, "_predictor") and self._predictor is not None:
            try:
                predictor_bytes: int = deep_sizeof(
                    self._predictor, with_overhead=with_overhead, verbose=verbose
                )
                if verbose and print_prefix:
                    print(
                        f"{print_prefix} Predictor (_predictor): {predictor_bytes} bytes"
                    )
                total_bytes += predictor_bytes
            except TypeError as e:
                if verbose:
                    print(
                        f"{print_prefix} Predictor type ({type(self._predictor)}) not directly supported by deep_sizeof, using basic sizeof: {e}"
                    )
                try:
                    predictor_bytes = sizeof(
                        self._predictor, with_overhead=with_overhead
                    )
                    if verbose:
                        print(
                            f"{print_prefix} Predictor (_predictor) (basic sizeof): {predictor_bytes} bytes"
                        )
                    total_bytes += predictor_bytes
                except Exception as e2:
                    if verbose:
                        print(
                            f"{print_prefix} Error calculating basic size of predictor: {e2}"
                        )
            except Exception as e:
                if verbose:
                    print(
                        f"{print_prefix} Error calculating deep size of predictor: {e}"
                    )

        if verbose:
            print(f"Total Calculated Actual Size: {total_bytes} bytes")
        return total_bytes

    def evaluate_fpr(
        self, test_neg_keys: Optional[Sequence[KeyType]] = None
    ) -> Tuple[float, int, int]:
        """
        Evaluates the actual False Positive Rate on a provided set of negative keys
        or the internal test set if available.

        Args:
            test_neg_keys (Optional[Sequence[KeyType]]): A sequence of known negative
                keys to test against. If None, uses `self.neg_keys_test`.

        Returns:
            Tuple[float, int, int]: Measured FPR, count of false positives, total keys tested.
                                    Returns (0.0, 0, 0) if no test keys are available.
        """
        keys_to_test: Sequence[KeyType]
        if test_neg_keys is not None:
            keys_to_test = test_neg_keys
        elif self.neg_keys_test:
            keys_to_test = self.neg_keys_test
        else:
            print("Warning: No negative keys provided or available for FPR evaluation.")
            return 0.0, 0, 0

        n_test = len(keys_to_test)
        if n_test == 0:
            return 0.0, 0, 0

        print(f"Evaluating FPR on {n_test} negative keys...")
        start_eval_time = time.time()
        results = self.contains_batch(keys_to_test)
        fp_count = sum(results)
        measured_fpr = fp_count / n_test
        end_eval_time = time.time()
        print(f"FPR evaluation took {end_eval_time - start_eval_time:.2f}s.")

        return measured_fpr, fp_count, n_test


# --- Main Execution Block ---
def main() -> None:
    """Main function to parse arguments, load data, train predictor,
    build LearnedBloomFilter, and test."""
    parser = argparse.ArgumentParser(
        description="Construct and test a Learned Bloom Filter with a trained predictor."
    )
    parser.add_argument(
        "--data_path",
        action="store",
        dest="data_path",
        type=str,
        required=True,
        help="Path of the dataset CSV file (needs 'key', 'label' columns)",
    )
    parser.add_argument(
        "--target_fpr",
        action="store",
        dest="target_fpr",
        type=float,
        required=True,
        help="F: The target overall false positive rate for the Learned BF",
    )
    parser.add_argument(
        "--lbf_neg_split",
        action="store",
        dest="lbf_neg_split",
        type=float,
        default=0.7,
        help="Fraction of negative samples used for LBF construction/evaluation "
        "(threshold tuning and internal testing) vs predictor training (default: 0.7)",
    )
    parser.add_argument(
        "--neg_val_split",
        action="store",
        dest="neg_val_split",
        type=float,
        default=0.5,
        help="Fraction of LBF negative samples used for threshold tuning (validation) "
        "vs internal testing (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        action="store",
        dest="seed",
        type=int,
        default=42,
        help="Random seed for train/test splits (default: 42)",
    )
    parser.add_argument(
        "--hash_features",
        action="store",
        dest="hash_features",
        type=int,
        default=2**12,  # Default number of features for HashingVectorizer
        help="Number of features for HashingVectorizer (default: 4096)",
    )

    results: argparse.Namespace = parser.parse_args()

    DATA_PATH: Final[str] = results.data_path
    TARGET_FPR: Final[float] = results.target_fpr
    LBF_NEG_SPLIT: Final[float] = results.lbf_neg_split
    NEG_VAL_SPLIT: Final[float] = results.neg_val_split
    SEED: Final[int] = results.seed
    HASH_FEATURES: Final[int] = results.hash_features

    # --- Data Loading and Preparation ---
    print(f"Loading data from: {DATA_PATH}")
    try:
        data: pd.DataFrame = pd.read_csv(DATA_PATH)
        required_cols: Final[set[str]] = {"key", "label"}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")
        data["key"] = data["key"].astype(str)
        data["label"] = pd.to_numeric(data["label"], errors="coerce")
        if data["label"].isnull().any():
            raise ValueError("Non-numeric values found in 'label' column.")
        # Convert labels to binary (1 for positive, 0 for negative)
        data["binary_label"] = (data["label"] == 1).astype(int)

    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error loading or validating data: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        sys.exit(1)

    # Separate positive and negative samples
    positive_sample: pd.DataFrame = data.loc[(data["binary_label"] == 1)].copy()
    negative_sample: pd.DataFrame = data.loc[(data["binary_label"] == 0)].copy()

    if len(positive_sample) == 0:
        print("Error: No positive samples (label=1) found in the data.")
        sys.exit(1)

    # --- Split Data for Predictor Training and LBF Construction ---
    # Split negatives: a portion for predictor training, the rest for LBF
    neg_train_pred: pd.DataFrame
    neg_lbf_construction: pd.DataFrame

    if len(negative_sample) == 0:
        print(
            "Warning: No negative samples found. Cannot train predictor or "
            "evaluate FPR accurately."
        )
        neg_train_pred = negative_sample.copy()
        neg_lbf_construction = negative_sample.copy()
    else:
        # Use (1 - LBF_NEG_SPLIT) fraction for predictor training
        # Use LBF_NEG_SPLIT fraction for LBF construction/evaluation
        neg_train_pred, neg_lbf_construction = train_test_split(
            negative_sample, train_size=(1.0 - LBF_NEG_SPLIT), random_state=SEED
        )

    print(f"Total positive samples: {len(positive_sample)}")
    print(f"Total negative samples: {len(negative_sample)}")
    print(f"  - Negatives for predictor training: {len(neg_train_pred)}")
    print(f"  - Negatives for LBF construction/evaluation: {len(neg_lbf_construction)}")
    assert len(neg_train_pred) + len(neg_lbf_construction) == len(negative_sample)

    # --- Train Predictor ---
    print(
        f"\nTraining predictor model (Logistic Regression with {HASH_FEATURES} hash features)..."
    )
    # Prepare data for predictor training (positives + subset of negatives)
    train_pred_data = pd.concat([positive_sample, neg_train_pred])
    X_train_pred: pd.Series[str] = train_pred_data["key"]
    y_train_pred: pd.Series[int] = train_pred_data["binary_label"]

    # Define model pipeline
    predictor_model = Pipeline(
        [
            (
                "vectorizer",
                HashingVectorizer(
                    n_features=HASH_FEATURES, alternate_sign=False, ngram_range=(1, 3)
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear", random_state=SEED, class_weight="balanced"
                ),
            ),
        ]
    )

    start_pred_train_time: float = time.time()
    try:
        # Avoid training if no training data exists (e.g., no negatives)
        if not X_train_pred.empty:
            predictor_model.fit(X_train_pred, y_train_pred)
            print("Predictor training complete.")
        else:
            print("Skipping predictor training: No training data available.")
            # Handle case where model cannot be fitted - predictor func needs to be robust
    except Exception as e:
        print(f"Error training predictor model: {e}")
        sys.exit(1)
    end_pred_train_time: float = time.time()
    print(f"Predictor training took {end_pred_train_time - start_pred_train_time:.2f}s")

    # --- Define the Predictor Function for LBF ---
    predictor_cache: dict[str, float] = {}  # Optional cache

    def trained_predictor(keys: Iterable[str]) -> List[float]:
        """Batched predictor function using the trained model."""
        keys_list = list(map(str, keys))
        results = [0.0] * len(keys_list)
        keys_to_predict_indices: List[int] = []
        keys_to_predict: List[str] = []

        # Check cache first
        for i, key in enumerate(keys_list):
            if key in predictor_cache:
                results[i] = predictor_cache[key]
            else:
                keys_to_predict_indices.append(i)
                keys_to_predict.append(key)

        # Predict uncached keys
        if keys_to_predict:
            try:
                probas: np.ndarray[Any, np.dtype[np.float64]] = (
                    predictor_model.predict_proba(keys_to_predict)
                )
                scores = probas[:, 1].astype(float).tolist()
                for idx, key, score in zip(
                    keys_to_predict_indices, keys_to_predict, scores
                ):
                    results[idx] = score
                    predictor_cache[key] = score  # Update cache
            except NotFittedError:
                print(
                    "Warning: Predictor model not fitted. Returning default score 0.0."
                )
                # Return default scores for keys that needed prediction
                for idx in keys_to_predict_indices:
                    results[idx] = 0.0
            except Exception as e:
                print(f"Error during prediction: {e}. Returning default score 0.0.")
                for idx in keys_to_predict_indices:
                    results[idx] = 0.0  # Fail-safe score

        return results

    predictor_func: Predictor[str] = trained_predictor

    # --- Prepare Key Lists for LBF ---
    pos_keys: List[str] = list(positive_sample["key"])
    # Use negatives reserved for LBF construction/evaluation
    neg_keys_lbf: List[str] = list(neg_lbf_construction["key"])

    # --- Construct LearnedBloomFilter ---
    print(f"\nConstructing LearnedBloomFilter...")
    print(f"Parameters: Target FPR={TARGET_FPR}, Neg Val Split={NEG_VAL_SPLIT}")
    _total_construct_start: float = time.time()
    lbf_instance: LearnedBloomFilter[str]
    try:
        # Pass the predictor function, positive keys, and the designated negative keys
        lbf_instance = LearnedBloomFilter(
            predictor=predictor_func,
            pos_keys=pos_keys,
            neg_keys=neg_keys_lbf,  # These will be split internally by LBF
            target_overall_fpr=TARGET_FPR,
            neg_val_split=NEG_VAL_SPLIT,  # Controls internal split for threshold tuning
            random_seed=SEED,
            # serializer= # Optional: Add serializer if needed for your key type
        )
    except RuntimeError as e:
        print(f"Error during LearnedBloomFilter construction: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during construction: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    total_construct_end: float = time.time()
    # Construction time reported by LBF includes score computation, threshold finding, building filter
    print(
        "Total time including predictor training and LBF construction: "
        f"{total_construct_end - start_pred_train_time:.4f} seconds."
    )

    # --- Verification: No False Negatives ---
    print("\nTODO: Verifying no false negatives...")

    # --- Testing: False Positive Rate ---
    print("\nCalculating false positive rate on internal test set...")
    # evaluate_fpr uses the internal neg_keys_test split by default
    measured_fpr, fp_cnt, neg_tested = lbf_instance.evaluate_fpr()

    # --- Output Results ---
    print("\n--- Learned Bloom Filter Results ---")
    print(f"Target Overall FPR: {TARGET_FPR:.6f}")
    if neg_tested > 0:
        print(
            f"Measured FPR on internal test set: {measured_fpr:.6f} "
            f"[{fp_cnt} / {neg_tested}]"
        )
    else:
        print("Measured FPR on internal test set: N/A (no internal test negatives)")

    print(f"Determined Score Threshold (tau): {lbf_instance.threshold:.6f}")
    print(f"False Negatives Stored in Backup BF: {lbf_instance.n_false_negatives}")

    # Memory Usage
    mem_backup_bytes: int = lbf_instance.memory_usage_backup_bf_bytes
    mem_backup_kib: float = mem_backup_bytes / 1024.0

    if mem_backup_kib >= 1.0:
        print(
            f"Memory Usage of Backup BF: {mem_backup_kib:.2f} KiB ({mem_backup_bytes} bytes)"
        )
    else:
        print(f"Memory Usage of Backup BF: {mem_backup_bytes} bytes")

    # Calculate total size (may include predictor size estimation)
    total_mem_bytes = lbf_instance.get_actual_size_bytes(
        verbose=False
    )  # Set verbose=True for breakdown
    total_mem_kib = total_mem_bytes / 1024.0
    total_mem_mib = total_mem_kib / 1024.0

    if total_mem_mib >= 1.0:
        print(
            f"Total Estimated Memory (Predictor + Backup BF + Threshold): {total_mem_mib:.2f} MiB ({total_mem_bytes} bytes)"
        )
    elif total_mem_kib >= 0.1:
        print(
            f"Total Estimated Memory (Predictor + Backup BF + Threshold): {total_mem_kib:.2f} KiB ({total_mem_bytes} bytes)"
        )
    else:
        print(
            f"Total Estimated Memory (Predictor + Backup BF + Threshold): {total_mem_bytes} bytes"
        )

    # Optional: Estimate predictor size separately if needed
    try:
        predictor_size = deep_sizeof(predictor_model)
        print(
            f"Estimated Predictor Size (using deep_sizeof): {predictor_size / 1024.0:.2f} KiB"
        )
    except NameError:
        print("Predictor size estimation skipped (deep_sizeof not available).")
    except Exception as e:
        print(f"Could not estimate predictor size: {e}")

    print("------------------------------------")


if __name__ == "__main__":
    main()
