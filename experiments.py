import time
import math
import logging
import statistics
import argparse
import sys # For deep_sizeof approximation
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Iterable,
    Final,
    Tuple,
)
from dataclasses import dataclass, field

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

from bloom_filter import BloomFilter
from fast_plbf import FastPLBF
# TODO: from learned_bloom_filter import LearnedBloomFilter

# --- Assume these imports exist from your project ---
from timers import (
    CpuTimer,
    CudaTimer,
    TimerError,
    NS_PER_S,
)

KeyType = str # for URL dataset

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- Configuration and Result Dataclasses ---

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run, taking a pre-built object."""
    # The actual, instantiated object to test (e.g., PLBF, BF)
    test_subject: Any
    # Identifier for this specific experiment configuration
    experiment_name: str
    # Data subsets needed for testing
    positive_samples: pd.DataFrame # Contains 'key' column
    negative_samples_test: pd.DataFrame # Contains 'key' column
    # Parameters for the run
    device: str # 'cpu' or 'cuda' (Hint for timer and potential data movement)
    batch_sizes: List[int]
    repetitions: int
    warmup_reps: int
    # Optional context parameters
    target_fpr: Optional[float] = None
    # Method name on test_subject to call for timing queries
    query_method_name: str = "contains_batch"

@dataclass
class TimingResult:
    """Stores timing results for one repetition."""
    batch_size: int
    repetition: int
    total_latency_ns: float
    # Add other per-repetition metrics if needed

@dataclass
class ExperimentResult:
    """Aggregated results for a specific configuration and batch size."""
    config_name: str
    batch_size: int
    avg_total_latency_ns: float
    std_total_latency_ns: float
    median_total_latency_ns: float
    p5_total_latency_ns: float
    p95_total_latency_ns: float
    avg_per_item_latency_ns: float = field(init=False)
    # Optional metrics calculated by the runner
    measured_fpr: Optional[float] = None
    memory_usage_bytes: Optional[float] = None # Example

    def __post_init__(self):
        # Avoid division by zero if batch_size is somehow 0
        self.avg_per_item_latency_ns = (
            self.avg_total_latency_ns / self.batch_size
            if self.batch_size > 0 else 0.0
        )

# --- Experiment Runner ---

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.test_subject = config.test_subject

        if config.device == 'cuda' and not torch.cuda.is_available():
             raise ValueError("CUDA device requested but CUDA is not available.")

        # Check if the test subject has the specified query method
        if not hasattr(self.test_subject, config.query_method_name):
            raise AttributeError(
                f"Test subject of type {type(self.test_subject).__name__} "
                f"does not have the method '{config.query_method_name}'"
            )
        self.query_method = getattr(self.test_subject, config.query_method_name)

        # Prepare keys from dataframes once
        self.pos_keys_test: List[KeyType] = list(config.positive_samples['key'].astype(str))
        self.neg_keys_test: List[KeyType] = list(config.negative_samples_test['key'].astype(str))
        if not self.pos_keys_test and not self.neg_keys_test:
             logging.warning("Both positive and negative test key lists are empty.")


    def _prepare_test_batch(self, batch_size: int) -> List[KeyType]:
        """Prepares a batch of keys for testing, mixing positive and negative."""
        if batch_size <= 0:
            return []

        num_pos = min(len(self.pos_keys_test), batch_size // 2)
        num_neg = min(len(self.neg_keys_test), batch_size - num_pos)

        # If we still haven't filled the batch, fill remaining with positives
        if num_pos + num_neg < batch_size:
            num_pos = min(len(self.pos_keys_test), batch_size - num_neg)

        # Sample without replacement if possible, otherwise with replacement
        pos_sample = np.random.choice(
            self.pos_keys_test, num_pos, replace=len(self.pos_keys_test) < num_pos
        ).tolist() if num_pos > 0 and self.pos_keys_test else []

        neg_sample = np.random.choice(
            self.neg_keys_test, num_neg, replace=len(self.neg_keys_test) < num_neg
        ).tolist() if num_neg > 0 and self.neg_keys_test else []

        batch = pos_sample + neg_sample
        np.random.shuffle(batch) # Shuffle the combined batch

        # --- Device Handling ---
        # This is heuristic. The runner doesn't *know* if the test_subject
        # needs data on the GPU. We assume it does if device='cuda'.
        # The test_subject's query method should handle the data type.
        # For PyTorch models, this often means converting to tensors.
        # For CPU-based structures (sklearn, pure Python), list is fine.
        # This part might need adjustment based on your specific test_subject needs.
        # if self.config.device == 'cuda':
        #    try:
        #        # Example: Convert to tensor if using PyTorch subject
        #        return torch.tensor(batch, device='cuda') # Adjust dtype as needed
        #    except Exception as e:
        #        logging.warning(f"Could not convert batch to CUDA tensor: {e}. Passing list.")
        #        return batch # Fallback to list
        # else:
        #    return batch # Keep as list for CPU

        # For now, let's assume the query_method handles the input type (list of keys)
        return batch


    def run(self) -> List[ExperimentResult]:
        """Runs the full experiment suite for all configured batch sizes."""
        results_all: Dict[int, List[TimingResult]] = {
            bs: [] for bs in self.config.batch_sizes
        }
        num_reps = self.config.repetitions
        num_warmup = self.config.warmup_reps
        timer_device_str = self.config.device

        logging.info(f"Starting experiment: {self.config.experiment_name}")
        logging.info(f"Test Subject: {type(self.test_subject).__name__}")
        logging.info(f"Device Hint: {timer_device_str}, Batch sizes: {self.config.batch_sizes}")
        logging.info(f"Repetitions: {num_reps}, Warmup: {num_warmup}")
        logging.info(f"Query method: {self.config.query_method_name}")

        # Determine timer type
        is_gpu_timer = timer_device_str == 'cuda'
        timer_cls = CudaTimer if is_gpu_timer else CpuTimer
        device = torch.device('cuda') if is_gpu_timer else None # CudaTimer needs device
        logging.info(f"Using {'CudaTimer' if is_gpu_timer else 'CpuTimer'}.")

        # --- Warmup Phase ---
        logging.info("Running warmup iterations...")
        if self.config.batch_sizes:
            warmup_batch_size = self.config.batch_sizes[0]
            warmup_batch = self._prepare_test_batch(warmup_batch_size)
            if warmup_batch:
                try:
                    for _ in range(num_warmup):
                        _ = self.query_method(warmup_batch)
                    if is_gpu_timer:
                        torch.cuda.synchronize(device)
                    logging.info("Warmup complete.")
                except Exception as e:
                    logging.error(f"Error during warmup: {e}. Aborting warmup.", exc_info=True)
            else:
                logging.warning("Warmup batch is empty, skipping warmup.")
        else:
             logging.warning("No batch sizes configured, skipping warmup.")

        # --- Measurement Phase ---
        total_queries = 0
        start_time_exp = time.time()

        for batch_size in self.config.batch_sizes:
            logging.info(f"--- Measuring Batch Size: {batch_size} ---")
            batch = self._prepare_test_batch(batch_size)
            if not batch:
                logging.warning(f"Prepared batch is empty for size {batch_size}. Skipping.")
                continue

            current_batch_timings: List[TimingResult] = []
            for rep in range(num_reps):
                try:
                    # Select timer instance based on device config
                    with CudaTimer(device) if is_gpu_timer else CpuTimer() as timer:
                        # The core operation using the configured method
                        _ = self.query_method(batch) # Result ignored for timing

                    total_latency_ns = timer.get_elapsed_ns()
                    current_batch_timings.append(
                        TimingResult(
                            batch_size=batch_size,
                            repetition=rep + 1,
                            total_latency_ns=total_latency_ns,
                        )
                    )
                    total_queries += batch_size

                except TimerError as e:
                    logging.error(f"Timer error (B={batch_size}, Rep={rep+1}): {e}. Skipping rep.")
                except Exception as e:
                     logging.error(f"Query error (B={batch_size}, Rep={rep+1}): {e}. Skipping rep.", exc_info=True)

            results_all[batch_size] = current_batch_timings
            if current_batch_timings:
                 avg_lat = statistics.mean(r.total_latency_ns for r in current_batch_timings)
                 logging.info(f"Batch Size {batch_size} complete. Avg total latency: {avg_lat / NS_PER_S:.6f} s")

        end_time_exp = time.time()
        logging.info(f"Measurement phase finished in {end_time_exp - start_time_exp:.2f} seconds. Total queries timed: {total_queries}")

        # --- Optional Metrics Calculation (Example: FPR and Memory) ---
        measured_fpr: Optional[float] = None
        memory_bytes: Optional[float] = None

        # Calculate FPR using the dedicated negative test set
        if self.neg_keys_test:
            logging.info("Calculating False Positive Rate...")
            try:
                fp_results = self.query_method(self.neg_keys_test)
                fp_count = sum(fp_results) # Assumes query_method returns list/array of bool/int
                measured_fpr = fp_count / len(self.neg_keys_test)
                logging.info(f"Measured FPR: {measured_fpr:.6f} ({fp_count}/{len(self.neg_keys_test)})")
            except Exception as e:
                logging.warning(f"Could not calculate FPR: {e}", exc_info=True)
        else:
            logging.warning("No negative test keys available to calculate FPR.")

        # Get memory usage if the object supports it
        if hasattr(self.test_subject, 'get_memory_usage_bytes'):
             logging.info("Getting memory usage...")
             try:
                 memory_bytes = self.test_subject.get_memory_usage_bytes()
                 logging.info(f"Reported memory usage: {memory_bytes / (1024*1024):.2f} MiB")
             except Exception as e:
                 logging.warning(f"Could not get memory usage: {e}", exc_info=True)
        elif hasattr(self.test_subject, 'memory_usage_of_backup_bf'): # Fallback for PLBF example
             logging.info("Getting backup BF memory usage (fallback)...")
             try:
                 mem_bits = self.test_subject.memory_usage_of_backup_bf
                 memory_bytes = mem_bits / 8.0
                 logging.info(f"Reported backup BF memory usage: {memory_bytes / (1024*1024):.2f} MiB")
             except Exception as e:
                 logging.warning(f"Could not get backup BF memory usage: {e}", exc_info=True)
        else:
             logging.info("Test subject does not provide a standard memory usage method.")


        # --- Aggregate Results ---
        aggregated_results: List[ExperimentResult] = []
        for batch_size, timings in results_all.items():
            if not timings:
                logging.warning(f"No valid timing results for batch size {batch_size}.")
                continue

            total_latencies = [r.total_latency_ns for r in timings]
            if not total_latencies: continue

            agg = ExperimentResult(
                config_name=self.config.experiment_name,
                batch_size=batch_size,
                avg_total_latency_ns=statistics.mean(total_latencies),
                std_total_latency_ns=statistics.stdev(total_latencies) if len(total_latencies) > 1 else 0.0,
                median_total_latency_ns=statistics.median(total_latencies),
                # Use numpy for percentiles if available and needed
                p5_total_latency_ns=float(np.percentile(total_latencies, 5)) if total_latencies else 0.0,
                p95_total_latency_ns=float(np.percentile(total_latencies, 95)) if total_latencies else 0.0,
                # Add optional metrics calculated once per experiment run
                measured_fpr=measured_fpr,
                memory_usage_bytes=memory_bytes,
            )
            aggregated_results.append(agg)

        return aggregated_results


# --- Helper Functions (from your example, potentially move to utils) ---

# Basic deep_sizeof approximation (use cautiously)
def deep_sizeof(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents. """
    dict_handler = lambda d: sum(deep_sizeof(k, handlers, verbose) + deep_sizeof(v, handlers, verbose) for k, v in d.items())
    all_handlers = {
        tuple: lambda t: sum(deep_sizeof(i, handlers, verbose) for i in t),
        list: lambda l: sum(deep_sizeof(i, handlers, verbose) for i in l),
        dict: dict_handler,
        set: lambda s: sum(deep_sizeof(i, handlers, verbose) for i in s),
        frozenset: lambda s: sum(deep_sizeof(i, handlers, verbose) for i in s),
    }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = sys.getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=sys.stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += handler(o)
                break
        return s

    return sizeof(o)

# --- Main Execution Block ---

def main() -> None:
    """Main function to parse arguments, load data, build the specified structure, and run experiments."""
    parser = argparse.ArgumentParser(
        description="Construct and benchmark different filter structures (PLBF, LBF, BF)."
    )
    # --- Structure Selection ---
    parser.add_argument(
        "--structure_type",
        type=str,
        required=True,
        choices=["PLBF", "FastPLBF", "LBF", "BF"],
        help="Type of filter structure to build and test.",
    )

    # --- Common Arguments ---
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV data ('key', 'label')")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batch_sizes", type=str, default="1,8,64,256,1024", help="Comma-separated batch sizes")
    parser.add_argument("--repetitions", type=int, default=20, help="Timing repetitions per batch size")
    parser.add_argument("--warmup_reps", type=int, default=5, help="Warmup repetitions")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device hint for timing")

    # --- Data Split Arguments (Common) ---
    parser.add_argument("--plbf_test_split", type=float, default=0.7, help="Negatives fraction for final structure testing")
    parser.add_argument("--predictor_neg_split", type=float, default=0.5, help="Negatives fraction for predictor train vs h-learn (if applicable)")

    # --- Predictor Arguments (for LBF, PLBF) ---
    parser.add_argument("--hash_features", type=int, default=2**12, help="HashingVectorizer features (for LBF/PLBF)")

    # --- PLBF Specific Arguments ---
    parser.add_argument("--N", type=int, help="Initial segments (Required for PLBF/FastPLBF)")
    parser.add_argument("--k", type=int, help="Final regions (Required for PLBF/FastPLBF)")
    parser.add_argument("--F", type=float, help="Target overall FPR (Required for PLBF/FastPLBF)")
    # --use_fast_dp is implicitly handled by structure_type="FastPLBF"

    # --- BF / LBF Backup Arguments ---
    parser.add_argument("--bf_capacity", type=int, help="Capacity of the Bloom Filter (Required for BF, LBF backup)")
    parser.add_argument("--bf_error_rate", type=float, help="Error rate of the Bloom Filter (Required for BF, LBF backup)")

    # --- LBF Specific Arguments ---
    parser.add_argument("--lbf_threshold", type=float, help="Threshold for Learned Bloom Filter (Required for LBF)")


    results: argparse.Namespace = parser.parse_args()

    # --- Argument Validation based on Structure Type ---
    stype = results.structure_type
    if stype in ["PLBF", "FastPLBF"]:
        if results.N is None or results.k is None or results.F is None:
            parser.error("--N, --k, and --F are required for PLBF/FastPLBF")
        # Use F as the target FPR for PLBF
        TARGET_FPR = results.F
    elif stype == "LBF":
        if results.bf_capacity is None or results.bf_error_rate is None or results.lbf_threshold is None:
            parser.error("--bf_capacity, --bf_error_rate, and --lbf_threshold are required for LBF")
        # LBF doesn't have a single 'target FPR' in the same way PLBF does during construction
        TARGET_FPR = None # Or maybe use bf_error_rate as a reference?
    elif stype == "BF":
        if results.bf_capacity is None or results.bf_error_rate is None:
            parser.error("--bf_capacity and --bf_error_rate are required for BF")
        TARGET_FPR = results.bf_error_rate # Use BF's configured FPR as the 'target'

    # --- Extract Common Args ---
    DATA_PATH: Final[str] = results.data_path
    SEED: Final[int] = results.seed
    BATCH_SIZES: Final[List[int]] = [int(bs.strip()) for bs in results.batch_sizes.split(',')]
    REPETITIONS: Final[int] = results.repetitions
    WARMUP_REPS: Final[int] = results.warmup_reps
    DEVICE: Final[str] = results.device
    PLBF_TEST_SPLIT: Final[float] = results.plbf_test_split
    PREDICTOR_NEG_SPLIT: Final[float] = results.predictor_neg_split
    HASH_FEATURES: Final[int] = results.hash_features


    # --- 1. Data Loading and Preparation (Common to all structures) ---
    print(f"Loading data from: {DATA_PATH}")
    # (Keep the data loading and initial split logic as before)
    try:
        data: pd.DataFrame = pd.read_csv(DATA_PATH)
        required_cols: Final[set[str]] = {"key", "label"}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")
        data["key"] = data["key"].astype(str)
        data["label"] = pd.to_numeric(data["label"], errors="coerce")
        if data["label"].isnull().any():
            raise ValueError("Non-numeric values found in 'label' column.")
        data["binary_label"] = (data["label"] == 1).astype(int)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    positive_sample: pd.DataFrame = data.loc[(data["binary_label"] == 1)].copy()
    negative_sample: pd.DataFrame = data.loc[(data["binary_label"] == 0)].copy()

    if len(positive_sample) == 0: print("Error: No positive samples."); exit(1)

    # --- 2. Split Data (Common) ---
    neg_final_test: pd.DataFrame # Renamed for clarity
    neg_for_pred_and_h: pd.DataFrame
    if len(negative_sample) > 0:
        neg_for_pred_and_h, neg_final_test = train_test_split(
            negative_sample, test_size=PLBF_TEST_SPLIT, random_state=SEED
        )
    else:
        print("Warning: No negative samples found.")
        neg_final_test = negative_sample.copy()
        neg_for_pred_and_h = negative_sample.copy()

    neg_train_pred: pd.DataFrame
    neg_plbf_h_learn: pd.DataFrame # Keep name, used by PLBF, potentially LBF setup?
    if len(neg_for_pred_and_h) > 0:
        neg_train_pred, neg_plbf_h_learn = train_test_split(
            neg_for_pred_and_h, train_size=PREDICTOR_NEG_SPLIT, random_state=SEED + 1
        )
    else:
        neg_train_pred = neg_for_pred_and_h.copy()
        neg_plbf_h_learn = neg_for_pred_and_h.copy()

    print(f"Data split: +ve={len(positive_sample)}, "
          f"neg_pred={len(neg_train_pred)}, neg_h={len(neg_plbf_h_learn)}, "
          f"neg_test={len(neg_final_test)}")

    pos_keys: List[str] = list(positive_sample["key"])

    # --- 3. Setup Predictor (Only if needed by structure type) ---
    predictor_model = None
    predictor_func = None
    if stype in ["PLBF", "FastPLBF", "LBF"]:
        print(f"\nTraining predictor (Logistic Regression w/ {HASH_FEATURES} features)...")
        train_pred_data = pd.concat([positive_sample, neg_train_pred])
        X_train_pred: pd.Series[str] = train_pred_data["key"]
        y_train_pred: pd.Series[int] = train_pred_data["binary_label"]

        predictor_model = Pipeline([
            ('vectorizer', HashingVectorizer(n_features=HASH_FEATURES, alternate_sign=False, ngram_range=(1, 3))),
            ('classifier', LogisticRegression(solver='liblinear', random_state=SEED, class_weight='balanced'))
        ])
        start_pred_train_time: float = time.time()
        try:
            predictor_model.fit(X_train_pred, y_train_pred)
        except Exception as e: print(f"Error training predictor: {e}"); exit(1)
        end_pred_train_time: float = time.time()
        print(f"Predictor training took {end_pred_train_time - start_pred_train_time:.2f}s")

        # Define Predictor Function Wrapper (common for LBF/PLBF)
        def trained_predictor(keys: Iterable[str]) -> List[float]:
            keys_list = list(map(str, keys))
            if not keys_list: return []
            try:
                probas: np.ndarray[Any, Any] = predictor_model.predict_proba(keys_list)
                return probas[:, 1].astype(float).tolist()
            except NotFittedError: raise RuntimeError("Predictor model not fitted!")
            except Exception as e:
                logging.error(f"Error during prediction: {e}", exc_info=True)
                return [0.0] * len(keys_list)
        predictor_func = trained_predictor


    # --- 4. Construct the Test Subject ---
    print(f"\nConstructing {stype}...")
    test_subject: Optional[Any] = None
    construction_start_time = time.time()

    try:
        if stype in ["PLBF", "FastPLBF"]:
            # --- PLBF Construction ---
            if predictor_func is None: raise RuntimeError("Predictor function not created for PLBF.")
            N_param: Final[int] = results.N
            k_param: Final[int] = results.k
            F_param: Final[float] = results.F
            neg_keys_h_learn: List[str] = list(neg_plbf_h_learn["key"])

            # Assume PLBF/FastPLBF classes are imported
            # from your_plbf_module import PLBF, FastPLBF
            PLBFClass: type = FastPLBF if stype == "FastPLBF" else PLBF # type: ignore
            test_subject = PLBFClass(
                predictor=predictor_func,
                pos_keys=pos_keys,
                neg_keys=neg_keys_h_learn,
                F=F_param,
                N=N_param,
                k=k_param,
            )
            experiment_name_suffix = f"_N{N_param}_k{k_param}_F{F_param:.1E}_HF{HASH_FEATURES}"

        elif stype == "LBF":
            # --- LBF Construction ---
            if predictor_func is None: raise RuntimeError("Predictor function not created for LBF.")
            bf_capacity: Final[int] = results.bf_capacity
            bf_error_rate: Final[float] = results.bf_error_rate
            lbf_threshold: Final[float] = results.lbf_threshold

            # Assume BloomFilter and LearnedBloomFilter classes are imported/defined
            # from bloom_filter import BloomFilter # Or your implementation
            # from learned_bloom_filter import LearnedBloomFilter # Or your implementation

            # Create and populate the backup Bloom Filter
            print(f"  Building backup Bloom Filter (Cap={bf_capacity}, Err={bf_error_rate})")
            backup_bf = BloomFilter[str](capacity=bf_capacity, error_rate=bf_error_rate)
            # Populate with *all* positive keys (common LBF assumption)
            for key in pos_keys:
                backup_bf.add(key)
            print(f"  Backup BF populated with {len(pos_keys)} positive keys.")

            # Instantiate LBF
            test_subject = LearnedBloomFilter( # type: ignore
                predictor=predictor_func,
                backup_filter=backup_bf,
                threshold=lbf_threshold
            )
            experiment_name_suffix = f"_Thresh{lbf_threshold:.2f}_Cap{bf_capacity}_Err{bf_error_rate:.1E}_HF{HASH_FEATURES}"

        elif stype == "BF":
            # --- BF Construction ---
            bf_capacity: Final[int] = results.bf_capacity
            bf_error_rate: Final[float] = results.bf_error_rate

            # Assume BloomFilter class is imported/defined
            # from bloom_filter import BloomFilter

            # Create and populate the Bloom Filter
            print(f"  Building Bloom Filter (Cap={bf_capacity}, Err={bf_error_rate})")
            bf_instance = BloomFilter[str](capacity=bf_capacity, error_rate=bf_error_rate)
            # Populate with *all* positive keys
            for key in pos_keys:
                bf_instance.add(key)
            print(f"  BF populated with {len(pos_keys)} positive keys.")

            # We need to ensure the BF instance conforms to the interface expected by the runner
            # Let's use a simple wrapper if needed, assuming BF has `__contains__` and `calc_memory_usage_bits`
            class WrappedBF:
                def __init__(self, bf: BloomFilter):
                    self._bf = bf
                def contains_batch(self, keys: Iterable[str]) -> List[bool]:
                    # Adapt this based on your BloomFilter's query method
                    return [key in self._bf for key in keys]
                def get_memory_usage_bytes(self) -> float:
                    # Adapt based on your BloomFilter's memory reporting
                    if hasattr(self._bf, 'calc_memory_usage_bits'):
                        return self._bf.calc_memory_usage_bits() / 8.0
                    elif hasattr(self._bf, 'num_bits'):
                         return self._bf.num_bits / 8.0
                    else: return 0.0 # Or raise error/warning

            test_subject = WrappedBF(bf_instance)
            experiment_name_suffix = f"_Cap{bf_capacity}_Err{bf_error_rate:.1E}"

        else:
            # Should be caught by argparse choices, but defensively check
            raise ValueError(f"Unhandled structure type: {stype}")

        # Add common memory reporting method if missing (example placeholder)
        if not hasattr(test_subject, 'get_memory_usage_bytes'):
             logging.warning(f"Structure type {stype} doesn't have 'get_memory_usage_bytes'. Adding default.")
             def get_mem_bytes_default(self): return 0.0
             setattr(test_subject.__class__, 'get_memory_usage_bytes', get_mem_bytes_default)

    except NameError as e:
         print(f"Error: A required class (e.g., PLBF, LBF, BF) not found. Import error? Details: {e}")
         exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during {stype} construction: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    construction_end_time = time.time()
    print(f"{stype} construction took: {construction_end_time - construction_start_time:.4f} seconds.")

    # --- 5. Define Experiment Configuration ---
    experiment_name = f"{stype}{experiment_name_suffix}"
    config = ExperimentConfig(
        test_subject=test_subject,
        experiment_name=experiment_name,
        positive_samples=positive_sample, # Used by runner for batch creation
        negative_samples_test=neg_final_test, # Used by runner for batch creation & FPR calc
        device=DEVICE,
        batch_sizes=BATCH_SIZES,
        repetitions=REPETITIONS,
        warmup_reps=WARMUP_REPS,
        target_fpr=TARGET_FPR, # Store the target/configured FPR for reference
        # Ensure the query method name is consistent or configurable
        query_method_name="contains_batch"
    )

    # --- 6. Create and Run Experiment Runner ---
    all_results: List[ExperimentResult] = []
    print(f"\nStarting experiment run for: {experiment_name}")
    try:
        runner = ExperimentRunner(config)
        results_list = runner.run()
        all_results.extend(results_list)
        logging.info(f"===== Experiment {experiment_name} Complete =====")
    except (ValueError, AttributeError, TimerError, RuntimeError) as e:
        logging.error(f"!!!!! Experiment {experiment_name} Failed: {e} !!!!!", exc_info=True)
    except Exception as e:
         logging.error(f"!!!!! Experiment {experiment_name} Failed with unexpected error: {e} !!!!!", exc_info=True)


    # --- 7. Process and Display Results ---
    print("\n===== Experiment Results Summary =====")
    # (Keep the result processing and display logic as before)
    if all_results:
        # Display results for each batch size
        for res in all_results:
            print(f"--- Batch Size: {res.batch_size} ---")
            print(f"  Config Name: {res.config_name}")
            print(f"  Avg Latency (Total): {res.avg_total_latency_ns / NS_PER_S:.6f} s")
            print(f"  Avg Latency (Per Item): {res.avg_per_item_latency_ns:.2f} ns")
            # ... (print other latency stats) ...

        # Display overall metrics
        first_result = all_results[0]
        print("\n--- Overall Metrics ---")
        if first_result.measured_fpr is not None:
             print(f"  Measured FPR on test set: {first_result.measured_fpr:.6f}")
        else:
             print("  Measured FPR: Not calculated")
        if config.target_fpr is not None:
             print(f"  Target/Configured FPR: {config.target_fpr:.6f}")
        else:
             print(f"  Target/Configured FPR: N/A")


        if first_result.memory_usage_bytes is not None and first_result.memory_usage_bytes > 0:
            mem_bytes = first_result.memory_usage_bytes
            mem_kib = mem_bytes / 1024.0
            mem_mib = mem_kib / 1024.0
            if mem_mib >= 1.0:
                print(f"  Memory Usage (Structure): {mem_mib:.2f} MiB ({mem_bytes:.0f} bytes)")
            else:
                print(f"  Memory Usage (Structure): {mem_kib:.2f} KiB ({mem_bytes:.0f} bytes)")
            # Optional: Add predictor size if applicable and model exists
            if predictor_model and stype != "BF":
                 predictor_size_bytes = deep_sizeof(predictor_model) # Approximation
                 print(f"  Estimated Predictor Size: {predictor_size_bytes / (1024*1024):.2f} MiB")
                 print(f"  Total Estimated Size: {(mem_bytes + predictor_size_bytes) / (1024*1024):.2f} MiB")
        else:
            print("  Memory Usage: Not reported or zero")

        # Optional: Save results
        # ... (saving logic as before, maybe include structure type) ...
        # results_df['structure_type'] = stype
        # results_df.to_csv(f"{experiment_name}_results.csv", index=False)

    else:
        print("No results were collected from the experiment run.")

    print("------------------------------------")


if __name__ == "__main__":
    main()