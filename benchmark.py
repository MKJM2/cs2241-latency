from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score

# --- third‑party filter structures -------------------------------------------------
from bloomfilter import BloomFilter  # type: ignore
from fast_plbf import FastPLBF  # type: ignore

# --- predictor model factories -----------------------------------------------------
from models import (  # type: ignore
    make_logistic_regression_predictor,
    make_pytorch_mlp_predictor,
    make_xgboost_predictor,
    make_xgboost_predictor_with_features,
)

PREDICTOR_MODEL_FACTORIES: List[Callable[..., "Predictor"]] = [
    make_logistic_regression_predictor,
    make_pytorch_mlp_predictor,
    make_xgboost_predictor,
    make_xgboost_predictor_with_features,
]

PREDICTOR_MODEL_NAMES = [
    "LogisticRegression (sklearn)",
    "MLP (PyTorch)",
    "XGBoost (CPU)",
    "XGBoost (with features)",
]

# --- timers ------------------------------------------------------------------------
from timers import CpuTimer, CudaTimer  # type: ignore
from timers import TimerError  # type: ignore

# -----------------------------------------------------------------------------------
TimedPrediction = Tuple[List[float], Dict[str, float]]
Predictor = Callable[[Iterable[str]], TimedPrediction]

# -----------------------------------------------------------------------------------
BATCH_SIZES: List[int] = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
]
BENCH_DIR = Path("_bench")
BENCH_DIR.mkdir(exist_ok=True)

# Number of timing repetitions for each primitive measurement point
CALIBRATION_RUNS = 30
# Number of keys to use for predictor training if not using full sample
# Set to None to use all keys_sample for predictor training
PREDICTOR_TRAINING_SUBSET_SIZE = 10000

# For PLBFs (TODO: make this overridable with args.parser)
PLBF_K_PARTITIONS = 6  

# The following account for Python overheads and such within
# this (a bit buggy) benchmarking script. They are currently
# hardcoded on my machine. For future work, we will make this
# not hardcoded and more robust in general.
PREDICTION_SCALING_FACTOR_BF = 1.0
PREDICTION_SCALING_FACTOR_PLBF = 1.0

@dataclass
class PrimitiveFit:
    """Stores a fitted latency primitive."""

    primitive: str  # T_mem | T_infer | T_lookup
    device: str  # cpu | cuda
    params: Dict[str, float]  # regression coefficients
    form: str  # affine | powerlaw
    r2: float # R-squared of the fit

    def predict(self, x: float) -> float:
        if x == 0: # Avoid issues with power law at x=0
            if self.form == "affine":
                return self.params["intercept"]
            return 0.0 # Or handle as an error/special case for power law

        if self.form == "affine":
            return self.params["intercept"] + self.params["coef"] * x
        # power‑law:  y = exp(a) * x ** b   with params {a,b}
        return float(np.exp(self.params["a"]) * (x**self.params["b"]))


# -----------------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------------

def generate_random_key(avg_len: int = 15) -> str:
    """Generates a random alphanumeric string key."""
    return str(uuid.uuid4()) # Simple, unique random strings


def load_dataset(path: str, sample_size: int = 4000) -> List[str]:
    """Load a newline‑delimited text file and return `sample_size` keys (shuffled)."""
    with open(path, "r", encoding="utf-8") as fh:
        keys = [line.rstrip("\n") for line in fh if line.strip()]
    random.shuffle(keys)
    return keys[:sample_size]


# -----------------------------------------------------------------------------------
# Primitive measurement routines
# -----------------------------------------------------------------------------------

def measure_T_mem(
    batch_sizes: List[int], runs_per_batch: int
) -> pd.DataFrame:
    """Measure host <> device round‑trip copy time on CUDA for synthetic payloads."""
    import torch  # local import to avoid mandatory torch on CPU‑only systems

    results: List[Dict[str, float]] = []
    if not torch.cuda.is_available():
        print("CUDA not available, skipping T_mem measurement.")
        return pd.DataFrame(results)

    device = torch.device("cuda")
    for B in batch_sizes:
        payload = torch.randn(
            (B, 4096), dtype=torch.float32
        )  # Fixed payload feature size
        timings_ns = []
        for _ in range(runs_per_batch):
            with CudaTimer(device) as cuda_timer:
                payload_gpu = payload.to(device)
                _ = payload_gpu.to("cpu")
            timings_ns.append(cuda_timer.get_elapsed_ns() * PREDICTION_SCALING_FACTOR_BF) # type: ignore
        results.append(
            {"B": B, "time_ns": median(timings_ns), "device": "cuda"}
        )
    return pd.DataFrame(results)


def measure_T_infer(
    predictor: Predictor,
    batch_sizes: List[int],
    device: str,
    runs_per_batch: int,
) -> pd.DataFrame:
    """Measure predictor inference time (+data movement) for given batch sizes."""
    max_B = max(batch_sizes) if batch_sizes else 0
    # Generate more diverse dummy keys for inference measurement
    dummy_keys = [generate_random_key() for _ in range(max_B + 1)]
    rows: List[Dict[str, float]] = []

    for B in batch_sizes:
        if B == 0: continue
        keys_slice = dummy_keys[:B]
        batch_timings_s = []
        for _ in range(runs_per_batch):
            _, timings = predictor(keys_slice)
            infer_s = timings.get("inference_time", 0.0)
            dm_s = timings.get("data_movement_time", 0.0) or 0.0
            batch_timings_s.append(infer_s + dm_s)
        scaled_median_time_s = median(batch_timings_s) * PREDICTION_SCALING_FACTOR_BF
        rows.append(
            {"B": B, "infer_s": median(batch_timings_s), "device": device}
        )
    return pd.DataFrame(rows)


def measure_T_lookup_fixed(
    bf_capacity_param: int, # This is effectively bf_bytes * 8 from args
    fpr: float,
    batch_sizes: List[int],
    num_items_to_populate: int,
    runs_per_batch: int,
) -> pd.DataFrame:
    """Measure lookup latency on CPU for a Bloom filter of fixed configuration,
    populated with a specific number of items. Lookups are for present keys.
    """
    bf = BloomFilter(bf_capacity_param, fpr)
    
    # Populate the filter with unique random keys
    # These keys will be present in the filter during timing.
    population_keys: List[str] = []
    if num_items_to_populate > 0:
        population_keys = [
            f"calib_present_key_{i}" for i in range(num_items_to_populate)
        ]
        for key in population_keys:
            bf.add(key)
    
    # If no keys populated, or not enough for max batch size, use placeholder keys
    # This scenario (lookup on empty/nearly empty BF) should be handled carefully
    # or avoided if not representative. For "present key" lookup, we need populated keys.
    if not population_keys: # Cannot measure present key lookup if BF is empty
        if num_items_to_populate > 0: # Should not happen if logic is correct
             print("Warning: Population keys empty despite num_items_to_populate > 0")
        # Fallback: use dummy keys (will be absent, measures absent key lookup)
        # This changes the nature of the measurement.
        # For now, we assume population_keys will be sufficient for B.
        # If B > len(population_keys), we cycle.
        print("Warning: BF population is empty for T_lookup. Measurement might be misleading.")
        # To prevent crash, fill with some dummy keys for lookup if needed
        # This part of logic needs to ensure lookup_key_pool is never empty if B > 0
        if max(batch_sizes) > 0 and not population_keys:
            population_keys = [f"dummy_absent_key_{i}" for i in range(max(batch_sizes))]


    rows: List[Dict[str, float]] = []
    max_b_val = max(batch_sizes) if batch_sizes else 0

    # Prepare lookup keys: all keys looked up will be from the populated set.
    lookup_key_pool: List[str]
    if population_keys:
        lookup_key_pool = [
            population_keys[i % len(population_keys)] for i in range(max_b_val)
        ]
    else: # Should ideally not be reached if we want to measure present key lookups
        lookup_key_pool = [f"placeholder_key_{i}" for i in range(max_b_val)]


    for B in batch_sizes:
        if B == 0: continue
        current_lookup_keys = lookup_key_pool[:B]
        if not current_lookup_keys: continue # Should not happen if B > 0

        batch_timings_ns = []
        for _ in range(runs_per_batch):
            with CpuTimer() as cpu_timer:
                for key_to_lookup in current_lookup_keys:
                    _ = key_to_lookup in bf
            batch_timings_ns.append(cpu_timer.get_elapsed_ns() * PREDICTION_SCALING_FACTOR_BF) # type: ignore
        rows.append(
            {"B": B, "time_ns": median(batch_timings_ns), "device": "cpu"}
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------------
# Regression helpers
# -----------------------------------------------------------------------------------

def fit_affine(
    df: pd.DataFrame, xcol: str, ycol: str
) -> Tuple[Dict[str, float], float]:
    X = df[[xcol]].values
    y = df[ycol].values
    if len(X) < 2: # Not enough data to fit
        return {"intercept": 0.0, "coef": 0.0}, 0.0
    mdl = HuberRegressor().fit(X, y)
    return {
        "intercept": float(mdl.intercept_),
        "coef": float(mdl.coef_[0]),
    }, r2_score(y, mdl.predict(X))


def fit_powerlaw(
    df: pd.DataFrame, xcol: str, ycol: str
) -> Tuple[Dict[str, float], float]:
    df_filtered = df[(df[xcol] > 0) & (df[ycol] > 0)]
    if len(df_filtered) < 2: # Not enough data to fit
        return {"a": 0.0, "b": 0.0}, 0.0
        
    X = np.log(df_filtered[[xcol]].values)
    y = np.log(df_filtered[ycol].values)
    mdl = HuberRegressor().fit(X, y)
    a = float(mdl.intercept_)
    b = float(mdl.coef_[0])
    # R2 on the log-transformed data
    return {"a": a, "b": b}, r2_score(y, mdl.predict(X))


def choose_best_fit(
    df: pd.DataFrame, xcol: str, ycol: str, primitive_name: str
) -> PrimitiveFit:
    """Return PrimitiveFit choosing affine if r² ≥ .95 else power‑law."""
    # Ensure ycol is not in seconds if time_ns is expected (e.g. T_infer)
    ycol_effective = ycol
    if ycol == "infer_s": # Convert T_infer to ns for consistency with T_mem, T_lookup
        df[ycol + "_ns"] = df[ycol] * 1e9
        ycol_effective = ycol + "_ns"

    affine_params, r2_aff = fit_affine(df, xcol, ycol_effective)
    pw_params, r2_pw = fit_powerlaw(df, xcol, ycol_effective)

    device = df["device"].iloc[0] if "device" in df.columns and not df.empty else "cpu"

    # Prefer affine if it's a very good fit, otherwise power law if it's better.
    # This threshold can be tuned.
    if r2_aff >= 0.95 and r2_aff >= r2_pw :
        print(f"Fit {primitive_name} ({device}): Affine (R2={r2_aff:.3f})")
        return PrimitiveFit(primitive_name, device, affine_params, "affine", r2_aff)
    
    # If powerlaw R2 is poor, but better than affine (which was <0.95)
    # still choose powerlaw if it's the best of the two.
    # If both are poor, powerlaw is often a safer bet for extrapolating.
    print(f"Fit {primitive_name} ({device}): PowerLaw (R2_log={r2_pw:.3f}, R2_affine={r2_aff:.3f})")
    return PrimitiveFit(primitive_name, device, pw_params, "powerlaw", r2_pw)


# -----------------------------------------------------------------------------------
# Calibration orchestrator
# -----------------------------------------------------------------------------------

def calibrate(
    keys_sample: List[str],
    bf_bytes_arg: int, # From --bf-bytes
    predictor_factory: Callable[..., Predictor] | None,
    device: str,
    runs_per_batch: int,
) -> Dict[str, PrimitiveFit]:
    fits: Dict[str, PrimitiveFit] = {}
    bf_fpr = 0.01
    # This is the 'expected_elements' parameter for the BloomFilter constructor
    bf_capacity_for_filter_init = bf_bytes_arg * 8

    # --- T_mem (CUDA only) ---------------------------------------------------------
    if device == "cuda":
        df_mem = measure_T_mem(BATCH_SIZES, runs_per_batch)
        if not df_mem.empty:
            fits["T_mem_cuda"] = choose_best_fit(df_mem, "B", "time_ns", "T_mem")

    # --- T_lookup (CPU, fixed filter size) ----------------------------------------
    # Populate BF with half the sample keys, similar to validation phase
    num_items_for_bf_population = len(keys_sample) // 2
    df_lu = measure_T_lookup_fixed(
        bf_capacity_for_filter_init,
        bf_fpr,
        BATCH_SIZES,
        num_items_for_bf_population,
        runs_per_batch,
    )
    if not df_lu.empty:
        fits["T_lookup_cpu"] = choose_best_fit(df_lu, "B", "time_ns", "T_lookup")

    # --- T_infer (if predictor supplied) ------------------------------------------
    if predictor_factory is not None:
        # Use a subset for faster predictor training if PREDICTOR_TRAINING_SUBSET_SIZE is set
        train_keys = keys_sample
        if PREDICTOR_TRAINING_SUBSET_SIZE is not None and PREDICTOR_TRAINING_SUBSET_SIZE < len(keys_sample):
            train_keys = random.sample(keys_sample, PREDICTOR_TRAINING_SUBSET_SIZE)
        
        half = len(train_keys) // 2
        X_train = train_keys
        y_train = [1.0] * half + [0.0] * (len(train_keys) - half)
        
        print(f"Training predictor for T_infer calibration with {len(X_train)} keys...")
        predictor = predictor_factory(X_train, y_train, device)
        print("Predictor training complete.")

        df_inf = measure_T_infer(predictor, BATCH_SIZES, device, runs_per_batch)
        if not df_inf.empty:
            # T_infer is measured in seconds, choose_best_fit will convert to ns
            fits[f"T_infer_{device}"] = choose_best_fit(df_inf, "B", "infer_s", "T_infer")
            
    # Persist fits
    fit_objects_to_persist = {name: fit for name, fit in fits.items() if fit is not None}
    if fit_objects_to_persist:
        joblib.dump(fit_objects_to_persist, BENCH_DIR / "fits.pkl")
        print(f"Primitive fits saved to {BENCH_DIR / 'fits.pkl'}")
        for name, fit in fit_objects_to_persist.items():
            print(f"  {name}: {fit.form}, R2={fit.r2:.3f}, Params={fit.params}")
    else:
        print("No fits were generated.")
    return fit_objects_to_persist


# -----------------------------------------------------------------------------------
# Latency equations using fitted primitives (output in nanoseconds)
# -----------------------------------------------------------------------------------

def latency_lookup_ns(B: int, fits: Dict[str, PrimitiveFit]) -> float:
    fit = fits.get("T_lookup_cpu")
    return fit.predict(B) if fit else 0.0


def latency_infer_ns(
    B: int, fits: Dict[str, PrimitiveFit], device: str
) -> float:
    key = f"T_infer_{device}"
    fit = fits.get(key)
    # T_infer fit is already in ns after choose_best_fit
    return fit.predict(B) if fit else 0.0


def latency_mem_ns(B: int, fits: Dict[str, PrimitiveFit]) -> float:
    # Relevant for CUDA operations if data movement is not part of T_infer
    # but T_infer as measured should include it. This is mostly for completeness.
    fit = fits.get("T_mem_cuda")
    return fit.predict(B) if fit else 0.0


def latency_BF_ns(B: int, fits: Dict[str, PrimitiveFit]) -> float:
    return latency_lookup_ns(B, fits)


def latency_LBF_ns(
    B: int,
    fits: Dict[str, PrimitiveFit],
    device: str,
    P_backup: float = 0.5, # Probability of consulting the backup filter
) -> float:
    t_inf_ns = latency_infer_ns(B, fits, device)
    # Assume backup lookups are batched if B > 1, or individual if B=1.
    # The P_backup * B items go to backup.
    # If t_lu is per item, then P_backup * B * t_lu_scalar.
    # If t_lu is for a batch of size X, then P_backup * t_lu(B).
    # The current T_lookup is for a batch of B items.
    # So, if P_backup fraction of B items go to backup, it's one lookup batch of size P_backup*B.
    num_backup_items = int(P_backup * B)
    if num_backup_items == 0 and P_backup > 0 and B > 0: # ensure at least 1 if any backup
        num_backup_items = 1

    t_lu_backup_ns = latency_lookup_ns(num_backup_items, fits)
    
    # If T_infer on CUDA, add T_mem for initial data to device if not already in T_infer
    # Assuming T_infer from measure_T_infer already includes host-device-host for keys/preds
    # t_mem_component = latency_mem_ns(B, fits) if device == "cuda" else 0.0

    return t_inf_ns + t_lu_backup_ns


def latency_PLBF_ns(
    B: int,
    fits: Dict[str, PrimitiveFit],
    device: str,
    k_partitions: int = 4, # Number of partitions in PLBF
    P_vec: List[float] | None = None, # Routing probabilities to each partition
) -> float:
    """Estimated PLBF latency. Assumes uniform routing if P_vec is None."""
    t_inf_ns = latency_infer_ns(B, fits, device)
    
    if P_vec is None:
        P_vec = [1.0 / k_partitions] * k_partitions
    if abs(sum(P_vec) - 1.0) > 1e-6: # Check sum
        print(f"Warning: P_vec sums to {sum(P_vec)}, not 1.0. Normalizing.")
        total_p = sum(P_vec)
        P_vec = [p / total_p for p in P_vec]


    # Each partition i handles P_vec[i] * B items.
    # Sum of latencies for lookups in each partition.
    total_lu_ns = 0.0
    for p_i in P_vec:
        items_in_partition_i = int(p_i * B)
        if items_in_partition_i > 0:
            total_lu_ns += latency_lookup_ns(items_in_partition_i, fits)
        # If items_in_partition_i is 0, its lookup cost is 0.
        # This assumes T_lookup_cpu is calibrated for the same type of sub-filter.

    return (t_inf_ns + total_lu_ns) * PREDICTION_SCALING_FACTOR_PLBF # type: ignore


# -----------------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------------

def validate(
    structure: str,
    fits: Dict[str, PrimitiveFit],
    bf_bytes_arg: int, # From --bf-bytes
    keys_sample: List[str],
    device: str, # For LBF/PLBF predictor
    predictor_factory: Callable[..., Predictor] | None, # For LBF
    validation_batch_sizes: List[int],
    runs_per_batch_validate: int = 5 # Fewer runs for validation timing
):
    """Run a timing sweep and print prediction vs reality."""
    print(f"\n--- Validating {structure} ---")
    bf_fpr = 0.01
    bf_capacity_for_filter_init = bf_bytes_arg * 8 # expected_elements

    # Keys to populate filters with (half the sample)
    num_populate = len(keys_sample) // 2
    population_keys = keys_sample[:num_populate]
    
    # Keys for lookup (can include keys not in population_keys for misses)
    # Validation will use slices of keys_sample for lookup.

    # --- Setup structure for real timing ---
    timed_contains_batch: Callable[[List[str]], Any]
    predictor_for_lbf: Predictor | None = None

    if structure == "BF":
        bf = BloomFilter(bf_capacity_for_filter_init, bf_fpr)
        for key in population_keys:
            bf.add(key)
        def bf_contains_batch(batch: List[str]):
            return [key in bf for key in batch]
        timed_contains_batch = bf_contains_batch

    elif structure == "LBF":
        if predictor_factory is None:
            print("LBF validation requires a predictor factory. Skipping.")
            return
        
        # Train predictor for LBF (consistent with calibration)
        train_keys_lbf = keys_sample
        if PREDICTOR_TRAINING_SUBSET_SIZE is not None and PREDICTOR_TRAINING_SUBSET_SIZE < len(keys_sample):
            train_keys_lbf = random.sample(keys_sample, PREDICTOR_TRAINING_SUBSET_SIZE)
        
        half_lbf = len(train_keys_lbf) // 2
        X_train_lbf = train_keys_lbf
        y_train_lbf = [1.0] * half_lbf + [0.0] * (len(train_keys_lbf) - half_lbf)
        
        print(f"Training LBF predictor for validation with {len(X_train_lbf)} keys...")
        predictor_for_lbf = predictor_factory(X_train_lbf, y_train_lbf, device)
        print("LBF predictor training complete.")

        # Backup Bloom Filter for LBF
        backup_bf = BloomFilter(bf_capacity_for_filter_init, bf_fpr)
        # Populate backup BF same as standalone BF for this example
        for key in population_keys:
            backup_bf.add(key)

        def lbf_timed_op(batch: List[str]):
            if not batch: return []
            results = [False] * len(batch)
            
            # 1. Predict
            # We need raw scores/probabilities if predictor gives them,
            # or just use its direct output.
            # Assuming predictor output is a list of scores/labels (0 or 1)
            predictions, _ = predictor_for_lbf(batch) # type: ignore

            keys_to_check_in_bf: List[str] = []
            indices_for_bf_results: List[int] = []

            for i, pred_val in enumerate(predictions):
                # Simple LBF: if model predicts positive (e.g., > 0.5 or == 1), check BF
                if pred_val > 0.5: # Threshold for positive prediction
                    keys_to_check_in_bf.append(batch[i])
                    indices_for_bf_results.append(i)
            
            # 2. Lookup in backup BF for those keys
            if keys_to_check_in_bf:
                bf_results = [key in backup_bf for key in keys_to_check_in_bf]
                for original_idx, bf_res in zip(indices_for_bf_results, bf_results):
                    results[original_idx] = bf_res
            return results
        timed_contains_batch = lbf_timed_op

    elif structure == "PLBF":
        if predictor_factory is None:
            print("PLBF validation requires a predictor factory. Skipping.")
            return

        # 1. Train the predictor for PLBF
        train_keys_plbf = keys_sample
        if PREDICTOR_TRAINING_SUBSET_SIZE is not None and PREDICTOR_TRAINING_SUBSET_SIZE < len(keys_sample):
            train_keys_plbf = random.sample(keys_sample, PREDICTOR_TRAINING_SUBSET_SIZE)
        
        half_plbf = len(train_keys_plbf) // 2
        X_train_plbf = train_keys_plbf # These are the keys the predictor is trained on
        y_train_plbf = [1.0] * half_plbf + [0.0] * (len(train_keys_plbf) - half_plbf)
        
        print(f"Training PLBF predictor for validation with {len(X_train_plbf)} keys...")
        predictor_for_plbf = predictor_factory(X_train_plbf, y_train_plbf, device)
        print("PLBF predictor training complete.")

        # 2. Define pos_keys and neg_keys_h_learn for FastPLBF constructor
        # These keys are used by FastPLBF to learn its routing or internal model.
        # It's typical to use the same keys the main predictor was trained on.
        pos_keys_for_plbf_init = X_train_plbf[:half_plbf]
        neg_keys_for_plbf_init = X_train_plbf[half_plbf:]

        # 3. Define other FastPLBF parameters
        plbf_fpr_param = bf_fpr  # F: False Positive Rate for sub-filters
        plbf_k_param = PLBF_K_PARTITIONS # k: Number of partitions

        # N: Capacity for each sub-filter
        if bf_capacity_for_filter_init < plbf_k_param :
            print(f"Warning: Total BF capacity ({bf_capacity_for_filter_init}) is less than k_partitions ({plbf_k_param}). Sub-filter capacity will be low.")
            plbf_subfilter_capacity_param = 1 # Ensure at least 1
        else:
            plbf_subfilter_capacity_param = bf_capacity_for_filter_init // plbf_k_param
        
        if plbf_subfilter_capacity_param == 0:
             print(f"Error: Calculated PLBF sub-filter capacity is 0. bf_capacity_for_filter_init={bf_capacity_for_filter_init}, k_partitions={plbf_k_param}. Skipping PLBF.")
             return


        print(f"Initializing FastPLBF with: k={plbf_k_param}, sub-filter_N={plbf_subfilter_capacity_param}, sub-filter_FPR={plbf_fpr_param}")
        print(f"  pos_keys for PLBF init: {len(pos_keys_for_plbf_init)}, neg_keys: {len(neg_keys_for_plbf_init)}")

        try:
            plbf_instance = FastPLBF(
                predictor=predictor_for_plbf,
                pos_keys=pos_keys_for_plbf_init,
                neg_keys=neg_keys_for_plbf_init,
                F=plbf_fpr_param,
                N=plbf_subfilter_capacity_param,
                k=plbf_k_param,
            )
            timed_contains_batch = plbf_instance.contains_batch
        except Exception as e:
            print(f"Could not initialize or use FastPLBF for validation: {e}. Skipping PLBF.")
            return
    else:
        print(f"Unknown structure {structure} for validation.")
        return

    print("B\tPred_ns\tReal_ns\tAPE%")
    apes: List[float] = []
    preds: List[float] = []
    reals: List[float] = []
    batches: List[int] = []

    for B_val in validation_batch_sizes:
        if B_val == 0: continue
        if B_val > len(keys_sample):
            print(f"Skipping B={B_val}, exceeds sample size {len(keys_sample)}")
            continue
            
        batch = keys_sample[:B_val]
        
        # --- Real timing ---
        real_timings_ns = []
        for _ in range(runs_per_batch_validate):
            with CpuTimer() as cpu_timer:
                _ = timed_contains_batch(batch)
            real_timings_ns.append(cpu_timer.get_elapsed_ns()) # type: ignore
        real_ns = median(real_timings_ns)

        # --- Predicted latency ---
        pred_ns = 0.0
        if structure == "BF":
            pred_ns = latency_BF_ns(B_val, fits)
        elif structure == "LBF":
            # For LBF, P_backup can be estimated from the predictor behavior on a sample
            # or use a fixed value as in the symbolic model.
            # Let's use the default P_backup for the prediction.
            pred_ns = latency_LBF_ns(B_val, fits, device, P_backup=0.5) # Default P_backup
        elif structure == "PLBF":
            # P_vec might be known from PLBF's model or assumed uniform.
            pred_ns = latency_PLBF_ns(B_val, fits, device) # Default k_partitions, uniform P_vec

        if real_ns > 0:
            ape = abs(pred_ns - real_ns) / real_ns * 100.0

            preds.append(pred_ns)
            reals.append(real_ns)
            batches.append(B_val)
            apes.append(ape)
            print(f"{B_val}\t{pred_ns:.0f}\t{real_ns:.0f}\t{ape:.1f}")
        else:
            print(f"{B_val}\t{pred_ns:.0f}\t{real_ns:.0f}\tN/A (real_ns is zero)")

    # --- scatter plot ---
    out_png = f"results/{structure.lower()}_scatter.png"
    plot_scatter(preds, reals, batches, structure, out_png)
            
    if apes:
        print(f"MAPE: {mean(apes):.1f}%")
    else:
        print("No APE values to calculate MAPE.")


# -----------------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate and validate latency models for BF/LBF/PLBF."
    )
    parser.add_argument(
        "structure",
        choices=["BF", "LBF", "PLBF"],
        help="Data structure to benchmark.",
    )
    parser.add_argument(
        "data_path", help="Path to newline‑delimited key file."
    )
    parser.add_argument(
        "--bf-bytes",
        type=int,
        required=True,
        help="Factor for Bloom filter capacity. Capacity = bf-bytes * 8. "
             "This is used as 'expected_elements' for BF init.",
    )
    parser.add_argument(
        "--model-idx",
        type=int,
        default=0,
        help="Predictor model index (for LBF/PLBF).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device hint for predictor.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10000, # Increased default sample size
        help="Number of keys to sample from dataset.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=CALIBRATION_RUNS,
        help="Number of repetitions for timing calibration primitives.",
    )
    parser.add_argument(
        "--force-recalibrate",
        action="store_true",
        help="Force recalibration even if cached fits exist.",
    )
    parser.add_argument(
        "--validation-batches",
        type=str,
        default="1,64,256,512, 1024", # More validation points
        help="Comma-separated list of batch sizes for validation.",
    )

    args = parser.parse_args()

    keys = load_dataset(args.data_path, args.sample)
    if not keys:
        sys.exit(f"No keys loaded from {args.data_path}. Exiting.")
    print(f"Loaded {len(keys)} keys for benchmark (sample size: {args.sample}).")

    # Determine predictor factory if needed
    predictor_factory_selected = None
    if args.structure in ("LBF", "PLBF"):
        if not 0 <= args.model_idx < len(PREDICTOR_MODEL_FACTORIES):
            sys.exit(
                f"Invalid --model-idx {args.model_idx}. "
                f"Max index is {len(PREDICTOR_MODEL_FACTORIES)-1}"
            )
        predictor_factory_selected = PREDICTOR_MODEL_FACTORIES[args.model_idx]
        print(f"Using predictor: {PREDICTOR_MODEL_NAMES[args.model_idx]} for {args.structure}")


    fits_path = BENCH_DIR / "fits.pkl"
    fits: Dict[str, PrimitiveFit]
    if not args.force_recalibrate and fits_path.exists():
        try:
            fits = joblib.load(fits_path)
            print(f"Loaded cached primitive fits from {fits_path}")
            # Basic check if fits look valid
            if not fits or not all(isinstance(f, PrimitiveFit) for f in fits.values()):
                print("Cached fits are invalid. Recalibrating.")
                args.force_recalibrate = True
            else:
                 for name, fit in fits.items():
                    print(f"  Loaded {name}: {fit.form}, R2={fit.r2:.3f}, Params={fit.params}")

        except Exception as e:
            print(f"Error loading cached fits: {e}. Recalibrating.")
            args.force_recalibrate = True

    if args.force_recalibrate or not fits_path.exists():
        if args.force_recalibrate and fits_path.exists():
            print("Forcing recalibration, removing old fits file.")
            os.remove(fits_path)
        print("Starting calibration...")
        fits = calibrate(
            keys,
            args.bf_bytes,
            predictor_factory_selected,
            args.device,
            args.runs,
        )
        if not fits:
            sys.exit("Calibration failed to produce any fits. Exiting.")
        print("Calibration done.")
    
    validation_batches = [int(b.strip()) for b in args.validation_batches.split(',') if b.strip()]
    if not validation_batches:
        sys.exit("No valid batch sizes provided for validation.")

    validate(
        args.structure,
        fits,
        args.bf_bytes,
        keys,
        args.device,
        predictor_factory_selected, # Pass for LBF validation
        validation_batches,
        runs_per_batch_validate=max(1, args.runs // 4) # Fewer runs for validation
    )

def plot_scatter(
    pred_ns: List[float],
    real_ns: List[float],
    batch_sizes: List[int],
    structure: str,
    out_path: str | None = None,
) -> None:
    """
    IEEE‑style scatter plot of predicted vs. measured latency.

    Parameters
    ----------
    pred_ns : List[float]
        Predicted latencies (nanoseconds).
    real_ns : List[float]
        Measured latencies (nanoseconds).
    batch_sizes : List[int]
        Corresponding batch sizes; used as point labels.
    structure : str
        Name of the data structure (BF, LBF, PLBF) – appears in the title.
    out_path : str, optional
        If provided, the figure is saved to this path; otherwise `plt.show()` is called.
    """
    if not (pred_ns and real_ns) or len(pred_ns) != len(real_ns):
        print("plot_scatter: input lists are empty or length‑mismatched.")
        return

    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    plt.style.use(["science", "grid", "ieee"])

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    if structure == "PLBF":
        color = "blue"
    elif structure == "LBF":
        color = "orange"
    else: # BF
        color = "green"
    ax.scatter(real_ns, pred_ns, s=20, marker="o", c=color, alpha=0.7)

    # Ideal y = x reference line
    lo, hi = min(real_ns + pred_ns), max(real_ns + pred_ns)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=0.8, color="black")

    # Small batch‑size labels
    for x, y, b in zip(real_ns, pred_ns, batch_sizes):
        ax.annotate(str(b), (x, y), textcoords="offset points",
                    xytext=(4, -4), fontsize=6)

    ax.set_xlabel("Measured latency (ns)")
    ax.set_ylabel("Predicted latency (ns)")
    # ax.set_title(f"{structure}: Predicted vs. Measured")

    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Scatter plot saved to {out_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
