"""benchmark.py
A lightweight calibration‑and‑validation utility for Bloom filters (BF),
Learned Bloom Filters (LBF) and Partitioned LBFs (PLBF).

The script samples a small key set, builds (or loads) a predictor model if
needed, calibrates simple latency primitives (memory copy, model inference,
lookup), fits affine / power‑law regressions, persists the fits to `_bench/`,
and finally validates the symbolic latency model on a handful of batch sizes.

Run:
    python benchmark.py STRUCTURE data.txt --bf-bytes 16384 --model-idx 1 \
           --device cuda --sample 4000

Requirements (assumed available in PYTHONPATH):
    * bloomfilter.BloomFilter  (see https://github.com/MKJM2/cpp-bf)
    * fast_plbf.FastPLBF      
    * timers.py providing CpuTimer and CudaTimer
    * models.py providing predictor factories
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, Iterable, List, Tuple

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
BATCH_SIZES: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
BENCH_DIR = Path("_bench")
BENCH_DIR.mkdir(exist_ok=True)

RUNS = 30  # number of runs for timing

@dataclass
class PrimitiveFit:
    """Stores a fitted latency primitive."""

    primitive: str  # T_mem | T_infer | T_lookup
    device: str  # cpu | cuda
    params: Dict[str, float]  # regression coefficients
    form: str  # affine | powerlaw

    def predict(self, x: float) -> float:
        if self.form == "affine":
            return self.params["intercept"] + self.params["coef"] * x
        # power‑law:  y = exp(a) * x ** b   with params {a,b}
        return float(np.exp(self.params["a"]) * (x ** self.params["b"]))

# -----------------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------------

def load_dataset(path: str, sample: int = 4000) -> List[str]:
    """Load a newline‑delimited text file and return `sample` keys (shuffled)."""
    with open(path, "r", encoding="utf-8") as fh:
        keys = [line.rstrip("\n") for line in fh]
    random.shuffle(keys)
    return keys[:sample]


# -----------------------------------------------------------------------------------
# Primitive measurement routines
# -----------------------------------------------------------------------------------

def measure_T_mem(batch_sizes: List[int]) -> pd.DataFrame:
    """Measure host <> device round‑trip copy time on CUDA for synthetic payloads."""
    import torch  # local import to avoid mandatory torch on CPU‑only systems

    results: List[Dict[str, float]] = []
    device = torch.device("cuda")
    for B in batch_sizes:
        x = torch.random((B, 4096), dtype=torch.float32)
        with CudaTimer(device) as cuda_timer:
            y = x.to(device)
            _ = y.to("cpu")
        # CudaTimer auto‑recorded elapsed_ns
        elapsed_ns = cuda_timer.get_elapsed_ns()  # type: ignore
        results.append({"B": B, "time_ns": elapsed_ns, "device": "cuda"})
    return pd.DataFrame(results)


def measure_T_infer(predictor: Predictor, batch_sizes: List[int], device: str) -> pd.DataFrame:
    """Measure predictor inference time (+data movement) for given batch sizes."""
    dummy_keys = [f"key{i}" for i in range(max(batch_sizes) + 1)]
    rows: List[Dict[str, float]] = []
    for B in batch_sizes:
        keys_slice = dummy_keys[:B]
        _, timings = predictor(keys_slice)
        infer = timings.get("inference_time", 0.0)
        dm = timings.get("data_movement_time", 0.0) or 0.0
        rows.append({"B": B, "infer_s": infer + dm, "device": device})
    return pd.DataFrame(rows)


def measure_T_lookup_fixed(bf_bits: int, batch_sizes: List[int]) -> pd.DataFrame:
    """Measure lookup latency on CPU for a Bloom filter of fixed size.

    Parameters
    ----------
    bf_bits : int
        Bloom filter size in bits.
    batch_sizes : List[int]
        Batch sizes to measure.
    """
    # capacity parameter in library is *expected item count*, not bits; we cheat by
    # using bits as capacity to keep latency roughly proportional.
    bf = BloomFilter(bf_bits, 0.01)
    rows: List[Dict[str, float]] = []
    for B in batch_sizes:
        for _ in range(RUNS):
            keys = [f"dummy{i}" for i in range(B)]
            bf.add(keys[0])
            with CpuTimer() as cpu_timer:
                for i in range(B):
                    _ = keys[i] in bf
            elapsed_ns = cpu_timer.get_elapsed_ns()  # type: ignore
            rows.append({"B": B, "time_ns": elapsed_ns, "device": "cpu"})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------------
# Regression helpers
# -----------------------------------------------------------------------------------

def fit_affine(df: pd.DataFrame, xcol: str, ycol: str) -> Tuple[Dict[str, float], float]:
    X = df[[xcol]].values
    y = df[ycol].values
    mdl = HuberRegressor().fit(X, y)
    return {"intercept": float(mdl.intercept_), "coef": float(mdl.coef_[0])}, r2_score(y, mdl.predict(X))


def fit_powerlaw(df: pd.DataFrame, xcol: str, ycol: str) -> Tuple[Dict[str, float], float]:
    df = df[df[xcol] > 0]
    X = np.log(df[[xcol]].values)
    y = np.log(df[ycol].values)
    mdl = HuberRegressor().fit(X, y)
    a = float(mdl.intercept_)
    b = float(mdl.coef_[0])
    preds = mdl.predict(X)
    return {"a": a, "b": b}, r2_score(y, preds)


def choose_best_fit(df: pd.DataFrame, xcol: str, ycol: str) -> PrimitiveFit:
    """Return PrimitiveFit choosing affine if r² ≥ .98 else power‑law."""
    affine_params, r2_aff = fit_affine(df, xcol, ycol)
    if r2_aff >= 0.98:
        return PrimitiveFit(ycol, df["device"].iloc[0], affine_params, "affine")
    pw_params, _ = fit_powerlaw(df, xcol, ycol)
    return PrimitiveFit(ycol, df["device"].iloc[0], pw_params, "powerlaw")


# -----------------------------------------------------------------------------------
# Calibration orchestrator
# -----------------------------------------------------------------------------------

def calibrate(
    structure: str,
    keys_sample: List[str],
    bf_bytes: int,
    predictor_factory: Callable[..., Predictor] | None,
    device: str,
) -> Dict[str, PrimitiveFit]:
    """Measure primitives and fit regression models.

    For BF we skip predictor / T_infer; for CPU‑only we skip CUDA T_mem.
    """
    fits: Dict[str, PrimitiveFit] = {}

    # --- T_mem (CUDA only) ---------------------------------------------------------
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        df_mem = measure_T_mem(BATCH_SIZES)
        fits["T_mem_cuda"] = choose_best_fit(df_mem, "B", "time_ns")

    # --- T_lookup (CPU, fixed filter size) ----------------------------------------
    df_lu = measure_T_lookup_fixed(bf_bytes * 8, BATCH_SIZES)
    fits["T_lookup_cpu"] = choose_best_fit(df_lu, "B", "time_ns")

    # --- T_infer (if predictor supplied) ------------------------------------------
    if predictor_factory is not None:
        # Simple supervised task: label 1 for first half, 0 for second.
        half = len(keys_sample) // 2
        X_train = keys_sample
        y_train = [1] * half + [0] * (len(keys_sample) - half)
        predictor = predictor_factory(X_train, y_train, device)
        df_inf = measure_T_infer(predictor, BATCH_SIZES, device)
        fits[f"T_infer_{device}"] = choose_best_fit(df_inf, "B", "infer_s")

    # persist fits
    joblib.dump(fits, BENCH_DIR / "fits.pkl")
    return fits


# -----------------------------------------------------------------------------------
# Latency equations using fitted primitives
# -----------------------------------------------------------------------------------

def latency_lookup(B: int, fits: Dict[str, PrimitiveFit]) -> float:
    return fits["T_lookup_cpu"].predict(B)


def latency_infer(B: int, fits: Dict[str, PrimitiveFit], device: str) -> float:
    key = f"T_infer_{device}"
    if key not in fits:
        return 0.0
    return fits[key].predict(B)


def latency_BF(B: int, fits: Dict[str, PrimitiveFit]) -> float:
    return latency_lookup(B, fits)


def latency_LBF(B: int, fits: Dict[str, PrimitiveFit], P_backup: float = 0.5) -> float:
    t_inf = latency_infer(B, fits, device="cpu")
    t_lu = latency_lookup(1, fits)  # scalar lookup cost (approx.)
    return t_inf + P_backup * t_lu


def latency_PLBF(
    B: int,
    fits: Dict[str, PrimitiveFit],
    k: int = 4,
    P_vec: List[float] | None = None,
) -> float:
    """Dummy PLBF latency with uniform routing if P_vec is None."""
    if P_vec is None:
        P_vec = [1.0 / k] * k
    t_inf = latency_infer(B, fits, device="cpu")
    t_lu = latency_lookup(1, fits)
    return t_inf + sum(p * t_lu for p in P_vec)


# -----------------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------------

def validate(
    structure: str,
    fits: Dict[str, PrimitiveFit],
    bf_bytes: int,
    keys_sample: List[str],
):
    """Run a timing sweep and print prediction vs reality."""

    # Build structures for real timing
    if structure == "BF":
        bf = BloomFilter(bf_bytes * 8, 0.01)
        for k in keys_sample[: len(keys_sample) // 2]:
            bf.add(k)
        def contains_batch(batch: List[str]):
            return [key in bf for key in batch]
    elif structure == "LBF":
        # Minimal stub: just reuse BF latency for demonstration
        bf = BloomFilter(bf_bytes * 8, 0.01)
        contains_batch = lambda batch: [key in bf for key in batch]  # type: ignore
    else:  # PLBF
        plbf = FastPLBF()  # type: ignore  # assume has contains_batch
        contains_batch = plbf.contains_batch  # type: ignore

    print("B\tpred_ns\treal_ns\tAPE%")
    apes: List[float] = []
    for B in [1, 64, 1024, 8192]:
        batch = keys_sample[:B]
        # --- real timing ---
        with CpuTimer() as cpu_timer:
            _ = contains_batch(batch)
        real_ns = cpu_timer.get_elapsed_ns()  # type: ignore
        # --- predicted ---
        if structure == "BF":
            pred_ns = latency_BF(B, fits)
        elif structure == "LBF":
            pred_ns = latency_LBF(B, fits)
        else:
            pred_ns = latency_PLBF(B, fits)
        ape = abs(pred_ns - real_ns) / real_ns * 100.0
        apes.append(ape)
        print(f"{B}\t{pred_ns:.0f}\t{real_ns:.0f}\t{ape:.1f}")
    print(f"MAPE: {mean(apes):.1f}%")


# -----------------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate and validate latency models for BF/LBF/PLBF.")
    parser.add_argument("structure", choices=["BF", "LBF", "PLBF"], help="Data structure to benchmark.")
    parser.add_argument("data_path", help="Path to newline‑delimited key file.")
    parser.add_argument("--bf-bytes", type=int, required=True, help="Bloom filter size in bytes (fixed).")
    parser.add_argument("--model-idx", type=int, default=0, help="Predictor model index (for LBF/PLBF).")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device hint for predictor.")
    parser.add_argument("--sample", type=int, default=4000, help="Number of keys to sample from dataset.")

    args = parser.parse_args()

    keys = load_dataset(args.data_path, args.sample)
    structure = args.structure

    # Determine predictor factory if needed
    predictor_factory = None
    if structure in ("LBF", "PLBF"):
        try:
            predictor_factory = PREDICTOR_MODEL_FACTORIES[args.model_idx]
            print(f"Using predictor: {PREDICTOR_MODEL_NAMES[args.model_idx]}")
        except IndexError:
            sys.exit("Invalid --model-idx")

    fits_path = BENCH_DIR / "fits.pkl"
    if fits_path.exists():
        fits: Dict[str, PrimitiveFit] = joblib.load(fits_path)
        print("Loaded cached primitive fits.")
    else:
        fits = calibrate(structure, keys, args.bf_bytes, predictor_factory, args.device)
        print("Calibration done and cached.")

    validate(structure, fits, args.bf_bytes, keys)


if __name__ == "__main__":
    main()
