"""
models.py: Defines predictor factory functions extracted from experiments.py.
"""
from __future__ import annotations

import time
import os
import hashlib
import urllib.parse
import re
import math
from typing import Iterable, List, Tuple, Optional, Dict, Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

if TYPE_CHECKING:
    from typing import TypedDict, Callable

    class PredictorTimings(TypedDict):
        data_movement_time: Optional[float]
        inference_time: float

    Predictor = Callable[[Iterable[str]], Tuple[List[float], PredictorTimings]]


def make_logistic_regression_predictor(
    X_train: Iterable[str], y_train: Iterable[float], device_hint: str = "cpu"
) -> Predictor:
    """Train a logistic regression model with text hashing and return a predictor."""
    vectorizer = HashingVectorizer(n_features=4096, alternate_sign=False)
    model = LogisticRegression(
        solver="liblinear", random_state=42, class_weight="balanced"
    )
    pipe = Pipeline([("vectorizer", vectorizer), ("classifier", model)])
    start_time = time.time()
    pipe.fit(X_train, y_train)
    duration = time.time() - start_time
    print(f"[Model] LogisticRegression training took {duration:.2f}s")

    def predictor(keys: Iterable[str]) -> Tuple[List[float], PredictorTimings]:
        timings: PredictorTimings = {"data_movement_time": None, "inference_time": 0.0}
        keys_list = list(map(str, keys))
        if not keys_list:
            return [], timings
        start = time.perf_counter()
        probas = pipe.predict_proba(keys_list)
        timings["inference_time"] = time.perf_counter() - start
        return probas[:, 1].astype(float).tolist(), timings

    return predictor


def make_pytorch_mlp_predictor(
    X_train: Iterable[str], y_train: Iterable[float], device_hint: str = "cpu"
) -> Predictor:
    """Train or load a simple PyTorch MLP model and return a predictor."""
    def simple_hash_vec(keys: Iterable[str], n_features: int = 4096) -> np.ndarray:
        arr = np.zeros((len(keys), n_features), dtype=np.float32)
        for i, k in enumerate(keys):
            digest = hashlib.sha256(k.encode()).digest()
            idx = int.from_bytes(digest, "big") % n_features
            arr[i, idx] = 1.0
        return arr

    X_np = simple_hash_vec(X_train)
    y_np = np.array(y_train, dtype=np.float32)
    device = (
        torch.device("cuda" if device_hint == "cuda" and torch.cuda.is_available() else "cpu")
    )

    class MLP(nn.Module):
        def __init__(self, n_features: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(n_features, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(self.relu(self.fc1(x)))

    n_features = 4096
    cache_dir = "_models"
    os.makedirs(cache_dir, exist_ok=True)

    def get_cache_key(
        X_arr: np.ndarray, y_arr: np.ndarray, n_feats: int
    ) -> str:
        # Fast metadata-based key: use array shapes, dtypes, and feature count
        meta = f"{X_arr.shape}_{y_arr.shape}_{n_feats}_{X_arr.dtype}_{y_arr.dtype}"
        return hashlib.sha256(meta.encode()).hexdigest()

    cache_key = get_cache_key(X_np, y_np, n_features)
    cache_path = os.path.join(cache_dir, f"mlp_{cache_key}.pt")
    model = MLP(n_features).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if os.path.exists(cache_path):
        print(f"[Model] Loading cached MLP from {cache_path}")
        checkpoint = torch.load(cache_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        X_tensor = torch.tensor(X_np, dtype=torch.float32, device="cpu")
        y_tensor = torch.tensor(y_np, dtype=torch.float32, device="cpu").view(-1, 1)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=1024, shuffle=True)
        epochs = 10
        print(f"[Model] Training MLP on {device} for {epochs} epochs")
        start_time = time.time()
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
        print(f"[Model] MLP training took {time.time() - start_time:.2f}s")
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, cache_path
        )

    def predictor(keys: Iterable[str]) -> Tuple[List[float], PredictorTimings]:
        timings: PredictorTimings = {"data_movement_time": None, "inference_time": 0.0}
        keys_list = list(map(str, keys))
        if not keys_list:
            return [], timings
        feat_start = time.perf_counter()
        X_keys = simple_hash_vec(keys_list)
        X_cpu = torch.tensor(X_keys, dtype=torch.float32, device="cpu")
        feat_time = time.perf_counter() - feat_start
        timings["inference_time"] = feat_time
        if device.type == "cuda":
            dm_start = time.perf_counter()
            X_gpu = X_cpu.to(device)
            timings["data_movement_time"] = time.perf_counter() - dm_start
            inf_start = time.perf_counter()
            with torch.no_grad():
                logits = model(X_gpu)
                probs = torch.sigmoid(logits)
            timings["inference_time"] += time.perf_counter() - inf_start
            dm_start = time.perf_counter()
            probs = probs.cpu()
            timings["data_movement_time"] += time.perf_counter() - dm_start
            probs = probs.numpy().flatten()
        else:
            inf_start = time.perf_counter()
            with torch.no_grad():
                logits = model(X_cpu)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
            timings["inference_time"] += time.perf_counter() - inf_start
        return probs.tolist(), timings

    return predictor

def make_xgboost_predictor(
    X_train: Iterable[str], y_train: Iterable[float], device_hint: str = "cpu"
) -> Predictor:
    """Train XGBoost with hashing features and return a predictor."""
    vectorizer = HashingVectorizer(n_features=512, alternate_sign=False)
    X_feat = vectorizer.transform(X_train)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=1)
    grid = GridSearchCV(
        xgb,
        {"max_depth": [5, 7, 9], "n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 0.3]},
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
    )
    start = time.perf_counter()
    grid.fit(X_feat, y_train)
    print(f"[Model] XGBoost training took {time.perf_counter() - start:.2f}s")
    model = grid.best_estimator_

    def predictor(keys: Iterable[str]) -> Tuple[List[float], PredictorTimings]:
        timings: PredictorTimings = {"data_movement_time": None, "inference_time": 0.0}
        keys_list = list(map(str, keys))
        if not keys_list:
            return [], timings
        inf_start = time.perf_counter()
        Xk = vectorizer.transform(keys_list)
        probas = model.predict_proba(Xk)
        timings["inference_time"] = time.perf_counter() - inf_start
        return probas[:, 1].astype(float).tolist(), timings

    return predictor

class UrlFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from URL strings into a DataFrame."""

    def fit(self, X: Iterable[str], y: Any = None) -> UrlFeatureExtractor:
        return self

    def transform(self, X: Iterable[str]) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        for url in X:
            parsed = urllib.parse.urlparse(url)
            record = {
                "scheme": parsed.scheme or "",
                "url_length": len(url),
                "hostname_length": len(parsed.hostname) if parsed.hostname else 0,
                "path_length": len(parsed.path),
                "query_length": len(parsed.query),
                "fragment_length": len(parsed.fragment),
                "num_dots_in_hostname": parsed.hostname.count(".") if parsed.hostname else 0,
                "num_slashes_in_path": parsed.path.count("/") if parsed.path else 0,
                "num_question_marks": url.count("?"),
                "num_equals_signs": url.count("="),
                "num_ampersands": url.count("&"),
                "num_hyphens": url.count("-"),
                "num_at_signs": url.count("@"),
                "has_port_in_hostname": bool(re.search(r":\d+", parsed.netloc)),
                "has_ip_in_hostname": bool(self._is_ip_address(parsed.hostname)) if parsed.hostname else False,
                "is_https": parsed.scheme == "https",
                "hostname": parsed.hostname or "",
                "path": parsed.path or "",
                "query": parsed.query or "",
                "has_suspicious_keywords": self._has_suspicious_keywords(url),
                "hostname_entropy": self._calculate_entropy(parsed.hostname) if parsed.hostname else 0.0,
                "tld": self._extract_tld(parsed.hostname) if parsed.hostname else "none",
            }
            records.append(record)
        return pd.DataFrame(records)

    @staticmethod
    def _is_ip_address(hostname: str) -> bool:
        ipv4 = r"^(\d{1,3}\.){3}\d{1,3}$"
        ipv6 = r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$"
        return bool(re.match(ipv4, hostname)) or bool(re.match(ipv6, hostname))

    @staticmethod
    def _has_suspicious_keywords(url: str) -> bool:
        keywords = [
            "login", "bank", "account", "update", "verify", "confirm",
            "secure", "webscr", "cmd=", "phish", "malware", "virus",
        ]
        return any(kw in url.lower() for kw in keywords)

    @staticmethod
    def _calculate_entropy(s: str) -> float:
        if not s:
            return 0.0
        probs = [s.count(c) / len(s) for c in set(s)]
        return -sum(p * math.log2(p) for p in probs)

    @staticmethod
    def _extract_tld(hostname: str) -> str:
        parts = hostname.split(".")
        return parts[-1] if len(parts) > 1 else "none"


def make_xgboost_predictor_with_features(
    X_train: Iterable[str], y_train: Iterable[float], device_hint: str = "cpu"
) -> Predictor:
    """Train XGBoost with URL feature extraction pipeline."""
    numerical = [
        "url_length", "hostname_length", "path_length", "query_length",
        "fragment_length", "num_dots_in_hostname", "num_slashes_in_path",
        "num_question_marks", "num_equals_signs", "num_ampersands",
        "num_hyphens", "num_at_signs", "hostname_entropy",
    ]
    categorical = ["scheme", "tld"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("hostname_text", TfidfVectorizer(ngram_range=(1,3), max_features=1000, analyzer="char"), "hostname"),
            ("path_text", TfidfVectorizer(ngram_range=(1,2), max_features=1000, analyzer="char"), "path"),
            ("query_text", TfidfVectorizer(ngram_range=(1,2), max_features=500, analyzer="char"), "query"),
        ], remainder="drop",
    )
    pipeline = Pipeline([
        ("extractor", UrlFeatureExtractor()),
        ("preprocessor", preprocess),
        ("xgb", XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", n_jobs=-1,
            tree_method="hist" if device_hint == "gpu" else "auto"
        )),
    ])
    grid = GridSearchCV(
        pipeline,
        {
            "xgb__max_depth": [5, 7],
            "xgb__n_estimators": [50, 100],
            "xgb__learning_rate": [0.05, 0.1],
        },
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
    )
    start_time = time.perf_counter()
    grid.fit(X_train, y_train)
    print(f"[Model] XGBoost (with features) training took {time.perf_counter() - start_time:.2f}s")
    model = grid.best_estimator_

    def predictor(keys: Iterable[str]) -> Tuple[List[float], PredictorTimings]:
        timings: PredictorTimings = {"data_movement_time": None, "inference_time": 0.0}
        keys_list = list(map(str, keys))
        if not keys_list:
            return [], timings
        inf_start = time.perf_counter()
        probas = model.predict_proba(keys_list)
        timings["inference_time"] = time.perf_counter() - inf_start
        return [float(p) for p in probas[:, 1]], timings

    return predictor
