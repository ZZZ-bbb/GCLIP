"""Controlled, full-batch out-of-core clustering without dimensionality reduction.

All models consume every row on every iteration. K-means is Euclidean Lloyd;
GMMs use MAP component prediction and a common covariance eigenvalue floor.
This module deliberately does not know about target images or their labels.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


@dataclass
class FrozenClusterModel:
    kind: str
    covariance_type: str
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.float64)
        self.means = np.asarray(self.means, dtype=np.float64)
        self.covariances = np.asarray(self.covariances, dtype=np.float64)
        if self.kind not in ("gmm", "kmeans"):
            raise ValueError(self.kind)
        if self.covariance_type not in ("diag", "spherical", "tied", "full", "none"):
            raise ValueError(self.covariance_type)
        if not all(np.isfinite(value).all() for value in (self.weights, self.means, self.covariances)):
            raise FloatingPointError("non-finite model parameters")
        if np.min(self.weights) <= 0 or not np.isclose(self.weights.sum(), 1, atol=1e-5):
            raise ValueError("invalid mixture occupancy")

    @property
    def components(self) -> int:
        return len(self.weights)

    def save(self, path: Any) -> None:
        # Use a handle to prevent NumPy from silently appending .npz to .tmp.
        with open(path, "wb") as handle:
            np.savez_compressed(handle, kind=self.kind, covariance_type=self.covariance_type,
                                weights=self.weights, means=self.means, covariances=self.covariances)

    @classmethod
    def load(cls, path: Any) -> "FrozenClusterModel":
        with np.load(path, allow_pickle=False) as payload:
            return cls(str(payload["kind"]), str(payload["covariance_type"]),
                       payload["weights"], payload["means"], payload["covariances"])


def parameters(model: FrozenClusterModel, device: torch.device) -> dict[str, Any]:
    means = torch.as_tensor(model.means, dtype=torch.float32, device=device)
    out: dict[str, Any] = {"kind": model.kind, "covariance_type": model.covariance_type,
                           "means": means, "log_weights": torch.as_tensor(np.log(model.weights),
                                                                         dtype=torch.float32, device=device)}
    if model.kind == "kmeans":
        return out
    d = means.shape[1]
    if model.covariance_type in ("diag", "spherical"):
        cov = model.covariances
        if model.covariance_type == "spherical":
            cov = np.repeat(cov[:, None], d, axis=1)
        inverse = torch.as_tensor(1.0 / cov, dtype=torch.float32, device=device)
        out.update(inverse=inverse, mean_inverse=means * inverse,
                   constant=(means.square() * inverse + torch.as_tensor(np.log(2 * math.pi * cov),
                             dtype=torch.float32, device=device)).sum(dim=1))
    else:
        covs = model.covariances[None] if model.covariance_type == "tied" else model.covariances
        chol = np.linalg.cholesky(covs)
        # x @ inv(L).T gives whitened row vectors for covariance L @ L.T.
        out["whitener"] = torch.as_tensor(np.linalg.inv(chol).transpose(0, 2, 1).copy(),
                                          dtype=torch.float32, device=device)
        out["logdet"] = torch.as_tensor(2 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1),
                                        dtype=torch.float32, device=device)
        if model.covariance_type == "tied":
            out["white_means"] = means @ out["whitener"][0]
    return out


def scores(x: torch.Tensor, p: dict[str, Any]) -> torch.Tensor:
    if p["kind"] == "kmeans":
        return 2 * x @ p["means"].T - p["means"].square().sum(dim=1) - x.square().sum(dim=1)[:, None]
    cov = p["covariance_type"]
    if cov in ("diag", "spherical"):
        return -0.5 * (x.square() @ p["inverse"].T - 2 * x @ p["mean_inverse"].T + p["constant"]) + p["log_weights"]
    if cov == "tied":
        white = x @ p["whitener"][0]
        means = p["white_means"]
        distance = white.square().sum(dim=1)[:, None] - 2 * white @ means.T + means.square().sum(dim=1)
        return -0.5 * (distance + p["logdet"][0] + x.shape[1] * math.log(2 * math.pi)) + p["log_weights"]
    return torch.stack([
        -0.5 * (((x - mean) @ p["whitener"][k]).square().sum(dim=1)
                + p["logdet"][k] + x.shape[1] * math.log(2 * math.pi)) + p["log_weights"][k]
        for k, mean in enumerate(p["means"])], dim=1)


def _floor_covariance(cov: np.ndarray, floor: float) -> np.ndarray:
    symmetric = 0.5 * (cov + cov.T)
    eigenvalues, vectors = np.linalg.eigh(symmetric)
    return (vectors * np.maximum(eigenvalues, floor)) @ vectors.T


def _update(kind: str, covtype: str, nk: np.ndarray, sx: np.ndarray,
            second: np.ndarray, floor: float) -> FrozenClusterModel:
    if np.min(nk) <= 1e-8:
        raise RuntimeError("empty component: fail explicitly; do not silently reseed or change K")
    means = sx / nk[:, None]
    weights = nk / nk.sum()
    if kind == "kmeans":
        cov = np.empty(0)
    elif covtype in ("diag", "spherical"):
        diag = second / nk[:, None] - means * means
        cov = np.maximum(diag, floor) if covtype == "diag" else np.maximum(diag.mean(axis=1), floor)
    elif covtype == "tied":
        cov = _floor_covariance(second / nk.sum() - (means.T * weights) @ means, floor)
    else:
        cov = np.stack([_floor_covariance(second[k] / nk[k] - np.outer(means[k], means[k]), floor)
                        for k in range(len(nk))])
    return FrozenClusterModel(kind, covtype, weights, means, cov)


@torch.inference_mode()
def accumulate(data: Any, model: FrozenClusterModel, device: torch.device, batch_size: int,
               hard: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n, d = data.shape
    k = model.components
    p = parameters(model, device)
    nk = np.zeros(k, dtype=np.float64)
    sx = np.zeros((k, d), dtype=np.float64)
    second_shape = (d, d) if model.covariance_type == "tied" else ((k, d, d) if model.covariance_type == "full" else (k, d))
    second = np.zeros(second_shape, dtype=np.float64)
    objective = 0.0
    for start in range(0, n, batch_size):
        x = F.normalize(torch.as_tensor(np.asarray(data[start:min(start + batch_size, n)]).copy(), device=device).float(), dim=1)
        score = scores(x, p)
        if not torch.isfinite(score).all():
            raise FloatingPointError("non-finite likelihood scores")
        if hard or model.kind == "kmeans":
            r = F.one_hot(score.argmax(dim=1), k).float()
            objective += float(score.max(dim=1).values.double().sum().item())
        else:
            norm = torch.logsumexp(score, dim=1)
            r = (score - norm[:, None]).exp()
            objective += float(norm.double().sum().item())
        nk += r.sum(dim=0).double().cpu().numpy()
        sx += (r.T @ x).double().cpu().numpy()
        if model.kind == "kmeans" and not hard:
            continue
        if model.covariance_type == "tied":
            second += (x.T @ x).double().cpu().numpy()
        elif model.covariance_type == "full":
            for component in range(k):
                second[component] += (x.T @ (x * r[:, component:component+1])).double().cpu().numpy()
        else:
            second += (r.T @ x.square()).double().cpu().numpy()
    return nk, sx, second, objective / n


def initialize(sample: np.ndarray, kind: str, covtype: str, components: int, seed: int,
               device: torch.device, batch_size: int, floor: float) -> tuple[FrozenClusterModel, dict[str, Any]]:
    x = np.asarray(sample, dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    started = time.perf_counter()
    km = KMeans(n_clusters=components, init="k-means++", n_init=1, max_iter=100,
                tol=1e-4, random_state=seed, algorithm="lloyd").fit(x)
    warm = FrozenClusterModel("kmeans", covtype, np.full(components, 1 / components),
                              km.cluster_centers_, np.empty(0))
    nk, sx, second, _ = accumulate(x, warm, device, batch_size, hard=True)
    out = _update(kind, covtype, nk, sx, second, floor)
    return out, {"seconds": time.perf_counter() - started, "sample_rows": len(x), "seed": seed,
                 "kmeans_iterations": int(km.n_iter_), "n_init": 1,
                 "rule": "common KMeans++ centers, hard-assignment moments; no target data"}


@torch.inference_mode()
def fit(data: Any, model: FrozenClusterModel, device: torch.device, batch_size: int = 65536,
        max_iter: int = 100, tol: float = 1e-3, floor: float = 1e-6,
        callback: Callable | None = None, history: list[dict] | None = None) -> tuple[FrozenClusterModel, dict]:
    history = list(history or [])
    previous = history[-1]["objective"] if history else None
    last_convergence = history[-1].get("convergence_value") if history else None
    converged = last_convergence is not None and last_convergence < tol
    start_iteration = max_iter + 1 if converged else len(history) + 1
    for iteration in range(start_iteration, max_iter + 1):
        started = time.perf_counter()
        nk, sx, second, objective = accumulate(data, model, device, batch_size)
        updated = _update(model.kind, model.covariance_type, nk, sx, second, floor)
        change = None if previous is None else objective - previous
        mean_shift = float(np.linalg.norm(updated.means - model.means))
        # GMM uses average loglikelihood change; KMeans uses relative center shift.
        convergence_value = (mean_shift / max(float(np.linalg.norm(model.means)), 1e-12)
                             if model.kind == "kmeans" else (None if change is None else abs(change)))
        record = {"iteration": iteration, "objective": objective, "change": change,
                  "seconds": time.perf_counter() - started, "weights": updated.weights.tolist(),
                  "mean_shift": mean_shift, "convergence_value": convergence_value,
                  "rows_visited": int(data.shape[0]), "minimum_occupancy_fraction": float(updated.weights.min())}
        history.append(record)
        model = updated
        converged = convergence_value is not None and convergence_value < tol
        if callback:
            callback(model, history, converged)
        if converged:
            break
        previous = objective
    return model, {"samples": int(data.shape[0]), "feature_dimension": int(data.shape[1]),
                   "iterations": len(history), "converged": converged, "max_iter": max_iter,
                   "tolerance": tol, "covariance_floor": floor, "batch_size": batch_size,
                   "fit_seconds": sum(row["seconds"] for row in history), "history": history,
                   "stop_reason": "tolerance_reached" if converged else "maximum_iterations",
                   "minimum_occupancy_fraction": float(model.weights.min()),
                   "compute_dtype": "float32", "cross_batch_accumulation_dtype": "float64"}
