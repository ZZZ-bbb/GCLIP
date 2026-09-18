"""Exact archived marker extraction. No watershed, boxes or position validation."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from scipy import ndimage as ndi
MIN_REGION_AREA=4
METHODS=("hysteresis_core","erosion_core","probability_peak","thickness_probability_peak")
def count_metrics(ground_truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    ground_truth = ground_truth.astype(np.float64)
    predicted = predicted.astype(np.float64)
    residual = predicted - ground_truth
    centered = ground_truth - ground_truth.mean()
    return {
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "r2": float(1.0 - np.square(residual).sum() / max(np.square(centered).sum(), 1e-12)),
        "mean_ground_truth_count": float(ground_truth.mean()),
        "mean_predicted_count": float(predicted.mean()),
        "mean_signed_error": float(residual.mean()),
        "images": int(len(ground_truth)),
    }

def calibration_indices(paths: list[Path], per_height: int = 30) -> np.ndarray:
    selected: list[int] = []
    for height in ("7m", "12m", "20m"):
        candidates = [index for index, path in enumerate(paths) if path.name.startswith(f"{height}__")]
        positions = np.linspace(0, len(candidates) - 1, min(per_height, len(candidates)), dtype=int)
        selected.extend(candidates[int(position)] for position in positions)
    return np.asarray(sorted(set(selected)), dtype=np.int64)

def component_point_markers(
    candidate: np.ndarray,
    score: np.ndarray,
    minimum_area: int,
) -> list[tuple[int, int]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    points: list[tuple[int, int]] = []
    for component_id in range(1, count):
        if int(stats[component_id, cv2.CC_STAT_AREA]) < minimum_area:
            continue
        positions = np.argwhere(labels == component_id)
        values = score[positions[:, 0], positions[:, 1]]
        y, x = positions[int(np.argmax(values))]
        points.append((int(y), int(x)))
    return points

def local_peak_points(
    score: np.ndarray,
    allowed: np.ndarray,
    radius: int,
    minimum_score: float,
) -> list[tuple[int, int]]:
    maximum = ndi.maximum_filter(score, size=2 * radius + 1, mode="nearest")
    candidate = allowed & (score >= minimum_score) & (score >= maximum - 1e-7)
    return component_point_markers(candidate, score, minimum_area=1)

def make_markers(
    probability: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = np.asarray(probability, dtype=np.float32)
    low_mask = probability >= float(config["low_threshold"])
    component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
        low_mask.astype(np.uint8), connectivity=8
    )
    valid_mask = np.zeros_like(low_mask)
    markers = np.zeros(low_mask.shape, dtype=np.int32)
    marker_id = 0
    method = str(config["method"])
    for component_id in range(1, component_count):
        x, y, width, height, area = stats[component_id]
        if int(area) < MIN_REGION_AREA:
            continue
        slices = (slice(y, y + height), slice(x, x + width))
        local_mask = component_labels[slices] == component_id
        local_probability = probability[slices]
        valid_mask[slices] |= local_mask
        if method == "hysteresis_core":
            candidate = local_mask & (
                local_probability >= float(config["high_threshold"])
            )
            points = component_point_markers(
                candidate, local_probability, int(config["seed_min_area"])
            )
        elif method == "erosion_core":
            candidate = ndi.binary_erosion(
                local_mask,
                structure=ndi.generate_binary_structure(2, 1),
                iterations=int(config["erosion_iterations"]),
                border_value=0,
            )
            points = component_point_markers(
                candidate, local_probability, int(config["seed_min_area"])
            )
        elif method == "probability_peak":
            smooth = ndi.gaussian_filter(local_probability, sigma=0.8)
            points = local_peak_points(
                smooth,
                local_mask & (local_probability >= float(config["seed_threshold"])),
                int(config["radius"]),
                float(config["seed_threshold"]),
            )
        elif method == "thickness_probability_peak":
            distance = ndi.distance_transform_edt(local_mask).astype(np.float32)
            normalized_distance = distance / max(float(distance.max()), 1e-6)
            normalized_probability = np.clip(
                (local_probability - float(config["low_threshold"]))
                / max(1.0 - float(config["low_threshold"]), 1e-6),
                0.0,
                1.0,
            )
            score = normalized_distance * normalized_probability
            points = local_peak_points(
                score,
                local_mask,
                int(config["radius"]),
                float(config["relative_peak"]) * float(score[local_mask].max()),
            )
        else:
            raise ValueError(method)
        if not points:
            best = np.unravel_index(
                int(np.argmax(np.where(local_mask, local_probability, -1.0))),
                local_probability.shape,
            )
            points = [(int(best[0]), int(best[1]))]
        for local_y, local_x in points:
            marker_id += 1
            markers[y + local_y, x + local_x] = marker_id
    return valid_mask, markers, component_labels

def marker_count(probability: np.ndarray, config: dict[str, Any]) -> int:
    _, markers, _ = make_markers(probability, config)
    return int(markers.max())

def evaluate_config(
    probability: np.ndarray,
    ground_truth: np.ndarray,
    indices: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    predicted = np.asarray(
        [marker_count(probability[index], config) for index in indices], dtype=np.float64
    )
    return {**config, **count_metrics(ground_truth[indices], predicted)}

def configurations() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for low in (0.30, 0.35, 0.40):
        for high in (0.50, 0.60, 0.70, 0.80):
            for seed_area in (1, 4):
                rows.append(
                    {
                        "method": "hysteresis_core",
                        "low_threshold": low,
                        "high_threshold": high,
                        "seed_min_area": seed_area,
                    }
                )
        for iterations in (1, 2, 3, 4, 6):
            for seed_area in (1, 4):
                rows.append(
                    {
                        "method": "erosion_core",
                        "low_threshold": low,
                        "erosion_iterations": iterations,
                        "seed_min_area": seed_area,
                    }
                )
        for seed_threshold in (0.45, 0.55, 0.65):
            for radius in (3, 5, 8, 12, 16):
                rows.append(
                    {
                        "method": "probability_peak",
                        "low_threshold": low,
                        "seed_threshold": seed_threshold,
                        "radius": radius,
                    }
                )
        for relative_peak in (0.20, 0.40, 0.60):
            for radius in (3, 5, 8, 12, 16):
                rows.append(
                    {
                        "method": "thickness_probability_peak",
                        "low_threshold": low,
                        "relative_peak": relative_peak,
                        "radius": radius,
                    }
                )
    return rows

def compact_config(row: dict[str, Any]) -> dict[str, Any]:
    metric_keys = {
        "mae",
        "rmse",
        "r2",
        "mean_ground_truth_count",
        "mean_predicted_count",
        "mean_signed_error",
        "images",
    }
    return {key: value for key, value in row.items() if key not in metric_keys}
