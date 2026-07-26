from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss


_EPS = 1e-12


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, _EPS, 1.0)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def ordered_log_loss(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    probabilities = _normalize(probabilities)
    index = {str(label): position for position, label in enumerate(classes)}
    target = np.asarray([index[str(value)] for value in y_true], dtype=int)
    selected = probabilities[np.arange(len(target)), target]
    return float(-np.mean(np.log(np.clip(selected, 1e-15, 1.0))))


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    bins: int = 15,
) -> tuple[float, list[dict[str, float | int]]]:
    probabilities = _normalize(probabilities)
    predicted_index = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    predicted = classes[predicted_index]
    correct = predicted == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    curve: list[dict[str, float | int]] = []
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        count = int(mask.sum())
        if count == 0:
            continue
        accuracy = float(correct[mask].mean())
        mean_confidence = float(confidence[mask].mean())
        ece += (count / len(y_true)) * abs(accuracy - mean_confidence)
        curve.append(
            {
                "bin_lower": float(low),
                "bin_upper": float(high),
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
            }
        )
    return float(ece), curve


@dataclass(frozen=True)
class TemperatureCalibrator:
    temperature: float = 1.0

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = _normalize(probabilities)
        logits = np.log(probabilities)
        scaled = logits / max(float(self.temperature), 1e-6)
        scaled -= scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        return exp / exp.sum(axis=1, keepdims=True)

    def to_dict(self) -> dict[str, float]:
        return {"method": "temperature_scaling", "temperature": float(self.temperature)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TemperatureCalibrator":
        if not payload:
            return cls()
        return cls(float(payload.get("temperature", 1.0)))


def fit_temperature_scaler(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> TemperatureCalibrator:
    probabilities = _normalize(probabilities)

    def objective(value: float) -> float:
        calibrated = TemperatureCalibrator(value).transform(probabilities)
        return ordered_log_loss(y_true, calibrated, classes)

    result = minimize_scalar(objective, bounds=(0.25, 8.0), method="bounded", options={"xatol": 1e-3})
    return TemperatureCalibrator(float(result.x if result.success else 1.0))


def calibration_report(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    calibrator: TemperatureCalibrator | None = None,
) -> dict[str, Any]:
    raw = _normalize(probabilities)
    calibrated = (calibrator or TemperatureCalibrator()).transform(raw)
    raw_ece, raw_curve = expected_calibration_error(y_true, raw, classes)
    calibrated_ece, calibrated_curve = expected_calibration_error(y_true, calibrated, classes)
    report: dict[str, Any] = {
        "raw_ece": raw_ece,
        "calibrated_ece": calibrated_ece,
        "raw_log_loss": ordered_log_loss(y_true, raw, classes),
        "calibrated_log_loss": ordered_log_loss(y_true, calibrated, classes),
        "raw_reliability_curve": raw_curve,
        "calibrated_reliability_curve": calibrated_curve,
    }
    if len(classes) == 2:
        positive = classes[1]
        binary = (y_true == positive).astype(int)
        report["raw_brier_score"] = float(brier_score_loss(binary, raw[:, 1]))
        report["calibrated_brier_score"] = float(brier_score_loss(binary, calibrated[:, 1]))
    return report
