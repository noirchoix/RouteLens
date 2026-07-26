from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class PreFitDecision:
    trainable: bool
    reason_code: str | None
    reasons: tuple[str, ...]
    retained_labels: tuple[str, ...] = ()
    dropped_labels: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trainable": self.trainable,
            "reason_code": self.reason_code,
            "reasons": list(self.reasons),
            "retained_labels": list(self.retained_labels),
            "dropped_labels": list(self.dropped_labels),
        }


def validate_classification_support(
    split_counts: dict[str, dict[str, int]],
    *,
    retained_classes: Iterable[str],
    task_kind: str,
) -> PreFitDecision:
    classes = tuple(str(item) for item in retained_classes)
    train = Counter({str(k): int(v) for k, v in split_counts.get("train", {}).items()})
    val = Counter({str(k): int(v) for k, v in split_counts.get("val", {}).items()})
    test = Counter({str(k): int(v) for k, v in split_counts.get("test", {}).items()})
    reasons: list[str] = []
    minimum_classes = 2
    observed_train = [label for label in classes if train.get(label, 0) > 0]
    if len(observed_train) < minimum_classes:
        reasons.append(f"{task_kind} training requires at least two supported classes.")
    if sum(val.get(label, 0) for label in classes) == 0:
        reasons.append("Validation split has no evaluable rows for retained classes.")
    if sum(test.get(label, 0) for label in classes) == 0:
        reasons.append("Test split has no evaluable rows for retained classes.")
    if task_kind == "binary":
        for split_name, counts in (("train", train), ("val", val), ("test", test)):
            present = [label for label in classes if counts.get(label, 0) > 0]
            if len(present) < 2:
                reasons.append(f"Binary {split_name} split lacks positive and negative class support.")
    return PreFitDecision(
        trainable=not reasons,
        reason_code=None if not reasons else "insufficient_class_support",
        reasons=tuple(dict.fromkeys(reasons)),
        retained_labels=classes,
    )


def validate_multilabel_support(
    split_totals: dict[str, int],
    positive_counts: dict[str, dict[str, int]],
    *,
    candidate_labels: Iterable[str],
    minimum_positive_train: int,
    minimum_negative_train: int | None = None,
) -> PreFitDecision:
    minimum_negative_train = minimum_negative_train or minimum_positive_train
    retained: list[str] = []
    dropped: list[dict[str, Any]] = []
    for label in candidate_labels:
        support: dict[str, dict[str, int]] = {}
        reasons: list[str] = []
        for split in ("train", "val", "test"):
            total = int(split_totals.get(split, 0))
            positives = int(positive_counts.get(split, {}).get(label, 0))
            negatives = max(total - positives, 0)
            support[split] = {"total": total, "positive": positives, "negative": negatives}
        if support["train"]["positive"] < minimum_positive_train:
            reasons.append("insufficient_train_positive_support")
        if support["train"]["negative"] < minimum_negative_train:
            reasons.append("insufficient_train_negative_support")
        for split in ("val", "test"):
            if support[split]["positive"] == 0:
                reasons.append(f"missing_{split}_positive_support")
            if support[split]["negative"] == 0:
                reasons.append(f"missing_{split}_negative_support")
        if reasons:
            dropped.append({"label": label, "reasons": reasons, "support": support})
        else:
            retained.append(label)
    if retained:
        return PreFitDecision(
            trainable=True,
            reason_code=None,
            reasons=(),
            retained_labels=tuple(retained),
            dropped_labels=tuple(dropped),
        )
    return PreFitDecision(
        trainable=False,
        reason_code="multilabel_no_valid_binary_targets",
        reasons=("No multilabel target has both positive and negative support across train, validation and test.",),
        retained_labels=(),
        dropped_labels=tuple(dropped),
    )
