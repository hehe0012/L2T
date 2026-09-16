"""Evaluation helpers for OTIB-style transfer."""

from __future__ import annotations

from collections.abc import Sequence


def exact_match_accuracy(predictions: Sequence[int], targets: Sequence[int]) -> float:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if not predictions:
        return 0.0
    correct = sum(int(pred == target) for pred, target in zip(predictions, targets))
    return correct / len(predictions)


def program_accuracy(predictions: Sequence[Sequence[int]], targets: Sequence[Sequence[int]]) -> float:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if not predictions:
        return 0.0
    correct = sum(tuple(pred) == tuple(target) for pred, target in zip(predictions, targets))
    return correct / len(predictions)

