"""Baseline model builders used by the modeling notebooks."""

from __future__ import annotations

from typing import Sequence

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.models.train_models import RANDOM_STATE, build_preprocessor, make_estimator_pipeline


def build_dummy_baseline() -> DummyClassifier:
    """Return a prior-based dummy classifier."""

    return DummyClassifier(strategy="prior", random_state=RANDOM_STATE)


def build_logistic_baseline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
):
    """Return a class-weighted logistic regression pipeline."""

    preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=True)
    estimator = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="liblinear",
    )
    return make_estimator_pipeline(preprocessor, estimator)


def build_decision_tree_baseline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] | None = None,
):
    """Return a class-weighted decision tree pipeline."""

    preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=False)
    estimator = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    return make_estimator_pipeline(preprocessor, estimator)


def build_baseline_registry() -> dict[str, object]:
    """Return a simple registry of baseline estimators/builders."""

    return {
        "dummy_classifier": build_dummy_baseline,
        "logistic_regression_balanced": build_logistic_baseline,
        "decision_tree_balanced": build_decision_tree_baseline,
    }

