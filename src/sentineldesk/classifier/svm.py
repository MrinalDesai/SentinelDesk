"""
SVM classifier (Round 2 Algo 5) — the ensemble partner to Mistral.

A linear-kernel SVC over TF-IDF features. We expose calibrated probabilities
(probability=True) because the VGAC ensemble (Round 2 R7) compares the SVM's
confidence against Mistral's and falls back to the higher-confidence
prediction on disagreement.

The trainer also runs stratified k-fold cross-validation and reports
mean +/- std for accuracy and macro-F1. That is the honest, reproducible
counterpart to a one-shot sample metric, and it is what should back any
headline numbers in the submission.

NOTE on speed: SVC(probability=True) fits an internal CV for Platt scaling and
gets slow on large corpora. For a few thousand tickets it is fine. If training
time becomes painful at full scale, swap SVC for LinearSVC wrapped in
CalibratedClassifierCV — same interface, much faster, marginally different
probabilities. Left as SVC to match the Round 2 spec exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from ..corpus import LabeledTicket


def build_pipeline(
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> Pipeline:
    """TF-IDF -> linear SVC with probability estimates."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    stop_words="english",
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
                ),
            ),
            ("svc", SVC(kernel="linear", probability=True, random_state=42)),
        ]
    )


@dataclass
class CVReport:
    folds: int
    n_samples: int
    accuracy_mean: float
    accuracy_std: float
    f1_macro_mean: float
    f1_macro_std: float

    def as_dict(self) -> dict:
        return asdict(self)


def cross_validate_svm(
    tickets: list[LabeledTicket],
    folds: int = 5,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> CVReport:
    """Stratified k-fold CV, reporting mean +/- std accuracy and macro-F1."""
    X = [t.text for t in tickets]
    y = [t.category for t in tickets]

    # don't ask for more folds than the smallest class supports
    min_class = min(y.count(c) for c in set(y))
    folds = max(2, min(folds, min_class))

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_validate(
        build_pipeline(max_features, ngram_range),
        X,
        y,
        cv=skf,
        scoring=["accuracy", "f1_macro"],
    )
    return CVReport(
        folds=folds,
        n_samples=len(tickets),
        accuracy_mean=float(np.mean(scores["test_accuracy"])),
        accuracy_std=float(np.std(scores["test_accuracy"])),
        f1_macro_mean=float(np.mean(scores["test_f1_macro"])),
        f1_macro_std=float(np.std(scores["test_f1_macro"])),
    )


def train_svm(
    tickets: list[LabeledTicket],
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> Pipeline:
    """Fit the full pipeline on all tickets and return it."""
    pipeline = build_pipeline(max_features, ngram_range)
    pipeline.fit([t.text for t in tickets], [t.category for t in tickets])
    return pipeline


def save_model(pipeline: Pipeline, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(Path(path))


class SVMClassifier:
    """Runtime wrapper used by the VGAC ensemble: text -> (category, probability)."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.classes_ = list(pipeline.classes_)

    @classmethod
    def from_path(cls, path: str | Path) -> "SVMClassifier":
        return cls(load_model(path))

    def predict(self, text: str) -> tuple[str, float]:
        """Return (predicted_category, probability_of_that_category).

        The label comes from SVC.predict() (the authoritative decision
        function); the probability is the calibrated confidence of *that*
        label. We deliberately do NOT argmax predict_proba for the label,
        because Platt-calibrated probabilities can disagree with the SVM
        decision on small data, which would make predict() and the reported
        label inconsistent.
        """
        label = str(self.pipeline.predict([text])[0])
        probs = self.pipeline.predict_proba([text])[0]
        idx = self.classes_.index(label)
        return label, float(probs[idx])

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        labels = self.pipeline.predict(texts)
        proba = self.pipeline.predict_proba(texts)
        out: list[tuple[str, float]] = []
        for label, row in zip(labels, proba):
            label = str(label)
            out.append((label, float(row[self.classes_.index(label)])))
        return out
