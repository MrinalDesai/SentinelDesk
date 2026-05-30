"""Classification: SVM ensemble partner (VGAC agent added in a later slice)."""

from .explain import explain
from .knn import KNNVote, KNNVoter
from .resolver import Decision, EdgeCaseResolver
from .scorer import DeterministicScorer, ScoreResult, VocabModel
from .svm import (
    CVReport,
    SVMClassifier,
    build_pipeline,
    cross_validate_svm,
    load_model,
    save_model,
    train_svm,
)

__all__ = [
    "build_pipeline",
    "train_svm",
    "cross_validate_svm",
    "CVReport",
    "save_model",
    "load_model",
    "SVMClassifier",
    "DeterministicScorer",
    "VocabModel",
    "ScoreResult",
    "KNNVoter",
    "KNNVote",
    "EdgeCaseResolver",
    "Decision",
    "explain",
]
