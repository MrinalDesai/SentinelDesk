"""
Tests for the SVM classifier.

Trained on the 42-ticket seed corpus. With mutually-exclusive categories the
model should classify obvious in-domain text correctly and emit sane
probabilities; cross-validation should return finite metrics; and a saved
model should reload and predict identically.
"""

from pathlib import Path

import pytest

from sentineldesk.classifier import (
    CVReport,
    SVMClassifier,
    cross_validate_svm,
    load_model,
    save_model,
    train_svm,
)
from sentineldesk.config import Domain
from sentineldesk.corpus import load_tickets_csv

SEED = Path(__file__).resolve().parents[1] / "data" / "seed_tickets.csv"


@pytest.fixture(scope="module")
def tickets():
    return load_tickets_csv(SEED)


@pytest.fixture(scope="module")
def model(tickets):
    return SVMClassifier(train_svm(tickets))


def test_predict_returns_valid_domain_and_prob(model):
    label, prob = model.predict("vpn keeps disconnecting on the office wifi")
    assert label in {d.value for d in Domain}
    assert 0.0 <= prob <= 1.0


def test_predicts_obvious_in_domain(model):
    # strongly-worded in-domain examples should land in the right category
    assert model.predict("database query is extremely slow missing index")[0] == "Database"
    assert model.predict("user account locked needs password reset")[0] == "Access Management"
    assert model.predict("disk volume is full backup failed no space")[0] == "Storage"


def test_predict_batch_matches_single(model):
    texts = ["firewall blocking the port", "phishing email reported"]
    batch = model.predict_batch(texts)
    assert len(batch) == 2
    for text, (label, prob) in zip(texts, batch):
        single = model.predict(text)
        assert single[0] == label
        assert abs(single[1] - prob) < 1e-9


def test_cross_validation_report(tickets):
    report = cross_validate_svm(tickets, folds=5)
    assert isinstance(report, CVReport)
    assert report.n_samples == 42
    assert 0.0 <= report.accuracy_mean <= 1.0
    assert 0.0 <= report.f1_macro_mean <= 1.0
    assert report.accuracy_std >= 0.0


def test_save_load_roundtrip(tickets, tmp_path):
    pipeline = train_svm(tickets)
    path = save_model(pipeline, tmp_path / "svm_model.pkl")
    assert path.exists()
    reloaded = SVMClassifier(load_model(path))
    text = "kubernetes node out of memory oom killer"
    assert reloaded.predict(text) == SVMClassifier(pipeline).predict(text)
