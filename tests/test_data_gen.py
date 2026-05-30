"""
Tests for the synthetic data generation pipeline.

A single stub LLM simulates Mistral by branching on prompt type (generation /
validation / augmentation), letting us exercise the entire pipeline -
including the score<threshold filter, augmentation fan-out, and dedup -
without a GPU.
"""

import json
import re

import pytest

from sentineldesk.data_gen import (
    SyntheticDataGenerator,
    parse_augmentations,
    parse_generated_tickets,
    parse_validation_score,
    write_tickets_csv,
)
from sentineldesk.llm import StubLLMClient


# --- simulated Mistral ------------------------------------------------------

def make_handler(score: int = 5):
    """Return a handler(prompt)->str that fakes Mistral for each prompt type."""

    def handler(prompt: str) -> str:
        if prompt.startswith("Generate") and "tickets" in prompt:
            m = re.search(r"Generate (\d+) distinct", prompt)
            n = int(m.group(1)) if m else 1
            cat = re.search(r'category "([^"]+)"', prompt)
            label = cat.group(1) if cat else "X"
            items = [
                {
                    "title": f"{label} issue {i}",
                    "description": f"a realistic {label} problem number {i}",
                    "resolution": "do the fix",
                    "priority": "High",
                    "request_type": "Incident",
                }
                for i in range(n)
            ]
            # object-wrapped, as Ollama format:json produces
            return json.dumps({"tickets": items})
        if prompt.startswith("Score how well"):
            return json.dumps({"score": score, "reason": "ok"})
        if prompt.startswith("Rewrite this IT ticket"):
            m = re.search(r"Rewrite this IT ticket (\d+)", prompt)
            n = int(m.group(1)) if m else 3
            return json.dumps(
                {"variants": [
                    {"title": f"variant {i}", "description": f"reworded {i}"}
                    for i in range(n)
                ]}
            )
        return ""

    return handler


CATS = ["Network", "Security"]


def test_generate_hits_target_per_category():
    gen = SyntheticDataGenerator(StubLLMClient(make_handler()), CATS)
    tickets = gen.generate(per_category=5, batch_size=2)
    assert len(tickets) == 10
    by_cat = {c: sum(1 for t in tickets if t["category"] == c) for c in CATS}
    assert by_cat == {"Network": 5, "Security": 5}


def test_validation_drops_low_scores():
    gen = SyntheticDataGenerator(StubLLMClient(make_handler(score=3)), CATS)
    tickets = gen.generate(per_category=4)
    kept = gen.validate(tickets, threshold=4)
    assert kept == []  # everything scored 3, below threshold


def test_validation_keeps_high_scores():
    gen = SyntheticDataGenerator(StubLLMClient(make_handler(score=5)), CATS)
    tickets = gen.generate(per_category=4)
    kept = gen.validate(tickets, threshold=4)
    assert len(kept) == len(tickets)


def test_augment_fans_out_and_preserves_category():
    gen = SyntheticDataGenerator(StubLLMClient(make_handler()), ["Network"])
    base = gen.generate(per_category=2)
    augmented = gen.augment(base, variations=3)
    # originals + 3 variants each = 2 + 6
    assert len(augmented) == 8
    assert all(t["category"] == "Network" for t in augmented)


def test_dedup_removes_identical():
    dupes = [
        {"title": "A", "description": "same thing", "category": "Network",
         "resolution": "", "priority": "Low", "request_type": "Incident"},
        {"title": "a", "description": "SAME   thing", "category": "Network",
         "resolution": "", "priority": "Low", "request_type": "Incident"},
    ]
    assert len(SyntheticDataGenerator.dedup(dupes)) == 1


def test_full_run_and_csv(tmp_path):
    gen = SyntheticDataGenerator(StubLLMClient(make_handler(score=5)), CATS)
    tickets = gen.run(per_category=3, threshold=4, variations=2)
    assert len(tickets) > 0
    out = write_tickets_csv(tickets, tmp_path / "synthetic_tickets.csv")
    assert out.exists()
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert header == "title,description,category,resolution,priority,request_type"


# --- parser edge cases ------------------------------------------------------

def test_parse_generated_normalises_bad_priority():
    raw = '[{"title":"t","description":"d","priority":"URGENT","request_type":"weird"}]'
    out = parse_generated_tickets(raw, "Network")
    assert out[0]["priority"] == "Medium"        # invalid -> default
    assert out[0]["request_type"] == "Incident"  # invalid -> default


def test_parse_generated_handles_fences_and_junk():
    raw = 'Sure!\n```json\n[{"title":"t","description":"d"}]\n```\nHope this helps'
    out = parse_generated_tickets(raw, "Security")
    assert len(out) == 1 and out[0]["category"] == "Security"


@pytest.mark.parametrize("raw,expected", [
    ('{"score": 5}', 5),
    ('{"score": 4.0}', 4),
    ('the score is 2 out of 5', 2),
    ('garbage', 0),
    ('{"score": 9}', 5),  # clamped
])
def test_parse_validation_score(raw, expected):
    assert parse_validation_score(raw) == expected


def test_parse_augmentations_drops_incomplete():
    raw = '[{"title":"a","description":"b"},{"title":"only title"}]'
    assert parse_augmentations(raw) == [{"title": "a", "description": "b"}]


def test_parse_generated_accepts_object_wrapped():
    # Ollama format:json emits a top-level object; both shapes must parse.
    bare = '[{"title":"t","description":"d"}]'
    wrapped = '{"tickets":[{"title":"t","description":"d"}]}'
    assert len(parse_generated_tickets(bare, "Network")) == 1
    assert len(parse_generated_tickets(wrapped, "Network")) == 1


def test_parse_augmentations_accepts_object_wrapped():
    wrapped = '{"variants":[{"title":"a","description":"b"}]}'
    assert parse_augmentations(wrapped) == [{"title": "a", "description": "b"}]


def test_parse_generated_finds_arbitrary_list_key():
    # if the model wraps under an unexpected key, we still find the array
    raw = '{"data":[{"title":"t","description":"d"}]}'
    assert len(parse_generated_tickets(raw, "Storage")) == 1
