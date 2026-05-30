"""Result contracts produced by stages 2-5 of the runtime pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from ..config import Action, Domain


class Classification(BaseModel):
    """Stage 2 output: the ensemble decision plus both raw predictions.

    We keep the individual Mistral and SVM predictions (not just the merged
    result) so disagreement is auditable and so the self-correction loop can
    learn from which model was wrong.
    """

    category: Domain
    confidence: float = Field(ge=0.0, le=1.0)
    matched_terms: list[str] = Field(default_factory=list)
    mistral_prediction: Optional[str] = None
    mistral_confidence: Optional[float] = None
    svm_prediction: Optional[str] = None
    svm_confidence: Optional[float] = None
    reasoning: Optional[str] = None
    agreed: bool = False  # did Mistral and SVM agree on the category?


class Routing(BaseModel):
    """Stage 3 output: which team owns the ticket and by when."""

    department: str
    contact: str
    sla_hours: int
    sla_deadline: datetime


class SimilarTicket(BaseModel):
    """A retrieved neighbour with its source, used as resolution evidence."""

    title: str
    resolution: str
    score: float
    method: str  # "bm25" | "semantic" | "graph"


class Resolution(BaseModel):
    """Stage 4 output: generated steps plus the evidence they cite."""

    steps: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    similar_tickets: list[SimilarTicket] = Field(default_factory=list)
    resolution_method: str = "rag"  # "rag" | "bm25_fallback" | "degraded"
    quality_score: Optional[int] = Field(default=None, ge=1, le=5)


class EscalationDecision(BaseModel):
    """Stage 5 terminal decision returned to the caller."""

    action: Action
    reason: str
    quality_score: Optional[int] = Field(default=None, ge=1, le=5)
    escalated_to: Optional[str] = None
    automation_suggested: bool = False
