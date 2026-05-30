"""Online learning: self-correction loop that grows the vocabulary from human corrections."""

from .self_correction import CorrectionResult, CorrectionStore

__all__ = ["CorrectionStore", "CorrectionResult"]
