"""Stage 0 deterministic safety gate."""

from .patterns import (
    DOCUMENTED_COUNT,
    HIGH_STAKES,
    PROPOSED_COUNT,
    HighStakesCategory,
)
from .safety_layer import safety_check

__all__ = [
    "safety_check",
    "HIGH_STAKES",
    "HighStakesCategory",
    "DOCUMENTED_COUNT",
    "PROPOSED_COUNT",
]
