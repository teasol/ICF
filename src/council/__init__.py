"""Council infrastructure: seat specs, round-card validation, and the runner client.

The research norm this implements lives in AGENTS.md (decision D-046). Nothing
here decides research questions -- it only enforces the mechanics that make a
multi-seat deliberation worth reading: isolation in Phase 1, a mandatory
refuter seat, a synthesis seat that is not the proposer, budgets fixed before
results are seen, and abstentions recorded rather than silently dropped.
"""

from src.council.spec import (  # noqa: F401
    ARCHETYPES,
    RoundCard,
    Seat,
    SeatResult,
    classify_response,
    round_invalid_reason,
    validate_card,
)
