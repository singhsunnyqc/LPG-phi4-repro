"""Policy books for evaluation-only datasets (HarmBench, WildGuard).

Each module exposes:
- POLICY_BOOK: dict[tax_id -> imperative one-sentence rule] used to build
  the policy list shown to LPG at evaluation time.

LPG reasons over a *given* policy list, so for benchmarks that don't ship
their own policy strings we synthesize them from the dataset's native
taxonomy. Wording matches the style used in the training corpus so the
train and eval distributions stay aligned.
"""

from .harmbench import POLICY_BOOK as HARMBENCH_POLICY_BOOK
from .wildguard import POLICY_BOOK as WILDGUARD_POLICY_BOOK

__all__ = [
    "HARMBENCH_POLICY_BOOK",
    "WILDGUARD_POLICY_BOOK",
]
