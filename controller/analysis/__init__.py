"""Post-run analysis. Imported by scripts and by the watchdog, never by a phase.

The controller never re-reads its own logs during a run — that is what makes
them append-only and trustworthy. These functions run afterwards, or against
events the watchdog already holds in memory for the cycle it is observing.
"""

from controller.analysis.metrics import (
    Series,
    coordination_strength,
    doctrine_stability,
    failure_modes,
    identity_texts,
)

__all__ = [
    "Series",
    "coordination_strength",
    "doctrine_stability",
    "failure_modes",
    "identity_texts",
]
