"""Adapters for external fairness baselines used by the experiments."""

from .fair_projection_adapter import (
    FairProjectionAdapter,
    FairProjectionConfig,
    FairProjectionResult,
    check_fair_projection_dependencies,
)
from .fair_transport_adapter import (
    FairTransportAdapter,
    FairTransportConfig,
    FairTransportResult,
    check_fair_transport_dependencies,
)

__all__ = [
    "FairProjectionAdapter",
    "FairProjectionConfig",
    "FairProjectionResult",
    "check_fair_projection_dependencies",
    "FairTransportAdapter",
    "FairTransportConfig",
    "FairTransportResult",
    "check_fair_transport_dependencies",
]
