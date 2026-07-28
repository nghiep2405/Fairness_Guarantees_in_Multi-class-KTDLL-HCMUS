"""Adapters for external fairness baselines used by the experiments."""

from .aif360_subprocess import (
    AIF360AdversarialConfig,
    AIF360AdversarialResult,
    probe_aif360_subprocess_runtime,
    run_aif360_adversarial_subprocess,
)
from .fair_projection_adapter import (
    FairProjectionAdapter,
    FairProjectionConfig,
    FairProjectionResult,
    check_fair_projection_dependencies,
    ensure_fair_projection_runtime,
)
from .fair_transport_adapter import (
    FairTransportAdapter,
    FairTransportConfig,
    FairTransportResult,
    check_fair_transport_dependencies,
)

__all__ = [
    "AIF360AdversarialConfig",
    "AIF360AdversarialResult",
    "probe_aif360_subprocess_runtime",
    "run_aif360_adversarial_subprocess",
    "FairProjectionAdapter",
    "FairProjectionConfig",
    "FairProjectionResult",
    "check_fair_projection_dependencies",
    "ensure_fair_projection_runtime",
    "FairTransportAdapter",
    "FairTransportConfig",
    "FairTransportResult",
    "check_fair_transport_dependencies",
]
