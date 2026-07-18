"""
Container for topological analysis results.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class TopologyResult:
    """
    Standardized container for results of topological analysis.

    This dataclass holds Betti numbers (β₀, β₁), their statistics, and
    per‑sample values. It is returned by :class:`TopologicalProfiler`.

    :param float beta_0_mean: Mean of β₀ across the batch.
    :param float beta_1_mean: Mean of β₁ across the batch, or None if not computed.
    :param float beta_0_std: Standard deviation of β₀.
    :param float beta_1_std: Standard deviation of β₁, or None if not computed.
    :param float beta_0_max: Maximum β₀ across the batch.
    :param float beta_1_max: Maximum β₁ across the batch, or None if not computed.
    :param list[int] per_sample_beta_0: β₀ values for each sample in the batch.
    :param list[Optional[int]] per_sample_beta_1: β₁ values for each sample, with None if not supported.
    :param bool success: Whether the computation succeeded.
    :param str error_msg: Error message if computation failed, or None.
    :param str backend_name: Name of the backend used (e.g., "gudhi").
    :param dict[str, Any] metadata: Additional metadata (batch size, thresholds, etc.).
    """
    beta_0_mean: float
    beta_1_mean: Optional[float] = None
    beta_0_std: float = 0.0
    beta_1_std: Optional[float] = None
    beta_0_max: float = 0.0
    beta_1_max: Optional[float] = None
    per_sample_beta_0: List[int] = field(default_factory=list)
    per_sample_beta_1: List[Optional[int]] = field(default_factory=list)
    success: bool = True
    error_msg: Optional[str] = None
    backend_name: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)