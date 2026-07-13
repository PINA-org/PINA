from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch

@dataclass
class TopologyResult:
    beta_0_mean: float
    beta_1_mean: Optional[float] = None  # Can be None if backend doesn't support β₁
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

class TopologyBackend(ABC):
    @abstractmethod
    def compute(self, tensor: torch.Tensor, **kwargs) -> TopologyResult:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass