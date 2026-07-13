"""
GUDHI-based backend using CubicalComplex.
Implements defensive programming for production use.
"""
import numpy as np
import torch
import warnings
from typing import Optional
from .base import TopologyBackend, TopologyResult

class GudhiBackend(TopologyBackend):
    def __init__(self, min_persistence: float = 0.1):
        self.min_persistence = min_persistence
        self._gudhi_available = self._check_gudhi()
    
    def _check_gudhi(self) -> bool:
        try:
            import gudhi
            return True
        except ImportError:
            return False
    
    @property
    def name(self) -> str:
        return "gudhi"
    
    def _validate_and_prepare(self, tensor: torch.Tensor, channel: Optional[int] = None) -> np.ndarray:
        """Fixes: GPU memory, dimensionality, channel selection."""
        tensor = tensor.detach().cpu()
        
        if tensor.ndim == 4:
            if channel is None:
                tensor = tensor[:, 0, :, :]
            else:
                if channel >= tensor.shape[1]:
                    raise ValueError(f"Channel {channel} requested, but tensor has only {tensor.shape[1]} channels.")
                tensor = tensor[:, channel, :, :]
        elif tensor.ndim == 3:
            pass
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}. Expected 3D or 4D.")
        
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        
        return tensor.numpy()
    
    def _compute_single_betti(self, grid_2d: np.ndarray, min_persistence: float) -> tuple:
        import gudhi
        cc = gudhi.CubicalComplex(top_dimensional_cells=grid_2d.astype(np.float64))
        cc.persistence()
        intervals_0 = cc.persistence_intervals_in_dimension(0)
        intervals_1 = cc.persistence_intervals_in_dimension(1)
        beta_0 = sum(1 for (b, d) in intervals_0 if (d - b) > min_persistence)
        beta_1 = sum(1 for (b, d) in intervals_1 if (d - b) > min_persistence)
        return beta_0, beta_1
    
    def compute(self, tensor: torch.Tensor, **kwargs) -> TopologyResult:
        if not self._gudhi_available:
            return TopologyResult(
                success=False,
                error_msg="GUDHI is not installed.",
                backend_name=self.name
            )
        
        try:
            channel = kwargs.get("channel", None)
            negate = kwargs.get("negate", False)
            min_pers = kwargs.get("min_persistence", self.min_persistence)
            
            data = self._validate_and_prepare(tensor, channel=channel)
            if negate:
                data = -data  # convert to superlevel sets
            
            batch_size = data.shape[0]
            
            beta_0_list = []
            beta_1_list = []
            
            for i in range(batch_size):
                b0, b1 = self._compute_single_betti(data[i], min_pers)
                beta_0_list.append(b0)
                beta_1_list.append(b1)
            
            beta_0_arr = np.array(beta_0_list)
            beta_1_arr = np.array(beta_1_list)
            
            return TopologyResult(
                beta_0_mean=float(beta_0_arr.mean()),
                beta_1_mean=float(beta_1_arr.mean()),
                beta_0_std=float(beta_0_arr.std()),
                beta_1_std=float(beta_1_arr.std()),
                beta_0_max=int(beta_0_arr.max()),
                beta_1_max=int(beta_1_arr.max()),
                per_sample_beta_0=beta_0_list,
                per_sample_beta_1=beta_1_list,
                success=True,
                backend_name=self.name,
                metadata={
                    "batch_size": batch_size,
                    "min_persistence": min_pers,
                    "channel": channel,
                    "negate": negate,
                    "tensor_shape": tensor.shape
                }
            )
        except Exception as e:
            return TopologyResult(
                success=False,
                error_msg=f"GUDHI computation failed: {str(e)}",
                backend_name=self.name,
                metadata={"tensor_shape": tensor.shape}
            )