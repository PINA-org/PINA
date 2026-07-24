"""
GUDHI-based backend for computing persistent homology on grid data.
"""
import numpy as np
import torch
import logging
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
from pina._src.callback.topology.topology_result import TopologyResult

logger = logging.getLogger(__name__)


def _parallel_betti_worker(args):
    """Worker function for parallel Betti computation."""
    grid, min_pers, idx = args
    try:
        import gudhi
        cc = gudhi.CubicalComplex(top_dimensional_cells=grid.astype(np.float64))
        cc.persistence()
        intervals_0 = cc.persistence_intervals_in_dimension(0)
        intervals_1 = cc.persistence_intervals_in_dimension(1)
        beta_0 = sum(1 for (b, d) in intervals_0 if (d - b) > min_pers)
        beta_1 = sum(1 for (b, d) in intervals_1 if (d - b) > min_pers)
        return beta_0, beta_1, None
    except Exception as e:
        return 0, 0, str(e)


class GudhiBackend:
    """
    GUDHI-based backend for computing persistent homology on grid data.

    Uses :class:`gudhi.CubicalComplex` to compute Betti numbers (β₀, β₁)
    from a 2D grid. Implements persistence filtering to ignore noise.

    .. seealso::
        - GUDHI library: https://gudhi.inria.fr/
        - Persistent homology: Edelsbrunner & Harer, "Computational Topology"

    :param float min_persistence: Minimum lifespan for a feature to be counted.
        Default is 0.1.
    """

    def __init__(self, min_persistence: float = 0.1):
        """
        Initialize the GUDHI backend.

        :param float min_persistence: Minimum persistence threshold.
        """
        self.min_persistence = min_persistence
        self._gudhi_available = self._check_gudhi()

    def _check_gudhi(self) -> bool:
        try:
            import gudhi  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def name(self) -> str:
        """Name of the backend."""
        return "gudhi"

    def _is_finite(self, arr: np.ndarray) -> bool:
        """Check if all values in the array are finite."""
        return np.isfinite(arr).all()

    def _validate_and_prepare(self, tensor: torch.Tensor, channel: Optional[int] = None) -> np.ndarray:
        """
        Validate and prepare the tensor for GUDHI.

        :param torch.Tensor tensor: Input tensor.
        :param int channel: Channel index to use, or None for the first channel.
        :return: Prepared numpy array of shape (B, H, W).
        :raises ValueError: If the tensor shape is invalid.
        """
        tensor = tensor.detach().cpu()

        # Handle 2D case first: (H, W) -> (1, H, W)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        # Now we have at least 3D
        if tensor.ndim == 4:
            # (B, C, H, W) -> (B, H, W) by selecting channel
            if channel is None:
                tensor = tensor[:, 0, :, :]
            else:
                if channel >= tensor.shape[1]:
                    raise ValueError(f"Channel {channel} requested, but tensor has {tensor.shape[1]} channels.")
                tensor = tensor[:, channel, :, :]
        elif tensor.ndim == 3:
            # (B, H, W) -> OK
            pass
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}. Expected 2D, 3D, or 4D.")

        return tensor.numpy()

    def _compute_single_betti(self, grid_2d: np.ndarray, min_persistence: float, sample_idx: int = 0) -> tuple:
        """Compute Betti numbers for a single 2D grid."""
        if not self._is_finite(grid_2d):
            logger.warning(f"Sample {sample_idx} contains NaN or Inf values. Skipping topology computation.")
            return 0, 0

        import gudhi

        cc = gudhi.CubicalComplex(top_dimensional_cells=grid_2d.astype(np.float64))
        cc.persistence()

        intervals_0 = cc.persistence_intervals_in_dimension(0)
        intervals_1 = cc.persistence_intervals_in_dimension(1)

        beta_0 = sum(1 for (b, d) in intervals_0 if (d - b) > min_persistence)
        beta_1 = sum(1 for (b, d) in intervals_1 if (d - b) > min_persistence)

        return beta_0, beta_1

    def compute(self, tensor: torch.Tensor, **kwargs) -> TopologyResult:
        """
        Compute Betti numbers from a batched tensor.

        :param torch.Tensor tensor: Input tensor. Shape (B, H, W) or (B, C, H, W).
        :param kwargs: Additional arguments:
            - ``channel`` (int): Channel to use.
            - ``negate`` (bool): If True, negate the tensor before computation.
            - ``min_persistence`` (float): Override the persistence threshold.
            - ``num_workers`` (int): Number of parallel workers for large batches.
        :return: A :class:`TopologyResult` object.
        :rtype: TopologyResult
        """
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
            num_workers = kwargs.get("num_workers", 0)

            data = self._validate_and_prepare(tensor, channel=channel)
            if negate:
                data = -data

            batch_size = data.shape[0]
            beta_0_list = []
            beta_1_list = []

            # Parallel execution for large batches
            if num_workers > 0 and batch_size > 8:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(_parallel_betti_worker, (data[i], min_pers, i)) for i in range(batch_size)]
                    for i, future in enumerate(futures):
                        b0, b1, err = future.result(timeout=60)
                        if err:
                            logger.error(f"Sample {i} failed: {err}")
                            return TopologyResult(
                                success=False,
                                error_msg=f"Sample {i} failed: {err}",
                                beta_0_mean=0.0,
                                beta_1_mean=None,
                                beta_0_std=0.0,
                                beta_1_std=None,
                                beta_0_max=0.0,
                                beta_1_max=None,
                                per_sample_beta_0=[],
                                per_sample_beta_1=[],
                                backend_name=self.name,
                                metadata={"tensor_shape": tensor.shape}
                            )
                        beta_0_list.append(b0)
                        beta_1_list.append(b1)
            else:
                # Sequential fallback
                for i in range(batch_size):
                    try:
                        b0, b1 = self._compute_single_betti(data[i], min_pers, i)
                    except Exception as e:
                        logger.error(f"Sample {i} failed: {e}")
                        return TopologyResult(
                            success=False,
                            error_msg=f"Sample {i} failed: {e}",
                            beta_0_mean=0.0,
                            beta_1_mean=None,
                            beta_0_std=0.0,
                            beta_1_std=None,
                            beta_0_max=0.0,
                            beta_1_max=None,
                            per_sample_beta_0=[],
                            per_sample_beta_1=[],
                            backend_name=self.name,
                            metadata={"tensor_shape": tensor.shape}
                        )
                    beta_0_list.append(b0)
                    beta_1_list.append(b1)

            beta_0_arr = np.array(beta_0_list)
            beta_1_arr = np.array(beta_1_list)

            return TopologyResult(
                beta_0_mean=float(beta_0_arr.mean()),
                beta_1_mean=float(beta_1_arr.mean()) if len(beta_1_arr) > 0 else None,
                beta_0_std=float(beta_0_arr.std()),
                beta_1_std=float(beta_1_arr.std()) if len(beta_1_arr) > 0 else None,
                beta_0_max=int(beta_0_arr.max()),
                beta_1_max=int(beta_1_arr.max()) if len(beta_1_arr) > 0 else None,
                per_sample_beta_0=beta_0_list,
                per_sample_beta_1=beta_1_list,
                success=True,
                backend_name=self.name,
                metadata={
                    "batch_size": batch_size,
                    "min_persistence": min_pers,
                    "channel": channel,
                    "negate": negate,
                    "num_workers": num_workers,
                    "tensor_shape": tensor.shape
                }
            )

        except Exception as e:
            logger.error(f"GUDHI computation failed: {e}")
            return TopologyResult(
                success=False,
                error_msg=f"GUDHI computation failed: {str(e)}",
                beta_0_mean=0.0,
                beta_1_mean=None,
                beta_0_std=0.0,
                beta_1_std=None,
                beta_0_max=0.0,
                beta_1_max=None,
                per_sample_beta_0=[],
                per_sample_beta_1=[],
                backend_name=self.name,
                metadata={"tensor_shape": tensor.shape}
            )