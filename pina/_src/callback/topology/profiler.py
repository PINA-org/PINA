"""
Topological Profiler: Lazy-loads GUDHI backend with state consistency.
"""
import torch
from typing import Optional
from pina._src.callback.topology.topology_result import TopologyResult


class TopologicalProfiler:
    """
    Lazy-loading profiler for topological analysis.

    This class wraps the GUDHI backend and provides a consistent interface
    for computing Betti numbers. The backend is only instantiated when
    :meth:`compute` is first called.

    :param float min_persistence: Minimum persistence threshold. Default is 0.1.
    :param int channel: Default channel to monitor. If None, uses the first channel.
    """

    def __init__(
        self,
        min_persistence: float = 0.1,
        channel: Optional[int] = None,
    ):
        """
        Initialize the profiler.

        See the class docstring for parameter descriptions.
        """
        self.min_persistence = min_persistence
        self.channel = channel
        self._backend = None

    def _get_backend(self):
        """
        Lazy-load the GUDHI backend.

        :return: The GUDHI backend instance.
        :rtype: GudhiBackend
        :raises ImportError: If GUDHI is not installed.
        """
        if self._backend is None:
            try:
                from pina._src.callback.topology.gudhi_backend import GudhiBackend
                self._backend = GudhiBackend(min_persistence=self.min_persistence)
            except ImportError as e:
                raise ImportError(
                    "\n" + "=" * 60 + "\n"
                    "GUDHI BACKEND REQUIRED FOR TOPOLOGICAL PROFILING.\n"
                    "Please install GUDHI:\n"
                    "    pip install gudhi\n"
                    "=" * 60
                ) from e
        return self._backend

    def compute(self, tensor: torch.Tensor, **kwargs) -> TopologyResult:
        """
        Compute Betti numbers from a tensor.

        :param torch.Tensor tensor: Input tensor.
        :param kwargs: Additional arguments:
            - ``channel`` (int): Override the default channel.
            - ``min_persistence`` (float): Override the persistence threshold.
            - ``num_workers`` (int): Number of parallel workers for large batches.
        :return: A :class:`TopologyResult` object.
        :rtype: TopologyResult
        """
        channel = kwargs.pop("channel", self.channel)
        min_pers = kwargs.pop("min_persistence", self.min_persistence)

        backend = self._get_backend()

        # CRITICAL FIX: Do NOT mutate shared backend state.
        # Pass min_persistence directly to backend.compute to avoid race conditions.
        return backend.compute(
            tensor,
            channel=channel,
            min_persistence=min_pers,
            **kwargs
        )

    @property
    def backend_name(self) -> str:
        """Name of the active backend."""
        return self._get_backend().name