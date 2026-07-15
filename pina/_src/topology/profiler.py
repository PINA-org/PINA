"""
Topological Profiler: Lazy-loads GUDHI backend with state consistency.
"""
import torch
from typing import Optional
from pina._src.topology.backends import TopologyResult

class TopologicalProfiler:
    def __init__(
        self,
        min_persistence: float = 0.1,
        channel: Optional[int] = None,
    ):
        self.min_persistence = min_persistence
        self.channel = channel
        self._backend = None

    def _get_backend(self):
        if self._backend is None:
            try:
                from pina._src.topology.backends import GudhiBackend
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
        channel = kwargs.pop("channel", self.channel)
        min_pers = kwargs.pop("min_persistence", self.min_persistence)

        backend = self._get_backend()

        if backend.min_persistence != min_pers:
            backend.min_persistence = min_pers

        return backend.compute(tensor, channel=channel, min_persistence=min_pers, **kwargs)

    @property
    def backend_name(self) -> str:
        return self._get_backend().name
