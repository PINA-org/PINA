"""
Topological Profiler: Extracts topology features from FNO outputs using GUDHI.
Implements lazy loading and state consistency.
"""
import torch
from typing import Optional, Dict, Any
from .backends import TopologyResult  # Only import the dataclass (safe)

class TopologicalProfiler:
    """
    Profiler that computes Betti numbers using GUDHI backend.
    Lazy-loads the backend to avoid import errors on library load.
    """
    def __init__(
        self,
        min_persistence: float = 0.1,
        channel: Optional[int] = None,
    ):
        """
        Args:
            min_persistence: Minimum lifespan for a feature to be counted.
            channel: Default channel to monitor (0-based index).
        """
        self.min_persistence = min_persistence
        self.channel = channel
        # FIX 1: Do NOT instantiate backend here. Use lazy loading.
        self._backend = None

    def _get_backend(self):
        """
        Lazy-loads the GUDHI backend.
        FIX 2: Error only happens when the user actually calls compute().
        """
        if self._backend is None:
            try:
                from .backends import GudhiBackend
                self._backend = GudhiBackend(min_persistence=self.min_persistence)
            except ImportError as e:
                raise ImportError(
                    "\n" + "=" * 60 + "\n"
                    "GUDHI BACKEND REQUIRED FOR TOPOLOGICAL PROFILING.\n"
                    "Morphological fallbacks are removed due to inaccuracy.\n"
                    "\n"
                    "Please install GUDHI:\n"
                    "    pip install gudhi\n"
                    "=" * 60
                ) from e
        return self._backend

    def compute(self, tensor: torch.Tensor, **kwargs) -> TopologyResult:
        """
        Compute Betti numbers from a tensor.
        
        Args:
            tensor: Input tensor (batch, ...).
            **kwargs: 
                channel: Override the default channel for this call.
                min_persistence: Override the default persistence for this call.
        """
        # FIX 3: Resolve channel override: kwargs take priority over self.channel
        channel = kwargs.pop("channel", self.channel)
        min_pers = kwargs.pop("min_persistence", self.min_persistence)

        # Get the backend (lazy load)
        backend = self._get_backend()

        # FIX 1: Sync state if the user changed the profiler's attributes
        if backend.min_persistence != min_pers:
            backend.min_persistence = min_pers

        # Pass the resolved parameters
        return backend.compute(tensor, channel=channel, min_persistence=min_pers, **kwargs)

    @property
    def backend_name(self) -> str:
        """Returns the name of the active backend."""
        return self._get_backend().name