"""
TopologyMonitor: PyTorch Lightning callback for topological health checking.
Uses the TopologicalProfiler with lazy-loaded GUDHI backend.
"""
import torch
import logging
import warnings
from pytorch_lightning import Callback, Trainer, LightningModule
from typing import Optional, Union, List
from .profiler import TopologicalProfiler
from .backends import TopologyResult

# Setup logger for this module
logger = logging.getLogger(__name__)

class TopologyMonitor(Callback):
    """
    Monitors topological health of model outputs during validation.
    
    Args:
        expected_beta_0: Expected number of connected components.
        expected_beta_1: Expected number of holes.
        monitor_freq: Run topology check every N validation epochs.
        threshold_beta_0: Max allowed beta_0 before triggering alert.
        threshold_beta_1: Max allowed beta_1 before triggering alert.
        warmup_epochs: Number of epochs to wait before monitoring starts.
        mode: 'warn' (log warning), 'stop' (graceful stop), 'log' (just log).
        min_persistence: Minimum lifespan for features to be counted.
        channel: Channel index to monitor (for multi-variate outputs).
        collect_predictions: Function to extract predictions from the model.
            If None, the user must override `on_validation_epoch_end`.
    """
    def __init__(
        self,
        expected_beta_0: int = 1,
        expected_beta_1: int = 0,
        monitor_freq: int = 10,
        threshold_beta_0: Optional[float] = None,
        threshold_beta_1: Optional[float] = None,
        warmup_epochs: int = 10,
        mode: str = "warn",
        min_persistence: float = 0.1,
        channel: Optional[int] = None,
        collect_predictions: Optional[callable] = None,
    ):
        super().__init__()
        self.expected_beta_0 = expected_beta_0
        self.expected_beta_1 = expected_beta_1
        self.monitor_freq = monitor_freq
        self.threshold_beta_0 = threshold_beta_0 if threshold_beta_0 is not None else expected_beta_0 + 0.5
        self.threshold_beta_1 = threshold_beta_1 if threshold_beta_1 is not None else expected_beta_1 + 0.5
        self.warmup_epochs = warmup_epochs
        self.mode = mode
        self.collect_predictions = collect_predictions
        
        # Lazy-load the profiler (GUDHI error deferred until first use)
        self.profiler = TopologicalProfiler(
            min_persistence=min_persistence,
            channel=channel
        )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Runs topology check at the end of each validation epoch."""
        # FIX 3: Use Lightning's state directly. No manual epoch counter.
        if trainer.current_epoch < self.warmup_epochs:
            return
        
        # FIX 3: Sync frequency with actual training schedule
        if (trainer.current_epoch - self.warmup_epochs) % self.monitor_freq != 0:
            return
        
        # Get predictions
        predictions = self._get_predictions(trainer, pl_module)
        if predictions is None:
            warnings.warn(
                "No predictions available for TopologyMonitor. "
                "Override `_get_predictions` or provide `collect_predictions`."
            )
            return
        
        # Compute topology
        result = self.profiler.compute(predictions)
        if not result.success:
            warnings.warn(f"Topology computation failed: {result.error_msg}")
            return
        
        # FIX 2: Use Lightning's agnostic logging interface (pl_module.log)
        self._log_results(pl_module, result)
        
        # FIX 1: Check both beta_0 and beta_1 thresholds independently
        trigger_beta_0 = result.beta_0_mean > self.threshold_beta_0
        trigger_beta_1 = (result.beta_1_mean is not None and 
                          result.beta_1_mean > self.threshold_beta_1)
        
        if trigger_beta_0 or trigger_beta_1:
            alert = self._generate_alert(result)
            
            if self.mode == "stop":
                # FIX 4: Graceful shutdown: log error, set flag, do NOT raise exception
                logger.error(alert)
                trainer.should_stop = True
                # Let Lightning handle the shutdown gracefully
            elif self.mode == "warn":
                warnings.warn(alert)
            # else 'log' mode: just log (already logged via `_log_results`)

    def _get_predictions(self, trainer: Trainer, pl_module: LightningModule) -> Optional[torch.Tensor]:
        """
        Extracts predictions from the model.
        If `collect_predictions` is provided, use it.
        Otherwise, attempt to get the last batch from the validation dataloader.
        """
        if self.collect_predictions is not None:
            return self.collect_predictions(trainer, pl_module)
        
        # Fallback: warn user to implement collection
        return None

    def _log_results(self, pl_module: LightningModule, result: TopologyResult):
        """
        Logs topology metrics using PyTorch Lightning's agnostic interface.
        FIX 2: Works with TensorBoard, WandB, CSV, Comet, etc.
        """
        # Log mean, std, max for beta_0
        pl_module.log("topology/beta_0_mean", result.beta_0_mean, sync_dist=True)
        pl_module.log("topology/beta_0_std", result.beta_0_std, sync_dist=True)
        pl_module.log("topology/beta_0_max", float(result.beta_0_max), sync_dist=True)
        
        # Log beta_1 only if available
        if result.beta_1_mean is not None:
            pl_module.log("topology/beta_1_mean", result.beta_1_mean, sync_dist=True)
            pl_module.log("topology/beta_1_std", result.beta_1_std, sync_dist=True)
            if result.beta_1_max is not None:
                pl_module.log("topology/beta_1_max", float(result.beta_1_max), sync_dist=True)

    def _generate_alert(self, result: TopologyResult) -> str:
        """
        Generates an actionable alert message.
        FIX 5: Diagnostics are based on the thresholds, not hardcoded offsets.
        """
        msg = (
            f"\n{'='*60}\n"
            f"[TOPOLOGY ALERT] Epoch: {self._current_epoch}\n"
        )
        
        if result.beta_0_mean > self.threshold_beta_0:
            msg += f"β₀ = {result.beta_0_mean:.2f} (threshold: {self.threshold_beta_0})\n"
        if result.beta_1_mean is not None and result.beta_1_mean > self.threshold_beta_1:
            msg += f"β₁ = {result.beta_1_mean:.2f} (threshold: {self.threshold_beta_1})\n"
        
        msg += "\nDiagnostics:\n"
        # FIX 5: Base diagnostic messages on thresholds, not expected+1
        if result.beta_0_mean > self.threshold_beta_0:
            msg += "  - Excessive connected components detected. Ghost islands may be present.\n"
        if result.beta_1_mean is not None and result.beta_1_mean > self.threshold_beta_1:
            msg += "  - Spurious holes detected. Check boundary conditions.\n"
        
        msg += "\nSuggested actions:\n"
        suggestions = []
        if result.beta_0_mean > self.threshold_beta_0:
            suggestions.append("Increase `n_modes` to capture high-frequency edges.")
        if result.beta_1_mean is not None and result.beta_1_mean > self.threshold_beta_1:
            suggestions.append("Check boundary conditions; spurious loops may arise from BC mismatches.")
        if not suggestions:
            suggestions.append("Reduce learning_rate to stabilize spectral weight convergence.")
        msg += "  - " + "\n  - ".join(suggestions)
        msg += f"\n{'='*60}"
        return msg