"""
TopologyMonitor: PyTorch Lightning callback for topological health checking.
"""
import torch
import logging
import warnings
from lightning.pytorch import Callback, Trainer, LightningModule
from typing import Optional
from pina._src.topology.profiler import TopologicalProfiler
from pina._src.topology.backends import TopologyResult

logger = logging.getLogger(__name__)

class TopologyMonitor(Callback):
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

        self.profiler = TopologicalProfiler(
            min_persistence=min_persistence,
            channel=channel
        )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.current_epoch < self.warmup_epochs:
            return

        if (trainer.current_epoch - self.warmup_epochs) % self.monitor_freq != 0:
            return

        predictions = self._get_predictions(trainer, pl_module)
        if predictions is None:
            warnings.warn(
                "No predictions available for TopologyMonitor. "
                "Override `_get_predictions` or provide `collect_predictions`."
            )
            return

        result = self.profiler.compute(predictions)
        if not result.success:
            warnings.warn(f"Topology computation failed: {result.error_msg}")
            return

        self._log_results(pl_module, result)

        trigger_beta_0 = result.beta_0_mean > self.threshold_beta_0
        trigger_beta_1 = (result.beta_1_mean is not None and
                          result.beta_1_mean > self.threshold_beta_1)

        if trigger_beta_0 or trigger_beta_1:
            alert = self._generate_alert(result, trainer.current_epoch)

            if self.mode == "stop":
                logger.error(alert)
                trainer.should_stop = True
            elif self.mode == "warn":
                warnings.warn(alert)

    def _get_predictions(self, trainer: Trainer, pl_module: LightningModule) -> Optional[torch.Tensor]:
        if self.collect_predictions is not None:
            return self.collect_predictions(trainer, pl_module)
        return None

    def _log_results(self, pl_module: LightningModule, result: TopologyResult):
        pl_module.log("topology/beta_0_mean", result.beta_0_mean, sync_dist=True)
        pl_module.log("topology/beta_0_std", result.beta_0_std, sync_dist=True)
        pl_module.log("topology/beta_0_max", float(result.beta_0_max), sync_dist=True)

        if result.beta_1_mean is not None:
            pl_module.log("topology/beta_1_mean", result.beta_1_mean, sync_dist=True)
            pl_module.log("topology/beta_1_std", result.beta_1_std, sync_dist=True)
            if result.beta_1_max is not None:
                pl_module.log("topology/beta_1_max", float(result.beta_1_max), sync_dist=True)

    def _generate_alert(self, result: TopologyResult, epoch: int) -> str:
        msg = (
            f"\n{'='*60}\n"
            f"[TOPOLOGY ALERT] Epoch: {epoch}\n"
        )

        if result.beta_0_mean > self.threshold_beta_0:
            msg += f"β₀ = {result.beta_0_mean:.2f} (threshold: {self.threshold_beta_0})\n"
        if result.beta_1_mean is not None and result.beta_1_mean > self.threshold_beta_1:
            msg += f"β₁ = {result.beta_1_mean:.2f} (threshold: {self.threshold_beta_1})\n"

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
