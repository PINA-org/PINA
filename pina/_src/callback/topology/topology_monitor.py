"""
TopologyMonitor: PyTorch Lightning callback for topological health checking.
"""
import torch
import logging
import warnings
from typing import Optional, Any, List, Dict
from lightning.pytorch import Callback, Trainer, LightningModule
from pina._src.callback.topology.profiler import TopologicalProfiler
from pina._src.callback.topology.topology_result import TopologyResult

logger = logging.getLogger(__name__)


class TopologyMonitor(Callback):
    """
    PyTorch Lightning callback for monitoring topological health during training.

    :param int expected_beta_0: Expected number of connected components. Default is 1.
    :param int expected_beta_1: Expected number of holes. Default is 0.
    :param int monitor_freq: Run topology check every N validation epochs. Default is 10.
    :param float threshold_beta_0: Max allowed β₀ before triggering alert.
    :param float threshold_beta_1: Max allowed β₁ before triggering alert.
    :param int warmup_epochs: Epochs to wait before monitoring starts. Default is 10.
    :param str mode: Alert mode: "warn", "stop", or "log".
    :param float min_persistence: Minimum lifespan for features. Default is 0.1.
    :param int channel: Channel index to monitor.
    :param callable collect_predictions: Custom extraction callable(batch, batch_idx, outputs).
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

        self.profiler = TopologicalProfiler(
            min_persistence=min_persistence,
            channel=channel
        )
        self._prediction_buffer: List[torch.Tensor] = []

    def _extract_input(self, batch: Any) -> Optional[torch.Tensor]:
        """Robustly extract the input tensor from various batch structures."""
        if isinstance(batch, list) and len(batch) > 0:
            item = batch[0]
            if isinstance(item, tuple) and len(item) == 2:
                _, data_dict = item
                if isinstance(data_dict, dict):
                    for key in ['input', 'x', 'inputs', 'prediction']:
                        if key in data_dict:
                            return data_dict[key]
        if isinstance(batch, dict):
            for key in ['input', 'x', 'inputs', 'prediction']:
                if key in batch:
                    return batch[key]
            for val in batch.values():
                if isinstance(val, torch.Tensor):
                    return val
        if isinstance(batch, (tuple, list)) and len(batch) >= 1:
            if isinstance(batch[0], torch.Tensor):
                return batch[0]
        if isinstance(batch, torch.Tensor):
            return batch
        return None

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect predictions on device (no CPU transfer yet)."""
        if trainer.current_epoch < self.warmup_epochs:
            return
        if (trainer.current_epoch - self.warmup_epochs) % self.monitor_freq != 0:
            return

        if self.collect_predictions is not None:
            pred = self.collect_predictions(batch, batch_idx, outputs)
            if pred is not None and isinstance(pred, torch.Tensor):
                self._prediction_buffer.append(pred)
            return

        pred = self._extract_input(batch)
        if pred is not None and isinstance(pred, torch.Tensor):
            self._prediction_buffer.append(pred)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Run topology check using weighted global statistics across DDP ranks."""
        if trainer.current_epoch < self.warmup_epochs:
            self._prediction_buffer = []
            return
        if (trainer.current_epoch - self.warmup_epochs) % self.monitor_freq != 0:
            self._prediction_buffer = []
            return

        local_len = len(self._prediction_buffer)
        
        # DDP-safe global empty check
        if trainer.world_size > 1:
            import torch.distributed as dist
            len_tensor = torch.tensor(local_len, device=pl_module.device)
            dist.all_reduce(len_tensor, op=dist.ReduceOp.SUM)
            global_len = len_tensor.item()
        else:
            global_len = local_len

        if global_len == 0:
            self._prediction_buffer = []
            return

        # Local computation
        if local_len == 0:
            N, sum_b0, sum_sq_b0, sum_b1, sum_sq_b1 = 0, 0.0, 0.0, 0.0, 0.0
        else:
            preds = torch.cat(self._prediction_buffer, dim=0).detach().cpu()
            result = self.profiler.compute(preds)
            if not result.success:
                warnings.warn(f"Topology failed: {result.error_msg}")
                self._prediction_buffer = []
                return
            N = len(result.per_sample_beta_0)
            sum_b0 = sum(result.per_sample_beta_0)
            sum_sq_b0 = sum(x**2 for x in result.per_sample_beta_0)
            if result.beta_1_mean is not None:
                sum_b1 = sum(result.per_sample_beta_1)
                sum_sq_b1 = sum(x**2 for x in result.per_sample_beta_1)
            else:
                sum_b1, sum_sq_b1 = 0.0, 0.0
            self._prediction_buffer = []

        # AllReduce weighted statistics
        if trainer.world_size > 1:
            import torch.distributed as dist
            N_t = torch.tensor(N, device=pl_module.device, dtype=torch.float32)
            sum_b0_t = torch.tensor(sum_b0, device=pl_module.device, dtype=torch.float32)
            sum_sq_b0_t = torch.tensor(sum_sq_b0, device=pl_module.device, dtype=torch.float32)
            sum_b1_t = torch.tensor(sum_b1, device=pl_module.device, dtype=torch.float32)
            sum_sq_b1_t = torch.tensor(sum_sq_b1, device=pl_module.device, dtype=torch.float32)
            dist.all_reduce(N_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_b0_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_sq_b0_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_b1_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_sq_b1_t, op=dist.ReduceOp.SUM)
            N, sum_b0, sum_sq_b0, sum_b1, sum_sq_b1 = N_t.item(), sum_b0_t.item(), sum_sq_b0_t.item(), sum_b1_t.item(), sum_sq_b1_t.item()

        if N == 0:
            return

        # Weighted global statistics (Law of Total Variance)
        mean_b0 = sum_b0 / N
        mean_sq_b0 = sum_sq_b0 / N
        std_b0 = max(0.0, (mean_sq_b0 - mean_b0**2) ** 0.5)
        
        if sum_b1 != 0.0:
            mean_b1 = sum_b1 / N
            mean_sq_b1 = sum_sq_b1 / N
            std_b1 = max(0.0, (mean_sq_b1 - mean_b1**2) ** 0.5)
        else:
            mean_b1, std_b1 = None, None

        # Pack results
        global_result = TopologyResult(
            beta_0_mean=mean_b0,
            beta_1_mean=mean_b1,
            beta_0_std=std_b0,
            beta_1_std=std_b1,
            beta_0_max=0.0,
            beta_1_max=None,
            per_sample_beta_0=[],
            per_sample_beta_1=[],
            success=True,
            backend_name="gudhi",
            metadata={"n_samples": int(N)}
        )

        self._log_results(pl_module, global_result)

        # Alert logic
        trigger_beta_0 = global_result.beta_0_mean > self.threshold_beta_0
        trigger_beta_1 = (global_result.beta_1_mean is not None and
                          global_result.beta_1_mean > self.threshold_beta_1)
        should_stop_local = trigger_beta_0 or trigger_beta_1

        if trainer.world_size > 1:
            import torch.distributed as dist
            stop_tensor = torch.tensor([1.0 if should_stop_local else 0.0], device=pl_module.device)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.SUM)
            should_stop_global = stop_tensor.item() > 0
        else:
            should_stop_global = should_stop_local

        if should_stop_global:
            alert = self._generate_alert(global_result, trainer.current_epoch)
            if self.mode == "stop":
                logger.error(alert)
                trainer.should_stop = True
            elif self.mode == "warn":
                warnings.warn(alert)

    def _log_results(self, pl_module: LightningModule, result: TopologyResult):
        device = pl_module.device
        pl_module.log("topology/beta_0_mean", torch.tensor(result.beta_0_mean, device=device), sync_dist=True)
        pl_module.log("topology/beta_0_std", torch.tensor(result.beta_0_std, device=device), sync_dist=True)
        if result.beta_1_mean is not None:
            pl_module.log("topology/beta_1_mean", torch.tensor(result.beta_1_mean, device=device), sync_dist=True)
            pl_module.log("topology/beta_1_std", torch.tensor(result.beta_1_std, device=device), sync_dist=True)

    def _generate_alert(self, result: TopologyResult, epoch: int) -> str:
        msg = f"\n{'='*60}\n[TOPOLOGY ALERT] Epoch: {epoch}\n"
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