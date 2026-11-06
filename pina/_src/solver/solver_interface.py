"""Module for the solver interface."""

from abc import ABCMeta, abstractmethod
import lightning


class SolverInterface(lightning.pytorch.LightningModule, metaclass=ABCMeta):
    """
    Abstract base class for PINA solvers. All specific solvers must inherit
    from this interface. This class extends
    :class:`~lightning.pytorch.core.LightningModule`, providing additional
    functionalities for defining and optimizing Deep Learning models.

    By inheriting from this base class, solvers gain access to built-in training
    loops, logging utilities, and optimization techniques.
    """

    @abstractmethod
    def training_step(self, batch, batch_idx):
        """
        Solver training step.

        :param batch: The condition-name/data pairs supplied by the data module.
        :type batch: list[tuple[str, dict[str, Any] | pina.graph.Graph |
            torch_geometric.data.Data]]
        :param int batch_idx: The index of the current batch.
        :return: The loss of the training step.
        :rtype: torch.Tensor
        """

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """
        Solver validation step.

        :param batch: The condition-name/data pairs supplied by the data module.
        :type batch: list[tuple[str, dict[str, Any] | pina.graph.Graph |
            torch_geometric.data.Data]]
        :param int batch_idx: The index of the current batch.
        :return: The loss of the validation step.
        :rtype: torch.Tensor
        """

    @abstractmethod
    def test_step(self, batch, batch_idx):
        """
        Solver test step.

        :param batch: The condition-name/data pairs supplied by the data module.
        :type batch: list[tuple[str, dict[str, Any] | pina.graph.Graph |
            torch_geometric.data.Data]]
        :param int batch_idx: The index of the current batch.
        :return: The loss of the test step.
        :rtype: torch.Tensor
        """

    @property
    @abstractmethod
    def problem(self):
        """
        The problem instance.

        :return: The problem instance.
        :rtype: pina.problem.BaseProblem | None
        """

    @property
    @abstractmethod
    def use_lt(self):
        """
        Whether the solver wraps forward with labeled input/output handling.

        :return: The use_lt attribute.
        :rtype: bool
        """

    @property
    @abstractmethod
    def weighting(self):
        """
        The weighting schema used by the solver.

        :return: The weighting schema used by the solver.
        :rtype: pina.weighting.BaseWeighting
        """

    @property
    @abstractmethod
    def loss(self):
        """
        The element-wise loss module used by the solver.

        :return: The element-wise loss module used by the solver.
        :rtype: torch.nn.modules.loss._Loss | pina.loss.BaseDualLoss
        """
