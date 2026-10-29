"""Module for the Condition interface."""

from abc import ABCMeta, abstractmethod


class ConditionInterface(metaclass=ABCMeta):
    """
    Abstract interface for all conditions.

    A condition binds a set of data points (inputs, targets, equations, ...)
    to a problem. The data module works with conditions through this interface
    only: it shuffles sample ids and asks the condition to :meth:`materialize`
    the requested ids into a batch. Storage and batching are therefore a
    condition concern, and new data types can be added without touching the
    data module.
    """

    @abstractmethod
    def __len__(self):
        """
        Return the number of data points in the condition.

        :return: The number of data points.
        :rtype: int
        """

    @abstractmethod
    def __getitem__(self, idx):
        """
        Materialize the data points at the specified index (or indices).

        :param idx: The index of the data point to retrieve.
        :type idx: int | slice | list[int] | torch.Tensor
        :return: The materialized batch.
        :rtype: dict
        """

    @abstractmethod
    def store_data(self, **kwargs):
        """
        Store the data for the condition in a suitable format.

        :param dict kwargs: The keyword arguments containing the data to be
            stored.
        :return: The stored data.
        :rtype: Any
        """

    @abstractmethod
    def materialize(self, ids, device=None, batch_fn=None):
        """
        Build a batch from the given data point ids.

        :param ids: The ids of the data points to batch.
        :type ids: int | list[int] | torch.Tensor
        :param device: The target device for the batch. Default is ``None``.
        :param batch_fn: Optional callable overriding the default batched
            construction. It receives the selected raw data and the ids and
            returns the batch. Default is ``None``.
        :return: The materialized batch.
        :rtype: dict
        """

    @abstractmethod
    def evaluate(self, batch, solver):
        """
        Evaluate the residual of the condition on the given batch using the
        solver.

        This method computes the non-aggregated, element-wise residual of the
        condition. A forward pass of the solver's model is performed on the
        input samples, and the condition residual is evaluated accordingly.

        The returned tensor is not reduced, preserving the per-sample residual
        values.

        :param dict batch: The batch containing the data required by the
            condition evaluation.
        :param BaseSolver solver: The solver used to perform the forward pass
            and compute the residual.
        :return: The non-aggregated residual tensor.
        :rtype: torch.Tensor | LabelTensor
        """

    @property
    @abstractmethod
    def problem(self):
        """
        The problem associated with this condition.

        :return: The problem associated with this condition.
        :rtype: BaseProblem
        """

    @problem.setter
    @abstractmethod
    def problem(self, value):
        """
        Set the problem associated with this condition.

        :param BaseProblem value: The problem to associate with this condition.
        """
