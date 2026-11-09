"""Module for the Condition interface."""

from abc import ABCMeta, abstractmethod


class ConditionInterface(metaclass=ABCMeta):
    """
    Abstract interface for all conditions.

    A condition binds a set of data points (inputs, targets, equations, ...)
    to a problem. The data module works with conditions through this interface
    only: it shuffles stored batch ids and asks the condition to
    :meth:`materialize` the requested batch. Storage and batching are a
    condition concern, and new data types can be added without touching the
    data module.
    """

    @abstractmethod
    def __len__(self):
        """
        Return the number of stored batches in the condition.

        :return: The number of stored batches.
        :rtype: int
        """

    @abstractmethod
    def __getitem__(self, idx):
        """
        Retrieve the stored batch at the specified index.

        :param idx: The index of the batch to retrieve. Tensor indices must
            contain a single integer value.
        :type idx: int | torch.Tensor
        :return: The materialized batch.
        :rtype: dict[str, Any] | pina.graph.Graph
        """

    @abstractmethod
    def store_data(self, **kwargs):
        """
        Store the data for the condition in a suitable format.

        :param dict[str, Any] kwargs: Storage settings interpreted by each
            implementation, such as ``batch_size`` for tensor conditions,
            ``n_subgraphs`` and ``num_hops`` for graph conditions, or temporal
            window settings for time-series conditions. Domain conditions
            instead accept replacement ``domain`` and ``equation`` values.
        :return: The stored batches, or ``None`` when an implementation only
            updates its configuration, as domain conditions do.
        :rtype: list[dict[str, Any]] | list[pina.graph.Graph] | None
        """

    @abstractmethod
    def materialize(self, ids, device=None, batch_fn=None):
        """
        Retrieve a stored batch by its identifier.

        :param ids: The batch identifier. Tensor identifiers must contain a
            single integer value. Concrete implementations name this ``id``.
        :type ids: int | torch.Tensor
        :param device: The target device. If ``None``, leave the batch on its
            existing device. Default is ``None``.
        :type device: str | torch.device | int | None
        :param batch_fn: The optional construction override reserved by this
            interface. If supported, it receives selected raw data and ids and
            returns a batch. ``None`` selects default construction. Current
            tensor and graph implementations omit this argument. Default is
            ``None``.
        :type batch_fn: Callable | None
        :return: The materialized batch.
        :rtype: dict[str, Any] | pina.graph.Graph
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

        :param batch: The stored tensor fields or graph required by the
            condition evaluation.
        :type batch: dict[str, Any] | pina.graph.Graph
        :param pina.solver.BaseSolver solver: The solver used to perform the
            forward pass and compute the residual.
        :return: The non-aggregated residual tensor.
        :rtype: torch.Tensor | LabelTensor
        """

    @property
    @abstractmethod
    def problem(self):
        """
        The problem associated with this condition.

        :return: The problem associated with this condition.
        :rtype: pina.problem.ProblemInterface
        """

    @problem.setter
    @abstractmethod
    def problem(self, value):
        """
        Set the problem associated with this condition.

        :param pina.problem.ProblemInterface value: The problem to associate
            with this condition.
        """
