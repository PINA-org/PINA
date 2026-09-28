"""Trainer utilities built on top of the PyTorch Lightning Trainer class."""

import warnings
import torch
import lightning
from pina._src.solver.mixin.physics_informed_mixin import PhysicsInformedMixin
from pina._src.solver.base_solver import BaseSolver
from pina._src.data.data_module import DataModule
from pina._src.core.utils import (
    check_consistency,
    custom_warning_format,
    check_positive_integer,
)

# Set custom warning format and filter warnings
warnings.formatwarning = custom_warning_format
warnings.filterwarnings("always", category=UserWarning)


class Trainer(lightning.pytorch.Trainer):
    """
    PINA-specific extension of :class:`lightning.pytorch.Trainer`.

    The trainer configures solver execution, dataset splitting, batching,
    logging, device placement for unknown parameters, and gradient tracking
    requirements for physics-informed solvers.

    :Example:

        >>> from pina.solver import BaseSolver
        >>> from pina.problem import BaseProblem
        >>> import torch
        >>> # Trainer is typically used with a solver and problem:
        >>> # trainer = Trainer(solver=solver_instance, batch_size=32,
        >>> #     max_epochs=100)
        >>> # trainer.train()
    """

    # Available batching modes
    _AVAIL_BATCHING_MODES = {
        "common_batch_size",
        "proportional",
    }

    def __init__(
        self,
        solver,
        batch_size=None,
        train_size=1.0,
        test_size=0.0,
        val_size=0.0,
        batching_mode="common_batch_size",
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        dataloader_cls=None,
        collate_fn=None,
        **kwargs,
    ):
        """
        Initialization of the :class:`Trainer` class.

        :param BaseSolver solver: The solver used to train, validate, and test
            the associated problem.
        :param int batch_size: The number of samples per batch. If ``None``, the
            entire dataset is processed as a single batch. Default is ``None``.
        :param float train_size: The fraction of samples assigned to the
            training split. Must belong to the interval ``[0, 1]``.
            Default is ``1.0``.
        :param float val_size: The fraction of samples assigned to the
            validation split. Must belong to the interval ``[0, 1]``.
            Default is ``0.0``.
        :param float test_size: The fraction of samples assigned to the test
            split. Must belong to the interval ``[0, 1]``. Default is ``0.0``.
        :param str batching_mode: The strategy used to aggregate batches across
            dataloaders. Available options are ``"common_batch_size"`` for
            uniform batch sizes across conditions, and ``"proportional"`` for
            batch sizes proportional to dataset sizes.
            Default is ``"common_batch_size"``.
        :param int num_workers: The number of worker processes used by
            dataloaders. Default is ``0`` for sequential loading.
        :param bool pin_memory: Whether pinned memory should be enabled during
            data loading. Default is ``False``.
        :param bool shuffle: Whether condition samples should be shuffled before
            splitting. Default is ``True``.
        :param dataloader_cls: The dataloader class to use for each condition,
            defaulting to :class:`torch.utils.data.DataLoader`. It can either
            be a single class applied to all conditions, or a mapping between
            condition names and dataloader classes. Custom classes must accept
            the ``batch_size``, ``sampler``, ``num_workers``, and ``pin_memory``
            arguments and yield batches of sample ids. Default is ``None``.
        :type dataloader_cls: type | dict[str, type] | None
        :param collate_fn: Optional callable (or mapping between condition
            names and callables) used to collapse the raw data selected by a
            condition into a batch. It receives the selected raw data and the
            ids, and returns the batch. Default is ``None``.
        :type collate_fn: Callable | dict[str, Callable] | None
        :param dict kwargs: Additional keyword arguments forwarded to the
            Lightning trainer.
        :raises ValueError: If ``solver`` is not a PINA solver.
        :raises ValueError: If ``train_size``, ``val_size``, or ``test_size`` is
            not a float in the interval ``[0, 1]``.
        :raises ValueError: If the sum of ``train_size``, ``val_size``, and
            ``test_size`` is not equal to 1.
        :raises ValueError: If ``pin_memory`` or ``shuffle`` is not a boolean.
        :raises AssertionError: If ``num_workers`` is a negative integer.
        :raises ValueError: If ``batch_size``, when provided, is not a positive
            integer.
        :raises ValueError: If ``batching_mode`` is not one of the available
             options.
        :raises ValueError: If ``dataloader_cls`` or ``collate_fn`` is a mapping
            referring to unknown conditions.
        :raises UserWarning: If the provided ``batching_mode`` is incompatible
            with the ``batch_size``.
        :raises RuntimeError: If any domain in the problem has not been
            discretised.
        """
        # Backward compatibility: compile has been removed
        if "compile" in kwargs:
            warnings.warn(
                "`compile` is deprecated and no longer used. Compilation is "
                "now disabled and the argument will be ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs.pop("compile")

        # Check consistency
        check_consistency(solver, BaseSolver)
        check_consistency(train_size, float)
        check_consistency(test_size, float)
        check_consistency(val_size, float)
        check_consistency(pin_memory, bool)
        check_consistency(shuffle, bool)
        check_positive_integer(num_workers, strict=False)
        if batch_size is not None:
            check_positive_integer(batch_size, strict=True)

        # Check that train_size, test_size and val_size sum to 1
        total = train_size + val_size + test_size
        if not torch.isclose(torch.tensor(total), torch.tensor(1.0)):
            raise ValueError(
                "`train_size`, `val_size`, and `test_size` must sum to 1."
            )

        # Check consistency
        if batching_mode not in self._AVAIL_BATCHING_MODES:
            raise ValueError(
                f"Invalid batching mode '{batching_mode}'. "
                f"Expected one of: {sorted(self._AVAIL_BATCHING_MODES)}."
            )

        # Normalize dataloader_cls and collate_fn to per-condition mappings
        conditions = set(solver.problem.conditions)
        dataloader_cls, collate_fn = self._resolve_dataloader_options(
            dataloader_cls=dataloader_cls,
            collate_fn=collate_fn,
            conditions=conditions,
        )

        # Set inference mode to false when usiing physics-informed mixin
        if isinstance(solver, PhysicsInformedMixin):
            kwargs["inference_mode"] = False

        # Set log_every_n_steps to 0 if batch_size is None, otherwise default
        kwargs["log_every_n_steps"] = (
            0 if batch_size is None else kwargs.get("log_every_n_steps", 50)
        )

        # Set default value for enable_progress_bar to True if not provided
        kwargs.setdefault("enable_progress_bar", True)

        # Initialize the parent class with the provided keyword arguments
        super().__init__(**kwargs)

        # Raise warning if batch size and batching mode are incompatible
        if batch_size is None and batching_mode != "common_batch_size":
            warnings.warn(
                f"Batching mode '{batching_mode}' is ignored when the batch "
                "size is None. Setting batching_mode to 'common_batch_size'.",
                UserWarning,
            )

            # Set batching mode to common_batch_size if incompatible
            batching_mode = "common_batch_size"

        # Raise warning if batch size and batching mode are incompatible
        if (
            batch_size is not None
            and batching_mode == "proportional"
            and batch_size <= len(solver.problem.conditions)
        ):
            warnings.warn(
                "Batching mode 'proportional' requires the batch size to be "
                "larger than the number of conditions. Setting batching_mode "
                "to 'common_batch_size'.",
                UserWarning,
            )

            # Set batching mode to common_batch_size if incompatible
            batching_mode = "common_batch_size"

        # Initialize the class attributes
        self.solver = solver
        self.batch_size = batch_size

        # Move the unknown parameters to the correct device
        self._move_to_device()

        # Check that all domains are discretised, otherwise raise an error
        if not self.solver.problem.are_all_domains_discretised:

            # Get the list of sampled domains from the problem
            sampled_domains = self.solver.problem.discretised_domains

            # Create a status message for each domain
            status = "\n".join(
                f"    - Domain '{name}': "
                f"{'sampled' if name in sampled_domains else 'not sampled'}"
                for name in self.solver.problem.domains
            )

            # Raise an error with the status of each domain
            raise RuntimeError(
                "Cannot create the Trainer because some domains have not been "
                f"sampled. Domain status:\n{status}"
            )

        # Create the data module
        self.data_module = DataModule(
            problem=self.solver.problem,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            batch_size=self.batch_size,
            batching_mode=batching_mode,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            dataloader_cls=dataloader_cls,
            collate_fn=collate_fn,
        )

        # Set logging kwargs
        self.logging_kwargs = {
            "sync_dist": bool(
                len(self._accelerator_connector._parallel_devices) > 1
            ),
            "on_step": bool(kwargs["log_every_n_steps"] > 0),
            "prog_bar": bool(kwargs["enable_progress_bar"]),
            "on_epoch": True,
        }

    @staticmethod
    def _resolve_dataloader_options(dataloader_cls, collate_fn, conditions):
        """
        Normalize the dataloader and collate options to per-condition mappings.

        :param dataloader_cls: The dataloader class to use. It can be a single
            class applied to all conditions, or a mapping between condition
            names and dataloader classes.
        :type dataloader_cls: type | dict[str, type] | None
        :param collate_fn: The collate function to use. It can be a single
            callable applied to all conditions, or a mapping between condition
            names and callables.
        :type collate_fn: Callable | dict[str, Callable] | None
        :param set[str] conditions: The names of the problem conditions.
        :raises ValueError: If a mapping refers to unknown conditions.
        :return: The normalized per-condition mappings.
        :rtype: tuple[dict[str, type | None], dict[str, Callable | None]]
        """

        def _normalize(value, label):
            if value is None:
                return {name: None for name in conditions}

            if isinstance(value, dict):
                if not set(value).issubset(conditions):
                    raise ValueError(
                        f"The {label} mapping refers to unknown conditions. "
                        f"Valid conditions are: {sorted(conditions)}."
                    )
                return value

            return {name: value for name in conditions}

        dataloader_cls = _normalize(dataloader_cls, "dataloader_cls")
        collate_fn = _normalize(collate_fn, "collate_fn")

        return dataloader_cls, collate_fn

    def _move_to_device(self):
        """
        Move problem unknown parameters to the trainer device.

        If the associated problem defines ``unknown_parameters``, each parameter
        is moved to the first device configured by the Lightning accelerator
        connector.
        """
        # Get the device from the accelerator connector
        device = self._accelerator_connector._parallel_devices[0]

        # Get the problem instance from the solver
        problem = self.solver.problem

        # Move the unknown parameters to the correct device if they exist
        if hasattr(problem, "unknown_parameters"):
            for key in problem.unknown_parameters:
                problem.unknown_parameters[key] = torch.nn.Parameter(
                    problem.unknown_parameters[key].data.to(device)
                )

    def train(self, **kwargs):
        """
        Fit the solver using the trainer data module.

        :param dict kwargs: Additional keyword arguments forwarded to the
            Lightning trainer ``fit`` method.
        :return: Result returned by Lightning's ``fit`` method.
        :rtype: Any
        """
        return super().fit(self.solver, datamodule=self.data_module, **kwargs)

    def test(self, **kwargs):
        """
        Test the solver using the trainer data module.

        :param dict kwargs: Additional keyword arguments forwarded to the
            Lightning trainer ``test`` method.
        :return: Result returned by Lightning's ``test`` method.
        :rtype: Any
        """
        return super().test(self.solver, datamodule=self.data_module, **kwargs)

    @property
    def solver(self):
        """
        Return the solver attached to the trainer.

        :return: The solver used by the trainer.
        :rtype: BaseSolver
        """
        return self._solver

    @solver.setter
    def solver(self, solver):
        """
        Set the solver attached to the trainer.

        :param BaseSolver solver: The solver instance to attach.
        """
        self._solver = solver
