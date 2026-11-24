"""Trainer utilities built on top of the PyTorch Lightning Trainer class."""

import warnings

import lightning
import torch

from pina._src.core.data_module import DataModule
from pina._src.core.utils import (
    check_consistency,
    custom_warning_format,
)
from pina._src.solver.base_solver import BaseSolver
from pina._src.solver.mixin.physics_informed_mixin import PhysicsInformedMixin

# Set custom warning format and filter warnings
warnings.formatwarning = custom_warning_format
warnings.filterwarnings("always", category=UserWarning)


class Trainer(lightning.pytorch.Trainer):
    """
    PINA-specific extension of :class:`lightning.pytorch.Trainer`.

    The trainer configures solver execution, logging, device placement for
    unknown parameters, and gradient tracking requirements for
    physics-informed solvers.

    Dataset creation is delegated to :class:`~pina.DataModule`. By default the
    trainer builds one from ``train_size``, ``val_size``, ``test_size``,
    ``batch_size``, and ``shuffle``. For advanced settings, such as per
    condition mini-batch rules or custom dataloader options, build a
    :class:`~pina.DataModule` explicitly and pass it through ``datamodule``.

    :Example:

        >>> from pina import DataModule, Trainer
        >>> from pina.problem import BaseProblem
        >>> # Trainer builds the data module from the split and batch options:
        >>> trainer = Trainer(
        ...     solver=solver_instance,
        ...     batch_size=32,
        ...     train_size=0.8,
        ...     val_size=0.1,
        ...     test_size=0.1,
        ...     max_epochs=100,
        ... )
        >>> # Advanced usage: provide a fully configured data module:
        >>> data_module = DataModule(
        ...     problem=solver_instance.problem,
        ...     batch_size=32,
        ...     mini_batch_rule={"boundary": {"batch_size": 16}},
        ... )
        >>> trainer = Trainer(solver=solver_instance, datamodule=data_module)
    """

    def __init__(
        self,
        solver,
        train_size=None,
        test_size=None,
        val_size=None,
        batch_size=None,
        shuffle=None,
        datamodule=None,
        *,
        batching_mode=None,
        num_workers=None,
        pin_memory=None,
        problem_setting=None,
        dataloader_cls=None,
        collate_fn=None,
        **kwargs,
    ):
        """
        Initialization of the :class:`Trainer` class.

        :param BaseSolver solver: The solver used to train, validate, and test
            the associated problem.
        :param float train_size: The fraction of samples assigned to the
            training split. Must belong to the interval ``[0, 1]``. If
            ``datamodule`` is not provided, defaults to ``1.0``.
        :param float test_size: The fraction of samples assigned to the test
            split. Must belong to the interval ``[0, 1]``. If ``datamodule``
            is not provided, defaults to ``0.0``.
        :param float val_size: The fraction of samples assigned to the
            validation split. Must belong to the interval ``[0, 1]``. If
            ``datamodule`` is not provided, defaults to ``0.0``.
        :param batch_size: The number of samples per batch. If ``None``, the
            entire dataset is processed as a single batch. If ``datamodule``
            is not provided, it is forwarded to the created data module.
            Default is ``None``.
        :type batch_size: int | None
        :param bool shuffle: Whether condition samples should be shuffled
            before splitting. If ``datamodule`` is not provided, defaults to
            ``True``.
        :param DataModule datamodule: A fully configured data module. When
            provided, ``train_size``, ``test_size``, ``val_size``,
            ``batch_size``, and ``shuffle`` are ignored with a warning.
            Default is ``None``.
        :param batching_mode: Deprecated and ignored. Configure it on the
            :class:`~pina.DataModule` instead.
        :param num_workers: Deprecated and ignored. Configure it on the
            :class:`~pina.DataModule` instead.
        :param pin_memory: Deprecated and ignored. Configure it on the
            :class:`~pina.DataModule` instead.
        :param problem_setting: Deprecated and ignored. Use the per condition
            ``mini_batch_rule`` of :class:`~pina.DataModule` instead.
        :param dataloader_cls: Deprecated and ignored.
        :param collate_fn: Deprecated and ignored.
        :param dict kwargs: Additional keyword arguments forwarded to the
            Lightning trainer.
        :raises ValueError: If ``solver`` is not a PINA solver.
        :raises ValueError: If ``datamodule`` is not a
            :class:`~pina.DataModule` or refers to a different problem.
        :raises ValueError: If ``train_size``, ``val_size``, or ``test_size``
            is not a float in the interval ``[0, 1]``.
        :raises ValueError: If the sum of ``train_size``, ``val_size``, and
            ``test_size`` is not equal to 1.
        :raises ValueError: If ``shuffle`` is not a boolean.
        :raises AssertionError: If ``batch_size``, when provided, is not a
            positive integer.
        :raises UserWarning: If the split or batch options are provided
            together with ``datamodule``.
        :raises DeprecationWarning: If a deprecated data option is provided.
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

        # Warn about deprecated data options, which are now configured on the
        # data module
        deprecated_kwargs = {
            "batching_mode": batching_mode,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "problem_setting": problem_setting,
            "dataloader_cls": dataloader_cls,
            "collate_fn": collate_fn,
        }
        for name, value in deprecated_kwargs.items():
            if value is not None:
                warnings.warn(
                    f"`{name}` is deprecated in `Trainer` and will be removed "
                    "in 0.4. It is ignored: configure it on a `pina.DataModule` "
                    "and pass it through `datamodule=` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        # Check consistency
        check_consistency(solver, BaseSolver)

        # Resolve the data module
        if datamodule is None:
            self.data_module = DataModule(
                problem=solver.problem,
                train_size=1.0 if train_size is None else train_size,
                val_size=0.0 if val_size is None else val_size,
                test_size=0.0 if test_size is None else test_size,
                batch_size=batch_size,
                shuffle=True if shuffle is None else shuffle,
            )
        else:
            check_consistency(datamodule, DataModule)

            if datamodule.problem is not solver.problem:
                raise ValueError(
                    "The provided `datamodule` refers to a different problem "
                    "than the solver. Make sure `datamodule.problem` is the "
                    "same object as `solver.problem`."
                )

            conflicting = [
                name
                for name, value in (
                    ("train_size", train_size),
                    ("val_size", val_size),
                    ("test_size", test_size),
                    ("batch_size", batch_size),
                    ("shuffle", shuffle),
                )
                if value is not None
            ]
            if conflicting:
                warnings.warn(
                    "The following arguments are ignored because a "
                    f"`datamodule` was provided: {conflicting}. Configure "
                    "them on the `pina.DataModule` instead.",
                    UserWarning,
                    stacklevel=2,
                )

            self.data_module = datamodule

        # Set inference mode to false when using physics-informed mixin
        if isinstance(solver, PhysicsInformedMixin):
            kwargs["inference_mode"] = False

        # Set log_every_n_steps to 0 if the whole dataset is a single batch,
        # otherwise default
        kwargs["log_every_n_steps"] = (
            0
            if not self.data_module.has_batch_size
            else kwargs.get("log_every_n_steps", 50)
        )

        # Set default value for enable_progress_bar to True if not provided
        kwargs.setdefault("enable_progress_bar", True)

        # Initialize the parent class with the provided keyword arguments
        super().__init__(**kwargs)

        # Initialize the class attributes
        self.solver = solver
        self.batch_size = self.data_module.batch_size

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

        # Set logging kwargs
        self.logging_kwargs = {
            "sync_dist": bool(
                len(self._accelerator_connector._parallel_devices) > 1
            ),
            "on_step": bool(kwargs["log_every_n_steps"] > 0),
            "prog_bar": bool(kwargs["enable_progress_bar"]),
            "on_epoch": True,
        }

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
