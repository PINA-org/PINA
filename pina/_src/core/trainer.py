"""Trainer utilities built on top of the PyTorch Lightning Trainer class."""

import warnings
import torch
import lightning
from pina._src.solver.mixin.physics_informed_mixin import PhysicsInformedMixin
from pina._src.solver.base_solver import BaseSolver
from pina._src.core.data_module import DataModule
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

        >>> import torch
        >>> from pina import LabelTensor, Trainer
        >>> from pina.problem.zoo import SupervisedProblem
        >>> from pina.solver import SupervisedSingleModelSolver
        >>> x = torch.linspace(0.0, 1.0, 8).reshape(-1, 1)
        >>> problem = SupervisedProblem(
        ...     LabelTensor(x, ["x"]), LabelTensor(x.square(), ["y"]),
        ...     input_variables=["x"], output_variables=["y"],
        ... )
        >>> solver = SupervisedSingleModelSolver(
        ...     problem=problem, model=torch.nn.Linear(1, 1)
        ... )
        >>> trainer = Trainer(
        ...     solver=solver, max_epochs=1, accelerator="cpu", logger=False,
        ...     enable_checkpointing=False, enable_progress_bar=False,
        ...     enable_model_summary=False,
        ... )
        >>> trainer.train()
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

        :param pina.solver.BaseSolver solver: The solver used to train and test
            the associated problem. All problem domains must be discretised.
        :param batch_size: The positive number of tensor samples per condition
            batch. ``None`` stores each tensor condition in one batch; graph
            conditions use their subgraph sampling settings instead. Default
            is ``None``.
        :type batch_size: int | None
        :param float train_size: The fraction of each condition's stored batches
            assigned to training. Positive fractions allocate at least one
            batch when available, otherwise round down. Default is ``1.0``.
        :param float val_size: The fraction of each condition's stored batches
            assigned to validation after training allocation. Positive fractions
            allocate at least one remaining batch when available, otherwise
            round down. Default is ``0.0``.
        :param float test_size: The requested test fraction. Used in the
            sum-to-one check, but not directly in splitting: all batches left
            after training and validation become test batches. Default is
            ``0.0``.
        :param str batching_mode: The batching identifier, either
            ``"common_batch_size"`` or ``"proportional"``. The current data
            module stores this setting without changing its batching behavior.
            Default is ``"common_batch_size"``.
        :param int num_workers: The nonnegative number of workers loading batch
            identifiers, not condition tensors. Default is ``0``.
        :param bool pin_memory: Whether dataloaders pin batch identifier tensors
            in memory, not the materialized condition data. Default is
            ``False``.
        :param bool shuffle: Whether to permute each condition's stored batch
            indices before splitting. Does not shuffle samples inside batches
            or enable dataloader shuffling. Default is ``True``.
        :param dataloader_cls: The custom dataloader class or per-condition
            mapping, normalized and stored but unused by the current data
            module. ``None`` produces a mapping of condition names to ``None``;
            standard PyTorch dataloaders are always used. Default is ``None``.
        :type dataloader_cls: type | dict[str, type | None] | None
        :param collate_fn: The custom collator or per-condition mapping,
            normalized and stored but unused by the current data module.
            ``None`` produces a mapping of condition names to ``None``; default
            collation is always used. Default is ``None``.
        :type collate_fn: Callable | dict[str, Callable | None] | None
        :param dict[str, Any] kwargs: Options forwarded to
            :class:`lightning.pytorch.Trainer`, including device and strategy
            options (``accelerator``, ``devices``, ``num_nodes``, ``strategy``,
            ``precision``), stopping limits (``max_epochs``, ``max_steps``,
            ``max_time``), ``logger``, ``callbacks``, ``default_root_dir``, and
            Lightning's validation, gradient, checkpointing, and debugging
            options. Their types and defaults follow Lightning's constructor
            except for the following PINA overrides:

            * ``inference_mode`` (bool): Whether evaluation uses inference mode.
              Forced to ``False`` for physics-informed solvers, even if supplied
              as ``True``. Otherwise Lightning's default is ``True``.
            * ``log_every_n_steps`` (int): The logging interval. Forced to ``0``
              when ``batch_size`` is ``None``. Default is ``50`` when batching
              is enabled.
            * ``enable_progress_bar`` (bool): Whether to enable progress
              reporting. Default is ``True``.
            * ``compile`` (Any): The deprecated option, removed before
              forwarding and ignored with a :class:`DeprecationWarning` for any
              value.
        :raises ValueError: If ``solver`` fails validation against
            :class:`pina.solver.BaseSolver`.
        :raises ValueError: If ``train_size``, ``val_size``, or ``test_size`` is
            not a float.
        :raises ValueError: If the sum of ``train_size``, ``val_size``, and
            ``test_size`` is not close to ``1.0`` under :func:`torch.isclose`.
        :raises ValueError: If ``pin_memory`` or ``shuffle`` is not a boolean.
        :raises AssertionError: If ``num_workers`` is not a nonnegative integer,
            or ``batch_size``, when provided, is not a positive integer.
        :raises ValueError: If ``batching_mode`` is not one of the available
             options.
        :raises ValueError: If ``dataloader_cls`` or ``collate_fn`` is a mapping
            referring to unknown conditions.
        :raises TypeError: If an unsupported keyword is forwarded to Lightning.
        :raises RuntimeError: If any domain in the problem has not been
             discretised.

        .. note::

            Split fractions are not individually range-checked. Constructor
            setup moves inverse-problem parameters to the first configured
            device and creates a data module, but stores batches only when a
            training or testing loop runs its setup.

        .. warning::

            Requesting ``"proportional"`` emits :class:`UserWarning` and resets
            the identifier to ``"common_batch_size"`` when ``batch_size`` is
            ``None`` or no larger than the number of conditions. With
            ``batch_size=None``, using workers or pinning also emits a warning,
            but those settings are still forwarded unchanged to the data module.
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

        # Raise warning if num_workers or pin_memory are set without a batch size
        if batch_size is None and (num_workers > 0 or pin_memory):
            warnings.warn(
                "`num_workers` and `pin_memory` have no effect when "
                "`batch_size` is None.",
                UserWarning,
            )

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

        Dictionary inputs are returned unchanged, including omitted condition
        names; only unknown keys are rejected. Scalar inputs are broadcast to
        every condition. Values are not checked for being classes or callables.

        :param dataloader_cls: The dataloader class or per-condition mapping.
            ``None`` fills every condition entry with ``None``.
        :type dataloader_cls: type | dict[str, type | None] | None
        :param collate_fn: The collator or per-condition mapping. ``None`` fills
            every condition entry with ``None``.
        :type collate_fn: Callable | dict[str, Callable | None] | None
        :param set[str] conditions: The names of the problem conditions.
        :raises ValueError: If a mapping refers to unknown conditions.
        :return: The normalized per-condition mappings.
        :rtype: tuple[dict[str, type | None], dict[str, Callable | None]]
        """

        def _normalize(value, label):
            """
            Expand a scalar option or validate a per-condition mapping's keys.

            :param Any value: The option value. ``None`` fills each condition
                entry with ``None``. Dictionaries are returned unchanged after
                their keys are checked; other values are broadcast to every
                condition without type validation.
            :param str label: The option name used in validation errors.
            :raises ValueError: If a dictionary key is not a known condition.
            :return: The normalized per-condition mapping.
            :rtype: dict[str, Any]
            """
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
        is replaced in that dictionary by a new :class:`torch.nn.Parameter`
        wrapping its detached data on the first device configured by the
        Lightning accelerator connector. This also replaces parameters already
        on that device; existing references to the old objects are not updated.
        If the problem has no ``unknown_parameters`` attribute, do nothing.
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

        The solver and ``data_module`` are supplied automatically.

        :param dict[str, Any] kwargs: Options forwarded to
            :meth:`lightning.pytorch.Trainer.fit`. Supported options include
            ``ckpt_path`` (str | pathlib.Path | None), the checkpoint selection;
            ``None`` normally keeps the supplied solver's current weights.
            Default is ``None``. In Lightning versions supporting
            ``weights_only`` (bool | None), it controls restricted checkpoint
            loading; ``None`` uses Lightning/PyTorch's loading policy. Default
            is ``None``. Do not supply ``model`` or ``datamodule``, which are
            already provided, or custom training/validation dataloaders.
        :raises TypeError: If a keyword is unsupported by the installed
            Lightning version, or ``model`` or ``datamodule`` is duplicated.
        :raises lightning.fabric.utilities.exceptions.MisconfigurationException:
            If custom training or validation dataloaders conflict with the
            automatically supplied data module.
        """
        return super().fit(self.solver, datamodule=self.data_module, **kwargs)

    def test(self, **kwargs):
        """
        Test the solver using the trainer data module.

        The solver and ``data_module`` are supplied automatically. Setup
        rebuilds condition batches and all dataset splits before testing.

        :param dict[str, Any] kwargs: Options forwarded to
            :meth:`lightning.pytorch.Trainer.test`:

            * ``ckpt_path`` (str | pathlib.Path | None): The checkpoint
              selection. ``None`` normally keeps the supplied solver's current
              weights. Default is ``None``.
            * ``verbose`` (bool): Whether to print test results. Default is
              ``True``.
            * ``weights_only`` (bool | None): The restricted checkpoint-loading
              option, in Lightning versions supporting it. ``None`` uses
              Lightning/PyTorch's loading policy. Default is ``None``.

            Do not supply ``model`` or ``datamodule``, which are already
            provided, or custom test dataloaders.
        :raises TypeError: If a keyword is unsupported by the installed
            Lightning version, or ``model`` or ``datamodule`` is duplicated.
        :raises lightning.fabric.utilities.exceptions.MisconfigurationException:
            If custom test dataloaders conflict with the supplied data module.
        :return: The logged test metrics, one dictionary per test dataloader.
            Returns an empty list if no test batches are available.
        :rtype: list[dict[str, float]]
        """
        return super().test(self.solver, datamodule=self.data_module, **kwargs)

    @property
    def solver(self):
        """
        Return the solver attached to the trainer.

        :return: The solver used by the trainer.
        :rtype: pina.solver.BaseSolver
        """
        return self._solver

    @solver.setter
    def solver(self, solver):
        """
        Set the solver attached to the trainer.

        Store the reference without repeating constructor validation or
        rebuilding the existing data module.

        :param pina.solver.BaseSolver solver: The solver instance to attach.
        """
        self._solver = solver
