# Solver Submodule Refactoring

## Motivation

The solver submodule uses a mixin-based architecture for modular construction. While extensible,
the growing number of mixins and override chains made it difficult to read and understand which
method is overridden by which mixin. This refactoring compresses methods, clarifies override
semantics, and makes it easier to add new mixins.

## Design Decisions

| Decision | Resolution |
|---|---|
| `_compute_condition_loss` has 4 variants | **Pro.** Different scientific methods coexist in one framework. |
| SelfAdaptive solver | Keep as solver class, not a weighting class. Trainable-pointwise-weight + min-max loop is solver-level logic. |
| Competitive solver | Fine as-is. |
| GradientEnhanced | Keep as mixin. Moving to Condition changes the interface too much. |
| RBA mixin | Keep as mixin. Operates at a different stage than GradientEnhanced. |
| All mixins stay | ConditionAggregator, SingleModel, MultiModel, Ensemble, ManualOpt, PhysicsInformed, Autoregressive, GradientEnhanced, RBA — all retained for future extensibility. |
| Forward override conflict | Resolved by removing stacking forward from MultiModelMixin — only EnsembleMixin owns it. |
| GradientEnhanced + RBA | Composable, not exclusive. GradientEnhanced overrides `_regularize_condition_loss` (additive), RBA overrides `_weight_condition_loss` (multiplicative). |

## Architecture

### Layer 0: Interface
`SolverInterface` — abstract contract (training_step, validation_step, test_step, properties).

### Layer 1: Core
`BaseSolver` — full pipeline: init, forward (default single-model), loss computation, hooks,
properties. All mixins and concrete solvers build on this.

### Layer 2: Structural Mixins
| Mixin | Responsibility |
|---|---|
| `SingleModelMixin` | Single-model forward, optimizer config, convenience properties (model, optimizer, scheduler) |
| `MultiModelMixin` | Multi-optimizer config, convenience properties (models, optimizers, schedulers) — no forward |
| `EnsembleMixin` | Stacking forward (all models), per-model loss iteration |
| `ManualOptimizationMixin` | Lightning manual optimization loop (zero-grad, backward, step) |
| `ConditionAggregatorMixin` | Batch iteration, per-condition loss, logging, weighting aggregation |

### Layer 3: Composed Base Solvers
| Solver | Inherits |
|---|---|
| `SingleModelSolver` | SingleModel + CondAgg + BaseSolver |
| `MultiModelSolver` | ManualOpt + MultiModel + CondAgg + BaseSolver |
| `EnsembleSolver` | ManualOpt + Ensemble + CondAgg + BaseSolver |

### Layer 4: Feature Mixins
| Mixin | Hook overridden | Behavior |
|---|---|---|
| `PhysicsInformedMixin` | `validation_step`, `test_step` | Wraps with `@torch.enable_grad()` |
| `AutoregressiveMixin` | `_loss_from_residual` | Adaptive temporal weighting |
| `GradientEnhancedMixin` | `_regularize_condition_loss` | Additive gradient penalty |
| `ResidualBasedAttentionMixin` | `_weight_condition_loss` | Multiplicative attention weighting |

### Layer 5: Concrete Solvers
All 14 concrete solver classes. Most are thin wrappers (own `__init__` + `accepted_conditions_types`).
Complex ones (Causal, SelfAdaptive, Competitive) override `_compute_condition_loss` and/or
`training_step`.

## Method x Mixin Table

### BaseSolver + Structural Mixins + Feature Mixins

| Method | BaseSolver | SingleModelMixin | MultiModelMixin | EnsembleMixin | ManualOptMixin | CondAggMixin | PhysicsInfMixin | AutoregressiveMixin | GradientEnhancedMixin | RBAMixin |
|---|---|---|---|---|---|---|---|---|---|---|
| `__init__` | **Merged:** validates problem, stores state, labelizes forward, stores models/opts/scheds, sets up weighting + loss | | | | | | | | | |
| `forward` | **Default:** `self._pina_models[0](x)` | **Overrides:** `self.model(x)` | | **Overrides:** if `_active_model_idx`, eval one model; else `torch.stack([m(x) for m in self.models])` | | | | | | |
| `configure_optimizers` | | **Defines:** hook single model + inverse params, return single lists | **Defines:** hook all models + inverse params, return lists | *(inherits MultiModel)* | | | | | | |
| `training_step` | Calls `batch_evaluation_step`, logs `train_loss` | | | | **Overrides:** zero-grad -> super() -> manual backward -> step all | | | | | |
| `validation_step` | Calls `batch_evaluation_step`, logs `val_loss` | | | | | | **Overrides:** wraps `super()` with `@torch.enable_grad()` | | | |
| `test_step` | Calls `batch_evaluation_step`, logs `test_loss` | | | | | | **Overrides:** wraps `super()` with `@torch.enable_grad()` | | | |
| `on_train_batch_end` | | | | | **Defines:** syncs Lightning manual-opt counters | | | | | |
| `batch_evaluation_step` | | | | | | **Defines:** iterates batch -> `_compute_condition_loss` per condition -> logs -> aggregates via weighting | | | | |
| `_compute_condition_loss` | Clone -> prepare -> evaluate -> loss -> regularize -> weight -> reduce | | | | | | | | | |
| `_prepare_condition_data` | Identity hook | | | | | | | | **Overrides:** sets `requires_grad_(True)` on input | |
| `_regularize_condition_loss` | Identity hook (additive) | | | | | | | | **Overrides:** computes `grad(residual, input)`, adds gradient penalty | |
| `_weight_condition_loss` | Identity hook (multiplicative) | | | | | | | | | **Overrides:** normalizes residual norms, updates attention weights, multiplies loss |
| `_loss_from_residual` | `loss_fn(residual, zeros)` | | | | | | | **Overrides:** adds adaptive temporal weighting via `_get_weights` | | |
| `_apply_reduction` | Applies none/mean/sum | | | | | | | | | |
| `get_batch_size` | Static: sums input lengths | | | | | | | | | |
| `on_train_epoch_start` | | | | | | | | **Defines:** optionally resets running averages | | |
| `_init_autoregressive_components` | | | | | | | | **Defines:** stores eps, running averages, step counters | | |
| `_get_weights` | | | | | | | | **Defines:** running average per condition -> `_compute_adaptive_weights` | | |
| `_compute_adaptive_weights` | | | | | | | | **Defines:** `exp(-eps * cumsum(step_loss))` | | |
| `preprocess_step` | | | | | | | | **Defines:** identity hook before rollout step | | |
| `postprocess_step` | | | | | | | | **Defines:** identity hook after rollout step | | |
| `predict` | | | | | | | | **Defines:** autoregressive rollout | | |
| `_init_gradient_enhanced_components` | | | | | | | | | **Defines:** validates params, ensures SpatialProblem + LabelTensor | |
| `_init_residual_attention_components` | | | | | | | | | | **Defines:** validates params, registers per-condition attention buffers |
| `problem` | Property -> `_pina_problem` | | | | | | | | | |
| `use_lt` | Property -> `_use_lt` | | | | | | | | | |
| `weighting` | Property -> `_pina_weighting` | | | | | | | | | |
| `loss` | Property -> `_loss_fn` | | | | | | | | | |
| `model` | | Property -> `_pina_models[0]` | | | | | | | | |
| `optimizer` | | Property -> `_pina_optimizers[0]` | | | | | | | | |
| `scheduler` | | Property -> `_pina_schedulers[0]` | | | | | | | | |
| `models` | | | Property -> `_pina_models` | *(inherits)* | | | | | | |
| `optimizers` | | | Property -> `_pina_optimizers` | *(inherits)* | | | | | | |
| `schedulers` | | | Property -> `_pina_schedulers` | *(inherits)* | | | | | | |
| `num_models` | | | Property -> `len(self.models)` | *(inherits)* | | | | | | |

### Concrete Solvers

| Solver | Inherits from | Overrides `training_step` | Overrides `_compute_condition_loss` | Extra own methods |
|---|---|---|---|---|
| **SingleModelSolver** | Single + CondAgg + Base | -- | -- | `__init__`, `accepted_conditions_types` |
| **MultiModelSolver** | ManualOpt + Multi + CondAgg + Base | -- | -- | `__init__` |
| **EnsembleSolver** | ManualOpt + Ensemble + CondAgg + Base | -- | -- | `__init__` |
| **SupervisedSingleModelSolver** | SingleModelSolver | -- | -- | `__init__`, `accepted_conditions_types` |
| **SupervisedEnsembleSolver** | EnsembleSolver | -- | -- | `__init__`, `accepted_conditions_types` |
| **PhysicsInformedSingleModelSolver** | PhysInf + SingleModelSolver | -- | -- | `__init__`, `accepted_conditions_types` |
| **PhysicsInformedEnsembleSolver** | PhysInf + EnsembleSolver | -- | -- | `__init__`, `accepted_conditions_types` |
| **AutoregressiveSingleModelSolver** | AutoReg + SingleModelSolver | -- | -- | `__init__`, `accepted_conditions_types` |
| **AutoregressiveEnsembleSolver** | AutoReg + EnsembleSolver | -- | -- | `__init__`, `accepted_conditions_types` |
| **GradientPhysicsInformedSingleModelSolver** | PhysInf + GradEnh + SingleModelSolver | -- | -- | `__init__`, `accepted_conditions_types` |
| **RBAPhysicsInformedSingleModelSolver** | PhysInf + RBA + SingleModelSolver | -- | -- | `__init__`, `accepted_conditions_types` |
| **CausalPhysicsInformedSingleModelSolver** | PhysInf + SingleModelSolver | -- | Time-step loop + causal weighting | `__init__`, `accepted_conditions_types`, `temporal_variable`, `spatial_variables`, `_compute_weights` |
| **SelfAdaptivePhysicsInformedSolver** | PhysInf + MultiModelSolver | Min-max: weights first, then model | Trainable weight application | `__init__`, `accepted_conditions_types`, `model`, `weights`, `optimizer_model`, `optimizer_weights`, `scheduler_model`, `scheduler_weights` |
| **CompetitivePhysicsInformedSolver** | PhysInf + MultiModelSolver | Min-max: model first, then discriminator with `-loss` | Multiplies residual by discriminator output | `__init__`, `accepted_conditions_types`, `model`, `discriminator`, `optimizer_model`, `optimizer_discriminator`, `scheduler_model`, `scheduler_discriminator` |

## Override Conflict Map

| Method | Overridden by | Conflict? |
|---|---|---|
| `forward` | BaseSolver (default), SingleModelMixin, EnsembleMixin | Clean. SingleModel overrides for single-model; Ensemble overrides for stacking. |
| `training_step` | BaseSolver, ManualOptimizationMixin, SelfAdaptive, Competitive | SelfAdaptive/Competitive completely replace ManualOpt's training_step with custom min-max loops. |
| `_compute_condition_loss` | BaseSolver, EnsembleMixin, Causal, SelfAdaptive, Competitive | 4 genuinely different implementations. Pro: different scientific methods coexist. |
| `_loss_from_residual` | BaseSolver, AutoregressiveMixin | Clean single override. |
| `_prepare_condition_data` | BaseSolver, GradientEnhancedMixin | Clean single override. |
| `_regularize_condition_loss` | BaseSolver, GradientEnhancedMixin | Clean. GradientEnhanced adds gradient penalty (additive). |
| `_weight_condition_loss` | BaseSolver, ResidualBasedAttentionMixin | Clean. RBA multiplies by attention weights (multiplicative). |
| `validation_step` | BaseSolver, PhysicsInformedMixin | Clean single override. |
| `test_step` | BaseSolver, PhysicsInformedMixin | Clean single override. |

## Changes Applied

### Change 1: BaseSolver init merge
Merged `_init_solver_components` + `_init_weighting_and_loss` into `BaseSolver.__init__`.
Concrete solvers now call `BaseSolver.__init__` with all args at once.

### Change 2: MultiModelMixin loses stacking forward
Removed `forward` from `MultiModelMixin`. It now only manages optimizers/schedulers and provides
list properties. Stacking forward is owned solely by `EnsembleMixin`.

### Change 3: ManualOptimizationMixin loses `_init_manual_optimization`
Removed `_init_manual_optimization`. `MultiModelSolver` and `EnsembleSolver` set
`self.automatic_optimization = False` directly in their `__init__`.

### Change 4: Split regularization into two hooks
Added `_weight_condition_loss` (multiplicative hook) alongside `_regularize_condition_loss`
(additive hook). `_compute_condition_loss` calls both in sequence. GradientEnhanced overrides the
additive hook; RBA overrides the multiplicative hook. They are now composable.

### Change 5: SelfAdaptive/Competitive lose forward override
Both removed their `forward` override. With `MultiModelMixin` no longer stacking, the default
single-model forward from `BaseSolver` is correct for both.

### Default forward on BaseSolver
Added `BaseSolver.forward(x)` returning `self._pina_models[0](x)`. Provides a sensible default
for any solver that doesn't explicitly override forward.
