.. docmeta::
   :last_reviewed: 2026-06-24


Code Documentation
==================

Welcome to PINA documentation! Here you can find the modules of the package divided in different sections.
The high-level structure of the package is depicted in our API.

.. figure:: ../index_files/PINA_API.png
    :alt: PINA application program interface
    :align: center
    :width: 400

The pipeline to solve differential equations with PINA follows just five steps:

1. Define the :doc:`problem <problem/index>` you aim to solve
2. Generate data using :doc:`pina.domain <domain/index>`, or load results as :doc:`LabelTensor <label_tensor>`
3. Choose or build a :doc:`model <model/index>`
4. Choose a :doc:`solver <solver/index>`, or build one from :doc:`mixins <solver/mixin/index>`
5. Train with the :doc:`Trainer <trainer>`, enhanced with :doc:`callbacks <callback/index>`

.. toctree::
    :hidden:

    pina <pina_index>
    pina.condition <condition/index>
    pina.problem <problem/index>
    pina.domain <domain/index>
    pina.equation <equation/index>
    pina.model <model/index>
    pina.model.block <model/block/index>
    pina.solver <solver/index>
    pina.solver.mixin <solver/mixin/index>
    pina.callback <callback/index>
    pina.loss <loss/index>
    pina.weighting <weighting/index>
    pina.optim <optim/index>
    pina.adaptive_function <adaptive_function/index>

----

pina
----

:class:`~pina.Trainer`, :class:`~pina.DataModule`, :class:`~pina.LabelTensor`, :class:`~pina.Graph`, :doc:`differential operators <operator>`.

pina.condition
--------------

:class:`~pina.condition.DomainEquationCondition`, :class:`~pina.condition.InputEquationCondition`,
:class:`~pina.condition.InputTargetCondition`, time-series and graph conditions.

pina.problem
------------

:class:`~pina.problem.SpatialProblem`, :class:`~pina.problem.TimeDependentProblem`,
:class:`~pina.problem.ParametricProblem`, :class:`~pina.problem.InverseProblem`, and benchmark zoo.

pina.domain
-----------

:class:`~pina.domain.CartesianDomain`, :class:`~pina.domain.EllipsoidDomain`,
:class:`~pina.domain.SimplexDomain`, and set operations (Union, Intersection, Difference, Exclusion).

pina.equation
-------------

:class:`~pina.equation.Equation`, :class:`~pina.equation.SystemEquation`,
and equation zoo (Poisson, Burgers, Helmholtz, Advection, Allen-Cahn, Diffusion-Reaction, Acoustic Wave).

pina.model
----------

:class:`~pina.model.FeedForward`, :class:`~pina.model.FNO`, :class:`~pina.model.DeepONet`,
:class:`~pina.model.PirateNet`, :class:`~pina.model.GraphNeuralOperator`, :class:`~pina.model.SINDy`, and more.

pina.model.block
----------------

Low-level building blocks for constructing custom architectures (spectral convolutions, message passing, embeddings, …).

pina.solver
-----------

:class:`~pina.solver.SupervisedSingleModelSolver`,
:class:`~pina.solver.PhysicsInformedSingleModelSolver`,
:class:`~pina.solver.AutoregressiveSingleModelSolver`, and variants
(Self-Adaptive, Competitive, Gradient, RBA, Causal).

pina.solver.mixin
-----------------

Composable mixins for building custom solvers:
:class:`~pina.solver.mixin.PhysicsInformedMixin`, :class:`~pina.solver.mixin.AutoregressiveMixin`,
:class:`~pina.solver.mixin.GradientEnhancedMixin`, and more.

pina.callback
-------------

:class:`~pina.callback.DataNormalizer`, :class:`~pina.callback.R3Refinement`,
:class:`~pina.callback.MetricTracker`, :class:`~pina.callback.SwitchOptimizer`.

pina.loss
---------

:class:`~pina.loss.LpLoss`, :class:`~pina.loss.PowerLoss`, :class:`~pina.loss.SinkhornLoss`.

pina.weighting
--------------

:class:`~pina.weighting.ScalarWeighting`, :class:`~pina.weighting.LinearWeighting`,
:class:`~pina.weighting.NeuralTangentKernelWeighting`, :class:`~pina.weighting.SelfAdaptiveWeighting`.

pina.optim
----------

:class:`~pina.optim.TorchOptimizer`, :class:`~pina.optim.TorchScheduler`.

pina.adaptive_function
----------------------

Learnable activation functions: :class:`~pina.adaptive_function.AdaptiveSIREN`,
:class:`~pina.adaptive_function.AdaptiveTanh`, :class:`~pina.adaptive_function.AdaptiveReLU`, and more.
