.. docmeta::
   :last_reviewed: 2026-06-24


Quickstart
==========

This guide gets you up and running with PINA in 5 minutes. By the end, you will have trained a Physics-Informed Neural Network (PINN) to solve the Poisson equation on a unit square.

Install
-------

.. code-block:: bash

   pip install pina-mathlab

Define a problem
----------------

Every PINA workflow starts by defining a :class:`~pina.problem.spatial_problem.SpatialProblem`.
A problem specifies the output variables, the computational domain, and the conditions
(PDE residual, boundary conditions, initial conditions, data) that the solver must satisfy.

.. code-block:: python

   from pina import Condition
   from pina.problem import SpatialProblem
   from pina.domain import CartesianDomain
   from pina.equation import Equation, FixedValue
   from pina.operator import laplacian

   def poisson_equation(input_, output_):
       u = output_.extract(["u"])
       lap_u = laplacian(u, input_, components=["u"], d=["x", "y"])
       return lap_u + (3.14159 ** 2) * u

   class PoissonProblem(SpatialProblem):
       output_variables = ["u"]
       spatial_domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})
       domains = {
           "domain": CartesianDomain({"x": [0, 1], "y": [0, 1]}),
           "boundary": CartesianDomain({"x": [0, 1], "y": [0, 1]}),
       }
       conditions = {
           "domain": Condition(
               domain="domain", equation=Equation(poisson_equation)
           ),
           "boundary": Condition(
               domain="boundary", equation=FixedValue(0.0)
           ),
       }

   problem = PoissonProblem()
   problem.discretise_domain(n=100, mode="grid", domains=["domain", "boundary"])

Create a model
--------------

Choose a neural network architecture. For standard PINNs, a :class:`~pina.model.feed_forward.FeedForward` (MLP) is a solid starting point.

.. code-block:: python

   from pina.model import FeedForward

   model = FeedForward(
       input_dimensions=2,
       output_dimensions=1,
       inner_size=20,
       n_layers=3,
   )

Train with a solver
-------------------

The :class:`~pina.solver.physics_informed_single_model_solver.PhysicsInformedSingleModelSolver` solver wraps the problem and model, and the :class:`~pina._src.core.trainer.Trainer` orchestrates the training loop.

.. code-block:: python

   from pina.solver import PhysicsInformedSingleModelSolver
   from pina import Trainer

   solver = PhysicsInformedSingleModelSolver(problem=problem, model=model)
   trainer = Trainer(solver=solver, max_epochs=1000, accelerator="cpu")
   trainer.train()

For advanced dataset settings, such as per-condition mini-batch rules or
custom split fractions, build a :class:`~pina.DataModule` and pass it to the
trainer. ``batch_size`` is the default for every condition, and
``mini_batch_rule`` overrides it per condition:

.. code-block:: python

   from pina import DataModule, Trainer

   data_module = DataModule(
       problem=problem,
       batch_size=64,
       mini_batch_rule={"boundary": {"batch_size": 32}},
       train_size=0.8,
       val_size=0.1,
       test_size=0.1,
   )
   trainer = Trainer(solver=solver, datamodule=data_module, max_epochs=1000)
   trainer.train()

Inspect results
---------------

After training, the model stores its solution in the solver. Evaluate at any point:

.. code-block:: python

   import torch

   x = torch.tensor([[0.5, 0.5]], requires_grad=True)
   u_pred = solver(x)
   print(u_pred)

What's next?
------------

* Walk through the `Introductory Tutorial <https://PINA-org.github.io/PINA-Tutorials/tutorial17/tutorial.html>`_ for a deeper introduction.
* Explore the :doc:`API reference </_rst/_code>` for all available components.
* Read the `tutorials <https://PINA-org.github.io/PINA-Tutorials/>`_ for domain-specific guides (Neural Operators, Supervised Learning, etc.).
