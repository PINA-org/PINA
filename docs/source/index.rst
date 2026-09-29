.. docmeta::
   :last_reviewed: 2026-06-24

:html_theme.sidebar_secondary.remove:

Welcome to PINA's documentation!
=======================================

.. raw:: html

   <div class="hero">

   <img src="_static/PINA_logo.png" alt="PINA" style="max-width: 320px;"/>

   <p class="tagline">A Unified Framework for Scientific Machine Learning</p>

   <a href="_installation.html" class="btn btn-primary" style="margin: 0.25rem;">Get Started</a>
   <a href="_quickstart.html" class="btn btn-primary" style="margin: 0.25rem;">Quickstart</a>
   <a href="https://github.com/PINA-org/PINA" class="btn btn-primary" style="margin: 0.25rem;">View on GitHub</a>

   </div>

.. grid:: 1 1 1 3
   :gutter: 2

   .. grid-item-card:: 🧩 Modular Architecture
      :padding: 3
      :class-body: text-center

      Designed with composable abstractions — plug, replace, or extend components freely.

   .. grid-item-card:: 🚀 Scalable Performance
      :padding: 3
      :class-body: text-center

      Native multi-device training with minimal overhead for large-scale problems.

   .. grid-item-card:: ⚙️ Highly Flexible
      :padding: 3
      :class-body: text-center

      Full automation or granular control — PINA adapts to your workflow.

**PINA** is an open-source Python library designed to simplify and accelerate
the development of Scientific Machine Learning (SciML) solutions.
Built on top of `PyTorch <https://pytorch.org/>`_, `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_,
and `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_,
PINA provides an intuitive framework for defining, experimenting with,
and solving complex problems using Neural Networks,
Physics-Informed Neural Networks (PINNs), Neural Operators, and more.

.. tab-set::

   .. tab-item:: Data Driven Learning

      .. code-block:: python

         import torch
         from pina import Trainer
         from pina.model import FeedForward
         from pina.problem.zoo import SupervisedProblem
         from pina.solver import SupervisedSingleModelSolver

         input_tensor  = torch.rand((10, 1))
         target_tensor = input_tensor.pow(3)

         # Step 1. Define problem
         problem = SupervisedProblem(input_tensor, target_tensor)

         # Step 2. Define model
         model = FeedForward(input_dimensions=1, output_dimensions=1, layers=[64, 64])

         # Step 3. Define solver
         solver = SupervisedSingleModelSolver(problem, model, use_lt=False)

         # Step 4. Train
         trainer = Trainer(solver, max_epochs=1000, accelerator="gpu")
         trainer.train()

   .. tab-item:: Physics-Informed Learning

      .. code-block:: python

         from pina.operator import grad
         from pina.model import FeedForward
         from pina.equation import Equation
         from pina import Trainer, Condition
         from pina.domain import CartesianDomain
         from pina.problem import SpatialProblem
         from pina.equation.zoo import FixedValue
         from pina.solver import PhysicsInformedSingleModelSolver

         def ode_equation(input_, output_):
            u_x = grad(output_, input_, components=["u"], d=["x"])
            u = output_.extract(["u"])
            return u_x - u

         class SimpleODE(SpatialProblem):
            output_variables = ["u"]
            spatial_domain = CartesianDomain({"x": [0, 1]})
            domains = {
               "x0": CartesianDomain({"x": 0.0}),
               "D": CartesianDomain({"x": [0, 1]}),
            }
            conditions = {
               "bound_cond": Condition(domain="x0", equation=FixedValue(1.0)),
               "phys_cond": Condition(domain="D", equation=Equation(ode_equation)),
            }

         # Step 1. Define problem
         problem = SimpleODE()
         problem.discretise_domain(n=100, mode="grid", domains=["D", "x0"])

         # Step 2. Define model
         model = FeedForward(input_dimensions=1, output_dimensions=1, layers=[64, 64])

         # Step 3. Define solver
         solver = PhysicsInformedSingleModelSolver(problem, model)

         # Step 4. Train
         trainer = Trainer(solver, max_epochs=1000, accelerator="gpu")
         trainer.train()

.. admonition:: What's New
   :class: tip

   .. readme-news::

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: 📖 Installation
      :link: _installation.html
      :class-card: sd-shadow-sm

      Install PINA via pip, from source, or with extras.

   .. grid-item-card:: 🚀 Quickstart
      :link: _quickstart.html
      :class-card: sd-shadow-sm

      Get up and running in 5 minutes.

   .. grid-item-card:: 📚 API Reference
      :link: _rst/_code.html
      :class-card: sd-shadow-sm

      Complete API docs for problems, solvers, models, and more.

   .. grid-item-card:: 🎓 Tutorials
      :link: _tutorial.html
      :class-card: sd-shadow-sm

      Step-by-step tutorials covering PINNs, Neural Operators, and more.

   .. grid-item-card:: 📄 Cite PINA
      :link: _cite.html
      :class-card: sd-shadow-sm

      Citation and BibTeX for academic use.

   .. grid-item-card:: 🤝 Contributing
      :link: _contributing.html
      :class-card: sd-shadow-sm

      Guide for contributors, bug reports, and PRs.

   .. grid-item-card:: 👥 Team
      :link: _team.html
      :class-card: sd-shadow-sm

      Meet the PINA team and our funding sources.

   .. grid-item-card:: 📜 License
      :link: _LICENSE.html
      :class-card: sd-shadow-sm

      Open-source license information.

.. toctree::
   :hidden:
   :maxdepth: 1

   Quickstart <_quickstart>
   API <_rst/_code>
   Tutorials <_tutorial>
   Contributing <_contributing>
   Team & Foundings <_team.rst>
   Installing <_installation>
   Cite PINA <_cite.rst>
   License <_LICENSE.rst>
