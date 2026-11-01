.. docmeta::
   :last_reviewed: 2026-06-24


Installation
============

**PINA** requires `torch`, `lightning`, `torch_geometric` and `matplotlib`.

Installing via PIP
__________________

Mac and Linux users can install pre-built binary packages using pip:

.. code-block:: bash

    pip install pina-mathlab

To uninstall the package:

.. code-block:: bash

    pip uninstall pina-mathlab

Installing from source
______________________

The official distribution is on GitHub. Clone the repository:

.. code-block:: bash

    git clone https://github.com/PINA-org/PINA

Then install in editable mode:

.. code-block:: bash

    pip install -e .

Install with extra packages
____________________________

To install extra dependencies required to run tests or tutorials, use:

.. code-block:: bash

    pip install "pina-mathlab[extras]"

Available extras include:

* ``dev`` — development tools (use this if you want to contribute).
* ``test`` — for running tests locally.
* ``doc`` — for building the documentation locally.
* ``tutorial`` — for running tutorials.

Requirements
____________

PINA is built on:

* `PyTorch <https://pytorch.org/>`_ — deep learning framework.
* `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_ — training loop orchestration.
* `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_ — graph neural network support.
* `Matplotlib <https://matplotlib.org/>`_ — plotting and visualisation.

See Also
--------

* :doc:`Quickstart guide <_quickstart>`
* :doc:`API Reference <_rst/_code>`
