.. docmeta::
   :last_reviewed: 2026-11-11

Logging
=======

PINA emits diagnostic messages through the standard :mod:`logging` module
using the ``pina`` logger hierarchy. Importing :mod:`pina` is silent by
default: the package only installs a :class:`logging.NullHandler` and never
configures the root logger.

Enable debug output
-------------------

From Python:

.. code-block:: python

    import pina

    pina.enable_debug_logging()

    # Or choose a less verbose level.
    pina.set_level("INFO")

From the environment, before importing PINA:

.. code-block:: bash

    PINA_LOG_LEVEL=DEBUG python train.py

Combine the level with a custom format:

.. code-block:: bash

    PINA_LOG_LEVEL=DEBUG \
    PINA_LOG_FORMAT="%(levelname)s | %(name)s | %(message)s" \
    python train.py

Module loggers
--------------

Internal modules use child loggers such as ``pina.core.data_module`` and
``pina.solver.base_solver``. Child loggers propagate to the ``pina`` logger,
so a single call to :func:`~pina.enable_debug_logging` enables all of them.

Custom modules should follow the same pattern:

.. code-block:: python

    from pina import get_logger

    logger = get_logger(__name__)
    logger.debug("Sampling %d points", n)

Disable logging
---------------

To remove the console handler and silence PINA messages:

.. code-block:: python

    import pina

    pina.disable_logging()

API reference
-------------

.. autofunction:: pina.enable_debug_logging

.. autofunction:: pina.set_level

.. autofunction:: pina.disable_logging

.. autofunction:: pina.get_logger
