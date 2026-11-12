import io
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

import pina
from pina._src.core.logging import (
    LOG_FORMAT_ENV_VAR,
    LOG_LEVEL_ENV_VAR,
    get_logger,
)
from pina._src.core.logging import logger as pina_logger

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore_logging_state():
    """Restore the PINA logger state after each test."""
    original_handlers = list(pina_logger.handlers)
    original_level = pina_logger.level
    original_propagate = pina_logger.propagate

    # Isolate tests from handlers installed by PINA_LOG_LEVEL.
    for handler in list(pina_logger.handlers):
        if getattr(handler, "_pina_handler", False):
            pina_logger.removeHandler(handler)

    yield

    pina_logger.handlers[:] = original_handlers
    pina_logger.setLevel(original_level)
    pina_logger.propagate = original_propagate


def _run_python(code, extra_env=None):
    """Run Python code in a fresh interpreter with the repository importable."""
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    )
    if extra_env is not None:
        env.update(extra_env)

    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_logger_uses_pina_namespace():
    assert pina_logger.name == "pina"
    assert pina.logger is pina_logger


def test_null_handler_is_installed_by_default():
    assert any(
        isinstance(handler, logging.NullHandler)
        for handler in pina_logger.handlers
    )


def test_get_logger_shortens_internal_names():
    assert get_logger() is pina_logger
    assert get_logger("pina") is pina_logger
    assert get_logger("pina._src") is pina_logger
    assert (
        get_logger("pina._src.core.data_module").name
        == "pina.core.data_module"
    )
    assert get_logger("custom").name == "pina.custom"
    assert get_logger("pina.solver.pinn").name == "pina.solver.pinn"


def test_debug_logging_is_silent_by_default(capsys):
    pina_logger.debug("hidden debug message")
    pina_logger.info("hidden info message")

    captured = capsys.readouterr()
    assert "hidden" not in captured.out
    assert "hidden" not in captured.err


def test_enable_debug_logging_is_idempotent(capsys):
    pina.enable_debug_logging()
    pina.enable_debug_logging()

    handlers = [
        handler
        for handler in pina_logger.handlers
        if getattr(handler, "_pina_handler", False)
    ]
    assert len(handlers) == 1
    assert pina_logger.level == logging.DEBUG

    pina_logger.debug("visible debug message")
    assert "visible debug message" in capsys.readouterr().err


def test_enable_debug_logging_with_custom_stream():
    stream = io.StringIO()

    pina.enable_debug_logging(stream=stream)
    pina_logger.debug("stream message")

    assert "stream message" in stream.getvalue()


def test_set_level_accepts_names_and_integers():
    assert pina.set_level("INFO") is pina_logger
    assert pina_logger.level == logging.INFO

    pina.set_level(logging.WARNING)
    assert pina_logger.level == logging.WARNING

    with pytest.raises(ValueError):
        pina.set_level("VERBOSE")

    with pytest.raises(TypeError):
        pina.set_level(1.0)


def test_disable_logging_removes_pina_handler():
    pina.enable_debug_logging()
    pina.disable_logging()

    assert not any(
        getattr(handler, "_pina_handler", False)
        for handler in pina_logger.handlers
    )
    assert pina_logger.level > logging.CRITICAL


def test_environment_variable_enables_debug_output():
    result = _run_python(
        "import pina; pina.logger.debug('env debug message')",
        extra_env={LOG_LEVEL_ENV_VAR: "DEBUG"},
    )

    assert result.returncode == 0, result.stderr
    assert "env debug message" in result.stderr


def test_environment_variable_sets_format():
    result = _run_python(
        "import pina; pina.logger.debug('formatted message')",
        extra_env={
            LOG_LEVEL_ENV_VAR: "DEBUG",
            LOG_FORMAT_ENV_VAR: "PINA|%(levelname)s|%(message)s",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "PINA|DEBUG|formatted message" in result.stderr


def test_invalid_environment_level_is_ignored():
    result = _run_python(
        "import pina; pina.logger.debug('hidden env message')",
        extra_env={LOG_LEVEL_ENV_VAR: "VERBOSE"},
    )

    assert result.returncode == 0, result.stderr
    assert "hidden env message" not in result.stderr
    assert "Ignoring invalid" in result.stderr
