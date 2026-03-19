"""Logging and output configuration for the aef-embeddings package."""

import contextlib
import pathlib
import warnings
from collections.abc import Iterator
from typing import Final

import loguru
import tqdm

_LOG_FORMAT: Final[str] = (
    "{time:%Y-%m-%dT%H:%M:%S.%f%z} | {level:<8} | {thread.name}:{thread.id} | {message}"
)


def _configure_logging(
    output_dirpath: pathlib.Path,
    *,
    console: bool = False,
) -> None:
    """Adds a rotating JSONL file sink and a console sink.

    The JSONL file sink is always added at all levels.  The console
    sink, routed through ``tqdm.write()`` to avoid disturbing progress
    bars, is always added for WARNING and above.  When *console* is
    True, the console sink threshold is lowered to DEBUG so that all
    log messages are visible.

    Removes previously registered sinks before re-adding so that
    repeated calls do not accumulate duplicate sinks.

    Args:
        output_dirpath: Directory for the rotating JSONL log file.
        console: If True, show all log levels on the console instead
            of only WARNING and above.
    """
    # Remove all existing sinks (including loguru's default stderr
    # handler) so we have full control over where output goes.
    loguru.logger.remove()

    log_dirpath = output_dirpath / "logs"
    log_dirpath.mkdir(parents=True, exist_ok=True)

    loguru.logger.add(
        log_dirpath / "log.jsonl",
        format="{message}",
        serialize=True,
        backtrace=True,
        diagnose=True,
        rotation="10 MB",
    )

    loguru.logger.add(
        lambda msg: tqdm.tqdm.write(msg, end=""),
        format=_LOG_FORMAT,
        colorize=True,
        backtrace=True,
        diagnose=True,
        level="DEBUG" if console else "WARNING",
    )


@contextlib.contextmanager
def _redirect_warnings_to_tqdm() -> Iterator[None]:
    """Routes ``warnings.warn()`` through ``tqdm.write()``.

    Temporarily overrides ``warnings.showwarning`` so that warning
    messages are printed via ``tqdm.write()`` instead of writing
    directly to *stderr*, which would disturb active progress bars.
    """
    original = warnings.showwarning

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        tqdm.tqdm.write(
            warnings.formatwarning(message, category, filename, lineno, line),
            end="",
        )

    warnings.showwarning = _showwarning
    try:
        yield
    finally:
        warnings.showwarning = original
