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
    """Add a rotating JSONL file sink and a console sink.

    The JSONL file sink is always added at all levels.
    The console sink routes through ``tqdm.write`` to avoid disturbing
    progress bars and is added for WARNING and above by default.
    When *console* is ``True``, the console sink threshold is lowered to
    DEBUG so that all messages are visible.

    Previously registered sinks are removed first so that repeated calls
    do not accumulate duplicates.

    Args:
        output_dirpath:
            Directory for the rotating JSONL log file.
        console:
            If ``True``, show all log levels on the console instead of
            only WARNING and above.
    """
    # Remove all existing sinks, including the default stderr handler,
    # so that output routing is fully controlled.
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
    """Route ``warnings.warn`` output through ``tqdm.write``.

    Temporarily overrides ``warnings.showwarning`` so that warning
    messages do not write directly to *stderr*, which would corrupt
    active progress bars.
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
