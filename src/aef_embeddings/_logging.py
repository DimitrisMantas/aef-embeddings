"""Logging and output configuration."""

import contextlib
import json
import pathlib
import warnings
from collections.abc import Iterator
from typing import Any, Final, TextIO

import loguru
import tqdm

_LOG_FORMAT: Final[str] = (
    "{time:%Y-%m-%dT%H:%M:%S.%f%z} | {level:<8} | {thread.name}:{thread.id} | {message}"
)


class _PointLog:
    """Accumulated log entry for a single query point."""

    def __init__(
        self,
        point_index: int,
        source_x: float,
        source_y: float,
        source_crs: str,
        utm_crs: str,
        year: int,
        region_size_pixels: int,
    ) -> None:
        self._data: dict[str, Any] = {
            "point": point_index,
            "source_x": source_x,
            "source_y": source_y,
            "source_crs": source_crs,
            "utm_crs": utm_crs,
            "year": year,
            "region_size_pixels": region_size_pixels,
            "utm_easting": None,
            "utm_northing": None,
            "snapped_easting": None,
            "snapped_northing": None,
            "tiles": [],
            "conflicts": [],
            "valid_pixels": 0,
            "missing_pixels": 0,
            "checksum_md5": None,
            "status": "pending",
            "error": None,
        }

    def mark_restored(self) -> None:
        """Mark the entry as restored from a previous checkpoint."""
        self._data["status"] = "restored"

    def record_success(self, events: dict[str, Any]) -> None:
        """Merge processing events and mark as successful."""
        self._data.update(events)
        self._data["status"] = "success"

    def record_failure(self, error: Exception) -> None:
        """Mark the entry as failed with an error message."""
        self._data["status"] = "failure"
        self._data["error"] = f"{type(error).__name__}: {error}"

    def to_dict(self) -> dict[str, Any]:
        """Return the entry as a plain dict for serialization."""
        return self._data


def _configure_logging(*, console: bool = False) -> None:
    """Configure the console log sink.

    Previously registered sinks are removed first to avoid
    duplicates.

    Args:
        console:
            If ``True``, show all log levels on the console.
            The default threshold is WARNING.
    """
    loguru.logger.remove()
    loguru.logger.add(
        lambda msg: tqdm.tqdm.write(msg, end=""),
        format=_LOG_FORMAT,
        colorize=True,
        backtrace=True,
        diagnose=True,
        level="DEBUG" if console else "WARNING",
    )


def _write_point_log(
    entries: list[dict[str, Any]],
    output_dirpath: pathlib.Path,
) -> None:
    """Write per-point log entries to a JSONL file.

    Each entry occupies one line, keyed on the query point index.
    The file is overwritten on each call so that checkpoint writes
    reflect the latest state.

    Args:
        entries:
            The accumulated log entries, one per query point.
        output_dirpath:
            The output directory.
    """
    log_dirpath = output_dirpath / "logs"
    log_dirpath.mkdir(parents=True, exist_ok=True)
    with open(log_dirpath / "log.jsonl", mode="w") as f:
        for entry in entries:
            json.dump(entry, f, separators=(",", ":"))
            f.write("\n")


@contextlib.contextmanager
def _redirect_warnings_to_tqdm() -> Iterator[None]:
    """Route ``warnings.warn`` output through ``tqdm.write``.

    Prevents warning messages from corrupting active progress bars
    by temporarily overriding ``warnings.showwarning``.
    """
    original = warnings.showwarning

    def _showwarning(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: TextIO | None = None,
        line: str | None = None,
    ):
        tqdm.tqdm.write(
            warnings.formatwarning(message, category, filename, lineno, line),
            end="",
        )

    warnings.showwarning = _showwarning
    try:
        yield
    finally:
        warnings.showwarning = original
