"""Atomic file replacement, shared rather than copied again.

Seven modules already carry this idiom verbatim — the canonical copy is
`krasnal_id.experiments.artifacts.write_experiment_result`. Rather than write an
eighth, the export uses these. The existing seven can migrate later with no
behaviour change; nothing here alters what they do.

The guarantee: a reader of the destination sees either the previous complete file
or the new complete file, never a partial one. That needs the temporary file in
the *same directory* as the destination (so the rename is a rename and not a
cross-device copy), the data flushed to the device before the rename, and the
temporary removed on every failure path.

`staged_path` exists alongside the text and bytes helpers because a parquet
writer wants to own the file itself, and buffering a 450 MB shard in memory to
satisfy a nicer signature would be a poor trade.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, TextIO


@contextmanager
def staged_path(path: Path, error: type[Exception]) -> Iterator[Path]:
    """Yield a temporary path that replaces `path` when the block exits cleanly.

    The caller writes to the yielded path however it likes, including handing it
    to a library that opens it itself. Durability of the caller's own writes is
    the caller's business; `atomic_text` and `atomic_bytes` handle it for the
    common case.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        yield temporary_path
        os.replace(temporary_path, path)
    except OSError as os_error:
        raise error(f"could not write {path}: {os_error}") from os_error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@contextmanager
def atomic_text(path: Path, error: type[Exception]) -> Iterator[TextIO]:
    """Replace one UTF-8 text file atomically."""
    with staged_path(path, error) as staged, staged.open("w", encoding="utf-8") as handle:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def atomic_bytes(path: Path, error: type[Exception]) -> Iterator[BinaryIO]:
    """Replace one binary file atomically."""
    with staged_path(path, error) as staged, staged.open("wb") as handle:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
