"""Uploading a built export to the Hub.

Build and push are separate verbs on purpose: what gets published is the
directory that was just written and hashed into `provenance.json`, byte for byte.
Handing the rows to a library that re-encodes and re-shards them would publish
something the receipt does not describe.

The token never enters this module. `huggingface_hub` resolves `HF_TOKEN` or the
stored login itself, so no call here passes one, no variable holds one, and no
error message can leak one.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


class PushError(RuntimeError):
    """Raised when an upload fails for an operational reason."""


class PushConfigurationError(ValueError):
    """Raised when a push is misconfigured before anything leaves the machine."""


class DatasetUploader(Protocol):
    """The slice of `HfApi` this needs, so a test can supply its own."""

    def create_repo(self, repo_id: str, *, repo_type: str, exist_ok: bool, private: bool) -> object:
        """Create the dataset repository if it does not exist."""
        ...

    def upload_folder(
        self, *, folder_path: str, repo_id: str, repo_type: str, commit_message: str
    ) -> str:
        """Upload a directory, returning the resulting commit or repository URL."""
        ...


@dataclass(frozen=True, slots=True)
class PushOutcome:
    """What one push did."""

    repo_id: str
    url: str
    private: bool


def _default_uploader() -> DatasetUploader:
    """Build the real uploader, imported lazily so the build path never needs it."""
    from huggingface_hub import HfApi

    return HfApi()


def validate_card(directory: Path) -> None:
    """Ask the Hub whether it will accept the card, before creating anything.

    Learned the hard way: `repos/create` succeeds, then `validate-yaml` rejects
    the metadata and the upload aborts, leaving an empty repository behind. The
    validator is the same one the Hub applies on push, so asking it first turns an
    orphaned repository into a message.
    """
    from huggingface_hub import DatasetCard

    card = directory / "README.md"
    if not card.is_file():
        raise PushConfigurationError(f"no dataset card at {card}; run the export first")
    try:
        DatasetCard.load(card).validate(repo_type="dataset")
    except PushConfigurationError:
        raise
    # The validator raises a family of value and HTTP errors; all of them mean the
    # card would be rejected, which is a configuration problem, not a transport one.
    except Exception as error:
        raise PushConfigurationError(f"the Hub rejected {card.name}: {error}") from error


def _token_present() -> bool:
    """Say whether a token is available, without ever reading its value."""
    from huggingface_hub import get_token

    return get_token() is not None


def push_export(
    directory: Path,
    repo_id: str,
    *,
    private: bool,
    commit_message: str,
    uploader: DatasetUploader | None = None,
    token_check: Callable[[], bool] | None = None,
) -> PushOutcome:
    """Upload a built export directory to the Hub as a dataset repository."""
    if REPO_ID_PATTERN.fullmatch(repo_id) is None:
        raise PushConfigurationError(
            f"'{repo_id}' is not a Hugging Face repository id; expected namespace/name"
        )
    if not directory.is_dir():
        raise PushConfigurationError(
            f"nothing to push: {directory} does not exist; run the export first"
        )

    if uploader is None:
        if not (token_check or _token_present)():
            raise PushConfigurationError(
                "no Hugging Face token available; run `hf auth login` or set HF_TOKEN"
            )
        validate_card(directory)
        uploader = _default_uploader()

    try:
        uploader.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
        url = uploader.upload_folder(
            folder_path=str(directory),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )
    # The hub raises a wide family of transport, auth and rate-limit errors; all of
    # them mean the same thing here, which is that the upload did not happen.
    except Exception as error:
        raise PushError(f"upload to {repo_id} failed: {error}") from error

    return PushOutcome(
        repo_id=repo_id,
        url=str(url) or f"https://huggingface.co/datasets/{repo_id}",
        private=private,
    )
