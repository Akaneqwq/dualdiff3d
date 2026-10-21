"""Hugging Face Hub helpers for released DualDiff checkpoints."""

import argparse
import os
from pathlib import Path
from typing import Optional, Union


DEFAULT_MODEL_ID = "Akaneqwq/DualDiff"
DEFAULT_MODEL_FILENAME = "model.pkl"


def _environment_flag(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {value!r}"
    )


def resolve_dualdiff_checkpoint(
    model_path: Optional[Union[str, os.PathLike]] = None,
    model_id: Optional[str] = None,
    filename: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    local_files_only: bool = False,
) -> str:
    """Return a local checkpoint path, downloading it from the Hub if needed.

    An explicit ``model_path`` always takes precedence and must exist. Otherwise
    the repository, filename, revision, and cache directory can be configured
    by arguments or the ``DUALDIFF_MODEL_*`` environment variables.
    """
    if model_path:
        local_path = Path(model_path).expanduser().resolve()
        if not local_path.is_file():
            raise FileNotFoundError(f"DualDiff checkpoint not found: {local_path}")
        print(f"Using local DualDiff checkpoint: {local_path}")
        return str(local_path)

    model_id = model_id or os.environ.get("DUALDIFF_MODEL_ID", DEFAULT_MODEL_ID)
    filename = filename or os.environ.get(
        "DUALDIFF_MODEL_FILENAME", DEFAULT_MODEL_FILENAME
    )
    revision = revision or os.environ.get("DUALDIFF_MODEL_REVISION") or None
    cache_dir = cache_dir or os.environ.get("DUALDIFF_CACHE_DIR") or None
    local_files_only = local_files_only or _environment_flag(
        "DUALDIFF_LOCAL_FILES_ONLY"
    )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Automatic DualDiff checkpoint download requires huggingface-hub. "
            "Install the repository environment with `conda env create -f "
            "environment.yml`."
        ) from exc

    print(
        f"Resolving DualDiff checkpoint from Hugging Face: "
        f"{model_id}/{filename}"
    )
    try:
        path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        mode = "local cache" if local_files_only else "Hugging Face Hub"
        raise RuntimeError(
            f"Could not resolve {model_id}/{filename} from {mode}. "
            "Check the repository name, network access, or pass a local "
            "checkpoint path explicitly."
        ) from exc

    print(f"Using cached DualDiff checkpoint: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download or locate the released DualDiff checkpoint."
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--model_filename", type=str, default=None)
    parser.add_argument("--model_revision", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()
    print(
        resolve_dualdiff_checkpoint(
            model_path=args.model_path,
            model_id=args.model_id,
            filename=args.model_filename,
            revision=args.model_revision,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
    )


if __name__ == "__main__":
    main()
