"""Version lookup shared by the lightweight and fully loaded CLI paths."""

from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Optional


def _read_local_project_version() -> Optional[str]:
    pyproject_path = Path(__file__).resolve().parents[4] / "pyproject.toml"
    try:
        for line in pyproject_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("version"):
                _, raw_value = line.split("=", 1)
                return raw_value.strip().strip('"').strip("'") or None
    except Exception:
        return None
    return None


def cli_version() -> str:
    """Return installed package version, falling back to a source checkout."""
    try:
        return importlib_metadata.version("mtdata-mcp")
    except importlib_metadata.PackageNotFoundError:
        return _read_local_project_version() or "unknown"
