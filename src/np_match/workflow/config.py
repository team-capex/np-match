"""Load workflow TOML settings and resolve their paths relative to the file."""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MISSING = object()


@dataclass(frozen=True)
class WorkflowConfig:
    """Parsed TOML configuration with config-relative path resolution."""

    source: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "WorkflowConfig":
        source = Path(path).expanduser().resolve()
        with source.open("rb") as config_file:
            data = tomllib.load(config_file)
        return cls(source=source, data=data)

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, dict):
            raise TypeError(
                f"Expected [{name}] table in workflow config {self.source}"
            )
        return value

    def value(self, section: str, key: str, default: Any = _MISSING) -> Any:
        values = self.section(section)
        if key not in values:
            if default is not _MISSING:
                return default
            raise KeyError(
                f"Missing {section}.{key} in workflow config {self.source}"
            )
        return values[key]

    def path(self, key: str) -> Path:
        value = self.value("paths", key)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"paths.{key} must be a non-empty string in {self.source}"
            )
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.source.parent / path
        return path.resolve()


def load_workflow_config(
    config_path: str | Path | None, script_file: str | Path
) -> WorkflowConfig:
    """Load an explicit config or the config adjacent to a workflow's scripts."""
    path = (
        Path(config_path)
        if config_path is not None
        else Path(script_file).resolve().parents[1] / "config.toml"
    )
    return WorkflowConfig.load(path)
