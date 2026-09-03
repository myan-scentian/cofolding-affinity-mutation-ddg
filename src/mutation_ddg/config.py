"""Load a dataset configuration from `configuration/`."""

from __future__ import annotations

from pathlib import Path

import yaml

from mutation_ddg.paths import ARTIFACT_ROOT, DEFAULT_ARTIFACT_ROOT, PROJECT_ROOT

ROOT = PROJECT_ROOT
CONFIGURATION = ROOT / "configuration"


def load(kind: str, name: str) -> dict:
    """Read `configuration/<kind>/<name>.yaml`."""
    path = CONFIGURATION / kind / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (CONFIGURATION / kind).glob("*.yaml"))
        raise FileNotFoundError(f"no {kind} config {name!r}; available: {available}")
    return yaml.safe_load(path.read_text())


def run(name: str) -> dict:
    """A dataset config with its model config merged in, dataset settings winning."""
    settings = load("datasets", name)
    model = load("models", settings.get("model", "boltz2"))
    merged = {**model, **settings}
    merged["gpu"] = {**model.get("gpu", {}), **settings.get("gpu", {})}
    return merged


def resolve(config: dict, key: str) -> Path:
    """Resolve a configured input/output path from the artifact root."""
    value = config.get(key)
    if value is None:
        raise KeyError(f"config has no {key!r}")
    path = Path(value)
    if path.is_absolute():
        try:
            return ARTIFACT_ROOT / path.relative_to(DEFAULT_ARTIFACT_ROOT)
        except ValueError:
            return path
    return ARTIFACT_ROOT / path
