"""Utilities for modules."""

from pathlib import Path
import importlib.util
import sys


def load_module(module_path: Path | str, add_to_sys_modules: bool = False):
    """Load a module from a file."""
    module_path = Path(module_path).resolve()
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from {module_path}")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    if add_to_sys_modules:
        sys.modules[module_path.stem] = config
    return config
