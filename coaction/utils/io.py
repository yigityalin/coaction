"""Utilities for IO operations."""

from collections.abc import Sequence
from functools import singledispatch
from pathlib import Path
import json
import pickle

import numpy as np


def _check_save_path(path: Path, allow_file_exists: bool = False) -> Path:
    """Check if a save path is valid.

    This function will create the parent directory if it does not exist.

    Parameters
    ----------
    path : Path
        The path to the file.
    allow_file_exists : bool
        Whether to allow overwriting an existing file.
    """
    path = Path(path).resolve()
    if not allow_file_exists and path.exists():
        raise FileExistsError(f"Path {path} already exists.")
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    return path


def _check_load_path(path: Path) -> Path:
    """Check if a load path is valid.

    Parameters
    ----------
    path : Path
        The path to the file.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist.")
    if not path.is_file():
        raise FileNotFoundError(f"Path {path} is not a file.")
    if not path.suffix:
        raise ValueError(f"Path {path} has no suffix.")
    return path


@singledispatch
def save_object(obj, path: Path, allow_file_exists: bool = False):
    """Save an object to a file.

    Parameters
    ----------
    obj : Any
        The object to save.
    path : Path
        The path to the file.
    allow_file_exists : bool
        Whether to allow overwriting an existing file.
    """
    if not path.suffix:
        path = path.with_suffix(".pkl")
    path = _check_save_path(path, allow_file_exists=allow_file_exists)
    with open(path, "wb") as file:
        pickle.dump(obj, file)


@save_object.register
def _(obj: dict, path: Path, allow_file_exists: bool = False):
    """Save a dictionary to a file.

    This function will save the dictionary as a JSON file.

    Parameters
    ----------
    obj : dict
        The dictionary to save.
    path : Path
        The path to the file.
    allow_file_exists : bool
        Whether to allow overwriting an existing file.
    """
    if not path.suffix:
        path = path.with_suffix(".json")
    path = _check_save_path(path, allow_file_exists=allow_file_exists)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, indent=4)


@save_object.register
def _(obj: Sequence | np.ndarray, path: Path, allow_file_exists: bool = False):
    """Save a sequence to a file.

    This function will save the sequence as a NumPy file.

    Parameters
    ----------
    obj : Sequence | np.ndarray
        The sequence to save.
    path : Path
        The path to the file.
    allow_file_exists : bool
        Whether to allow overwriting an existing file.
    """
    if not path.suffix:
        path = path.with_suffix(".npy")
    path = _check_save_path(path, allow_file_exists=allow_file_exists)
    np.save(path, obj)


def find_and_load_object(path: Path):
    """Find and load an object from a file.

    This function will try to load an object from a file with the following suffixes:
    - .json
    - .npy
    - .pkl

    Parameters
    ----------
    path : Path
        The path to the file.
    """
    for suffix in [".json", ".npy", ".pkl"]:
        path = path.with_suffix(suffix)
        if path.exists():
            return load_object(path)
    raise FileNotFoundError(f"Path {path} does not exist.")


def load_object(path: Path):
    """Load an object from a file.

    Parameters
    ----------
    path : Path
        The path to the file.
    """
    path = _check_load_path(path)
    match path.suffix:
        case ".json":
            return load_json(path)
        case ".npy":
            return load_npy(path)
        case ".pkl":
            return load_pkl(path)
        case _:
            raise ValueError(f"Path {path} has an invalid suffix.")


def load_json(path: Path) -> dict:
    """Load a JSON file.

    Parameters
    ----------
    path : Path
        The path to the file.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_npy(path: Path) -> np.ndarray:
    """Load a NumPy file.

    Parameters
    ----------
    path : Path
        The path to the file.
    """
    return np.load(path)


def load_pkl(path: Path):
    """Load a pickle file.

    Parameters
    ----------
    path : Path
        The path to the file.
    """
    with open(path, "rb") as file:
        return pickle.load(file)
