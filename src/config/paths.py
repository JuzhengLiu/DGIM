"""Load repository-relative resource paths without importing the model stack.

The default configuration is ``configs/paths.yml``. An explicit configuration
path (or ``DGIM_PATHS_CONFIG``) overlays the defaults; otherwise an optional
``configs/paths.local.yml`` overlays them. Relative paths in every file are
anchored to the repository root, not to the working directory or YML file.
"""

import os
from pathlib import Path, PureWindowsPath
import re

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "paths.yml"
LOCAL_CONFIG = REPO_ROOT / "configs" / "paths.local.yml"

PATH_KEYS = frozenset({
    "data_config", "model_config", "pretrain_config", "finetune_config",
    "train_data_root", "train_pose_root", "train_npz_root",
    "train_list_path", "train_intrinsic_path",
    "val_data_root", "val_pose_root", "val_npz_root",
    "val_list_path", "val_intrinsic_path",
    "test_data_root", "test_pose_root", "test_npz_root",
    "test_list_path", "test_intrinsic_path",
    "style_dir", "style_checkpoint", "matcher_checkpoint",
    "repvit_source_dir", "repvit_checkpoint",
    "teacher_source_dir", "teacher_checkpoint", "teacher_cache_dir",
    "pretrain_init_checkpoint", "finetune_init_checkpoint",
    "pretrain_resume_checkpoint", "finetune_resume_checkpoint",
    "log_dir", "checkpoint_dir", "sample_dir",
    "demo_output_dir",
})

REQUIRED_PATH_KEYS = frozenset({
    "data_config", "model_config", "train_data_root", "style_dir",
    "style_checkpoint", "matcher_checkpoint", "log_dir", "sample_dir",
    "demo_output_dir",
})

# Support the same variable syntax on Windows and Linux.
_ENV_VARIABLE = re.compile(r"\$\{([^}]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)|%([^%]+)%")


def resolve_path(value):
    """Return an absolute path string, or None for an optional unset path.

    Expand ``~``, ``$VAR``, ``${VAR}``, and ``%VAR%``. Unknown or still-unexpanded
    variables raise ValueError. Resource existence is deliberately not checked.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Path values must be strings or null.")
    if not value.strip():
        raise ValueError("Path values must not be empty; use null for optional paths.")

    def expand_variable(match):
        variable = next(group for group in match.groups() if group is not None)
        if variable not in os.environ:
            raise ValueError("Environment variable {!r} is not set.".format(variable))
        return os.environ[variable]

    expanded = _ENV_VARIABLE.sub(expand_variable, value)
    remaining = _ENV_VARIABLE.search(expanded)
    if remaining is not None or "${" in expanded:
        raise ValueError("Path contains an unexpanded environment variable: {!r}.".format(value))
    expanded = os.path.expanduser(expanded)
    if expanded.startswith("~"):
        raise ValueError("Cannot expand the home directory in path {!r}.".format(value))
    if not expanded.strip():
        raise ValueError("Path expands to an empty value; use null for optional paths.")

    windows_path = PureWindowsPath(expanded)
    if windows_path.drive and not windows_path.root:
        raise ValueError("Drive-relative paths are ambiguous; use C:/absolute/path or a repository-relative path.")
    if os.name != 'nt' and windows_path.is_absolute():
        raise ValueError("A Windows absolute path cannot be used on this platform; select a Linux path configuration.")
    path = Path(expanded)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve(strict=False))


def _read_paths(config_file):
    try:
        with config_file.open("r", encoding="utf-8-sig") as stream:
            document = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError("Cannot read path configuration {}: {}".format(config_file, error)) from error

    if not isinstance(document, dict) or set(document) != {"paths"}:
        raise ValueError("{} must contain exactly one top-level object named 'paths'.".format(config_file))
    paths = document["paths"]
    if not isinstance(paths, dict):
        raise ValueError("The 'paths' value in {} must be an object.".format(config_file))
    unknown = set(paths) - PATH_KEYS
    if unknown:
        raise ValueError("Unknown path keys in {}: {}".format(config_file, ", ".join(sorted(unknown))))

    resolved = {}
    for key, value in paths.items():
        if value is None and key in REQUIRED_PATH_KEYS:
            raise ValueError("Required path {!r} in {} must not be null.".format(key, config_file))
        try:
            resolved[key] = resolve_path(value)
        except (ValueError, OSError, RuntimeError) as error:
            raise ValueError("Invalid path {!r} in {}: {}".format(key, config_file, error)) from error
    return resolved


def load_paths(config_path=None):
    """Return all configured paths as absolute strings or optional None values.

    Precedence: defaults < explicit file / DGIM_PATHS_CONFIG. Only when neither
    selector is present does paths.local.yml override the defaults. An explicit
    argument takes precedence over DGIM_PATHS_CONFIG. Selecting a file does not
    change how relative resource paths are resolved.
    """
    paths = _read_paths(DEFAULT_CONFIG)
    selected = config_path if config_path is not None else os.environ.get("DGIM_PATHS_CONFIG")
    if selected is not None:
        if isinstance(selected, os.PathLike):
            selected = os.fspath(selected)
        try:
            selected_file = Path(resolve_path(selected))
        except (ValueError, TypeError, OSError, RuntimeError) as error:
            raise ValueError("Invalid path configuration selector: {}".format(error)) from error
        if selected_file != DEFAULT_CONFIG:
            paths.update(_read_paths(selected_file))
    elif LOCAL_CONFIG.exists():
        paths.update(_read_paths(LOCAL_CONFIG))

    missing = PATH_KEYS - set(paths)
    if missing:
        raise ValueError("Missing path configuration keys: {}".format(", ".join(sorted(missing))))
    return paths
