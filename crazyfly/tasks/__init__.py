# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

import sys
from pathlib import Path


def _ensure_isaaclab_tasks_on_path() -> None:
    """Make bundled ``isaaclab_tasks`` importable in pip-only IsaacLab installs."""
    try:
        import isaaclab_tasks  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    try:
        import isaaclab
    except ModuleNotFoundError:
        return

    tasks_root = Path(isaaclab.__file__).resolve().parent / "source" / "isaaclab_tasks"
    if tasks_root.is_dir():
        tasks_root_str = str(tasks_root)
        if tasks_root_str not in sys.path:
            sys.path.append(tasks_root_str)


_ensure_isaaclab_tasks_on_path()

##
# Register Gym environments.
##

from isaaclab_tasks.utils import import_packages  # noqa: E402

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
