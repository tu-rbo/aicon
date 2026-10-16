"""
MuJoCo Viewer on macOS

On macOS, MuJoCo's interactive viewer (used by robosuite's "mjviewer" renderer) only runs if the script is started
with MuJoCo's `mjpython` launcher instead of `python`, which keeps the main thread free for the GUI.

    if args.render:
        relaunch_under_mjpython()   # no-op on Linux and when already running under mjpython

Authors: Adrian Pfisterer
License: MIT
"""

__author__ = "Adrian Pfisterer"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import os
import shutil
import sys


def relaunch_under_mjpython():
    """On macOS, replace the current process with the same command run under `mjpython` (installed with MuJoCo next
    to the Python interpreter). Does nothing on other platforms or when already running under `mjpython`."""
    if sys.platform != "darwin" or "MJPYTHON_BIN" in os.environ:  # mjpython sets MJPYTHON_BIN
        return
    mjpython = os.path.join(os.path.dirname(sys.executable), "mjpython")
    if not os.path.exists(mjpython):
        mjpython = shutil.which("mjpython")
    if mjpython is None:
        sys.exit("The MuJoCo viewer needs `mjpython` on macOS, which was not found (it is installed with mujoco)")
    spec = sys.modules["__main__"].__spec__
    target = ["-m", spec.name] if spec is not None else [sys.argv[0]]
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(mjpython, [mjpython, *target, *sys.argv[1:]])
