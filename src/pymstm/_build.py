"""Build helper for compiling the pyMSTM f2py extension.

Usage:
    python -m pymstm._build

Requires gfortran, numpy (with f2py), meson, and ninja (meson/ninja are
invoked internally by numpy.f2py on Python >=3.12), and the MSTM submodule
at external/mstm/.
"""

import glob
import os
import subprocess
import sys


def build_library():
    """Compile the f2py extension via `make f2py-ext`."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    mstm_submodule = os.path.join(project_root, "external", "mstm", "december2023")
    if not os.path.isdir(mstm_submodule):
        sys.exit(
            "MSTM submodule not found at external/mstm/\n"
            "Initialize it with:\n"
            "  git submodule update --init --recursive"
        )

    makefile = os.path.join(project_root, "Makefile")
    if not os.path.isfile(makefile):
        sys.exit(f"Makefile not found at {makefile}")

    result = subprocess.run(
        ["make", "-C", project_root, "f2py-ext"],
        capture_output=False,
    )

    if result.returncode != 0:
        sys.exit(result.returncode)

    ext_glob = os.path.join(project_root, "src", "pymstm", "_mstm_ext*.so")
    matches = glob.glob(ext_glob)
    if matches:
        print(f"Built: {matches[0]}")
    else:
        sys.exit("Build succeeded but _mstm_ext*.so not found")


if __name__ == "__main__":
    build_library()
