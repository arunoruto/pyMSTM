#!/usr/bin/env python3
"""Build the pyMSTM f2py extension and place it at the path meson expects.

numpy.f2py's "-c" compile mode always writes its output to the current
working directory, using a filename that includes the Python/platform ABI
tag -- there's no flag to make it write directly to an arbitrary path (see
--build-dir, which only controls where *intermediate* meson files land,
not the final .so). This driver runs the exact same command
`make f2py-ext` runs locally, in a scratch directory, then copies the
resulting extension module to the exact output path meson's custom_target
requires.

Usage: build_f2py_ext.py <output_path> <comma_separated_only_names> <source...>
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile


def main():
    output_path = os.path.abspath(sys.argv[1])
    only_names = sys.argv[2].split(",")
    # f2py's meson backend always runs relative to the scratch cwd below;
    # resolve source paths to absolute first so they still resolve there.
    sources = [os.path.abspath(s) for s in sys.argv[3:]]

    env = dict(os.environ)
    # -z noexecstack is a GNU ld (ELF) hardening flag with no equivalent
    # meaning on macOS's Mach-O linker, which doesn't understand -z at
    # all and aborts with "unknown options: -z" -- Linux-only. Never
    # caught before because this extension was never actually built on
    # macOS until cibuildwheel's macOS legs (see .github/workflows/
    # publish.yml) tried it for the first time.
    if sys.platform.startswith("linux"):
        ldflags = "-Wl,-z,noexecstack"
        env["LDFLAGS"] = f"{env['LDFLAGS']} {ldflags}" if env.get("LDFLAGS") else ldflags

    with tempfile.TemporaryDirectory() as scratch:
        cmd = [
            sys.executable,
            "-m",
            "numpy.f2py",
            "-c",
            *sources,
            "only:",
            *only_names,
            ":",
            "-m",
            "_mstm_ext",
        ]
        subprocess.run(cmd, cwd=scratch, env=env, check=True)
        matches = glob.glob(os.path.join(scratch, "_mstm_ext*.so")) + glob.glob(
            os.path.join(scratch, "_mstm_ext*.pyd")
        )
        if not matches:
            sys.exit("f2py build did not produce _mstm_ext*.so/.pyd")
        shutil.copy(matches[0], output_path)


if __name__ == "__main__":
    main()
