#!/usr/bin/env python3
"""meson dist script: drop the parts of the MSTM submodule the sdist
does not need.

The sdist only has to build the extension from external/mstm/december2023.
Upstream's documentation (notebook, preprint PDFs), examples and the older
code/ tree are 25+ MB, and upstream's random_configuration-10.f90 contains
a GPL-licensed routine; pyMSTM compiles its own modified copy instead
(src/pymstm/_fortran/random_configuration-10.f90).
"""

import os
import shutil
from pathlib import Path

root = Path(os.environ["MESON_PROJECT_DIST_ROOT"]) / "external" / "mstm"
for sub in ("doc", "examples", "code"):
    shutil.rmtree(root / sub, ignore_errors=True)
(root / "december2023" / "random_configuration-10.f90").unlink(missing_ok=True)
