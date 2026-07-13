{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  # pyMSTM's Python extension (src/pymstm/_mstm_ext*.so) is built by
  # meson-python (see meson.build / pyproject.toml's [build-system]),
  # which shells out to `numpy.f2py -c` (tools/build_f2py_ext.py) against
  # the git submodule at external/mstm/. That needs, at minimum, gfortran
  # on PATH -- provided below. meson, ninja, and meson-python itself are
  # Python-level build requirements pulled in via the `dev` extra in
  # pyproject.toml, installed into this shell's venv by
  # languages.python.uv.sync (allExtras below) rather than as Nix packages
  # -- pyproject.toml's [tool.uv] no-build-isolation-package = ["pymstm"]
  # requires them to live in *this* persistent venv (not an ephemeral
  # isolated build env), since meson-python's editable-install rebuild
  # hook records an absolute path to `ninja` the first time it builds and
  # reuses that same path on every later `import pymstm`.
  #
  # Once this shell is set up, `import pymstm` alone keeps the extension
  # up to date automatically -- `make f2py-ext`/`make cli` (both still
  # present in the Makefile) are only needed for one-off builds outside a
  # full editable install, e.g. producing a standalone .so or the CLI
  # binary without touching the venv.
  packages = [ pkgs.gfortran ];

  enterShell = ''
    if [ ! -L "$DEVENV_ROOT/.venv" ]; then
        ln -s "$DEVENV_STATE/venv/" "$DEVENV_ROOT/.venv"
    fi
  '';

  languages.python = {
    enable = true;

    uv = {
      enable = true;
      sync = {
        enable = true;
        allExtras = true;
      };
    };

    libraries = with pkgs; [ zlib ];
  };
}
