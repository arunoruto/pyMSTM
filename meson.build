project(
  'pymstm',
  'c',
  'fortran',
  version: '0.1.0',
  meson_version: '>=1.1.0',
)

py = import('python').find_installation(pure: false)

ext_suffix = run_command(
  py, '-c', 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))',
  check: true,
).stdout().strip()

mstm_dir = meson.project_source_root() / 'external' / 'mstm' / 'december2023'
wrapper_dir = meson.project_source_root() / 'src' / 'pymstm' / '_fortran'
patches_dir = wrapper_dir / 'patches'

# --- Patched copies of two legacy source files ---
#
# numpy.f2py's Fortran parser (crackfortran) has real bugs that crash on
# certain DATA statements, and it refuses to expose any module-level
# variable of a module that itself defines a derived type. Both patches
# are pure Fortran refactors (see the .patch files for exactly what and
# why) -- no algorithm or runtime behavior changes -- applied only to
# scratch copies used for this extension. The standalone MSTM CLI binary
# keeps building from the literal unmodified upstream sources.
mstm_modules_patched = custom_target(
  'mstm-modules-33-patched',
  input: [
    mstm_dir / 'mstm-modules-33.f90',
    patches_dir / 'mstm-modules-33-f2py-data-fix.patch',
  ],
  output: 'mstm-modules-33.f90',
  command: [py, files('tools' / 'apply_patch.py'), '@INPUT0@', '@INPUT1@', '@OUTPUT@'],
)

random_config_patched = custom_target(
  'random-configuration-10-patched',
  input: [
    mstm_dir / 'random_configuration-10.f90',
    patches_dir / 'random_configuration-10-f2py-derivedtype-fix.patch',
  ],
  output: 'random_configuration-10.f90',
  command: [py, files('tools' / 'apply_patch.py'), '@INPUT0@', '@INPUT1@', '@OUTPUT@'],
)

# A second, independent patch on top of the one above: replaces
# hpsort_eps_epw, which upstream MSTM adapted from a GPLv2-licensed
# Quantum ESPRESSO/EPW routine, with a clean-room heapsort implementation
# (same signature/contract, written from scratch -- see the patch file).
# Needed to keep the whole compiled extension MIT-clean, since
# mstm-input-37.f90 has a compile-time `use random_sphere_configuration`
# dependency on this file even when the random-configuration feature
# itself isn't exposed to Python.
random_config_relicensed = custom_target(
  'random-configuration-10-relicensed',
  input: [
    random_config_patched,
    patches_dir / 'random_configuration-10-license-fix.patch',
  ],
  output: 'random_configuration-10-relicensed.f90',
  command: [py, files('tools' / 'apply_patch.py'), '@INPUT0@', '@INPUT1@', '@OUTPUT@'],
)

# --- The f2py extension itself ---
#
# Built via numpy.f2py's own "-c" compile mode (the same command
# `make f2py-ext` runs for local dev, see tools/build_f2py_ext.py for why
# a driver script is needed rather than a plain command: line -c always
# writes to the current working directory, with no way to target an
# arbitrary meson-controlled output path directly).
#
# Deliberately not using a .pyf signature file or SciPy's generate_f2pymod
# multi-step (.pyf -> .c -> compile) approach: this package has a single
# f2py extension module, not dozens sharing infrastructure, so the extra
# machinery isn't worth the added maintenance surface.
#
# Grows as more of mstm_f2py.f90 is written -- keep in sync with the
# Makefile's F2PY_ONLY.
f2py_only = [
  'mstm_init', 'mstm_set_verbose', 'mstm_finalize', 'mstm_set_spheres',
  'mstm_set_medium_ref', 'mstm_set_incident', 'mstm_set_solver_params',
  'mstm_set_layer_count', 'mstm_set_layer_thickness', 'mstm_set_layer_ref_index',
  'mstm_set_lattice', 'mstm_clear_lattice', 'mstm_prepare', 'mstm_solve',
  'mstm_scattering_angle', 'mstm_compute_tmatrix', 'mstm_ranorient_smatrix',
  'mstm_ranorient_smatrix_at_angle', 'mstm_set_scattering_map',
  'mstm_set_excitation_switch',
]

f2py_sources = [
  mstm_dir / 'mstm-intrinsics.f90',
  mstm_dir / 'mpidefs-serial.f90',
  mstm_modules_patched,
  mstm_dir / 'fft_translation-5.f90',
  mstm_dir / 'mstm-scatprops-26.f90',
  mstm_dir / 'mstm-solver-8.f90',
  random_config_relicensed,
  mstm_dir / 'lmfit.f90',
  mstm_dir / 'mstm-input-37.f90',
  wrapper_dir / 'mstm_f2py.f90',
]

mstm_ext = custom_target(
  '_mstm_ext',
  input: f2py_sources,
  output: '_mstm_ext' + ext_suffix,
  command: [
    py, files('tools' / 'build_f2py_ext.py'),
    '@OUTPUT@', ','.join(f2py_only), '@INPUT@',
  ],
  install: true,
  install_dir: py.get_install_dir(pure: false) / 'pymstm',
)

# --- Pure Python package ---

python_sources = [
  'src' / 'pymstm' / '__init__.py',
  'src' / 'pymstm' / '_build.py',
  'src' / 'pymstm' / '_cli.py',
  'src' / 'pymstm' / '_config.py',
  'src' / 'pymstm' / '_convert.py',
  'src' / 'pymstm' / '_inp.py',
  'src' / 'pymstm' / '_inp_parser.py',
  'src' / 'pymstm' / '_mstm.py',
  'src' / 'pymstm' / '_parser.py',
  'src' / 'pymstm' / '_solver.py',
]

py.install_sources(
  python_sources,
  subdir: 'pymstm',
)
