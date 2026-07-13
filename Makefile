# Makefile for building the pyMSTM f2py extension and the standalone
# MSTM CLI binary. Requires the MSTM submodule at external/mstm/

FC = gfortran
BUILD_DIR = build
FFLAGS = -O2 -fPIC -fno-strict-aliasing -J$(BUILD_DIR) -I$(BUILD_DIR)

MSTM = external/mstm/december2023
WRAPPER = src/pymstm/_fortran

CLI_OUTPUT = $(BUILD_DIR)/mstm

# Module sources for the CLI binary
MODULE_SRCS = $(MSTM)/mstm-intrinsics.f90 \
              $(MSTM)/mpidefs-serial.f90 \
              $(MSTM)/mstm-modules-33.f90 \
              $(MSTM)/fft_translation-5.f90 \
              $(MSTM)/mstm-scatprops-26.f90 \
              $(MSTM)/mstm-solver-8.f90 \
              $(MSTM)/random_configuration-10.f90 \
              $(MSTM)/lmfit.f90 \
              $(MSTM)/mstm-input-37.f90

MODULE_OBJS = $(patsubst %.f90,$(BUILD_DIR)/%.o,$(notdir $(MODULE_SRCS)))

# --- f2py extension (the Python-facing pyMSTM backend) ---
#
# The actual build recipe (which sources, the two patches needed to work
# around real numpy.f2py parser bugs, the `only:` function list) lives in
# meson.build -- this target is a thin wrapper so `make f2py-ext` still
# works as a quick one-off, without needing a full `pip install -e .`.
# Normal local development doesn't need this at all: meson-python's
# editable install (see pyproject.toml's [tool.uv] no-build-isolation
# -package) rebuilds the extension automatically on `import pymstm`
# whenever a source file changed. The CLI binary below keeps building
# from the unmodified upstream sources, so it remains an independent
# reference (see tests/test_compatibility.py).
F2PY_MESON_BUILD_DIR = $(BUILD_DIR)/f2py-meson

.PHONY: f2py-ext
f2py-ext:
	test -f $(F2PY_MESON_BUILD_DIR)/build.ninja || meson setup $(F2PY_MESON_BUILD_DIR)
	meson compile -C $(F2PY_MESON_BUILD_DIR)
	rm -f src/pymstm/_mstm_ext*.so
	cp $(F2PY_MESON_BUILD_DIR)/_mstm_ext*.so src/pymstm/

.PHONY: all clean cli

all: f2py-ext cli

cli: $(CLI_OUTPUT)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# --- Module compilation rules (CLI only) ---

$(BUILD_DIR)/mstm-intrinsics.o: $(MSTM)/mstm-intrinsics.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD_DIR)/mpidefs-serial.o: $(MSTM)/mpidefs-serial.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD_DIR)/mstm-modules-33.o: $(MSTM)/mstm-modules-33.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD_DIR)/fft_translation-5.o: $(MSTM)/fft_translation-5.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD_DIR)/mstm-scatprops-26.o: $(MSTM)/mstm-scatprops-26.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD_DIR)/mstm-solver-8.o: $(MSTM)/mstm-solver-8.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD_DIR)/random_configuration-10.o: $(MSTM)/random_configuration-10.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD_DIR)/lmfit.o: $(MSTM)/lmfit.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD_DIR)/mstm-input-37.o: $(MSTM)/mstm-input-37.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -c $< -o $@

# --- Main program compilation (for CLI binary) ---

$(BUILD_DIR)/mstm-main-3.o: $(MSTM)/mstm-main-3.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -c $< -o $@

# --- Linking ---

$(CLI_OUTPUT): $(MODULE_OBJS) $(BUILD_DIR)/mstm-main-3.o
	$(FC) $(FFLAGS) -o $@ $(MODULE_OBJS) $(BUILD_DIR)/mstm-main-3.o

clean:
	rm -rf $(BUILD_DIR)
	rm -f src/pymstm/_mstm_ext*.so
	rm -rf .mesonpy-*
	rm -f *.mod *.o
