{
  description = "Python bindings for the MSTM (Multiple Sphere T-Matrix) Fortran library";

  inputs = {
    nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0";
    mstm-src = {
      url = "github:dmckwski/MSTM/a0c982121cf9ac352531f4816639a07d814385bd";
      flake = false;
    };
  };

  outputs =
    { self, nixpkgs, mstm-src, ... }@inputs:
    let
      inherit (nixpkgs) lib;

      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      forEachSupportedSystem =
        f:
        lib.genAttrs supportedSystems (
          system:
          f {
            inherit system;
            pkgs = import nixpkgs {
              inherit system;
              config.allowUnfree = true;
            };
          }
        );

      mstmModuleSrcs = [
        "mstm-intrinsics.f90"
        "mstm-modules-33.f90"
        "fft_translation-5.f90"
        "mstm-scatprops-26.f90"
        "mstm-solver-8.f90"
        "random_configuration-10.f90"
        "lmfit.f90"
        "mstm-input-37.f90"
      ];

      compileMstmModules =
        fc: mpidefsSrc:
        ''
          ${fc} -O2 -fPIC -fno-strict-aliasing -J. -I. -c mstm-intrinsics.f90
          ${fc} -O2 -fPIC -fno-strict-aliasing -J. -I. -c ${mpidefsSrc}
          ${builtins.concatStringsSep "\n" (builtins.map (s: "${fc} -O2 -fPIC -fno-strict-aliasing -J. -I. -c ${s}") (builtins.filter (s: s != "mstm-intrinsics.f90") mstmModuleSrcs))}
        '';

      mstmSrc = mstm-src + "/december2023";

      pyprojectVersion = (lib.importTOML ./pyproject.toml).project.version;
    in
    {
      packages = forEachSupportedSystem (
        { pkgs, system }:
        let
          inherit (pkgs) stdenv;
        in
        rec {
          mstm = stdenv.mkDerivation {
            pname = "mstm";
            version = "december2023";
            src = mstmSrc;
            nativeBuildInputs = [ pkgs.gfortran ];
            buildPhase =
              (compileMstmModules "gfortran" "mpidefs-serial.f90")
              + ''
                gfortran -O2 -fPIC -fno-strict-aliasing -J. -I. -c mstm-main-3.f90
                gfortran -O2 -o mstm *.o
              '';
            installPhase = ''
              mkdir -p $out/bin
              cp mstm $out/bin/
            '';

            meta = with lib; {
              description = "Multiple Sphere T-Matrix code in Fortran (serial build, december2023 module layout)";
              homepage = "https://github.com/dmckwski/MSTM";
              license = licenses.mit;
              platforms = platforms.unix;
              maintainers = with maintainers; [ arunoruto ];
              mainProgram = "mstm";
            };
          };

          mstm-mpi = stdenv.mkDerivation {
            pname = "mstm-mpi";
            version = "december2023";
            src = mstmSrc;
            nativeBuildInputs = [ pkgs.gfortran pkgs.openmpi ];
            buildPhase =
              (compileMstmModules "mpifort" "mpidefs-parallel-2.f90")
              + ''
                mpifort -O2 -fPIC -fno-strict-aliasing -J. -I. -c mstm-main-3.f90
                mpifort -O2 -o mstm-mpi *.o
              '';
            installPhase = ''
              mkdir -p $out/bin
              cp mstm-mpi $out/bin/
            '';

            meta = with lib; {
              description = "Multiple Sphere T-Matrix code in Fortran (MPI-parallel build, december2023 module layout)";
              homepage = "https://github.com/dmckwski/MSTM";
              license = licenses.mit;
              platforms = platforms.unix;
              maintainers = with maintainers; [ arunoruto ];
              mainProgram = "mstm-mpi";
            };
          };

          # The Python bindings, as a real Nix derivation -- meson-python
          # drives the exact same meson.build/tools/build_f2py_ext.py that
          # `uv sync` triggers locally, so the compiled extension is the
          # same numpy.f2py output either way.
          #
          # external/mstm is a git submodule, and Nix flakes only ever see
          # a git-tracked copy of the flake's own source (via `self`) --
          # a submodule's checked-out content is invisible to that copy
          # (confirmed directly: `self`'s `external/mstm` doesn't even
          # exist as a directory, since git records a submodule as a
          # single gitlink entry, not the files inside it). Rather than
          # fetch the source a second time, postPatch repopulates
          # external/mstm from the *same* mstm-src input the CLI
          # derivations above already use.
          pymstm = pkgs.python3Packages.buildPythonPackage {
            pname = "pymstm";
            version = pyprojectVersion;
            pyproject = true;
            src = self;

            postPatch = ''
              rm -rf external/mstm
              mkdir -p external
              cp -r ${mstm-src} external/mstm
              chmod -R u+w external/mstm
            '';

            # nixpkgs' meson-python setup hook pre-runs `meson setup` as
            # its own configurePhase and hands `pypaBuildHook` a
            # `-Cbuild-dir=` pointing at it -- on this nixpkgs/meson-python
            # pairing that pre-configured dir confuses `python -m build`
            # into treating the *build* dir as the source root ("Source
            # .../build does not appear to be a Python project"). Skipping
            # that pre-configure step lets meson-python's own build
            # backend invoke meson itself from the real source root,
            # which is its normal, fully self-contained mode of operation.
            dontUseMesonConfigure = true;

            build-system = [ pkgs.python3Packages.meson-python ];
            nativeBuildInputs = [
              pkgs.meson
              pkgs.ninja
              pkgs.gfortran
              pkgs.gnupatch # applies the f2py-compatibility patches under src/pymstm/_fortran/patches/
            ];
            dependencies = [ pkgs.python3Packages.numpy ];

            # The test suite cross-checks against the standalone `mstm`
            # CLI binary (see the derivation above) and real MPI runs --
            # neither is wired up as a build input here, so skip pytest
            # during the Nix build; test.yml's CI job covers that.
            doCheck = false;
            pythonImportsCheck = [ "pymstm" ];

            meta = with lib; {
              description = "Python bindings for the MSTM (Multiple Sphere T-Matrix) Fortran library";
              homepage = "https://github.com/arunoruto/pyMSTM";
              license = licenses.mit;
              platforms = platforms.unix;
              maintainers = with maintainers; [ arunoruto ];
            };
          };

          default = mstm;
        }
      );

      devShells = forEachSupportedSystem (
        { pkgs, system }:
        {
          default = pkgs.mkShellNoCC {
            packages =
              with pkgs;
              [
                self.formatter.${system}
                gfortran
              ]
              ++ lib.optionals (!pkgs.stdenv.isDarwin) [
                self.packages.${system}.mstm
              ];
          };
        }
      );

      formatter = forEachSupportedSystem ({ pkgs, ... }: pkgs.nixfmt);
    };
}
