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
    in
    {
      packages = forEachSupportedSystem (
        { pkgs, system }:
        let
          inherit (pkgs) stdenv;
        in
        {
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
          };

          # Note: pyMSTM's Python extension (src/pymstm/_mstm_ext*.so) is no
          # longer built as a Nix derivation (that was the ctypes-era
          # libmstm.so, built from the now-deleted mstm_wrapper.f90). It's
          # built by meson-python (see meson.build / pyproject.toml) against
          # the local git submodule at external/mstm/ -- gfortran below
          # covers the one system-level dependency that needs. Unlike
          # devenv.nix, this plain flake shell doesn't run `uv sync`
          # automatically -- run `uv sync --all-extras` once after entering
          # (pulls in meson/ninja/meson-python from the `dev` extra; see
          # pyproject.toml's [tool.uv] no-build-isolation-package comment
          # for why those need to land in the persistent venv, not an
          # ephemeral build env), then `import pymstm` rebuilds the
          # extension automatically whenever its Fortran sources change.
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
