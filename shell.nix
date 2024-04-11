{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell rec {
  buildInputs = with pkgs; [
    rustup
    cargo-nextest cargo-fuzz
    typos
    fd yq

    gdb lldb rr
    clang llvmPackages.libclang
    libgcc.lib
    openssl pkg-config cmake
    evcxr tokei
    libxkbcommon
    wayland xorg.libX11 xorg.libXcursor xorg.libXrandr xorg.libXi
    alsa-lib
    fontconfig freetype

    shaderc directx-shader-compiler
    libGL
    vulkan-headers vulkan-loader
    vulkan-tools vulkan-tools-lunarg
    vulkan-validation-layers
    vulkan-extension-layer

    python3 valgrind
    renderdoc
    gnuplot
  ];

  LIBCLANG_PATH = pkgs.lib.makeLibraryPath [pkgs.llvmPackages.libclang.lib];

  BINDGEN_EXTRA_CLANG_ARGS = with pkgs.llvmPackages_latest.libclang; [
    ''-I"${lib}/lib/clang/${version}"''
  ];

  shellHook = ''
    export RUSTC_VERSION="$(tomlq -r .toolchain.channel rust-toolchain.toml)"
    export PATH="$PATH:''${CARGO_HOME:-~/.cargo}/bin"
    export PATH="$PATH:''${RUSTUP_HOME:-~/.rustup/toolchains/$RUSTC_VERSION-x86_64-unknown-linux/bin}"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${builtins.toString (pkgs.lib.makeLibraryPath buildInputs)}";

    rustup default $RUSTC_VERSION
    rustup component add rust-src rust-analyzer
  '';
}
