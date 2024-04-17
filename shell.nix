# This file is only relevant for Nix and NixOS users.
# What's actually meant by "Nix" here is not UNIX, but the *package manager* Nix, see https://nixos.org/.
# If you are
#   on macOS (and not using nix-darwin)
#   or on Windows (and not using Nix in WSL),
# you can carelessly ignore this file.
#
# Otherwise, if you *do* use Nix the package manager,
# this file declares
#   common dependencies
#   and some nice tools
# which you'll most likely need when working with wgpu.
# Feel free to copy it into your own project if deemed useful.
#
# To use this file, just run `nix-shell` in this folder,
# which will drop you into a shell
# with all the deps needed for building wgpu available.
#
# Or if you're using direnv (https://direnv.net/),
# use `direnv allow` to automatically always use this file
# if you're navigating into this or a subfolder.

{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell rec {
  buildInputs = with pkgs; [
    # necessary for building wgpu in 3rd party packages (in most cases)
    libxkbcommon
    wayland xorg.libX11 xorg.libXcursor xorg.libXrandr xorg.libXi
    alsa-lib
    fontconfig freetype
    shaderc directx-shader-compiler
    pkg-config cmake
    mold # could use any linker, needed for rustix (but mold is fast)

    libGL
    vulkan-headers vulkan-loader
    vulkan-tools vulkan-tools-lunarg
    vulkan-extension-layer
    vulkan-validation-layers # don't need them *strictly* but immensely helpful

    # necessary for developing (all of) wgpu itself
    cargo-nextest cargo-fuzz

    # nice for developing wgpu itself
    typos 

    # if you don't already have rust installed through other means,
    # this shell.nix can do that for you with this below
    yq # for tomlq below
    rustup

    # nice tools
    gdb rr
    evcxr
    valgrind
    renderdoc
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
