{ inputs, ... }:
{
  perSystem =
    {
      self',
      pkgs,
      system,
      ...
    }:
    let
      fnx = inputs.fenix.packages.${system};
    in
    {
      # rust toolchain
      packages.rust = fnx.combine [
        fnx.stable.cargo
        fnx.stable.rust-src
        fnx.stable.rustc

        # it's generally recommended to use nightly rustfmt
        fnx.complete.rustfmt

        # utils
        fnx.stable.clippy
        fnx.stable.rust-analyzer
      ];

      devShells.rust =
        let
          # window manager deps, could probably split this based on system but
          # this is simpler and doesn't hurt that much
          wmDeps = [
            # wayland
            pkgs.wayland
            pkgs.libxkbcommon

            # x
            pkgs.xorg.libX11
            pkgs.xorg.libXcursor
            pkgs.xorg.libXrandr
            pkgs.xorg.libXi
          ];
          vulkanDeps = [
            # this is the main one to get it running
            pkgs.vulkan-loader
            # this is just for extra features
            pkgs.vulkan-headers
            pkgs.vulkan-tools
            pkgs.vulkan-tools-lunarg
            pkgs.vulkan-extension-layer
            pkgs.vulkan-validation-layers # don't need them *strictly* but immensely helpful
          ];
          nixRustDeps = [
            pkgs.pkg-config
            pkgs.udev
          ];
          allDeps = wmDeps ++ vulkanDeps ++ nixRustDeps;
        in
        pkgs.mkShell {
          name = "Development shell for working on wgpu";
          packages = [ self'.packages.rust ];
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath allDeps;
        };
    };
}
