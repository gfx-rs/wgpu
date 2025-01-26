{
  description = "Front-end for chat backends";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";  # Specify the Nixpkgs version
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
  in
  {
		devShells.${system}.default = pkgs.mkShell {
    	    packages = with pkgs; [
    	      cargo
    	      rustc
    	      rust-analyzer
    	      rustfmt

    	      libxkbcommon
    	      wayland

			  vulkan-loader
			  vulkan-validation-layers
			  vulkan-tools

			  libappindicator

			  openssl
			  pkgs.pkg-config
    	    ];	
			LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
				pkgs.libxkbcommon
				pkgs.wayland
				pkgs.vulkan-loader
    		
				pkgs.freetype
				pkgs.fontconfig
			];

    	    RUST_BACKTRACE = "full";
    	    WINIT_UNIX_BACKEND = "wayland";
    	};
	};
}
