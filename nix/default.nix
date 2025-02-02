{
  imports = [ ./rust.nix ];

  perSystem =
    { self', ... }:
    {
      # rust + minimal wgpu deps for standard dev shell
      devShells.default = self'.devShells.rust;
    };
}
