# Requested changes to CLI

* Prettier outputs (colors, etc) when it makes sense, like for errors and also for the help message
* Allow using statically linked DXC, dynamically linked (through dlload or whatever its called), or the CLI
  * Same for spirv-val and spirv-opt I think, default to statically linking all of it
* DXC output to stdout by using a tempfile or something
* Consider allowing stdout output with json info somehow, maybe json goes to stderr?
* Ensure every current naga option is exposed
* Somehow find some way to indicate to contributors that they should expose their new options and features to the CLI
* Snapshot that displays reflection info
* Improve reflection, and have snapshots that test every useful kind of reflection
  * Every shader stage, so mesh outputs & task payload sizes
  * Vertex attributes
  * Fragment outputs
  * Interstage variables
  * Structs used with buffers, push constants, uniforms etc
* What does this mean and should it be changed?
  > --print-config-schema omits `spv_out.capabilities` (its type has no JsonSchema impl), but
  > --config / --config-json still accept it.
* Wtf is `--block-ctx-dir`
* Exhaustively list all GLSL profile options and make `--profile` into `--glsl-profile`
  * Same for other options like `--metal-version`