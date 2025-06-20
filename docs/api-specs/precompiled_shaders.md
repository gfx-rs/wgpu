# Precompiled shaders
There are two main issues an implementation needs to cover
* Including and using reflection info
* Exposing how individual backends compile shaders outside of the backends
What changes need to be made
* I propose making a new crate, `wgpu-shaders`
  * This crate would be a "wrapper" around `naga`, that would include all shader compiling logic
  * This logic could then be used by both compile time macros and `wgpu-hal` itself
  * This crate would include "backend"-specific parts, but it wouldn't need actual access to backends
* I also propose moving many `naga` types into `wgpu-types`, primarily those useful for reflection.
  * The type to look out for here is `wgpu_core::validation::Interface`. This would also need to be moved into `wgpu-types`