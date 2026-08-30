/*!
## Feature flags
*/
#![doc = document_features::document_features!()]
/*!

### Feature Aliases

These features aren't actually features on the crate itself, but a convenient shorthand for
complicated cases.

- **`wgpu_core`** --- Enabled on every non-wasm target, and on wasm when any non-webgpu backend is
  enabled.
- **`naga`** --- Enabled when the `naga-ir`, `spirv`, or `glsl` feature is enabled. `wgpu_core` does
  not imply it.
*/

use crate::Adapter;
