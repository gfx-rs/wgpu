/*!
# Vulkan Requirements

`wgpu`'s Vulkan backend requires the following from a physical device:

- `fragmentStoresAndAtomics` must be enabled (see
  <https://github.com/gpuweb/gpuweb/issues/639>).
- `independentBlending` must be enabled (see
  [wgpu#2498](https://github.com/gfx-rs/wgpu/pull/2498)).
- The `maxDrawIndexedIndexValue` limit must be 32-bit.
- `VK_KHR_storage_buffer_storage_class` must be supported.
*/
