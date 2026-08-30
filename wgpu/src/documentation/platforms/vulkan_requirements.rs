/*!
# Vulkan Requirements

`wgpu`'s Vulkan backend hides a physical device that does not meet all of the
following:

- Vulkan 1.1, or `VK_KHR_maintenance1`.
- `VK_KHR_storage_buffer_storage_class`, on Vulkan 1.0 devices.
- A first queue family that supports graphics.
- A driver conformance version with a non-zero major number, unless the driver
  is MoltenVK or [`InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER`] is
  set.

A physical device that lacks any of the following is still exposed, but the
matching [`DownlevelFlags`] bit is cleared:

- `fragmentStoresAndAtomics`, for [`DownlevelFlags::FRAGMENT_WRITABLE_STORAGE`]
  (see <https://github.com/gpuweb/gpuweb/issues/639>).
- `independentBlend`, for [`DownlevelFlags::INDEPENDENT_BLEND`] (see
  [wgpu#2498](https://github.com/gfx-rs/wgpu/pull/2498)).
- `fullDrawIndexUint32`, for [`DownlevelFlags::FULL_DRAW_INDEX_UINT32`].
*/

use crate::{DownlevelFlags, InstanceFlags};
