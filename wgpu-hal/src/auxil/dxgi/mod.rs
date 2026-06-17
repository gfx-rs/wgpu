// The bulk of this module is DX12-only. Only `hdr` (the pure
// `DXGI_OUTPUT_DESC1` → `DisplayHdrInfo` mapping) is shared with the
// Vulkan-on-Windows backend, so the rest stays gated behind `dx12`.
#[cfg(dx12)]
pub mod conv;
#[cfg(dx12)]
pub mod exception;
#[cfg(dx12)]
pub mod factory;
pub mod hdr;
#[cfg(dx12)]
pub mod name;
#[cfg(dx12)]
pub mod result;
#[cfg(dx12)]
pub mod time;
