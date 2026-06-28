// Everything here is shared between DX12 and the Windows Vulkan backend's
// DXGI presentation path, so cfgs are used to include the appropriate bits for each backend.
pub mod conv;
pub mod dcomp;
pub mod dxgi_lib;
#[cfg(dx12)]
pub mod exception;
pub mod factory;
pub mod hdr;
#[cfg(dx12)]
pub mod name;
pub mod result;
pub mod swapchain;
#[cfg(dx12)]
pub mod time;
#[cfg(dx12)]
pub mod types;
