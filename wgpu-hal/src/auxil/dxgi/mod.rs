// Every module here is gated behind `dx12` except `hdr` (the
// `DXGI_OUTPUT_DESC1` -> `DisplayHdrInfo` mapping), which the
// Vulkan-on-Windows backend also uses.
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
