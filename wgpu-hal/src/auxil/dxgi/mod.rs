use core::mem;

use windows::core::Interface;

// Everything here is shared between DX12 and the Windows Vulkan backend's
// DXGI presentation path, so cfgs are used to include the appropriate bits for each backend.
pub mod conv;
pub mod d3d12_lib;
pub mod dcomp;
pub mod device_factory;
pub mod dxgi_lib;
#[cfg(dx12)]
pub mod exception;
pub mod factory;
#[cfg(vulkan)]
pub mod handles;
pub mod hdr;
#[cfg(dx12)]
pub mod name;
pub mod result;
pub mod swapchain;
#[cfg(dx12)]
pub mod time;
#[cfg(dx12)]
pub mod types;

/// Create a temporary "owned" copy inside a [`mem::ManuallyDrop`] without increasing the refcount or
/// moving away the source variable.
///
/// This is a common pattern when needing to pass interface pointers ("borrows") into Windows
/// structs.  Moving/cloning ownership is impossible/inconvenient because:
///
/// - The caller does _not_ assume ownership (and decrement the refcount at a later time);
/// - Unnecessarily increasing and decrementing the refcount;
/// - [`Drop`] destructors cannot run inside `union` structures (when the created structure is
///   implicitly dropped after a call).
///
/// See also <https://github.com/microsoft/windows-rs/pull/2361#discussion_r1150799401> and
/// <https://github.com/microsoft/windows-rs/issues/2386>.
///
/// # Safety
/// The returned value holds the pointer of `src` without a reference of its own. It must not be
/// dropped, and must not be used after `src` is released.
pub(crate) unsafe fn borrow_interface_temporarily<I: Interface>(
    src: &I,
) -> mem::ManuallyDrop<Option<I>> {
    mem::ManuallyDrop::new(Some(unsafe { I::from_raw(src.as_raw()) }))
}

/// See [`borrow_interface_temporarily()`]
#[cfg(dx12)]
pub(crate) unsafe fn borrow_optional_interface_temporarily<I: Interface>(
    src: &Option<I>,
) -> mem::ManuallyDrop<Option<I>> {
    mem::ManuallyDrop::new(src.as_ref().map(|src| unsafe { I::from_raw(src.as_raw()) }))
}
