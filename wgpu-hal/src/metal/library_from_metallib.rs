use core::{ffi::c_void, ptr::NonNull};

use objc2::{
    msg_send,
    rc::Retained,
    runtime::{AnyObject, ProtocolObject},
};
use objc2_foundation::NSError;
use objc2_metal::{MTLDevice, MTLLibrary};

#[link(name = "System")] // libdispatch lives inside libSystem
extern "C" {
    fn dispatch_data_create(
        buffer: NonNull<c_void>,
        size: usize,
        queue: Option<NonNull<c_void>>, // dispatch_queue_t _Nullable
        destructor: *mut c_void,        // dispatch_block_t _Nullable
    ) -> *mut AnyObject; // dispatch_data_t

    fn dispatch_release(object: *mut AnyObject); // dispatch_data_t
}

const DISPATCH_DATA_DESTRUCTOR_DEFAULT: *mut c_void = core::ptr::null_mut();

pub(crate) fn new_library_from_metallib_bytes(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[u8],
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, Retained<NSError>> {
    let buffer = NonNull::new(data.as_ptr().cast_mut()).unwrap().cast();
    let data =
        unsafe { dispatch_data_create(buffer, data.len(), None, DISPATCH_DATA_DESTRUCTOR_DEFAULT) };
    assert!(!data.is_null());

    let res: Result<Retained<ProtocolObject<dyn MTLLibrary>>, Retained<NSError>> =
        unsafe { msg_send![device, newLibraryWithData: data, error: _] };

    unsafe { dispatch_release(data) };
    res
}
