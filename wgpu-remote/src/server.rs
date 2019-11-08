/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::slice;

#[no_mangle]
pub extern "C" fn wgpu_server_new() -> *mut wgn::Global {
    log::info!("Initializing WGPU server");
    Box::into_raw(Box::new(wgn::Global::new("wgpu")))
}

#[no_mangle]
pub extern "C" fn wgpu_server_delete(global: *mut wgn::Global) {
    log::info!("Terminating WGPU server");
    unsafe { Box::from_raw(global) }.delete();
    log::info!("\t...done");
}

/// Request an adapter according to the specified options.
/// Provide the list of IDs to pick from.
///
/// Returns the index in this list, or -1 if unable to pick.
#[no_mangle]
pub extern "C" fn wgpu_server_instance_request_adapter(
    global: &wgn::Global,
    desc: &wgn::RequestAdapterOptions,
    ids: *const wgn::AdapterId,
    id_length: usize,
) -> i8 {
    extern "C" fn request_adapter_callback(
        data: *const wgn::AdapterId,
        user_data: *mut std::ffi::c_void,
    ) {
        unsafe {
            *(user_data as *mut wgn::AdapterId) = *data;
        }
    }

    let ids = unsafe { slice::from_raw_parts(ids, id_length) };
    let mut adapter_id: wgn::AdapterId = wgn::AdapterId::ERROR;
    let adapter_id_ref = &mut adapter_id;
    wgn::request_adapter_async(global, desc, ids, request_adapter_callback, adapter_id_ref as *mut _ as *mut std::ffi::c_void);
    if adapter_id == wgn::AdapterId::ERROR {
        -1
    } else {
        ids.iter().position(|&i| i == adapter_id).unwrap() as i8
    }
}

#[no_mangle]
pub extern "C" fn wgpu_server_adapter_request_device(
    global: &wgn::Global,
    self_id: wgn::AdapterId,
    desc: &wgn::DeviceDescriptor,
    new_id: wgn::DeviceId,
) {
    use wgn::adapter_request_device as func;
    wgn::gfx_select!(self_id => func(global, self_id, desc, new_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_device_destroy(global: &wgn::Global, self_id: wgn::DeviceId) {
    use wgn::device_destroy as func;
    wgn::gfx_select!(self_id => func(global, self_id))
}
