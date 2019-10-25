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
    //TODO: proper cleanup
    let _ = unsafe { Box::from_raw(global) };
}

#[no_mangle]
pub extern "C" fn wgpu_server_instance_request_adapter(
    global: &wgn::Global,
    desc: &wgn::RequestAdapterOptions,
    ids: *const wgn::AdapterId,
    id_length: usize,
) -> wgn::AdapterId {
    let ids = unsafe { slice::from_raw_parts(ids, id_length) };
    wgn::request_adapter(global, desc, ids).unwrap()
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
