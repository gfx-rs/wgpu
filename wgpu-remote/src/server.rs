/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use core::{gfx_select, hub::Global, id};

use std::slice;

#[no_mangle]
pub extern "C" fn wgpu_server_new() -> *mut Global<()> {
    log::info!("Initializing WGPU server");
    Box::into_raw(Box::new(Global::new("wgpu")))
}

#[no_mangle]
pub extern "C" fn wgpu_server_delete(global: *mut Global<()>) {
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
    global: &Global<()>,
    desc: &core::instance::RequestAdapterOptions,
    ids: *const id::AdapterId,
    id_length: usize,
) -> i8 {
    let ids = unsafe { slice::from_raw_parts(ids, id_length) };
    match global.pick_adapter(
        desc,
        core::instance::AdapterInputs::IdSet(ids, |i| i.backend()),
    ) {
        Some(id) => ids.iter().position(|&i| i == id).unwrap() as i8,
        None => -1,
    }
}

#[no_mangle]
pub extern "C" fn wgpu_server_adapter_request_device(
    global: &Global<()>,
    self_id: id::AdapterId,
    desc: &core::instance::DeviceDescriptor,
    new_id: id::DeviceId,
) {
    gfx_select!(self_id => global.adapter_request_device(self_id, desc, new_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_device_destroy(global: &Global<()>, self_id: id::DeviceId) {
    gfx_select!(self_id => global.device_destroy(self_id))
}
