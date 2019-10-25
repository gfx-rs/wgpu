/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use wgn::{AdapterId, Backend, DeviceId, IdentityManager, SurfaceId};

use parking_lot::Mutex;

use std::{ptr, slice};

pub mod server;

#[derive(Debug)]
struct IdentityHub {
    adapters: IdentityManager<AdapterId>,
    devices: IdentityManager<DeviceId>,
}

impl IdentityHub {
    fn new(backend: Backend) -> Self {
        IdentityHub {
            adapters: IdentityManager::new(backend),
            devices: IdentityManager::new(backend),
        }
    }
}

#[derive(Debug)]
struct Identities {
    surfaces: IdentityManager<SurfaceId>,
    vulkan: IdentityHub,
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    metal: IdentityHub,
    #[cfg(windows)]
    dx12: IdentityHub,
}

impl Identities {
    fn new() -> Self {
        Identities {
            surfaces: IdentityManager::new(Backend::Empty),
            vulkan: IdentityHub::new(Backend::Vulkan),
            #[cfg(any(target_os = "ios", target_os = "macos"))]
            metal: IdentityHub::new(Backend::Metal),
            #[cfg(windows)]
            dx12: IdentityHub::new(Backend::Dx12),
        }
    }

    fn select(&mut self, backend: Backend) -> &mut IdentityHub {
        match backend {
            Backend::Vulkan => &mut self.vulkan,
            #[cfg(any(target_os = "ios", target_os = "macos"))]
            Backend::Metal => &mut self.metal,
            #[cfg(windows)]
            Backend::Dx12 => &mut self.dx12,
            _ => panic!("Unexpected backend: {:?}", backend),
        }
    }
}

#[derive(Debug)]
pub struct Client {
    identities: Mutex<Identities>,
}

#[repr(C)]
#[derive(Debug)]
pub struct Infrastructure {
    pub client: *mut Client,
    pub error: *const u8,
}

#[no_mangle]
pub extern "C" fn wgpu_client_new() -> Infrastructure {
    log::info!("Initializing WGPU client");
    let client = Box::new(Client {
        identities: Mutex::new(Identities::new()),
    });
    Infrastructure {
        client: Box::into_raw(client),
        error: ptr::null(),
    }
}

#[no_mangle]
pub extern "C" fn wgpu_client_delete(client: *mut Client) {
    log::info!("Terminating WGPU client");
    let _client = unsafe { Box::from_raw(client) };
}

#[no_mangle]
pub extern "C" fn wgpu_client_make_adapter_ids(
    client: &Client,
    ids: *mut wgn::AdapterId,
    id_length: usize,
) -> usize {
    let mut identities = client.identities.lock();
    assert_ne!(id_length, 0);
    let mut ids = unsafe { slice::from_raw_parts_mut(ids, id_length) }.iter_mut();

    *ids.next().unwrap() = identities.vulkan.adapters.alloc();

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    {
        *ids.next().unwrap() = identities.metal.adapters.alloc();
    }
    #[cfg(windows)]
    {
        *ids.next().unwrap() = identities.dx12.adapters.alloc();
    }

    id_length - ids.len()
}

#[no_mangle]
pub extern "C" fn wgpu_client_kill_adapter_ids(
    client: &Client,
    ids: *const wgn::AdapterId,
    id_length: usize,
) {
    let mut identity = client.identities.lock();
    let ids = unsafe { slice::from_raw_parts(ids, id_length) };
    for &id in ids {
        identity.select(id.backend()).adapters.free(id)
    }
}

#[no_mangle]
pub extern "C" fn wgpu_client_make_device_id(
    client: &Client,
    adapter_id: wgn::AdapterId,
) -> wgn::DeviceId {
    client
        .identities
        .lock()
        .select(adapter_id.backend())
        .devices
        .alloc()
}

#[no_mangle]
pub extern "C" fn wgpu_client_kill_device_id(client: &Client, id: wgn::DeviceId) {
    client
        .identities
        .lock()
        .select(id.backend())
        .devices
        .free(id)
}
