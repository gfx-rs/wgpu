/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//TODO: remove once `cbindgen` is smart enough
extern crate wgpu_native as wgn;
use wgn::{AdapterId, Backend, DeviceId, IdentityManager, SurfaceId};

use crate::server::Server;

use ipc_channel::ipc;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use std::{ptr, sync::Arc};

mod server;

/// A message on the timeline of devices, queues, and resources.
#[derive(Serialize, Deserialize, Debug)]
enum GlobalMessage {
    RequestAdapter(wgn::RequestAdapterOptions, Vec<wgn::AdapterId>),
    AdapterRequestDevice(wgn::AdapterId, wgn::DeviceDescriptor, wgn::DeviceId),
    //Device(DeviceMessage),
    //Queue(QueueMessage),
    //Texture(TextureMessage),
    //Command(CommandMessage),
    Terminate,
}

#[derive(Debug)]
struct IdentityHub {
    adapters: IdentityManager<AdapterId>,
    devices: IdentityManager<DeviceId>,
    /*
    pipeline_layouts: IdentityManager<PipelineLayoutId>,
    shader_modules: IdentityManager<ShaderModuleId>,
    bind_group_layouts: IdentityManager<BindGroupLayoutId>,
    bind_groups: IdentityManager<BindGroupId>,
    command_buffers: IdentityManager<CommandBufferId>,
    render_passes: IdentityManager<RenderPassId>,
    render_pipelines: IdentityManager<RenderPipelineId>,
    compute_passes: IdentityManager<ComputePassId>,
    compute_pipelines: IdentityManager<ComputePipelineId>,
    buffers: IdentityManager<BufferId>,
    textures: IdentityManager<TextureId>,
    texture_views: IdentityManager<TextureViewId>,
    samplers: IdentityManager<SamplerId>,
    */
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


#[derive(Clone, Debug)]
pub struct Client {
    channel: ipc::IpcSender<GlobalMessage>,
    identities: Arc<Mutex<Identities>>,
}

#[repr(C)]
#[derive(Debug)]
pub struct Infrastructure {
    pub client: *mut Client,
    pub server: *mut Server,
    pub error: *const u8,
}

#[no_mangle]
pub extern "C" fn wgpu_initialize() -> Infrastructure {
    match ipc::channel() {
        Ok((sender, receiver)) => {
            let client = Client {
                channel: sender,
                identities: Arc::new(Mutex::new(Identities::new())),
            };
            let server = Server::new(receiver);
            Infrastructure {
                client: Box::into_raw(Box::new(client)),
                server: Box::into_raw(Box::new(server)),
                error: ptr::null(),
            }
        }
        Err(e) => {
            log::error!("WGPU initialize failed: {:?}", e);
            Infrastructure {
                client: ptr::null_mut(),
                server: ptr::null_mut(),
                error: ptr::null(), //TODO_remote_
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_terminate(client: *mut Client) {
    let client = unsafe { Box::from_raw(client) };
    let msg = GlobalMessage::Terminate;
    let _ = client.channel.send(msg);
}

#[no_mangle]
pub extern "C" fn wgpu_client_request_adapter(
    client: &Client,
    desc: &wgn::RequestAdapterOptions,
) -> wgn::AdapterId {
    let mut identities = client.identities.lock();
    let ids = vec![
        identities.vulkan.adapters.alloc(),
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        identities.metal.adapters.alloc(),
        #[cfg(windows)]
        identities.dx12.adapters.alloc(),
    ];
    let msg = GlobalMessage::RequestAdapter(desc.clone(), ids);
    client.channel.send(msg).unwrap();
    unimplemented!()
}

#[no_mangle]
pub extern "C" fn wgpu_client_adapter_create_device(
    client: &Client,
    adapter_id: wgn::AdapterId,
    desc: &wgn::DeviceDescriptor,
) -> wgn::DeviceId {
    let device_id = client
        .identities
        .lock()
        .select(adapter_id.backend())
        .devices
        .alloc();
    let msg = GlobalMessage::AdapterRequestDevice(adapter_id, desc.clone(), device_id);
    client.channel.send(msg).unwrap();
    device_id
}
