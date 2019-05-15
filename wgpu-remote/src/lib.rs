//TODO: remove once `cbindgen` is smart enough
extern crate wgpu_native as wgn;
use wgn::{AdapterId, DeviceId, InstanceId};

use crate::server::Server;

use ipc_channel::ipc;
use log::error;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use std::ptr;

mod server;

#[derive(Serialize, Deserialize)]
enum InstanceMessage {
    Create(wgn::InstanceId),
    InstanceGetAdapter(wgn::InstanceId, wgn::AdapterDescriptor, wgn::AdapterId),
    AdapterCreateDevice(wgn::AdapterId, wgn::DeviceDescriptor, wgn::DeviceId),
    Destroy(wgn::InstanceId),
}

/// A message on the timeline of devices, queues, and resources.
#[derive(Serialize, Deserialize)]
enum GlobalMessage {
    Instance(InstanceMessage),
    //Device(DeviceMessage),
    //Queue(QueueMessage),
    //Texture(TextureMessage),
    //Command(CommandMessage),
    Terminate,
}

#[derive(Default)]
struct IdentityHub {
    adapters: wgn::IdentityManager<AdapterId>,
    devices: wgn::IdentityManager<DeviceId>,
}

pub struct Client {
    channel: ipc::IpcSender<GlobalMessage>,
    instance_id: wgn::InstanceId,
    identity: Mutex<IdentityHub>,
}

pub struct ClientFactory {
    channel: ipc::IpcSender<GlobalMessage>,
    instance_identities: Mutex<wgn::IdentityManager<InstanceId>>,
}

#[repr(C)]
pub struct Infrastructure {
    pub factory: *mut ClientFactory,
    pub server: *mut Server,
    pub error: *const u8,
}

#[no_mangle]
pub extern "C" fn wgpu_initialize() -> Infrastructure {
    match ipc::channel() {
        Ok((sender, receiver)) => {
            let factory = ClientFactory {
                channel: sender,
                instance_identities: Mutex::new(wgn::IdentityManager::default()),
            };
            let server = Server::new(receiver);
            Infrastructure {
                factory: Box::into_raw(Box::new(factory)),
                server: Box::into_raw(Box::new(server)),
                error: ptr::null(),
            }
        }
        Err(e) => {
            error!("WGPU initialize failed: {:?}", e);
            Infrastructure {
                factory: ptr::null_mut(),
                server: ptr::null_mut(),
                error: ptr::null(), //TODO_remote_
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_terminate(factory: *mut ClientFactory) {
    let factory = unsafe { Box::from_raw(factory) };
    let _ = factory.channel.send(GlobalMessage::Terminate);
}

#[no_mangle]
pub extern "C" fn wgpu_client_create(factory: &ClientFactory) -> *mut Client {
    let instance_id = factory.instance_identities.lock().alloc();
    let msg = GlobalMessage::Instance(InstanceMessage::Create(instance_id));
    factory.channel.send(msg).unwrap();
    let client = Client {
        channel: factory.channel.clone(),
        instance_id,
        identity: Mutex::new(IdentityHub::default()),
    };
    Box::into_raw(Box::new(client))
}

#[no_mangle]
pub extern "C" fn wgpu_client_destroy(factory: &ClientFactory, client: *mut Client) {
    let client = unsafe { Box::from_raw(client) };
    factory.instance_identities.lock().free(client.instance_id);
    let msg = GlobalMessage::Instance(InstanceMessage::Destroy(client.instance_id));
    client.channel.send(msg).unwrap();
}

#[no_mangle]
pub extern "C" fn wgpu_client_get_adapter(
    client: &Client,
    desc: &wgn::AdapterDescriptor,
) -> wgn::AdapterId {
    let adapter_id = client.identity.lock().adapters.alloc();
    let msg = GlobalMessage::Instance(InstanceMessage::InstanceGetAdapter(
        client.instance_id,
        desc.clone(),
        adapter_id,
    ));
    client.channel.send(msg).unwrap();
    adapter_id
}

#[no_mangle]
pub extern "C" fn wgpu_client_adapter_create_device(
    client: &Client,
    adapter_id: wgn::AdapterId,
    desc: &wgn::DeviceDescriptor,
) -> wgn::DeviceId {
    let device_id = client.identity.lock().devices.alloc();
    let msg = GlobalMessage::Instance(InstanceMessage::AdapterCreateDevice(
        adapter_id,
        desc.clone(),
        device_id,
    ));
    client.channel.send(msg).unwrap();
    device_id
}
