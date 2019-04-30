use crate::server::Server;

use ipc_channel::ipc;
use lazy_static::lazy_static;
use log::error;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use wgn;

use std::ptr;

mod server;


lazy_static! {
    static ref INSTANCE_IDENTITIES: Mutex<wgn::IdentityManager> = Mutex::new(wgn::IdentityManager::default());
}

#[derive(Serialize, Deserialize)]
enum InstanceMessage {
    InstanceGetAdapter(wgn::InstanceId, wgn::AdapterDescriptor, wgn::AdapterId),
    AdapterCreateDevice(wgn::AdapterId, wgn::DeviceDescriptor, wgn::DeviceId),
    Terminate,
}

/// A message on the timeline of devices, queues, and resources.
#[derive(Serialize, Deserialize)]
enum GlobalMessage {
    Instance(InstanceMessage),
    //Device(DeviceMessage),
    //Queue(QueueMessage),
    //Texture(TextureMessage),
    //Command(CommandMessage),
}

#[derive(Default)]
struct IdentityHub {
    adapters: wgn::IdentityManager,
    devices: wgn::IdentityManager,
}

pub struct Client {
    channel: ipc::IpcSender<GlobalMessage>,
    instance_id: wgn::InstanceId,
    identity: Mutex<IdentityHub>,
}

impl Client {
    fn new(
        channel: ipc::IpcSender<GlobalMessage>,
        instance_id: wgn::InstanceId,
    ) -> Self {
        Client {
            channel,
            instance_id,
            identity: Mutex::new(IdentityHub::default()),
        }
    }
}

#[repr(C)]
pub struct Infrastructure {
    pub client: *mut Client,
    pub server: *mut Server,
    pub error: *const u8,
}

#[no_mangle]
pub extern "C" fn wgpu_initialize() -> Infrastructure {
    match ipc::channel() {
        Ok((sender, receiver)) => {
            let instance_id = INSTANCE_IDENTITIES.lock().alloc();
            let client = Client::new(sender, instance_id);
            let server = Server::new(receiver, instance_id);
            Infrastructure {
                client: Box::into_raw(Box::new(client)),
                server: Box::into_raw(Box::new(server)),
                error: ptr::null(),
            }
        }
        Err(e) => {
            error!("WGPU initialize failed: {:?}", e);
            Infrastructure {
                client: ptr::null_mut(),
                server: ptr::null_mut(),
                error: ptr::null(), //TODO
            }
        }
    }

}

#[no_mangle]
pub extern "C" fn wgpu_terminate(client: *mut Client) {
    let client = unsafe {
        Box::from_raw(client)
    };
    INSTANCE_IDENTITIES.lock().free(client.instance_id);
    let _ = client.channel.send(GlobalMessage::Instance(InstanceMessage::Terminate));
}

#[no_mangle]
pub extern "C" fn wgpu_instance_get_adapter(
    client: &Client,
    desc: &wgn::AdapterDescriptor,
) -> wgn::AdapterId {
    let id = client.identity.lock().adapters.alloc();
    let msg = GlobalMessage::Instance(InstanceMessage::InstanceGetAdapter(
        client.instance_id,
        desc.clone(),
        id,
    ));
    client.channel.send(msg).unwrap();
    id
}

#[no_mangle]
pub extern "C" fn wgpu_adapter_create_device(
    client: &Client,
    adapter_id: wgn::AdapterId,
    desc: &wgn::DeviceDescriptor,
) -> wgn::DeviceId {
    let id = client.identity.lock().devices.alloc();
    let msg = GlobalMessage::Instance(InstanceMessage::AdapterCreateDevice(
        adapter_id,
        desc.clone(),
        id,
    ));
    client.channel.send(msg).unwrap();
    id
}
