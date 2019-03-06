use ipc_channel::ipc::IpcSender;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use wgpu_native as wgn;

#[derive(Serialize, Deserialize)]
pub enum InstanceMessage {
    InstanceGetAdapter(wgn::InstanceId, wgn::AdapterDescriptor, wgn::AdapterId),
    AdapterCreateDevice(wgn::AdapterId, wgn::DeviceDescriptor, wgn::DeviceId),
}

/// A message on the timeline of devices, queues, and resources.
#[derive(Serialize, Deserialize)]
pub enum GlobalMessage {
    Instance(InstanceMessage),
    //Device(DeviceMessage),
    //Queue(QueueMessage),
    //Texture(TextureMessage),
    //Command(CommandMessage),
}

#[derive(Default)]
pub struct IdentityHub {
    adapters: wgn::IdentityManager,
    devices: wgn::IdentityManager,
}

pub struct Client {
    channel: IpcSender<GlobalMessage>,
    identity: Mutex<IdentityHub>,
}

impl Client {
    pub fn new(channel: IpcSender<GlobalMessage>) -> Self {
        Client {
            channel,
            identity: Mutex::new(IdentityHub::default()),
        }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_instance_get_adapter(
    client: &Client,
    instance_id: wgn::InstanceId,
    desc: &wgn::AdapterDescriptor,
) -> wgn::AdapterId {
    let id = client.identity.lock().adapters.alloc();
    let msg = GlobalMessage::Instance(InstanceMessage::InstanceGetAdapter(
        instance_id,
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
