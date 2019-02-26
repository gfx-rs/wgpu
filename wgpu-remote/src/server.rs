use crate::{GlobalMessage, InstanceMessage}

use ipc_channel::ipc::IpcReceiver;

use wgn;


struct Server {
    channel: IpcReceiver<GlobalMessage>,
    instance_id: wgn::IntanceId,
}

impl Server {
    pub fn new(channel: IpcReceiver<GlobalMessage>) -> Self {
        Server {
            channel,
            instance_id: wgn::wgpu_create_instance(),
        }
    }
}

pub fn process(message: GlobalMessage) {
    match message {
        GlobalMessage::Instance(msg) => match msg {
            InstanceMessage::InstanceGetAdapter(instance_id, ref desc, id) => {
                let adapter = wgn::instance_get_adapter(instance_id, desc);
                wgn::HUB.adapters.register(id, adapter);
            }
            InstanceMessage::AdapterCreateDevice(adapter_id, ref desc, id) => {
                let device = wgn::adapter_create_device(adapter_id, desc);
                wgn::HUB.devices.register(id, device);
            }
        },
    }
}
