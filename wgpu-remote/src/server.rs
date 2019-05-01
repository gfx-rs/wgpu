use crate::{GlobalMessage, InstanceMessage};

use ipc_channel::ipc::IpcReceiver;
use wgn;


pub struct Server {
    channel: IpcReceiver<GlobalMessage>,
}

impl Server {
    pub(crate) fn new(channel: IpcReceiver<GlobalMessage>) -> Self {
        Server {
            channel,
        }
    }
}

enum ControlFlow {
    Continue,
    Terminate,
}

fn process(message: GlobalMessage) -> ControlFlow {
    match message {
        GlobalMessage::Instance(msg) => match msg {
            InstanceMessage::Create(instance_id) => {
                let instance = wgn::create_instance();
                wgn::HUB.instances.register(instance_id, instance);
            }
            InstanceMessage::InstanceGetAdapter(instance_id, ref desc, id) => {
                let adapter = wgn::instance_get_adapter(instance_id, desc);
                wgn::HUB.adapters.register(id, adapter);
            }
            InstanceMessage::AdapterCreateDevice(adapter_id, ref desc, id) => {
                let device = wgn::adapter_create_device(adapter_id, desc);
                wgn::HUB.devices.register(id, device);
            }
            InstanceMessage::Destroy(instance_id) => {
                wgn::HUB.instances.unregister(instance_id);
            }
        },
        GlobalMessage::Terminate => return ControlFlow::Terminate,
    }

    ControlFlow::Continue
}


#[no_mangle]
pub extern "C" fn wgpu_server_process(server: &Server) {
    while let Ok(message) = server.channel.try_recv() {
        match process(message) {
            ControlFlow::Continue => {},
            ControlFlow::Terminate => break,
        }
    }
}
