use crate::{GlobalMessage};

use ipc_channel::ipc::IpcReceiver;
use wgn;

#[derive(Debug)]
pub struct Server {
    channel: IpcReceiver<GlobalMessage>,
}

impl Server {
    pub(crate) fn new(channel: IpcReceiver<GlobalMessage>) -> Self {
        Server { channel }
    }
}

enum ControlFlow {
    Continue,
    Terminate,
}

fn process(message: GlobalMessage) -> ControlFlow {
    match message {
        GlobalMessage::RequestAdapter(ref desc, ref ids) => {
            wgn::request_adapter(desc, ids);
        }
        GlobalMessage::AdapterRequestDevice(adapter_id, ref desc, id) => {
            use wgn::adapter_request_device as fun;
            wgn::gfx_select!(adapter_id => fun(adapter_id, desc, id));
        }
        GlobalMessage::Terminate => return ControlFlow::Terminate,
    }

    ControlFlow::Continue
}

#[no_mangle]
pub extern "C" fn wgpu_server_process(server: &Server) {
    while let Ok(message) = server.channel.try_recv() {
        match process(message) {
            ControlFlow::Continue => {}
            ControlFlow::Terminate => break,
        }
    }
}
