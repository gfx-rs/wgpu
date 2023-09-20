use crate::{com::ComPtr, sync::Fence, CommandList, HRESULT};
use winapi::um::d3d12;

#[repr(u32)]
pub enum Priority {
    Normal = d3d12::D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
    High = d3d12::D3D12_COMMAND_QUEUE_PRIORITY_HIGH,
    GlobalRealtime = d3d12::D3D12_COMMAND_QUEUE_PRIORITY_GLOBAL_REALTIME,
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct CommandQueueFlags: u32 {
        const DISABLE_GPU_TIMEOUT = d3d12::D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;
    }
}

pub type CommandQueue = ComPtr<d3d12::ID3D12CommandQueue>;

impl CommandQueue {
    pub fn execute_command_lists(&self, command_lists: &[CommandList]) {
        let command_lists = command_lists
            .iter()
            .map(CommandList::as_mut_ptr)
            .collect::<Box<[_]>>();
        unsafe { self.ExecuteCommandLists(command_lists.len() as _, command_lists.as_ptr()) }
    }

    pub fn signal(&self, fence: &Fence, value: u64) -> HRESULT {
        unsafe { self.Signal(fence.as_mut_ptr(), value) }
    }
}
