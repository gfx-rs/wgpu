//! Command Allocator

use crate::com::WeakPtr;
use winapi::um::d3d12;

pub type CommandAllocator = WeakPtr<d3d12::ID3D12CommandAllocator>;

impl CommandAllocator {
    pub fn reset(&self) {
        unsafe {
            self.Reset();
        }
    }
}
