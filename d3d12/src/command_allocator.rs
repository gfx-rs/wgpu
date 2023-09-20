//! Command Allocator

use crate::com::ComPtr;
use winapi::um::d3d12;

pub type CommandAllocator = ComPtr<d3d12::ID3D12CommandAllocator>;

impl CommandAllocator {
    pub fn reset(&self) {
        unsafe {
            self.Reset();
        }
    }
}
