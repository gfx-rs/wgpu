use crate::{com::ComPtr, HRESULT};
use std::ptr;
use winapi::um::{d3d12, synchapi, winnt};

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Event(pub winnt::HANDLE);
impl Event {
    pub fn create(manual_reset: bool, initial_state: bool) -> Self {
        Event(unsafe {
            synchapi::CreateEventA(
                ptr::null_mut(),
                manual_reset as _,
                initial_state as _,
                ptr::null(),
            )
        })
    }

    // TODO: return value
    pub fn wait(&self, timeout_ms: u32) -> u32 {
        unsafe { synchapi::WaitForSingleObject(self.0, timeout_ms) }
    }
}

pub type Fence = ComPtr<d3d12::ID3D12Fence>;
impl Fence {
    pub fn set_event_on_completion(&self, event: Event, value: u64) -> HRESULT {
        unsafe { self.SetEventOnCompletion(value, event.0) }
    }

    pub fn get_value(&self) -> u64 {
        unsafe { self.GetCompletedValue() }
    }

    pub fn signal(&self, value: u64) -> HRESULT {
        unsafe { self.Signal(value) }
    }
}
