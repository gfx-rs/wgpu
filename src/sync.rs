use com::WeakPtr;
use winapi::um::d3d12;
use winapi::um::winnt;
use HRESULT;

pub type Event = winnt::HANDLE;
pub type Fence = WeakPtr<d3d12::ID3D12Fence>;

impl Fence {
    pub fn set_event_on_completion(&self, event: Event, value: u64) -> HRESULT {
        unsafe { self.SetEventOnCompletion(value, event) }
    }

    pub fn get_value(&self) -> u64 {
        unsafe { self.GetCompletedValue() }
    }

    pub fn signal(&self, value: u64) -> HRESULT {
        unsafe { self.Signal(value) }
    }
}
