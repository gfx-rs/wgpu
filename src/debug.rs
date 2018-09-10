use com::WeakPtr;
use winapi::um::{d3d12, d3d12sdklayers};
use winapi::Interface;
use D3DResult;

pub type Debug = WeakPtr<d3d12sdklayers::ID3D12Debug>;

impl Debug {
    pub fn get_interface() -> D3DResult<Self> {
        let mut debug = Debug::null();
        let hr = unsafe {
            d3d12::D3D12GetDebugInterface(&d3d12sdklayers::ID3D12Debug::uuidof(), debug.mut_void())
        };

        (debug, hr)
    }

    pub fn enable_layer(&self) {
        unsafe { self.EnableDebugLayer() }
    }
}
