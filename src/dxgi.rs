use com::WeakPtr;
use winapi::shared::{dxgi, dxgi1_3, dxgi1_4};
use winapi::Interface;
use D3DResult;

bitflags! {
    pub struct FactoryCreationFlags: u32 {
        const DEBUG = dxgi1_3::DXGI_CREATE_FACTORY_DEBUG;
    }
}

pub type Adapter1 = WeakPtr<dxgi::IDXGIAdapter1>;
pub type Factory4 = WeakPtr<dxgi1_4::IDXGIFactory4>;

impl Factory4 {
    pub fn create(flags: FactoryCreationFlags) -> D3DResult<Self> {
        let mut factory = Factory4::null();
        let hr = unsafe {
            dxgi1_3::CreateDXGIFactory2(
                flags.bits(),
                &dxgi1_4::IDXGIFactory4::uuidof(),
                factory.mut_void(),
            )
        };

        (factory, hr)
    }

    pub fn enumerate_adapters(&self, id: u32) -> D3DResult<Adapter1> {
        let mut adapter = Adapter1::null();
        let hr = unsafe { self.EnumAdapters1(id, adapter.mut_void() as *mut *mut _) };

        (adapter, hr)
    }
}
