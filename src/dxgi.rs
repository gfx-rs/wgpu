use crate::{com::WeakPtr, CommandQueue, D3DResult, Resource, SampleDesc, HRESULT};
use std::ptr;
use winapi::{
    shared::{dxgi, dxgi1_2, dxgi1_3, dxgi1_4, dxgiformat, dxgitype, windef::HWND},
    um::{d3d12, dxgidebug},
    Interface,
};

bitflags! {
    pub struct FactoryCreationFlags: u32 {
        const DEBUG = dxgi1_3::DXGI_CREATE_FACTORY_DEBUG;
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone)]
pub enum Scaling {
    Stretch = dxgi1_2::DXGI_SCALING_STRETCH,
    Identity = dxgi1_2::DXGI_SCALING_NONE,
    Aspect = dxgi1_2::DXGI_SCALING_ASPECT_RATIO_STRETCH,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone)]
pub enum SwapEffect {
    Discard = dxgi::DXGI_SWAP_EFFECT_DISCARD,
    Sequential = dxgi::DXGI_SWAP_EFFECT_SEQUENTIAL,
    FlipDiscard = dxgi::DXGI_SWAP_EFFECT_FLIP_DISCARD,
    FlipSequential = dxgi::DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone)]
pub enum AlphaMode {
    Unspecified = dxgi1_2::DXGI_ALPHA_MODE_UNSPECIFIED,
    Premultiplied = dxgi1_2::DXGI_ALPHA_MODE_PREMULTIPLIED,
    Straight = dxgi1_2::DXGI_ALPHA_MODE_STRAIGHT,
    Ignore = dxgi1_2::DXGI_ALPHA_MODE_IGNORE,
    ForceDword = dxgi1_2::DXGI_ALPHA_MODE_FORCE_DWORD,
}

pub type Adapter1 = WeakPtr<dxgi::IDXGIAdapter1>;
pub type Factory2 = WeakPtr<dxgi1_2::IDXGIFactory2>;
pub type Factory4 = WeakPtr<dxgi1_4::IDXGIFactory4>;
pub type InfoQueue = WeakPtr<dxgidebug::IDXGIInfoQueue>;
pub type SwapChain = WeakPtr<dxgi::IDXGISwapChain>;
pub type SwapChain1 = WeakPtr<dxgi1_2::IDXGISwapChain1>;
pub type SwapChain3 = WeakPtr<dxgi1_4::IDXGISwapChain3>;

#[cfg(feature = "libloading")]
#[derive(Debug)]
pub struct DxgiLib {
    lib: libloading::Library,
}

#[cfg(feature = "libloading")]
impl DxgiLib {
    pub fn new() -> Result<Self, libloading::Error> {
        unsafe { libloading::Library::new("dxgi.dll").map(|lib| DxgiLib { lib }) }
    }

    pub fn create_factory2(
        &self,
        flags: FactoryCreationFlags,
    ) -> Result<D3DResult<Factory4>, libloading::Error> {
        type Fun = extern "system" fn(
            winapi::shared::minwindef::UINT,
            winapi::shared::guiddef::REFIID,
            *mut *mut winapi::ctypes::c_void,
        ) -> HRESULT;

        let mut factory = Factory4::null();
        let hr = unsafe {
            let func: libloading::Symbol<Fun> = self.lib.get(b"CreateDXGIFactory2")?;
            func(
                flags.bits(),
                &dxgi1_4::IDXGIFactory4::uuidof(),
                factory.mut_void(),
            )
        };

        Ok((factory, hr))
    }

    pub fn get_debug_interface1(&self) -> Result<D3DResult<InfoQueue>, libloading::Error> {
        type Fun = extern "system" fn(
            winapi::shared::minwindef::UINT,
            winapi::shared::guiddef::REFIID,
            *mut *mut winapi::ctypes::c_void,
        ) -> HRESULT;

        let mut queue = InfoQueue::null();
        let hr = unsafe {
            let func: libloading::Symbol<Fun> = self.lib.get(b"DXGIGetDebugInterface1")?;
            func(0, &dxgidebug::IDXGIInfoQueue::uuidof(), queue.mut_void())
        };
        Ok((queue, hr))
    }
}

// TODO: strong types
pub struct SwapchainDesc {
    pub width: u32,
    pub height: u32,
    pub format: dxgiformat::DXGI_FORMAT,
    pub stereo: bool,
    pub sample: SampleDesc,
    pub buffer_usage: dxgitype::DXGI_USAGE,
    pub buffer_count: u32,
    pub scaling: Scaling,
    pub swap_effect: SwapEffect,
    pub alpha_mode: AlphaMode,
    pub flags: u32,
}

impl Factory2 {
    // TODO: interface not complete
    pub fn create_swapchain_for_hwnd(
        &self,
        queue: CommandQueue,
        hwnd: HWND,
        desc: &SwapchainDesc,
    ) -> D3DResult<SwapChain1> {
        let desc = dxgi1_2::DXGI_SWAP_CHAIN_DESC1 {
            AlphaMode: desc.alpha_mode as _,
            BufferCount: desc.buffer_count,
            Width: desc.width,
            Height: desc.height,
            Format: desc.format,
            Flags: desc.flags,
            BufferUsage: desc.buffer_usage,
            SampleDesc: dxgitype::DXGI_SAMPLE_DESC {
                Count: desc.sample.count,
                Quality: desc.sample.quality,
            },
            Scaling: desc.scaling as _,
            Stereo: desc.stereo as _,
            SwapEffect: desc.swap_effect as _,
        };

        let mut swap_chain = SwapChain1::null();
        let hr = unsafe {
            self.CreateSwapChainForHwnd(
                queue.as_mut_ptr() as *mut _,
                hwnd,
                &desc,
                ptr::null(),
                ptr::null_mut(),
                swap_chain.mut_void() as *mut *mut _,
            )
        };

        (swap_chain, hr)
    }
}

impl Factory4 {
    #[cfg(feature = "implicit-link")]
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

    pub fn as_factory2(&self) -> Factory2 {
        unsafe { Factory2::from_raw(self.as_mut_ptr() as *mut _) }
    }

    pub fn enumerate_adapters(&self, id: u32) -> D3DResult<Adapter1> {
        let mut adapter = Adapter1::null();
        let hr = unsafe { self.EnumAdapters1(id, adapter.mut_void() as *mut *mut _) };

        (adapter, hr)
    }
}

bitflags! {
    pub struct SwapChainPresentFlags: u32 {
        const DXGI_PRESENT_DO_NOT_SEQUENCE = dxgi::DXGI_PRESENT_DO_NOT_SEQUENCE;
        const DXGI_PRESENT_TEST = dxgi::DXGI_PRESENT_TEST;
        const DXGI_PRESENT_RESTART = dxgi::DXGI_PRESENT_RESTART;
        const DXGI_PRESENT_DO_NOT_WAIT = dxgi::DXGI_PRESENT_DO_NOT_WAIT;
        const DXGI_PRESENT_RESTRICT_TO_OUTPUT = dxgi::DXGI_PRESENT_RESTRICT_TO_OUTPUT;
        const DXGI_PRESENT_STEREO_PREFER_RIGHT = dxgi::DXGI_PRESENT_STEREO_PREFER_RIGHT;
        const DXGI_PRESENT_STEREO_TEMPORARY_MONO = dxgi::DXGI_PRESENT_STEREO_TEMPORARY_MONO;
        const DXGI_PRESENT_USE_DURATION = dxgi::DXGI_PRESENT_USE_DURATION;
        const DXGI_PRESENT_ALLOW_TEARING = dxgi::DXGI_PRESENT_ALLOW_TEARING;
    }
}

impl SwapChain {
    pub fn get_buffer(&self, id: u32) -> D3DResult<Resource> {
        let mut resource = Resource::null();
        let hr =
            unsafe { self.GetBuffer(id, &d3d12::ID3D12Resource::uuidof(), resource.mut_void()) };

        (resource, hr)
    }

    //TODO: replace by present_flags
    pub fn present(&self, interval: u32, flags: u32) -> HRESULT {
        unsafe { self.Present(interval, flags) }
    }

    pub fn present_flags(&self, interval: u32, flags: SwapChainPresentFlags) -> HRESULT {
        unsafe { self.Present(interval, flags.bits()) }
    }
}

impl SwapChain1 {
    pub fn as_swapchain0(&self) -> SwapChain {
        unsafe { SwapChain::from_raw(self.as_mut_ptr() as *mut _) }
    }
}

impl SwapChain3 {
    pub fn as_swapchain0(&self) -> SwapChain {
        unsafe { SwapChain::from_raw(self.as_mut_ptr() as *mut _) }
    }

    pub fn get_current_back_buffer_index(&self) -> u32 {
        unsafe { self.GetCurrentBackBufferIndex() }
    }
}
