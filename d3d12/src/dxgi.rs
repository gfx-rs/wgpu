use crate::{com::ComPtr, D3DResult, Resource, SampleDesc, HRESULT};
use std::ptr;
use winapi::{
    shared::{
        dxgi, dxgi1_2, dxgi1_3, dxgi1_4, dxgi1_5, dxgi1_6, dxgiformat, dxgitype, minwindef::TRUE,
        windef::HWND,
    },
    um::{d3d12, dxgidebug, unknwnbase::IUnknown, winnt::HANDLE},
    Interface,
};

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
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

pub type InfoQueue = ComPtr<dxgidebug::IDXGIInfoQueue>;

pub type Adapter1 = ComPtr<dxgi::IDXGIAdapter1>;
pub type Adapter2 = ComPtr<dxgi1_2::IDXGIAdapter2>;
pub type Adapter3 = ComPtr<dxgi1_4::IDXGIAdapter3>;
pub type Adapter4 = ComPtr<dxgi1_6::IDXGIAdapter4>;
crate::weak_com_inheritance_chain! {
    #[derive(Debug, Clone, PartialEq, Hash)]
    pub enum DxgiAdapter {
        Adapter1(dxgi::IDXGIAdapter1), from_adapter1, as_adapter1, adapter1;
        Adapter2(dxgi1_2::IDXGIAdapter2), from_adapter2, as_adapter2, unwrap_adapter2;
        Adapter3(dxgi1_4::IDXGIAdapter3), from_adapter3, as_adapter3, unwrap_adapter3;
        Adapter4(dxgi1_6::IDXGIAdapter4), from_adapter4, as_adapter4, unwrap_adapter4;
    }
}

pub type Factory1 = ComPtr<dxgi::IDXGIFactory1>;
pub type Factory2 = ComPtr<dxgi1_2::IDXGIFactory2>;
pub type Factory3 = ComPtr<dxgi1_3::IDXGIFactory3>;
pub type Factory4 = ComPtr<dxgi1_4::IDXGIFactory4>;
pub type Factory5 = ComPtr<dxgi1_5::IDXGIFactory5>;
pub type Factory6 = ComPtr<dxgi1_6::IDXGIFactory6>;
crate::weak_com_inheritance_chain! {
    #[derive(Debug, Clone, PartialEq, Hash)]
    pub enum DxgiFactory {
        Factory1(dxgi::IDXGIFactory1), from_factory1, as_factory1, factory1;
        Factory2(dxgi1_2::IDXGIFactory2), from_factory2, as_factory2, unwrap_factory2;
        Factory3(dxgi1_3::IDXGIFactory3), from_factory3, as_factory3, unwrap_factory3;
        Factory4(dxgi1_4::IDXGIFactory4), from_factory4, as_factory4, unwrap_factory4;
        Factory5(dxgi1_5::IDXGIFactory5), from_factory5, as_factory5, unwrap_factory5;
        Factory6(dxgi1_6::IDXGIFactory6), from_factory6, as_factory6, unwrap_factory6;
    }
}

pub type FactoryMedia = ComPtr<dxgi1_3::IDXGIFactoryMedia>;

pub type SwapChain = ComPtr<dxgi::IDXGISwapChain>;
pub type SwapChain1 = ComPtr<dxgi1_2::IDXGISwapChain1>;
pub type SwapChain2 = ComPtr<dxgi1_3::IDXGISwapChain2>;
pub type SwapChain3 = ComPtr<dxgi1_4::IDXGISwapChain3>;
crate::weak_com_inheritance_chain! {
    #[derive(Debug, Clone, PartialEq, Hash)]
    pub enum DxgiSwapchain {
        SwapChain(dxgi::IDXGISwapChain), from_swap_chain, as_swap_chain, swap_chain;
        SwapChain1(dxgi1_2::IDXGISwapChain1), from_swap_chain1, as_swap_chain1, unwrap_swap_chain1;
        SwapChain2(dxgi1_3::IDXGISwapChain2), from_swap_chain2, as_swap_chain2, unwrap_swap_chain2;
        SwapChain3(dxgi1_4::IDXGISwapChain3), from_swap_chain3, as_swap_chain3, unwrap_swap_chain3;
    }
}

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

    pub fn create_factory1(&self) -> Result<D3DResult<Factory1>, libloading::Error> {
        type Fun = extern "system" fn(
            winapi::shared::guiddef::REFIID,
            *mut *mut winapi::ctypes::c_void,
        ) -> HRESULT;

        let mut factory = Factory1::null();
        let hr = unsafe {
            let func: libloading::Symbol<Fun> = self.lib.get(b"CreateDXGIFactory1")?;
            func(&dxgi::IDXGIFactory1::uuidof(), factory.mut_void())
        };

        Ok((factory, hr))
    }

    pub fn create_factory_media(&self) -> Result<D3DResult<FactoryMedia>, libloading::Error> {
        type Fun = extern "system" fn(
            winapi::shared::guiddef::REFIID,
            *mut *mut winapi::ctypes::c_void,
        ) -> HRESULT;

        let mut factory = FactoryMedia::null();
        let hr = unsafe {
            // https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_3/nn-dxgi1_3-idxgifactorymedia
            let func: libloading::Symbol<Fun> = self.lib.get(b"CreateDXGIFactory1")?;
            func(&dxgi1_3::IDXGIFactoryMedia::uuidof(), factory.mut_void())
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
impl SwapchainDesc {
    pub fn to_desc1(&self) -> dxgi1_2::DXGI_SWAP_CHAIN_DESC1 {
        dxgi1_2::DXGI_SWAP_CHAIN_DESC1 {
            AlphaMode: self.alpha_mode as _,
            BufferCount: self.buffer_count,
            Width: self.width,
            Height: self.height,
            Format: self.format,
            Flags: self.flags,
            BufferUsage: self.buffer_usage,
            SampleDesc: dxgitype::DXGI_SAMPLE_DESC {
                Count: self.sample.count,
                Quality: self.sample.quality,
            },
            Scaling: self.scaling as _,
            Stereo: self.stereo as _,
            SwapEffect: self.swap_effect as _,
        }
    }
}

impl Factory1 {
    pub fn create_swapchain(
        &self,
        queue: *mut IUnknown,
        hwnd: HWND,
        desc: &SwapchainDesc,
    ) -> D3DResult<SwapChain> {
        let mut desc = dxgi::DXGI_SWAP_CHAIN_DESC {
            BufferDesc: dxgitype::DXGI_MODE_DESC {
                Width: desc.width,
                Height: desc.width,
                RefreshRate: dxgitype::DXGI_RATIONAL {
                    Numerator: 1,
                    Denominator: 60,
                },
                Format: desc.format,
                ScanlineOrdering: dxgitype::DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED,
                Scaling: dxgitype::DXGI_MODE_SCALING_UNSPECIFIED,
            },
            SampleDesc: dxgitype::DXGI_SAMPLE_DESC {
                Count: desc.sample.count,
                Quality: desc.sample.quality,
            },
            BufferUsage: desc.buffer_usage,
            BufferCount: desc.buffer_count,
            OutputWindow: hwnd,
            Windowed: TRUE,
            SwapEffect: desc.swap_effect as _,
            Flags: desc.flags,
        };

        let mut swapchain = SwapChain::null();
        let hr =
            unsafe { self.CreateSwapChain(queue, &mut desc, swapchain.mut_void() as *mut *mut _) };

        (swapchain, hr)
    }
}

impl Factory2 {
    // TODO: interface not complete
    pub fn create_swapchain_for_hwnd(
        &self,
        queue: *mut IUnknown,
        hwnd: HWND,
        desc: &SwapchainDesc,
    ) -> D3DResult<SwapChain1> {
        let mut swap_chain = SwapChain1::null();
        let hr = unsafe {
            self.CreateSwapChainForHwnd(
                queue,
                hwnd,
                &desc.to_desc1(),
                ptr::null(),
                ptr::null_mut(),
                swap_chain.mut_void() as *mut *mut _,
            )
        };

        (swap_chain, hr)
    }

    pub fn create_swapchain_for_composition(
        &self,
        queue: *mut IUnknown,
        desc: &SwapchainDesc,
    ) -> D3DResult<SwapChain1> {
        let mut swap_chain = SwapChain1::null();
        let hr = unsafe {
            self.CreateSwapChainForComposition(
                queue,
                &desc.to_desc1(),
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

    pub fn enumerate_adapters(&self, id: u32) -> D3DResult<Adapter1> {
        let mut adapter = Adapter1::null();
        let hr = unsafe { self.EnumAdapters1(id, adapter.mut_void() as *mut *mut _) };

        (adapter, hr)
    }
}

impl FactoryMedia {
    pub fn create_swapchain_for_composition_surface_handle(
        &self,
        queue: *mut IUnknown,
        surface_handle: HANDLE,
        desc: &SwapchainDesc,
    ) -> D3DResult<SwapChain1> {
        let mut swap_chain = SwapChain1::null();
        let hr = unsafe {
            self.CreateSwapChainForCompositionSurfaceHandle(
                queue,
                surface_handle,
                &desc.to_desc1(),
                ptr::null_mut(),
                swap_chain.mut_void() as *mut *mut _,
            )
        };

        (swap_chain, hr)
    }
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
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

impl SwapChain3 {
    pub fn get_current_back_buffer_index(&self) -> u32 {
        unsafe { self.GetCurrentBackBufferIndex() }
    }
}
