use super::result::HResult as _;
use crate::auxil::dyn_lib::DynLib;
use core::ffi;
use windows::core::Interface as _;
use windows::Win32::Graphics::Dxgi;

#[derive(Debug)]
pub struct DxgiLib {
    lib: DynLib,
}

impl DxgiLib {
    pub fn new() -> Result<Self, libloading::Error> {
        unsafe { DynLib::new("dxgi.dll").map(|lib| Self { lib }) }
    }

    /// Will error with crate::DeviceError::Unexpected if DXGI 1.3 is not available.
    #[cfg_attr(not(dx12), expect(dead_code))]
    pub fn debug_interface1(&self) -> Result<Option<Dxgi::IDXGIInfoQueue>, crate::DeviceError> {
        // Calls windows::Win32::Graphics::Dxgi::DXGIGetDebugInterface1 on dxgi.dll
        type Fun = extern "system" fn(
            flags: u32,
            riid: *const windows_core::GUID,
            pdebug: *mut *mut ffi::c_void,
        ) -> windows_core::HRESULT;
        let func: libloading::Symbol<Fun> =
            unsafe { self.lib.get(c"DXGIGetDebugInterface1".to_bytes()) }?;

        let mut result__ = None;

        let res = (func)(0, &Dxgi::IDXGIInfoQueue::IID, <*mut _>::cast(&mut result__)).ok();

        if let Err(ref err) = res {
            match err.code() {
                Dxgi::DXGI_ERROR_SDK_COMPONENT_MISSING => return Ok(None),
                _ => {}
            }
        }

        res.into_device_result("debug_interface1")?;

        result__.ok_or(crate::DeviceError::Unexpected).map(Some)
    }

    /// Will error with crate::DeviceError::Unexpected if DXGI 1.1 is not available.
    pub fn create_factory1(&self) -> Result<Dxgi::IDXGIFactory1, crate::DeviceError> {
        // Calls windows::Win32::Graphics::Dxgi::CreateDXGIFactory1 on dxgi.dll
        type Fun = extern "system" fn(
            riid: *const windows_core::GUID,
            ppfactory: *mut *mut ffi::c_void,
        ) -> windows_core::HRESULT;
        let func: libloading::Symbol<Fun> =
            unsafe { self.lib.get(c"CreateDXGIFactory1".to_bytes()) }?;

        let mut result__ = None;

        (func)(&Dxgi::IDXGIFactory1::IID, <*mut _>::cast(&mut result__))
            .ok()
            .into_device_result("create_factory1")?;

        result__.ok_or(crate::DeviceError::Unexpected)
    }

    /// Will error with crate::DeviceError::Unexpected if DXGI 1.4 is not available.
    #[cfg_attr(not(dx12), expect(dead_code))]
    pub fn create_factory4(
        &self,
        factory_flags: Dxgi::DXGI_CREATE_FACTORY_FLAGS,
    ) -> Result<Dxgi::IDXGIFactory4, crate::DeviceError> {
        // Calls windows::Win32::Graphics::Dxgi::CreateDXGIFactory2 on dxgi.dll
        type Fun = extern "system" fn(
            flags: Dxgi::DXGI_CREATE_FACTORY_FLAGS,
            riid: *const windows_core::GUID,
            ppfactory: *mut *mut ffi::c_void,
        ) -> windows_core::HRESULT;
        let func: libloading::Symbol<Fun> =
            unsafe { self.lib.get(c"CreateDXGIFactory2".to_bytes()) }?;

        let mut result__ = None;

        (func)(
            factory_flags,
            &Dxgi::IDXGIFactory4::IID,
            <*mut _>::cast(&mut result__),
        )
        .ok()
        .into_device_result("create_factory4")?;

        result__.ok_or(crate::DeviceError::Unexpected)
    }

    /// Will error with crate::DeviceError::Unexpected if DXGI 1.3 is not available.
    #[cfg_attr(not(dx12), expect(dead_code))]
    pub fn create_factory_media(&self) -> Result<Dxgi::IDXGIFactoryMedia, crate::DeviceError> {
        // Calls windows::Win32::Graphics::Dxgi::CreateDXGIFactory1 on dxgi.dll
        type Fun = extern "system" fn(
            riid: *const windows_core::GUID,
            ppfactory: *mut *mut ffi::c_void,
        ) -> windows_core::HRESULT;
        let func: libloading::Symbol<Fun> =
            unsafe { self.lib.get(c"CreateDXGIFactory1".to_bytes()) }?;

        let mut result__ = None;

        // https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_3/nn-dxgi1_3-idxgifactorymedia
        (func)(&Dxgi::IDXGIFactoryMedia::IID, <*mut _>::cast(&mut result__))
            .ok()
            .into_device_result("create_factory_media")?;

        result__.ok_or(crate::DeviceError::Unexpected)
    }
}
