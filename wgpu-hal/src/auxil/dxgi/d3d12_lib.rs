#[cfg(dx12)]
use core::ffi;
use core::{error::Error, fmt};

use wgpu_sync::OnceCell;
use windows::{
    core::{IUnknown, Interface as _, Ref},
    Win32::Graphics::{Direct3D, Direct3D12, Dxgi},
};

use crate::auxil::dxgi::{factory::DxgiAdapter, result::HResult as _};
use crate::auxil::dyn_lib::DynLib;

/// Loads `d3d12.dll` at runtime, the same way [`DxgiLib`](super::dxgi_lib::DxgiLib) loads
/// `dxgi.dll`, so the DLL is never a load-time import and the binary stays loadable on systems
/// without D3D12 (e.g. Windows 7/8, where Vulkan still needs to work). Shared by the DX12 backend
/// and the Vulkan DXGI interop swapchain.
#[derive(Debug)]
pub(crate) struct D3D12Lib {
    lib: DynLib,
    // Entry points, loaded lazily by `cached_proc`.
    create_device_fn: OnceCell<Direct3D12::PFN_D3D12_CREATE_DEVICE>,
    debug_interface_fn: OnceCell<Direct3D12::PFN_D3D12_GET_DEBUG_INTERFACE>,
    #[cfg(dx12)]
    serialize_root_signature_fn: OnceCell<Direct3D12::PFN_D3D12_SERIALIZE_ROOT_SIGNATURE>,
    #[cfg(dxgi)]
    get_interface_fn: OnceCell<Direct3D12::PFN_D3D12_GET_INTERFACE>,
}

impl D3D12Lib {
    pub(crate) fn new() -> Result<Self, libloading::Error> {
        unsafe {
            DynLib::new("d3d12.dll").map(|lib| Self {
                lib,
                create_device_fn: OnceCell::new(),
                debug_interface_fn: OnceCell::new(),
                #[cfg(dx12)]
                serialize_root_signature_fn: OnceCell::new(),
                #[cfg(dxgi)]
                get_interface_fn: OnceCell::new(),
            })
        }
    }

    /// Loads the entry point `symbol` on first use, caching it in `cell`.
    fn cached_proc<T: Copy>(
        &self,
        cell: &OnceCell<T>,
        symbol: &[u8],
    ) -> Result<T, crate::DeviceError> {
        cell.get_or_try_init(|| {
            let sym: libloading::Symbol<T> = unsafe { self.lib.get(symbol) }?;
            Ok(*sym)
        })
        .copied()
    }

    pub(crate) fn create_device(
        &self,
        adapter: &DxgiAdapter,
        feature_level: Direct3D::D3D_FEATURE_LEVEL,
    ) -> Result<Direct3D12::ID3D12Device, CreateDeviceError> {
        // Calls windows::Win32::Graphics::Direct3D12::D3D12CreateDevice on d3d12.dll.
        let func = self
            .cached_proc(&self.create_device_fn, c"D3D12CreateDevice".to_bytes())
            .map_err(|_| CreateDeviceError::GetProcAddress)?
            .ok_or(CreateDeviceError::GetProcAddress)?;

        // `Ref<IUnknown>` is a `repr(transparent)` non-owning borrow of the COM pointer (it does not
        // touch the refcount); the adapter outlives this synchronous call.
        let adapter: Ref<IUnknown> = unsafe { core::mem::transmute(adapter.as_raw()) };

        let mut result__: Option<Direct3D12::ID3D12Device> = None;
        let res = unsafe {
            func(
                adapter,
                feature_level,
                &Direct3D12::ID3D12Device::IID,
                <*mut _>::cast(&mut result__),
            )
        };

        if res.is_err() {
            return Err(CreateDeviceError::D3D12CreateDevice(res));
        }
        result__.ok_or(CreateDeviceError::RetDeviceIsNull)
    }

    #[cfg(dx12)]
    pub(crate) fn serialize_root_signature(
        &self,
        version: Direct3D12::D3D_ROOT_SIGNATURE_VERSION,
        parameters: &[Direct3D12::D3D12_ROOT_PARAMETER],
        static_samplers: &[Direct3D12::D3D12_STATIC_SAMPLER_DESC],
        flags: Direct3D12::D3D12_ROOT_SIGNATURE_FLAGS,
    ) -> Result<Direct3D::ID3DBlob, crate::DeviceError> {
        // Calls windows::Win32::Graphics::Direct3D12::D3D12SerializeRootSignature on d3d12.dll.
        let func = self
            .cached_proc(
                &self.serialize_root_signature_fn,
                c"D3D12SerializeRootSignature".to_bytes(),
            )?
            .ok_or(crate::DeviceError::Unexpected)?;

        let desc = Direct3D12::D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: parameters.len() as _,
            pParameters: parameters.as_ptr(),
            NumStaticSamplers: static_samplers.len() as _,
            pStaticSamplers: static_samplers.as_ptr(),
            Flags: flags,
        };

        let mut blob: Option<Direct3D::ID3DBlob> = None;
        let mut error: Option<Direct3D::ID3DBlob> = None;
        unsafe { func(&desc, version, (&mut blob).into(), (&mut error).into()) }
            .ok()
            .into_device_result("Root signature serialization")?;

        if let Some(error) = error {
            let message = unsafe {
                let slice = core::slice::from_raw_parts(
                    error.GetBufferPointer().cast::<u8>(),
                    error.GetBufferSize(),
                );
                ffi::CStr::from_bytes_until_nul(slice)
            };
            log::error!(
                "Root signature serialization error: {:?}",
                message.unwrap().to_str().unwrap()
            );
            return Err(crate::DeviceError::Unexpected); // could be hal_usage_error or hal_internal_error
        }

        blob.ok_or(crate::DeviceError::Unexpected)
    }

    pub(crate) fn debug_interface(
        &self,
    ) -> Result<Option<Direct3D12::ID3D12Debug>, crate::DeviceError> {
        // Calls windows::Win32::Graphics::Direct3D12::D3D12GetDebugInterface on d3d12.dll.
        let func = self
            .cached_proc(
                &self.debug_interface_fn,
                c"D3D12GetDebugInterface".to_bytes(),
            )?
            .ok_or(crate::DeviceError::Unexpected)?;

        let mut result__ = None;
        let res =
            unsafe { func(&Direct3D12::ID3D12Debug::IID, <*mut _>::cast(&mut result__)) }.ok();

        if let Err(ref err) = res {
            if err.code() == Dxgi::DXGI_ERROR_SDK_COMPONENT_MISSING {
                return Ok(None);
            }
        }

        res.into_device_result("GetDebugInterface")?;

        result__.ok_or(crate::DeviceError::Unexpected).map(Some)
    }

    /// Calls D3D12GetInterface to obtain a COM interface by CLSID and IID.
    ///
    /// This is used by the Independent Devices API to obtain `ID3D12SDKConfiguration1`.
    #[cfg(dxgi)]
    pub(crate) fn get_interface<T: windows_core::Interface>(
        &self,
        clsid: &windows_core::GUID,
    ) -> Result<T, GetInterfaceError> {
        // Calls windows::Win32::Graphics::Direct3D12::D3D12GetInterface on d3d12.dll.
        let func = self
            .cached_proc(&self.get_interface_fn, c"D3D12GetInterface".to_bytes())
            .map_err(|_| GetInterfaceError::GetProcAddress)?
            .ok_or(GetInterfaceError::GetProcAddress)?;

        let mut result__: Option<T> = None;
        let res = unsafe { func(clsid, &T::IID, <*mut _>::cast(&mut result__)) };

        if res.is_err() {
            return Err(GetInterfaceError::D3D12GetInterface(res));
        }
        result__.ok_or(GetInterfaceError::RetIsNull)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum CreateDeviceError {
    GetProcAddress,
    D3D12CreateDevice(windows_core::HRESULT),
    RetDeviceIsNull,
}

impl fmt::Display for CreateDeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GetProcAddress => write!(f, "D3D12CreateDevice not found in d3d12.dll"),
            Self::D3D12CreateDevice(hr) => write!(f, "D3D12CreateDevice failed: {hr}"),
            Self::RetDeviceIsNull => write!(f, "D3D12CreateDevice returned null"),
        }
    }
}

impl Error for CreateDeviceError {}

#[cfg(dxgi)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum GetInterfaceError {
    GetProcAddress,
    D3D12GetInterface(windows_core::HRESULT),
    RetIsNull,
}

#[cfg(dxgi)]
impl fmt::Display for GetInterfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GetProcAddress => write!(f, "D3D12GetInterface not found in d3d12.dll"),
            Self::D3D12GetInterface(hr) => write!(f, "D3D12GetInterface failed: {hr}"),
            Self::RetIsNull => write!(f, "D3D12GetInterface returned null"),
        }
    }
}

#[cfg(dxgi)]
impl Error for GetInterfaceError {}
