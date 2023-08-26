use std::ptr;

use winapi::{
    shared::{
        dxgi,
        minwindef::{HMODULE, UINT},
        winerror,
    },
    um::{d3d11, d3d11_1, d3d11_2, d3dcommon},
};

use crate::auxil::dxgi::result::HResult;

type D3D11CreateDeviceFun = unsafe extern "system" fn(
    *mut dxgi::IDXGIAdapter,
    d3dcommon::D3D_DRIVER_TYPE,
    HMODULE,
    UINT,
    *const d3dcommon::D3D_FEATURE_LEVEL,
    UINT,
    UINT,
    *mut *mut d3d11::ID3D11Device,
    *mut d3dcommon::D3D_FEATURE_LEVEL,
    *mut *mut d3d11::ID3D11DeviceContext,
) -> d3d12::HRESULT;

pub(super) struct D3D11Lib {
    // We use the os specific symbol to drop the lifetime parameter.
    //
    // SAFETY: we must ensure this outlives the Library.
    d3d11_create_device: libloading::os::windows::Symbol<D3D11CreateDeviceFun>,

    lib: libloading::Library,
}
impl D3D11Lib {
    pub fn new() -> Option<Self> {
        unsafe {
            let lib = libloading::Library::new("d3d11.dll").ok()?;

            let d3d11_create_device = lib
                .get::<D3D11CreateDeviceFun>(b"D3D11CreateDevice")
                .ok()?
                .into_raw();

            Some(Self {
                lib,
                d3d11_create_device,
            })
        }
    }

    pub fn create_device(
        &self,
        adapter: d3d12::DxgiAdapter,
    ) -> Option<(super::D3D11Device, d3dcommon::D3D_FEATURE_LEVEL)> {
        let feature_levels = [
            d3dcommon::D3D_FEATURE_LEVEL_11_1,
            d3dcommon::D3D_FEATURE_LEVEL_11_0,
            d3dcommon::D3D_FEATURE_LEVEL_10_1,
            d3dcommon::D3D_FEATURE_LEVEL_10_0,
            d3dcommon::D3D_FEATURE_LEVEL_9_3,
            d3dcommon::D3D_FEATURE_LEVEL_9_2,
            d3dcommon::D3D_FEATURE_LEVEL_9_1,
        ];

        let mut device = d3d12::ComPtr::<d3d11::ID3D11Device>::null();
        let mut feature_level: d3dcommon::D3D_FEATURE_LEVEL = 0;

        // We need to try this twice. If the first time fails due to E_INVALIDARG
        // we are running on a machine without a D3D11.1 runtime, and need to
        // retry without the feature level 11_1 feature level.
        //
        // Why they thought this was a good API, who knows.

        let mut hr = unsafe {
            (self.d3d11_create_device)(
                adapter.as_mut_ptr() as *mut _,
                d3dcommon::D3D_DRIVER_TYPE_UNKNOWN,
                ptr::null_mut(), // software implementation DLL???
                0,               // flags
                feature_levels.as_ptr(),
                feature_levels.len() as u32,
                d3d11::D3D11_SDK_VERSION,
                device.mut_self(),
                &mut feature_level,
                ptr::null_mut(), // device context
            )
        };

        // Try again without FL11_1
        if hr == winerror::E_INVALIDARG {
            hr = unsafe {
                (self.d3d11_create_device)(
                    adapter.as_mut_ptr() as *mut _,
                    d3dcommon::D3D_DRIVER_TYPE_UNKNOWN,
                    ptr::null_mut(), // software implementation DLL???
                    0,               // flags
                    feature_levels[1..].as_ptr(),
                    feature_levels[1..].len() as u32,
                    d3d11::D3D11_SDK_VERSION,
                    device.mut_self(),
                    &mut feature_level,
                    ptr::null_mut(), // device context
                )
            };
        }

        // Any errors here are real and we should complain about
        if let Err(err) = hr.into_result() {
            log::error!("Failed to make a D3D11 device: {}", err);
            return None;
        }

        // We always try to upcast in highest -> lowest order

        // Device -> Device2
        unsafe {
            match device.cast::<d3d11_2::ID3D11Device2>().into_result() {
                Ok(device2) => {
                    return Some((super::D3D11Device::Device2(device2), feature_level));
                }
                Err(hr) => {
                    log::info!("Failed to cast device to ID3D11Device2: {}", hr)
                }
            }
        }

        // Device -> Device1
        unsafe {
            match device.cast::<d3d11_1::ID3D11Device1>().into_result() {
                Ok(device1) => {
                    return Some((super::D3D11Device::Device1(device1), feature_level));
                }
                Err(hr) => {
                    log::info!("Failed to cast device to ID3D11Device1: {}", hr)
                }
            }
        }

        Some((super::D3D11Device::Device(device), feature_level))
    }
}
