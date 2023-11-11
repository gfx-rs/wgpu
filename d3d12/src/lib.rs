#![cfg(windows)]
#![allow(
    clippy::missing_safety_doc,
    clippy::too_many_arguments,
    clippy::not_unsafe_ptr_arg_deref
)]

use std::{convert::TryFrom, ffi::CStr};
use winapi::{
    shared::dxgiformat,
    um::{d3d12, d3dcommon},
};

mod com;
mod command_allocator;
mod command_list;
mod debug;
mod descriptor;
mod device;
mod dxgi;
mod heap;
mod pso;
mod query;
mod queue;
mod resource;
mod sync;

pub use crate::com::*;
pub use crate::command_allocator::*;
pub use crate::command_list::*;
pub use crate::debug::*;
pub use crate::descriptor::*;
pub use crate::device::*;
pub use crate::dxgi::*;
pub use crate::heap::*;
pub use crate::pso::*;
pub use crate::query::*;
pub use crate::queue::*;
pub use crate::resource::*;
pub use crate::sync::*;

pub use winapi::shared::winerror::HRESULT;

pub type D3DResult<T> = (T, HRESULT);
pub type GpuAddress = d3d12::D3D12_GPU_VIRTUAL_ADDRESS;
pub type Format = dxgiformat::DXGI_FORMAT;
pub type Rect = d3d12::D3D12_RECT;
pub type NodeMask = u32;

/// Index into the root signature.
pub type RootIndex = u32;
/// Draw vertex count.
pub type VertexCount = u32;
/// Draw vertex base offset.
pub type VertexOffset = i32;
/// Draw number of indices.
pub type IndexCount = u32;
/// Draw number of instances.
pub type InstanceCount = u32;
/// Number of work groups.
pub type WorkGroupCount = [u32; 3];

pub type TextureAddressMode = [d3d12::D3D12_TEXTURE_ADDRESS_MODE; 3];

pub struct SampleDesc {
    pub count: u32,
    pub quality: u32,
}

#[repr(u32)]
#[non_exhaustive]
pub enum FeatureLevel {
    L9_1 = d3dcommon::D3D_FEATURE_LEVEL_9_1,
    L9_2 = d3dcommon::D3D_FEATURE_LEVEL_9_2,
    L9_3 = d3dcommon::D3D_FEATURE_LEVEL_9_3,
    L10_0 = d3dcommon::D3D_FEATURE_LEVEL_10_0,
    L10_1 = d3dcommon::D3D_FEATURE_LEVEL_10_1,
    L11_0 = d3dcommon::D3D_FEATURE_LEVEL_11_0,
    L11_1 = d3dcommon::D3D_FEATURE_LEVEL_11_1,
    L12_0 = d3dcommon::D3D_FEATURE_LEVEL_12_0,
    L12_1 = d3dcommon::D3D_FEATURE_LEVEL_12_1,
}

impl TryFrom<u32> for FeatureLevel {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            d3dcommon::D3D_FEATURE_LEVEL_9_1 => Self::L9_1,
            d3dcommon::D3D_FEATURE_LEVEL_9_2 => Self::L9_2,
            d3dcommon::D3D_FEATURE_LEVEL_9_3 => Self::L9_3,
            d3dcommon::D3D_FEATURE_LEVEL_10_0 => Self::L10_0,
            d3dcommon::D3D_FEATURE_LEVEL_10_1 => Self::L10_1,
            d3dcommon::D3D_FEATURE_LEVEL_11_0 => Self::L11_0,
            d3dcommon::D3D_FEATURE_LEVEL_11_1 => Self::L11_1,
            d3dcommon::D3D_FEATURE_LEVEL_12_0 => Self::L12_0,
            d3dcommon::D3D_FEATURE_LEVEL_12_1 => Self::L12_1,
            _ => return Err(()),
        })
    }
}

pub type Blob = ComPtr<d3dcommon::ID3DBlob>;

pub type Error = ComPtr<d3dcommon::ID3DBlob>;
impl Error {
    pub unsafe fn as_c_str(&self) -> &CStr {
        debug_assert!(!self.is_null());
        let data = self.GetBufferPointer();
        CStr::from_ptr(data as *const _ as *const _)
    }
}

#[cfg(feature = "libloading")]
#[derive(Debug)]
pub struct D3D12Lib {
    lib: libloading::Library,
}

#[cfg(feature = "libloading")]
impl D3D12Lib {
    pub fn new() -> Result<Self, libloading::Error> {
        unsafe { libloading::Library::new("d3d12.dll").map(|lib| D3D12Lib { lib }) }
    }
}
