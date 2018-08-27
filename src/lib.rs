extern crate winapi;
#[macro_use]
extern crate bitflags;

use std::ffi::CStr;
use winapi::shared::dxgiformat;
use winapi::um::{d3d12, d3dcommon};

mod com;
pub mod command_allocator;
pub mod command_list;
pub mod descriptor;
pub mod device;
pub mod pso;
pub mod query;
pub mod queue;
pub mod resource;
pub mod sync;

pub use self::com::WeakPtr;
pub use self::command_allocator::CommandAllocator;
pub use self::command_list::{CommandSignature, GraphicsCommandList};
pub use self::descriptor::{CpuDescriptor, DescriptorHeap, GpuDescriptor, RootSignature};
pub use self::device::Device;
pub use self::pso::{CachedPSO, PipelineState, Shader};
pub use self::query::QueryHeap;
pub use self::queue::CommandQueue;
pub use self::resource::{Heap, Resource};
pub use self::sync::Fence;

pub use winapi::shared::winerror::HRESULT;

pub type D3DResult<T> = (T, HRESULT);
pub type GpuAddress = d3d12::D3D12_GPU_VIRTUAL_ADDRESS;
pub type Format = dxgiformat::DXGI_FORMAT;
pub type Rect = d3d12::D3D12_RECT;
pub type NodeMask = u32;

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

#[repr(u32)]
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

pub type Blob = self::com::WeakPtr<d3dcommon::ID3DBlob>;

pub type Error = self::com::WeakPtr<d3dcommon::ID3DBlob>;
impl Error {
    pub unsafe fn as_c_str(&self) -> &CStr {
        debug_assert!(!self.is_null());
        let data = self.GetBufferPointer();
        CStr::from_ptr(data as *const _ as *const _)
    }
}
