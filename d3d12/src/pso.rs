//! Pipeline state

use crate::{com::ComPtr, Blob, D3DResult, Error};
use std::{
    ffi::{self, c_void},
    marker::PhantomData,
    ops::Deref,
    ptr,
};
use winapi::um::{d3d12, d3dcompiler};

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct PipelineStateFlags: u32 {
        const TOOL_DEBUG = d3d12::D3D12_PIPELINE_STATE_FLAG_TOOL_DEBUG;
    }
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct ShaderCompileFlags: u32 {
        const DEBUG = d3dcompiler::D3DCOMPILE_DEBUG;
        const SKIP_VALIDATION = d3dcompiler::D3DCOMPILE_SKIP_VALIDATION;
        const SKIP_OPTIMIZATION = d3dcompiler::D3DCOMPILE_SKIP_OPTIMIZATION;
        const PACK_MATRIX_ROW_MAJOR = d3dcompiler::D3DCOMPILE_PACK_MATRIX_ROW_MAJOR;
        const PACK_MATRIX_COLUMN_MAJOR = d3dcompiler::D3DCOMPILE_PACK_MATRIX_COLUMN_MAJOR;
        const PARTIAL_PRECISION = d3dcompiler::D3DCOMPILE_PARTIAL_PRECISION;
        // TODO: add missing flags
    }
}

#[derive(Copy, Clone)]
pub struct Shader<'a>(d3d12::D3D12_SHADER_BYTECODE, PhantomData<&'a c_void>);
impl<'a> Shader<'a> {
    pub fn null() -> Self {
        Shader(
            d3d12::D3D12_SHADER_BYTECODE {
                BytecodeLength: 0,
                pShaderBytecode: ptr::null(),
            },
            PhantomData,
        )
    }

    pub fn from_raw(data: &'a [u8]) -> Self {
        Shader(
            d3d12::D3D12_SHADER_BYTECODE {
                BytecodeLength: data.len() as _,
                pShaderBytecode: data.as_ptr() as _,
            },
            PhantomData,
        )
    }

    // `blob` may not be null.
    pub fn from_blob(blob: &'a Blob) -> Self {
        Shader(
            d3d12::D3D12_SHADER_BYTECODE {
                BytecodeLength: unsafe { blob.GetBufferSize() },
                pShaderBytecode: unsafe { blob.GetBufferPointer() },
            },
            PhantomData,
        )
    }

    /// Compile a shader from raw HLSL.
    ///
    /// * `target`: example format: `ps_5_1`.
    pub fn compile(
        code: &[u8],
        target: &ffi::CStr,
        entry: &ffi::CStr,
        flags: ShaderCompileFlags,
    ) -> D3DResult<(Blob, Error)> {
        let mut shader = Blob::null();
        let mut error = Error::null();

        let hr = unsafe {
            d3dcompiler::D3DCompile(
                code.as_ptr() as *const _,
                code.len(),
                ptr::null(), // defines
                ptr::null(), // include
                ptr::null_mut(),
                entry.as_ptr() as *const _,
                target.as_ptr() as *const _,
                flags.bits(),
                0,
                shader.mut_void() as *mut *mut _,
                error.mut_void() as *mut *mut _,
            )
        };

        ((shader, error), hr)
    }
}

impl<'a> Deref for Shader<'a> {
    type Target = d3d12::D3D12_SHADER_BYTECODE;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Copy, Clone)]
pub struct CachedPSO<'a>(d3d12::D3D12_CACHED_PIPELINE_STATE, PhantomData<&'a c_void>);
impl<'a> CachedPSO<'a> {
    pub fn null() -> Self {
        CachedPSO(
            d3d12::D3D12_CACHED_PIPELINE_STATE {
                CachedBlobSizeInBytes: 0,
                pCachedBlob: ptr::null(),
            },
            PhantomData,
        )
    }

    // `blob` may not be null.
    pub fn from_blob(blob: &'a Blob) -> Self {
        CachedPSO(
            d3d12::D3D12_CACHED_PIPELINE_STATE {
                CachedBlobSizeInBytes: unsafe { blob.GetBufferSize() },
                pCachedBlob: unsafe { blob.GetBufferPointer() },
            },
            PhantomData,
        )
    }
}

impl<'a> Deref for CachedPSO<'a> {
    type Target = d3d12::D3D12_CACHED_PIPELINE_STATE;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub type PipelineState = ComPtr<d3d12::ID3D12PipelineState>;

#[repr(u32)]
pub enum Subobject {
    RootSignature = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
    VS = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS,
    PS = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS,
    DS = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS,
    HS = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS,
    GS = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS,
    CS = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS,
    StreamOutput = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT,
    Blend = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND,
    SampleMask = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK,
    Rasterizer = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,
    DepthStencil = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL,
    InputLayout = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT,
    IBStripCut = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE,
    PrimitiveTopology = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
    RTFormats = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
    DSFormat = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
    SampleDesc = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC,
    NodeMask = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK,
    CachedPSO = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO,
    Flags = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS,
    DepthStencil1 = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1,
    // ViewInstancing = d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING,
}

/// Subobject of a pipeline stream description
#[repr(C)]
pub struct PipelineStateSubobject<T> {
    subobject_align: [usize; 0], // Subobjects must have the same alignment as pointers.
    subobject_type: d3d12::D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
    subobject: T,
}

impl<T> PipelineStateSubobject<T> {
    pub fn new(subobject_type: Subobject, subobject: T) -> Self {
        PipelineStateSubobject {
            subobject_align: [],
            subobject_type: subobject_type as _,
            subobject,
        }
    }
}
