use com::WeakPtr;
use std::mem;
use std::ops::Range;
use winapi::um::d3d12;
use {Blob, D3DResult, Error, TextureAddressMode};

pub type CpuDescriptor = d3d12::D3D12_CPU_DESCRIPTOR_HANDLE;
pub type GpuDescriptor = d3d12::D3D12_GPU_DESCRIPTOR_HANDLE;

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum HeapType {
    CbvSrvUav = d3d12::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
    Sampler = d3d12::D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
    Rtv = d3d12::D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
    Dsv = d3d12::D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
}

bitflags! {
    pub struct HeapFlags: u32 {
        const SHADER_VISIBLE = d3d12::D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    }
}

pub type DescriptorHeap = WeakPtr<d3d12::ID3D12DescriptorHeap>;

impl DescriptorHeap {
    pub fn start_cpu_descriptor(&self) -> CpuDescriptor {
        unsafe { self.GetCPUDescriptorHandleForHeapStart() }
    }

    pub fn start_gpu_descriptor(&self) -> GpuDescriptor {
        unsafe { self.GetGPUDescriptorHandleForHeapStart() }
    }
}

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum ShaderVisibility {
    All = d3d12::D3D12_SHADER_VISIBILITY_ALL,
    VS = d3d12::D3D12_SHADER_VISIBILITY_VERTEX,
    HS = d3d12::D3D12_SHADER_VISIBILITY_HULL,
    DS = d3d12::D3D12_SHADER_VISIBILITY_DOMAIN,
    GS = d3d12::D3D12_SHADER_VISIBILITY_GEOMETRY,
    PS = d3d12::D3D12_SHADER_VISIBILITY_PIXEL,
}

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum DescriptorRangeType {
    SRV = d3d12::D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
    UAV = d3d12::D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
    CBV = d3d12::D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
    Sampler = d3d12::D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
}

#[repr(transparent)]
pub struct DescriptorRange(d3d12::D3D12_DESCRIPTOR_RANGE);
impl DescriptorRange {
    pub fn new(
        ty: DescriptorRangeType,
        count: u32,
        base_register: u32,
        register_space: u32,
        offset: u32,
    ) -> Self {
        DescriptorRange(d3d12::D3D12_DESCRIPTOR_RANGE {
            RangeType: ty as _,
            NumDescriptors: count,
            BaseShaderRegister: base_register,
            RegisterSpace: register_space,
            OffsetInDescriptorsFromTableStart: offset,
        })
    }
}

#[repr(transparent)]
pub struct RootParameter(d3d12::D3D12_ROOT_PARAMETER);
impl RootParameter {
    // TODO: DescriptorRange must outlive Self
    pub fn descriptor_table(visibility: ShaderVisibility, ranges: &[DescriptorRange]) -> Self {
        let mut param = d3d12::D3D12_ROOT_PARAMETER {
            ParameterType: d3d12::D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            ShaderVisibility: visibility as _,
            ..unsafe { mem::zeroed() }
        };

        *unsafe { param.u.DescriptorTable_mut() } = d3d12::D3D12_ROOT_DESCRIPTOR_TABLE {
            NumDescriptorRanges: ranges.len() as _,
            pDescriptorRanges: ranges.as_ptr() as *const _,
        };

        RootParameter(param)
    }

    pub fn constants(
        visibility: ShaderVisibility,
        register: u32,
        register_space: u32,
        num: u32,
    ) -> Self {
        let mut param = d3d12::D3D12_ROOT_PARAMETER {
            ParameterType: d3d12::D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            ShaderVisibility: visibility as _,
            ..unsafe { mem::zeroed() }
        };

        *unsafe { param.u.Constants_mut() } = d3d12::D3D12_ROOT_CONSTANTS {
            ShaderRegister: register,
            RegisterSpace: register_space,
            Num32BitValues: num,
        };

        RootParameter(param)
    }
}

#[repr(u32)]
pub enum StaticBorderColor {
    TransparentBlack = d3d12::D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
    OpaqueBlack = d3d12::D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
    OpaqueWhite = d3d12::D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
}

#[repr(transparent)]
pub struct StaticSampler(d3d12::D3D12_STATIC_SAMPLER_DESC);
impl StaticSampler {
    pub fn new(
        visibility: ShaderVisibility,
        register: u32,
        register_space: u32,

        filter: d3d12::D3D12_FILTER,
        address_mode: TextureAddressMode,
        mip_lod_bias: f32,
        max_anisotropy: u32,
        comparison_op: d3d12::D3D12_COMPARISON_FUNC,
        border_color: StaticBorderColor,
        lod: Range<f32>,
    ) -> Self {
        StaticSampler(d3d12::D3D12_STATIC_SAMPLER_DESC {
            Filter: filter,
            AddressU: address_mode[0],
            AddressV: address_mode[1],
            AddressW: address_mode[2],
            MipLODBias: mip_lod_bias,
            MaxAnisotropy: max_anisotropy,
            ComparisonFunc: comparison_op,
            BorderColor: border_color as _,
            MinLOD: lod.start,
            MaxLOD: lod.end,
            ShaderRegister: register,
            RegisterSpace: register_space,
            ShaderVisibility: visibility as _,
        })
    }
}

#[repr(u32)]
pub enum RootSignatureVersion {
    V1_0 = d3d12::D3D_ROOT_SIGNATURE_VERSION_1_0,
    V1_1 = d3d12::D3D_ROOT_SIGNATURE_VERSION_1_1,
}

bitflags! {
    pub struct RootSignatureFlags: u32 {
        const ALLOW_IA_INPUT_LAYOUT = d3d12::D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        const DENY_VS_ROOT_ACCESS = d3d12::D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS;
        const DENY_HS_ROOT_ACCESS = d3d12::D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS;
        const DENY_DS_ROOT_ACCESS = d3d12::D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS;
        const DENY_GS_ROOT_ACCESS = d3d12::D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;
        const DENY_PS_ROOT_ACCESS = d3d12::D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS;
    }
}

pub type RootSignature = WeakPtr<d3d12::ID3D12RootSignature>;

impl RootSignature {
    pub fn serialize(
        version: RootSignatureVersion,
        parameters: &[RootParameter],
        static_samplers: &[StaticSampler],
        flags: RootSignatureFlags,
    ) -> D3DResult<(Blob, Error)> {
        let mut blob = Blob::null();
        let mut error = Error::null();

        let desc = d3d12::D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: parameters.len() as _,
            pParameters: parameters.as_ptr() as *const _,
            NumStaticSamplers: static_samplers.len() as _,
            pStaticSamplers: static_samplers.as_ptr() as _,
            Flags: flags.bits(),
        };

        let hr = unsafe {
            d3d12::D3D12SerializeRootSignature(
                &desc,
                version as _,
                blob.mut_void() as *mut *mut _,
                error.mut_void() as *mut *mut _,
            )
        };

        ((blob, error), hr)
    }
}
