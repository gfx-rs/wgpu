use crate::{com::ComPtr, Blob, D3DResult, Error, TextureAddressMode};
use std::{fmt, mem, ops::Range};
use winapi::{shared::dxgiformat, um::d3d12};

pub type CpuDescriptor = d3d12::D3D12_CPU_DESCRIPTOR_HANDLE;
pub type GpuDescriptor = d3d12::D3D12_GPU_DESCRIPTOR_HANDLE;

#[derive(Clone, Copy, Debug)]
pub struct Binding {
    pub space: u32,
    pub register: u32,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum DescriptorHeapType {
    CbvSrvUav = d3d12::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
    Sampler = d3d12::D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
    Rtv = d3d12::D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
    Dsv = d3d12::D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct DescriptorHeapFlags: u32 {
        const SHADER_VISIBLE = d3d12::D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    }
}

pub type DescriptorHeap = ComPtr<d3d12::ID3D12DescriptorHeap>;

impl DescriptorHeap {
    pub fn start_cpu_descriptor(&self) -> CpuDescriptor {
        unsafe { self.GetCPUDescriptorHandleForHeapStart() }
    }

    pub fn start_gpu_descriptor(&self) -> GpuDescriptor {
        unsafe { self.GetGPUDescriptorHandleForHeapStart() }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum ShaderVisibility {
    All = d3d12::D3D12_SHADER_VISIBILITY_ALL,
    VS = d3d12::D3D12_SHADER_VISIBILITY_VERTEX,
    HS = d3d12::D3D12_SHADER_VISIBILITY_HULL,
    DS = d3d12::D3D12_SHADER_VISIBILITY_DOMAIN,
    GS = d3d12::D3D12_SHADER_VISIBILITY_GEOMETRY,
    PS = d3d12::D3D12_SHADER_VISIBILITY_PIXEL,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum DescriptorRangeType {
    SRV = d3d12::D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
    UAV = d3d12::D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
    CBV = d3d12::D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
    Sampler = d3d12::D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
}

#[repr(transparent)]
pub struct DescriptorRange(d3d12::D3D12_DESCRIPTOR_RANGE);
impl DescriptorRange {
    pub fn new(ty: DescriptorRangeType, count: u32, base_binding: Binding, offset: u32) -> Self {
        DescriptorRange(d3d12::D3D12_DESCRIPTOR_RANGE {
            RangeType: ty as _,
            NumDescriptors: count,
            BaseShaderRegister: base_binding.register,
            RegisterSpace: base_binding.space,
            OffsetInDescriptorsFromTableStart: offset,
        })
    }
}

impl fmt::Debug for DescriptorRange {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter
            .debug_struct("DescriptorRange")
            .field("range_type", &self.0.RangeType)
            .field("num", &self.0.NumDescriptors)
            .field("register_space", &self.0.RegisterSpace)
            .field("base_register", &self.0.BaseShaderRegister)
            .field("table_offset", &self.0.OffsetInDescriptorsFromTableStart)
            .finish()
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

    pub fn constants(visibility: ShaderVisibility, binding: Binding, num: u32) -> Self {
        let mut param = d3d12::D3D12_ROOT_PARAMETER {
            ParameterType: d3d12::D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            ShaderVisibility: visibility as _,
            ..unsafe { mem::zeroed() }
        };

        *unsafe { param.u.Constants_mut() } = d3d12::D3D12_ROOT_CONSTANTS {
            ShaderRegister: binding.register,
            RegisterSpace: binding.space,
            Num32BitValues: num,
        };

        RootParameter(param)
    }

    //TODO: should this be unsafe?
    pub fn descriptor(
        ty: d3d12::D3D12_ROOT_PARAMETER_TYPE,
        visibility: ShaderVisibility,
        binding: Binding,
    ) -> Self {
        let mut param = d3d12::D3D12_ROOT_PARAMETER {
            ParameterType: ty,
            ShaderVisibility: visibility as _,
            ..unsafe { mem::zeroed() }
        };

        *unsafe { param.u.Descriptor_mut() } = d3d12::D3D12_ROOT_DESCRIPTOR {
            ShaderRegister: binding.register,
            RegisterSpace: binding.space,
        };

        RootParameter(param)
    }

    pub fn cbv_descriptor(visibility: ShaderVisibility, binding: Binding) -> Self {
        Self::descriptor(d3d12::D3D12_ROOT_PARAMETER_TYPE_CBV, visibility, binding)
    }

    pub fn srv_descriptor(visibility: ShaderVisibility, binding: Binding) -> Self {
        Self::descriptor(d3d12::D3D12_ROOT_PARAMETER_TYPE_SRV, visibility, binding)
    }

    pub fn uav_descriptor(visibility: ShaderVisibility, binding: Binding) -> Self {
        Self::descriptor(d3d12::D3D12_ROOT_PARAMETER_TYPE_UAV, visibility, binding)
    }
}

impl fmt::Debug for RootParameter {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        #[derive(Debug)]
        #[allow(dead_code)] // False-positive
        enum Inner<'a> {
            Table(&'a [DescriptorRange]),
            Constants { binding: Binding, num: u32 },
            SingleCbv(Binding),
            SingleSrv(Binding),
            SingleUav(Binding),
        }
        let kind = match self.0.ParameterType {
            d3d12::D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE => unsafe {
                let raw = self.0.u.DescriptorTable();
                Inner::Table(std::slice::from_raw_parts(
                    raw.pDescriptorRanges as *const _,
                    raw.NumDescriptorRanges as usize,
                ))
            },
            d3d12::D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS => unsafe {
                let raw = self.0.u.Constants();
                Inner::Constants {
                    binding: Binding {
                        space: raw.RegisterSpace,
                        register: raw.ShaderRegister,
                    },
                    num: raw.Num32BitValues,
                }
            },
            _ => unsafe {
                let raw = self.0.u.Descriptor();
                let binding = Binding {
                    space: raw.RegisterSpace,
                    register: raw.ShaderRegister,
                };
                match self.0.ParameterType {
                    d3d12::D3D12_ROOT_PARAMETER_TYPE_CBV => Inner::SingleCbv(binding),
                    d3d12::D3D12_ROOT_PARAMETER_TYPE_SRV => Inner::SingleSrv(binding),
                    d3d12::D3D12_ROOT_PARAMETER_TYPE_UAV => Inner::SingleUav(binding),
                    other => panic!("Unexpected type {:?}", other),
                }
            },
        };

        formatter
            .debug_struct("RootParameter")
            .field("visibility", &self.0.ShaderVisibility)
            .field("kind", &kind)
            .finish()
    }
}

#[repr(u32)]
#[derive(Copy, Clone, Debug)]
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
        binding: Binding,
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
            ShaderRegister: binding.register,
            RegisterSpace: binding.space,
            ShaderVisibility: visibility as _,
        })
    }
}

#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum RootSignatureVersion {
    V1_0 = d3d12::D3D_ROOT_SIGNATURE_VERSION_1_0,
    V1_1 = d3d12::D3D_ROOT_SIGNATURE_VERSION_1_1,
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct RootSignatureFlags: u32 {
        const ALLOW_IA_INPUT_LAYOUT = d3d12::D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        const DENY_VS_ROOT_ACCESS = d3d12::D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS;
        const DENY_HS_ROOT_ACCESS = d3d12::D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS;
        const DENY_DS_ROOT_ACCESS = d3d12::D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS;
        const DENY_GS_ROOT_ACCESS = d3d12::D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;
        const DENY_PS_ROOT_ACCESS = d3d12::D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS;
    }
}

pub type RootSignature = ComPtr<d3d12::ID3D12RootSignature>;
pub type BlobResult = D3DResult<(Blob, Error)>;

#[cfg(feature = "libloading")]
impl crate::D3D12Lib {
    pub fn serialize_root_signature(
        &self,
        version: RootSignatureVersion,
        parameters: &[RootParameter],
        static_samplers: &[StaticSampler],
        flags: RootSignatureFlags,
    ) -> Result<BlobResult, libloading::Error> {
        use winapi::um::d3dcommon::ID3DBlob;
        type Fun = extern "system" fn(
            *const d3d12::D3D12_ROOT_SIGNATURE_DESC,
            d3d12::D3D_ROOT_SIGNATURE_VERSION,
            *mut *mut ID3DBlob,
            *mut *mut ID3DBlob,
        ) -> crate::HRESULT;

        let desc = d3d12::D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: parameters.len() as _,
            pParameters: parameters.as_ptr() as *const _,
            NumStaticSamplers: static_samplers.len() as _,
            pStaticSamplers: static_samplers.as_ptr() as _,
            Flags: flags.bits(),
        };

        let mut blob = Blob::null();
        let mut error = Error::null();
        let hr = unsafe {
            let func: libloading::Symbol<Fun> = self.lib.get(b"D3D12SerializeRootSignature")?;
            func(
                &desc,
                version as _,
                blob.mut_void() as *mut *mut _,
                error.mut_void() as *mut *mut _,
            )
        };

        Ok(((blob, error), hr))
    }
}

impl RootSignature {
    #[cfg(feature = "implicit-link")]
    pub fn serialize(
        version: RootSignatureVersion,
        parameters: &[RootParameter],
        static_samplers: &[StaticSampler],
        flags: RootSignatureFlags,
    ) -> BlobResult {
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

#[repr(transparent)]
pub struct RenderTargetViewDesc(pub(crate) d3d12::D3D12_RENDER_TARGET_VIEW_DESC);

impl RenderTargetViewDesc {
    pub fn texture_2d(format: dxgiformat::DXGI_FORMAT, mip_slice: u32, plane_slice: u32) -> Self {
        let mut desc = d3d12::D3D12_RENDER_TARGET_VIEW_DESC {
            Format: format,
            ViewDimension: d3d12::D3D12_RTV_DIMENSION_TEXTURE2D,
            ..unsafe { mem::zeroed() }
        };

        *unsafe { desc.u.Texture2D_mut() } = d3d12::D3D12_TEX2D_RTV {
            MipSlice: mip_slice,
            PlaneSlice: plane_slice,
        };

        RenderTargetViewDesc(desc)
    }
}
