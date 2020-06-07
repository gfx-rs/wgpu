/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    binding_model::{BindEntryMap, BindGroupLayoutEntry, BindingType},
    device::RenderPassContext,
    id::{DeviceId, PipelineLayoutId, ShaderModuleId},
    FastHashMap, LifeGuard, RawString, RefCount, Stored, U32Array,
};
use spirv_headers as spirv;
use std::borrow::Borrow;
use wgt::{
    BufferAddress, ColorStateDescriptor, DepthStencilStateDescriptor, IndexFormat, InputStepMode,
    PrimitiveTopology, RasterizationStateDescriptor, VertexAttributeDescriptor,
};

#[repr(C)]
#[derive(Debug)]
pub struct VertexBufferLayoutDescriptor {
    pub array_stride: BufferAddress,
    pub step_mode: InputStepMode,
    pub attributes: *const VertexAttributeDescriptor,
    pub attributes_length: usize,
}

#[repr(C)]
#[derive(Debug)]
pub struct VertexStateDescriptor {
    pub index_format: IndexFormat,
    pub vertex_buffers: *const VertexBufferLayoutDescriptor,
    pub vertex_buffers_length: usize,
}

#[repr(C)]
#[derive(Debug)]
pub struct ShaderModuleDescriptor {
    pub code: U32Array,
}

#[derive(Debug)]
pub struct ShaderModule<B: hal::Backend> {
    pub(crate) raw: B::ShaderModule,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) module: Option<naga::Module>,
}

#[repr(C)]
#[derive(Debug)]
pub struct ProgrammableStageDescriptor {
    pub module: ShaderModuleId,
    pub entry_point: RawString,
}

#[derive(Clone, Debug)]
pub enum BindingError {
    /// The binding is missing from the pipeline layout.
    Missing,
    /// The visibility flags don't include the shader stage.
    Invisible,
    /// The load/store access flags don't match the shader.
    WrongUsage(naga::GlobalUse),
    /// The type on the shader side does not match the pipeline binding.
    WrongType,
    /// The view dimension doesn't match the shader.
    WrongTextureViewDimension { dim: spirv::Dim, is_array: bool },
    /// The component type of a sampled texture doesn't match the shader.
    WrongTextureComponentType(Option<naga::ScalarKind>),
    /// Texture sampling capability doesn't match with the shader.
    WrongTextureSampled,
    /// The multisampled flag doesn't match.
    WrongTextureMultisampled,
}

#[derive(Clone, Debug)]
pub enum InputError {
    /// The input is not provided by the earlier stage in the pipeline.
    Missing,
    /// The input type is not compatible with the provided.
    WrongType,
}

/// Errors produced when validating a programmable stage of a pipeline.
#[derive(Clone, Debug)]
pub enum ProgrammableStageError {
    /// Unable to find an entry point matching the specified execution model.
    MissingEntryPoint(spirv::ExecutionModel),
    /// Error matching a global binding against the pipeline layout.
    Binding {
        set: u32,
        binding: u32,
        error: BindingError,
    },
    /// Error matching the stage input against the previous stage outputs.
    Input {
        location: wgt::ShaderLocation,
        error: InputError,
    },
}

fn validate_binding(
    module: &naga::Module,
    var: &naga::GlobalVariable,
    entry: &BindGroupLayoutEntry,
    usage: naga::GlobalUse,
) -> Result<(), BindingError> {
    let mut ty_inner = &module.types[var.ty].inner;
    //TODO: change naga's IR to avoid a pointer here
    if let naga::TypeInner::Pointer { base, class: _ } = *ty_inner {
        ty_inner = &module.types[base].inner;
    }
    let allowed_usage = match *ty_inner {
        naga::TypeInner::Struct { .. } => match entry.ty {
            BindingType::UniformBuffer => naga::GlobalUse::LOAD,
            BindingType::StorageBuffer => naga::GlobalUse::all(),
            BindingType::ReadonlyStorageBuffer => naga::GlobalUse::LOAD,
            _ => return Err(BindingError::WrongType),
        },
        naga::TypeInner::Sampler => match entry.ty {
            BindingType::Sampler | BindingType::ComparisonSampler => naga::GlobalUse::empty(),
            _ => return Err(BindingError::WrongType),
        },
        naga::TypeInner::Image { base, dim, flags } => {
            if entry.multisampled != flags.contains(naga::ImageFlags::MULTISAMPLED) {
                return Err(BindingError::WrongTextureMultisampled);
            }
            if flags.contains(naga::ImageFlags::ARRAYED) {
                match (dim, entry.view_dimension) {
                    (spirv::Dim::Dim2D, wgt::TextureViewDimension::D2Array) => (),
                    (spirv::Dim::DimCube, wgt::TextureViewDimension::CubeArray) => (),
                    _ => {
                        return Err(BindingError::WrongTextureViewDimension {
                            dim,
                            is_array: true,
                        })
                    }
                }
            } else {
                match (dim, entry.view_dimension) {
                    (spirv::Dim::Dim1D, wgt::TextureViewDimension::D1) => (),
                    (spirv::Dim::Dim2D, wgt::TextureViewDimension::D2) => (),
                    (spirv::Dim::Dim3D, wgt::TextureViewDimension::D3) => (),
                    (spirv::Dim::DimCube, wgt::TextureViewDimension::Cube) => (),
                    _ => {
                        return Err(BindingError::WrongTextureViewDimension {
                            dim,
                            is_array: false,
                        })
                    }
                }
            }
            let (allowed_usage, is_sampled) = match entry.ty {
                BindingType::SampledTexture => {
                    let expected_scalar_kind = match entry.texture_component_type {
                        wgt::TextureComponentType::Float => naga::ScalarKind::Float,
                        wgt::TextureComponentType::Sint => naga::ScalarKind::Sint,
                        wgt::TextureComponentType::Uint => naga::ScalarKind::Uint,
                    };
                    match module.types[base].inner {
                        naga::TypeInner::Scalar { kind, .. }
                        | naga::TypeInner::Vector { kind, .. }
                            if kind == expected_scalar_kind => {}
                        naga::TypeInner::Scalar { kind, .. }
                        | naga::TypeInner::Vector { kind, .. } => {
                            return Err(BindingError::WrongTextureComponentType(Some(kind)))
                        }
                        _ => return Err(BindingError::WrongTextureComponentType(None)),
                    };
                    (naga::GlobalUse::LOAD, true)
                }
                BindingType::ReadonlyStorageTexture => {
                    //TODO: check entry.storage_texture_format
                    (naga::GlobalUse::LOAD, false)
                }
                BindingType::WriteonlyStorageTexture => (naga::GlobalUse::STORE, false),
                _ => return Err(BindingError::WrongType),
            };
            if is_sampled != flags.contains(naga::ImageFlags::SAMPLED) {
                return Err(BindingError::WrongTextureSampled);
            }
            allowed_usage
        }
        _ => return Err(BindingError::WrongType),
    };
    if allowed_usage.contains(usage) {
        Ok(())
    } else {
        Err(BindingError::WrongUsage(usage))
    }
}

fn is_sub_type(sub: &naga::TypeInner, provided: &naga::TypeInner) -> bool {
    use naga::TypeInner as Ti;

    match (sub, provided) {
        (
            &Ti::Scalar {
                kind: k0,
                width: w0,
            },
            &Ti::Scalar {
                kind: k1,
                width: w1,
            },
        ) => k0 == k1 && w0 <= w1,
        (
            &Ti::Scalar {
                kind: k0,
                width: w0,
            },
            &Ti::Vector {
                size: _,
                kind: k1,
                width: w1,
            },
        ) => k0 == k1 && w0 <= w1,
        (
            &Ti::Vector {
                size: s0,
                kind: k0,
                width: w0,
            },
            &Ti::Vector {
                size: s1,
                kind: k1,
                width: w1,
            },
        ) => s0 as u8 <= s1 as u8 && k0 == k1 && w0 <= w1,
        (
            &Ti::Matrix {
                columns: c0,
                rows: r0,
                kind: k0,
                width: w0,
            },
            &Ti::Matrix {
                columns: c1,
                rows: r1,
                kind: k1,
                width: w1,
            },
        ) => c0 == c1 && r0 == r1 && k0 == k1 && w0 <= w1,
        (&Ti::Struct { members: ref m0 }, &Ti::Struct { members: ref m1 }) => m0 == m1,
        _ => false,
    }
}

pub(crate) enum MaybeOwned<'a, T> {
    Owned(T),
    Borrowed(&'a T),
}

impl<'a, T> std::ops::Deref for MaybeOwned<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        match *self {
            MaybeOwned::Owned(ref value) => value,
            MaybeOwned::Borrowed(value) => value,
        }
    }
}

pub(crate) fn construct_vertex_format<'a>(
    format: wgt::VertexFormat,
) -> MaybeOwned<'a, naga::TypeInner> {
    use naga::TypeInner as Ti;
    use wgt::VertexFormat as Vf;
    MaybeOwned::Owned(match format {
        Vf::Uchar2 => Ti::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Uint,
            width: 8,
        },
        Vf::Uchar4 => Ti::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Uint,
            width: 8,
        },
        Vf::Char2 => Ti::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Sint,
            width: 8,
        },
        Vf::Char4 => Ti::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Sint,
            width: 8,
        },
        Vf::Uchar2Norm => Ti::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Float,
            width: 8,
        },
        Vf::Uchar4Norm => Ti::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Float,
            width: 8,
        },
        Vf::Char2Norm => Ti::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Float,
            width: 8,
        },
        Vf::Char4Norm => Ti::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Float,
            width: 8,
        },
        Vf::Ushort2 => Ti::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Uint,
            width: 16,
        },
        Vf::Ushort4 => Ti::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Uint,
            width: 16,
        },
        Vf::Short2 => Ti::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Sint,
            width: 16,
        },
        Vf::Short4 => Ti::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Sint,
            width: 16,
        },
        Vf::Ushort2Norm | Vf::Short2Norm | Vf::Half2 => Ti::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Float,
            width: 16,
        },
        Vf::Ushort4Norm | Vf::Short4Norm | Vf::Half4 => Ti::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Float,
            width: 16,
        },
        Vf::Float => Ti::Scalar {
            kind: naga::ScalarKind::Float,
            width: 32,
        },
        Vf::Float2 => Ti::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Float,
            width: 32,
        },
        Vf::Float3 => Ti::Vector {
            size: naga::VectorSize::Tri,
            kind: naga::ScalarKind::Float,
            width: 32,
        },
        Vf::Float4 => Ti::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Float,
            width: 32,
        },
        Vf::Uint => Ti::Scalar {
            kind: naga::ScalarKind::Uint,
            width: 32,
        },
        Vf::Uint2 => Ti::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Uint,
            width: 32,
        },
        Vf::Uint3 => Ti::Vector {
            size: naga::VectorSize::Tri,
            kind: naga::ScalarKind::Uint,
            width: 32,
        },
        Vf::Uint4 => Ti::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Uint,
            width: 32,
        },
        Vf::Int => Ti::Scalar {
            kind: naga::ScalarKind::Sint,
            width: 32,
        },
        Vf::Int2 => Ti::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Sint,
            width: 32,
        },
        Vf::Int3 => Ti::Vector {
            size: naga::VectorSize::Tri,
            kind: naga::ScalarKind::Sint,
            width: 32,
        },
        Vf::Int4 => Ti::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Sint,
            width: 32,
        },
    })
}

/// Return true if the fragment `format` is covered by the provided `output`.
pub(crate) fn check_texture_format(
    format: wgt::TextureFormat,
    output: &MaybeOwned<naga::TypeInner>,
) -> bool {
    use naga::ScalarKind as Sk;
    use wgt::TextureFormat as Tf;

    let (components, kind, width) = match *&**output {
        naga::TypeInner::Scalar { kind, width } => (1, kind, width),
        naga::TypeInner::Vector { size, kind, width } => (size as u8, kind, width),
        _ => return false,
    };
    let (req_components, req_kind, req_width) = match format {
        Tf::R8Unorm | Tf::R8Snorm => (1, Sk::Float, 8),
        Tf::R8Uint => (1, Sk::Uint, 8),
        Tf::R8Sint => (1, Sk::Sint, 8),
        Tf::R16Uint => (1, Sk::Uint, 16),
        Tf::R16Sint => (1, Sk::Sint, 16),
        Tf::R16Float => (1, Sk::Float, 16),
        Tf::Rg8Unorm | Tf::Rg8Snorm => (2, Sk::Float, 8),
        Tf::Rg8Uint => (2, Sk::Uint, 8),
        Tf::Rg8Sint => (2, Sk::Sint, 8),
        Tf::R32Uint => (1, Sk::Uint, 32),
        Tf::R32Sint => (1, Sk::Sint, 32),
        Tf::R32Float => (1, Sk::Float, 32),
        Tf::Rg16Uint => (2, Sk::Uint, 16),
        Tf::Rg16Sint => (2, Sk::Sint, 16),
        Tf::Rg16Float => (2, Sk::Float, 16),
        Tf::Rgba8Unorm
        | Tf::Rgba8UnormSrgb
        | Tf::Rgba8Snorm
        | Tf::Bgra8Unorm
        | Tf::Bgra8UnormSrgb => (4, Sk::Float, 8),
        Tf::Rgba8Uint => (4, Sk::Uint, 8),
        Tf::Rgba8Sint => (4, Sk::Sint, 8),
        Tf::Rgb10a2Unorm => (4, Sk::Float, 10),
        Tf::Rg11b10Float => (3, Sk::Float, 11),
        Tf::Rg32Uint => (2, Sk::Uint, 32),
        Tf::Rg32Sint => (2, Sk::Sint, 32),
        Tf::Rg32Float => (2, Sk::Float, 32),
        Tf::Rgba16Uint => (4, Sk::Uint, 16),
        Tf::Rgba16Sint => (4, Sk::Sint, 16),
        Tf::Rgba16Float => (4, Sk::Float, 16),
        Tf::Rgba32Uint => (4, Sk::Uint, 32),
        Tf::Rgba32Sint => (4, Sk::Sint, 32),
        Tf::Rgba32Float => (4, Sk::Float, 32),
        Tf::Depth32Float | Tf::Depth24Plus | Tf::Depth24PlusStencil8 => return false,
    };

    components >= req_components && kind == req_kind && width >= req_width
}

pub(crate) type StageInterface<'a> =
    FastHashMap<wgt::ShaderLocation, MaybeOwned<'a, naga::TypeInner>>;

pub(crate) fn validate_stage<'a>(
    module: &'a naga::Module,
    group_layouts: &[&BindEntryMap],
    entry_point_name: &str,
    execution_model: spirv::ExecutionModel,
    inputs: StageInterface<'a>,
) -> Result<StageInterface<'a>, ProgrammableStageError> {
    // Since a shader module can have multiple entry points with the same name,
    // we need to look for one with the right execution model.
    let entry_point = module
        .entry_points
        .iter()
        .find(|entry_point| {
            entry_point.name == entry_point_name && entry_point.exec_model == execution_model
        })
        .ok_or(ProgrammableStageError::MissingEntryPoint(execution_model))?;
    let stage_bit = match execution_model {
        spirv::ExecutionModel::Vertex => wgt::ShaderStage::VERTEX,
        spirv::ExecutionModel::Fragment => wgt::ShaderStage::FRAGMENT,
        spirv::ExecutionModel::GLCompute => wgt::ShaderStage::COMPUTE,
        // the entry point wouldn't match otherwise
        _ => unreachable!(),
    };

    let function = &module.functions[entry_point.function];
    let mut outputs = StageInterface::default();
    for ((_, var), &usage) in module.global_variables.iter().zip(&function.global_usage) {
        if usage.is_empty() {
            continue;
        }
        match var.binding {
            Some(naga::Binding::Descriptor { set, binding }) => {
                let result = group_layouts
                    .get(set as usize)
                    .and_then(|map| map.get(&binding))
                    .ok_or(BindingError::Missing)
                    .and_then(|entry| {
                        if entry.visibility.contains(stage_bit) {
                            Ok(entry)
                        } else {
                            Err(BindingError::Invisible)
                        }
                    })
                    .and_then(|entry| validate_binding(module, var, entry, usage));
                if let Err(error) = result {
                    return Err(ProgrammableStageError::Binding {
                        set,
                        binding,
                        error,
                    });
                }
            }
            Some(naga::Binding::Location(location)) => {
                let mut ty = &module.types[var.ty].inner;
                //TODO: change naga's IR to not have pointer for varyings
                if let naga::TypeInner::Pointer { base, class: _ } = *ty {
                    ty = &module.types[base].inner;
                }
                if usage.contains(naga::GlobalUse::STORE) {
                    outputs.insert(location, MaybeOwned::Borrowed(ty));
                } else {
                    let result =
                        inputs
                            .get(&location)
                            .ok_or(InputError::Missing)
                            .and_then(|provided| {
                                if is_sub_type(ty, provided) {
                                    Ok(())
                                } else {
                                    Err(InputError::WrongType)
                                }
                            });
                    if let Err(error) = result {
                        return Err(ProgrammableStageError::Input { location, error });
                    }
                }
            }
            _ => {}
        }
    }
    Ok(outputs)
}

#[repr(C)]
#[derive(Debug)]
pub struct ComputePipelineDescriptor {
    pub layout: PipelineLayoutId,
    pub compute_stage: ProgrammableStageDescriptor,
}

#[derive(Clone, Debug)]
pub enum ComputePipelineError {
    Stage(ProgrammableStageError),
}

#[derive(Debug)]
pub struct ComputePipeline<B: hal::Backend> {
    pub(crate) raw: B::ComputePipeline,
    pub(crate) layout_id: Stored<PipelineLayoutId>,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) life_guard: LifeGuard,
}

impl<B: hal::Backend> Borrow<RefCount> for ComputePipeline<B> {
    fn borrow(&self) -> &RefCount {
        self.life_guard.ref_count.as_ref().unwrap()
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct RenderPipelineDescriptor {
    pub layout: PipelineLayoutId,
    pub vertex_stage: ProgrammableStageDescriptor,
    pub fragment_stage: *const ProgrammableStageDescriptor,
    pub primitive_topology: PrimitiveTopology,
    pub rasterization_state: *const RasterizationStateDescriptor,
    pub color_states: *const ColorStateDescriptor,
    pub color_states_length: usize,
    pub depth_stencil_state: *const DepthStencilStateDescriptor,
    pub vertex_state: VertexStateDescriptor,
    pub sample_count: u32,
    pub sample_mask: u32,
    pub alpha_to_coverage_enabled: bool,
}

bitflags::bitflags! {
    #[repr(transparent)]
    pub struct PipelineFlags: u32 {
        const BLEND_COLOR = 1;
        const STENCIL_REFERENCE = 2;
        const DEPTH_STENCIL_READ_ONLY = 4;
    }
}

#[derive(Debug)]
pub struct RenderPipeline<B: hal::Backend> {
    pub(crate) raw: B::GraphicsPipeline,
    pub(crate) layout_id: Stored<PipelineLayoutId>,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) pass_context: RenderPassContext,
    pub(crate) flags: PipelineFlags,
    pub(crate) index_format: IndexFormat,
    pub(crate) sample_count: u8,
    pub(crate) vertex_strides: Vec<(BufferAddress, InputStepMode)>,
    pub(crate) life_guard: LifeGuard,
}

impl<B: hal::Backend> Borrow<RefCount> for RenderPipeline<B> {
    fn borrow(&self) -> &RefCount {
        self.life_guard.ref_count.as_ref().unwrap()
    }
}
