/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    binding_model::{BindEntryMap, BindGroupLayoutEntry, BindingType},
    device::RenderPassContext,
    id::{DeviceId, PipelineLayoutId, ShaderModuleId},
    LifeGuard, RawString, RefCount, Stored, U32Array,
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

/// Errors produced when validating a programmable stage of a pipeline.
#[derive(Clone, Debug)]
pub enum ProgrammableStageError {
    /// Unable to find an entry point matching the specified execution model.
    MissingEntryPoint(spirv::ExecutionModel),
    /// Error matching a global binding to the pipeline layout.
    Binding {
        set: u32,
        binding: u32,
        error: BindingError,
    },
}

fn validate_binding(
    module: &naga::Module,
    var: &naga::GlobalVariable,
    entry: &BindGroupLayoutEntry,
    usage: naga::GlobalUse,
) -> Result<(), BindingError> {
    let allowed_usage = match module.types[var.ty].inner {
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
                            if kind == expected_scalar_kind =>
                        {
                            ()
                        }
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

pub(crate) fn validate_stage(
    module: &naga::Module,
    group_layouts: &[&BindEntryMap],
    entry_point_name: &str,
    execution_model: spirv::ExecutionModel,
) -> Result<(), ProgrammableStageError> {
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
            _ => {} //TODO
        }
    }
    Ok(())
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
