/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    device::RenderPassContext,
    id::{PipelineLayoutId, ShaderModuleId},
    RawString,
    U32Array
};
use wgt::{BufferAddress, ColorStateDescriptor, DepthStencilStateDescriptor, IndexFormat, InputStepMode, PrimitiveTopology, RasterizationStateDescriptor, VertexAttributeDescriptor};

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

#[repr(C)]
#[derive(Debug)]
pub struct ProgrammableStageDescriptor {
    pub module: ShaderModuleId,
    pub entry_point: RawString,
}

#[repr(C)]
#[derive(Debug)]
pub struct ComputePipelineDescriptor {
    pub layout: PipelineLayoutId,
    pub compute_stage: ProgrammableStageDescriptor,
}

#[derive(Debug)]
pub struct ComputePipeline<B: hal::Backend> {
    pub(crate) raw: B::ComputePipeline,
    pub(crate) layout_id: PipelineLayoutId,
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
    }
}

#[derive(Debug)]
pub struct RenderPipeline<B: hal::Backend> {
    pub(crate) raw: B::GraphicsPipeline,
    pub(crate) layout_id: PipelineLayoutId,
    pub(crate) pass_context: RenderPassContext,
    pub(crate) flags: PipelineFlags,
    pub(crate) index_format: IndexFormat,
    pub(crate) sample_count: u8,
    pub(crate) vertex_strides: Vec<(BufferAddress, InputStepMode)>,
}
