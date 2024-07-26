// Copyright 2018-2024 the Deno authors. All rights reserved. MIT license.

use deno_core::error::type_error;
use deno_core::error::AnyError;
use deno_core::op2;
use deno_core::OpState;
use deno_core::Resource;
use deno_core::ResourceId;
use serde::Deserialize;
use std::borrow::Cow;
use std::cell::RefCell;

use super::error::WebGpuResult;

pub(crate) struct WebGpuRenderPass(pub(crate) RefCell<Box<dyn wgpu_core::command::DynRenderPass>>);
impl Resource for WebGpuRenderPass {
    fn name(&self) -> Cow<str> {
        "webGPURenderPass".into()
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderPassSetViewportArgs {
    render_pass_rid: ResourceId,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    min_depth: f32,
    max_depth: f32,
}

#[op2]
#[serde]
pub fn op_webgpu_render_pass_set_viewport(
    state: &mut OpState,
    #[serde] args: RenderPassSetViewportArgs,
) -> Result<WebGpuResult, AnyError> {
    let render_pass_resource = state
        .resource_table
        .get::<WebGpuRenderPass>(args.render_pass_rid)?;

    render_pass_resource.0.borrow_mut().set_viewport(
        state.borrow(),
        args.x,
        args.y,
        args.width,
        args.height,
        args.min_depth,
        args.max_depth,
    )?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_render_pass_set_scissor_rect(
    state: &mut OpState,
    #[smi] render_pass_rid: ResourceId,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<WebGpuResult, AnyError> {
    let render_pass_resource = state
        .resource_table
        .get::<WebGpuRenderPass>(render_pass_rid)?;

    render_pass_resource
        .0
        .borrow_mut()
        .set_scissor_rect(state.borrow(), x, y, width, height)?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_render_pass_set_blend_constant(
    state: &mut OpState,
    #[smi] render_pass_rid: ResourceId,
    #[serde] color: wgpu_types::Color,
) -> Result<WebGpuResult, AnyError> {
    let render_pass_resource = state
        .resource_table
        .get::<WebGpuRenderPass>(render_pass_rid)?;

    render_pass_resource
        .0
        .borrow_mut()
        .set_blend_constant(state.borrow(), color)?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_render_pass_set_stencil_reference(
    state: &mut OpState,
    #[smi] render_pass_rid: ResourceId,
    reference: u32,
) -> Result<WebGpuResult, AnyError> {
    let render_pass_resource = state
        .resource_table
        .get::<WebGpuRenderPass>(render_pass_rid)?;

    render_pass_resource
        .0
        .borrow_mut()
        .set_stencil_reference(state.borrow(), reference)?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_render_pass_begin_occlusion_query(
    state: &mut OpState,
    #[smi] render_pass_rid: ResourceId,
    query_index: u32,
) -> Result<WebGpuResult, AnyError> {
    let render_pass_resource = state
        .resource_table
        .get::<WebGpuRenderPass>(render_pass_rid)?;

    render_pass_resource
        .0
        .borrow_mut()
        .begin_occlusion_query(state.borrow(), query_index)?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_render_pass_end_occlusion_query(
    state: &mut OpState,
    #[smi] render_pass_rid: ResourceId,
) -> Result<WebGpuResult, AnyError> {
    let render_pass_resource = state
        .resource_table
        .get::<WebGpuRenderPass>(render_pass_rid)?;

    render_pass_resource
        .0
        .borrow_mut()
        .end_occlusion_query(state.borrow())?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_render_pass_execute_bundles(
    state: &mut OpState,
    #[smi] render_pass_rid: ResourceId,
    #[serde] bundles: Vec<u32>,
) -> Result<WebGpuResult, AnyError> {
    let bundles = bundles
        .iter()
        .map(|rid| {
            let render_bundle_resource = state
                .resource_table
                .get::<super::bundle::WebGpuRenderBundle>(*rid)?;
            Ok(render_bundle_resource.1)
        })
        .collect::<Result<Vec<_>, AnyError>>()?;

    let render_pass_resource = state
        .resource_table
        .get::<WebGpuRenderPass>(render_pass_rid)?;

    render_pass_resource
        .0
        .borrow_mut()
        .execute_bundles(state.borrow(), &bundles)?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_render_pass_end(
    state: &mut OpState,
    #[smi] render_pass_rid: ResourceId,
) -> Result<WebGpuResult, AnyError> {
    let render_pass_resource = state
        .resource_table
        .take::<WebGpuRenderPass>(render_pass_rid)?;

    render_pass_resource.0.borrow_mut().end(state.borrow())?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_render_pass_set_bind_group(
    state: &mut OpState,
    #[smi] render_pass_rid: ResourceId,
    index: u32,
    bind_group: u32,
    #[buffer] dynamic_offsets_data: &[u32],
    #[number] dynamic_offsets_data_start: usize,
    #[number] dynamic_offsets_data_length: usize,
) -> Result<WebGpuResult, AnyError> {
    let bind_group_resource = state
        .resource_table
        .get::<super::binding::WebGpuBindGroup>(bind_group)?;
    let render_pass_resource = state
        .resource_table
        .get::<WebGpuRenderPass>(render_pass_rid)?;

    let start = dynamic_offsets_data_start;
    let len = dynamic_offsets_data_length;

    // Assert that length and start are both in bounds
    assert!(start <= dynamic_offsets_data.len());
    assert!(len <= dynamic_offsets_data.len() - start);

    let dynamic_offsets_data: &[u32] = &dynamic_offsets_data[start..start + len];

    render_pass_resource.0.borrow_mut().set_bind_group(
        state.borrow(),
        index,
        bind_group_resource.1,
        dynamic_offsets_data,
    )?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_render_pass_push_debug_group(
    state: &mut OpState,
    #[smi] render_pass_rid: ResourceId,
    #[string] group_label: &str,
) -> Result<WebGpuResult, AnyError> {
    let render_pass_resource = state
        .resource_table
        .get::<WebGpuRenderPass>(render_pass_rid)?;

    render_pass_resource.0.borrow_mut().push_debug_group(
        state.borrow(),
        group_label,
        0, // wgpu#975
    )?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_render_pass_pop_debug_group(
    state: &mut OpState,
    #[smi] render_pass_rid: ResourceId,
) -> Result<WebGpuResult, AnyError> {
    let render_pass_resource = state
        .resource_table
        .get::<WebGpuRenderPass>(render_pass_rid)?;

    render_pass_resource
        .0
        .borrow_mut()
        .pop_debug_group(state.borrow())?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_render_pass_insert_debug_marker(
    state: &mut OpState,
    #[smi] render_pass_rid: ResourceId,
    #[string] marker_label: &str,
) -> Result<WebGpuResult, AnyError> {
    let render_pass_resource = state
        .resource_table
        .get::<WebGpuRenderPass>(render_pass_rid)?;

    render_pass_resource.0.borrow_mut().insert_debug_marker(
        state.borrow(),
        marker_label,
        0, // wgpu#975
    )?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_render_pass_set_pipeline(
    state: &mut OpState,
    #[smi] render_pass_rid: ResourceId,
    pipeline: u32,
) -> Result<WebGpuResult, AnyError> {
    let render_pipeline_resource = state
        .resource_table
        .get::<super::pipeline::WebGpuRenderPipeline>(pipeline)?;
    let render_pass_resource = state
        .resource_table
        .get::<WebGpuRenderPass>(render_pass_rid)?;

    render_pass_resource
        .0
        .borrow_mut()
        .set_pipeline(state.borrow(), render_pipeline_resource.1)?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_render_pass_set_index_buffer(
    state: &mut OpState,
    #[smi] render_pass_rid: ResourceId,
    buffer: u32,
    #[serde] index_format: wgpu_types::IndexFormat,
    #[number] offset: u64,
    #[number] size: Option<u64>,
) -> Result<WebGpuResult, AnyError> {
    let buffer_resource = state
        .resource_table
        .get::<super::buffer::WebGpuBuffer>(buffer)?;
    let render_pass_resource = state
        .resource_table
        .get::<WebGpuRenderPass>(render_pass_rid)?;

    let size = if let Some(size) = size {
        Some(
            std::num::NonZeroU64::new(size)
                .ok_or_else(|| type_error("size must be larger than 0"))?,
        )
    } else {
        None
    };

    render_pass_resource.0.borrow_mut().set_index_buffer(
        state.borrow(),
        buffer_resource.1,
        index_format,
        offset,
        size,
    )?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_render_pass_set_vertex_buffer(
    state: &mut OpState,
    #[smi] render_pass_rid: ResourceId,
    slot: u32,
    buffer: u32,
    #[number] offset: u64,
    #[number] size: Option<u64>,
) -> Result<WebGpuResult, AnyError> {
    let buffer_resource = state
        .resource_table
        .get::<super::buffer::WebGpuBuffer>(buffer)?;
    let render_pass_resource = state
        .resource_table
        .get::<WebGpuRenderPass>(render_pass_rid)?;

    let size = if let Some(size) = size {
        Some(
            std::num::NonZeroU64::new(size)
                .ok_or_else(|| type_error("size must be larger than 0"))?,
        )
    } else {
        None
    };

    render_pass_resource.0.borrow_mut().set_vertex_buffer(
        state.borrow(),
        slot,
        buffer_resource.1,
        offset,
        size,
    )?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_render_pass_draw(
    state: &mut OpState,
    #[smi] render_pass_rid: ResourceId,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) -> Result<WebGpuResult, AnyError> {
    let render_pass_resource = state
        .resource_table
        .get::<WebGpuRenderPass>(render_pass_rid)?;

    render_pass_resource.0.borrow_mut().draw(
        state.borrow(),
        vertex_count,
        instance_count,
        first_vertex,
        first_instance,
    )?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_render_pass_draw_indexed(
    state: &mut OpState,
    #[smi] render_pass_rid: ResourceId,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) -> Result<WebGpuResult, AnyError> {
    let render_pass_resource = state
        .resource_table
        .get::<WebGpuRenderPass>(render_pass_rid)?;

    render_pass_resource.0.borrow_mut().draw_indexed(
        state.borrow(),
        index_count,
        instance_count,
        first_index,
        base_vertex,
        first_instance,
    )?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_render_pass_draw_indirect(
    state: &mut OpState,
    #[smi] render_pass_rid: ResourceId,
    indirect_buffer: u32,
    #[number] indirect_offset: u64,
) -> Result<WebGpuResult, AnyError> {
    let buffer_resource = state
        .resource_table
        .get::<super::buffer::WebGpuBuffer>(indirect_buffer)?;
    let render_pass_resource = state
        .resource_table
        .get::<WebGpuRenderPass>(render_pass_rid)?;

    render_pass_resource.0.borrow_mut().draw_indirect(
        state.borrow(),
        buffer_resource.1,
        indirect_offset,
    )?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_render_pass_draw_indexed_indirect(
    state: &mut OpState,
    #[smi] render_pass_rid: ResourceId,
    indirect_buffer: u32,
    #[number] indirect_offset: u64,
) -> Result<WebGpuResult, AnyError> {
    let buffer_resource = state
        .resource_table
        .get::<super::buffer::WebGpuBuffer>(indirect_buffer)?;
    let render_pass_resource = state
        .resource_table
        .get::<WebGpuRenderPass>(render_pass_rid)?;

    render_pass_resource.0.borrow_mut().draw_indexed_indirect(
        state.borrow(),
        buffer_resource.1,
        indirect_offset,
    )?;

    Ok(WebGpuResult::empty())
}
