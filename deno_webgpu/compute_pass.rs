// Copyright 2018-2024 the Deno authors. All rights reserved. MIT license.

use deno_core::error::AnyError;
use deno_core::op2;
use deno_core::OpState;
use deno_core::Resource;
use deno_core::ResourceId;
use std::borrow::Cow;
use std::cell::RefCell;

use super::error::WebGpuResult;

pub(crate) struct WebGpuComputePass(
    pub(crate) RefCell<Box<dyn wgpu_core::command::DynComputePass>>,
);
impl Resource for WebGpuComputePass {
    fn name(&self) -> Cow<str> {
        "webGPUComputePass".into()
    }
}

#[op2]
#[serde]
pub fn op_webgpu_compute_pass_set_pipeline(
    state: &mut OpState,
    #[smi] compute_pass_rid: ResourceId,
    #[smi] pipeline: ResourceId,
) -> Result<WebGpuResult, AnyError> {
    let compute_pipeline_resource = state
        .resource_table
        .get::<super::pipeline::WebGpuComputePipeline>(pipeline)?;
    let compute_pass_resource = state
        .resource_table
        .get::<WebGpuComputePass>(compute_pass_rid)?;

    compute_pass_resource
        .0
        .borrow_mut()
        .set_pipeline(state.borrow(), compute_pipeline_resource.1)?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_compute_pass_dispatch_workgroups(
    state: &mut OpState,
    #[smi] compute_pass_rid: ResourceId,
    x: u32,
    y: u32,
    z: u32,
) -> Result<WebGpuResult, AnyError> {
    let compute_pass_resource = state
        .resource_table
        .get::<WebGpuComputePass>(compute_pass_rid)?;

    compute_pass_resource
        .0
        .borrow_mut()
        .dispatch_workgroups(state.borrow(), x, y, z)?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_compute_pass_dispatch_workgroups_indirect(
    state: &mut OpState,
    #[smi] compute_pass_rid: ResourceId,
    #[smi] indirect_buffer: ResourceId,
    #[number] indirect_offset: u64,
) -> Result<WebGpuResult, AnyError> {
    let buffer_resource = state
        .resource_table
        .get::<super::buffer::WebGpuBuffer>(indirect_buffer)?;
    let compute_pass_resource = state
        .resource_table
        .get::<WebGpuComputePass>(compute_pass_rid)?;

    compute_pass_resource
        .0
        .borrow_mut()
        .dispatch_workgroups_indirect(state.borrow(), buffer_resource.1, indirect_offset)?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_compute_pass_end(
    state: &mut OpState,
    #[smi] compute_pass_rid: ResourceId,
) -> Result<WebGpuResult, AnyError> {
    let compute_pass_resource = state
        .resource_table
        .take::<WebGpuComputePass>(compute_pass_rid)?;

    compute_pass_resource.0.borrow_mut().end(state.borrow())?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_compute_pass_set_bind_group(
    state: &mut OpState,
    #[smi] compute_pass_rid: ResourceId,
    index: u32,
    #[smi] bind_group: ResourceId,
    #[buffer] dynamic_offsets_data: &[u32],
    #[number] dynamic_offsets_data_start: usize,
    #[number] dynamic_offsets_data_length: usize,
) -> Result<WebGpuResult, AnyError> {
    let bind_group_resource = state
        .resource_table
        .get::<super::binding::WebGpuBindGroup>(bind_group)?;
    let compute_pass_resource = state
        .resource_table
        .get::<WebGpuComputePass>(compute_pass_rid)?;

    let start = dynamic_offsets_data_start;
    let len = dynamic_offsets_data_length;

    // Assert that length and start are both in bounds
    assert!(start <= dynamic_offsets_data.len());
    assert!(len <= dynamic_offsets_data.len() - start);

    let dynamic_offsets_data: &[u32] = &dynamic_offsets_data[start..start + len];

    compute_pass_resource.0.borrow_mut().set_bind_group(
        state.borrow(),
        index,
        bind_group_resource.1,
        dynamic_offsets_data,
    )?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_compute_pass_push_debug_group(
    state: &mut OpState,
    #[smi] compute_pass_rid: ResourceId,
    #[string] group_label: &str,
) -> Result<WebGpuResult, AnyError> {
    let compute_pass_resource = state
        .resource_table
        .get::<WebGpuComputePass>(compute_pass_rid)?;

    compute_pass_resource.0.borrow_mut().push_debug_group(
        state.borrow(),
        group_label,
        0, // wgpu#975
    )?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_compute_pass_pop_debug_group(
    state: &mut OpState,
    #[smi] compute_pass_rid: ResourceId,
) -> Result<WebGpuResult, AnyError> {
    let compute_pass_resource = state
        .resource_table
        .get::<WebGpuComputePass>(compute_pass_rid)?;

    compute_pass_resource
        .0
        .borrow_mut()
        .pop_debug_group(state.borrow())?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_compute_pass_insert_debug_marker(
    state: &mut OpState,
    #[smi] compute_pass_rid: ResourceId,
    #[string] marker_label: &str,
) -> Result<WebGpuResult, AnyError> {
    let compute_pass_resource = state
        .resource_table
        .get::<WebGpuComputePass>(compute_pass_rid)?;

    compute_pass_resource.0.borrow_mut().insert_debug_marker(
        state.borrow(),
        marker_label,
        0, // wgpu#975
    )?;

    Ok(WebGpuResult::empty())
}
