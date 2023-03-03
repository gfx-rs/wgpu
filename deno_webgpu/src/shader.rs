// Copyright 2018-2023 the Deno authors. All rights reserved. MIT license.

use deno_core::error::AnyError;
use deno_core::op;
use deno_core::OpState;
use deno_core::Resource;
use deno_core::ResourceId;
use std::borrow::Cow;
use std::rc::Rc;

use super::error::WebGpuResult;

pub(crate) struct WebGpuShaderModule(
    pub(crate) super::Instance,
    pub(crate) wgpu_core::id::ShaderModuleId,
);
impl Resource for WebGpuShaderModule {
    fn name(&self) -> Cow<str> {
        "webGPUShaderModule".into()
    }

    fn close(self: Rc<Self>) {
        let instance = &self.0;
        gfx_select!(self.1 => instance.shader_module_drop(self.1));
    }
}

#[op]
pub fn op_webgpu_create_shader_module(
    state: &mut OpState,
    device_rid: ResourceId,
    label: Option<String>,
    code: String,
) -> Result<WebGpuResult, AnyError> {
    let instance = state.borrow::<super::Instance>();
    let device_resource = state
        .resource_table
        .get::<super::WebGpuDevice>(device_rid)?;
    let device = device_resource.1;

    let source = wgpu_core::pipeline::ShaderModuleSource::Wgsl(Cow::from(code));

    let descriptor = wgpu_core::pipeline::ShaderModuleDescriptor {
        label: label.map(Cow::from),
        shader_bound_checks: wgpu_types::ShaderBoundChecks::default(),
    };

    gfx_put!(device => instance.device_create_shader_module(
    device,
    &descriptor,
    source,
    ()
  ) => state, WebGpuShaderModule)
}
