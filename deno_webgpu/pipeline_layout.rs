// Copyright 2018-2025 the Deno authors. MIT license.

use deno_core::cppgc::Ptr;
use deno_core::op2;
use deno_core::webidl::Nullable;
use deno_core::webidl::WebIdlInterfaceConverter;
use deno_core::GarbageCollected;
use deno_core::WebIDL;
use std::sync::Arc;

use crate::error::GPUGenericError;

pub struct GPUPipelineLayout {
  pub wgpu_pipeline_layout: Arc<wgpu_core::binding_model::PipelineLayout>,
  pub label: String,
}

impl WebIdlInterfaceConverter for GPUPipelineLayout {
  const NAME: &'static str = "GPUPipelineLayout";
}

impl GarbageCollected for GPUPipelineLayout {
  fn get_name(&self) -> &'static std::ffi::CStr {
    c"GPUPipelineLayout"
  }
}

#[op2]
impl GPUPipelineLayout {
  #[constructor]
  #[cppgc]
  fn constructor(_: bool) -> Result<GPUPipelineLayout, GPUGenericError> {
    Err(GPUGenericError::InvalidConstructor)
  }

  #[getter]
  #[string]
  fn label(&self) -> String {
    self.label.clone()
  }
  #[setter]
  #[string]
  fn label(&self, #[webidl] _label: String) {
    // TODO(@crowlKats): no-op, needs wpgu to implement changing the label
  }
}

#[derive(WebIDL)]
#[webidl(dictionary)]
pub(crate) struct GPUPipelineLayoutDescriptor {
  #[webidl(default = String::new())]
  pub label: String,

  pub bind_group_layouts:
    Vec<Nullable<Ptr<super::bind_group_layout::GPUBindGroupLayout>>>,
  #[webidl(default = 0)]
  pub immediate_size: u32,
}
