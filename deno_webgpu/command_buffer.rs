// Copyright 2018-2025 the Deno authors. MIT license.

use std::sync::Arc;

use deno_core::op2;
use deno_core::GarbageCollected;
use deno_core::WebIDL;

use crate::error::GPUGenericError;

pub struct GPUCommandBuffer {
  pub wgpu_command_buffer: Arc<wgpu_core::command::CommandBuffer>,
  pub label: String,
}

impl deno_core::webidl::WebIdlInterfaceConverter for GPUCommandBuffer {
  const NAME: &'static str = "GPUCommandBuffer";
}

impl GarbageCollected for GPUCommandBuffer {
  fn get_name(&self) -> &'static std::ffi::CStr {
    c"GPUCommandBuffer"
  }
}

#[op2]
impl GPUCommandBuffer {
  #[constructor]
  #[cppgc]
  fn constructor(_: bool) -> Result<GPUCommandBuffer, GPUGenericError> {
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
pub(crate) struct GPUCommandBufferDescriptor {
  #[webidl(default = String::new())]
  pub label: String,
}
