// Copyright 2018-2025 the Deno authors. MIT license.

use std::sync::Arc;

use deno_core::op2;
use deno_core::webidl::WebIdlInterfaceConverter;
use deno_core::GarbageCollected;
use deno_core::WebIDL;
use deno_error::JsErrorBox;
use wgpu_core::resource::Labeled;

use crate::error::GPUGenericError;

pub struct GPUQuerySet {
  pub wgpu_query_set: Arc<wgpu_core::resource::QuerySet>,
}

impl WebIdlInterfaceConverter for GPUQuerySet {
  const NAME: &'static str = "GPUQuerySet";
}

impl GarbageCollected for GPUQuerySet {
  fn get_name(&self) -> &'static std::ffi::CStr {
    c"GPUQuerySet"
  }
}

#[op2]
impl GPUQuerySet {
  #[constructor]
  #[cppgc]
  fn constructor(_: bool) -> Result<GPUQuerySet, GPUGenericError> {
    Err(GPUGenericError::InvalidConstructor)
  }

  #[getter]
  #[string]
  fn label(&self) -> String {
    self.wgpu_query_set.label().to_string()
  }
  #[setter]
  #[string]
  fn label(&self, #[webidl] _label: String) {
    // TODO(@crowlKats): no-op, needs wpgu to implement changing the label
  }

  #[fast]
  #[undefined]
  fn destroy(&self) -> Result<(), JsErrorBox> {
    self.wgpu_query_set.destroy();
    Ok(())
  }

  #[getter]
  #[string]
  #[rename("type")]
  fn r#type(&self) -> String {
    self.wgpu_query_set.descriptor().ty.to_string()
  }

  #[getter]
  fn count(&self) -> u32 {
    self.wgpu_query_set.descriptor().count
  }
}

#[derive(WebIDL)]
#[webidl(dictionary)]
pub(crate) struct GPUQuerySetDescriptor {
  #[webidl(default = String::new())]
  pub label: String,

  pub r#type: GPUQueryType,
  #[options(enforce_range = true)]
  pub count: u32,
}

#[derive(WebIDL, Clone)]
#[webidl(enum)]
pub(crate) enum GPUQueryType {
  Occlusion,
  Timestamp,
}
impl From<GPUQueryType> for wgpu_types::QueryType {
  fn from(value: GPUQueryType) -> Self {
    match value {
      GPUQueryType::Occlusion => Self::Occlusion,
      GPUQueryType::Timestamp => Self::Timestamp,
    }
  }
}
