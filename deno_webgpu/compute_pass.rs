// Copyright 2018-2025 the Deno authors. MIT license.

use std::borrow::Cow;
use std::cell::RefCell;

use deno_core::cppgc::Ptr;
use deno_core::op2;
use deno_core::v8;
use deno_core::webidl::IntOptions;
use deno_core::webidl::Nullable;
use deno_core::webidl::WebIdlConverter;
use deno_core::webidl::WebIdlError;
use deno_core::GarbageCollected;
use deno_core::WebIDL;
use deno_error::JsErrorBox;

use crate::error::GPUGenericError;
use crate::get_data_slice;

pub struct GPUComputePassEncoder {
  pub compute_pass: RefCell<wgpu_core::command::ComputePass>,
  pub label: String,
}

impl GarbageCollected for GPUComputePassEncoder {
  fn get_name(&self) -> &'static std::ffi::CStr {
    c"GPUComputePassEncoder"
  }
}

#[op2]
impl GPUComputePassEncoder {
  #[constructor]
  #[cppgc]
  fn constructor(_: bool) -> Result<GPUComputePassEncoder, GPUGenericError> {
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

  #[undefined]
  fn set_pipeline(
    &self,
    #[webidl] pipeline: Ptr<crate::compute_pipeline::GPUComputePipeline>,
  ) {
    self
      .compute_pass
      .borrow_mut()
      .set_pipeline(pipeline.wgpu_compute_pipeline.clone());
  }

  #[undefined]
  fn dispatch_workgroups(
    &self,
    #[webidl(options(enforce_range = true))] work_group_count_x: u32,
    #[webidl(default = 1, options(enforce_range = true))]
    work_group_count_y: u32,
    #[webidl(default = 1, options(enforce_range = true))]
    work_group_count_z: u32,
  ) {
    self.compute_pass.borrow_mut().dispatch_workgroups(
      work_group_count_x,
      work_group_count_y,
      work_group_count_z,
    );
  }

  #[undefined]
  fn dispatch_workgroups_indirect(
    &self,
    #[webidl] indirect_buffer: Ptr<crate::buffer::GPUBuffer>,
    #[webidl(options(enforce_range = true))] indirect_offset: u64,
  ) {
    self.compute_pass.borrow_mut().dispatch_workgroups_indirect(
      indirect_buffer.wgpu_buffer.clone(),
      indirect_offset,
    );
  }

  #[fast]
  #[undefined]
  fn end(&self) {
    self.compute_pass.borrow_mut().end();
  }

  #[undefined]
  fn push_debug_group(&self, #[webidl] group_label: String) {
    self.compute_pass.borrow_mut().push_debug_group(
      &group_label,
      0, // wgpu#975
    );
  }

  #[fast]
  #[undefined]
  fn pop_debug_group(&self) {
    self.compute_pass.borrow_mut().pop_debug_group();
  }

  #[undefined]
  fn insert_debug_marker(&self, #[webidl] marker_label: String) {
    self.compute_pass.borrow_mut().insert_debug_marker(
      &marker_label,
      0, // wgpu#975
    );
  }

  #[undefined]
  fn set_bind_group<'a>(
    &self,
    scope: &mut v8::HandleScope<'a>,
    #[webidl(options(enforce_range = true))] index: u32,
    #[webidl] bind_group: Nullable<Ptr<crate::bind_group::GPUBindGroup>>,
    dynamic_offsets: v8::Local<'a, v8::Value>,
    dynamic_offsets_data_start: v8::Local<'a, v8::Value>,
    dynamic_offsets_data_length: v8::Local<'a, v8::Value>,
  ) -> Result<(), WebIdlError> {
    const PREFIX: &str =
      "Failed to execute 'setBindGroup' on 'GPUComputePassEncoder'";
    if let Ok(uint_32) = dynamic_offsets.try_cast::<v8::Uint32Array>() {
      let start = u64::convert(
        scope,
        dynamic_offsets_data_start,
        Cow::Borrowed(PREFIX),
        (|| Cow::Borrowed("Argument 4")).into(),
        &IntOptions {
          clamp: false,
          enforce_range: true,
        },
      )? as usize;
      let len = u32::convert(
        scope,
        dynamic_offsets_data_length,
        Cow::Borrowed(PREFIX),
        (|| Cow::Borrowed("Argument 5")).into(),
        &IntOptions {
          clamp: false,
          enforce_range: true,
        },
      )? as usize;

      let ab = uint_32.buffer(scope).unwrap();
      let ptr = ab.data().unwrap();
      let ab_len = ab.byte_length() / 4;

      // SAFETY: compute_pass_set_bind_group internally calls extend_from_slice with this slice
      let data =
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as _, ab_len) };

      let offsets = &data[start..(start + len)];

      self.compute_pass.borrow_mut().set_bind_group(
        index,
        bind_group
          .into_option()
          .map(|bind_group| bind_group.wgpu_bind_group.clone()),
        offsets,
      )
    } else {
      let offsets = <Option<Vec<u32>>>::convert(
        scope,
        dynamic_offsets,
        Cow::Borrowed(PREFIX),
        (|| Cow::Borrowed("Argument 3")).into(),
        &IntOptions {
          clamp: false,
          enforce_range: true,
        },
      )?
      .unwrap_or_default();

      self.compute_pass.borrow_mut().set_bind_group(
        index,
        bind_group
          .into_option()
          .map(|bind_group| bind_group.wgpu_bind_group.clone()),
        &offsets,
      )
    };

    Ok(())
  }

  #[required(2)]
  #[undefined]
  fn set_immediates<'a>(
    &self,
    scope: &mut v8::HandleScope<'a>,
    #[webidl(options(enforce_range = true))] offset: u32,
    data_arg: v8::Local<'a, v8::Value>,
    #[webidl(default = 0, options(enforce_range = true))] data_offset: u64,
    #[webidl(options(enforce_range = true))] data_size: Option<u64>,
  ) -> Result<(), JsErrorBox> {
    let data = get_data_slice(scope, data_arg, data_offset, data_size)?;

    self.compute_pass.borrow_mut().set_immediates(offset, data);
    Ok(())
  }
}

#[derive(WebIDL)]
#[webidl(dictionary)]
pub(crate) struct GPUComputePassDescriptor {
  #[webidl(default = String::new())]
  pub label: String,

  pub timestamp_writes: Option<GPUComputePassTimestampWrites>,
}

#[derive(WebIDL)]
#[webidl(dictionary)]
pub(crate) struct GPUComputePassTimestampWrites {
  pub query_set: Ptr<crate::query_set::GPUQuerySet>,
  #[options(enforce_range = true)]
  pub beginning_of_pass_write_index: Option<u32>,
  #[options(enforce_range = true)]
  pub end_of_pass_write_index: Option<u32>,
}
