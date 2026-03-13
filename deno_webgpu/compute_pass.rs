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
use wgpu_core::binding_model::ImmediateUploadError;

use crate::error::GPUError;
use crate::error::GPUGenericError;
use crate::Instance;

pub struct GPUComputePassEncoder {
  pub instance: Instance,
  pub error_handler: super::error::ErrorHandler,

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
    let err = self
      .instance
      .compute_pass_set_pipeline(
        &mut self.compute_pass.borrow_mut(),
        pipeline.id,
      )
      .err();
    self.error_handler.push_error(err);
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
    let err = self
      .instance
      .compute_pass_dispatch_workgroups(
        &mut self.compute_pass.borrow_mut(),
        work_group_count_x,
        work_group_count_y,
        work_group_count_z,
      )
      .err();
    self.error_handler.push_error(err);
  }

  #[undefined]
  fn dispatch_workgroups_indirect(
    &self,
    #[webidl] indirect_buffer: Ptr<crate::buffer::GPUBuffer>,
    #[webidl(options(enforce_range = true))] indirect_offset: u64,
  ) {
    let err = self
      .instance
      .compute_pass_dispatch_workgroups_indirect(
        &mut self.compute_pass.borrow_mut(),
        indirect_buffer.id,
        indirect_offset,
      )
      .err();
    self.error_handler.push_error(err);
  }

  #[fast]
  #[undefined]
  fn end(&self) {
    let err = self
      .instance
      .compute_pass_end(&mut self.compute_pass.borrow_mut())
      .err();
    self.error_handler.push_error(err);
  }

  #[undefined]
  fn push_debug_group(&self, #[webidl] group_label: String) {
    let err = self
      .instance
      .compute_pass_push_debug_group(
        &mut self.compute_pass.borrow_mut(),
        &group_label,
        0, // wgpu#975
      )
      .err();
    self.error_handler.push_error(err);
  }

  #[fast]
  #[undefined]
  fn pop_debug_group(&self) {
    let err = self
      .instance
      .compute_pass_pop_debug_group(&mut self.compute_pass.borrow_mut())
      .err();
    self.error_handler.push_error(err);
  }

  #[undefined]
  fn insert_debug_marker(&self, #[webidl] marker_label: String) {
    let err = self
      .instance
      .compute_pass_insert_debug_marker(
        &mut self.compute_pass.borrow_mut(),
        &marker_label,
        0, // wgpu#975
      )
      .err();
    self.error_handler.push_error(err);
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
    let err = if let Ok(uint_32) = dynamic_offsets.try_cast::<v8::Uint32Array>()
    {
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

      self
        .instance
        .compute_pass_set_bind_group(
          &mut self.compute_pass.borrow_mut(),
          index,
          bind_group.into_option().map(|bind_group| bind_group.id),
          offsets,
        )
        .err()
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

      self
        .instance
        .compute_pass_set_bind_group(
          &mut self.compute_pass.borrow_mut(),
          index,
          bind_group.into_option().map(|bind_group| bind_group.id),
          &offsets,
        )
        .err()
    };

    self.error_handler.push_error(err);

    Ok(())
  }

  // TODO: share marshalling code
  #[required(2)]
  #[undefined]
  fn set_immediates<'a>(
    &self,
    scope: &mut v8::HandleScope<'a>,
    #[webidl(options(enforce_range = true))] range_offset: u32,
    data: v8::Local<v8::Value>,
    data_offset: v8::Local<'a, v8::Value>,
    size: v8::Local<'a, v8::Value>,
  ) -> Result<(), WebIdlError> {
    const PREFIX: &str =
      "Failed to execute 'setImmediateData' on 'GPUComputePassEncoder'";

    let data_size;
    let data_byte_len;
    let data_buffer;
    // TODO: Surely some helpers for this must exist?
    if let Ok(typed_array) = data.try_cast::<v8::TypedArray>() {
      data_byte_len = typed_array.byte_length();
      data_size = typed_array.length();
      data_buffer = typed_array.buffer(scope).and_then(|b| b.data());
    } else if let Ok(array_buf_view) = data.try_cast::<v8::ArrayBufferView>() {
      data_byte_len = array_buf_view.byte_length();
      data_size = 1;
      data_buffer = array_buf_view.buffer(scope).and_then(|b| b.data());
    } else if let Ok(array_buf) = data.try_cast::<v8::ArrayBuffer>() {
      data_byte_len = array_buf.byte_length();
      data_size = 1;
      data_buffer = array_buf.data();
    } else if let Ok(shared_array_buf) =
      data.try_cast::<v8::SharedArrayBuffer>()
    {
      data_byte_len = shared_array_buf.byte_length();
      data_size = 1;
      // TODO: file issue about `data` convenience accessor
      data_buffer = shared_array_buf.get_backing_store().data();
    } else {
      return Err(WebIdlError::new(
        Cow::Borrowed(PREFIX),
        (|| Cow::Borrowed("Argument 2")).into(),
        deno_core::webidl::WebIdlErrorKind::ConvertToConverterType(
          "AllowSharedBufferSource",
        ),
      ));
    }

    let data_offset = <u64 as WebIdlConverter<'a>>::convert(
      scope,
      data_offset,
      Cow::Borrowed(PREFIX),
      (|| Cow::Borrowed("Argument 3")).into(),
      &IntOptions {
        clamp: false,
        enforce_range: true,
      },
    )
    .map(|o| (o as usize) * data_size)?;

    let size = u64::convert(
      scope,
      size,
      Cow::Borrowed(PREFIX),
      (|| Cow::Borrowed("Argument 4")).into(),
      &IntOptions {
        clamp: false,
        enforce_range: true,
      },
    )
    .map(|o| (o as usize) * data_size)?;

    let data: &[_] = data_buffer.map_or(&[], |b| unsafe {
      // SAFETY: created from an array buffer, slice is dropped at end of function call
      std::slice::from_raw_parts(b.as_ptr() as *const u8, data_byte_len)
    });

    let data = match data.get(data_offset..(data_offset + size)) {
      Some(some) => some,
      None => {
        let err = if data_offset > data.len() {
          ImmediateUploadError::StartOffsetOverrun {
            start_offset: data_offset as u32,
            immediate_size: data_byte_len as u32,
          }
        } else {
          ImmediateUploadError::EndOffsetOverrun {
            start_offset: data_offset as u32,
            size: size as u32,
            immediate_size: data_byte_len as u32,
          }
        };

        self
          .error_handler
          .push_error(Some(GPUError::Validation(err.to_string())));

        return Ok(());
      }
    };

    let err = self
      .instance
      .compute_pass_set_immediates(
        &mut self.compute_pass.borrow_mut(),
        range_offset,
        data,
      )
      .err();
    self.error_handler.push_error(err);

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
