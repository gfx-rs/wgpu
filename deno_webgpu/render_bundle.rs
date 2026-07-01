// Copyright 2018-2025 the Deno authors. MIT license.

use std::borrow::Cow;
use std::cell::RefCell;
use std::num::NonZeroU64;

use deno_core::cppgc::Ptr;
use deno_core::op2;
use deno_core::v8;
use deno_core::webidl::IntOptions;
use deno_core::webidl::Nullable;
use deno_core::webidl::WebIdlConverter;
use deno_core::webidl::WebIdlError;
use deno_core::webidl::WebIdlInterfaceConverter;
use deno_core::GarbageCollected;
use deno_core::WebIDL;
use deno_error::JsErrorBox;

use crate::buffer::GPUBuffer;
use crate::error::GPUGenericError;
use crate::get_data_slice;
use crate::texture::GPUTextureFormat;
use crate::Instance;

pub struct GPURenderBundleEncoder {
  pub instance: Instance,
  pub error_handler: super::error::ErrorHandler,

  pub encoder: RefCell<Box<wgpu_core::command::RenderBundleEncoder>>,
  pub label: String,
}

impl GarbageCollected for GPURenderBundleEncoder {
  fn get_name(&self) -> &'static std::ffi::CStr {
    c"GPURenderBundleEncoder"
  }
}

#[op2]
impl GPURenderBundleEncoder {
  #[constructor]
  #[cppgc]
  fn constructor(_: bool) -> Result<GPURenderBundleEncoder, GPUGenericError> {
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

  #[cppgc]
  fn finish(
    &self,
    #[webidl] descriptor: GPURenderBundleDescriptor,
  ) -> GPURenderBundle {
    let wgpu_descriptor = wgpu_core::command::RenderBundleDescriptor {
      label: crate::transform_label(descriptor.label.clone()),
    };

    let (id, err) = self.instance.render_bundle_encoder_finish(
      &mut self.encoder.borrow_mut(),
      &wgpu_descriptor,
      None,
    );

    self.error_handler.push_error(err);

    GPURenderBundle {
      instance: self.instance.clone(),
      id,
      label: descriptor.label.clone(),
    }
  }

  #[undefined]
  fn push_debug_group(
    &self,
    #[webidl] group_label: String,
  ) -> Result<(), JsErrorBox> {
    let mut encoder = self.encoder.borrow_mut();
    let encoder = encoder.as_mut();

    let err = self
      .instance
      .render_bundle_encoder_push_debug_group(encoder, &group_label)
      .err();

    self.error_handler.push_error(err);

    Ok(())
  }

  #[fast]
  #[undefined]
  fn pop_debug_group(&self) -> Result<(), JsErrorBox> {
    let mut encoder = self.encoder.borrow_mut();
    let encoder = encoder.as_mut();

    let err = self
      .instance
      .render_bundle_encoder_pop_debug_group(encoder)
      .err();

    self.error_handler.push_error(err);

    Ok(())
  }

  #[undefined]
  fn insert_debug_marker(
    &self,
    #[webidl] marker_label: String,
  ) -> Result<(), JsErrorBox> {
    let mut encoder = self.encoder.borrow_mut();
    let encoder = encoder.as_mut();

    let err = self
      .instance
      .render_bundle_encoder_insert_debug_marker(encoder, &marker_label)
      .err();

    self.error_handler.push_error(err);
    Ok(())
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
  ) -> Result<(), SetBindGroupError> {
    let mut encoder = self.encoder.borrow_mut();
    let encoder = encoder.as_mut();

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

      // SAFETY: created from an array buffer, slice is dropped at end of function call
      let data =
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as _, ab_len) };

      let offsets = &data[start..(start + len)];

      let err = self
        .instance
        .render_bundle_encoder_set_bind_group(
          encoder,
          index,
          bind_group.into_option().map(|bind_group| bind_group.id),
          offsets,
        )
        .err();

      self.error_handler.push_error(err);
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

      let err = self
        .instance
        .render_bundle_encoder_set_bind_group(
          encoder,
          index,
          bind_group.into_option().map(|bind_group| bind_group.id),
          &offsets,
        )
        .err();

      self.error_handler.push_error(err);
    }

    Ok(())
  }

  #[undefined]
  fn set_pipeline(
    &self,
    #[webidl] pipeline: Ptr<crate::render_pipeline::GPURenderPipeline>,
  ) -> Result<(), JsErrorBox> {
    let mut encoder = self.encoder.borrow_mut();
    let encoder = encoder.as_mut();

    let err = self
      .instance
      .render_bundle_encoder_set_pipeline(encoder, pipeline.id)
      .err();

    self.error_handler.push_error(err);
    Ok(())
  }

  #[required(2)]
  #[undefined]
  fn set_index_buffer(
    &self,
    #[webidl] buffer: Ptr<GPUBuffer>,
    #[webidl] index_format: crate::render_pipeline::GPUIndexFormat,
    #[webidl(default = 0, options(enforce_range = true))] offset: u64,
    #[webidl(options(enforce_range = true))] size: Option<u64>,
  ) -> Result<(), JsErrorBox> {
    let mut encoder = self.encoder.borrow_mut();
    let encoder = encoder.as_mut();

    let err = self
      .instance
      .render_bundle_encoder_set_index_buffer(
        encoder,
        buffer.id,
        index_format.into(),
        offset,
        size.and_then(NonZeroU64::new),
      )
      .err();

    self.error_handler.push_error(err);

    Ok(())
  }

  #[required(2)]
  #[undefined]
  fn set_vertex_buffer(
    &self,
    #[webidl(options(enforce_range = true))] slot: u32,
    #[webidl] buffer: Nullable<Ptr<GPUBuffer>>,
    #[webidl(default = 0, options(enforce_range = true))] offset: u64,
    #[webidl(options(enforce_range = true))] size: Option<u64>,
  ) -> Result<(), JsErrorBox> {
    let mut encoder = self.encoder.borrow_mut();
    let encoder = encoder.as_mut();

    let err = self
      .instance
      .render_bundle_encoder_set_vertex_buffer(
        encoder,
        slot,
        buffer.into_option().map(|buffer| buffer.id),
        offset,
        size.and_then(NonZeroU64::new),
      )
      .err();
    self.error_handler.push_error(err);
    Ok(())
  }

  #[required(1)]
  #[undefined]
  fn draw(
    &self,
    #[webidl(options(enforce_range = true))] vertex_count: u32,
    #[webidl(default = 1, options(enforce_range = true))] instance_count: u32,
    #[webidl(default = 0, options(enforce_range = true))] first_vertex: u32,
    #[webidl(default = 0, options(enforce_range = true))] first_instance: u32,
  ) -> Result<(), JsErrorBox> {
    let mut encoder = self.encoder.borrow_mut();
    let encoder = encoder.as_mut();

    let err = self
      .instance
      .render_bundle_encoder_draw(
        encoder,
        vertex_count,
        instance_count,
        first_vertex,
        first_instance,
      )
      .err();
    self.error_handler.push_error(err);
    Ok(())
  }

  #[required(1)]
  #[undefined]
  fn draw_indexed(
    &self,
    #[webidl(options(enforce_range = true))] index_count: u32,
    #[webidl(default = 1, options(enforce_range = true))] instance_count: u32,
    #[webidl(default = 0, options(enforce_range = true))] first_index: u32,
    #[webidl(default = 0, options(enforce_range = true))] base_vertex: i32,
    #[webidl(default = 0, options(enforce_range = true))] first_instance: u32,
  ) -> Result<(), JsErrorBox> {
    let mut encoder = self.encoder.borrow_mut();
    let encoder = encoder.as_mut();

    let err = self
      .instance
      .render_bundle_encoder_draw_indexed(
        encoder,
        index_count,
        instance_count,
        first_index,
        base_vertex,
        first_instance,
      )
      .err();
    self.error_handler.push_error(err);
    Ok(())
  }

  #[required(2)]
  #[undefined]
  fn draw_indirect(
    &self,
    #[webidl] indirect_buffer: Ptr<GPUBuffer>,
    #[webidl(options(enforce_range = true))] indirect_offset: u64,
  ) -> Result<(), JsErrorBox> {
    let mut encoder = self.encoder.borrow_mut();
    let encoder = encoder.as_mut();

    let err = self
      .instance
      .render_bundle_encoder_draw_indirect(
        encoder,
        indirect_buffer.id,
        indirect_offset,
      )
      .err();
    self.error_handler.push_error(err);
    Ok(())
  }

  #[required(2)]
  #[undefined]
  fn draw_indexed_indirect(
    &self,
    #[webidl] indirect_buffer: Ptr<GPUBuffer>,
    #[webidl(options(enforce_range = true))] indirect_offset: u64,
  ) -> Result<(), JsErrorBox> {
    let mut encoder = self.encoder.borrow_mut();
    let encoder = encoder.as_mut();

    let err = self
      .instance
      .render_bundle_encoder_draw_indexed_indirect(
        encoder,
        indirect_buffer.id,
        indirect_offset,
      )
      .err();
    self.error_handler.push_error(err);
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

    let mut encoder = self.encoder.borrow_mut();
    let encoder = encoder.as_mut();

    let err = self
      .instance
      .render_bundle_encoder_set_immediates(encoder, offset, data)
      .err();
    self.error_handler.push_error(err);
    Ok(())
  }
}

#[derive(WebIDL)]
#[webidl(dictionary)]
pub(crate) struct GPURenderBundleEncoderDescriptor {
  #[webidl(default = String::new())]
  pub label: String,

  pub color_formats: Vec<Nullable<GPUTextureFormat>>,
  pub depth_stencil_format: Option<GPUTextureFormat>,
  #[webidl(default = 1)]
  #[options(enforce_range = true)]
  pub sample_count: u32,

  #[webidl(default = false)]
  pub depth_read_only: bool,
  #[webidl(default = false)]
  pub stencil_read_only: bool,
}

#[derive(Debug, thiserror::Error, deno_error::JsError)]
enum SetBindGroupError {
  #[class(inherit)]
  #[error(transparent)]
  WebIDL(#[from] WebIdlError),
  #[class(inherit)]
  #[error(transparent)]
  Other(#[from] JsErrorBox),
}

pub struct GPURenderBundle {
  pub instance: Instance,
  pub id: wgpu_core::id::RenderBundleId,
  pub label: String,
}

impl Drop for GPURenderBundle {
  fn drop(&mut self) {
    self.instance.render_bundle_drop(self.id);
  }
}

impl WebIdlInterfaceConverter for GPURenderBundle {
  const NAME: &'static str = "GPURenderBundle";
}

impl GarbageCollected for GPURenderBundle {
  fn get_name(&self) -> &'static std::ffi::CStr {
    c"GPURenderBundle"
  }
}

#[op2]
impl GPURenderBundle {
  #[constructor]
  #[cppgc]
  fn constructor(_: bool) -> Result<GPURenderBundle, GPUGenericError> {
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
pub(crate) struct GPURenderBundleDescriptor {
  #[webidl(default = String::new())]
  pub label: String,
}
