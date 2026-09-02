// Copyright 2018-2025 the Deno authors. MIT license.

use std::borrow::Cow;
use std::cell::RefCell;
use std::num::NonZeroU64;
use std::sync::Arc;

use deno_core::cppgc::Ptr;
use deno_core::op2;
use deno_core::v8;
use deno_core::v8::HandleScope;
use deno_core::v8::Local;
use deno_core::v8::Value;
use deno_core::webidl::ContextFn;
use deno_core::webidl::IntOptions;
use deno_core::webidl::Nullable;
use deno_core::webidl::WebIdlConverter;
use deno_core::webidl::WebIdlError;
use deno_core::GarbageCollected;
use deno_core::WebIDL;
use deno_error::JsErrorBox;

use crate::buffer::GPUBuffer;
use crate::error::GPUGenericError;
use crate::get_data_slice;
use crate::render_bundle::GPURenderBundle;
use crate::texture::GPUTexture;
use crate::texture::GPUTextureView;
use crate::webidl::GPUColor;

pub struct GPURenderPassEncoder {
  pub render_pass: RefCell<wgpu_core::command::RenderPass>,
  pub label: String,
}

impl GarbageCollected for GPURenderPassEncoder {
  fn get_name(&self) -> &'static std::ffi::CStr {
    c"GPURenderPassEncoder"
  }
}

#[op2]
impl GPURenderPassEncoder {
  #[constructor]
  #[cppgc]
  fn constructor(_: bool) -> Result<GPURenderPassEncoder, GPUGenericError> {
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

  #[required(6)]
  #[undefined]
  fn set_viewport(
    &self,
    #[webidl] x: f32,
    #[webidl] y: f32,
    #[webidl] width: f32,
    #[webidl] height: f32,
    #[webidl] min_depth: f32,
    #[webidl] max_depth: f32,
  ) {
    self
      .render_pass
      .borrow_mut()
      .set_viewport(x, y, width, height, min_depth, max_depth);
  }

  #[required(4)]
  #[undefined]
  fn set_scissor_rect(
    &self,
    #[webidl(options(enforce_range = true))] x: u32,
    #[webidl(options(enforce_range = true))] y: u32,
    #[webidl(options(enforce_range = true))] width: u32,
    #[webidl(options(enforce_range = true))] height: u32,
  ) {
    self
      .render_pass
      .borrow_mut()
      .set_scissor_rect(x, y, width, height);
  }

  #[reentrant]
  #[required(1)]
  #[undefined]
  fn set_blend_constant(&self, #[webidl] color: GPUColor) {
    self
      .render_pass
      .borrow_mut()
      .set_blend_constant(color.into());
  }

  #[required(1)]
  #[undefined]
  fn set_stencil_reference(
    &self,
    #[webidl(options(enforce_range = true))] reference: u32,
  ) {
    self
      .render_pass
      .borrow_mut()
      .set_stencil_reference(reference);
  }

  #[required(1)]
  #[undefined]
  fn begin_occlusion_query(
    &self,
    #[webidl(options(enforce_range = true))] query_index: u32,
  ) {
    self
      .render_pass
      .borrow_mut()
      .begin_occlusion_query(query_index);
  }

  #[fast]
  #[undefined]
  fn end_occlusion_query(&self) {
    self.render_pass.borrow_mut().end_occlusion_query();
  }

  #[reentrant]
  #[required(1)]
  #[undefined]
  fn execute_bundles(&self, #[webidl] bundles: Vec<Ptr<GPURenderBundle>>) {
    self.render_pass.borrow_mut().execute_bundles(
      &bundles
        .into_iter()
        .map(|bundle| bundle.wgpu_render_bundle.clone())
        .collect::<Vec<_>>(),
    );
  }

  #[fast]
  #[undefined]
  fn end(&self) {
    self.render_pass.borrow_mut().end();
  }

  #[undefined]
  fn push_debug_group(&self, #[webidl] group_label: String) {
    self.render_pass.borrow_mut().push_debug_group(
      &group_label,
      0, // wgpu#975
    );
  }

  #[fast]
  #[undefined]
  fn pop_debug_group(&self) {
    self.render_pass.borrow_mut().pop_debug_group();
  }

  #[undefined]
  fn insert_debug_marker(&self, #[webidl] marker_label: String) {
    self.render_pass.borrow_mut().insert_debug_marker(
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

      // SAFETY: created from an array buffer, slice is dropped at end of function call
      let data =
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as _, ab_len) };

      let offsets = &data[start..(start + len)];

      self.render_pass.borrow_mut().set_bind_group(
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

      self.render_pass.borrow_mut().set_bind_group(
        index,
        bind_group
          .into_option()
          .map(|bind_group| bind_group.wgpu_bind_group.clone()),
        &offsets,
      )
    };

    Ok(())
  }

  #[undefined]
  fn set_pipeline(
    &self,
    #[webidl] pipeline: Ptr<crate::render_pipeline::GPURenderPipeline>,
  ) {
    self
      .render_pass
      .borrow_mut()
      .set_pipeline(pipeline.wgpu_render_pipeline.clone());
  }

  #[required(2)]
  #[undefined]
  fn set_index_buffer(
    &self,
    #[webidl] buffer: Ptr<GPUBuffer>,
    #[webidl] index_format: crate::render_pipeline::GPUIndexFormat,
    #[webidl(default = 0, options(enforce_range = true))] offset: u64,
    #[webidl(options(enforce_range = true))] size: Option<u64>,
  ) {
    self.render_pass.borrow_mut().set_index_buffer(
      buffer.wgpu_buffer.clone(),
      index_format.into(),
      offset,
      size.and_then(NonZeroU64::new),
    );
  }

  #[required(2)]
  #[undefined]
  fn set_vertex_buffer(
    &self,
    #[webidl(options(enforce_range = true))] slot: u32,
    #[webidl] buffer: Nullable<Ptr<GPUBuffer>>,
    #[webidl(default = 0, options(enforce_range = true))] offset: u64,
    #[webidl(options(enforce_range = true))] size: Option<u64>,
  ) {
    self.render_pass.borrow_mut().set_vertex_buffer(
      slot,
      buffer
        .into_option()
        .map(|buffer| buffer.wgpu_buffer.clone()),
      offset,
      size.and_then(NonZeroU64::new),
    );
  }

  #[required(1)]
  #[undefined]
  fn draw(
    &self,
    #[webidl(options(enforce_range = true))] vertex_count: u32,
    #[webidl(default = 1, options(enforce_range = true))] instance_count: u32,
    #[webidl(default = 0, options(enforce_range = true))] first_vertex: u32,
    #[webidl(default = 0, options(enforce_range = true))] first_instance: u32,
  ) {
    self.render_pass.borrow_mut().draw(
      vertex_count,
      instance_count,
      first_vertex,
      first_instance,
    );
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
  ) {
    self.render_pass.borrow_mut().draw_indexed(
      index_count,
      instance_count,
      first_index,
      base_vertex,
      first_instance,
    );
  }

  #[required(2)]
  #[undefined]
  fn draw_indirect(
    &self,
    #[webidl] indirect_buffer: Ptr<GPUBuffer>,
    #[webidl(options(enforce_range = true))] indirect_offset: u64,
  ) {
    self
      .render_pass
      .borrow_mut()
      .draw_indirect(indirect_buffer.wgpu_buffer.clone(), indirect_offset);
  }

  #[required(2)]
  #[undefined]
  fn draw_indexed_indirect(
    &self,
    #[webidl] indirect_buffer: Ptr<GPUBuffer>,
    #[webidl(options(enforce_range = true))] indirect_offset: u64,
  ) {
    self.render_pass.borrow_mut().draw_indexed_indirect(
      indirect_buffer.wgpu_buffer.clone(),
      indirect_offset,
    );
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

    self.render_pass.borrow_mut().set_immediates(offset, data);
    Ok(())
  }
}

#[derive(WebIDL)]
#[webidl(dictionary)]
pub(crate) struct GPURenderPassDescriptor {
  #[webidl(default = String::new())]
  pub label: String,

  pub color_attachments: Vec<Nullable<GPURenderPassColorAttachment>>,
  pub depth_stencil_attachment: Option<GPURenderPassDepthStencilAttachment>,
  pub occlusion_query_set: Option<Ptr<crate::query_set::GPUQuerySet>>,
  pub timestamp_writes: Option<GPURenderPassTimestampWrites>,
  /*#[webidl(default = 50000000)]
  #[options(enforce_range = true)]
  pub max_draw_count: u64,*/
  #[webidl(default = 0)]
  #[options(enforce_range = true)]
  pub multiview_mask: u32,
}

#[derive(WebIDL)]
#[webidl(dictionary)]
pub(crate) struct GPURenderPassColorAttachment {
  pub view: GPUTextureOrView,
  #[options(enforce_range = true)]
  pub depth_slice: Option<u32>,
  pub resolve_target: Option<GPUTextureOrView>,
  pub clear_value: Option<GPUColor>,
  pub load_op: GPULoadOp,
  pub store_op: GPUStoreOp,
}

#[derive(WebIDL)]
#[webidl(enum)]
pub(crate) enum GPULoadOp {
  Load,
  Clear,
}
impl GPULoadOp {
  pub fn with_default_value<V: Default>(
    self,
    val: Option<V>,
  ) -> wgpu_core::command::LoadOp<V> {
    match self {
      GPULoadOp::Load => wgpu_core::command::LoadOp::Load,
      GPULoadOp::Clear => {
        wgpu_core::command::LoadOp::Clear(val.unwrap_or_default())
      }
    }
  }

  pub fn with_value<V>(self, val: V) -> wgpu_core::command::LoadOp<V> {
    match self {
      GPULoadOp::Load => wgpu_core::command::LoadOp::Load,
      GPULoadOp::Clear => wgpu_core::command::LoadOp::Clear(val),
    }
  }
}

#[derive(WebIDL)]
#[webidl(enum)]
pub(crate) enum GPUStoreOp {
  Store,
  Discard,
}
impl From<GPUStoreOp> for wgpu_core::command::StoreOp {
  fn from(value: GPUStoreOp) -> Self {
    match value {
      GPUStoreOp::Store => Self::Store,
      GPUStoreOp::Discard => Self::Discard,
    }
  }
}

#[derive(WebIDL)]
#[webidl(dictionary)]
pub(crate) struct GPURenderPassDepthStencilAttachment {
  pub view: GPUTextureOrView,
  pub depth_clear_value: Option<f32>,
  pub depth_load_op: Option<GPULoadOp>,
  pub depth_store_op: Option<GPUStoreOp>,
  #[webidl(default = false)]
  pub depth_read_only: bool,
  #[webidl(default = 0)]
  #[options(enforce_range = true)]
  pub stencil_clear_value: u32,
  pub stencil_load_op: Option<GPULoadOp>,
  pub stencil_store_op: Option<GPUStoreOp>,
  #[webidl(default = false)]
  pub stencil_read_only: bool,
}

#[derive(WebIDL)]
#[webidl(dictionary)]
pub(crate) struct GPURenderPassTimestampWrites {
  pub query_set: Ptr<crate::query_set::GPUQuerySet>,
  #[options(enforce_range = true)]
  pub beginning_of_pass_write_index: Option<u32>,
  #[options(enforce_range = true)]
  pub end_of_pass_write_index: Option<u32>,
}

pub(crate) enum GPUTextureOrView {
  Texture(Ptr<GPUTexture>),
  TextureView(Ptr<GPUTextureView>),
}

impl GPUTextureOrView {
  pub(crate) fn to_view(&self) -> Arc<wgpu_core::resource::TextureView> {
    match self {
      Self::Texture(texture) => texture.default_view(),
      Self::TextureView(texture_view) => texture_view.wgpu_texture_view.clone(),
    }
  }
}

impl<'a> WebIdlConverter<'a> for GPUTextureOrView {
  type Options = ();

  fn convert<'b>(
    scope: &mut HandleScope<'a>,
    value: Local<'a, Value>,
    prefix: Cow<'static, str>,
    context: ContextFn<'b>,
    options: &Self::Options,
  ) -> Result<Self, WebIdlError> {
    <Ptr<GPUTexture>>::convert(
      scope,
      value,
      prefix.clone(),
      context.borrowed(),
      options,
    )
    .map(Self::Texture)
    .or_else(|_| {
      <Ptr<GPUTextureView>>::convert(
        scope,
        value,
        prefix.clone(),
        context.borrowed(),
        options,
      )
      .map(Self::TextureView)
    })
  }
}
