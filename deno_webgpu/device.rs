// Copyright 2018-2025 the Deno authors. MIT license.

use std::borrow::Cow;
use std::cell::RefCell;
use std::num::NonZeroU64;
use std::rc::Rc;
use std::sync::Arc;

use deno_core::cppgc::{make_cppgc_object, SameObject};
use deno_core::op2;
use deno_core::v8;
use deno_core::webidl::WebIdlInterfaceConverter;
use deno_core::GarbageCollected;
use deno_error::JsErrorBox;
use wgpu_core::binding_model::BindingResource;
use wgpu_core::error::EmptyErrorScopeStack;
use wgpu_core::pipeline::ProgrammableStageDescriptor;
use wgpu_types::BindingType;

use super::bind_group::GPUBindGroup;
use super::bind_group::GPUBindingResource;
use super::bind_group_layout::GPUBindGroupLayout;
use super::buffer::GPUBuffer;
use super::compute_pipeline::GPUComputePipeline;
use super::pipeline_layout::GPUPipelineLayout;
use super::sampler::GPUSampler;
use super::shader::GPUShaderModule;
use super::texture::GPUTexture;
use crate::adapter::GPUAdapterInfo;
use crate::adapter::GPUSupportedFeatures;
use crate::adapter::GPUSupportedLimits;
use crate::command_encoder::GPUCommandEncoder;
use crate::error::{fmt_err, make_pipeline_error, GPUError};
use crate::error::{GPUGenericError, GPUPipelineErrorReason};
use crate::query_set::GPUQuerySet;
use crate::render_bundle::GPURenderBundleEncoder;
use crate::render_pipeline::GPURenderPipeline;
use crate::shader::GPUCompilationInfo;
use crate::webidl::GPUTextureUsageFlags;

/// External memory associated with device and queue, to encourage V8 to garbage
/// collect devices promptly. This seems to be particularly important when
/// running CTS tests under `webgpu:api,validation,capability_checks,limits,*`
/// on DX12 in wgpu CI, where any smaller power of two results in OOM errors.
pub(crate) const DEVICE_EXTERNAL_MEMORY_SIZE: i64 = 1 << 24; // 16 MB

pub struct GPUDevice {
  pub wgpu_device: Arc<wgpu_core::device::Device>,
  pub wgpu_adapter: Arc<wgpu_core::instance::Adapter>,

  pub label: String,

  pub features: SameObject<GPUSupportedFeatures>,
  pub limits: SameObject<GPUSupportedLimits>,
  pub adapter_info: Rc<SameObject<GPUAdapterInfo>>,

  pub queue_obj: v8::Global<v8::Object>,

  pub error_handler: super::error::ErrorHandler,
  pub lost_promise: v8::Global<v8::Promise>,

  // Weak reference to the JS object so we can attach a finalizer.
  pub(crate) weak: std::sync::OnceLock<v8::Weak<v8::Object>>,
}

impl GPUDevice {
  /// <https://www.w3.org/TR/webgpu/#abstract-opdef-validate-texture-format-required-features>
  fn validate_texture_format_required_feature(
    &self,
    format: wgpu_types::TextureFormat,
  ) -> Result<(), JsErrorBox> {
    self
      .wgpu_device
      .require_features(format.required_features())
      .map_err(|err| {
        let err = fmt_err(&err);
        JsErrorBox::type_error(err)
      })
  }
}

impl WebIdlInterfaceConverter for GPUDevice {
  const NAME: &'static str = "GPUDevice";
}

impl GarbageCollected for GPUDevice {
  fn get_name(&self) -> &'static std::ffi::CStr {
    c"GPUDevice"
  }
}

// EventTarget is extended in JS
#[op2]
impl GPUDevice {
  #[constructor]
  #[cppgc]
  fn constructor(_: bool) -> Result<GPUDevice, GPUGenericError> {
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

  #[getter]
  #[global]
  fn features(&self, scope: &mut v8::HandleScope) -> v8::Global<v8::Object> {
    self.features.get(scope, |scope| {
      let features = self.wgpu_device.features();
      GPUSupportedFeatures::new(scope, *features)
    })
  }

  #[getter]
  #[global]
  fn limits(&self, scope: &mut v8::HandleScope) -> v8::Global<v8::Object> {
    self.limits.get(scope, |_| {
      let limits = self.wgpu_device.limits();
      GPUSupportedLimits(limits.clone())
    })
  }

  #[getter]
  #[global]
  fn adapter_info(
    &self,
    scope: &mut v8::HandleScope,
  ) -> v8::Global<v8::Object> {
    self.adapter_info.get(scope, |_| {
      let info = self.wgpu_adapter.get_info();

      GPUAdapterInfo { info }
    })
  }

  #[getter]
  #[global]
  fn queue(&self) -> v8::Global<v8::Object> {
    self.queue_obj.clone()
  }

  #[fast]
  #[undefined]
  fn destroy(&self) {
    self.wgpu_device.destroy();
  }

  #[reentrant]
  #[required(1)]
  #[cppgc]
  fn create_buffer(
    &self,
    #[webidl] descriptor: super::buffer::GPUBufferDescriptor,
  ) -> Result<GPUBuffer, JsErrorBox> {
    // wgpu-core would also check this, but it needs to be reported via a JS
    // error, not a validation error. (WebGPU specifies this check on the
    // content timeline.)
    if descriptor.mapped_at_creation
      && !descriptor
        .size
        .is_multiple_of(wgpu_types::COPY_BUFFER_ALIGNMENT)
    {
      return Err(JsErrorBox::range_error(
        format!(
          "The size of a buffer that is mapped at creation must be a multiple of {}",
          wgpu_types::COPY_BUFFER_ALIGNMENT,
        )
      ));
    }

    // Validation of the usage needs to happen on the device timeline, so
    // don't raise an error immediately if it isn't valid. wgpu will
    // reject `BufferUsages::empty()`.
    let usage = wgpu_types::BufferUsages::from_bits(descriptor.usage)
      .unwrap_or(wgpu_types::BufferUsages::empty());

    let wgpu_descriptor = wgpu_core::resource::BufferDescriptor {
      label: crate::transform_label(descriptor.label.clone()),
      size: descriptor.size,
      usage,
      mapped_at_creation: descriptor.mapped_at_creation,
    };

    let wgpu_buffer = self.wgpu_device.create_buffer(&wgpu_descriptor);

    Ok(GPUBuffer {
      wgpu_buffer,
      wgpu_device: self.wgpu_device.clone(),
      label: descriptor.label,
      size: descriptor.size,
      usage: descriptor.usage,
      map_state: RefCell::new(if descriptor.mapped_at_creation {
        "mapped"
      } else {
        "unmapped"
      }),
      map_mode: RefCell::new(if descriptor.mapped_at_creation {
        Some(wgpu_core::device::HostMap::Write)
      } else {
        None
      }),
      mapped_js_buffers: RefCell::new(vec![]),
    })
  }

  #[reentrant]
  #[required(1)]
  #[cppgc]
  fn create_texture(
    &self,
    #[webidl] descriptor: super::texture::GPUTextureDescriptor,
  ) -> Result<GPUTexture, JsErrorBox> {
    // Validation of the usage needs to happen on the device timeline, so
    // don't raise an error immediately if it isn't valid. wgpu will
    // reject `TextureUsages::empty()`.
    let usage = wgpu_types::TextureUsages::from_bits(descriptor.usage)
      .unwrap_or(wgpu_types::TextureUsages::empty());

    let wgpu_descriptor = wgpu_types::TextureDescriptor {
      label: crate::transform_label(descriptor.label.clone()),
      size: descriptor.size.into(),
      mip_level_count: descriptor.mip_level_count,
      sample_count: descriptor.sample_count,
      dimension: descriptor.dimension.clone().into(),
      format: descriptor.format.clone().into(),
      usage,
      view_formats: descriptor
        .view_formats
        .into_iter()
        .map(Into::into)
        .collect::<Vec<_>>(),
    };

    // 2. ? Validate texture format required features of descriptor.format with this.[[device]].
    self.validate_texture_format_required_feature(wgpu_descriptor.format)?;

    // 3. Validate texture format required features of each element of descriptor.viewFormats with this.[[device]].
    for format in &wgpu_descriptor.view_formats {
      self.validate_texture_format_required_feature(*format)?;
    }

    let wgpu_texture = self.wgpu_device.create_texture(&wgpu_descriptor);

    Ok(GPUTexture {
      wgpu_texture,
      default_view: Default::default(),
      label: descriptor.label,
      size: wgpu_descriptor.size,
      mip_level_count: wgpu_descriptor.mip_level_count,
      sample_count: wgpu_descriptor.sample_count,
      dimension: descriptor.dimension,
      format: descriptor.format,
      usage: GPUTextureUsageFlags(usage),
    })
  }

  #[reentrant]
  #[cppgc]
  fn create_sampler(
    &self,
    #[webidl] descriptor: super::sampler::GPUSamplerDescriptor,
  ) -> Result<GPUSampler, JsErrorBox> {
    let wgpu_descriptor = wgpu_core::resource::SamplerDescriptor {
      label: crate::transform_label(descriptor.label.clone()),
      address_modes: [
        descriptor.address_mode_u.into(),
        descriptor.address_mode_v.into(),
        descriptor.address_mode_w.into(),
      ],
      mag_filter: descriptor.mag_filter.into(),
      min_filter: descriptor.min_filter.into(),
      mipmap_filter: descriptor.mipmap_filter.into(),
      lod_min_clamp: descriptor.lod_min_clamp,
      lod_max_clamp: descriptor.lod_max_clamp,
      compare: descriptor.compare.map(Into::into),
      anisotropy_clamp: descriptor.max_anisotropy,
      border_color: None,
    };

    let wgpu_sampler = self.wgpu_device.create_sampler(&wgpu_descriptor);

    Ok(GPUSampler {
      wgpu_sampler,
      label: descriptor.label,
    })
  }

  #[reentrant]
  #[required(1)]
  #[cppgc]
  fn create_bind_group_layout(
    &self,
    #[webidl]
    descriptor: super::bind_group_layout::GPUBindGroupLayoutDescriptor,
  ) -> Result<GPUBindGroupLayout, JsErrorBox> {
    let mut entries = Vec::with_capacity(descriptor.entries.len());

    for entry in descriptor.entries {
      let n_entries = [
        entry.buffer.is_some(),
        entry.sampler.is_some(),
        entry.texture.is_some(),
        entry.storage_texture.is_some(),
        entry.external_texture.is_some(),
      ]
      .into_iter()
      .filter(|t| *t)
      .count();

      if n_entries != 1 {
        return Err(JsErrorBox::type_error(
          "Only one of 'buffer', 'sampler', 'texture' and 'storageTexture' may be specified",
        ));
      }

      let ty = if let Some(buffer) = entry.buffer {
        BindingType::Buffer {
          ty: buffer.r#type.into(),
          has_dynamic_offset: buffer.has_dynamic_offset,
          min_binding_size: NonZeroU64::new(buffer.min_binding_size),
        }
      } else if let Some(sampler) = entry.sampler {
        BindingType::Sampler(sampler.r#type.into())
      } else if let Some(texture) = entry.texture {
        BindingType::Texture {
          sample_type: texture.sample_type.into(),
          view_dimension: texture.view_dimension.into(),
          multisampled: texture.multisampled,
        }
      } else if let Some(storage_texture) = entry.storage_texture {
        let format = storage_texture.format.into();
        self.validate_texture_format_required_feature(format)?;
        BindingType::StorageTexture {
          access: storage_texture.access.into(),
          format,
          view_dimension: storage_texture.view_dimension.into(),
        }
      } else if entry.external_texture.is_some() {
        BindingType::ExternalTexture
      } else {
        unreachable!()
      };

      entries.push(wgpu_types::BindGroupLayoutEntry {
        binding: entry.binding,
        visibility: entry.visibility.into(),
        ty,
        count: None, // native-only
      });
    }

    let wgpu_descriptor = wgpu_core::binding_model::BindGroupLayoutDescriptor {
      label: crate::transform_label(descriptor.label.clone()),
      entries: Cow::Owned(entries),
    };

    let wgpu_bind_group_layout =
      self.wgpu_device.create_bind_group_layout(&wgpu_descriptor);

    Ok(GPUBindGroupLayout {
      wgpu_bind_group_layout,
      label: descriptor.label,
    })
  }

  #[reentrant]
  #[required(1)]
  #[cppgc]
  fn create_pipeline_layout(
    &self,
    #[webidl] descriptor: super::pipeline_layout::GPUPipelineLayoutDescriptor,
  ) -> GPUPipelineLayout {
    let bind_group_layouts = descriptor
      .bind_group_layouts
      .into_iter()
      .map(|bind_group_layout| {
        bind_group_layout.into_option().map(|bind_group_layout| {
          bind_group_layout.wgpu_bind_group_layout.clone()
        })
      })
      .collect();

    let wgpu_descriptor = wgpu_core::binding_model::PipelineLayoutDescriptor {
      label: crate::transform_label(descriptor.label.clone()),
      bind_group_layouts: Cow::Owned(bind_group_layouts),
      immediate_size: descriptor.immediate_size,
    };

    let wgpu_pipeline_layout =
      self.wgpu_device.create_pipeline_layout(&wgpu_descriptor);

    GPUPipelineLayout {
      wgpu_pipeline_layout,
      label: descriptor.label,
    }
  }

  #[reentrant]
  #[required(1)]
  #[cppgc]
  fn create_bind_group(
    &self,
    #[webidl] descriptor: super::bind_group::GPUBindGroupDescriptor,
  ) -> GPUBindGroup {
    let entries = descriptor
      .entries
      .into_iter()
      .map(|entry| wgpu_core::binding_model::BindGroupEntry {
        binding: entry.binding,
        resource: match entry.resource {
          GPUBindingResource::Sampler(sampler) => {
            BindingResource::Sampler(sampler.wgpu_sampler.clone())
          }
          GPUBindingResource::Texture(texture) => {
            BindingResource::TextureView(texture.default_view())
          }
          GPUBindingResource::TextureView(texture_view) => {
            BindingResource::TextureView(texture_view.wgpu_texture_view.clone())
          }
          GPUBindingResource::Buffer(buffer) => {
            BindingResource::Buffer(wgpu_core::binding_model::BufferBinding {
              buffer: buffer.wgpu_buffer.clone(),
              offset: 0,
              size: Some(buffer.size),
            })
          }
          GPUBindingResource::BufferBinding(buffer_binding) => {
            BindingResource::Buffer(wgpu_core::binding_model::BufferBinding {
              buffer: buffer_binding.buffer.wgpu_buffer.clone(),
              offset: buffer_binding.offset,
              size: buffer_binding.size,
            })
          }
          GPUBindingResource::ExternalTexture(external_texture) => {
            BindingResource::ExternalTexture(
              external_texture.wgpu_external_texture.clone(),
            )
          }
        },
      })
      .collect::<Vec<_>>();

    let wgpu_descriptor = wgpu_core::binding_model::BindGroupDescriptor {
      label: crate::transform_label(descriptor.label.clone()),
      layout: descriptor.layout.wgpu_bind_group_layout.clone(),
      entries: Cow::Owned(entries),
    };

    let wgpu_bind_group = self.wgpu_device.create_bind_group(&wgpu_descriptor);

    GPUBindGroup {
      wgpu_bind_group,
      label: descriptor.label,
    }
  }

  #[reentrant]
  #[required(1)]
  #[cppgc]
  fn create_shader_module(
    &self,
    scope: &mut v8::HandleScope<'_>,
    #[webidl] descriptor: super::shader::GPUShaderModuleDescriptor,
  ) -> GPUShaderModule {
    let wgpu_descriptor = wgpu_core::pipeline::ShaderModuleDescriptor {
      label: crate::transform_label(descriptor.label.clone()),
      runtime_checks: wgpu_types::ShaderRuntimeChecks::default(),
    };

    let (wgpu_shader_module, err) = self.wgpu_device.create_shader_module(
      &wgpu_descriptor,
      wgpu_core::pipeline::ShaderModuleSource::Wgsl(Cow::Borrowed(
        &descriptor.code,
      )),
    );

    let compilation_info =
      GPUCompilationInfo::new(scope, err.iter(), &descriptor.code);
    let compilation_info = make_cppgc_object(scope, compilation_info);
    let compilation_info = v8::Global::new(scope, compilation_info);
    self.error_handler.push_error(err);

    GPUShaderModule {
      wgpu_shader_module,
      label: descriptor.label,
      compilation_info,
    }
  }

  #[reentrant]
  #[required(1)]
  #[cppgc]
  fn create_compute_pipeline(
    &self,
    #[webidl] descriptor: super::compute_pipeline::GPUComputePipelineDescriptor,
  ) -> GPUComputePipeline {
    let label = descriptor.label.clone();
    let wgpu_descriptor = transform_compute_pipeline_descriptor(descriptor);
    let wgpu_compute_pipeline =
      self.wgpu_device.create_compute_pipeline(wgpu_descriptor);
    GPUComputePipeline {
      wgpu_compute_pipeline,
      label,
    }
  }

  #[reentrant]
  #[required(1)]
  #[cppgc]
  fn create_render_pipeline(
    &self,
    #[webidl] descriptor: super::render_pipeline::GPURenderPipelineDescriptor,
  ) -> Result<GPURenderPipeline, JsErrorBox> {
    let label = descriptor.label.clone();
    let wgpu_descriptor =
      self.transform_render_pipeline_descriptor(descriptor)?;
    let wgpu_render_pipeline =
      self.wgpu_device.create_render_pipeline(wgpu_descriptor);
    Ok(GPURenderPipeline {
      wgpu_render_pipeline,
      label,
    })
  }

  #[async_method(fake)]
  #[reentrant]
  #[required(1)]
  #[cppgc]
  #[global]
  fn create_compute_pipeline_async(
    &self,
    scope: &mut v8::HandleScope,
    #[webidl] descriptor: super::compute_pipeline::GPUComputePipelineDescriptor,
  ) -> v8::Global<v8::Promise> {
    let resolver = v8::PromiseResolver::new(scope).unwrap();
    let promise = resolver.get_promise(scope);

    let label = descriptor.label.clone();
    let wgpu_descriptor = transform_compute_pipeline_descriptor(descriptor);
    match self
      .wgpu_device
      .create_compute_pipeline_or_error(wgpu_descriptor)
    {
      Ok(wgpu_compute_pipeline) => {
        let pipeline = GPUComputePipeline {
          wgpu_compute_pipeline,
          label,
        };
        let val = make_cppgc_object(scope, pipeline).into();
        resolver.resolve(scope, val);
      }
      Err(err) => {
        let err = make_pipeline_error(
          scope,
          GPUPipelineErrorReason::Validation,
          &fmt_err(&err),
        );
        resolver.reject(scope, err.into());
      }
    }

    v8::Global::new(scope, promise)
  }

  #[async_method(fake)]
  #[reentrant]
  #[required(1)]
  #[cppgc]
  #[global]
  fn create_render_pipeline_async(
    &self,
    scope: &mut v8::HandleScope,
    #[webidl] descriptor: super::render_pipeline::GPURenderPipelineDescriptor,
  ) -> Result<v8::Global<v8::Promise>, JsErrorBox> {
    let label = descriptor.label.clone();

    let wgpu_descriptor =
      self.transform_render_pipeline_descriptor(descriptor)?;

    let resolver = v8::PromiseResolver::new(scope).unwrap();
    let promise = resolver.get_promise(scope);

    match self
      .wgpu_device
      .create_render_pipeline_or_error(wgpu_descriptor)
    {
      Ok(wgpu_render_pipeline) => {
        let render_pipeline = GPURenderPipeline {
          wgpu_render_pipeline,
          label,
        };
        let val = make_cppgc_object(scope, render_pipeline).into();
        resolver.resolve(scope, val);
      }
      Err(err) => {
        let err = make_pipeline_error(
          scope,
          GPUPipelineErrorReason::Validation,
          &fmt_err(&err),
        );
        resolver.reject(scope, err.into());
      }
    }
    Ok(v8::Global::new(scope, promise))
  }

  fn create_command_encoder<'a>(
    &self,
    scope: &mut v8::HandleScope<'a>,
    #[webidl] descriptor: Option<
      super::command_encoder::GPUCommandEncoderDescriptor,
    >,
  ) -> v8::Local<'a, v8::Object> {
    // Metal imposes a limit on the number of outstanding command buffers.
    // Attempting to create another command buffer after reaching that limit
    // will block, which can result in a deadlock if GC is required to
    // recover old command buffers. To encourage V8 to garbage collect
    // command buffers before that happens, we associate some external
    // memory with each command buffer.
    #[cfg(target_vendor = "apple")]
    const EXTERNAL_MEMORY_AMOUNT: i64 = 1 << 16;

    let label = descriptor.map(|d| d.label).unwrap_or_default();
    let wgpu_descriptor = wgpu_types::CommandEncoderDescriptor {
      label: Some(Cow::Owned(label.clone())),
    };

    #[cfg(target_vendor = "apple")]
    scope.adjust_amount_of_external_allocated_memory(EXTERNAL_MEMORY_AMOUNT);

    let wgpu_command_encoder =
      self.wgpu_device.create_command_encoder(&wgpu_descriptor);

    let encoder = GPUCommandEncoder {
      wgpu_command_encoder,
      label,
      #[cfg(target_vendor = "apple")]
      weak: std::sync::OnceLock::new(),
    };

    let obj = make_cppgc_object(scope, encoder);

    #[cfg(target_vendor = "apple")]
    {
      let finalizer = v8::Weak::with_finalizer(
        scope,
        obj,
        Box::new(|isolate: &mut v8::Isolate| {
          isolate.adjust_amount_of_external_allocated_memory(
            -EXTERNAL_MEMORY_AMOUNT,
          );
        }),
      );
      deno_core::cppgc::try_unwrap_cppgc_object::<GPUCommandEncoder>(
        scope,
        obj.into(),
      )
      .unwrap()
      .weak
      .set(finalizer)
      .unwrap();
    }

    obj
  }

  #[reentrant]
  #[required(1)]
  #[cppgc]
  fn create_render_bundle_encoder(
    &self,
    #[webidl]
    descriptor: super::render_bundle::GPURenderBundleEncoderDescriptor,
  ) -> Result<GPURenderBundleEncoder, JsErrorBox> {
    let wgpu_descriptor = wgpu_core::command::RenderBundleEncoderDescriptor {
      label: crate::transform_label(descriptor.label.clone()),
      color_formats: Cow::Owned(
        descriptor
          .color_formats
          .into_iter()
          .map(|format| format.into_option().map(Into::into))
          .collect::<Vec<_>>(),
      ),
      depth_stencil: descriptor.depth_stencil_format.map(|format| {
        wgpu_types::RenderBundleDepthStencil {
          format: format.into(),
          depth_read_only: descriptor.depth_read_only,
          stencil_read_only: descriptor.stencil_read_only,
        }
      }),
      sample_count: descriptor.sample_count,
      multiview: None,
    };

    // 1. Validate texture format required features of each non-null element of descriptor.colorFormats with this.[[device]].
    for &format in wgpu_descriptor.color_formats.iter().flatten() {
      self.validate_texture_format_required_feature(format)?;
    }

    // 2. If descriptor.depthStencilFormat is provided:
    if let Some(ds) = wgpu_descriptor.depth_stencil {
      // Validate texture format required features of descriptor.depthStencilFormat with this.[[device]].
      self.validate_texture_format_required_feature(ds.format)?;
    }

    let encoder = self
      .wgpu_device
      .create_render_bundle_encoder(&wgpu_descriptor);

    Ok(GPURenderBundleEncoder {
      encoder: RefCell::new(encoder),
      label: descriptor.label,
    })
  }

  #[reentrant]
  #[required(1)]
  #[cppgc]
  fn create_query_set(
    &self,
    #[webidl] descriptor: crate::query_set::GPUQuerySetDescriptor,
  ) -> Result<GPUQuerySet, JsErrorBox> {
    let wgpu_descriptor = wgpu_core::resource::QuerySetDescriptor {
      label: crate::transform_label(descriptor.label.clone()),
      ty: descriptor.r#type.clone().into(),
      count: descriptor.count,
    };

    if matches!(wgpu_descriptor.ty, wgpu_types::QueryType::Timestamp) {
      self
        .wgpu_device
        .require_features(wgpu_types::Features::TIMESTAMP_QUERY)
        .map_err(|err| {
          let err = fmt_err(&err);
          JsErrorBox::type_error(err)
        })?;
    }

    let wgpu_query_set = self.wgpu_device.create_query_set(&wgpu_descriptor);

    Ok(GPUQuerySet {
      wgpu_query_set,
      r#type: descriptor.r#type,
      count: descriptor.count,
      label: descriptor.label,
    })
  }

  #[getter]
  #[global]
  fn lost(&self) -> v8::Global<v8::Promise> {
    self.lost_promise.clone()
  }

  #[required(1)]
  #[undefined]
  fn push_error_scope(&self, #[webidl] filter: super::error::GPUErrorFilter) {
    self.wgpu_device.push_error_scope(filter.into());
  }

  #[async_method(fake)]
  #[global]
  fn pop_error_scope(
    &self,
    scope: &mut v8::HandleScope,
  ) -> Result<v8::Global<v8::Value>, JsErrorBox> {
    match self.wgpu_device.pop_error_scope() {
      Ok(maybe_error) => {
        let val = if let Some(err) = maybe_error {
          let err: GPUError = err.into();
          deno_core::error::to_v8_error(scope, &err)
        } else {
          v8::null(scope).cast::<v8::Value>()
        };
        Ok(v8::Global::new(scope, val))
      }
      Err(EmptyErrorScopeStack {}) => Err(JsErrorBox::new(
        "DOMExceptionOperationError",
        "There are no error scopes on the error scope stack",
      )),
    }
  }

  #[fast]
  fn start_capture(&self) {
    unsafe { self.wgpu_device.start_graphics_debugger_capture() };
  }
  #[fast]
  fn stop_capture(&self) {
    self
      .wgpu_device
      .poll(wgpu_types::PollType::wait_indefinitely())
      .unwrap();
    unsafe { self.wgpu_device.stop_graphics_debugger_capture() };
  }
}

fn transform_compute_pipeline_descriptor(
  descriptor: super::compute_pipeline::GPUComputePipelineDescriptor,
) -> wgpu_core::pipeline::ComputePipelineDescriptor<'static> {
  wgpu_core::pipeline::ComputePipelineDescriptor {
    label: crate::transform_label(descriptor.label.clone()),
    layout: descriptor.layout.into(),
    stage: ProgrammableStageDescriptor {
      module: descriptor.compute.module.wgpu_shader_module.clone(),
      entry_point: descriptor.compute.entry_point.map(Into::into),
      constants: descriptor.compute.constants.into_iter().collect(),
      zero_initialize_workgroup_memory: true,
    },
    cache: None,
  }
}

impl GPUDevice {
  fn transform_render_pipeline_descriptor(
    &self,
    descriptor: super::render_pipeline::GPURenderPipelineDescriptor,
  ) -> Result<
    wgpu_core::pipeline::ResolvedGeneralRenderPipelineDescriptor<'static>,
    JsErrorBox,
  > {
    let vertex = wgpu_core::pipeline::VertexState {
      stage: ProgrammableStageDescriptor {
        module: descriptor.vertex.module.wgpu_shader_module.clone(),
        entry_point: descriptor.vertex.entry_point.map(Into::into),
        constants: descriptor.vertex.constants.into_iter().collect(),
        zero_initialize_workgroup_memory: true,
      },
      buffers: Cow::Owned(
        descriptor
          .vertex
          .buffers
          .into_iter()
          .map(|b| {
            b.into_option().map(|layout| {
              wgpu_core::pipeline::VertexBufferLayout {
                array_stride: layout.array_stride,
                step_mode: layout.step_mode.into(),
                attributes: Cow::Owned(
                  layout
                    .attributes
                    .into_iter()
                    .map(|attr| wgpu_types::VertexAttribute {
                      format: attr.format.into(),
                      offset: attr.offset,
                      shader_location: attr.shader_location,
                    })
                    .collect(),
                ),
              }
            })
          })
          .collect(),
      ),
    };

    let primitive = wgpu_types::PrimitiveState {
      topology: descriptor.primitive.topology.into(),
      strip_index_format: descriptor
        .primitive
        .strip_index_format
        .map(Into::into),
      front_face: descriptor.primitive.front_face.into(),
      cull_mode: descriptor.primitive.cull_mode.into(),
      unclipped_depth: descriptor.primitive.unclipped_depth,
      polygon_mode: Default::default(),
      conservative: false,
    };

    let depth_stencil = descriptor
      .depth_stencil
      .map(|depth_stencil| -> Result<_, JsErrorBox> {
        let front = wgpu_types::StencilFaceState {
          compare: depth_stencil.stencil_front.compare.into(),
          fail_op: depth_stencil.stencil_front.fail_op.into(),
          depth_fail_op: depth_stencil.stencil_front.depth_fail_op.into(),
          pass_op: depth_stencil.stencil_front.pass_op.into(),
        };
        let back = wgpu_types::StencilFaceState {
          compare: depth_stencil.stencil_back.compare.into(),
          fail_op: depth_stencil.stencil_back.fail_op.into(),
          depth_fail_op: depth_stencil.stencil_back.depth_fail_op.into(),
          pass_op: depth_stencil.stencil_back.pass_op.into(),
        };

        let format = depth_stencil.format.into();
        self.validate_texture_format_required_feature(format)?;
        Ok(wgpu_types::DepthStencilState {
          format,
          depth_write_enabled: depth_stencil.depth_write_enabled,
          depth_compare: depth_stencil.depth_compare.map(Into::into),
          stencil: wgpu_types::StencilState {
            front,
            back,
            read_mask: depth_stencil.stencil_read_mask,
            write_mask: depth_stencil.stencil_write_mask,
          },
          bias: wgpu_types::DepthBiasState {
            constant: depth_stencil.depth_bias,
            slope_scale: depth_stencil.depth_bias_slope_scale,
            clamp: depth_stencil.depth_bias_clamp,
          },
        })
      })
      .transpose()?;

    let multisample = wgpu_types::MultisampleState {
      count: descriptor.multisample.count,
      mask: descriptor.multisample.mask as u64,
      alpha_to_coverage_enabled: descriptor
        .multisample
        .alpha_to_coverage_enabled,
    };

    let fragment = descriptor
      .fragment
      .map(|fragment| -> Result<_, JsErrorBox> {
        Ok(wgpu_core::pipeline::FragmentState {
          stage: ProgrammableStageDescriptor {
            module: fragment.module.wgpu_shader_module.clone(),
            entry_point: fragment.entry_point.map(Into::into),
            constants: fragment.constants.into_iter().collect(),
            zero_initialize_workgroup_memory: true,
          },
          targets: Cow::Owned(
            fragment
              .targets
              .into_iter()
              .map(|target| -> Result<_, JsErrorBox> {
                target
                  .into_option()
                  .map(|target| -> Result<_, JsErrorBox> {
                    let format = target.format.into();
                    self.validate_texture_format_required_feature(format)?;
                    Ok(wgpu_types::ColorTargetState {
                      format,
                      blend: target.blend.map(|blend| wgpu_types::BlendState {
                        color: wgpu_types::BlendComponent {
                          src_factor: blend.color.src_factor.into(),
                          dst_factor: blend.color.dst_factor.into(),
                          operation: blend.color.operation.into(),
                        },
                        alpha: wgpu_types::BlendComponent {
                          src_factor: blend.alpha.src_factor.into(),
                          dst_factor: blend.alpha.dst_factor.into(),
                          operation: blend.alpha.operation.into(),
                        },
                      }),
                      write_mask: target.write_mask.into(),
                    })
                  })
                  .transpose()
              })
              .collect::<Result<Vec<_>, JsErrorBox>>()?,
          ),
        })
      })
      .transpose()?;

    Ok(
      wgpu_core::pipeline::ResolvedGeneralRenderPipelineDescriptor {
        label: crate::transform_label(descriptor.label.clone()),
        layout: descriptor.layout.into(),
        vertex: wgpu_core::pipeline::RenderPipelineVertexProcessor::Vertex(
          vertex,
        ),
        primitive,
        depth_stencil,
        multisample,
        fragment,
        cache: None,
        multiview_mask: None,
      },
    )
  }
}

#[derive(Clone, Debug, Default, Hash, Eq, PartialEq)]
pub enum GPUDeviceLostReason {
  #[default]
  Unknown,
  Destroyed,
}

impl From<wgpu_types::DeviceLostReason> for GPUDeviceLostReason {
  fn from(value: wgpu_types::DeviceLostReason) -> Self {
    match value {
      wgpu_types::DeviceLostReason::Unknown => Self::Unknown,
      wgpu_types::DeviceLostReason::Destroyed => Self::Destroyed,
    }
  }
}

#[derive(Default)]
pub struct GPUDeviceLostInfo {
  pub reason: GPUDeviceLostReason,
}

impl GarbageCollected for GPUDeviceLostInfo {
  fn get_name(&self) -> &'static std::ffi::CStr {
    c"GPUDeviceLostInfo"
  }
}

#[op2]
impl GPUDeviceLostInfo {
  #[constructor]
  #[cppgc]
  fn constructor(_: bool) -> Result<GPUDeviceLostInfo, GPUGenericError> {
    Err(GPUGenericError::InvalidConstructor)
  }

  #[getter]
  #[string]
  fn reason(&self) -> &'static str {
    use GPUDeviceLostReason::*;
    match self.reason {
      Unknown => "unknown",
      Destroyed => "destroyed",
    }
  }

  #[getter]
  #[string]
  fn message(&self) -> &'static str {
    "device was lost"
  }
}

#[op2(fast)]
pub fn op_webgpu_device_start_capture(#[cppgc] device: &GPUDevice) {
  unsafe {
    device.wgpu_device.start_graphics_debugger_capture();
  }
}

#[op2(fast)]
pub fn op_webgpu_device_stop_capture(#[cppgc] device: &GPUDevice) {
  unsafe {
    device.wgpu_device.stop_graphics_debugger_capture();
  }
}
