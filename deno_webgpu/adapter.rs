// Copyright 2018-2025 the Deno authors. MIT license.

use std::ops::BitOr;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::Mutex;

use deno_core::cppgc::make_cppgc_object;
use deno_core::cppgc::SameObject;
use deno_core::op2;
use deno_core::v8;
use deno_core::GarbageCollected;
use deno_core::JsRuntime;
use deno_core::OpState;
use deno_core::V8CrossThreadTaskSpawner;
use deno_core::WebIDL;
use wgpu_types::DeviceLostReason;

use super::device::GPUDevice;
use super::device::DEVICE_EXTERNAL_MEMORY_SIZE;
use super::queue::GPUQueue;
use crate::device::GPUDeviceLostInfo;
use crate::error::GPUError;
use crate::error::GPUGenericError;
use crate::webidl::GPUFeatureName;
use crate::LostPromiseResolverHM;
use crate::WeakDeviceHM;

#[derive(WebIDL)]
#[webidl(dictionary)]
pub(crate) struct GPURequestAdapterOptions {
  #[webidl(default = "core".into())]
  pub feature_level: String,
  pub power_preference: Option<GPUPowerPreference>,
  #[webidl(default = false)]
  pub force_fallback_adapter: bool,
}

#[derive(WebIDL)]
#[webidl(enum)]
pub(crate) enum GPUPowerPreference {
  LowPower,
  HighPerformance,
}

#[derive(WebIDL)]
#[webidl(dictionary)]
struct GPUQueueDescriptor {
  #[webidl(default = String::new())]
  label: String,
}

#[derive(WebIDL)]
#[webidl(dictionary)]
struct GPUDeviceDescriptor {
  #[webidl(default = String::new())]
  label: String,

  #[webidl(default = vec![])]
  required_features: Vec<GPUFeatureName>,
  #[webidl(default = Default::default())]
  #[options(enforce_range = true)]
  required_limits: indexmap::IndexMap<String, Option<u64>>,
  #[webidl(default = GPUQueueDescriptor { label: String::new() })]
  default_queue: GPUQueueDescriptor,
}

pub struct GPUAdapter {
  pub wgpu_adapter: Arc<wgpu_core::instance::Adapter>,

  pub features: SameObject<GPUSupportedFeatures>,
  pub limits: SameObject<GPUSupportedLimits>,
  pub info: Rc<SameObject<GPUAdapterInfo>>,
}

impl GarbageCollected for GPUAdapter {
  fn get_name(&self) -> &'static std::ffi::CStr {
    c"GPUAdapter"
  }
}

#[op2]
impl GPUAdapter {
  #[constructor]
  #[cppgc]
  fn constructor(_: bool) -> Result<GPUAdapter, GPUGenericError> {
    Err(GPUGenericError::InvalidConstructor)
  }

  #[getter]
  #[global]
  fn info(&self, scope: &mut v8::HandleScope) -> v8::Global<v8::Object> {
    self.info.get(scope, |_| {
      let info = self.wgpu_adapter.get_info();

      GPUAdapterInfo { info }
    })
  }

  #[getter]
  #[global]
  fn features(&self, scope: &mut v8::HandleScope) -> v8::Global<v8::Object> {
    self.features.get(scope, |scope| {
      let features = self.wgpu_adapter.features();
      // Only expose WebGPU features, not wgpu native-only features
      let features = features & wgpu_types::Features::all_webgpu_mask();
      GPUSupportedFeatures::new(scope, features)
    })
  }

  #[getter]
  #[global]
  fn limits(&self, scope: &mut v8::HandleScope) -> v8::Global<v8::Object> {
    self.limits.get(scope, |_| {
      let adapter_limits = self.wgpu_adapter.limits();
      GPUSupportedLimits(adapter_limits)
    })
  }

  #[async_method(fake)]
  #[reentrant]
  #[global]
  fn request_device(
    &self,
    state: &mut OpState,
    scope: &mut v8::HandleScope,
    #[webidl] descriptor: GPUDeviceDescriptor,
  ) -> Result<v8::Global<v8::Value>, CreateDeviceError> {
    let supported_features = self.wgpu_adapter.features();
    let required_features = descriptor
      .required_features
      .iter()
      .map(|f| wgpu_types::Features::from(*f))
      .fold(wgpu_types::Features::empty(), BitOr::bitor);

    // External textures are a required part of WebGPU, and `external-texture`
    // is not a WebGPU-defined feature. `wgpu` has it behind a feature for now,
    // because support is not complete. Allow applications to request that
    // feature even though it is not reported as an adapter-supported feature.
    //
    // There is probably not anything useful that Deno applications can do with
    // external textures, but it is useful to be able to enable it in
    // `cts_runner`.
    if !required_features
      .difference(supported_features | wgpu_types::Features::EXTERNAL_TEXTURE)
      .is_empty()
    {
      return Err(CreateDeviceError::RequiredFeaturesNotASubset);
    }

    // When support for compatibility mode is added, this will need to look
    // at whether the adapter is "compatibility-defaulting" or
    // "core-defaulting", and choose the appropriate set of defaults.
    //
    // Support for compatibility mode is tracked in
    // https://github.com/gfx-rs/wgpu/issues/8124.
    let required_limits = serde_json::from_value::<wgpu_types::Limits>(
      serde_json::to_value(descriptor.required_limits)?,
    )?
    .or_better_values_from(&wgpu_types::Limits::default());

    let trace = std::env::var_os("DENO_WEBGPU_TRACE")
      .map(|path| wgpu_types::Trace::Directory(std::path::PathBuf::from(path)))
      .unwrap_or_default();

    let wgpu_descriptor = wgpu_types::DeviceDescriptor {
      label: crate::transform_label(descriptor.label.clone()),
      required_features,
      required_limits,
      default_queue: wgpu_types::QueueDescriptor {
        label: crate::transform_label(descriptor.default_queue.label),
      },
      experimental_features: wgpu_types::ExperimentalFeatures::disabled(),
      memory_hints: Default::default(),
      trace,
    };

    let (wgpu_device, queue) =
      self.wgpu_adapter.request_device(&wgpu_descriptor)?;

    // Associate external memory with the device to encourage V8 to garbage
    // collect devices promptly.
    scope
      .adjust_amount_of_external_allocated_memory(DEVICE_EXTERNAL_MEMORY_SIZE);

    let error_handler =
      Rc::new(super::error::DeviceErrorHandler::new(wgpu_device.clone()));

    let wgpu_device_id = Arc::as_ptr(&wgpu_device) as usize;

    let lost_resolver = v8::PromiseResolver::new(scope).unwrap();
    let lost_promise = lost_resolver.get_promise(scope);
    let cross_thread_spawner =
      state.borrow::<V8CrossThreadTaskSpawner>().clone();
    state
      .borrow_mut::<LostPromiseResolverHM>()
      .insert(wgpu_device_id, v8::Global::new(scope, lost_resolver));
    let wake = Arc::new(Mutex::new(move |reason: DeviceLostReason| {
      cross_thread_spawner.spawn(move |scope| {
        let lost_resolver = JsRuntime::op_state_from(scope)
          .borrow_mut()
          .borrow_mut::<LostPromiseResolverHM>()
          .remove(&wgpu_device_id)
          .unwrap();
        let lost_resolver = v8::Local::new(scope, lost_resolver);
        let info = make_cppgc_object(
          scope,
          GPUDeviceLostInfo {
            reason: reason.into(),
          },
        );
        let info = v8::Local::new(scope, info);
        lost_resolver.resolve(scope, info.into());
      });
    }));
    wgpu_device.set_device_lost_closure(Box::new(move |reason, _| {
      wake.lock().unwrap()(reason);
    }));

    // Create the queue object eagerly so that the wgpu-core queue resource
    // gets cleaned up when the device is garbage collected, even if JS code
    // never accesses the queue property.
    let queue_obj = deno_core::cppgc::make_cppgc_object(
      scope,
      GPUQueue {
        label: descriptor.label.clone(),
        wgpu_queue: queue,
        wgpu_device: wgpu_device.clone(),
      },
    );
    let queue_obj = v8::Global::new(scope, queue_obj);

    let device = GPUDevice {
      wgpu_device: wgpu_device.clone(),
      label: descriptor.label,
      queue_obj,
      adapter_info: self.info.clone(),
      error_handler,
      wgpu_adapter: self.wgpu_adapter.clone(),
      lost_promise: v8::Global::new(scope, lost_promise),
      limits: SameObject::new(),
      features: SameObject::new(),
      weak: std::sync::OnceLock::new(),
    };
    let device = deno_core::cppgc::make_cppgc_object(scope, device);
    let weak_device = v8::Weak::new(scope, device);
    let event_target_setup = state.borrow::<crate::EventTargetSetup>();
    let webidl_brand = v8::Local::new(scope, event_target_setup.brand.clone());
    device.set(scope, webidl_brand, webidl_brand);
    let set_event_target_data =
      v8::Local::new(scope, event_target_setup.set_event_target_data.clone())
        .cast::<v8::Function>();
    let null = v8::null(scope);
    set_event_target_data.call(scope, null.into(), &[device.into()]);

    let finalizer = v8::Weak::with_finalizer(
      scope,
      device,
      Box::new(move |isolate| {
        isolate.adjust_amount_of_external_allocated_memory(
          -DEVICE_EXTERNAL_MEMORY_SIZE,
        );
        let op_state = JsRuntime::op_state_from(isolate);
        let mut op_state = op_state.borrow_mut();
        op_state
          .borrow_mut::<LostPromiseResolverHM>()
          .remove(&wgpu_device_id);
        op_state
          .borrow_mut::<WeakDeviceHM>()
          .remove(&wgpu_device_id);
      }),
    );

    // Now that the device is fully constructed, give the error handler a
    // weak reference to it, and store the finalizer weak reference.
    let device = device.cast::<v8::Value>();
    let device_ref =
      deno_core::cppgc::try_unwrap_cppgc_object::<GPUDevice>(scope, device)
        .unwrap();

    let cross_thread_spawner =
      state.borrow::<V8CrossThreadTaskSpawner>().clone();
    state
      .borrow_mut::<WeakDeviceHM>()
      .insert(wgpu_device_id, weak_device);
    let wake = Arc::new(Mutex::new(move |error: wgpu_types::error::Error| {
      cross_thread_spawner.spawn(move |scope| {
        let state = JsRuntime::op_state_from(&*scope);
        let state = state.borrow();
        let weak_device =
          state.borrow::<WeakDeviceHM>().get(&wgpu_device_id).unwrap();
        let err: GPUError = error.into();
        let weak_device = weak_device.clone();

        let Some(device) = weak_device.to_local(scope) else {
          // The device has already gone away, so we don't have
          // anywhere to report the error.
          return;
        };
        let key = v8::String::new(scope, "dispatchEvent").unwrap();
        let val = device.get(scope, key.into()).unwrap();
        let func =
          v8::Global::new(scope, val.try_cast::<v8::Function>().unwrap());
        let device = v8::Global::new(scope, device.cast::<v8::Value>());
        let error_event_class =
          state.borrow::<crate::ErrorEventClass>().0.clone();

        let error = deno_core::error::to_v8_error(scope, &err);

        let error_event_class =
          v8::Local::new(scope, error_event_class.clone());
        let constructor =
          v8::Local::<v8::Function>::try_from(error_event_class).unwrap();
        let kind = v8::String::new(scope, "uncapturederror").unwrap();

        let obj = v8::Object::new(scope);
        let key = v8::String::new(scope, "error").unwrap();
        obj.set(scope, key.into(), error);

        let event = constructor
          .new_instance(scope, &[kind.into(), obj.into()])
          .unwrap();

        let recv = v8::Local::new(scope, device);
        func.open(scope).call(scope, recv, &[event.into()]);
      });
    }));
    wgpu_device.on_uncaptured_error(Arc::new(move |error| {
      wake.lock().unwrap()(error);
    }));
    device_ref.weak.set(finalizer).unwrap();

    Ok(v8::Global::new(scope, device))
  }
}

#[derive(Debug, thiserror::Error, deno_error::JsError)]
pub enum CreateDeviceError {
  #[class(type)]
  #[error("requiredFeatures must be a subset of the adapter features")]
  RequiredFeaturesNotASubset,
  #[class(inherit)]
  #[error(transparent)]
  Serde(#[from] serde_json::Error),
  #[class("DOMExceptionOperationError")]
  #[error(transparent)]
  Device(#[from] wgpu_core::instance::RequestDeviceError),
}

pub struct GPUSupportedLimits(pub wgpu_types::Limits);

impl GarbageCollected for GPUSupportedLimits {
  fn get_name(&self) -> &'static std::ffi::CStr {
    c"GPUSupportedLimits"
  }
}

#[op2]
impl GPUSupportedLimits {
  #[constructor]
  #[cppgc]
  fn constructor(_: bool) -> Result<GPUSupportedLimits, GPUGenericError> {
    Err(GPUGenericError::InvalidConstructor)
  }

  #[getter]
  fn maxTextureDimension1D(&self) -> u32 {
    self.0.max_texture_dimension_1d
  }

  #[getter]
  fn maxTextureDimension2D(&self) -> u32 {
    self.0.max_texture_dimension_2d
  }

  #[getter]
  fn maxTextureDimension3D(&self) -> u32 {
    self.0.max_texture_dimension_3d
  }

  #[getter]
  fn maxTextureArrayLayers(&self) -> u32 {
    self.0.max_texture_array_layers
  }

  #[getter]
  fn maxBindGroups(&self) -> u32 {
    self.0.max_bind_groups
  }

  #[getter]
  fn maxBindGroupsPlusVertexBuffers(&self) -> u32 {
    self.0.max_bind_groups_plus_vertex_buffers
  }

  #[getter]
  fn maxBindingsPerBindGroup(&self) -> u32 {
    self.0.max_bindings_per_bind_group
  }

  #[getter]
  fn maxDynamicUniformBuffersPerPipelineLayout(&self) -> u32 {
    self.0.max_dynamic_uniform_buffers_per_pipeline_layout
  }

  #[getter]
  fn maxDynamicStorageBuffersPerPipelineLayout(&self) -> u32 {
    self.0.max_dynamic_storage_buffers_per_pipeline_layout
  }

  #[getter]
  fn maxSampledTexturesPerShaderStage(&self) -> u32 {
    self.0.max_sampled_textures_per_shader_stage
  }

  #[getter]
  fn maxSamplersPerShaderStage(&self) -> u32 {
    self.0.max_samplers_per_shader_stage
  }

  #[getter]
  fn maxStorageBuffersPerShaderStage(&self) -> u32 {
    self.0.max_storage_buffers_per_shader_stage
  }

  #[getter]
  fn maxStorageBuffersInVertexStage(&self) -> u32 {
    // TODO(https://github.com/gfx-rs/wgpu/issues/8748): InVertexStage limit
    // not implemented; return the PerShaderStage limit.
    self.0.max_storage_buffers_per_shader_stage
  }

  #[getter]
  fn maxStorageBuffersInFragmentStage(&self) -> u32 {
    // TODO(https://github.com/gfx-rs/wgpu/issues/8748): InFragmentStage limit
    // not implemented; return the PerShaderStage limit.
    self.0.max_storage_buffers_per_shader_stage
  }

  #[getter]
  fn maxStorageTexturesPerShaderStage(&self) -> u32 {
    self.0.max_storage_textures_per_shader_stage
  }

  #[getter]
  fn maxStorageTexturesInVertexStage(&self) -> u32 {
    // TODO(https://github.com/gfx-rs/wgpu/issues/8748): InVertexStage limit
    // not implemented; return the PerShaderStage limit.
    self.0.max_storage_textures_per_shader_stage
  }

  #[getter]
  fn maxStorageTexturesInFragmentStage(&self) -> u32 {
    // TODO(https://github.com/gfx-rs/wgpu/issues/8748): InFragmentStage limit
    // not implemented; return the PerShaderStage limit.
    self.0.max_storage_textures_per_shader_stage
  }

  #[getter]
  fn maxUniformBuffersPerShaderStage(&self) -> u32 {
    self.0.max_uniform_buffers_per_shader_stage
  }

  #[getter]
  #[number]
  fn maxUniformBufferBindingSize(&self) -> u64 {
    self.0.max_uniform_buffer_binding_size
  }

  #[getter]
  #[number]
  fn maxStorageBufferBindingSize(&self) -> u64 {
    self.0.max_storage_buffer_binding_size
  }

  #[getter]
  fn minUniformBufferOffsetAlignment(&self) -> u32 {
    self.0.min_uniform_buffer_offset_alignment
  }

  #[getter]
  fn minStorageBufferOffsetAlignment(&self) -> u32 {
    self.0.min_storage_buffer_offset_alignment
  }

  #[getter]
  fn maxVertexBuffers(&self) -> u32 {
    self.0.max_vertex_buffers
  }

  #[getter]
  #[number]
  fn maxBufferSize(&self) -> u64 {
    self.0.max_buffer_size
  }

  #[getter]
  fn maxVertexAttributes(&self) -> u32 {
    self.0.max_vertex_attributes
  }

  #[getter]
  fn maxVertexBufferArrayStride(&self) -> u32 {
    self.0.max_vertex_buffer_array_stride
  }

  #[getter]
  fn maxInterStageShaderVariables(&self) -> u32 {
    self.0.max_inter_stage_shader_variables
  }

  #[getter]
  fn maxColorAttachments(&self) -> u32 {
    self.0.max_color_attachments
  }

  #[getter]
  fn maxColorAttachmentBytesPerSample(&self) -> u32 {
    self.0.max_color_attachment_bytes_per_sample
  }

  #[getter]
  fn maxComputeWorkgroupStorageSize(&self) -> u32 {
    self.0.max_compute_workgroup_storage_size
  }

  #[getter]
  fn maxComputeInvocationsPerWorkgroup(&self) -> u32 {
    self.0.max_compute_invocations_per_workgroup
  }

  #[getter]
  fn maxComputeWorkgroupSizeX(&self) -> u32 {
    self.0.max_compute_workgroup_size_x
  }

  #[getter]
  fn maxComputeWorkgroupSizeY(&self) -> u32 {
    self.0.max_compute_workgroup_size_y
  }

  #[getter]
  fn maxComputeWorkgroupSizeZ(&self) -> u32 {
    self.0.max_compute_workgroup_size_z
  }

  #[getter]
  fn maxComputeWorkgroupsPerDimension(&self) -> u32 {
    self.0.max_compute_workgroups_per_dimension
  }

  #[getter]
  fn maxImmediateSize(&self) -> u32 {
    self.0.max_immediate_size
  }
}

pub struct GPUSupportedFeatures(v8::Global<v8::Value>);

impl GarbageCollected for GPUSupportedFeatures {
  fn get_name(&self) -> &'static std::ffi::CStr {
    c"GPUSupportedFeatures"
  }
}

impl GPUSupportedFeatures {
  pub fn new(
    scope: &mut v8::HandleScope,
    features: wgpu_types::Features,
  ) -> Self {
    let set = v8::Set::new(scope);

    for feature in features.iter() {
      let key = v8::String::new(scope, feature.as_str().unwrap()).unwrap();
      set.add(scope, key.into());
    }

    Self(v8::Global::new(scope, <v8::Local<v8::Value>>::from(set)))
  }
}

#[op2]
impl GPUSupportedFeatures {
  #[constructor]
  #[cppgc]
  fn constructor(_: bool) -> Result<GPUSupportedFeatures, GPUGenericError> {
    Err(GPUGenericError::InvalidConstructor)
  }

  #[global]
  #[symbol("setlike_set")]
  fn set(&self) -> v8::Global<v8::Value> {
    self.0.clone()
  }
}

pub struct GPUAdapterInfo {
  pub info: wgpu_types::AdapterInfo,
}

impl GarbageCollected for GPUAdapterInfo {
  fn get_name(&self) -> &'static std::ffi::CStr {
    c"GPUAdapterInfo"
  }
}

#[op2]
impl GPUAdapterInfo {
  #[constructor]
  #[cppgc]
  fn constructor(_: bool) -> Result<GPUAdapterInfo, GPUGenericError> {
    Err(GPUGenericError::InvalidConstructor)
  }

  #[getter]
  #[string]
  fn vendor(&self) -> String {
    self.info.vendor.to_string()
  }

  #[getter]
  #[string]
  fn architecture(&self) -> &'static str {
    "" // TODO(https://github.com/gfx-rs/wgpu/issues/8649): implement when wgpu has architecture detection
  }

  #[getter]
  #[string]
  fn device(&self) -> String {
    self.info.device.to_string()
  }

  #[getter]
  #[string]
  fn description(&self) -> String {
    self.info.name.clone()
  }

  #[getter]
  fn subgroup_min_size(&self) -> u32 {
    self.info.subgroup_min_size
  }

  #[getter]
  fn subgroup_max_size(&self) -> u32 {
    self.info.subgroup_max_size
  }

  #[getter]
  fn is_fallback_adapter(&self) -> bool {
    self.info.device_type == wgpu_types::DeviceType::Cpu
  }
}
