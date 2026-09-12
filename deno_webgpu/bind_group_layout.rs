// Copyright 2018-2025 the Deno authors. MIT license.

use std::sync::Arc;

use deno_core::op2;
use deno_core::GarbageCollected;
use deno_core::WebIDL;
use wgpu_core::resource::Labeled as _;

use crate::error::GPUGenericError;
use crate::texture::GPUTextureViewDimension;
use crate::webidl::GPUShaderStageFlags;

pub struct GPUBindGroupLayout {
  pub wgpu_bind_group_layout: Arc<wgpu_core::binding_model::BindGroupLayout>,
}

impl deno_core::webidl::WebIdlInterfaceConverter for GPUBindGroupLayout {
  const NAME: &'static str = "GPUBindGroupLayout";
}

impl GarbageCollected for GPUBindGroupLayout {
  fn get_name(&self) -> &'static std::ffi::CStr {
    c"GPUBindGroupLayout"
  }
}

#[op2]
impl GPUBindGroupLayout {
  #[constructor]
  #[cppgc]
  fn constructor(_: bool) -> Result<GPUBindGroupLayout, GPUGenericError> {
    Err(GPUGenericError::InvalidConstructor)
  }

  #[getter]
  #[string]
  fn label(&self) -> String {
    self.wgpu_bind_group_layout.label().to_string()
  }
  #[setter]
  #[string]
  fn label(&self, #[webidl] _label: String) {
    // TODO(@crowlKats): no-op, needs wpgu to implement changing the label
  }
}

#[derive(WebIDL)]
#[webidl(dictionary)]
pub(crate) struct GPUBindGroupLayoutDescriptor {
  #[webidl(default = String::new())]
  pub label: String,
  pub entries: Vec<GPUBindGroupLayoutEntry>,
}

#[derive(WebIDL, Copy, Clone)]
#[webidl(dictionary)]
pub(crate) struct GPUBindGroupLayoutEntry {
  #[options(enforce_range = true)]
  pub binding: u32,
  pub visibility: GPUShaderStageFlags,
  pub buffer: Option<GPUBufferBindingLayout>,
  pub sampler: Option<GPUSamplerBindingLayout>,
  pub texture: Option<GPUTextureBindingLayout>,
  pub storage_texture: Option<GPUStorageTextureBindingLayout>,
  pub external_texture: Option<GPUExternalTextureBindingLayout>,
}

impl TryFrom<GPUBindGroupLayoutEntry> for wgpu_types::BindGroupLayoutEntry {
  type Error = wgpu_core::binding_model::CreateBindGroupLayoutError;

  fn try_from(value: GPUBindGroupLayoutEntry) -> Result<Self, Self::Error> {
    let GPUBindGroupLayoutEntry {
      binding,
      visibility,
      buffer,
      sampler,
      texture,
      storage_texture,
      external_texture,
    } = value;

    let mut binding_type = None;
    if let Some(buffer) = buffer {
      if binding_type.is_some() {
        return Err(wgpu_core::binding_model::CreateBindGroupLayoutError::Entry { binding, error: wgpu_core::binding_model::BindGroupLayoutEntryError::MultipleBindingTypesProvided });
      }
      binding_type = Some(wgpu_types::BindingType::Buffer {
        ty: buffer.r#type.into(),
        has_dynamic_offset: buffer.has_dynamic_offset,
        min_binding_size: wgpu_types::BufferSize::new(buffer.min_binding_size),
      });
    }

    if let Some(sampler) = sampler {
      if binding_type.is_some() {
        return Err(wgpu_core::binding_model::CreateBindGroupLayoutError::Entry { binding, error: wgpu_core::binding_model::BindGroupLayoutEntryError::MultipleBindingTypesProvided });
      }
      binding_type =
        Some(wgpu_types::BindingType::Sampler(sampler.r#type.into()));
    }

    if let Some(texture) = texture {
      if binding_type.is_some() {
        return Err(wgpu_core::binding_model::CreateBindGroupLayoutError::Entry { binding, error: wgpu_core::binding_model::BindGroupLayoutEntryError::MultipleBindingTypesProvided });
      }
      binding_type = Some(wgpu_types::BindingType::Texture {
        sample_type: texture.sample_type.into(),
        view_dimension: texture.view_dimension.into(),
        multisampled: texture.multisampled,
      });
    }

    if let Some(storage_texture) = storage_texture {
      if binding_type.is_some() {
        return Err(wgpu_core::binding_model::CreateBindGroupLayoutError::Entry { binding, error: wgpu_core::binding_model::BindGroupLayoutEntryError::MultipleBindingTypesProvided });
      }
      binding_type = Some(wgpu_types::BindingType::StorageTexture {
        access: storage_texture.access.into(),
        format: storage_texture.format.into(),
        view_dimension: storage_texture.view_dimension.into(),
      });
    }

    if let Some(GPUExternalTextureBindingLayout {}) = external_texture {
      if binding_type.is_some() {
        return Err(wgpu_core::binding_model::CreateBindGroupLayoutError::Entry { binding, error: wgpu_core::binding_model::BindGroupLayoutEntryError::MultipleBindingTypesProvided });
      }
      binding_type = Some(wgpu_types::BindingType::ExternalTexture);
    }

    Ok(wgpu_types::BindGroupLayoutEntry {
      binding,
      visibility: visibility.into(),
      ty: binding_type.ok_or(wgpu_core::binding_model::CreateBindGroupLayoutError::Entry { binding, error: wgpu_core::binding_model::BindGroupLayoutEntryError::NoBindingTypesProvided })?,
      count: None, // native-only
    })
  }
}

impl From<GPUBindGroupLayoutEntry>
  for wgpu_core::binding_model::BindGroupLayoutEntry
{
  fn from(value: GPUBindGroupLayoutEntry) -> Self {
    let buffer = value.buffer.map(|buffer| {
      wgpu_core::binding_model::BufferBindingLayout {
        ty: buffer.r#type.into(),
        has_dynamic_offset: buffer.has_dynamic_offset,
        min_binding_size: wgpu_types::BufferSize::new(buffer.min_binding_size),
      }
    });

    let sampler = value.sampler.map(|sampler| {
      wgpu_core::binding_model::SamplerBindingLayout {
        ty: sampler.r#type.into(),
      }
    });

    let texture = value.texture.map(|texture| {
      wgpu_core::binding_model::TextureBindingLayout {
        sample_type: texture.sample_type.into(),
        view_dimension: texture.view_dimension.into(),
        multisampled: texture.multisampled,
      }
    });

    let storage_texture = value.storage_texture.map(|storage_texture| {
      wgpu_core::binding_model::StorageTextureBindingLayout {
        access: storage_texture.access.into(),
        format: storage_texture.format.into(),
        view_dimension: storage_texture.view_dimension.into(),
      }
    });

    let external_texture = value
      .external_texture
      .map(|_| wgpu_core::binding_model::ExternalTextureBindingLayout {});

    Self {
      binding: value.binding,
      visibility: value.visibility.into(),
      buffer,
      sampler,
      texture,
      storage_texture,
      external_texture,
      acceleration_structure: None,
      count: None, // native-only
    }
  }
}

#[derive(WebIDL, Copy, Clone)]
#[webidl(dictionary)]
pub(crate) struct GPUBufferBindingLayout {
  #[webidl(default = GPUBufferBindingType::Uniform)]
  pub r#type: GPUBufferBindingType,
  #[webidl(default = false)]
  pub has_dynamic_offset: bool,
  #[webidl(default = 0)]
  pub min_binding_size: u64,
}

#[derive(WebIDL, Copy, Clone)]
#[webidl(enum)]
pub(crate) enum GPUBufferBindingType {
  Uniform,
  Storage,
  ReadOnlyStorage,
}

impl From<GPUBufferBindingType> for wgpu_types::BufferBindingType {
  fn from(value: GPUBufferBindingType) -> Self {
    match value {
      GPUBufferBindingType::Uniform => Self::Uniform,
      GPUBufferBindingType::Storage => Self::Storage { read_only: false },
      GPUBufferBindingType::ReadOnlyStorage => {
        Self::Storage { read_only: true }
      }
    }
  }
}

#[derive(WebIDL, Copy, Clone)]
#[webidl(dictionary)]
pub(crate) struct GPUSamplerBindingLayout {
  #[webidl(default = GPUSamplerBindingType::Filtering)]
  pub r#type: GPUSamplerBindingType,
}

#[derive(WebIDL, Copy, Clone)]
#[webidl(enum)]
pub(crate) enum GPUSamplerBindingType {
  Filtering,
  NonFiltering,
  Comparison,
}

impl From<GPUSamplerBindingType> for wgpu_types::SamplerBindingType {
  fn from(value: GPUSamplerBindingType) -> Self {
    match value {
      GPUSamplerBindingType::Filtering => Self::Filtering,
      GPUSamplerBindingType::NonFiltering => Self::NonFiltering,
      GPUSamplerBindingType::Comparison => Self::Comparison,
    }
  }
}

#[derive(WebIDL, Copy, Clone)]
#[webidl(dictionary)]
pub(crate) struct GPUTextureBindingLayout {
  #[webidl(default = GPUTextureSampleType::Float)]
  pub sample_type: GPUTextureSampleType,
  #[webidl(default = GPUTextureViewDimension::D2)]
  pub view_dimension: GPUTextureViewDimension,
  #[webidl(default = false)]
  pub multisampled: bool,
}

#[derive(WebIDL, Copy, Clone)]
#[webidl(enum)]
pub(crate) enum GPUTextureSampleType {
  Float,
  UnfilterableFloat,
  Depth,
  Sint,
  Uint,
}

impl From<GPUTextureSampleType> for wgpu_types::TextureSampleType {
  fn from(value: GPUTextureSampleType) -> Self {
    match value {
      GPUTextureSampleType::Float => Self::Float { filterable: true },
      GPUTextureSampleType::UnfilterableFloat => {
        Self::Float { filterable: false }
      }
      GPUTextureSampleType::Depth => Self::Depth,
      GPUTextureSampleType::Sint => Self::Sint,
      GPUTextureSampleType::Uint => Self::Uint,
    }
  }
}

#[derive(WebIDL, Copy, Clone)]
#[webidl(dictionary)]
pub(crate) struct GPUStorageTextureBindingLayout {
  #[webidl(default = GPUStorageTextureAccess::WriteOnly)]
  pub access: GPUStorageTextureAccess,
  pub format: super::texture::GPUTextureFormat,
  #[webidl(default = GPUTextureViewDimension::D2)]
  pub view_dimension: GPUTextureViewDimension,
}

#[derive(WebIDL, Copy, Clone)]
#[webidl(enum)]
pub(crate) enum GPUStorageTextureAccess {
  WriteOnly,
  ReadOnly,
  ReadWrite,
}

impl From<GPUStorageTextureAccess> for wgpu_types::StorageTextureAccess {
  fn from(value: GPUStorageTextureAccess) -> Self {
    match value {
      GPUStorageTextureAccess::WriteOnly => Self::WriteOnly,
      GPUStorageTextureAccess::ReadOnly => Self::ReadOnly,
      GPUStorageTextureAccess::ReadWrite => Self::ReadWrite,
    }
  }
}

#[derive(WebIDL, Copy, Clone)]
#[webidl(dictionary)]
pub(crate) struct GPUExternalTextureBindingLayout {}
