/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    backend, conv,
    device::{Device, DeviceDescriptor},
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Input, Token},
    id::{AdapterId, DeviceId, SurfaceId, Valid},
    LabelHelpers, LifeGuard, PrivateFeatures, Stored, DOWNLEVEL_WARNING_MESSAGE, MAX_BIND_GROUPS,
};

use wgt::{Backend, BackendBit, PowerPreference, BIND_BUFFER_ALIGNMENT};

use hal::{
    adapter::PhysicalDevice as _, queue::QueueFamily as _, window::Surface as _, Instance as _,
};
use thiserror::Error;

/// Size that is guaranteed to be available in push constants.
///
/// This is needed because non-vulkan backends might not
/// provide a push-constant size limit.
const MIN_PUSH_CONSTANT_SIZE: u32 = 128;

pub type RequestAdapterOptions = wgt::RequestAdapterOptions<SurfaceId>;

#[derive(Debug)]
pub struct Instance {
    #[cfg(vulkan)]
    pub vulkan: Option<gfx_backend_vulkan::Instance>,
    #[cfg(metal)]
    pub metal: Option<gfx_backend_metal::Instance>,
    #[cfg(dx12)]
    pub dx12: Option<gfx_backend_dx12::Instance>,
    #[cfg(dx11)]
    pub dx11: Option<gfx_backend_dx11::Instance>,
    #[cfg(gl)]
    pub gl: Option<gfx_backend_gl::Instance>,
}

impl Instance {
    pub fn new(name: &str, version: u32, backends: BackendBit) -> Self {
        backends_map! {
            let map = |(backend, backend_create)| {
                if backends.contains(backend.into()) {
                    backend_create(name, version).ok()
                } else {
                    None
                }
            };
            Self {
                #[cfg(vulkan)]
                vulkan: map((Backend::Vulkan, gfx_backend_vulkan::Instance::create)),
                #[cfg(metal)]
                metal: map((Backend::Metal, gfx_backend_metal::Instance::create)),
                #[cfg(dx12)]
                dx12: map((Backend::Dx12, gfx_backend_dx12::Instance::create)),
                #[cfg(dx11)]
                dx11: map((Backend::Dx11, gfx_backend_dx11::Instance::create)),
                #[cfg(gl)]
                gl: map((Backend::Gl, gfx_backend_gl::Instance::create)),
            }
        }
    }

    pub(crate) fn destroy_surface(&self, surface: Surface) {
        backends_map! {
            let map = |(surface_backend, self_backend)| {
                unsafe {
                    if let Some(suf) = surface_backend {
                        self_backend.as_ref().unwrap().destroy_surface(suf);
                    }
                }
            };

            #[cfg(vulkan)]
            map((surface.vulkan, &self.vulkan)),
            #[cfg(metal)]
            map((surface.metal, &self.metal)),
            #[cfg(dx12)]
            map((surface.dx12, &self.dx12)),
            #[cfg(dx11)]
            map((surface.dx11, &self.dx11)),
            #[cfg(gl)]
            map((surface.gl, &self.gl)),
        }
    }
}

type GfxSurface<B> = <B as hal::Backend>::Surface;

#[derive(Debug)]
pub struct Surface {
    #[cfg(vulkan)]
    pub vulkan: Option<GfxSurface<backend::Vulkan>>,
    #[cfg(metal)]
    pub metal: Option<GfxSurface<backend::Metal>>,
    #[cfg(dx12)]
    pub dx12: Option<GfxSurface<backend::Dx12>>,
    #[cfg(dx11)]
    pub dx11: Option<GfxSurface<backend::Dx11>>,
    #[cfg(gl)]
    pub gl: Option<GfxSurface<backend::Gl>>,
}

impl crate::hub::Resource for Surface {
    const TYPE: &'static str = "Surface";

    fn life_guard(&self) -> &LifeGuard {
        unreachable!()
    }

    fn label(&self) -> &str {
        "<Surface>"
    }
}

#[derive(Debug)]
pub struct Adapter<B: hal::Backend> {
    pub(crate) raw: hal::adapter::Adapter<B>,
    features: wgt::Features,
    pub(crate) private_features: PrivateFeatures,
    limits: wgt::Limits,
    downlevel: wgt::DownlevelProperties,
    life_guard: LifeGuard,
}

impl<B: GfxBackend> Adapter<B> {
    fn new(raw: hal::adapter::Adapter<B>) -> Self {
        profiling::scope!("Adapter::new");

        let adapter_features = raw.physical_device.features();
        let properties = raw.physical_device.properties();

        let mut features = wgt::Features::default()
            | wgt::Features::MAPPABLE_PRIMARY_BUFFERS
            | wgt::Features::PUSH_CONSTANTS
            | wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        features.set(
            wgt::Features::DEPTH_CLAMPING,
            adapter_features.contains(hal::Features::DEPTH_CLAMP),
        );
        features.set(
            wgt::Features::TEXTURE_COMPRESSION_BC,
            adapter_features.contains(hal::Features::FORMAT_BC),
        );
        features.set(
            wgt::Features::TEXTURE_COMPRESSION_ETC2,
            adapter_features.contains(hal::Features::FORMAT_ETC2),
        );
        features.set(
            wgt::Features::TEXTURE_COMPRESSION_ASTC_LDR,
            adapter_features.contains(hal::Features::FORMAT_ASTC_LDR),
        );
        features.set(
            wgt::Features::SAMPLED_TEXTURE_BINDING_ARRAY,
            adapter_features.contains(hal::Features::TEXTURE_DESCRIPTOR_ARRAY),
        );
        features.set(
            wgt::Features::SAMPLED_TEXTURE_ARRAY_DYNAMIC_INDEXING,
            adapter_features.contains(hal::Features::SHADER_SAMPLED_IMAGE_ARRAY_DYNAMIC_INDEXING),
        );
        features.set(
            wgt::Features::SAMPLED_TEXTURE_ARRAY_NON_UNIFORM_INDEXING,
            adapter_features.contains(hal::Features::SAMPLED_TEXTURE_DESCRIPTOR_INDEXING),
        );
        features.set(
            wgt::Features::UNSIZED_BINDING_ARRAY,
            adapter_features.contains(hal::Features::UNSIZED_DESCRIPTOR_ARRAY),
        );
        features.set(
            wgt::Features::MULTI_DRAW_INDIRECT,
            adapter_features.contains(hal::Features::MULTI_DRAW_INDIRECT),
        );
        features.set(
            wgt::Features::MULTI_DRAW_INDIRECT_COUNT,
            adapter_features.contains(hal::Features::DRAW_INDIRECT_COUNT),
        );
        features.set(
            wgt::Features::NON_FILL_POLYGON_MODE,
            adapter_features.contains(hal::Features::NON_FILL_POLYGON_MODE),
        );
        features.set(
            wgt::Features::TIMESTAMP_QUERY,
            properties.limits.timestamp_compute_and_graphics,
        );
        features.set(
            wgt::Features::PIPELINE_STATISTICS_QUERY,
            adapter_features.contains(hal::Features::PIPELINE_STATISTICS_QUERY),
        );
        features.set(
            wgt::Features::SHADER_FLOAT64,
            adapter_features.contains(hal::Features::SHADER_FLOAT64),
        );
        features.set(
            wgt::Features::CONSERVATIVE_RASTERIZATION,
            adapter_features.contains(hal::Features::CONSERVATIVE_RASTERIZATION),
        );
        #[cfg(not(target_os = "ios"))]
        //TODO: https://github.com/gfx-rs/gfx/issues/3346
        features.set(wgt::Features::ADDRESS_MODE_CLAMP_TO_BORDER, true);

        let private_features = PrivateFeatures {
            anisotropic_filtering: adapter_features.contains(hal::Features::SAMPLER_ANISOTROPY),
            texture_d24: raw
                .physical_device
                .format_properties(Some(hal::format::Format::X8D24Unorm))
                .optimal_tiling
                .contains(hal::format::ImageFeature::DEPTH_STENCIL_ATTACHMENT),
            texture_d24_s8: raw
                .physical_device
                .format_properties(Some(hal::format::Format::D24UnormS8Uint))
                .optimal_tiling
                .contains(hal::format::ImageFeature::DEPTH_STENCIL_ATTACHMENT),
        };

        let default_limits = wgt::Limits::default();

        // All these casts to u32 are safe as the underlying vulkan types are u32s.
        // If another backend provides larger limits than u32, we need to clamp them to u32::MAX.
        // TODO: fix all gfx-hal backends to produce limits we care about, and remove .max
        let desc_limits = &properties.limits.descriptor_limits;
        let limits = wgt::Limits {
            max_texture_dimension_1d: properties
                .limits
                .max_image_1d_size
                .max(default_limits.max_texture_dimension_1d),
            max_texture_dimension_2d: properties
                .limits
                .max_image_2d_size
                .max(default_limits.max_texture_dimension_1d),
            max_texture_dimension_3d: properties
                .limits
                .max_image_3d_size
                .max(default_limits.max_texture_dimension_1d),
            max_texture_array_layers: (properties.limits.max_image_array_layers as u32)
                .max(default_limits.max_texture_array_layers),
            max_bind_groups: (properties.limits.max_bound_descriptor_sets as u32)
                .min(MAX_BIND_GROUPS as u32)
                .max(default_limits.max_bind_groups),
            max_dynamic_uniform_buffers_per_pipeline_layout: desc_limits
                .max_descriptor_set_uniform_buffers_dynamic
                .max(default_limits.max_dynamic_uniform_buffers_per_pipeline_layout),
            max_dynamic_storage_buffers_per_pipeline_layout: desc_limits
                .max_descriptor_set_storage_buffers_dynamic
                .max(default_limits.max_dynamic_storage_buffers_per_pipeline_layout),
            max_sampled_textures_per_shader_stage: desc_limits
                .max_per_stage_descriptor_sampled_images
                .max(default_limits.max_sampled_textures_per_shader_stage),
            max_samplers_per_shader_stage: desc_limits
                .max_per_stage_descriptor_samplers
                .max(default_limits.max_samplers_per_shader_stage),
            max_storage_buffers_per_shader_stage: desc_limits
                .max_per_stage_descriptor_storage_buffers
                .max(default_limits.max_storage_buffers_per_shader_stage),
            max_storage_textures_per_shader_stage: desc_limits
                .max_per_stage_descriptor_storage_images
                .max(default_limits.max_storage_textures_per_shader_stage),
            max_uniform_buffers_per_shader_stage: desc_limits
                .max_per_stage_descriptor_uniform_buffers
                .max(default_limits.max_uniform_buffers_per_shader_stage),
            max_uniform_buffer_binding_size: (properties.limits.max_uniform_buffer_range as u32)
                .max(default_limits.max_uniform_buffer_binding_size),
            max_storage_buffer_binding_size: (properties.limits.max_storage_buffer_range as u32)
                .max(default_limits.max_storage_buffer_binding_size),
            max_vertex_buffers: (properties.limits.max_vertex_input_bindings as u32)
                .max(default_limits.max_vertex_buffers),
            max_vertex_attributes: (properties.limits.max_vertex_input_attributes as u32)
                .max(default_limits.max_vertex_attributes),
            max_vertex_buffer_array_stride: (properties.limits.max_vertex_input_binding_stride
                as u32)
                .max(default_limits.max_vertex_buffer_array_stride),
            max_push_constant_size: (properties.limits.max_push_constants_size as u32)
                .max(MIN_PUSH_CONSTANT_SIZE), // As an extension, the default is always 0, so define a separate minimum.
        };

        let mut downlevel_flags = wgt::DownlevelFlags::empty();
        downlevel_flags.set(
            wgt::DownlevelFlags::COMPUTE_SHADERS,
            properties.downlevel.compute_shaders,
        );
        downlevel_flags.set(
            wgt::DownlevelFlags::STORAGE_IMAGES,
            properties.downlevel.storage_images,
        );
        downlevel_flags.set(
            wgt::DownlevelFlags::READ_ONLY_DEPTH_STENCIL,
            properties.downlevel.read_only_depth_stencil,
        );
        downlevel_flags.set(
            wgt::DownlevelFlags::DEVICE_LOCAL_IMAGE_COPIES,
            properties.downlevel.device_local_image_copies,
        );
        downlevel_flags.set(
            wgt::DownlevelFlags::NON_POWER_OF_TWO_MIPMAPPED_TEXTURES,
            properties.downlevel.non_power_of_two_mipmapped_textures,
        );
        downlevel_flags.set(
            wgt::DownlevelFlags::ANISOTROPIC_FILTERING,
            private_features.anisotropic_filtering,
        );

        let downlevel = wgt::DownlevelProperties {
            flags: downlevel_flags,
            shader_model: match properties.downlevel.shader_model {
                hal::DownlevelShaderModel::ShaderModel2 => wgt::ShaderModel::Sm2,
                hal::DownlevelShaderModel::ShaderModel4 => wgt::ShaderModel::Sm4,
                hal::DownlevelShaderModel::ShaderModel5 => wgt::ShaderModel::Sm5,
            },
        };

        Self {
            raw,
            features,
            private_features,
            limits,
            downlevel,
            life_guard: LifeGuard::new("<Adapter>"),
        }
    }

    pub fn get_swap_chain_preferred_format(
        &self,
        surface: &mut Surface,
    ) -> Result<wgt::TextureFormat, GetSwapChainPreferredFormatError> {
        profiling::scope!("Adapter::get_swap_chain_preferred_format");

        let formats = {
            let surface = B::get_surface_mut(surface);
            let queue_family = &self.raw.queue_families[0];
            if !surface.supports_queue_family(queue_family) {
                return Err(GetSwapChainPreferredFormatError::UnsupportedQueueFamily);
            }
            surface.supported_formats(&self.raw.physical_device)
        };
        if let Some(formats) = formats {
            // Check the four formats mentioned in the WebGPU spec:
            // Bgra8UnormSrgb, Rgba8UnormSrgb, Bgra8Unorm, Rgba8Unorm
            // Also, prefer sRGB over linear as it is better in
            // representing perceived colors.
            if formats.contains(&hal::format::Format::Bgra8Srgb) {
                return Ok(wgt::TextureFormat::Bgra8UnormSrgb);
            }
            if formats.contains(&hal::format::Format::Rgba8Srgb) {
                return Ok(wgt::TextureFormat::Rgba8UnormSrgb);
            }
            if formats.contains(&hal::format::Format::Bgra8Unorm) {
                return Ok(wgt::TextureFormat::Bgra8Unorm);
            }
            if formats.contains(&hal::format::Format::Rgba8Unorm) {
                return Ok(wgt::TextureFormat::Rgba8Unorm);
            }
            return Err(GetSwapChainPreferredFormatError::NotFound);
        }

        // If no formats were returned, use Bgra8UnormSrgb
        Ok(wgt::TextureFormat::Bgra8UnormSrgb)
    }

    pub(crate) fn get_texture_format_features(
        &self,
        format: wgt::TextureFormat,
    ) -> wgt::TextureFormatFeatures {
        let texture_format_properties = self
            .raw
            .physical_device
            .format_properties(Some(conv::map_texture_format(
                format,
                self.private_features,
            )))
            .optimal_tiling;

        let mut allowed_usages = format.describe().guaranteed_format_features.allowed_usages;
        if texture_format_properties.contains(hal::format::ImageFeature::SAMPLED) {
            allowed_usages |= wgt::TextureUsage::SAMPLED;
        }
        if texture_format_properties.contains(hal::format::ImageFeature::STORAGE) {
            allowed_usages |= wgt::TextureUsage::STORAGE;
        }
        if texture_format_properties.contains(hal::format::ImageFeature::COLOR_ATTACHMENT) {
            allowed_usages |= wgt::TextureUsage::RENDER_ATTACHMENT;
        }
        if texture_format_properties.contains(hal::format::ImageFeature::DEPTH_STENCIL_ATTACHMENT) {
            allowed_usages |= wgt::TextureUsage::RENDER_ATTACHMENT;
        }

        let mut flags = wgt::TextureFormatFeatureFlags::empty();
        if texture_format_properties.contains(hal::format::ImageFeature::STORAGE_ATOMIC) {
            flags |= wgt::TextureFormatFeatureFlags::STORAGE_ATOMICS;
        }
        if texture_format_properties.contains(hal::format::ImageFeature::STORAGE_READ_WRITE) {
            flags |= wgt::TextureFormatFeatureFlags::STORAGE_READ_WRITE;
        }

        let filterable =
            texture_format_properties.contains(hal::format::ImageFeature::SAMPLED_LINEAR);

        wgt::TextureFormatFeatures {
            allowed_usages,
            flags,
            filterable,
        }
    }

    fn create_device(
        &self,
        self_id: AdapterId,
        desc: &DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
    ) -> Result<Device<B>, RequestDeviceError> {
        // Verify all features were exposed by the adapter
        if !self.features.contains(desc.features) {
            return Err(RequestDeviceError::UnsupportedFeature(
                desc.features - self.features,
            ));
        }

        if !self.downlevel.is_webgpu_compliant() {
            log::warn!("{}", DOWNLEVEL_WARNING_MESSAGE);
        }

        // Verify feature preconditions
        if desc
            .features
            .contains(wgt::Features::MAPPABLE_PRIMARY_BUFFERS)
            && self.raw.info.device_type == hal::adapter::DeviceType::DiscreteGpu
        {
            log::warn!("Feature MAPPABLE_PRIMARY_BUFFERS enabled on a discrete gpu. This is a massive performance footgun and likely not what you wanted");
        }

        let phd = &self.raw.physical_device;
        let available_features = phd.features();

        // Check features that are always needed
        let wishful_features = hal::Features::ROBUST_BUFFER_ACCESS
            | hal::Features::VERTEX_STORES_AND_ATOMICS
            | hal::Features::FRAGMENT_STORES_AND_ATOMICS
            | hal::Features::NDC_Y_UP
            | hal::Features::INDEPENDENT_BLENDING
            | hal::Features::SAMPLER_ANISOTROPY
            | hal::Features::IMAGE_CUBE_ARRAY;
        let mut enabled_features = available_features & wishful_features;
        if enabled_features != wishful_features {
            log::warn!(
                "Missing internal features: {:?}",
                wishful_features - enabled_features
            );
        }

        // Features
        enabled_features.set(
            hal::Features::DEPTH_CLAMP,
            desc.features.contains(wgt::Features::DEPTH_CLAMPING),
        );
        enabled_features.set(
            hal::Features::FORMAT_BC,
            desc.features
                .contains(wgt::Features::TEXTURE_COMPRESSION_BC),
        );
        enabled_features.set(
            hal::Features::FORMAT_ETC2,
            desc.features
                .contains(wgt::Features::TEXTURE_COMPRESSION_ETC2),
        );
        enabled_features.set(
            hal::Features::FORMAT_ASTC_LDR,
            desc.features
                .contains(wgt::Features::TEXTURE_COMPRESSION_ASTC_LDR),
        );
        enabled_features.set(
            hal::Features::TEXTURE_DESCRIPTOR_ARRAY,
            desc.features
                .contains(wgt::Features::SAMPLED_TEXTURE_BINDING_ARRAY),
        );
        enabled_features.set(
            hal::Features::SHADER_SAMPLED_IMAGE_ARRAY_DYNAMIC_INDEXING,
            desc.features
                .contains(wgt::Features::SAMPLED_TEXTURE_ARRAY_DYNAMIC_INDEXING),
        );
        enabled_features.set(
            hal::Features::SAMPLED_TEXTURE_DESCRIPTOR_INDEXING,
            desc.features
                .contains(wgt::Features::SAMPLED_TEXTURE_ARRAY_NON_UNIFORM_INDEXING),
        );
        enabled_features.set(
            hal::Features::UNSIZED_DESCRIPTOR_ARRAY,
            desc.features.contains(wgt::Features::UNSIZED_BINDING_ARRAY),
        );
        enabled_features.set(
            hal::Features::MULTI_DRAW_INDIRECT,
            desc.features.contains(wgt::Features::MULTI_DRAW_INDIRECT),
        );
        enabled_features.set(
            hal::Features::DRAW_INDIRECT_COUNT,
            desc.features
                .contains(wgt::Features::MULTI_DRAW_INDIRECT_COUNT),
        );
        enabled_features.set(
            hal::Features::NON_FILL_POLYGON_MODE,
            desc.features.contains(wgt::Features::NON_FILL_POLYGON_MODE),
        );
        enabled_features.set(
            hal::Features::PIPELINE_STATISTICS_QUERY,
            desc.features
                .contains(wgt::Features::PIPELINE_STATISTICS_QUERY),
        );
        enabled_features.set(
            hal::Features::SHADER_FLOAT64,
            desc.features.contains(wgt::Features::SHADER_FLOAT64),
        );
        enabled_features.set(
            hal::Features::CONSERVATIVE_RASTERIZATION,
            desc.features
                .contains(wgt::Features::CONSERVATIVE_RASTERIZATION),
        );

        let family = self
            .raw
            .queue_families
            .iter()
            .find(|family| family.queue_type().supports_graphics())
            .ok_or(RequestDeviceError::NoGraphicsQueue)?;

        let mut gpu =
            unsafe { phd.open(&[(family, &[1.0])], enabled_features) }.map_err(|err| {
                use hal::device::CreationError::*;
                match err {
                    DeviceLost => RequestDeviceError::DeviceLost,
                    InitializationFailed => RequestDeviceError::Internal,
                    OutOfMemory(_) => RequestDeviceError::OutOfMemory,
                    _ => panic!("failed to create `gfx-hal` device: {}", err),
                }
            })?;

        if let Some(_) = desc.label {
            //TODO
        }

        let limits = phd.properties().limits;
        assert_eq!(
            0,
            BIND_BUFFER_ALIGNMENT % limits.min_storage_buffer_offset_alignment,
            "Adapter storage buffer offset alignment not compatible with WGPU"
        );
        assert_eq!(
            0,
            BIND_BUFFER_ALIGNMENT % limits.min_uniform_buffer_offset_alignment,
            "Adapter uniform buffer offset alignment not compatible with WGPU"
        );
        if self.limits < desc.limits {
            return Err(RequestDeviceError::LimitsExceeded);
        }

        let mem_props = phd.memory_properties();

        Device::new(
            gpu.device,
            Stored {
                value: Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            gpu.queue_groups.swap_remove(0),
            mem_props,
            limits,
            self.private_features,
            self.downlevel,
            desc,
            trace_path,
        )
        .or(Err(RequestDeviceError::OutOfMemory))
    }
}

impl<B: hal::Backend> crate::hub::Resource for Adapter<B> {
    const TYPE: &'static str = "Adapter";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

#[derive(Clone, Debug, Error)]
pub enum GetSwapChainPreferredFormatError {
    #[error("no suitable format found")]
    NotFound,
    #[error("invalid adapter")]
    InvalidAdapter,
    #[error("invalid surface")]
    InvalidSurface,
    #[error("surface does not support the adapter's queue family")]
    UnsupportedQueueFamily,
}

#[derive(Clone, Debug, Error)]
/// Error when requesting a device from the adaptor
pub enum RequestDeviceError {
    #[error("parent adapter is invalid")]
    InvalidAdapter,
    #[error("connection to device was lost during initialization")]
    DeviceLost,
    #[error("device initialization failed due to implementation specific errors")]
    Internal,
    #[error("some of the requested device limits are not supported")]
    LimitsExceeded,
    #[error("device has no queue supporting graphics")]
    NoGraphicsQueue,
    #[error("not enough memory left")]
    OutOfMemory,
    #[error("unsupported features were requested: {0:?}")]
    UnsupportedFeature(wgt::Features),
}

pub enum AdapterInputs<'a, I> {
    IdSet(&'a [I], fn(&I) -> Backend),
    Mask(BackendBit, fn(Backend) -> I),
}

impl<I: Clone> AdapterInputs<'_, I> {
    fn find(&self, b: Backend) -> Option<I> {
        match *self {
            Self::IdSet(ids, ref fun) => ids.iter().find(|id| fun(id) == b).cloned(),
            Self::Mask(bits, ref fun) => {
                if bits.contains(b.into()) {
                    Some(fun(b))
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Clone, Debug, Error)]
#[error("adapter is invalid")]
pub struct InvalidAdapter;

#[derive(Clone, Debug, Error)]
pub enum RequestAdapterError {
    #[error("no suitable adapter found")]
    NotFound,
    #[error("surface {0:?} is invalid")]
    InvalidSurface(SurfaceId),
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    #[cfg(feature = "raw-window-handle")]
    pub fn instance_create_surface(
        &self,
        handle: &impl raw_window_handle::HasRawWindowHandle,
        id_in: Input<G, SurfaceId>,
    ) -> SurfaceId {
        profiling::scope!("Instance::create_surface");

        let surface = unsafe {
            backends_map! {
                let map = |inst| {
                    inst
                    .as_ref()
                    .and_then(|inst| inst.create_surface(handle).map_err(|e| {
                        log::warn!("Error: {:?}", e);
                    }).ok())
                };

                Surface {
                    #[cfg(vulkan)]
                    vulkan: map(&self.instance.vulkan),
                    #[cfg(metal)]
                    metal: map(&self.instance.metal),
                    #[cfg(dx12)]
                    dx12: map(&self.instance.dx12),
                    #[cfg(dx11)]
                    dx11: map(&self.instance.dx11),
                    #[cfg(gl)]
                    gl: map(&self.instance.gl),
                }
            }
        };

        let mut token = Token::root();
        let id = self.surfaces.prepare(id_in).assign(surface, &mut token);
        id.0
    }

    #[cfg(metal)]
    pub fn instance_create_surface_metal(
        &self,
        layer: *mut std::ffi::c_void,
        id_in: Input<G, SurfaceId>,
    ) -> SurfaceId {
        profiling::scope!("Instance::instance_create_surface_metal");

        let surface = Surface {
            #[cfg(feature = "gfx-backend-vulkan")]
            vulkan: None, //TODO: create_surface_from_layer ?
            metal: self.instance.metal.as_ref().map(|inst| {
                // we don't want to link to metal-rs for this
                #[allow(clippy::transmute_ptr_to_ref)]
                inst.create_surface_from_layer(unsafe { std::mem::transmute(layer) })
            }),
        };

        let mut token = Token::root();
        let id = self.surfaces.prepare(id_in).assign(surface, &mut token);
        id.0
    }

    pub fn surface_drop(&self, id: SurfaceId) {
        profiling::scope!("Surface::drop");
        let mut token = Token::root();
        let (surface, _) = self.surfaces.unregister(id, &mut token);
        self.instance.destroy_surface(surface.unwrap());
    }

    pub fn enumerate_adapters(&self, inputs: AdapterInputs<Input<G, AdapterId>>) -> Vec<AdapterId> {
        profiling::scope!("Instance::enumerate_adapters");

        let instance = &self.instance;
        let mut token = Token::root();
        let mut adapters = Vec::new();

        backends_map! {
            let map = |(instance_field, backend, backend_info, backend_hub)| {
                if let Some(ref inst) = *instance_field {
                    let hub = backend_hub(self);
                    if let Some(id_backend) = inputs.find(backend) {
                        for raw in inst.enumerate_adapters() {
                            let adapter = Adapter::new(raw);
                            log::info!("Adapter {} {:?}", backend_info, adapter.raw.info);
                            let id = hub.adapters
                                .prepare(id_backend.clone())
                                .assign(adapter, &mut token);
                            adapters.push(id.0);
                        }
                    }
                }
            };

            #[cfg(vulkan)]
            map((&instance.vulkan, Backend::Vulkan, "Vulkan", backend::Vulkan::hub)),
            #[cfg(metal)]
            map((&instance.metal, Backend::Metal, "Metal", backend::Metal::hub)),
            #[cfg(dx12)]
            map((&instance.dx12, Backend::Dx12, "Dx12", backend::Dx12::hub)),
            #[cfg(dx11)]
            map((&instance.dx11, Backend::Dx11, "Dx11", backend::Dx11::hub)),
            #[cfg(gl)]
            map((&instance.gl, Backend::Gl, "GL", backend::Gl::hub)),
        }

        adapters
    }

    pub fn request_adapter(
        &self,
        desc: &RequestAdapterOptions,
        inputs: AdapterInputs<Input<G, AdapterId>>,
    ) -> Result<AdapterId, RequestAdapterError> {
        profiling::scope!("Instance::pick_adapter");

        let instance = &self.instance;
        let mut token = Token::root();
        let (surface_guard, mut token) = self.surfaces.read(&mut token);
        let compatible_surface = desc
            .compatible_surface
            .map(|id| {
                surface_guard
                    .get(id)
                    .map_err(|_| RequestAdapterError::InvalidSurface(id))
            })
            .transpose()?;
        let mut device_types = Vec::new();

        let mut id_vulkan = inputs.find(Backend::Vulkan);
        let mut id_metal = inputs.find(Backend::Metal);
        let mut id_dx12 = inputs.find(Backend::Dx12);
        let mut id_dx11 = inputs.find(Backend::Dx11);
        let mut id_gl = inputs.find(Backend::Gl);

        backends_map! {
            let map = |(instance_backend, id_backend, surface_backend)| {
                match *instance_backend {
                    Some(ref inst) if id_backend.is_some() => {
                        let mut adapters = inst.enumerate_adapters();
                        if let Some(surface_backend) = compatible_surface.and_then(surface_backend) {
                            adapters.retain(|a| {
                                a.queue_families
                                    .iter()
                                    .find(|qf| qf.queue_type().supports_graphics())
                                    .map_or(false, |qf| surface_backend.supports_queue_family(qf))
                            });
                        }
                        device_types.extend(adapters.iter().map(|ad| ad.info.device_type.clone()));
                        adapters
                    }
                    _ => Vec::new(),
                }
            };

            // NB: The internal function definitions are a workaround for Rust
            // being weird with lifetimes for closure literals...
            #[cfg(vulkan)]
            let adapters_vk = map((&instance.vulkan, &id_vulkan, {
                fn surface_vulkan(surf: &Surface) -> Option<&GfxSurface<backend::Vulkan>> {
                    surf.vulkan.as_ref()
                }
                surface_vulkan
            }));
            #[cfg(metal)]
            let adapters_mtl = map((&instance.metal, &id_metal, {
                fn surface_metal(surf: &Surface) -> Option<&GfxSurface<backend::Metal>> {
                    surf.metal.as_ref()
                }
                surface_metal
            }));
            #[cfg(dx12)]
            let adapters_dx12 = map((&instance.dx12, &id_dx12, {
                fn surface_dx12(surf: &Surface) -> Option<&GfxSurface<backend::Dx12>> {
                    surf.dx12.as_ref()
                }
                surface_dx12
            }));
            #[cfg(dx11)]
            let adapters_dx11 = map((&instance.dx11, &id_dx11, {
                fn surface_dx11(surf: &Surface) -> Option<&GfxSurface<backend::Dx11>> {
                    surf.dx11.as_ref()
                }
                surface_dx11
            }));
            #[cfg(gl)]
            let adapters_gl = map((&instance.gl, &id_gl, {
                fn surface_gl(surf: &Surface) -> Option<&GfxSurface<backend::Gl>> {
                    surf.gl.as_ref()
                }
                surface_gl
            }));
        }

        if device_types.is_empty() {
            return Err(RequestAdapterError::NotFound);
        }

        let (mut integrated, mut discrete, mut virt, mut cpu, mut other) =
            (None, None, None, None, None);

        for (i, ty) in device_types.into_iter().enumerate() {
            match ty {
                hal::adapter::DeviceType::IntegratedGpu => {
                    integrated = integrated.or(Some(i));
                }
                hal::adapter::DeviceType::DiscreteGpu => {
                    discrete = discrete.or(Some(i));
                }
                hal::adapter::DeviceType::VirtualGpu => {
                    virt = virt.or(Some(i));
                }
                hal::adapter::DeviceType::Cpu => {
                    cpu = cpu.or(Some(i));
                }
                hal::adapter::DeviceType::Other => {
                    other = other.or(Some(i));
                }
            }
        }

        let preferred_gpu = match desc.power_preference {
            PowerPreference::LowPower => integrated.or(other).or(discrete).or(virt).or(cpu),
            PowerPreference::HighPerformance => discrete.or(other).or(integrated).or(virt).or(cpu),
        };

        let mut selected = preferred_gpu.unwrap_or(0);

        backends_map! {
            let map = |(info_adapter, id_backend, mut adapters_backend, backend_hub)| {
                if selected < adapters_backend.len() {
                    let adapter = Adapter::new(adapters_backend.swap_remove(selected));
                    log::info!("Adapter {} {:?}", info_adapter, adapter.raw.info);
                    let id = backend_hub(self).adapters
                        .prepare(id_backend.take().unwrap())
                        .assign(adapter, &mut token);
                    return Ok(id.0);
                }
                selected -= adapters_backend.len();
            };

            #[cfg(vulkan)]
            map(("Vulkan", &mut id_vulkan, adapters_vk, backend::Vulkan::hub)),
            #[cfg(metal)]
            map(("Metal", &mut id_metal, adapters_mtl, backend::Metal::hub)),
            #[cfg(dx12)]
            map(("Dx12", &mut id_dx12, adapters_dx12, backend::Dx12::hub)),
            #[cfg(dx11)]
            map(("Dx11", &mut id_dx11, adapters_dx11, backend::Dx11::hub)),
            #[cfg(gl)]
            map(("GL", &mut id_gl, adapters_gl, backend::Gl::hub)),
        }

        let _ = (
            selected,
            id_vulkan.take(),
            id_metal.take(),
            id_dx12.take(),
            id_dx11.take(),
            id_gl.take(),
        );
        log::warn!("Some adapters are present, but enumerating them failed!");
        Err(RequestAdapterError::NotFound)
    }

    pub fn adapter_get_info<B: GfxBackend>(
        &self,
        adapter_id: AdapterId,
    ) -> Result<wgt::AdapterInfo, InvalidAdapter> {
        profiling::scope!("Adapter::get_info");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        adapter_guard
            .get(adapter_id)
            .map(|adapter| conv::map_adapter_info(adapter.raw.info.clone(), adapter_id.backend()))
            .map_err(|_| InvalidAdapter)
    }

    pub fn adapter_get_texture_format_features<B: GfxBackend>(
        &self,
        adapter_id: AdapterId,
        format: wgt::TextureFormat,
    ) -> Result<wgt::TextureFormatFeatures, InvalidAdapter> {
        profiling::scope!("Adapter::get_texture_format_features");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        adapter_guard
            .get(adapter_id)
            .map(|adapter| adapter.get_texture_format_features(format))
            .map_err(|_| InvalidAdapter)
    }

    pub fn adapter_features<B: GfxBackend>(
        &self,
        adapter_id: AdapterId,
    ) -> Result<wgt::Features, InvalidAdapter> {
        profiling::scope!("Adapter::features");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        adapter_guard
            .get(adapter_id)
            .map(|adapter| adapter.features)
            .map_err(|_| InvalidAdapter)
    }

    pub fn adapter_limits<B: GfxBackend>(
        &self,
        adapter_id: AdapterId,
    ) -> Result<wgt::Limits, InvalidAdapter> {
        profiling::scope!("Adapter::limits");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        adapter_guard
            .get(adapter_id)
            .map(|adapter| adapter.limits.clone())
            .map_err(|_| InvalidAdapter)
    }

    pub fn adapter_downlevel_properties<B: GfxBackend>(
        &self,
        adapter_id: AdapterId,
    ) -> Result<wgt::DownlevelProperties, InvalidAdapter> {
        profiling::scope!("Adapter::downlevel_properties");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        adapter_guard
            .get(adapter_id)
            .map(|adapter| adapter.downlevel)
            .map_err(|_| InvalidAdapter)
    }

    pub fn adapter_drop<B: GfxBackend>(&self, adapter_id: AdapterId) {
        profiling::scope!("Adapter::drop");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (mut adapter_guard, _) = hub.adapters.write(&mut token);

        let free = match adapter_guard.get_mut(adapter_id) {
            Ok(adapter) => adapter.life_guard.ref_count.take().unwrap().load() == 1,
            Err(_) => true,
        };
        if free {
            hub.adapters
                .unregister_locked(adapter_id, &mut *adapter_guard);
        }
    }
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn adapter_request_device<B: GfxBackend>(
        &self,
        adapter_id: AdapterId,
        desc: &DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
        id_in: Input<G, DeviceId>,
    ) -> (DeviceId, Option<RequestDeviceError>) {
        profiling::scope!("Adapter::request_device");

        let hub = B::hub(self);
        let mut token = Token::root();
        let fid = hub.devices.prepare(id_in);

        let error = loop {
            let (adapter_guard, mut token) = hub.adapters.read(&mut token);
            let adapter = match adapter_guard.get(adapter_id) {
                Ok(adapter) => adapter,
                Err(_) => break RequestDeviceError::InvalidAdapter,
            };
            let device = match adapter.create_device(adapter_id, desc, trace_path) {
                Ok(device) => device,
                Err(e) => break e,
            };
            let id = fid.assign(device, &mut token);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }
}
