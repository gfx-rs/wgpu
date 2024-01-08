use super::conv;

use ash::{extensions::khr, vk};
use parking_lot::Mutex;

use std::{
    collections::BTreeMap,
    ffi::CStr,
    sync::{atomic::AtomicIsize, Arc},
};

fn depth_stencil_required_flags() -> vk::FormatFeatureFlags {
    vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
}

//TODO: const fn?
fn indexing_features() -> wgt::Features {
    wgt::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | wgt::Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
        | wgt::Features::PARTIALLY_BOUND_BINDING_ARRAY
}

/// Aggregate of the `vk::PhysicalDevice*Features` structs used by `gfx`.
#[derive(Debug, Default)]
pub struct PhysicalDeviceFeatures {
    core: vk::PhysicalDeviceFeatures,
    pub(super) descriptor_indexing: Option<vk::PhysicalDeviceDescriptorIndexingFeaturesEXT>,
    imageless_framebuffer: Option<vk::PhysicalDeviceImagelessFramebufferFeaturesKHR>,
    timeline_semaphore: Option<vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>,
    image_robustness: Option<vk::PhysicalDeviceImageRobustnessFeaturesEXT>,
    robustness2: Option<vk::PhysicalDeviceRobustness2FeaturesEXT>,
    multiview: Option<vk::PhysicalDeviceMultiviewFeaturesKHR>,
    sampler_ycbcr_conversion: Option<vk::PhysicalDeviceSamplerYcbcrConversionFeatures>,
    astc_hdr: Option<vk::PhysicalDeviceTextureCompressionASTCHDRFeaturesEXT>,
    shader_float16: Option<(
        vk::PhysicalDeviceShaderFloat16Int8Features,
        vk::PhysicalDevice16BitStorageFeatures,
    )>,
    acceleration_structure: Option<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>,
    buffer_device_address: Option<vk::PhysicalDeviceBufferDeviceAddressFeaturesKHR>,
    ray_query: Option<vk::PhysicalDeviceRayQueryFeaturesKHR>,
    zero_initialize_workgroup_memory:
        Option<vk::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures>,
}

// This is safe because the structs have `p_next: *mut c_void`, which we null out/never read.
unsafe impl Send for PhysicalDeviceFeatures {}
unsafe impl Sync for PhysicalDeviceFeatures {}

impl PhysicalDeviceFeatures {
    /// Add the members of `self` into `info.enabled_features` and its `p_next` chain.
    pub fn add_to_device_create_builder<'a>(
        &'a mut self,
        mut info: vk::DeviceCreateInfoBuilder<'a>,
    ) -> vk::DeviceCreateInfoBuilder<'a> {
        info = info.enabled_features(&self.core);
        if let Some(ref mut feature) = self.descriptor_indexing {
            info = info.push_next(feature);
        }
        if let Some(ref mut feature) = self.imageless_framebuffer {
            info = info.push_next(feature);
        }
        if let Some(ref mut feature) = self.timeline_semaphore {
            info = info.push_next(feature);
        }
        if let Some(ref mut feature) = self.image_robustness {
            info = info.push_next(feature);
        }
        if let Some(ref mut feature) = self.robustness2 {
            info = info.push_next(feature);
        }
        if let Some(ref mut feature) = self.astc_hdr {
            info = info.push_next(feature);
        }
        if let Some((ref mut f16_i8_feature, ref mut _16bit_feature)) = self.shader_float16 {
            info = info.push_next(f16_i8_feature);
            info = info.push_next(_16bit_feature);
        }
        if let Some(ref mut feature) = self.zero_initialize_workgroup_memory {
            info = info.push_next(feature);
        }
        if let Some(ref mut feature) = self.acceleration_structure {
            info = info.push_next(feature);
        }
        if let Some(ref mut feature) = self.buffer_device_address {
            info = info.push_next(feature);
        }
        if let Some(ref mut feature) = self.ray_query {
            info = info.push_next(feature);
        }
        info
    }

    /// Create a `PhysicalDeviceFeatures` that will be used to create a logical device.
    ///
    /// `requested_features` should be the same as what was used to generate `enabled_extensions`.
    fn from_extensions_and_requested_features(
        device_api_version: u32,
        enabled_extensions: &[&'static CStr],
        requested_features: wgt::Features,
        downlevel_flags: wgt::DownlevelFlags,
        private_caps: &super::PrivateCapabilities,
    ) -> Self {
        let needs_sampled_image_non_uniform = requested_features.contains(
            wgt::Features::TEXTURE_BINDING_ARRAY
                | wgt::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        );
        let needs_storage_buffer_non_uniform = requested_features.contains(
            wgt::Features::BUFFER_BINDING_ARRAY
                | wgt::Features::STORAGE_RESOURCE_BINDING_ARRAY
                | wgt::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        );
        let needs_uniform_buffer_non_uniform = requested_features.contains(
            wgt::Features::TEXTURE_BINDING_ARRAY
                | wgt::Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING,
        );
        let needs_storage_image_non_uniform = requested_features.contains(
            wgt::Features::TEXTURE_BINDING_ARRAY
                | wgt::Features::STORAGE_RESOURCE_BINDING_ARRAY
                | wgt::Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING,
        );
        let needs_partially_bound =
            requested_features.intersects(wgt::Features::PARTIALLY_BOUND_BINDING_ARRAY);

        Self {
            // vk::PhysicalDeviceFeatures is a struct composed of Bool32's while
            // Features is a bitfield so we need to map everything manually
            core: vk::PhysicalDeviceFeatures::builder()
                .robust_buffer_access(private_caps.robust_buffer_access)
                .independent_blend(downlevel_flags.contains(wgt::DownlevelFlags::INDEPENDENT_BLEND))
                .sample_rate_shading(
                    downlevel_flags.contains(wgt::DownlevelFlags::MULTISAMPLED_SHADING),
                )
                .image_cube_array(
                    downlevel_flags.contains(wgt::DownlevelFlags::CUBE_ARRAY_TEXTURES),
                )
                .draw_indirect_first_instance(
                    requested_features.contains(wgt::Features::INDIRECT_FIRST_INSTANCE),
                )
                //.dual_src_blend(requested_features.contains(wgt::Features::DUAL_SRC_BLENDING))
                .multi_draw_indirect(
                    requested_features.contains(wgt::Features::MULTI_DRAW_INDIRECT),
                )
                .fill_mode_non_solid(requested_features.intersects(
                    wgt::Features::POLYGON_MODE_LINE | wgt::Features::POLYGON_MODE_POINT,
                ))
                //.depth_bounds(requested_features.contains(wgt::Features::DEPTH_BOUNDS))
                //.alpha_to_one(requested_features.contains(wgt::Features::ALPHA_TO_ONE))
                //.multi_viewport(requested_features.contains(wgt::Features::MULTI_VIEWPORTS))
                .sampler_anisotropy(
                    downlevel_flags.contains(wgt::DownlevelFlags::ANISOTROPIC_FILTERING),
                )
                .texture_compression_etc2(
                    requested_features.contains(wgt::Features::TEXTURE_COMPRESSION_ETC2),
                )
                .texture_compression_astc_ldr(
                    requested_features.contains(wgt::Features::TEXTURE_COMPRESSION_ASTC),
                )
                .texture_compression_bc(
                    requested_features.contains(wgt::Features::TEXTURE_COMPRESSION_BC),
                )
                //.occlusion_query_precise(requested_features.contains(wgt::Features::PRECISE_OCCLUSION_QUERY))
                .pipeline_statistics_query(
                    requested_features.contains(wgt::Features::PIPELINE_STATISTICS_QUERY),
                )
                .vertex_pipeline_stores_and_atomics(
                    requested_features.contains(wgt::Features::VERTEX_WRITABLE_STORAGE),
                )
                .fragment_stores_and_atomics(
                    downlevel_flags.contains(wgt::DownlevelFlags::FRAGMENT_WRITABLE_STORAGE),
                )
                //.shader_image_gather_extended(
                //.shader_storage_image_extended_formats(
                .shader_uniform_buffer_array_dynamic_indexing(
                    requested_features.contains(wgt::Features::BUFFER_BINDING_ARRAY),
                )
                .shader_storage_buffer_array_dynamic_indexing(requested_features.contains(
                    wgt::Features::BUFFER_BINDING_ARRAY
                        | wgt::Features::STORAGE_RESOURCE_BINDING_ARRAY,
                ))
                .shader_sampled_image_array_dynamic_indexing(
                    requested_features.contains(wgt::Features::TEXTURE_BINDING_ARRAY),
                )
                .shader_storage_buffer_array_dynamic_indexing(requested_features.contains(
                    wgt::Features::TEXTURE_BINDING_ARRAY
                        | wgt::Features::STORAGE_RESOURCE_BINDING_ARRAY,
                ))
                //.shader_storage_image_array_dynamic_indexing(
                //.shader_clip_distance(requested_features.contains(wgt::Features::SHADER_CLIP_DISTANCE))
                //.shader_cull_distance(requested_features.contains(wgt::Features::SHADER_CULL_DISTANCE))
                .shader_float64(requested_features.contains(wgt::Features::SHADER_F64))
                //.shader_int64(requested_features.contains(wgt::Features::SHADER_INT64))
                .shader_int16(requested_features.contains(wgt::Features::SHADER_I16))
                //.shader_resource_residency(requested_features.contains(wgt::Features::SHADER_RESOURCE_RESIDENCY))
                .geometry_shader(requested_features.contains(wgt::Features::SHADER_PRIMITIVE_INDEX))
                .depth_clamp(requested_features.contains(wgt::Features::DEPTH_CLIP_CONTROL))
                .dual_src_blend(requested_features.contains(wgt::Features::DUAL_SOURCE_BLENDING))
                .build(),
            descriptor_indexing: if requested_features.intersects(indexing_features()) {
                Some(
                    vk::PhysicalDeviceDescriptorIndexingFeaturesEXT::builder()
                        .shader_sampled_image_array_non_uniform_indexing(
                            needs_sampled_image_non_uniform,
                        )
                        .shader_storage_image_array_non_uniform_indexing(
                            needs_storage_image_non_uniform,
                        )
                        .shader_uniform_buffer_array_non_uniform_indexing(
                            needs_uniform_buffer_non_uniform,
                        )
                        .shader_storage_buffer_array_non_uniform_indexing(
                            needs_storage_buffer_non_uniform,
                        )
                        .descriptor_binding_partially_bound(needs_partially_bound)
                        .build(),
                )
            } else {
                None
            },
            imageless_framebuffer: if device_api_version >= vk::API_VERSION_1_2
                || enabled_extensions.contains(&vk::KhrImagelessFramebufferFn::name())
            {
                Some(
                    vk::PhysicalDeviceImagelessFramebufferFeaturesKHR::builder()
                        .imageless_framebuffer(private_caps.imageless_framebuffers)
                        .build(),
                )
            } else {
                None
            },
            timeline_semaphore: if device_api_version >= vk::API_VERSION_1_2
                || enabled_extensions.contains(&vk::KhrTimelineSemaphoreFn::name())
            {
                Some(
                    vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR::builder()
                        .timeline_semaphore(private_caps.timeline_semaphores)
                        .build(),
                )
            } else {
                None
            },
            image_robustness: if device_api_version >= vk::API_VERSION_1_3
                || enabled_extensions.contains(&vk::ExtImageRobustnessFn::name())
            {
                Some(
                    vk::PhysicalDeviceImageRobustnessFeaturesEXT::builder()
                        .robust_image_access(private_caps.robust_image_access)
                        .build(),
                )
            } else {
                None
            },
            robustness2: if enabled_extensions.contains(&vk::ExtRobustness2Fn::name()) {
                // Note: enabling `robust_buffer_access2` isn't requires, strictly speaking
                // since we can enable `robust_buffer_access` all the time. But it improves
                // program portability, so we opt into it if they are supported.
                Some(
                    vk::PhysicalDeviceRobustness2FeaturesEXT::builder()
                        .robust_buffer_access2(private_caps.robust_buffer_access2)
                        .robust_image_access2(private_caps.robust_image_access2)
                        .build(),
                )
            } else {
                None
            },
            multiview: if device_api_version >= vk::API_VERSION_1_1
                || enabled_extensions.contains(&vk::KhrMultiviewFn::name())
            {
                Some(
                    vk::PhysicalDeviceMultiviewFeatures::builder()
                        .multiview(requested_features.contains(wgt::Features::MULTIVIEW))
                        .build(),
                )
            } else {
                None
            },
            sampler_ycbcr_conversion: if device_api_version >= vk::API_VERSION_1_1
                || enabled_extensions.contains(&vk::KhrSamplerYcbcrConversionFn::name())
            {
                Some(
                    vk::PhysicalDeviceSamplerYcbcrConversionFeatures::builder()
                        // .sampler_ycbcr_conversion(requested_features.contains(wgt::Features::TEXTURE_FORMAT_NV12))
                        .build(),
                )
            } else {
                None
            },
            astc_hdr: if enabled_extensions.contains(&vk::ExtTextureCompressionAstcHdrFn::name()) {
                Some(
                    vk::PhysicalDeviceTextureCompressionASTCHDRFeaturesEXT::builder()
                        .texture_compression_astc_hdr(true)
                        .build(),
                )
            } else {
                None
            },
            shader_float16: if requested_features.contains(wgt::Features::SHADER_F16) {
                Some((
                    vk::PhysicalDeviceShaderFloat16Int8Features::builder()
                        .shader_float16(true)
                        .build(),
                    vk::PhysicalDevice16BitStorageFeatures::builder()
                        .storage_buffer16_bit_access(true)
                        .uniform_and_storage_buffer16_bit_access(true)
                        .build(),
                ))
            } else {
                None
            },
            acceleration_structure: if enabled_extensions
                .contains(&vk::KhrAccelerationStructureFn::name())
            {
                Some(
                    vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder()
                        .acceleration_structure(true)
                        .build(),
                )
            } else {
                None
            },
            buffer_device_address: if enabled_extensions
                .contains(&vk::KhrBufferDeviceAddressFn::name())
            {
                Some(
                    vk::PhysicalDeviceBufferDeviceAddressFeaturesKHR::builder()
                        .buffer_device_address(true)
                        .build(),
                )
            } else {
                None
            },
            ray_query: if enabled_extensions.contains(&vk::KhrRayQueryFn::name()) {
                Some(
                    vk::PhysicalDeviceRayQueryFeaturesKHR::builder()
                        .ray_query(true)
                        .build(),
                )
            } else {
                None
            },
            zero_initialize_workgroup_memory: if device_api_version >= vk::API_VERSION_1_3
                || enabled_extensions.contains(&vk::KhrZeroInitializeWorkgroupMemoryFn::name())
            {
                Some(
                    vk::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures::builder()
                        .shader_zero_initialize_workgroup_memory(
                            private_caps.zero_initialize_workgroup_memory,
                        )
                        .build(),
                )
            } else {
                None
            },
        }
    }

    fn to_wgpu(
        &self,
        instance: &ash::Instance,
        phd: vk::PhysicalDevice,
        caps: &PhysicalDeviceCapabilities,
    ) -> (wgt::Features, wgt::DownlevelFlags) {
        use crate::auxil::db;
        use wgt::{DownlevelFlags as Df, Features as F};
        let mut features = F::empty()
            | F::SPIRV_SHADER_PASSTHROUGH
            | F::MAPPABLE_PRIMARY_BUFFERS
            | F::PUSH_CONSTANTS
            | F::ADDRESS_MODE_CLAMP_TO_BORDER
            | F::ADDRESS_MODE_CLAMP_TO_ZERO
            | F::TIMESTAMP_QUERY
            | F::TIMESTAMP_QUERY_INSIDE_PASSES
            | F::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | F::CLEAR_TEXTURE;

        let mut dl_flags = Df::COMPUTE_SHADERS
            | Df::BASE_VERTEX
            | Df::READ_ONLY_DEPTH_STENCIL
            | Df::NON_POWER_OF_TWO_MIPMAPPED_TEXTURES
            | Df::COMPARISON_SAMPLERS
            | Df::VERTEX_STORAGE
            | Df::FRAGMENT_STORAGE
            | Df::DEPTH_TEXTURE_AND_BUFFER_COPIES
            | Df::BUFFER_BINDINGS_NOT_16_BYTE_ALIGNED
            | Df::UNRESTRICTED_INDEX_BUFFER
            | Df::INDIRECT_EXECUTION
            | Df::VIEW_FORMATS
            | Df::UNRESTRICTED_EXTERNAL_TEXTURE_COPIES
            | Df::NONBLOCKING_QUERY_RESOLVE
            | Df::VERTEX_AND_INSTANCE_INDEX_RESPECTS_RESPECTIVE_FIRST_VALUE_IN_INDIRECT_DRAW;

        dl_flags.set(
            Df::SURFACE_VIEW_FORMATS,
            caps.supports_extension(vk::KhrSwapchainMutableFormatFn::name()),
        );
        dl_flags.set(Df::CUBE_ARRAY_TEXTURES, self.core.image_cube_array != 0);
        dl_flags.set(Df::ANISOTROPIC_FILTERING, self.core.sampler_anisotropy != 0);
        dl_flags.set(
            Df::FRAGMENT_WRITABLE_STORAGE,
            self.core.fragment_stores_and_atomics != 0,
        );
        dl_flags.set(Df::MULTISAMPLED_SHADING, self.core.sample_rate_shading != 0);
        dl_flags.set(Df::INDEPENDENT_BLEND, self.core.independent_blend != 0);
        dl_flags.set(
            Df::FULL_DRAW_INDEX_UINT32,
            self.core.full_draw_index_uint32 != 0,
        );
        dl_flags.set(Df::DEPTH_BIAS_CLAMP, self.core.depth_bias_clamp != 0);

        features.set(
            F::INDIRECT_FIRST_INSTANCE,
            self.core.draw_indirect_first_instance != 0,
        );
        //if self.core.dual_src_blend != 0
        features.set(F::MULTI_DRAW_INDIRECT, self.core.multi_draw_indirect != 0);
        features.set(F::POLYGON_MODE_LINE, self.core.fill_mode_non_solid != 0);
        features.set(F::POLYGON_MODE_POINT, self.core.fill_mode_non_solid != 0);
        //if self.core.depth_bounds != 0 {
        //if self.core.alpha_to_one != 0 {
        //if self.core.multi_viewport != 0 {
        features.set(
            F::TEXTURE_COMPRESSION_ETC2,
            self.core.texture_compression_etc2 != 0,
        );
        features.set(
            F::TEXTURE_COMPRESSION_ASTC,
            self.core.texture_compression_astc_ldr != 0,
        );
        features.set(
            F::TEXTURE_COMPRESSION_BC,
            self.core.texture_compression_bc != 0,
        );
        features.set(
            F::PIPELINE_STATISTICS_QUERY,
            self.core.pipeline_statistics_query != 0,
        );
        features.set(
            F::VERTEX_WRITABLE_STORAGE,
            self.core.vertex_pipeline_stores_and_atomics != 0,
        );
        //if self.core.shader_image_gather_extended != 0 {
        //if self.core.shader_storage_image_extended_formats != 0 {
        features.set(
            F::BUFFER_BINDING_ARRAY,
            self.core.shader_uniform_buffer_array_dynamic_indexing != 0,
        );
        features.set(
            F::TEXTURE_BINDING_ARRAY,
            self.core.shader_sampled_image_array_dynamic_indexing != 0,
        );
        features.set(F::SHADER_PRIMITIVE_INDEX, self.core.geometry_shader != 0);
        if Self::all_features_supported(
            &features,
            &[
                (
                    F::BUFFER_BINDING_ARRAY,
                    self.core.shader_storage_buffer_array_dynamic_indexing,
                ),
                (
                    F::TEXTURE_BINDING_ARRAY,
                    self.core.shader_storage_image_array_dynamic_indexing,
                ),
            ],
        ) {
            features.insert(F::STORAGE_RESOURCE_BINDING_ARRAY);
        }
        //if self.core.shader_storage_image_array_dynamic_indexing != 0 {
        //if self.core.shader_clip_distance != 0 {
        //if self.core.shader_cull_distance != 0 {
        features.set(F::SHADER_F64, self.core.shader_float64 != 0);
        //if self.core.shader_int64 != 0 {
        features.set(F::SHADER_I16, self.core.shader_int16 != 0);

        //if caps.supports_extension(vk::KhrSamplerMirrorClampToEdgeFn::name()) {
        //if caps.supports_extension(vk::ExtSamplerFilterMinmaxFn::name()) {
        features.set(
            F::MULTI_DRAW_INDIRECT_COUNT,
            caps.supports_extension(vk::KhrDrawIndirectCountFn::name()),
        );
        features.set(
            F::CONSERVATIVE_RASTERIZATION,
            caps.supports_extension(vk::ExtConservativeRasterizationFn::name()),
        );

        let intel_windows = caps.properties.vendor_id == db::intel::VENDOR && cfg!(windows);

        if let Some(ref descriptor_indexing) = self.descriptor_indexing {
            const STORAGE: F = F::STORAGE_RESOURCE_BINDING_ARRAY;
            if Self::all_features_supported(
                &features,
                &[
                    (
                        F::TEXTURE_BINDING_ARRAY,
                        descriptor_indexing.shader_sampled_image_array_non_uniform_indexing,
                    ),
                    (
                        F::BUFFER_BINDING_ARRAY | STORAGE,
                        descriptor_indexing.shader_storage_buffer_array_non_uniform_indexing,
                    ),
                ],
            ) {
                features.insert(F::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING);
            }
            if Self::all_features_supported(
                &features,
                &[
                    (
                        F::BUFFER_BINDING_ARRAY,
                        descriptor_indexing.shader_uniform_buffer_array_non_uniform_indexing,
                    ),
                    (
                        F::TEXTURE_BINDING_ARRAY | STORAGE,
                        descriptor_indexing.shader_storage_image_array_non_uniform_indexing,
                    ),
                ],
            ) {
                features.insert(F::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING);
            }
            if descriptor_indexing.descriptor_binding_partially_bound != 0 && !intel_windows {
                features |= F::PARTIALLY_BOUND_BINDING_ARRAY;
            }
        }

        features.set(F::DEPTH_CLIP_CONTROL, self.core.depth_clamp != 0);
        features.set(F::DUAL_SOURCE_BLENDING, self.core.dual_src_blend != 0);

        if let Some(ref multiview) = self.multiview {
            features.set(F::MULTIVIEW, multiview.multiview != 0);
        }

        features.set(
            F::TEXTURE_FORMAT_16BIT_NORM,
            is_format_16bit_norm_supported(instance, phd),
        );

        if let Some(ref astc_hdr) = self.astc_hdr {
            features.set(
                F::TEXTURE_COMPRESSION_ASTC_HDR,
                astc_hdr.texture_compression_astc_hdr != 0,
            );
        }

        if let Some((ref f16_i8, ref bit16)) = self.shader_float16 {
            features.set(
                F::SHADER_F16,
                f16_i8.shader_float16 != 0
                    && bit16.storage_buffer16_bit_access != 0
                    && bit16.uniform_and_storage_buffer16_bit_access != 0,
            );
        }

        let supports_depth_format = |format| {
            supports_format(
                instance,
                phd,
                format,
                vk::ImageTiling::OPTIMAL,
                depth_stencil_required_flags(),
            )
        };

        let texture_s8 = supports_depth_format(vk::Format::S8_UINT);
        let texture_d32 = supports_depth_format(vk::Format::D32_SFLOAT);
        let texture_d24_s8 = supports_depth_format(vk::Format::D24_UNORM_S8_UINT);
        let texture_d32_s8 = supports_depth_format(vk::Format::D32_SFLOAT_S8_UINT);

        let stencil8 = texture_s8 || texture_d24_s8;
        let depth24_plus_stencil8 = texture_d24_s8 || texture_d32_s8;

        dl_flags.set(
            Df::WEBGPU_TEXTURE_FORMAT_SUPPORT,
            stencil8 && depth24_plus_stencil8 && texture_d32,
        );

        features.set(F::DEPTH32FLOAT_STENCIL8, texture_d32_s8);

        features.set(
            F::RAY_TRACING_ACCELERATION_STRUCTURE,
            caps.supports_extension(vk::KhrDeferredHostOperationsFn::name())
                && caps.supports_extension(vk::KhrAccelerationStructureFn::name())
                && caps.supports_extension(vk::KhrBufferDeviceAddressFn::name()),
        );

        features.set(
            F::RAY_QUERY,
            caps.supports_extension(vk::KhrRayQueryFn::name()),
        );

        let rg11b10ufloat_renderable = supports_format(
            instance,
            phd,
            vk::Format::B10G11R11_UFLOAT_PACK32,
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::COLOR_ATTACHMENT
                | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND,
        );
        features.set(F::RG11B10UFLOAT_RENDERABLE, rg11b10ufloat_renderable);
        features.set(F::SHADER_UNUSED_VERTEX_OUTPUT, true);

        features.set(
            F::BGRA8UNORM_STORAGE,
            supports_bgra8unorm_storage(instance, phd, caps.device_api_version),
        );

        features.set(
            F::FLOAT32_FILTERABLE,
            is_float32_filterable_supported(instance, phd),
        );

        if let Some(ref _sampler_ycbcr_conversion) = self.sampler_ycbcr_conversion {
            features.set(
                F::TEXTURE_FORMAT_NV12,
                supports_format(
                    instance,
                    phd,
                    vk::Format::G8_B8R8_2PLANE_420_UNORM,
                    vk::ImageTiling::OPTIMAL,
                    vk::FormatFeatureFlags::SAMPLED_IMAGE
                        | vk::FormatFeatureFlags::TRANSFER_SRC
                        | vk::FormatFeatureFlags::TRANSFER_DST,
                ) && !caps
                    .driver
                    .map(|driver| driver.driver_id == vk::DriverId::MOLTENVK)
                    .unwrap_or_default(),
            );
        }

        (features, dl_flags)
    }

    fn all_features_supported(
        features: &wgt::Features,
        implications: &[(wgt::Features, vk::Bool32)],
    ) -> bool {
        implications
            .iter()
            .all(|&(flag, support)| !features.contains(flag) || support != 0)
    }
}

/// Information gathered about a physical device capabilities.
#[derive(Default, Debug)]
pub struct PhysicalDeviceCapabilities {
    supported_extensions: Vec<vk::ExtensionProperties>,
    properties: vk::PhysicalDeviceProperties,
    maintenance_3: Option<vk::PhysicalDeviceMaintenance3Properties>,
    descriptor_indexing: Option<vk::PhysicalDeviceDescriptorIndexingPropertiesEXT>,
    acceleration_structure: Option<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>,
    driver: Option<vk::PhysicalDeviceDriverPropertiesKHR>,
    /// The device API version.
    ///
    /// Which is the version of Vulkan supported for device-level functionality.
    ///
    /// It is associated with a `VkPhysicalDevice` and its children.
    device_api_version: u32,
}

// This is safe because the structs have `p_next: *mut c_void`, which we null out/never read.
unsafe impl Send for PhysicalDeviceCapabilities {}
unsafe impl Sync for PhysicalDeviceCapabilities {}

impl PhysicalDeviceCapabilities {
    pub fn properties(&self) -> vk::PhysicalDeviceProperties {
        self.properties
    }

    pub fn supports_extension(&self, extension: &CStr) -> bool {
        use crate::auxil::cstr_from_bytes_until_nul;
        self.supported_extensions
            .iter()
            .any(|ep| cstr_from_bytes_until_nul(&ep.extension_name) == Some(extension))
    }

    /// Map `requested_features` to the list of Vulkan extension strings required to create the logical device.
    fn get_required_extensions(&self, requested_features: wgt::Features) -> Vec<&'static CStr> {
        let mut extensions = Vec::new();

        // Note that quite a few extensions depend on the `VK_KHR_get_physical_device_properties2` instance extension.
        // We enable `VK_KHR_get_physical_device_properties2` unconditionally (if available).

        // Require `VK_KHR_swapchain`
        extensions.push(vk::KhrSwapchainFn::name());

        if self.device_api_version < vk::API_VERSION_1_1 {
            // Require either `VK_KHR_maintenance1` or `VK_AMD_negative_viewport_height`
            if self.supports_extension(vk::KhrMaintenance1Fn::name()) {
                extensions.push(vk::KhrMaintenance1Fn::name());
            } else {
                // `VK_AMD_negative_viewport_height` is obsoleted by `VK_KHR_maintenance1` and must not be enabled alongside it
                extensions.push(vk::AmdNegativeViewportHeightFn::name());
            }

            // Optional `VK_KHR_maintenance2`
            if self.supports_extension(vk::KhrMaintenance2Fn::name()) {
                extensions.push(vk::KhrMaintenance2Fn::name());
            }

            // Optional `VK_KHR_maintenance3`
            if self.supports_extension(vk::KhrMaintenance3Fn::name()) {
                extensions.push(vk::KhrMaintenance3Fn::name());
            }

            // Require `VK_KHR_storage_buffer_storage_class`
            extensions.push(vk::KhrStorageBufferStorageClassFn::name());

            // Require `VK_KHR_multiview` if the associated feature was requested
            if requested_features.contains(wgt::Features::MULTIVIEW) {
                extensions.push(vk::KhrMultiviewFn::name());
            }

            // Require `VK_KHR_sampler_ycbcr_conversion` if the associated feature was requested
            if requested_features.contains(wgt::Features::TEXTURE_FORMAT_NV12) {
                extensions.push(vk::KhrSamplerYcbcrConversionFn::name());
            }
        }

        if self.device_api_version < vk::API_VERSION_1_2 {
            // Optional `VK_KHR_image_format_list`
            if self.supports_extension(vk::KhrImageFormatListFn::name()) {
                extensions.push(vk::KhrImageFormatListFn::name());
            }

            // Optional `VK_KHR_imageless_framebuffer`
            if self.supports_extension(vk::KhrImagelessFramebufferFn::name()) {
                extensions.push(vk::KhrImagelessFramebufferFn::name());
                // Require `VK_KHR_maintenance2` due to it being a dependency
                if self.device_api_version < vk::API_VERSION_1_1 {
                    extensions.push(vk::KhrMaintenance2Fn::name());
                }
            }

            // Optional `VK_KHR_driver_properties`
            if self.supports_extension(vk::KhrDriverPropertiesFn::name()) {
                extensions.push(vk::KhrDriverPropertiesFn::name());
            }

            // Optional `VK_KHR_timeline_semaphore`
            if self.supports_extension(vk::KhrTimelineSemaphoreFn::name()) {
                extensions.push(vk::KhrTimelineSemaphoreFn::name());
            }

            // Require `VK_EXT_descriptor_indexing` if one of the associated features was requested
            if requested_features.intersects(indexing_features()) {
                extensions.push(vk::ExtDescriptorIndexingFn::name());
            }

            // Require `VK_KHR_shader_float16_int8` and `VK_KHR_16bit_storage` if the associated feature was requested
            if requested_features.contains(wgt::Features::SHADER_F16) {
                extensions.push(vk::KhrShaderFloat16Int8Fn::name());
                // `VK_KHR_16bit_storage` requires `VK_KHR_storage_buffer_storage_class`, however we require that one already
                if self.device_api_version < vk::API_VERSION_1_1 {
                    extensions.push(vk::Khr16bitStorageFn::name());
                }
            }

            //extensions.push(vk::KhrSamplerMirrorClampToEdgeFn::name());
            //extensions.push(vk::ExtSamplerFilterMinmaxFn::name());
        }

        if self.device_api_version < vk::API_VERSION_1_3 {
            // Optional `VK_EXT_image_robustness`
            if self.supports_extension(vk::ExtImageRobustnessFn::name()) {
                extensions.push(vk::ExtImageRobustnessFn::name());
            }
        }

        // Optional `VK_KHR_swapchain_mutable_format`
        if self.supports_extension(vk::KhrSwapchainMutableFormatFn::name()) {
            extensions.push(vk::KhrSwapchainMutableFormatFn::name());
        }

        // Optional `VK_EXT_robustness2`
        if self.supports_extension(vk::ExtRobustness2Fn::name()) {
            extensions.push(vk::ExtRobustness2Fn::name());
        }

        // Require `VK_KHR_draw_indirect_count` if the associated feature was requested
        // Even though Vulkan 1.2 has promoted the extension to core, we must require the extension to avoid
        // large amounts of spaghetti involved with using PhysicalDeviceVulkan12Features.
        if requested_features.contains(wgt::Features::MULTI_DRAW_INDIRECT_COUNT) {
            extensions.push(vk::KhrDrawIndirectCountFn::name());
        }

        // Require `VK_KHR_deferred_host_operations`, `VK_KHR_acceleration_structure` and `VK_KHR_buffer_device_address` if the feature `RAY_TRACING` was requested
        if requested_features.contains(wgt::Features::RAY_TRACING_ACCELERATION_STRUCTURE) {
            extensions.push(vk::KhrDeferredHostOperationsFn::name());
            extensions.push(vk::KhrAccelerationStructureFn::name());
            extensions.push(vk::KhrBufferDeviceAddressFn::name());
        }

        // Require `VK_KHR_ray_query` if the associated feature was requested
        if requested_features.contains(wgt::Features::RAY_QUERY) {
            extensions.push(vk::KhrRayQueryFn::name());
        }

        // Require `VK_EXT_conservative_rasterization` if the associated feature was requested
        if requested_features.contains(wgt::Features::CONSERVATIVE_RASTERIZATION) {
            extensions.push(vk::ExtConservativeRasterizationFn::name());
        }

        // Require `VK_KHR_portability_subset` on macOS/iOS
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        extensions.push(vk::KhrPortabilitySubsetFn::name());

        // Require `VK_EXT_texture_compression_astc_hdr` if the associated feature was requested
        if requested_features.contains(wgt::Features::TEXTURE_COMPRESSION_ASTC_HDR) {
            extensions.push(vk::ExtTextureCompressionAstcHdrFn::name());
        }

        extensions
    }

    fn to_wgpu_limits(&self) -> wgt::Limits {
        let limits = &self.properties.limits;

        let max_compute_workgroup_sizes = limits.max_compute_work_group_size;
        let max_compute_workgroups_per_dimension = limits.max_compute_work_group_count[0]
            .min(limits.max_compute_work_group_count[1])
            .min(limits.max_compute_work_group_count[2]);

        // Prevent very large buffers on mesa and most android devices.
        let is_nvidia = self.properties.vendor_id == crate::auxil::db::nvidia::VENDOR;
        let max_buffer_size =
            if (cfg!(target_os = "linux") || cfg!(target_os = "android")) && !is_nvidia {
                i32::MAX as u64
            } else {
                u64::MAX
            };

        wgt::Limits {
            max_texture_dimension_1d: limits.max_image_dimension1_d,
            max_texture_dimension_2d: limits.max_image_dimension2_d,
            max_texture_dimension_3d: limits.max_image_dimension3_d,
            max_texture_array_layers: limits.max_image_array_layers,
            max_bind_groups: limits
                .max_bound_descriptor_sets
                .min(crate::MAX_BIND_GROUPS as u32),
            max_bindings_per_bind_group: wgt::Limits::default().max_bindings_per_bind_group,
            max_dynamic_uniform_buffers_per_pipeline_layout: limits
                .max_descriptor_set_uniform_buffers_dynamic,
            max_dynamic_storage_buffers_per_pipeline_layout: limits
                .max_descriptor_set_storage_buffers_dynamic,
            max_sampled_textures_per_shader_stage: limits.max_per_stage_descriptor_sampled_images,
            max_samplers_per_shader_stage: limits.max_per_stage_descriptor_samplers,
            max_storage_buffers_per_shader_stage: limits.max_per_stage_descriptor_storage_buffers,
            max_storage_textures_per_shader_stage: limits.max_per_stage_descriptor_storage_images,
            max_uniform_buffers_per_shader_stage: limits.max_per_stage_descriptor_uniform_buffers,
            max_uniform_buffer_binding_size: limits
                .max_uniform_buffer_range
                .min(crate::auxil::MAX_I32_BINDING_SIZE),
            max_storage_buffer_binding_size: limits
                .max_storage_buffer_range
                .min(crate::auxil::MAX_I32_BINDING_SIZE),
            max_vertex_buffers: limits
                .max_vertex_input_bindings
                .min(crate::MAX_VERTEX_BUFFERS as u32),
            max_vertex_attributes: limits.max_vertex_input_attributes,
            max_vertex_buffer_array_stride: limits.max_vertex_input_binding_stride,
            max_push_constant_size: limits.max_push_constants_size,
            min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment as u32,
            min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment as u32,
            max_inter_stage_shader_components: limits
                .max_vertex_output_components
                .min(limits.max_fragment_input_components),
            max_compute_workgroup_storage_size: limits.max_compute_shared_memory_size,
            max_compute_invocations_per_workgroup: limits.max_compute_work_group_invocations,
            max_compute_workgroup_size_x: max_compute_workgroup_sizes[0],
            max_compute_workgroup_size_y: max_compute_workgroup_sizes[1],
            max_compute_workgroup_size_z: max_compute_workgroup_sizes[2],
            max_compute_workgroups_per_dimension,
            max_buffer_size,
            max_non_sampler_bindings: std::u32::MAX,
        }
    }

    fn to_hal_alignments(&self) -> crate::Alignments {
        let limits = &self.properties.limits;
        crate::Alignments {
            buffer_copy_offset: wgt::BufferSize::new(limits.optimal_buffer_copy_offset_alignment)
                .unwrap(),
            buffer_copy_pitch: wgt::BufferSize::new(limits.optimal_buffer_copy_row_pitch_alignment)
                .unwrap(),
        }
    }
}

impl super::InstanceShared {
    #[allow(trivial_casts)] // false positives
    fn inspect(
        &self,
        phd: vk::PhysicalDevice,
    ) -> (PhysicalDeviceCapabilities, PhysicalDeviceFeatures) {
        let capabilities = {
            let mut capabilities = PhysicalDeviceCapabilities::default();
            capabilities.supported_extensions =
                unsafe { self.raw.enumerate_device_extension_properties(phd).unwrap() };
            capabilities.properties = unsafe { self.raw.get_physical_device_properties(phd) };
            capabilities.device_api_version = capabilities.properties.api_version;

            if let Some(ref get_device_properties) = self.get_physical_device_properties {
                // Get these now to avoid borrowing conflicts later
                let supports_maintenance3 = capabilities.device_api_version >= vk::API_VERSION_1_1
                    || capabilities.supports_extension(vk::KhrMaintenance3Fn::name());
                let supports_descriptor_indexing = capabilities.device_api_version
                    >= vk::API_VERSION_1_2
                    || capabilities.supports_extension(vk::ExtDescriptorIndexingFn::name());
                let supports_driver_properties = capabilities.device_api_version
                    >= vk::API_VERSION_1_2
                    || capabilities.supports_extension(vk::KhrDriverPropertiesFn::name());

                let supports_acceleration_structure =
                    capabilities.supports_extension(vk::KhrAccelerationStructureFn::name());

                let mut builder = vk::PhysicalDeviceProperties2KHR::builder();
                if supports_maintenance3 {
                    capabilities.maintenance_3 =
                        Some(vk::PhysicalDeviceMaintenance3Properties::default());
                    builder = builder.push_next(capabilities.maintenance_3.as_mut().unwrap());
                }

                if supports_descriptor_indexing {
                    let next = capabilities
                        .descriptor_indexing
                        .insert(vk::PhysicalDeviceDescriptorIndexingPropertiesEXT::default());
                    builder = builder.push_next(next);
                }

                if supports_acceleration_structure {
                    let next = capabilities
                        .acceleration_structure
                        .insert(vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default());
                    builder = builder.push_next(next);
                }

                if supports_driver_properties {
                    let next = capabilities
                        .driver
                        .insert(vk::PhysicalDeviceDriverPropertiesKHR::default());
                    builder = builder.push_next(next);
                }

                let mut properties2 = builder.build();
                unsafe {
                    get_device_properties.get_physical_device_properties2(phd, &mut properties2);
                }

                if is_intel_igpu_outdated_for_robustness2(
                    capabilities.properties,
                    capabilities.driver,
                ) {
                    use crate::auxil::cstr_from_bytes_until_nul;
                    capabilities.supported_extensions.retain(|&x| {
                        cstr_from_bytes_until_nul(&x.extension_name)
                            != Some(vk::ExtRobustness2Fn::name())
                    });
                }
            };
            capabilities
        };

        let mut features = PhysicalDeviceFeatures::default();
        features.core = if let Some(ref get_device_properties) = self.get_physical_device_properties
        {
            let core = vk::PhysicalDeviceFeatures::default();
            let mut builder = vk::PhysicalDeviceFeatures2KHR::builder().features(core);

            // `VK_KHR_multiview` is promoted to 1.1
            if capabilities.device_api_version >= vk::API_VERSION_1_1
                || capabilities.supports_extension(vk::KhrMultiviewFn::name())
            {
                let next = features
                    .multiview
                    .insert(vk::PhysicalDeviceMultiviewFeatures::default());
                builder = builder.push_next(next);
            }

            // `VK_KHR_sampler_ycbcr_conversion` is promoted to 1.1
            if capabilities.device_api_version >= vk::API_VERSION_1_1
                || capabilities.supports_extension(vk::KhrSamplerYcbcrConversionFn::name())
            {
                let next = features
                    .sampler_ycbcr_conversion
                    .insert(vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default());
                builder = builder.push_next(next);
            }

            if capabilities.supports_extension(vk::ExtDescriptorIndexingFn::name()) {
                let next = features
                    .descriptor_indexing
                    .insert(vk::PhysicalDeviceDescriptorIndexingFeaturesEXT::default());
                builder = builder.push_next(next);
            }

            // `VK_KHR_imageless_framebuffer` is promoted to 1.2, but has no changes, so we can keep using the extension unconditionally.
            if capabilities.supports_extension(vk::KhrImagelessFramebufferFn::name()) {
                let next = features
                    .imageless_framebuffer
                    .insert(vk::PhysicalDeviceImagelessFramebufferFeaturesKHR::default());
                builder = builder.push_next(next);
            }

            // `VK_KHR_timeline_semaphore` is promoted to 1.2, but has no changes, so we can keep using the extension unconditionally.
            if capabilities.supports_extension(vk::KhrTimelineSemaphoreFn::name()) {
                let next = features
                    .timeline_semaphore
                    .insert(vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR::default());
                builder = builder.push_next(next);
            }

            if capabilities.supports_extension(vk::ExtImageRobustnessFn::name()) {
                let next = features
                    .image_robustness
                    .insert(vk::PhysicalDeviceImageRobustnessFeaturesEXT::default());
                builder = builder.push_next(next);
            }
            if capabilities.supports_extension(vk::ExtRobustness2Fn::name()) {
                let next = features
                    .robustness2
                    .insert(vk::PhysicalDeviceRobustness2FeaturesEXT::default());
                builder = builder.push_next(next);
            }
            if capabilities.supports_extension(vk::ExtTextureCompressionAstcHdrFn::name()) {
                let next = features
                    .astc_hdr
                    .insert(vk::PhysicalDeviceTextureCompressionASTCHDRFeaturesEXT::default());
                builder = builder.push_next(next);
            }
            if capabilities.supports_extension(vk::KhrShaderFloat16Int8Fn::name())
                && capabilities.supports_extension(vk::Khr16bitStorageFn::name())
            {
                let next = features.shader_float16.insert((
                    vk::PhysicalDeviceShaderFloat16Int8FeaturesKHR::default(),
                    vk::PhysicalDevice16BitStorageFeaturesKHR::default(),
                ));
                builder = builder.push_next(&mut next.0);
                builder = builder.push_next(&mut next.1);
            }
            if capabilities.supports_extension(vk::KhrAccelerationStructureFn::name()) {
                let next = features
                    .acceleration_structure
                    .insert(vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default());
                builder = builder.push_next(next);
            }

            // `VK_KHR_zero_initialize_workgroup_memory` is promoted to 1.3
            if capabilities.device_api_version >= vk::API_VERSION_1_3
                || capabilities.supports_extension(vk::KhrZeroInitializeWorkgroupMemoryFn::name())
            {
                let next = features
                    .zero_initialize_workgroup_memory
                    .insert(vk::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures::default());
                builder = builder.push_next(next);
            }

            let mut features2 = builder.build();
            unsafe {
                get_device_properties.get_physical_device_features2(phd, &mut features2);
            }
            features2.features
        } else {
            unsafe { self.raw.get_physical_device_features(phd) }
        };

        (capabilities, features)
    }
}

impl super::Instance {
    pub fn expose_adapter(
        &self,
        phd: vk::PhysicalDevice,
    ) -> Option<crate::ExposedAdapter<super::Api>> {
        use crate::auxil::cstr_from_bytes_until_nul;
        use crate::auxil::db;

        let (phd_capabilities, phd_features) = self.shared.inspect(phd);

        let info = wgt::AdapterInfo {
            name: {
                cstr_from_bytes_until_nul(&phd_capabilities.properties.device_name)
                    .and_then(|info| info.to_str().ok())
                    .unwrap_or("?")
                    .to_owned()
            },
            vendor: phd_capabilities.properties.vendor_id,
            device: phd_capabilities.properties.device_id,
            device_type: match phd_capabilities.properties.device_type {
                ash::vk::PhysicalDeviceType::OTHER => wgt::DeviceType::Other,
                ash::vk::PhysicalDeviceType::INTEGRATED_GPU => wgt::DeviceType::IntegratedGpu,
                ash::vk::PhysicalDeviceType::DISCRETE_GPU => wgt::DeviceType::DiscreteGpu,
                ash::vk::PhysicalDeviceType::VIRTUAL_GPU => wgt::DeviceType::VirtualGpu,
                ash::vk::PhysicalDeviceType::CPU => wgt::DeviceType::Cpu,
                _ => wgt::DeviceType::Other,
            },
            driver: {
                phd_capabilities
                    .driver
                    .as_ref()
                    .and_then(|driver| cstr_from_bytes_until_nul(&driver.driver_name))
                    .and_then(|name| name.to_str().ok())
                    .unwrap_or("?")
                    .to_owned()
            },
            driver_info: {
                phd_capabilities
                    .driver
                    .as_ref()
                    .and_then(|driver| cstr_from_bytes_until_nul(&driver.driver_info))
                    .and_then(|name| name.to_str().ok())
                    .unwrap_or("?")
                    .to_owned()
            },
            backend: wgt::Backend::Vulkan,
        };

        let (available_features, downlevel_flags) =
            phd_features.to_wgpu(&self.shared.raw, phd, &phd_capabilities);
        let mut workarounds = super::Workarounds::empty();
        {
            // TODO: only enable for particular devices
            workarounds |= super::Workarounds::SEPARATE_ENTRY_POINTS;
            workarounds.set(
                super::Workarounds::EMPTY_RESOLVE_ATTACHMENT_LISTS,
                phd_capabilities.properties.vendor_id == db::qualcomm::VENDOR,
            );
            workarounds.set(
                super::Workarounds::FORCE_FILL_BUFFER_WITH_SIZE_GREATER_4096_ALIGNED_OFFSET_16,
                phd_capabilities.properties.vendor_id == db::nvidia::VENDOR,
            );
        };

        if let Some(driver) = phd_capabilities.driver {
            if driver.conformance_version.major == 0 {
                if driver.driver_id == ash::vk::DriverId::MOLTENVK {
                    log::debug!("Adapter is not Vulkan compliant, but is MoltenVK, continuing");
                } else if self
                    .shared
                    .flags
                    .contains(wgt::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER)
                {
                    log::warn!("Adapter is not Vulkan compliant: {}", info.name);
                } else {
                    log::warn!(
                        "Adapter is not Vulkan compliant, hiding adapter: {}",
                        info.name
                    );
                    return None;
                }
            }
        }
        if phd_capabilities.device_api_version == vk::API_VERSION_1_0
            && !phd_capabilities.supports_extension(vk::KhrStorageBufferStorageClassFn::name())
        {
            log::warn!(
                "SPIR-V storage buffer class is not supported, hiding adapter: {}",
                info.name
            );
            return None;
        }
        if !phd_capabilities.supports_extension(vk::AmdNegativeViewportHeightFn::name())
            && !phd_capabilities.supports_extension(vk::KhrMaintenance1Fn::name())
            && phd_capabilities.device_api_version < vk::API_VERSION_1_1
        {
            log::warn!(
                "viewport Y-flip is not supported, hiding adapter: {}",
                info.name
            );
            return None;
        }

        let queue_families = unsafe {
            self.shared
                .raw
                .get_physical_device_queue_family_properties(phd)
        };
        let queue_flags = queue_families.first()?.queue_flags;
        if !queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            log::warn!("The first queue only exposes {:?}", queue_flags);
            return None;
        }

        let private_caps = super::PrivateCapabilities {
            flip_y_requires_shift: phd_capabilities.device_api_version >= vk::API_VERSION_1_1
                || phd_capabilities.supports_extension(vk::KhrMaintenance1Fn::name()),
            imageless_framebuffers: match phd_features.imageless_framebuffer {
                Some(features) => features.imageless_framebuffer == vk::TRUE,
                None => phd_features
                    .imageless_framebuffer
                    .map_or(false, |ext| ext.imageless_framebuffer != 0),
            },
            image_view_usage: phd_capabilities.device_api_version >= vk::API_VERSION_1_1
                || phd_capabilities.supports_extension(vk::KhrMaintenance2Fn::name()),
            timeline_semaphores: match phd_features.timeline_semaphore {
                Some(features) => features.timeline_semaphore == vk::TRUE,
                None => phd_features
                    .timeline_semaphore
                    .map_or(false, |ext| ext.timeline_semaphore != 0),
            },
            texture_d24: supports_format(
                &self.shared.raw,
                phd,
                vk::Format::X8_D24_UNORM_PACK32,
                vk::ImageTiling::OPTIMAL,
                depth_stencil_required_flags(),
            ),
            texture_d24_s8: supports_format(
                &self.shared.raw,
                phd,
                vk::Format::D24_UNORM_S8_UINT,
                vk::ImageTiling::OPTIMAL,
                depth_stencil_required_flags(),
            ),
            texture_s8: supports_format(
                &self.shared.raw,
                phd,
                vk::Format::S8_UINT,
                vk::ImageTiling::OPTIMAL,
                depth_stencil_required_flags(),
            ),
            non_coherent_map_mask: phd_capabilities.properties.limits.non_coherent_atom_size - 1,
            can_present: true,
            //TODO: make configurable
            robust_buffer_access: phd_features.core.robust_buffer_access != 0,
            robust_image_access: match phd_features.robustness2 {
                Some(ref f) => f.robust_image_access2 != 0,
                None => phd_features
                    .image_robustness
                    .map_or(false, |ext| ext.robust_image_access != 0),
            },
            robust_buffer_access2: phd_features
                .robustness2
                .as_ref()
                .map(|r| r.robust_buffer_access2 == 1)
                .unwrap_or_default(),
            robust_image_access2: phd_features
                .robustness2
                .as_ref()
                .map(|r| r.robust_image_access2 == 1)
                .unwrap_or_default(),
            zero_initialize_workgroup_memory: phd_features
                .zero_initialize_workgroup_memory
                .map_or(false, |ext| {
                    ext.shader_zero_initialize_workgroup_memory == vk::TRUE
                }),
            image_format_list: phd_capabilities.device_api_version >= vk::API_VERSION_1_2
                || phd_capabilities.supports_extension(vk::KhrImageFormatListFn::name()),
        };
        let capabilities = crate::Capabilities {
            limits: phd_capabilities.to_wgpu_limits(),
            alignments: phd_capabilities.to_hal_alignments(),
            downlevel: wgt::DownlevelCapabilities {
                flags: downlevel_flags,
                limits: wgt::DownlevelLimits {},
                shader_model: wgt::ShaderModel::Sm5, //TODO?
            },
        };

        let adapter = super::Adapter {
            raw: phd,
            instance: Arc::clone(&self.shared),
            //queue_families,
            known_memory_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::HOST_CACHED
                | vk::MemoryPropertyFlags::LAZILY_ALLOCATED,
            phd_capabilities,
            //phd_features,
            downlevel_flags,
            private_caps,
            workarounds,
        };

        Some(crate::ExposedAdapter {
            adapter,
            info,
            features: available_features,
            capabilities,
        })
    }
}

impl super::Adapter {
    pub fn raw_physical_device(&self) -> ash::vk::PhysicalDevice {
        self.raw
    }

    pub fn physical_device_capabilities(&self) -> &PhysicalDeviceCapabilities {
        &self.phd_capabilities
    }

    pub fn shared_instance(&self) -> &super::InstanceShared {
        &self.instance
    }

    pub fn required_device_extensions(&self, features: wgt::Features) -> Vec<&'static CStr> {
        let (supported_extensions, unsupported_extensions) = self
            .phd_capabilities
            .get_required_extensions(features)
            .iter()
            .partition::<Vec<&CStr>, _>(|&&extension| {
                self.phd_capabilities.supports_extension(extension)
            });

        if !unsupported_extensions.is_empty() {
            log::warn!("Missing extensions: {:?}", unsupported_extensions);
        }

        log::debug!("Supported extensions: {:?}", supported_extensions);
        supported_extensions
    }

    /// `features` must be the same features used to create `enabled_extensions`.
    pub fn physical_device_features(
        &self,
        enabled_extensions: &[&'static CStr],
        features: wgt::Features,
    ) -> PhysicalDeviceFeatures {
        PhysicalDeviceFeatures::from_extensions_and_requested_features(
            self.phd_capabilities.device_api_version,
            enabled_extensions,
            features,
            self.downlevel_flags,
            &self.private_caps,
        )
    }

    /// # Safety
    ///
    /// - `raw_device` must be created from this adapter.
    /// - `raw_device` must be created using `family_index`, `enabled_extensions` and `physical_device_features()`
    /// - `enabled_extensions` must be a superset of `required_device_extensions()`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn device_from_raw(
        &self,
        raw_device: ash::Device,
        handle_is_owned: bool,
        enabled_extensions: &[&'static CStr],
        features: wgt::Features,
        family_index: u32,
        queue_index: u32,
    ) -> Result<crate::OpenDevice<super::Api>, crate::DeviceError> {
        let mem_properties = {
            profiling::scope!("vkGetPhysicalDeviceMemoryProperties");
            unsafe {
                self.instance
                    .raw
                    .get_physical_device_memory_properties(self.raw)
            }
        };
        let memory_types =
            &mem_properties.memory_types[..mem_properties.memory_type_count as usize];
        let valid_ash_memory_types = memory_types.iter().enumerate().fold(0, |u, (i, mem)| {
            if self.known_memory_flags.contains(mem.property_flags) {
                u | (1 << i)
            } else {
                u
            }
        });

        let swapchain_fn = khr::Swapchain::new(&self.instance.raw, &raw_device);

        let indirect_count_fn = if enabled_extensions.contains(&khr::DrawIndirectCount::name()) {
            Some(khr::DrawIndirectCount::new(&self.instance.raw, &raw_device))
        } else {
            None
        };
        let timeline_semaphore_fn = if enabled_extensions.contains(&khr::TimelineSemaphore::name())
        {
            Some(super::ExtensionFn::Extension(khr::TimelineSemaphore::new(
                &self.instance.raw,
                &raw_device,
            )))
        } else if self.phd_capabilities.device_api_version >= vk::API_VERSION_1_2 {
            Some(super::ExtensionFn::Promoted)
        } else {
            None
        };
        let ray_tracing_fns = if enabled_extensions.contains(&khr::AccelerationStructure::name())
            && enabled_extensions.contains(&khr::BufferDeviceAddress::name())
        {
            Some(super::RayTracingDeviceExtensionFunctions {
                acceleration_structure: khr::AccelerationStructure::new(
                    &self.instance.raw,
                    &raw_device,
                ),
                buffer_device_address: khr::BufferDeviceAddress::new(
                    &self.instance.raw,
                    &raw_device,
                ),
            })
        } else {
            None
        };

        let naga_options = {
            use naga::back::spv;

            // The following capabilities are always available
            // see https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap52.html#spirvenv-capabilities
            let mut capabilities = vec![
                spv::Capability::Shader,
                spv::Capability::Matrix,
                spv::Capability::Sampled1D,
                spv::Capability::Image1D,
                spv::Capability::ImageQuery,
                spv::Capability::DerivativeControl,
                spv::Capability::StorageImageExtendedFormats,
            ];

            if self
                .downlevel_flags
                .contains(wgt::DownlevelFlags::CUBE_ARRAY_TEXTURES)
            {
                capabilities.push(spv::Capability::SampledCubeArray);
            }

            if self
                .downlevel_flags
                .contains(wgt::DownlevelFlags::MULTISAMPLED_SHADING)
            {
                capabilities.push(spv::Capability::SampleRateShading);
            }

            if features.contains(wgt::Features::MULTIVIEW) {
                capabilities.push(spv::Capability::MultiView);
            }

            if features.contains(wgt::Features::SHADER_PRIMITIVE_INDEX) {
                capabilities.push(spv::Capability::Geometry);
            }

            if features.intersects(
                wgt::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
                    | wgt::Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING,
            ) {
                capabilities.push(spv::Capability::ShaderNonUniform);
            }
            if features.contains(wgt::Features::BGRA8UNORM_STORAGE) {
                capabilities.push(spv::Capability::StorageImageWriteWithoutFormat);
            }

            if features.contains(wgt::Features::RAY_QUERY) {
                capabilities.push(spv::Capability::RayQueryKHR);
            }

            let mut flags = spv::WriterFlags::empty();
            flags.set(
                spv::WriterFlags::DEBUG,
                self.instance.flags.contains(wgt::InstanceFlags::DEBUG),
            );
            flags.set(
                spv::WriterFlags::LABEL_VARYINGS,
                self.phd_capabilities.properties.vendor_id != crate::auxil::db::qualcomm::VENDOR,
            );
            flags.set(
                spv::WriterFlags::FORCE_POINT_SIZE,
                //Note: we could technically disable this when we are compiling separate entry points,
                // and we know exactly that the primitive topology is not `PointList`.
                // But this requires cloning the `spv::Options` struct, which has heap allocations.
                true, // could check `super::Workarounds::SEPARATE_ENTRY_POINTS`
            );
            spv::Options {
                lang_version: (1, 0),
                flags,
                capabilities: Some(capabilities.iter().cloned().collect()),
                bounds_check_policies: naga::proc::BoundsCheckPolicies {
                    index: naga::proc::BoundsCheckPolicy::Restrict,
                    buffer: if self.private_caps.robust_buffer_access {
                        naga::proc::BoundsCheckPolicy::Unchecked
                    } else {
                        naga::proc::BoundsCheckPolicy::Restrict
                    },
                    image_load: if self.private_caps.robust_image_access {
                        naga::proc::BoundsCheckPolicy::Unchecked
                    } else {
                        naga::proc::BoundsCheckPolicy::Restrict
                    },
                    image_store: naga::proc::BoundsCheckPolicy::Unchecked,
                    // TODO: support bounds checks on binding arrays
                    binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
                },
                zero_initialize_workgroup_memory: if self
                    .private_caps
                    .zero_initialize_workgroup_memory
                {
                    spv::ZeroInitializeWorkgroupMemoryMode::Native
                } else {
                    spv::ZeroInitializeWorkgroupMemoryMode::Polyfill
                },
                // We need to build this separately for each invocation, so just default it out here
                binding_map: BTreeMap::default(),
                debug_info: None,
            }
        };

        let raw_queue = {
            profiling::scope!("vkGetDeviceQueue");
            unsafe { raw_device.get_device_queue(family_index, queue_index) }
        };

        let shared = Arc::new(super::DeviceShared {
            raw: raw_device,
            family_index,
            queue_index,
            raw_queue,
            handle_is_owned,
            instance: Arc::clone(&self.instance),
            physical_device: self.raw,
            enabled_extensions: enabled_extensions.into(),
            extension_fns: super::DeviceExtensionFunctions {
                draw_indirect_count: indirect_count_fn,
                timeline_semaphore: timeline_semaphore_fn,
                ray_tracing: ray_tracing_fns,
            },
            vendor_id: self.phd_capabilities.properties.vendor_id,
            timestamp_period: self.phd_capabilities.properties.limits.timestamp_period,
            private_caps: self.private_caps.clone(),
            workarounds: self.workarounds,
            render_passes: Mutex::new(Default::default()),
            framebuffers: Mutex::new(Default::default()),
        });
        let mut relay_semaphores = [vk::Semaphore::null(); 2];
        for sem in relay_semaphores.iter_mut() {
            unsafe {
                *sem = shared
                    .raw
                    .create_semaphore(&vk::SemaphoreCreateInfo::builder(), None)?
            };
        }
        let queue = super::Queue {
            raw: raw_queue,
            swapchain_fn,
            device: Arc::clone(&shared),
            family_index,
            relay_semaphores,
            relay_index: AtomicIsize::new(-1),
        };

        let mem_allocator = {
            let limits = self.phd_capabilities.properties.limits;
            let config = gpu_alloc::Config::i_am_prototyping(); //TODO
            let max_memory_allocation_size =
                if let Some(maintenance_3) = self.phd_capabilities.maintenance_3 {
                    maintenance_3.max_memory_allocation_size
                } else {
                    u64::max_value()
                };
            let properties = gpu_alloc::DeviceProperties {
                max_memory_allocation_count: limits.max_memory_allocation_count,
                max_memory_allocation_size,
                non_coherent_atom_size: limits.non_coherent_atom_size,
                memory_types: memory_types
                    .iter()
                    .map(|memory_type| gpu_alloc::MemoryType {
                        props: gpu_alloc::MemoryPropertyFlags::from_bits_truncate(
                            memory_type.property_flags.as_raw() as u8,
                        ),
                        heap: memory_type.heap_index,
                    })
                    .collect(),
                memory_heaps: mem_properties.memory_heaps
                    [..mem_properties.memory_heap_count as usize]
                    .iter()
                    .map(|&memory_heap| gpu_alloc::MemoryHeap {
                        size: memory_heap.size,
                    })
                    .collect(),
                buffer_device_address: enabled_extensions
                    .contains(&khr::BufferDeviceAddress::name()),
            };
            gpu_alloc::GpuAllocator::new(config, properties)
        };
        let desc_allocator = gpu_descriptor::DescriptorAllocator::new(
            if let Some(di) = self.phd_capabilities.descriptor_indexing {
                di.max_update_after_bind_descriptors_in_all_pools
            } else {
                0
            },
        );

        let device = super::Device {
            shared,
            mem_allocator: Mutex::new(mem_allocator),
            desc_allocator: Mutex::new(desc_allocator),
            valid_ash_memory_types,
            naga_options,
            #[cfg(feature = "renderdoc")]
            render_doc: Default::default(),
        };

        Ok(crate::OpenDevice { device, queue })
    }
}

impl crate::Adapter<super::Api> for super::Adapter {
    unsafe fn open(
        &self,
        features: wgt::Features,
        _limits: &wgt::Limits,
    ) -> Result<crate::OpenDevice<super::Api>, crate::DeviceError> {
        let enabled_extensions = self.required_device_extensions(features);
        let mut enabled_phd_features = self.physical_device_features(&enabled_extensions, features);

        let family_index = 0; //TODO
        let family_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(family_index)
            .queue_priorities(&[1.0])
            .build();
        let family_infos = [family_info];

        let str_pointers = enabled_extensions
            .iter()
            .map(|&s| {
                // Safe because `enabled_extensions` entries have static lifetime.
                s.as_ptr()
            })
            .collect::<Vec<_>>();

        let pre_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&family_infos)
            .enabled_extension_names(&str_pointers);
        let info = enabled_phd_features
            .add_to_device_create_builder(pre_info)
            .build();
        let raw_device = {
            profiling::scope!("vkCreateDevice");
            unsafe { self.instance.raw.create_device(self.raw, &info, None)? }
        };

        unsafe {
            self.device_from_raw(
                raw_device,
                true,
                &enabled_extensions,
                features,
                family_info.queue_family_index,
                0,
            )
        }
    }

    unsafe fn texture_format_capabilities(
        &self,
        format: wgt::TextureFormat,
    ) -> crate::TextureFormatCapabilities {
        use crate::TextureFormatCapabilities as Tfc;

        let vk_format = self.private_caps.map_texture_format(format);
        let properties = unsafe {
            self.instance
                .raw
                .get_physical_device_format_properties(self.raw, vk_format)
        };
        let features = properties.optimal_tiling_features;

        let mut flags = Tfc::empty();
        flags.set(
            Tfc::SAMPLED,
            features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE),
        );
        flags.set(
            Tfc::SAMPLED_LINEAR,
            features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR),
        );
        // flags.set(
        //     Tfc::SAMPLED_MINMAX,
        //     features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_MINMAX),
        // );
        flags.set(
            Tfc::STORAGE | Tfc::STORAGE_READ_WRITE,
            features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE),
        );
        flags.set(
            Tfc::STORAGE_ATOMIC,
            features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE_ATOMIC),
        );
        flags.set(
            Tfc::COLOR_ATTACHMENT,
            features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT),
        );
        flags.set(
            Tfc::COLOR_ATTACHMENT_BLEND,
            features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND),
        );
        flags.set(
            Tfc::DEPTH_STENCIL_ATTACHMENT,
            features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT),
        );
        flags.set(
            Tfc::COPY_SRC,
            features.intersects(vk::FormatFeatureFlags::TRANSFER_SRC),
        );
        flags.set(
            Tfc::COPY_DST,
            features.intersects(vk::FormatFeatureFlags::TRANSFER_DST),
        );
        // Vulkan is very permissive about MSAA
        flags.set(Tfc::MULTISAMPLE_RESOLVE, !format.is_compressed());

        // get the supported sample counts
        let format_aspect = crate::FormatAspects::from(format);
        let limits = self.phd_capabilities.properties.limits;

        let sample_flags = if format_aspect.contains(crate::FormatAspects::DEPTH) {
            limits
                .framebuffer_depth_sample_counts
                .min(limits.sampled_image_depth_sample_counts)
        } else if format_aspect.contains(crate::FormatAspects::STENCIL) {
            limits
                .framebuffer_stencil_sample_counts
                .min(limits.sampled_image_stencil_sample_counts)
        } else {
            let first_aspect = format_aspect
                .iter()
                .next()
                .expect("All texture should at least one aspect")
                .map();

            // We should never get depth or stencil out of this, due to the above.
            assert_ne!(first_aspect, wgt::TextureAspect::DepthOnly);
            assert_ne!(first_aspect, wgt::TextureAspect::StencilOnly);

            match format.sample_type(Some(first_aspect), None).unwrap() {
                wgt::TextureSampleType::Float { .. } => limits
                    .framebuffer_color_sample_counts
                    .min(limits.sampled_image_color_sample_counts),
                wgt::TextureSampleType::Sint | wgt::TextureSampleType::Uint => {
                    limits.sampled_image_integer_sample_counts
                }
                _ => unreachable!(),
            }
        };

        flags.set(
            Tfc::MULTISAMPLE_X2,
            sample_flags.contains(vk::SampleCountFlags::TYPE_2),
        );
        flags.set(
            Tfc::MULTISAMPLE_X4,
            sample_flags.contains(vk::SampleCountFlags::TYPE_4),
        );
        flags.set(
            Tfc::MULTISAMPLE_X8,
            sample_flags.contains(vk::SampleCountFlags::TYPE_8),
        );
        flags.set(
            Tfc::MULTISAMPLE_X16,
            sample_flags.contains(vk::SampleCountFlags::TYPE_16),
        );

        flags
    }

    unsafe fn surface_capabilities(
        &self,
        surface: &super::Surface,
    ) -> Option<crate::SurfaceCapabilities> {
        if !self.private_caps.can_present {
            return None;
        }
        let queue_family_index = 0; //TODO
        {
            profiling::scope!("vkGetPhysicalDeviceSurfaceSupportKHR");
            match unsafe {
                surface.functor.get_physical_device_surface_support(
                    self.raw,
                    queue_family_index,
                    surface.raw,
                )
            } {
                Ok(true) => (),
                Ok(false) => return None,
                Err(e) => {
                    log::error!("get_physical_device_surface_support: {}", e);
                    return None;
                }
            }
        }

        let caps = {
            profiling::scope!("vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
            match unsafe {
                surface
                    .functor
                    .get_physical_device_surface_capabilities(self.raw, surface.raw)
            } {
                Ok(caps) => caps,
                Err(e) => {
                    log::error!("get_physical_device_surface_capabilities: {}", e);
                    return None;
                }
            }
        };

        // If image count is 0, the support number of images is unlimited.
        let max_image_count = if caps.max_image_count == 0 {
            !0
        } else {
            caps.max_image_count
        };

        // `0xFFFFFFFF` indicates that the extent depends on the created swapchain.
        let current_extent = if caps.current_extent.width != !0 && caps.current_extent.height != !0
        {
            Some(wgt::Extent3d {
                width: caps.current_extent.width,
                height: caps.current_extent.height,
                depth_or_array_layers: 1,
            })
        } else {
            None
        };

        let raw_present_modes = {
            profiling::scope!("vkGetPhysicalDeviceSurfacePresentModesKHR");
            match unsafe {
                surface
                    .functor
                    .get_physical_device_surface_present_modes(self.raw, surface.raw)
            } {
                Ok(present_modes) => present_modes,
                Err(e) => {
                    log::error!("get_physical_device_surface_present_modes: {}", e);
                    Vec::new()
                }
            }
        };

        let raw_surface_formats = {
            profiling::scope!("vkGetPhysicalDeviceSurfaceFormatsKHR");
            match unsafe {
                surface
                    .functor
                    .get_physical_device_surface_formats(self.raw, surface.raw)
            } {
                Ok(formats) => formats,
                Err(e) => {
                    log::error!("get_physical_device_surface_formats: {}", e);
                    Vec::new()
                }
            }
        };

        let formats = raw_surface_formats
            .into_iter()
            .filter_map(conv::map_vk_surface_formats)
            .collect();
        Some(crate::SurfaceCapabilities {
            formats,
            swap_chain_sizes: caps.min_image_count..=max_image_count,
            current_extent,
            usage: conv::map_vk_image_usage(caps.supported_usage_flags),
            present_modes: raw_present_modes
                .into_iter()
                .flat_map(conv::map_vk_present_mode)
                .collect(),
            composite_alpha_modes: conv::map_vk_composite_alpha(caps.supported_composite_alpha),
        })
    }

    unsafe fn get_presentation_timestamp(&self) -> wgt::PresentationTimestamp {
        // VK_GOOGLE_display_timing is the only way to get presentation
        // timestamps on vulkan right now and it is only ever available
        // on android and linux. This includes mac, but there's no alternative
        // on mac, so this is fine.
        #[cfg(unix)]
        {
            let mut timespec = libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            unsafe {
                libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut timespec);
            }

            wgt::PresentationTimestamp(
                timespec.tv_sec as u128 * 1_000_000_000 + timespec.tv_nsec as u128,
            )
        }
        #[cfg(not(unix))]
        {
            wgt::PresentationTimestamp::INVALID_TIMESTAMP
        }
    }
}

fn is_format_16bit_norm_supported(instance: &ash::Instance, phd: vk::PhysicalDevice) -> bool {
    let tiling = vk::ImageTiling::OPTIMAL;
    let features = vk::FormatFeatureFlags::SAMPLED_IMAGE
        | vk::FormatFeatureFlags::STORAGE_IMAGE
        | vk::FormatFeatureFlags::TRANSFER_SRC
        | vk::FormatFeatureFlags::TRANSFER_DST;
    let r16unorm = supports_format(instance, phd, vk::Format::R16_UNORM, tiling, features);
    let r16snorm = supports_format(instance, phd, vk::Format::R16_SNORM, tiling, features);
    let rg16unorm = supports_format(instance, phd, vk::Format::R16G16_UNORM, tiling, features);
    let rg16snorm = supports_format(instance, phd, vk::Format::R16G16_SNORM, tiling, features);
    let rgba16unorm = supports_format(
        instance,
        phd,
        vk::Format::R16G16B16A16_UNORM,
        tiling,
        features,
    );
    let rgba16snorm = supports_format(
        instance,
        phd,
        vk::Format::R16G16B16A16_SNORM,
        tiling,
        features,
    );

    r16unorm && r16snorm && rg16unorm && rg16snorm && rgba16unorm && rgba16snorm
}

fn is_float32_filterable_supported(instance: &ash::Instance, phd: vk::PhysicalDevice) -> bool {
    let tiling = vk::ImageTiling::OPTIMAL;
    let features = vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR;
    let r_float = supports_format(instance, phd, vk::Format::R32_SFLOAT, tiling, features);
    let rg_float = supports_format(instance, phd, vk::Format::R32G32_SFLOAT, tiling, features);
    let rgba_float = supports_format(
        instance,
        phd,
        vk::Format::R32G32B32A32_SFLOAT,
        tiling,
        features,
    );
    r_float && rg_float && rgba_float
}

fn supports_format(
    instance: &ash::Instance,
    phd: vk::PhysicalDevice,
    format: vk::Format,
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> bool {
    let properties = unsafe { instance.get_physical_device_format_properties(phd, format) };
    match tiling {
        vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
        vk::ImageTiling::OPTIMAL => properties.optimal_tiling_features.contains(features),
        _ => false,
    }
}

fn supports_bgra8unorm_storage(
    instance: &ash::Instance,
    phd: vk::PhysicalDevice,
    device_api_version: u32,
) -> bool {
    // See https://github.com/KhronosGroup/Vulkan-Docs/issues/2027#issuecomment-1380608011

    // This check gates the function call and structures used below.
    // TODO: check for (`VK_KHR_get_physical_device_properties2` or VK1.1) and (`VK_KHR_format_feature_flags2` or VK1.3).
    // Right now we only check for VK1.3.
    if device_api_version < vk::API_VERSION_1_3 {
        return false;
    }

    unsafe {
        let mut properties3 = vk::FormatProperties3::default();
        let mut properties2 = vk::FormatProperties2::builder().push_next(&mut properties3);

        instance.get_physical_device_format_properties2(
            phd,
            vk::Format::B8G8R8A8_UNORM,
            &mut properties2,
        );

        let features2 = properties2.format_properties.optimal_tiling_features;
        let features3 = properties3.optimal_tiling_features;

        features2.contains(vk::FormatFeatureFlags::STORAGE_IMAGE)
            && features3.contains(vk::FormatFeatureFlags2::STORAGE_WRITE_WITHOUT_FORMAT)
    }
}

// For https://github.com/gfx-rs/wgpu/issues/4599
// Intel iGPUs with outdated drivers can break rendering if `VK_EXT_robustness2` is used.
// Driver version 31.0.101.2115 works, but there's probably an earlier functional version.
fn is_intel_igpu_outdated_for_robustness2(
    props: vk::PhysicalDeviceProperties,
    driver: Option<vk::PhysicalDeviceDriverPropertiesKHR>,
) -> bool {
    const DRIVER_VERSION_WORKING: u32 = (101 << 14) | 2115; // X.X.101.2115

    let is_outdated = props.vendor_id == crate::auxil::db::intel::VENDOR
        && props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
        && props.driver_version < DRIVER_VERSION_WORKING
        && driver
            .map(|driver| driver.driver_id == vk::DriverId::INTEL_PROPRIETARY_WINDOWS)
            .unwrap_or_default();

    if is_outdated {
        log::warn!(
            "Disabling robustBufferAccess2 and robustImageAccess2: IntegratedGpu Intel Driver is outdated. Found with version 0x{:X}, less than the known good version 0x{:X} (31.0.101.2115)",
            props.driver_version,
            DRIVER_VERSION_WORKING
        );
    }
    is_outdated
}
