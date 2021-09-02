use super::conv;

use ash::{extensions::khr, vk};
use parking_lot::Mutex;

use std::{ffi::CStr, mem, ptr, sync::Arc};

//TODO: const fn?
fn indexing_features() -> wgt::Features {
    wgt::Features::BUFFER_BINDING_ARRAY | wgt::Features::TEXTURE_BINDING_ARRAY
}

/// Aggregate of the `vk::PhysicalDevice*Features` structs used by `gfx`.
#[derive(Debug, Default)]
pub struct PhysicalDeviceFeatures {
    core: vk::PhysicalDeviceFeatures,
    vulkan_1_2: Option<vk::PhysicalDeviceVulkan12Features>,
    descriptor_indexing: Option<vk::PhysicalDeviceDescriptorIndexingFeaturesEXT>,
    imageless_framebuffer: Option<vk::PhysicalDeviceImagelessFramebufferFeaturesKHR>,
    timeline_semaphore: Option<vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>,
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
        if let Some(ref mut feature) = self.vulkan_1_2 {
            info = info.push_next(feature);
        }
        if let Some(ref mut feature) = self.descriptor_indexing {
            info = info.push_next(feature);
        }
        if let Some(ref mut feature) = self.imageless_framebuffer {
            info = info.push_next(feature);
        }
        info
    }

    /// Create a `PhysicalDeviceFeatures` that will be used to create a logical device.
    ///
    /// `requested_features` should be the same as what was used to generate `enabled_extensions`.
    fn from_extensions_and_requested_features(
        api_version: u32,
        enabled_extensions: &[&'static CStr],
        requested_features: wgt::Features,
        downlevel_flags: wgt::DownlevelFlags,
        private_caps: &super::PrivateCapabilities,
    ) -> Self {
        //TODO: make configurable
        let rba = !(cfg!(target_os = "macos") || cfg!(target_os = "ios"));
        Self {
            // vk::PhysicalDeviceFeatures is a struct composed of Bool32's while
            // Features is a bitfield so we need to map everything manually
            core: vk::PhysicalDeviceFeatures::builder()
                .robust_buffer_access(rba)
                .independent_blend(true)
                .sample_rate_shading(true)
                .image_cube_array(
                    downlevel_flags.contains(wgt::DownlevelFlags::CUBE_ARRAY_TEXTURES),
                )
                //.dual_src_blend(requested_features.contains(wgt::Features::DUAL_SRC_BLENDING))
                .multi_draw_indirect(
                    requested_features.contains(wgt::Features::MULTI_DRAW_INDIRECT),
                )
                .depth_clamp(requested_features.contains(wgt::Features::DEPTH_CLAMPING))
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
                    requested_features.contains(wgt::Features::TEXTURE_COMPRESSION_ASTC_LDR),
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
                .shader_float64(requested_features.contains(wgt::Features::SHADER_FLOAT64))
                //.shader_int64(requested_features.contains(wgt::Features::SHADER_INT64))
                //.shader_int16(requested_features.contains(wgt::Features::SHADER_INT16))
                //.shader_resource_residency(requested_features.contains(wgt::Features::SHADER_RESOURCE_RESIDENCY))
                .geometry_shader(requested_features.contains(wgt::Features::SHADER_PRIMITIVE_INDEX))
                .build(),
            vulkan_1_2: if api_version >= vk::API_VERSION_1_2 {
                Some(
                    vk::PhysicalDeviceVulkan12Features::builder()
                        //.sampler_mirror_clamp_to_edge(requested_features.contains(wgt::Features::SAMPLER_MIRROR_CLAMP_EDGE))
                        .draw_indirect_count(
                            requested_features.contains(wgt::Features::MULTI_DRAW_INDIRECT_COUNT),
                        )
                        .descriptor_indexing(requested_features.intersects(indexing_features()))
                        .shader_sampled_image_array_non_uniform_indexing(
                            requested_features.contains(
                                wgt::Features::TEXTURE_BINDING_ARRAY
                                    | wgt::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                            ),
                        )
                        .shader_storage_image_array_non_uniform_indexing(
                            requested_features.contains(
                                wgt::Features::TEXTURE_BINDING_ARRAY
                                    | wgt::Features::STORAGE_RESOURCE_BINDING_ARRAY
                                    | wgt::Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING,
                            ),
                        )
                        //.shader_storage_buffer_array_non_uniform_indexing(
                        .shader_uniform_buffer_array_non_uniform_indexing(
                            requested_features.contains(
                                wgt::Features::BUFFER_BINDING_ARRAY
                                    | wgt::Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING,
                            ),
                        )
                        .shader_storage_buffer_array_non_uniform_indexing(
                            requested_features.contains(
                                wgt::Features::BUFFER_BINDING_ARRAY
                                    | wgt::Features::STORAGE_RESOURCE_BINDING_ARRAY
                                    | wgt::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                            ),
                        )
                        .runtime_descriptor_array(
                            requested_features.contains(wgt::Features::UNSIZED_BINDING_ARRAY),
                        )
                        //.sampler_filter_minmax(requested_features.contains(wgt::Features::SAMPLER_REDUCTION))
                        .imageless_framebuffer(private_caps.imageless_framebuffers)
                        .timeline_semaphore(private_caps.timeline_semaphores)
                        .build(),
                )
            } else {
                None
            },
            descriptor_indexing: if enabled_extensions
                .contains(&vk::ExtDescriptorIndexingFn::name())
            {
                Some(
                    vk::PhysicalDeviceDescriptorIndexingFeaturesEXT::builder()
                        .shader_sampled_image_array_non_uniform_indexing(
                            requested_features.contains(
                                wgt::Features::TEXTURE_BINDING_ARRAY
                                    | wgt::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                            ),
                        )
                        .shader_storage_image_array_non_uniform_indexing(
                            requested_features.contains(
                                wgt::Features::TEXTURE_BINDING_ARRAY
                                    | wgt::Features::STORAGE_RESOURCE_BINDING_ARRAY
                                    | wgt::Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING,
                            ),
                        )
                        //.shader_storage_buffer_array_non_uniform_indexing(
                        .shader_uniform_buffer_array_non_uniform_indexing(
                            requested_features.contains(
                                wgt::Features::BUFFER_BINDING_ARRAY
                                    | wgt::Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING,
                            ),
                        )
                        .shader_storage_buffer_array_non_uniform_indexing(
                            requested_features.contains(
                                wgt::Features::BUFFER_BINDING_ARRAY
                                    | wgt::Features::STORAGE_RESOURCE_BINDING_ARRAY
                                    | wgt::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                            ),
                        )
                        .runtime_descriptor_array(
                            requested_features.contains(wgt::Features::UNSIZED_BINDING_ARRAY),
                        )
                        .build(),
                )
            } else {
                None
            },
            imageless_framebuffer: if enabled_extensions
                .contains(&vk::KhrImagelessFramebufferFn::name())
            {
                Some(
                    vk::PhysicalDeviceImagelessFramebufferFeaturesKHR::builder()
                        .imageless_framebuffer(true)
                        .build(),
                )
            } else {
                None
            },
            timeline_semaphore: if enabled_extensions.contains(&vk::KhrTimelineSemaphoreFn::name())
            {
                Some(
                    vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR::builder()
                        .timeline_semaphore(true)
                        .build(),
                )
            } else {
                None
            },
        }
    }

    fn to_wgpu(&self, caps: &PhysicalDeviceCapabilities) -> (wgt::Features, wgt::DownlevelFlags) {
        use wgt::{DownlevelFlags as Df, Features as F};
        let mut features = F::empty()
            | F::SPIRV_SHADER_PASSTHROUGH
            | F::MAPPABLE_PRIMARY_BUFFERS
            | F::PUSH_CONSTANTS
            | F::ADDRESS_MODE_CLAMP_TO_BORDER
            | F::TIMESTAMP_QUERY
            | F::PIPELINE_STATISTICS_QUERY
            | F::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        let mut dl_flags = Df::all();

        dl_flags.set(Df::CUBE_ARRAY_TEXTURES, self.core.image_cube_array != 0);
        dl_flags.set(Df::ANISOTROPIC_FILTERING, self.core.sampler_anisotropy != 0);
        dl_flags.set(
            Df::FRAGMENT_WRITABLE_STORAGE,
            self.core.fragment_stores_and_atomics != 0,
        );

        //if self.core.dual_src_blend != 0
        features.set(F::MULTI_DRAW_INDIRECT, self.core.multi_draw_indirect != 0);
        features.set(F::DEPTH_CLAMPING, self.core.depth_clamp != 0);
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
            F::TEXTURE_COMPRESSION_ASTC_LDR,
            self.core.texture_compression_astc_ldr != 0,
        );
        features.set(
            F::TEXTURE_COMPRESSION_BC,
            self.core.texture_compression_bc != 0,
        );
        //if self.core.occlusion_query_precise != 0 {
        //if self.core.pipeline_statistics_query != 0 { //TODO
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
        features.set(F::SHADER_FLOAT64, self.core.shader_float64 != 0);
        //if self.core.shader_int64 != 0 {
        //if self.core.shader_int16 != 0 {

        //if caps.supports_extension(vk::KhrSamplerMirrorClampToEdgeFn::name()) {
        //if caps.supports_extension(vk::ExtSamplerFilterMinmaxFn::name()) {
        features.set(
            F::MULTI_DRAW_INDIRECT_COUNT,
            caps.supports_extension(khr::DrawIndirectCount::name()),
        );
        features.set(
            F::CONSERVATIVE_RASTERIZATION,
            caps.supports_extension(vk::ExtConservativeRasterizationFn::name()),
        );

        if let Some(ref vulkan_1_2) = self.vulkan_1_2 {
            const STORAGE: F = F::STORAGE_RESOURCE_BINDING_ARRAY;
            if Self::all_features_supported(
                &features,
                &[
                    (
                        F::TEXTURE_BINDING_ARRAY,
                        vulkan_1_2.shader_sampled_image_array_non_uniform_indexing,
                    ),
                    (
                        F::BUFFER_BINDING_ARRAY | STORAGE,
                        vulkan_1_2.shader_storage_buffer_array_non_uniform_indexing,
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
                        vulkan_1_2.shader_uniform_buffer_array_non_uniform_indexing,
                    ),
                    (
                        F::BUFFER_BINDING_ARRAY | STORAGE,
                        vulkan_1_2.shader_storage_buffer_array_non_uniform_indexing,
                    ),
                ],
            ) {
                features.insert(F::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING);
            }
            if vulkan_1_2.runtime_descriptor_array != 0 {
                features |= F::UNSIZED_BINDING_ARRAY;
            }
            //if vulkan_1_2.sampler_mirror_clamp_to_edge != 0 {
            //if vulkan_1_2.sampler_filter_minmax != 0 {
            if vulkan_1_2.draw_indirect_count != 0 {
                features |= F::MULTI_DRAW_INDIRECT_COUNT;
            }
        }

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
            if descriptor_indexing.runtime_descriptor_array != 0 {
                features |= F::UNSIZED_BINDING_ARRAY;
            }
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
pub struct PhysicalDeviceCapabilities {
    supported_extensions: Vec<vk::ExtensionProperties>,
    properties: vk::PhysicalDeviceProperties,
}

impl PhysicalDeviceCapabilities {
    fn supports_extension(&self, extension: &CStr) -> bool {
        self.supported_extensions
            .iter()
            .any(|ep| unsafe { CStr::from_ptr(ep.extension_name.as_ptr()) } == extension)
    }

    /// Map `requested_features` to the list of Vulkan extension strings required to create the logical device.
    fn get_required_extensions(&self, requested_features: wgt::Features) -> Vec<&'static CStr> {
        let mut extensions = Vec::new();

        extensions.push(khr::Swapchain::name());

        if self.properties.api_version < vk::API_VERSION_1_1 {
            extensions.push(vk::KhrMaintenance1Fn::name());
            extensions.push(vk::KhrMaintenance2Fn::name());

            // `VK_AMD_negative_viewport_height` is obsoleted by `VK_KHR_maintenance1` and must not be enabled alongside `VK_KHR_maintenance1` or a 1.1+ device.
            if !self.supports_extension(vk::KhrMaintenance1Fn::name()) {
                extensions.push(vk::AmdNegativeViewportHeightFn::name());
            }
        }

        if self.properties.api_version < vk::API_VERSION_1_2 {
            if self.supports_extension(vk::KhrImagelessFramebufferFn::name()) {
                extensions.push(vk::KhrImagelessFramebufferFn::name());
                extensions.push(vk::KhrImageFormatListFn::name()); // Required for `KhrImagelessFramebufferFn`
            }

            extensions.push(vk::ExtSamplerFilterMinmaxFn::name());
            extensions.push(vk::KhrTimelineSemaphoreFn::name());

            if requested_features.intersects(indexing_features()) {
                extensions.push(vk::ExtDescriptorIndexingFn::name());

                if self.properties.api_version < vk::API_VERSION_1_1 {
                    extensions.push(vk::KhrMaintenance3Fn::name());
                }
            }

            //extensions.push(vk::KhrSamplerMirrorClampToEdgeFn::name());
            //extensions.push(vk::ExtSamplerFilterMinmaxFn::name());

            if requested_features.contains(wgt::Features::MULTI_DRAW_INDIRECT_COUNT) {
                extensions.push(khr::DrawIndirectCount::name());
            }
        }

        if requested_features.contains(wgt::Features::CONSERVATIVE_RASTERIZATION) {
            extensions.push(vk::ExtConservativeRasterizationFn::name());
        }

        extensions
    }

    fn to_wgpu_limits(&self) -> wgt::Limits {
        let limits = &self.properties.limits;
        wgt::Limits {
            max_texture_dimension_1d: limits.max_image_dimension1_d,
            max_texture_dimension_2d: limits.max_image_dimension2_d,
            max_texture_dimension_3d: limits.max_image_dimension3_d,
            max_texture_array_layers: limits.max_image_array_layers,
            max_bind_groups: limits
                .max_bound_descriptor_sets
                .min(crate::MAX_BIND_GROUPS as u32),
            max_dynamic_uniform_buffers_per_pipeline_layout: limits
                .max_descriptor_set_uniform_buffers_dynamic,
            max_dynamic_storage_buffers_per_pipeline_layout: limits
                .max_descriptor_set_storage_buffers_dynamic,
            max_sampled_textures_per_shader_stage: limits.max_per_stage_descriptor_sampled_images,
            max_samplers_per_shader_stage: limits.max_per_stage_descriptor_samplers,
            max_storage_buffers_per_shader_stage: limits.max_per_stage_descriptor_storage_buffers,
            max_storage_textures_per_shader_stage: limits.max_per_stage_descriptor_storage_images,
            max_uniform_buffers_per_shader_stage: limits.max_per_stage_descriptor_uniform_buffers,
            max_uniform_buffer_binding_size: limits.max_uniform_buffer_range,
            max_storage_buffer_binding_size: limits.max_storage_buffer_range,
            max_vertex_buffers: limits
                .max_vertex_input_bindings
                .min(crate::MAX_VERTEX_BUFFERS as u32),
            max_vertex_attributes: limits.max_vertex_input_attributes,
            max_vertex_buffer_array_stride: limits.max_vertex_input_binding_stride,
            max_push_constant_size: limits.max_push_constants_size,
        }
    }

    fn to_hal_alignments(&self) -> crate::Alignments {
        let limits = &self.properties.limits;
        crate::Alignments {
            buffer_copy_offset: wgt::BufferSize::new(limits.optimal_buffer_copy_offset_alignment)
                .unwrap(),
            buffer_copy_pitch: wgt::BufferSize::new(limits.optimal_buffer_copy_row_pitch_alignment)
                .unwrap(),
            storage_buffer_offset: wgt::BufferSize::new(limits.min_storage_buffer_offset_alignment)
                .unwrap(),
            uniform_buffer_offset: wgt::BufferSize::new(limits.min_uniform_buffer_offset_alignment)
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
        let capabilities = unsafe {
            PhysicalDeviceCapabilities {
                supported_extensions: self.raw.enumerate_device_extension_properties(phd).unwrap(),
                properties: self.raw.get_physical_device_properties(phd),
            }
        };

        let mut features = PhysicalDeviceFeatures::default();
        features.core = if let Some(ref get_device_properties) = self.get_physical_device_properties
        {
            let core = vk::PhysicalDeviceFeatures::builder().build();
            let mut features2 = vk::PhysicalDeviceFeatures2KHR::builder()
                .features(core)
                .build();

            if capabilities.properties.api_version >= vk::API_VERSION_1_2 {
                features.vulkan_1_2 = Some(vk::PhysicalDeviceVulkan12Features::builder().build());

                let mut_ref = features.vulkan_1_2.as_mut().unwrap();
                mut_ref.p_next = mem::replace(&mut features2.p_next, mut_ref as *mut _ as *mut _);
            }

            if capabilities.supports_extension(vk::ExtDescriptorIndexingFn::name()) {
                features.descriptor_indexing =
                    Some(vk::PhysicalDeviceDescriptorIndexingFeaturesEXT::builder().build());

                let mut_ref = features.descriptor_indexing.as_mut().unwrap();
                mut_ref.p_next = mem::replace(&mut features2.p_next, mut_ref as *mut _ as *mut _);
            }

            // `VK_KHR_imageless_framebuffer` is promoted to 1.2, but has no changes, so we can keep using the extension unconditionally.
            if capabilities.supports_extension(vk::KhrImagelessFramebufferFn::name()) {
                features.imageless_framebuffer =
                    Some(vk::PhysicalDeviceImagelessFramebufferFeaturesKHR::builder().build());

                let mut_ref = features.imageless_framebuffer.as_mut().unwrap();
                mut_ref.p_next = mem::replace(&mut features2.p_next, mut_ref as *mut _ as *mut _);
            }

            // `VK_KHR_timeline_semaphore` is promoted to 1.2, but has no changes, so we can keep using the extension unconditionally.
            if capabilities.supports_extension(vk::KhrTimelineSemaphoreFn::name()) {
                features.timeline_semaphore =
                    Some(vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR::builder().build());

                let mut_ref = features.timeline_semaphore.as_mut().unwrap();
                mut_ref.p_next = mem::replace(&mut features2.p_next, mut_ref as *mut _ as *mut _);
            }

            unsafe {
                get_device_properties.get_physical_device_features2_khr(phd, &mut features2);
            }
            features2.features
        } else {
            unsafe { self.raw.get_physical_device_features(phd) }
        };

        /// # Safety
        /// `T` must be a struct bigger than `vk::BaseOutStructure`.
        unsafe fn null_p_next<T>(features: &mut Option<T>) {
            if let Some(ref mut features) = *features {
                // This is technically invalid since `vk::BaseOutStructure` and `T` will probably never have the same size.
                (*(features as *mut T as *mut vk::BaseOutStructure)).p_next = ptr::null_mut();
            }
        }

        unsafe {
            null_p_next(&mut features.vulkan_1_2);
            null_p_next(&mut features.descriptor_indexing);
            null_p_next(&mut features.imageless_framebuffer);
            null_p_next(&mut features.timeline_semaphore);
        }

        (capabilities, features)
    }
}

impl super::Instance {
    pub fn expose_adapter(
        &self,
        phd: vk::PhysicalDevice,
    ) -> Option<crate::ExposedAdapter<super::Api>> {
        let (phd_capabilities, phd_features) = self.shared.inspect(phd);

        let info = wgt::AdapterInfo {
            name: unsafe {
                CStr::from_ptr(phd_capabilities.properties.device_name.as_ptr())
                    .to_str()
                    .unwrap_or("?")
                    .to_owned()
            },
            vendor: phd_capabilities.properties.vendor_id as usize,
            device: phd_capabilities.properties.device_id as usize,
            device_type: match phd_capabilities.properties.device_type {
                ash::vk::PhysicalDeviceType::OTHER => wgt::DeviceType::Other,
                ash::vk::PhysicalDeviceType::INTEGRATED_GPU => wgt::DeviceType::IntegratedGpu,
                ash::vk::PhysicalDeviceType::DISCRETE_GPU => wgt::DeviceType::DiscreteGpu,
                ash::vk::PhysicalDeviceType::VIRTUAL_GPU => wgt::DeviceType::VirtualGpu,
                ash::vk::PhysicalDeviceType::CPU => wgt::DeviceType::Cpu,
                _ => wgt::DeviceType::Other,
            },
            backend: wgt::Backend::Vulkan,
        };

        let (available_features, downlevel_flags) = phd_features.to_wgpu(&phd_capabilities);
        {
            use crate::auxil::db;
            // see https://github.com/gfx-rs/gfx/issues/1930
            let _is_windows_intel_dual_src_bug = cfg!(windows)
                && phd_capabilities.properties.vendor_id == db::intel::VENDOR
                && (phd_capabilities.properties.device_id & db::intel::DEVICE_KABY_LAKE_MASK
                    == db::intel::DEVICE_KABY_LAKE_MASK
                    || phd_capabilities.properties.device_id & db::intel::DEVICE_SKY_LAKE_MASK
                        == db::intel::DEVICE_SKY_LAKE_MASK);
        };

        if phd_features.core.sample_rate_shading == 0 {
            log::error!(
                "sample_rate_shading feature is not supported, hiding the adapter: {}",
                info.name
            );
            return None;
        }
        if !phd_capabilities.supports_extension(vk::AmdNegativeViewportHeightFn::name())
            && !phd_capabilities.supports_extension(vk::KhrMaintenance1Fn::name())
            && phd_capabilities.properties.api_version < vk::API_VERSION_1_2
        {
            log::error!(
                "viewport Y-flip is not supported, hiding the adapter: {}",
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
            flip_y_requires_shift: phd_capabilities.properties.api_version >= vk::API_VERSION_1_1
                || phd_capabilities.supports_extension(vk::KhrMaintenance1Fn::name()),
            imageless_framebuffers: match phd_features.vulkan_1_2 {
                Some(features) => features.imageless_framebuffer == vk::TRUE,
                None => phd_capabilities.supports_extension(vk::KhrImagelessFramebufferFn::name()),
            },
            image_view_usage: phd_capabilities.properties.api_version >= vk::API_VERSION_1_1
                || phd_capabilities.supports_extension(vk::KhrMaintenance2Fn::name()),
            timeline_semaphores: match phd_features.vulkan_1_2 {
                Some(features) => features.timeline_semaphore == vk::TRUE,
                None => phd_capabilities.supports_extension(vk::KhrTimelineSemaphoreFn::name()),
            },
            texture_d24: unsafe {
                self.shared
                    .raw
                    .get_physical_device_format_properties(phd, vk::Format::X8_D24_UNORM_PACK32)
                    .optimal_tiling_features
                    .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
            },
            texture_d24_s8: unsafe {
                self.shared
                    .raw
                    .get_physical_device_format_properties(phd, vk::Format::D24_UNORM_S8_UINT)
                    .optimal_tiling_features
                    .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
            },
            non_coherent_map_mask: phd_capabilities.properties.limits.non_coherent_atom_size - 1,
            can_present: true,
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
            self.phd_capabilities.properties.api_version,
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
    pub unsafe fn device_from_raw(
        &self,
        raw_device: ash::Device,
        handle_is_owned: bool,
        enabled_extensions: &[&'static CStr],
        family_index: u32,
        queue_index: u32,
    ) -> Result<crate::OpenDevice<super::Api>, crate::DeviceError> {
        let mem_properties = self
            .instance
            .raw
            .get_physical_device_memory_properties(self.raw);
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
            Some(super::ExtensionFn::Extension(khr::DrawIndirectCount::new(
                &self.instance.raw,
                &raw_device,
            )))
        } else if self.phd_capabilities.properties.api_version >= vk::API_VERSION_1_2 {
            Some(super::ExtensionFn::Promoted)
        } else {
            None
        };
        let timeline_semaphore_fn = if enabled_extensions.contains(&khr::TimelineSemaphore::name())
        {
            Some(super::ExtensionFn::Extension(khr::TimelineSemaphore::new(
                &self.instance.entry,
                &self.instance.raw,
            )))
        } else if self.phd_capabilities.properties.api_version >= vk::API_VERSION_1_2 {
            Some(super::ExtensionFn::Promoted)
        } else {
            None
        };

        let naga_options = {
            use naga::back::spv;
            let capabilities = [
                spv::Capability::Shader,
                spv::Capability::Matrix,
                spv::Capability::Sampled1D,
                spv::Capability::Image1D,
                spv::Capability::ImageQuery,
                spv::Capability::DerivativeControl,
                //Note: this is requested always, no matter what the actual
                // adapter supports. It's not the responsibility of SPV-out
                // translation to handle the storage support for formats.
                spv::Capability::StorageImageExtendedFormats,
                //TODO: fill out the rest
            ];
            let mut flags = spv::WriterFlags::empty();
            flags.set(
                spv::WriterFlags::DEBUG,
                self.instance.flags.contains(crate::InstanceFlags::DEBUG),
            );
            spv::Options {
                lang_version: (1, 0),
                flags,
                capabilities: Some(capabilities.iter().cloned().collect()),
                index_bounds_check_policy: naga::back::BoundsCheckPolicy::Restrict,
                image_bounds_check_policy: naga::back::BoundsCheckPolicy::Restrict,
            }
        };

        log::info!("Private capabilities: {:?}", self.private_caps);
        let raw_queue = raw_device.get_device_queue(family_index, queue_index);

        let shared = Arc::new(super::DeviceShared {
            raw: raw_device,
            handle_is_owned,
            instance: Arc::clone(&self.instance),
            extension_fns: super::DeviceExtensionFunctions {
                draw_indirect_count: indirect_count_fn,
                timeline_semaphore: timeline_semaphore_fn,
            },
            vendor_id: self.phd_capabilities.properties.vendor_id,
            downlevel_flags: self.downlevel_flags,
            private_caps: self.private_caps.clone(),
            _timestamp_period: self.phd_capabilities.properties.limits.timestamp_period,
            render_passes: Mutex::new(Default::default()),
            framebuffers: Mutex::new(Default::default()),
        });
        let queue = super::Queue {
            raw: raw_queue,
            swapchain_fn,
            device: Arc::clone(&shared),
            family_index,
            relay_semaphore: shared
                .raw
                .create_semaphore(&vk::SemaphoreCreateInfo::builder(), None)?,
            relay_active: false,
        };

        let mem_allocator = {
            let limits = self.phd_capabilities.properties.limits;
            let config = gpu_alloc::Config::i_am_prototyping(); //TODO
            let properties = gpu_alloc::DeviceProperties {
                max_memory_allocation_count: limits.max_memory_allocation_count,
                max_memory_allocation_size: u64::max_value(), // TODO
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
                buffer_device_address: false,
            };
            gpu_alloc::GpuAllocator::new(config, properties)
        };
        let desc_allocator = gpu_descriptor::DescriptorAllocator::new(0);

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
        let raw_device = self.instance.raw.create_device(self.raw, &info, None)?;

        self.device_from_raw(
            raw_device,
            true,
            &enabled_extensions,
            family_info.queue_family_index,
            0,
        )
    }

    unsafe fn texture_format_capabilities(
        &self,
        format: wgt::TextureFormat,
    ) -> crate::TextureFormatCapabilities {
        use crate::TextureFormatCapabilities as Tfc;
        let vk_format = self.private_caps.map_texture_format(format);
        let properties = self
            .instance
            .raw
            .get_physical_device_format_properties(self.raw, vk_format);
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
        flags.set(
            Tfc::SAMPLED_MINMAX,
            features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_MINMAX),
        );
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
            features.intersects(
                vk::FormatFeatureFlags::TRANSFER_SRC | vk::FormatFeatureFlags::BLIT_SRC,
            ),
        );
        flags.set(
            Tfc::COPY_DST,
            features.intersects(
                vk::FormatFeatureFlags::TRANSFER_DST | vk::FormatFeatureFlags::BLIT_DST,
            ),
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
        match surface.functor.get_physical_device_surface_support(
            self.raw,
            queue_family_index,
            surface.raw,
        ) {
            Ok(true) => (),
            Ok(false) => return None,
            Err(e) => {
                log::error!("get_physical_device_surface_support: {}", e);
                return None;
            }
        }

        let caps = match surface
            .functor
            .get_physical_device_surface_capabilities(self.raw, surface.raw)
        {
            Ok(caps) => caps,
            Err(e) => {
                log::error!("get_physical_device_surface_capabilities: {}", e);
                return None;
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

        let min_extent = wgt::Extent3d {
            width: caps.min_image_extent.width,
            height: caps.min_image_extent.height,
            depth_or_array_layers: 1,
        };

        let max_extent = wgt::Extent3d {
            width: caps.max_image_extent.width,
            height: caps.max_image_extent.height,
            depth_or_array_layers: caps.max_image_array_layers,
        };

        let raw_present_modes = match surface
            .functor
            .get_physical_device_surface_present_modes(self.raw, surface.raw)
        {
            Ok(present_modes) => present_modes,
            Err(e) => {
                log::error!("get_physical_device_surface_present_modes: {}", e);
                Vec::new()
            }
        };

        let raw_surface_formats = match surface
            .functor
            .get_physical_device_surface_formats(self.raw, surface.raw)
        {
            Ok(formats) => formats,
            Err(e) => {
                log::error!("get_physical_device_surface_formats: {}", e);
                Vec::new()
            }
        };

        let supported_formats = [
            wgt::TextureFormat::Rgba8Unorm,
            wgt::TextureFormat::Rgba8UnormSrgb,
            wgt::TextureFormat::Bgra8Unorm,
            wgt::TextureFormat::Bgra8UnormSrgb,
        ];
        let formats = supported_formats
            .iter()
            .cloned()
            .filter(|&format| {
                let vk_format = self.private_caps.map_texture_format(format);
                raw_surface_formats
                    .iter()
                    .any(|sf| sf.format == vk_format || sf.format == vk::Format::UNDEFINED)
            })
            .collect();

        Some(crate::SurfaceCapabilities {
            formats,
            swap_chain_sizes: caps.min_image_count..=max_image_count,
            current_extent,
            extents: min_extent..=max_extent,
            usage: conv::map_vk_image_usage(caps.supported_usage_flags),
            present_modes: raw_present_modes
                .into_iter()
                .flat_map(conv::map_vk_present_mode)
                .collect(),
            composite_alpha_modes: conv::map_vk_composite_alpha(caps.supported_composite_alpha),
        })
    }
}
