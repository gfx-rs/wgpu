use std::{ptr, sync::Arc};

use super::conv;

type DeviceResult<T> = Result<T, crate::DeviceError>;

impl crate::Device<super::Api> for super::Device {
    unsafe fn create_buffer(&self, desc: &crate::BufferDescriptor) -> DeviceResult<super::Buffer> {
        let map_read = desc.usage.contains(crate::BufferUse::MAP_READ);
        let map_write = desc.usage.contains(crate::BufferUse::MAP_WRITE);

        let mut options = mtl::MTLResourceOptions::empty();
        options |= if map_read || map_write {
            mtl::MTLResourceOptions::StorageModeShared
        } else {
            mtl::MTLResourceOptions::StorageModePrivate
        };
        options.set(mtl::MTLResourceOptions::CPUCacheModeDefaultCache, map_read);

        //TODO: HazardTrackingModeUntracked

        let raw = self.shared.device.lock().new_buffer(desc.size, options);
        if let Some(label) = desc.label {
            raw.set_label(label);
        }
        Ok(super::Buffer {
            raw,
            size: desc.size,
            options,
        })
    }
    unsafe fn destroy_buffer(&self, _buffer: super::Buffer) {}

    unsafe fn map_buffer(
        &self,
        buffer: &super::Buffer,
        range: crate::MemoryRange,
    ) -> DeviceResult<ptr::NonNull<u8>> {
        let ptr = buffer.raw.contents() as *mut u8;
        assert!(!ptr.is_null());
        Ok(ptr::NonNull::new(ptr.offset(range.start as isize)).unwrap())
    }

    unsafe fn unmap_buffer(&self, _buffer: &super::Buffer) -> DeviceResult<()> {
        Ok(())
    }
    unsafe fn flush_mapped_ranges<I>(&self, _buffer: &super::Buffer, _ranges: I) {}
    unsafe fn invalidate_mapped_ranges<I>(&self, _buffer: &super::Buffer, _ranges: I) {}

    unsafe fn create_texture(
        &self,
        desc: &crate::TextureDescriptor,
    ) -> DeviceResult<super::Texture> {
        let mtl_format = self.shared.private_caps.map_format(desc.format);

        let descriptor = mtl::TextureDescriptor::new();
        let mut array_layers = desc.size.depth_or_array_layers;
        let mtl_type = match desc.dimension {
            wgt::TextureDimension::D1 => {
                if desc.size.depth_or_array_layers > 1 {
                    descriptor.set_array_length(desc.size.depth_or_array_layers as u64);
                    mtl::MTLTextureType::D1Array
                } else {
                    mtl::MTLTextureType::D1
                }
            }
            wgt::TextureDimension::D2 => {
                if desc.sample_count > 1 {
                    descriptor.set_sample_count(desc.sample_count as u64);
                    mtl::MTLTextureType::D2Multisample
                } else if desc.size.depth_or_array_layers > 1 {
                    descriptor.set_array_length(desc.size.depth_or_array_layers as u64);
                    mtl::MTLTextureType::D2Array
                } else {
                    mtl::MTLTextureType::D2
                }
            }
            wgt::TextureDimension::D3 => {
                descriptor.set_depth(desc.size.depth_or_array_layers as u64);
                array_layers = 1;
                mtl::MTLTextureType::D3
            }
        };

        descriptor.set_texture_type(mtl_type);
        descriptor.set_width(desc.size.width as u64);
        descriptor.set_height(desc.size.height as u64);
        descriptor.set_mipmap_level_count(desc.mip_level_count as u64);
        descriptor.set_pixel_format(mtl_format);
        descriptor.set_usage(conv::map_texture_usage(desc.usage));
        descriptor.set_storage_mode(mtl::MTLStorageMode::Private);

        let raw = self.shared.device.lock().new_texture(&descriptor);
        if let Some(label) = desc.label {
            raw.set_label(label);
        }

        Ok(super::Texture {
            raw,
            raw_format: mtl_format,
            raw_type: mtl_type,
            mip_levels: desc.mip_level_count,
            array_layers,
        })
    }

    unsafe fn destroy_texture(&self, _texture: super::Texture) {}

    unsafe fn create_texture_view(
        &self,
        texture: &super::Texture,
        desc: &crate::TextureViewDescriptor,
    ) -> DeviceResult<super::TextureView> {
        let mtl_format = self.shared.private_caps.map_format(desc.format);

        let mtl_type = if texture.raw_type == mtl::MTLTextureType::D2Multisample {
            texture.raw_type
        } else {
            conv::map_texture_view_dimension(desc.dimension)
        };

        let raw = if mtl_format == texture.raw_format
            && mtl_type == texture.raw_type
            && desc.range == wgt::ImageSubresourceRange::default()
        {
            // Some images are marked as framebuffer-only, and we can't create aliases of them.
            // Also helps working around Metal bugs with aliased array textures.
            texture.raw.to_owned()
        } else {
            let mip_level_count = match desc.range.mip_level_count {
                Some(count) => count.get(),
                None => texture.mip_levels - desc.range.base_mip_level,
            };
            let array_layer_count = match desc.range.array_layer_count {
                Some(count) => count.get(),
                None => texture.array_layers - desc.range.base_array_layer,
            };

            texture.raw.new_texture_view_from_slice(
                mtl_format,
                mtl_type,
                mtl::NSRange {
                    location: desc.range.base_mip_level as _,
                    length: mip_level_count as _,
                },
                mtl::NSRange {
                    location: desc.range.base_array_layer as _,
                    length: array_layer_count as _,
                },
            )
        };

        Ok(super::TextureView { raw })
    }
    unsafe fn destroy_texture_view(&self, _view: super::TextureView) {}

    unsafe fn create_sampler(
        &self,
        desc: &crate::SamplerDescriptor,
    ) -> DeviceResult<super::Sampler> {
        let caps = &self.shared.private_caps;
        let descriptor = mtl::SamplerDescriptor::new();

        descriptor.set_min_filter(conv::map_filter_mode(desc.min_filter));
        descriptor.set_mag_filter(conv::map_filter_mode(desc.mag_filter));
        descriptor.set_mip_filter(match desc.mipmap_filter {
            wgt::FilterMode::Nearest if desc.lod_clamp.is_none() => {
                mtl::MTLSamplerMipFilter::NotMipmapped
            }
            wgt::FilterMode::Nearest => mtl::MTLSamplerMipFilter::Nearest,
            wgt::FilterMode::Linear => mtl::MTLSamplerMipFilter::Linear,
        });

        if let Some(aniso) = desc.anisotropy_clamp {
            descriptor.set_max_anisotropy(aniso.get() as _);
        }

        let [s, t, r] = desc.address_modes;
        descriptor.set_address_mode_s(conv::map_address_mode(s));
        descriptor.set_address_mode_t(conv::map_address_mode(t));
        descriptor.set_address_mode_r(conv::map_address_mode(r));

        if let Some(ref range) = desc.lod_clamp {
            descriptor.set_lod_min_clamp(range.start);
            descriptor.set_lod_max_clamp(range.end);
        }

        if caps.sampler_lod_average {
            descriptor.set_lod_average(true); // optimization
        }

        if let Some(fun) = desc.compare {
            descriptor.set_compare_function(conv::map_compare_function(fun));
        }
        if let Some(border_color) = desc.border_color {
            descriptor.set_border_color(conv::map_border_color(border_color));
        }

        let raw = self.shared.device.lock().new_sampler(&descriptor);

        Ok(super::Sampler { raw })
    }
    unsafe fn destroy_sampler(&self, _sampler: super::Sampler) {}

    unsafe fn create_command_buffer(
        &self,
        desc: &crate::CommandBufferDescriptor,
    ) -> DeviceResult<super::Encoder> {
        Ok(super::Encoder)
    }
    unsafe fn destroy_command_buffer(&self, cmd_buf: super::Encoder) {}

    unsafe fn create_bind_group_layout(
        &self,
        desc: &crate::BindGroupLayoutDescriptor,
    ) -> DeviceResult<super::BindGroupLayout> {
        let map = desc
            .entries
            .iter()
            .cloned()
            .map(|entry| (entry.binding, entry))
            .collect();
        Ok(super::BindGroupLayout {
            entries: Arc::new(map),
        })
    }
    unsafe fn destroy_bind_group_layout(&self, _bg_layout: super::BindGroupLayout) {}

    unsafe fn create_pipeline_layout(
        &self,
        desc: &crate::PipelineLayoutDescriptor<super::Api>,
    ) -> DeviceResult<super::PipelineLayout> {
        #[derive(Debug)]
        struct StageInfo {
            stage: naga::ShaderStage,
            counters: super::ResourceData<super::ResourceIndex>,
            pc_buffer: Option<super::ResourceIndex>,
            pc_limit: u32,
            sizes_buffer: Option<super::ResourceIndex>,
            sizes_count: u8,
        }
        impl StageInfo {
            fn stage_bit(&self) -> wgt::ShaderStage {
                crate::aux::map_naga_stage(self.stage)
            }
        }

        let mut stage_data = super::NAGA_STAGES.map(|&stage| StageInfo {
            stage,
            counters: super::ResourceData::default(),
            pc_buffer: None,
            pc_limit: 0,
            sizes_buffer: None,
            sizes_count: 0,
        });
        let mut binding_map = std::collections::BTreeMap::default();
        let mut bind_group_infos = arrayvec::ArrayVec::new();

        // First, place the push constants
        for info in stage_data.iter_mut() {
            for pcr in desc.push_constant_ranges {
                if pcr.stages.contains(info.stage_bit()) {
                    debug_assert_eq!(pcr.range.end % 4, 0);
                    info.pc_limit = (pcr.range.end / 4).max(info.pc_limit);
                }
            }

            // round up the limits alignment to 4, so that it matches MTL compiler logic
            const LIMIT_MASK: u32 = 3;
            //TODO: figure out what and how exactly does the alignment. Clearly, it's not
            // straightforward, given that value of 2 stays non-aligned.
            if info.pc_limit > LIMIT_MASK {
                info.pc_limit = (info.pc_limit + LIMIT_MASK) & !LIMIT_MASK;
            }

            // handle the push constant buffer assignment and shader overrides
            if info.pc_limit != 0 {
                info.pc_buffer = Some(info.counters.buffers);
                info.counters.buffers += 1;
            }
        }

        // Second, place the described resources
        for (group_index, &bgl) in desc.bind_group_layouts.iter().enumerate() {
            // remember where the resources for this set start at each shader stage
            let mut dynamic_buffers = Vec::new();
            let mut sized_buffer_bindings = Vec::new();
            let base_resource_indices = stage_data.map(|info| info.counters.clone());

            for entry in bgl.entries.values() {
                match entry.ty {
                    wgt::BindingType::Buffer {
                        ty,
                        has_dynamic_offset,
                        min_binding_size: _,
                    } => {
                        if has_dynamic_offset {
                            dynamic_buffers.push(stage_data.map(|info| {
                                if entry.visibility.contains(info.stage_bit()) {
                                    info.counters.buffers
                                } else {
                                    !0
                                }
                            }));
                        }
                        match ty {
                            wgt::BufferBindingType::Storage { .. } => {
                                sized_buffer_bindings.push((entry.binding, entry.visibility));
                                for info in stage_data.iter_mut() {
                                    if entry.visibility.contains(info.stage_bit()) {
                                        info.sizes_count += 1;
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }

                for info in stage_data.iter_mut() {
                    if !entry.visibility.contains(info.stage_bit()) {
                        continue;
                    }

                    let mut target = naga::back::msl::BindTarget::default();
                    match entry.ty {
                        wgt::BindingType::Buffer { ty, .. } => {
                            target.buffer = Some(info.counters.buffers as _);
                            info.counters.buffers += 1;
                            if let wgt::BufferBindingType::Storage { read_only } = ty {
                                target.mutable = !read_only;
                            }
                        }
                        wgt::BindingType::Sampler { .. } => {
                            target.sampler = Some(naga::back::msl::BindSamplerTarget::Resource(
                                info.counters.samplers as _,
                            ));
                            info.counters.samplers += 1;
                        }
                        wgt::BindingType::Texture { .. } => {
                            target.texture = Some(info.counters.textures as _);
                            info.counters.textures += 1;
                        }
                        wgt::BindingType::StorageTexture { access, .. } => {
                            target.texture = Some(info.counters.textures as _);
                            info.counters.textures += 1;
                            target.mutable = match access {
                                wgt::StorageTextureAccess::ReadOnly => false,
                                wgt::StorageTextureAccess::WriteOnly => true,
                                wgt::StorageTextureAccess::ReadWrite => true,
                            };
                        }
                    }

                    let source = naga::back::msl::BindSource {
                        stage: info.stage,
                        group: group_index as u32,
                        binding: entry.binding,
                    };
                    binding_map.insert(source, target);
                }
            }

            bind_group_infos.push(super::BindGroupLayoutInfo {
                base_resource_indices,
                dynamic_buffers,
                sized_buffer_bindings,
            });
        }

        // Finally, make sure we fit the limits
        for info in stage_data.iter_mut() {
            // handle the sizes buffer assignment and shader overrides
            if info.sizes_count != 0 {
                info.sizes_buffer = Some(info.counters.buffers);
                info.counters.buffers += 1;
            }
            if info.counters.buffers > self.shared.private_caps.max_buffers_per_stage
                || info.counters.textures > self.shared.private_caps.max_textures_per_stage
                || info.counters.samplers > self.shared.private_caps.max_samplers_per_stage
            {
                log::error!("Resource limit exceeded: {:?}", info);
                return Err(crate::DeviceError::OutOfMemory);
            }
        }

        let per_stage_map = stage_data.map(|info| naga::back::msl::PerStageResources {
            push_constant_buffer: info
                .pc_buffer
                .map(|buffer_index| buffer_index as naga::back::msl::Slot),
            sizes_buffer: info
                .sizes_buffer
                .map(|buffer_index| buffer_index as naga::back::msl::Slot),
        });

        let naga_options = naga::back::msl::Options {
            lang_version: match self.shared.private_caps.msl_version {
                mtl::MTLLanguageVersion::V1_0 => (1, 0),
                mtl::MTLLanguageVersion::V1_1 => (1, 1),
                mtl::MTLLanguageVersion::V1_2 => (1, 2),
                mtl::MTLLanguageVersion::V2_0 => (2, 0),
                mtl::MTLLanguageVersion::V2_1 => (2, 1),
                mtl::MTLLanguageVersion::V2_2 => (2, 2),
                mtl::MTLLanguageVersion::V2_3 => (2, 3),
            },
            binding_map,
            inline_samplers: Default::default(),
            spirv_cross_compatibility: false,
            fake_missing_bindings: false,
            per_stage_map: naga::back::msl::PerStageMap {
                vs: per_stage_map.vs,
                fs: per_stage_map.fs,
                cs: per_stage_map.cs,
            },
        };

        Ok(super::PipelineLayout {
            naga_options,
            bind_group_infos,
            push_constants_infos: stage_data.map(|info| {
                info.pc_buffer
                    .map(|buffer_index| super::PushConstantsStage {
                        count: info.pc_limit,
                        buffer_index,
                    })
            }),
        })
    }
    unsafe fn destroy_pipeline_layout(&self, _pipeline_layout: super::PipelineLayout) {}

    unsafe fn create_bind_group(
        &self,
        desc: &crate::BindGroupDescriptor<super::Api>,
    ) -> DeviceResult<super::BindGroup> {
        //TODO: avoid heap allocation
        let mut entries = desc.entries.to_vec();
        entries.sort_by_key(|e| e.binding);

        let mut bg = super::BindGroup::default();
        for (&stage, counter) in super::NAGA_STAGES.iter().zip(bg.counters.iter_mut()) {
            let stage_bit = crate::aux::map_naga_stage(stage);
            for entry in entries.iter() {
                let layout = &desc.layout.entries[&entry.binding];
                if !layout.visibility.contains(stage_bit) {
                    continue;
                }
                match layout.ty {
                    wgt::BindingType::Buffer { .. } => {
                        let source = &desc.buffers[entry.resource_index as usize];
                        bg.buffers.push(super::BufferResource {
                            ptr: source.buffer.as_raw(),
                            offset: source.offset,
                        });
                        counter.buffers += 1;
                    }
                    wgt::BindingType::Sampler { .. } => {
                        let res = desc.samplers[entry.resource_index as usize].as_raw();
                        bg.samplers.push(res);
                        counter.samplers += 1;
                    }
                    wgt::BindingType::Texture { .. } | wgt::BindingType::StorageTexture { .. } => {
                        let res = desc.textures[entry.resource_index as usize].view.as_raw();
                        bg.textures.push(res);
                        counter.textures += 1;
                    }
                }
            }
        }

        Ok(bg)
    }

    unsafe fn destroy_bind_group(&self, _group: super::BindGroup) {}

    unsafe fn create_shader_module(
        &self,
        desc: &crate::ShaderModuleDescriptor,
        shader: crate::NagaShader,
    ) -> Result<Resource, (crate::ShaderError, crate::NagaShader)> {
        Ok(Resource)
    }
    unsafe fn destroy_shader_module(&self, module: Resource) {}
    unsafe fn create_render_pipeline(
        &self,
        desc: &crate::RenderPipelineDescriptor<Api>,
    ) -> Result<Resource, crate::PipelineError> {
        Ok(Resource)
    }
    unsafe fn destroy_render_pipeline(&self, pipeline: Resource) {}
    unsafe fn create_compute_pipeline(
        &self,
        desc: &crate::ComputePipelineDescriptor<Api>,
    ) -> Result<Resource, crate::PipelineError> {
        Ok(Resource)
    }
    unsafe fn destroy_compute_pipeline(&self, pipeline: Resource) {}

    unsafe fn create_query_set(&self, desc: &wgt::QuerySetDescriptor) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_query_set(&self, set: Resource) {}
    unsafe fn create_fence(&self) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_fence(&self, fence: Resource) {}
    unsafe fn get_fence_value(&self, fence: &Resource) -> DeviceResult<crate::FenceValue> {
        Ok(0)
    }
    unsafe fn wait(
        &self,
        fence: &Resource,
        value: crate::FenceValue,
        timeout_ms: u32,
    ) -> DeviceResult<bool> {
        Ok(true)
    }

    unsafe fn start_capture(&self) -> bool {
        false
    }
    unsafe fn stop_capture(&self) {}
}
