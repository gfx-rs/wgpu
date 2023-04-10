use parking_lot::Mutex;
use std::{
    num::NonZeroU32,
    ptr,
    sync::{atomic, Arc},
    thread, time,
};

use super::conv;
use crate::auxil::map_naga_stage;

type DeviceResult<T> = Result<T, crate::DeviceError>;

struct CompiledShader {
    library: metal::Library,
    function: metal::Function,
    wg_size: metal::MTLSize,
    wg_memory_sizes: Vec<u32>,

    /// Bindings of WGSL `storage` globals that contain variable-sized arrays.
    ///
    /// In order to implement bounds checks and the `arrayLength` function for
    /// WGSL runtime-sized arrays, we pass the entry point a struct with a
    /// member for each global variable that contains such an array. That member
    /// is a `u32` holding the variable's total size in bytes---which is simply
    /// the size of the `Buffer` supplying that variable's contents for the
    /// draw call.
    sized_bindings: Vec<naga::ResourceBinding>,

    immutable_buffer_mask: usize,
}

fn create_stencil_desc(
    face: &wgt::StencilFaceState,
    read_mask: u32,
    write_mask: u32,
) -> metal::StencilDescriptor {
    let desc = metal::StencilDescriptor::new();
    desc.set_stencil_compare_function(conv::map_compare_function(face.compare));
    desc.set_read_mask(read_mask);
    desc.set_write_mask(write_mask);
    desc.set_stencil_failure_operation(conv::map_stencil_op(face.fail_op));
    desc.set_depth_failure_operation(conv::map_stencil_op(face.depth_fail_op));
    desc.set_depth_stencil_pass_operation(conv::map_stencil_op(face.pass_op));
    desc
}

fn create_depth_stencil_desc(state: &wgt::DepthStencilState) -> metal::DepthStencilDescriptor {
    let desc = metal::DepthStencilDescriptor::new();
    desc.set_depth_compare_function(conv::map_compare_function(state.depth_compare));
    desc.set_depth_write_enabled(state.depth_write_enabled);
    let s = &state.stencil;
    if s.is_enabled() {
        let front_desc = create_stencil_desc(&s.front, s.read_mask, s.write_mask);
        desc.set_front_face_stencil(Some(&front_desc));
        let back_desc = create_stencil_desc(&s.back, s.read_mask, s.write_mask);
        desc.set_back_face_stencil(Some(&back_desc));
    }
    desc
}

impl super::Device {
    fn load_shader(
        &self,
        stage: &crate::ProgrammableStage<super::Api>,
        layout: &super::PipelineLayout,
        primitive_class: metal::MTLPrimitiveTopologyClass,
        naga_stage: naga::ShaderStage,
    ) -> Result<CompiledShader, crate::PipelineError> {
        let stage_bit = map_naga_stage(naga_stage);

        let module = &stage.module.naga.module;
        let ep_resources = &layout.per_stage_map[naga_stage];

        let bounds_check_policy = if stage.module.runtime_checks {
            naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite
        } else {
            naga::proc::BoundsCheckPolicy::Unchecked
        };

        let options = naga::back::msl::Options {
            lang_version: match self.shared.private_caps.msl_version {
                metal::MTLLanguageVersion::V1_0 => (1, 0),
                metal::MTLLanguageVersion::V1_1 => (1, 1),
                metal::MTLLanguageVersion::V1_2 => (1, 2),
                metal::MTLLanguageVersion::V2_0 => (2, 0),
                metal::MTLLanguageVersion::V2_1 => (2, 1),
                metal::MTLLanguageVersion::V2_2 => (2, 2),
                metal::MTLLanguageVersion::V2_3 => (2, 3),
                metal::MTLLanguageVersion::V2_4 => (2, 4),
            },
            inline_samplers: Default::default(),
            spirv_cross_compatibility: false,
            fake_missing_bindings: false,
            per_entry_point_map: naga::back::msl::EntryPointResourceMap::from([(
                stage.entry_point.to_string(),
                ep_resources.clone(),
            )]),
            bounds_check_policies: naga::proc::BoundsCheckPolicies {
                index: bounds_check_policy,
                buffer: bounds_check_policy,
                image: bounds_check_policy,
                // TODO: support bounds checks on binding arrays
                binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
            },
            zero_initialize_workgroup_memory: true,
        };

        let pipeline_options = naga::back::msl::PipelineOptions {
            allow_point_size: match primitive_class {
                metal::MTLPrimitiveTopologyClass::Point => true,
                _ => false,
            },
        };

        let (source, info) = naga::back::msl::write_string(
            module,
            &stage.module.naga.info,
            &options,
            &pipeline_options,
        )
        .map_err(|e| crate::PipelineError::Linkage(stage_bit, format!("MSL: {:?}", e)))?;

        log::debug!(
            "Naga generated shader for entry point '{}' and stage {:?}\n{}",
            stage.entry_point,
            naga_stage,
            &source
        );

        let options = metal::CompileOptions::new();
        options.set_language_version(self.shared.private_caps.msl_version);

        if self.shared.private_caps.supports_preserve_invariance {
            options.set_preserve_invariance(true);
        }

        let library = self
            .shared
            .device
            .lock()
            .new_library_with_source(source.as_ref(), &options)
            .map_err(|err| {
                log::warn!("Naga generated shader:\n{}", source);
                crate::PipelineError::Linkage(stage_bit, format!("Metal: {}", err))
            })?;

        let ep_index = module
            .entry_points
            .iter()
            .position(|ep| ep.stage == naga_stage && ep.name == stage.entry_point)
            .ok_or(crate::PipelineError::EntryPoint(naga_stage))?;
        let ep = &module.entry_points[ep_index];
        let ep_name = info.entry_point_names[ep_index]
            .as_ref()
            .map_err(|e| crate::PipelineError::Linkage(stage_bit, format!("{}", e)))?;

        let wg_size = metal::MTLSize {
            width: ep.workgroup_size[0] as _,
            height: ep.workgroup_size[1] as _,
            depth: ep.workgroup_size[2] as _,
        };

        let function = library.get_function(ep_name, None).map_err(|e| {
            log::error!("get_function: {:?}", e);
            crate::PipelineError::EntryPoint(naga_stage)
        })?;

        // collect sizes indices, immutable buffers, and work group memory sizes
        let ep_info = &stage.module.naga.info.get_entry_point(ep_index);
        let mut wg_memory_sizes = Vec::new();
        let mut sized_bindings = Vec::new();
        let mut immutable_buffer_mask = 0;
        for (var_handle, var) in module.global_variables.iter() {
            match var.space {
                naga::AddressSpace::WorkGroup => {
                    if !ep_info[var_handle].is_empty() {
                        let size = module.types[var.ty].inner.size(&module.constants);
                        wg_memory_sizes.push(size);
                    }
                }
                naga::AddressSpace::Uniform | naga::AddressSpace::Storage { .. } => {
                    let br = match var.binding {
                        Some(ref br) => br.clone(),
                        None => continue,
                    };
                    let storage_access_store = match var.space {
                        naga::AddressSpace::Storage { access } => {
                            access.contains(naga::StorageAccess::STORE)
                        }
                        _ => false,
                    };

                    // check for an immutable buffer
                    if !ep_info[var_handle].is_empty() && !storage_access_store {
                        let slot = ep_resources.resources[&br].buffer.unwrap();
                        immutable_buffer_mask |= 1 << slot;
                    }

                    let mut dynamic_array_container_ty = var.ty;
                    if let naga::TypeInner::Struct { ref members, .. } = module.types[var.ty].inner
                    {
                        dynamic_array_container_ty = members.last().unwrap().ty;
                    }
                    if let naga::TypeInner::Array {
                        size: naga::ArraySize::Dynamic,
                        ..
                    } = module.types[dynamic_array_container_ty].inner
                    {
                        sized_bindings.push(br);
                    }
                }
                _ => {}
            }
        }

        Ok(CompiledShader {
            library,
            function,
            wg_size,
            wg_memory_sizes,
            sized_bindings,
            immutable_buffer_mask,
        })
    }

    fn set_buffers_mutability(
        buffers: &metal::PipelineBufferDescriptorArrayRef,
        mut immutable_mask: usize,
    ) {
        while immutable_mask != 0 {
            let slot = immutable_mask.trailing_zeros();
            immutable_mask ^= 1 << slot;
            buffers
                .object_at(slot as u64)
                .unwrap()
                .set_mutability(metal::MTLMutability::Immutable);
        }
    }

    pub unsafe fn texture_from_raw(
        raw: metal::Texture,
        format: wgt::TextureFormat,
        raw_type: metal::MTLTextureType,
        array_layers: u32,
        mip_levels: u32,
        copy_size: crate::CopyExtent,
    ) -> super::Texture {
        super::Texture {
            raw,
            format,
            raw_type,
            array_layers,
            mip_levels,
            copy_size,
        }
    }

    pub unsafe fn device_from_raw(raw: metal::Device, features: wgt::Features) -> super::Device {
        super::Device {
            shared: Arc::new(super::AdapterShared::new(raw)),
            features,
        }
    }

    pub fn raw_device(&self) -> &Mutex<metal::Device> {
        &self.shared.device
    }
}

impl crate::Device<super::Api> for super::Device {
    unsafe fn exit(self, _queue: super::Queue) {}

    unsafe fn create_buffer(&self, desc: &crate::BufferDescriptor) -> DeviceResult<super::Buffer> {
        let map_read = desc.usage.contains(crate::BufferUses::MAP_READ);
        let map_write = desc.usage.contains(crate::BufferUses::MAP_WRITE);

        let mut options = metal::MTLResourceOptions::empty();
        options |= if map_read || map_write {
            // `crate::MemoryFlags::PREFER_COHERENT` is ignored here
            metal::MTLResourceOptions::StorageModeShared
        } else {
            metal::MTLResourceOptions::StorageModePrivate
        };
        options.set(
            metal::MTLResourceOptions::CPUCacheModeWriteCombined,
            map_write,
        );

        //TODO: HazardTrackingModeUntracked

        objc::rc::autoreleasepool(|| {
            let raw = self.shared.device.lock().new_buffer(desc.size, options);
            if let Some(label) = desc.label {
                raw.set_label(label);
            }
            Ok(super::Buffer {
                raw,
                size: desc.size,
            })
        })
    }
    unsafe fn destroy_buffer(&self, _buffer: super::Buffer) {}

    unsafe fn map_buffer(
        &self,
        buffer: &super::Buffer,
        range: crate::MemoryRange,
    ) -> DeviceResult<crate::BufferMapping> {
        let ptr = buffer.raw.contents() as *mut u8;
        assert!(!ptr.is_null());
        Ok(crate::BufferMapping {
            ptr: ptr::NonNull::new(unsafe { ptr.offset(range.start as isize) }).unwrap(),
            is_coherent: true,
        })
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
        use foreign_types::ForeignTypeRef;

        let mtl_format = self.shared.private_caps.map_format(desc.format);

        objc::rc::autoreleasepool(|| {
            let descriptor = metal::TextureDescriptor::new();

            let mtl_type = match desc.dimension {
                wgt::TextureDimension::D1 => metal::MTLTextureType::D1,
                wgt::TextureDimension::D2 => {
                    if desc.sample_count > 1 {
                        descriptor.set_sample_count(desc.sample_count as u64);
                        metal::MTLTextureType::D2Multisample
                    } else if desc.size.depth_or_array_layers > 1 {
                        descriptor.set_array_length(desc.size.depth_or_array_layers as u64);
                        metal::MTLTextureType::D2Array
                    } else {
                        metal::MTLTextureType::D2
                    }
                }
                wgt::TextureDimension::D3 => {
                    descriptor.set_depth(desc.size.depth_or_array_layers as u64);
                    metal::MTLTextureType::D3
                }
            };

            descriptor.set_texture_type(mtl_type);
            descriptor.set_width(desc.size.width as u64);
            descriptor.set_height(desc.size.height as u64);
            descriptor.set_mipmap_level_count(desc.mip_level_count as u64);
            descriptor.set_pixel_format(mtl_format);
            descriptor.set_usage(conv::map_texture_usage(desc.format, desc.usage));
            descriptor.set_storage_mode(metal::MTLStorageMode::Private);

            let raw = self.shared.device.lock().new_texture(&descriptor);
            if raw.as_ptr().is_null() {
                return Err(crate::DeviceError::OutOfMemory);
            }
            if let Some(label) = desc.label {
                raw.set_label(label);
            }

            Ok(super::Texture {
                raw,
                format: desc.format,
                raw_type: mtl_type,
                mip_levels: desc.mip_level_count,
                array_layers: desc.array_layer_count(),
                copy_size: desc.copy_extent(),
            })
        })
    }

    unsafe fn destroy_texture(&self, _texture: super::Texture) {}

    unsafe fn create_texture_view(
        &self,
        texture: &super::Texture,
        desc: &crate::TextureViewDescriptor,
    ) -> DeviceResult<super::TextureView> {
        let raw_type = if texture.raw_type == metal::MTLTextureType::D2Multisample {
            texture.raw_type
        } else {
            conv::map_texture_view_dimension(desc.dimension)
        };

        let aspects = crate::FormatAspects::new(desc.format, desc.range.aspect);

        let raw_format = self
            .shared
            .private_caps
            .map_view_format(desc.format, aspects);

        let format_equal = raw_format == self.shared.private_caps.map_format(texture.format);
        let type_equal = raw_type == texture.raw_type;
        let range_full_resource =
            desc.range
                .is_full_resource(desc.format, texture.mip_levels, texture.array_layers);

        let raw = if format_equal && type_equal && range_full_resource {
            // Some images are marked as framebuffer-only, and we can't create aliases of them.
            // Also helps working around Metal bugs with aliased array textures.
            texture.raw.to_owned()
        } else {
            let mip_level_count = desc
                .range
                .mip_level_count
                .unwrap_or(texture.mip_levels - desc.range.base_mip_level);
            let array_layer_count = desc
                .range
                .array_layer_count
                .unwrap_or(texture.array_layers - desc.range.base_array_layer);

            objc::rc::autoreleasepool(|| {
                let raw = texture.raw.new_texture_view_from_slice(
                    raw_format,
                    raw_type,
                    metal::NSRange {
                        location: desc.range.base_mip_level as _,
                        length: mip_level_count as _,
                    },
                    metal::NSRange {
                        location: desc.range.base_array_layer as _,
                        length: array_layer_count as _,
                    },
                );
                if let Some(label) = desc.label {
                    raw.set_label(label);
                }
                raw
            })
        };

        Ok(super::TextureView { raw, aspects })
    }
    unsafe fn destroy_texture_view(&self, _view: super::TextureView) {}

    unsafe fn create_sampler(
        &self,
        desc: &crate::SamplerDescriptor,
    ) -> DeviceResult<super::Sampler> {
        objc::rc::autoreleasepool(|| {
            let descriptor = metal::SamplerDescriptor::new();

            descriptor.set_min_filter(conv::map_filter_mode(desc.min_filter));
            descriptor.set_mag_filter(conv::map_filter_mode(desc.mag_filter));
            descriptor.set_mip_filter(match desc.mipmap_filter {
                wgt::FilterMode::Nearest if desc.lod_clamp == (0.0..0.0) => {
                    metal::MTLSamplerMipFilter::NotMipmapped
                }
                wgt::FilterMode::Nearest => metal::MTLSamplerMipFilter::Nearest,
                wgt::FilterMode::Linear => metal::MTLSamplerMipFilter::Linear,
            });

            let [s, t, r] = desc.address_modes;
            descriptor.set_address_mode_s(conv::map_address_mode(s));
            descriptor.set_address_mode_t(conv::map_address_mode(t));
            descriptor.set_address_mode_r(conv::map_address_mode(r));

            // Anisotropy is always supported on mac up to 16x
            descriptor.set_max_anisotropy(desc.anisotropy_clamp as _);

            descriptor.set_lod_min_clamp(desc.lod_clamp.start);
            descriptor.set_lod_max_clamp(desc.lod_clamp.end);

            if let Some(fun) = desc.compare {
                descriptor.set_compare_function(conv::map_compare_function(fun));
            }

            if let Some(border_color) = desc.border_color {
                if let wgt::SamplerBorderColor::Zero = border_color {
                    if s == wgt::AddressMode::ClampToBorder {
                        descriptor.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToZero);
                    }

                    if t == wgt::AddressMode::ClampToBorder {
                        descriptor.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToZero);
                    }

                    if r == wgt::AddressMode::ClampToBorder {
                        descriptor.set_address_mode_r(metal::MTLSamplerAddressMode::ClampToZero);
                    }
                } else {
                    descriptor.set_border_color(conv::map_border_color(border_color));
                }
            }

            if let Some(label) = desc.label {
                descriptor.set_label(label);
            }
            let raw = self.shared.device.lock().new_sampler(&descriptor);

            Ok(super::Sampler { raw })
        })
    }
    unsafe fn destroy_sampler(&self, _sampler: super::Sampler) {}

    unsafe fn create_command_encoder(
        &self,
        desc: &crate::CommandEncoderDescriptor<super::Api>,
    ) -> Result<super::CommandEncoder, crate::DeviceError> {
        Ok(super::CommandEncoder {
            shared: Arc::clone(&self.shared),
            raw_queue: Arc::clone(&desc.queue.raw),
            raw_cmd_buf: None,
            state: super::CommandState::default(),
            temp: super::Temp::default(),
        })
    }
    unsafe fn destroy_command_encoder(&self, _encoder: super::CommandEncoder) {}

    unsafe fn create_bind_group_layout(
        &self,
        desc: &crate::BindGroupLayoutDescriptor,
    ) -> DeviceResult<super::BindGroupLayout> {
        Ok(super::BindGroupLayout {
            entries: Arc::from(desc.entries),
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
            resources: naga::back::msl::BindingMap,
        }

        let mut stage_data = super::NAGA_STAGES.map(|stage| StageInfo {
            stage,
            counters: super::ResourceData::default(),
            pc_buffer: None,
            pc_limit: 0,
            sizes_buffer: None,
            sizes_count: 0,
            resources: Default::default(),
        });
        let mut bind_group_infos = arrayvec::ArrayVec::new();

        // First, place the push constants
        let mut total_push_constants = 0;
        for info in stage_data.iter_mut() {
            for pcr in desc.push_constant_ranges {
                if pcr.stages.contains(map_naga_stage(info.stage)) {
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

            total_push_constants = total_push_constants.max(info.pc_limit);
        }

        // Second, place the described resources
        for (group_index, &bgl) in desc.bind_group_layouts.iter().enumerate() {
            // remember where the resources for this set start at each shader stage
            let base_resource_indices = stage_data.map_ref(|info| info.counters.clone());

            for entry in bgl.entries.iter() {
                if let wgt::BindingType::Buffer {
                    ty: wgt::BufferBindingType::Storage { .. },
                    ..
                } = entry.ty
                {
                    for info in stage_data.iter_mut() {
                        if entry.visibility.contains(map_naga_stage(info.stage)) {
                            info.sizes_count += 1;
                        }
                    }
                }

                for info in stage_data.iter_mut() {
                    if !entry.visibility.contains(map_naga_stage(info.stage)) {
                        continue;
                    }

                    let mut target = naga::back::msl::BindTarget::default();
                    let count = entry.count.map_or(1, NonZeroU32::get);
                    target.binding_array_size = entry.count.map(NonZeroU32::get);
                    match entry.ty {
                        wgt::BindingType::Buffer { ty, .. } => {
                            target.buffer = Some(info.counters.buffers as _);
                            info.counters.buffers += count;
                            if let wgt::BufferBindingType::Storage { read_only } = ty {
                                target.mutable = !read_only;
                            }
                        }
                        wgt::BindingType::Sampler { .. } => {
                            target.sampler = Some(naga::back::msl::BindSamplerTarget::Resource(
                                info.counters.samplers as _,
                            ));
                            info.counters.samplers += count;
                        }
                        wgt::BindingType::Texture { .. } => {
                            target.texture = Some(info.counters.textures as _);
                            info.counters.textures += count;
                        }
                        wgt::BindingType::StorageTexture { access, .. } => {
                            target.texture = Some(info.counters.textures as _);
                            info.counters.textures += count;
                            target.mutable = match access {
                                wgt::StorageTextureAccess::ReadOnly => false,
                                wgt::StorageTextureAccess::WriteOnly => true,
                                wgt::StorageTextureAccess::ReadWrite => true,
                            };
                        }
                    }

                    let br = naga::ResourceBinding {
                        group: group_index as u32,
                        binding: entry.binding,
                    };
                    info.resources.insert(br, target);
                }
            }

            bind_group_infos.push(super::BindGroupLayoutInfo {
                base_resource_indices,
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

        let push_constants_infos = stage_data.map_ref(|info| {
            info.pc_buffer.map(|buffer_index| super::PushConstantsInfo {
                count: info.pc_limit,
                buffer_index,
            })
        });

        let total_counters = stage_data.map_ref(|info| info.counters.clone());

        let per_stage_map = stage_data.map(|info| naga::back::msl::EntryPointResources {
            push_constant_buffer: info
                .pc_buffer
                .map(|buffer_index| buffer_index as naga::back::msl::Slot),
            sizes_buffer: info
                .sizes_buffer
                .map(|buffer_index| buffer_index as naga::back::msl::Slot),
            resources: info.resources,
        });

        Ok(super::PipelineLayout {
            bind_group_infos,
            push_constants_infos,
            total_counters,
            total_push_constants,
            per_stage_map,
        })
    }
    unsafe fn destroy_pipeline_layout(&self, _pipeline_layout: super::PipelineLayout) {}

    unsafe fn create_bind_group(
        &self,
        desc: &crate::BindGroupDescriptor<super::Api>,
    ) -> DeviceResult<super::BindGroup> {
        let mut bg = super::BindGroup::default();
        for (&stage, counter) in super::NAGA_STAGES.iter().zip(bg.counters.iter_mut()) {
            let stage_bit = map_naga_stage(stage);
            let mut dynamic_offsets_count = 0u32;
            for (entry, layout) in desc.entries.iter().zip(desc.layout.entries.iter()) {
                let size = layout.count.map_or(1, |c| c.get());
                if let wgt::BindingType::Buffer {
                    has_dynamic_offset: true,
                    ..
                } = layout.ty
                {
                    dynamic_offsets_count += size;
                }
                if !layout.visibility.contains(stage_bit) {
                    continue;
                }
                match layout.ty {
                    wgt::BindingType::Buffer {
                        ty,
                        has_dynamic_offset,
                        ..
                    } => {
                        let start = entry.resource_index as usize;
                        let end = start + size as usize;
                        bg.buffers
                            .extend(desc.buffers[start..end].iter().map(|source| {
                                // Given the restrictions on `BufferBinding::offset`,
                                // this should never be `None`.
                                let remaining_size =
                                    wgt::BufferSize::new(source.buffer.size - source.offset);
                                let binding_size = match ty {
                                    wgt::BufferBindingType::Storage { .. } => {
                                        source.size.or(remaining_size)
                                    }
                                    _ => None,
                                };
                                super::BufferResource {
                                    ptr: source.buffer.as_raw(),
                                    offset: source.offset,
                                    dynamic_index: if has_dynamic_offset {
                                        Some(dynamic_offsets_count - 1)
                                    } else {
                                        None
                                    },
                                    binding_size,
                                    binding_location: layout.binding,
                                }
                            }));
                        counter.buffers += 1;
                    }
                    wgt::BindingType::Sampler { .. } => {
                        let start = entry.resource_index as usize;
                        let end = start + size as usize;
                        bg.samplers
                            .extend(desc.samplers[start..end].iter().map(|samp| samp.as_raw()));
                        counter.samplers += size;
                    }
                    wgt::BindingType::Texture { .. } | wgt::BindingType::StorageTexture { .. } => {
                        let start = entry.resource_index as usize;
                        let end = start + size as usize;
                        bg.textures.extend(
                            desc.textures[start..end]
                                .iter()
                                .map(|tex| tex.view.as_raw()),
                        );
                        counter.textures += size;
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
        shader: crate::ShaderInput,
    ) -> Result<super::ShaderModule, crate::ShaderError> {
        match shader {
            crate::ShaderInput::Naga(naga) => Ok(super::ShaderModule {
                naga,
                runtime_checks: desc.runtime_checks,
            }),
            crate::ShaderInput::SpirV(_) => {
                panic!("SPIRV_SHADER_PASSTHROUGH is not enabled for this backend")
            }
        }
    }
    unsafe fn destroy_shader_module(&self, _module: super::ShaderModule) {}

    unsafe fn create_render_pipeline(
        &self,
        desc: &crate::RenderPipelineDescriptor<super::Api>,
    ) -> Result<super::RenderPipeline, crate::PipelineError> {
        objc::rc::autoreleasepool(|| {
            let descriptor = metal::RenderPipelineDescriptor::new();

            let raw_triangle_fill_mode = match desc.primitive.polygon_mode {
                wgt::PolygonMode::Fill => metal::MTLTriangleFillMode::Fill,
                wgt::PolygonMode::Line => metal::MTLTriangleFillMode::Lines,
                wgt::PolygonMode::Point => panic!(
                    "{:?} is not enabled for this backend",
                    wgt::Features::POLYGON_MODE_POINT
                ),
            };

            let (primitive_class, raw_primitive_type) =
                conv::map_primitive_topology(desc.primitive.topology);

            // Vertex shader
            let (vs_lib, vs_info) = {
                let vs = self.load_shader(
                    &desc.vertex_stage,
                    desc.layout,
                    primitive_class,
                    naga::ShaderStage::Vertex,
                )?;

                descriptor.set_vertex_function(Some(&vs.function));
                if self.shared.private_caps.supports_mutability {
                    Self::set_buffers_mutability(
                        descriptor.vertex_buffers().unwrap(),
                        vs.immutable_buffer_mask,
                    );
                }

                let info = super::PipelineStageInfo {
                    push_constants: desc.layout.push_constants_infos.vs,
                    sizes_slot: desc.layout.per_stage_map.vs.sizes_buffer,
                    sized_bindings: vs.sized_bindings,
                };

                (vs.library, info)
            };

            // Fragment shader
            let (fs_lib, fs_info) = match desc.fragment_stage {
                Some(ref stage) => {
                    let fs = self.load_shader(
                        stage,
                        desc.layout,
                        primitive_class,
                        naga::ShaderStage::Fragment,
                    )?;

                    descriptor.set_fragment_function(Some(&fs.function));
                    if self.shared.private_caps.supports_mutability {
                        Self::set_buffers_mutability(
                            descriptor.fragment_buffers().unwrap(),
                            fs.immutable_buffer_mask,
                        );
                    }

                    let info = super::PipelineStageInfo {
                        push_constants: desc.layout.push_constants_infos.fs,
                        sizes_slot: desc.layout.per_stage_map.fs.sizes_buffer,
                        sized_bindings: fs.sized_bindings,
                    };

                    (Some(fs.library), Some(info))
                }
                None => {
                    // TODO: This is a workaround for what appears to be a Metal validation bug
                    // A pixel format is required even though no attachments are provided
                    if desc.color_targets.is_empty() && desc.depth_stencil.is_none() {
                        descriptor
                            .set_depth_attachment_pixel_format(metal::MTLPixelFormat::Depth32Float);
                    }
                    (None, None)
                }
            };

            for (i, ct) in desc.color_targets.iter().enumerate() {
                let at_descriptor = descriptor.color_attachments().object_at(i as u64).unwrap();
                let ct = if let Some(color_target) = ct.as_ref() {
                    color_target
                } else {
                    at_descriptor.set_pixel_format(metal::MTLPixelFormat::Invalid);
                    continue;
                };

                let raw_format = self.shared.private_caps.map_format(ct.format);
                at_descriptor.set_pixel_format(raw_format);
                at_descriptor.set_write_mask(conv::map_color_write(ct.write_mask));

                if let Some(ref blend) = ct.blend {
                    at_descriptor.set_blending_enabled(true);
                    let (color_op, color_src, color_dst) = conv::map_blend_component(&blend.color);
                    let (alpha_op, alpha_src, alpha_dst) = conv::map_blend_component(&blend.alpha);

                    at_descriptor.set_rgb_blend_operation(color_op);
                    at_descriptor.set_source_rgb_blend_factor(color_src);
                    at_descriptor.set_destination_rgb_blend_factor(color_dst);

                    at_descriptor.set_alpha_blend_operation(alpha_op);
                    at_descriptor.set_source_alpha_blend_factor(alpha_src);
                    at_descriptor.set_destination_alpha_blend_factor(alpha_dst);
                }
            }

            let depth_stencil = match desc.depth_stencil {
                Some(ref ds) => {
                    let raw_format = self.shared.private_caps.map_format(ds.format);
                    let aspects = crate::FormatAspects::from(ds.format);
                    if aspects.contains(crate::FormatAspects::DEPTH) {
                        descriptor.set_depth_attachment_pixel_format(raw_format);
                    }
                    if aspects.contains(crate::FormatAspects::STENCIL) {
                        descriptor.set_stencil_attachment_pixel_format(raw_format);
                    }

                    let ds_descriptor = create_depth_stencil_desc(ds);
                    let raw = self
                        .shared
                        .device
                        .lock()
                        .new_depth_stencil_state(&ds_descriptor);
                    Some((raw, ds.bias))
                }
                None => None,
            };

            if desc.layout.total_counters.vs.buffers + (desc.vertex_buffers.len() as u32)
                > self.shared.private_caps.max_vertex_buffers
            {
                let msg = format!(
                    "pipeline needs too many buffers in the vertex stage: {} vertex and {} layout",
                    desc.vertex_buffers.len(),
                    desc.layout.total_counters.vs.buffers
                );
                return Err(crate::PipelineError::Linkage(
                    wgt::ShaderStages::VERTEX,
                    msg,
                ));
            }

            if !desc.vertex_buffers.is_empty() {
                let vertex_descriptor = metal::VertexDescriptor::new();
                for (i, vb) in desc.vertex_buffers.iter().enumerate() {
                    let buffer_index =
                        self.shared.private_caps.max_vertex_buffers as u64 - 1 - i as u64;
                    let buffer_desc = vertex_descriptor.layouts().object_at(buffer_index).unwrap();

                    // Metal expects the stride to be the actual size of the attributes.
                    // The semantics of array_stride == 0 can be achieved by setting
                    // the step function to constant and rate to 0.
                    if vb.array_stride == 0 {
                        let stride = vb
                            .attributes
                            .iter()
                            .map(|attribute| attribute.offset + attribute.format.size())
                            .max()
                            .unwrap_or(0);
                        buffer_desc.set_stride(wgt::math::align_to(stride, 4));
                        buffer_desc.set_step_function(metal::MTLVertexStepFunction::Constant);
                        buffer_desc.set_step_rate(0);
                    } else {
                        buffer_desc.set_stride(vb.array_stride);
                        buffer_desc.set_step_function(conv::map_step_mode(vb.step_mode));
                    }

                    for at in vb.attributes {
                        let attribute_desc = vertex_descriptor
                            .attributes()
                            .object_at(at.shader_location as u64)
                            .unwrap();
                        attribute_desc.set_format(conv::map_vertex_format(at.format));
                        attribute_desc.set_buffer_index(buffer_index);
                        attribute_desc.set_offset(at.offset);
                    }
                }
                descriptor.set_vertex_descriptor(Some(vertex_descriptor));
            }

            if desc.multisample.count != 1 {
                //TODO: handle sample mask
                descriptor.set_sample_count(desc.multisample.count as u64);
                descriptor
                    .set_alpha_to_coverage_enabled(desc.multisample.alpha_to_coverage_enabled);
                //descriptor.set_alpha_to_one_enabled(desc.multisample.alpha_to_one_enabled);
            }

            if let Some(name) = desc.label {
                descriptor.set_label(name);
            }

            let raw = self
                .shared
                .device
                .lock()
                .new_render_pipeline_state(&descriptor)
                .map_err(|e| {
                    crate::PipelineError::Linkage(
                        wgt::ShaderStages::VERTEX | wgt::ShaderStages::FRAGMENT,
                        format!("new_render_pipeline_state: {:?}", e),
                    )
                })?;

            Ok(super::RenderPipeline {
                raw,
                vs_lib,
                fs_lib,
                vs_info,
                fs_info,
                raw_primitive_type,
                raw_triangle_fill_mode,
                raw_front_winding: conv::map_winding(desc.primitive.front_face),
                raw_cull_mode: conv::map_cull_mode(desc.primitive.cull_mode),
                raw_depth_clip_mode: if self.features.contains(wgt::Features::DEPTH_CLIP_CONTROL) {
                    Some(if desc.primitive.unclipped_depth {
                        metal::MTLDepthClipMode::Clamp
                    } else {
                        metal::MTLDepthClipMode::Clip
                    })
                } else {
                    None
                },
                depth_stencil,
            })
        })
    }
    unsafe fn destroy_render_pipeline(&self, _pipeline: super::RenderPipeline) {}

    unsafe fn create_compute_pipeline(
        &self,
        desc: &crate::ComputePipelineDescriptor<super::Api>,
    ) -> Result<super::ComputePipeline, crate::PipelineError> {
        objc::rc::autoreleasepool(|| {
            let descriptor = metal::ComputePipelineDescriptor::new();

            let cs = self.load_shader(
                &desc.stage,
                desc.layout,
                metal::MTLPrimitiveTopologyClass::Unspecified,
                naga::ShaderStage::Compute,
            )?;
            descriptor.set_compute_function(Some(&cs.function));

            if self.shared.private_caps.supports_mutability {
                Self::set_buffers_mutability(
                    descriptor.buffers().unwrap(),
                    cs.immutable_buffer_mask,
                );
            }

            let cs_info = super::PipelineStageInfo {
                push_constants: desc.layout.push_constants_infos.cs,
                sizes_slot: desc.layout.per_stage_map.cs.sizes_buffer,
                sized_bindings: cs.sized_bindings,
            };

            if let Some(name) = desc.label {
                descriptor.set_label(name);
            }

            let raw = self
                .shared
                .device
                .lock()
                .new_compute_pipeline_state(&descriptor)
                .map_err(|e| {
                    crate::PipelineError::Linkage(
                        wgt::ShaderStages::COMPUTE,
                        format!("new_compute_pipeline_state: {:?}", e),
                    )
                })?;

            Ok(super::ComputePipeline {
                raw,
                cs_info,
                cs_lib: cs.library,
                work_group_size: cs.wg_size,
                work_group_memory_sizes: cs.wg_memory_sizes,
            })
        })
    }
    unsafe fn destroy_compute_pipeline(&self, _pipeline: super::ComputePipeline) {}

    unsafe fn create_query_set(
        &self,
        desc: &wgt::QuerySetDescriptor<crate::Label>,
    ) -> DeviceResult<super::QuerySet> {
        objc::rc::autoreleasepool(|| {
            match desc.ty {
                wgt::QueryType::Occlusion => {
                    let size = desc.count as u64 * crate::QUERY_SIZE;
                    let options = metal::MTLResourceOptions::empty();
                    //TODO: HazardTrackingModeUntracked
                    let raw_buffer = self.shared.device.lock().new_buffer(size, options);
                    if let Some(label) = desc.label {
                        raw_buffer.set_label(label);
                    }
                    Ok(super::QuerySet {
                        raw_buffer,
                        ty: desc.ty,
                    })
                }
                wgt::QueryType::Timestamp | wgt::QueryType::PipelineStatistics(_) => {
                    Err(crate::DeviceError::OutOfMemory)
                }
            }
        })
    }
    unsafe fn destroy_query_set(&self, _set: super::QuerySet) {}

    unsafe fn create_fence(&self) -> DeviceResult<super::Fence> {
        Ok(super::Fence {
            completed_value: Arc::new(atomic::AtomicU64::new(0)),
            pending_command_buffers: Vec::new(),
        })
    }
    unsafe fn destroy_fence(&self, _fence: super::Fence) {}
    unsafe fn get_fence_value(&self, fence: &super::Fence) -> DeviceResult<crate::FenceValue> {
        let mut max_value = fence.completed_value.load(atomic::Ordering::Acquire);
        for &(value, ref cmd_buf) in fence.pending_command_buffers.iter() {
            if cmd_buf.status() == metal::MTLCommandBufferStatus::Completed {
                max_value = value;
            }
        }
        Ok(max_value)
    }
    unsafe fn wait(
        &self,
        fence: &super::Fence,
        wait_value: crate::FenceValue,
        timeout_ms: u32,
    ) -> DeviceResult<bool> {
        if wait_value <= fence.completed_value.load(atomic::Ordering::Acquire) {
            return Ok(true);
        }

        let cmd_buf = match fence
            .pending_command_buffers
            .iter()
            .find(|&&(value, _)| value >= wait_value)
        {
            Some(&(_, ref cmd_buf)) => cmd_buf,
            None => {
                log::error!("No active command buffers for fence value {}", wait_value);
                return Err(crate::DeviceError::Lost);
            }
        };

        let start = time::Instant::now();
        loop {
            if let metal::MTLCommandBufferStatus::Completed = cmd_buf.status() {
                return Ok(true);
            }
            if start.elapsed().as_millis() >= timeout_ms as u128 {
                return Ok(false);
            }
            thread::sleep(time::Duration::from_millis(1));
        }
    }

    unsafe fn start_capture(&self) -> bool {
        if !self.shared.private_caps.supports_capture_manager {
            return false;
        }
        let device = self.shared.device.lock();
        let shared_capture_manager = metal::CaptureManager::shared();
        let default_capture_scope = shared_capture_manager.new_capture_scope_with_device(&device);
        shared_capture_manager.set_default_capture_scope(&default_capture_scope);
        shared_capture_manager.start_capture_with_scope(&default_capture_scope);
        default_capture_scope.begin_scope();
        true
    }
    unsafe fn stop_capture(&self) {
        let shared_capture_manager = metal::CaptureManager::shared();
        if let Some(default_capture_scope) = shared_capture_manager.default_capture_scope() {
            default_capture_scope.end_scope();
        }
        shared_capture_manager.stop_capture();
    }
}
