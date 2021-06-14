use super::conv;

use arrayvec::ArrayVec;
use ash::{extensions::khr, version::DeviceV1_0, vk};
use inplace_it::inplace_or_alloc_from_iter;
use parking_lot::Mutex;

use std::{ptr, sync::Arc};

impl super::DeviceShared {
    unsafe fn set_object_name(
        &self,
        object_type: vk::ObjectType,
        object: impl vk::Handle,
        name: &str,
    ) {
        use std::ffi::CStr;

        let extension = match self.instance.debug_utils {
            Some(ref debug_utils) => &debug_utils.extension,
            None => return,
        };

        // Keep variables outside the if-else block to ensure they do not
        // go out of scope while we hold a pointer to them
        let mut buffer: [u8; 64] = [0u8; 64];
        let buffer_vec: Vec<u8>;

        // Append a null terminator to the string
        let name_bytes = if name.len() < buffer.len() {
            // Common case, string is very small. Allocate a copy on the stack.
            buffer.copy_from_slice(name.as_bytes());
            // Add null terminator
            buffer[name.len()] = 0;
            &buffer[..name.len() + 1]
        } else {
            // Less common case, the string is large.
            // This requires a heap allocation.
            buffer_vec = name
                .as_bytes()
                .iter()
                .cloned()
                .chain(std::iter::once(0))
                .collect();
            &buffer_vec
        };

        let _result = extension.debug_utils_set_object_name(
            self.raw.handle(),
            &vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(object_type)
                .object_handle(object.as_raw())
                .object_name(CStr::from_bytes_with_nul_unchecked(name_bytes)),
        );
    }
}

impl gpu_alloc::MemoryDevice<vk::DeviceMemory> for super::DeviceShared {
    unsafe fn allocate_memory(
        &self,
        size: u64,
        memory_type: u32,
        flags: gpu_alloc::AllocationFlags,
    ) -> Result<vk::DeviceMemory, gpu_alloc::OutOfMemory> {
        let mut info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(memory_type);

        let mut info_flags;

        if flags.contains(gpu_alloc::AllocationFlags::DEVICE_ADDRESS) {
            info_flags = vk::MemoryAllocateFlagsInfo::builder()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            info = info.push_next(&mut info_flags);
        }

        match self.raw.allocate_memory(&info, None) {
            Ok(memory) => Ok(memory),
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                Err(gpu_alloc::OutOfMemory::OutOfDeviceMemory)
            }
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => {
                Err(gpu_alloc::OutOfMemory::OutOfHostMemory)
            }
            Err(vk::Result::ERROR_TOO_MANY_OBJECTS) => panic!("Too many objects"),
            Err(err) => panic!("Unexpected Vulkan error: `{}`", err),
        }
    }

    unsafe fn deallocate_memory(&self, memory: vk::DeviceMemory) {
        self.raw.free_memory(memory, None);
    }

    unsafe fn map_memory(
        &self,
        memory: &mut vk::DeviceMemory,
        offset: u64,
        size: u64,
    ) -> Result<ptr::NonNull<u8>, gpu_alloc::DeviceMapError> {
        match self
            .raw
            .map_memory(*memory, offset, size, vk::MemoryMapFlags::empty())
        {
            Ok(ptr) => Ok(ptr::NonNull::new(ptr as *mut u8)
                .expect("Pointer to memory mapping must not be null")),
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                Err(gpu_alloc::DeviceMapError::OutOfDeviceMemory)
            }
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => {
                Err(gpu_alloc::DeviceMapError::OutOfHostMemory)
            }
            Err(vk::Result::ERROR_MEMORY_MAP_FAILED) => Err(gpu_alloc::DeviceMapError::MapFailed),
            Err(err) => panic!("Unexpected Vulkan error: `{}`", err),
        }
    }

    unsafe fn unmap_memory(&self, memory: &mut vk::DeviceMemory) {
        self.raw.unmap_memory(*memory);
    }

    unsafe fn invalidate_memory_ranges(
        &self,
        _ranges: &[gpu_alloc::MappedMemoryRange<'_, vk::DeviceMemory>],
    ) -> Result<(), gpu_alloc::OutOfMemory> {
        // should never be called
        unimplemented!()
    }

    unsafe fn flush_memory_ranges(
        &self,
        _ranges: &[gpu_alloc::MappedMemoryRange<'_, vk::DeviceMemory>],
    ) -> Result<(), gpu_alloc::OutOfMemory> {
        // should never be called
        unimplemented!()
    }
}

impl
    gpu_descriptor::DescriptorDevice<vk::DescriptorSetLayout, vk::DescriptorPool, vk::DescriptorSet>
    for super::DeviceShared
{
    unsafe fn create_descriptor_pool(
        &self,
        descriptor_count: &gpu_descriptor::DescriptorTotalCount,
        max_sets: u32,
        flags: gpu_descriptor::DescriptorPoolCreateFlags,
    ) -> Result<vk::DescriptorPool, gpu_descriptor::CreatePoolError> {
        //Note: ignoring other types, since they can't appear here
        let unfiltered_counts = [
            (vk::DescriptorType::SAMPLER, descriptor_count.sampler),
            (
                vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count.sampled_image,
            ),
            (
                vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count.storage_image,
            ),
            (
                vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count.uniform_buffer,
            ),
            (
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                descriptor_count.uniform_buffer_dynamic,
            ),
            (
                vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count.storage_buffer,
            ),
            (
                vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                descriptor_count.storage_buffer_dynamic,
            ),
        ];

        let filtered_counts = unfiltered_counts
            .iter()
            .cloned()
            .filter(|&(_, count)| count != 0)
            .map(|(ty, count)| vk::DescriptorPoolSize {
                ty,
                descriptor_count: count,
            })
            .collect::<ArrayVec<[_; 8]>>();

        let mut vk_flags = vk::DescriptorPoolCreateFlags::empty();
        if flags.contains(gpu_descriptor::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET) {
            vk_flags |= vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET;
        }
        let vk_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(max_sets)
            .flags(vk_flags)
            .pool_sizes(&filtered_counts)
            .build();

        match self.raw.create_descriptor_pool(&vk_info, None) {
            Ok(pool) => Ok(pool),
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => {
                Err(gpu_descriptor::CreatePoolError::OutOfHostMemory)
            }
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                Err(gpu_descriptor::CreatePoolError::OutOfDeviceMemory)
            }
            Err(vk::Result::ERROR_FRAGMENTATION) => {
                Err(gpu_descriptor::CreatePoolError::Fragmentation)
            }
            Err(other) => {
                log::error!("create_descriptor_pool: {:?}", other);
                Err(gpu_descriptor::CreatePoolError::OutOfHostMemory)
            }
        }
    }

    unsafe fn destroy_descriptor_pool(&self, pool: vk::DescriptorPool) {
        self.raw.destroy_descriptor_pool(pool, None)
    }

    unsafe fn alloc_descriptor_sets<'a>(
        &self,
        pool: &mut vk::DescriptorPool,
        layouts: impl ExactSizeIterator<Item = &'a vk::DescriptorSetLayout>,
        sets: &mut impl Extend<vk::DescriptorSet>,
    ) -> Result<(), gpu_descriptor::DeviceAllocationError> {
        let result = inplace_or_alloc_from_iter(layouts.cloned(), |layouts_slice| {
            let vk_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(*pool)
                .set_layouts(layouts_slice)
                .build();
            self.raw.allocate_descriptor_sets(&vk_info)
        });

        match result {
            Ok(vk_sets) => {
                sets.extend(vk_sets);
                Ok(())
            }
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY)
            | Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY) => {
                Err(gpu_descriptor::DeviceAllocationError::OutOfHostMemory)
            }
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                Err(gpu_descriptor::DeviceAllocationError::OutOfDeviceMemory)
            }
            Err(vk::Result::ERROR_FRAGMENTED_POOL) => {
                Err(gpu_descriptor::DeviceAllocationError::FragmentedPool)
            }
            Err(other) => {
                log::error!("allocate_descriptor_sets: {:?}", other);
                Err(gpu_descriptor::DeviceAllocationError::OutOfHostMemory)
            }
        }
    }

    unsafe fn dealloc_descriptor_sets<'a>(
        &self,
        pool: &mut vk::DescriptorPool,
        sets: impl Iterator<Item = vk::DescriptorSet>,
    ) {
        let result = inplace_or_alloc_from_iter(sets, |sets_slice| {
            self.raw.free_descriptor_sets(*pool, sets_slice)
        });
        match result {
            Ok(()) => {}
            Err(err) => log::error!("free_descriptor_sets: {:?}", err),
        }
    }
}

impl super::Device {
    pub(super) unsafe fn create_swapchain(
        &self,
        surface: &mut super::Surface,
        config: &crate::SurfaceConfiguration,
        provided_old_swapchain: Option<super::Swapchain>,
    ) -> Result<super::Swapchain, crate::SurfaceError> {
        let functor = khr::Swapchain::new(&surface.instance.raw, &self.shared.raw);

        let old_swapchain = match provided_old_swapchain {
            Some(osc) => osc.raw,
            None => vk::SwapchainKHR::null(),
        };

        let info = vk::SwapchainCreateInfoKHR::builder()
            .flags(vk::SwapchainCreateFlagsKHR::empty())
            .surface(surface.raw)
            .min_image_count(config.swap_chain_size)
            .image_format(self.shared.private_caps.map_texture_format(config.format))
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(vk::Extent2D {
                width: config.extent.width,
                height: config.extent.height,
            })
            .image_array_layers(config.extent.depth_or_array_layers)
            .image_usage(conv::map_texture_usage(config.usage))
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(conv::map_composite_alpha_mode(config.composite_alpha_mode))
            .present_mode(conv::map_present_mode(config.present_mode))
            .clipped(true)
            .old_swapchain(old_swapchain);

        let result = functor.create_swapchain(&info, None);

        // doing this before bailing out with error
        if old_swapchain != vk::SwapchainKHR::null() {
            functor.destroy_swapchain(old_swapchain, None)
        }

        let raw = match result {
            Ok(swapchain) => swapchain,
            Err(error) => {
                return Err(match error {
                    vk::Result::ERROR_SURFACE_LOST_KHR => crate::SurfaceError::Lost,
                    vk::Result::ERROR_NATIVE_WINDOW_IN_USE_KHR => {
                        crate::SurfaceError::Other("Native window is in use")
                    }
                    other => crate::DeviceError::from(other).into(),
                })
            }
        };

        let images = functor
            .get_swapchain_images(raw)
            .map_err(crate::DeviceError::from)?;

        let vk_info = vk::FenceCreateInfo::builder().build();
        let fence = self
            .shared
            .raw
            .create_fence(&vk_info, None)
            .map_err(crate::DeviceError::from)?;

        Ok(super::Swapchain {
            raw,
            functor,
            device: Arc::clone(&self.shared),
            fence,
            images,
        })
    }
}

use super::{DeviceResult, Resource}; //temporary

impl crate::Device<super::Api> for super::Device {
    unsafe fn create_buffer(
        &self,
        desc: &crate::BufferDescriptor,
    ) -> Result<super::Buffer, crate::DeviceError> {
        let vk_info = vk::BufferCreateInfo::builder()
            .size(desc.size)
            .usage(conv::map_buffer_usage(desc.usage))
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let raw = self.shared.raw.create_buffer(&vk_info, None)?;
        let req = self.shared.raw.get_buffer_memory_requirements(raw);

        let mut alloc_usage = if desc
            .usage
            .intersects(crate::BufferUse::MAP_READ | crate::BufferUse::MAP_WRITE)
        {
            let mut flags = gpu_alloc::UsageFlags::HOST_ACCESS;
            flags.set(
                gpu_alloc::UsageFlags::DOWNLOAD,
                desc.usage.contains(crate::BufferUse::MAP_READ),
            );
            flags.set(
                gpu_alloc::UsageFlags::UPLOAD,
                desc.usage.contains(crate::BufferUse::MAP_WRITE),
            );
            flags
        } else {
            gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS
        };
        alloc_usage.set(
            gpu_alloc::UsageFlags::TRANSIENT,
            desc.memory_flags.contains(crate::MemoryFlag::TRANSIENT),
        );

        let block = self.mem_allocator.lock().alloc(
            &*self.shared,
            gpu_alloc::Request {
                size: req.size,
                align_mask: req.alignment - 1,
                usage: alloc_usage,
                memory_types: req.memory_type_bits & self.valid_ash_memory_types,
            },
        )?;

        self.shared
            .raw
            .bind_buffer_memory(raw, *block.memory(), block.offset())?;

        if let Some(label) = desc.label {
            self.shared
                .set_object_name(vk::ObjectType::BUFFER, raw, label);
        }

        Ok(super::Buffer {
            raw,
            block: Mutex::new(block),
        })
    }
    unsafe fn destroy_buffer(&self, buffer: super::Buffer) {
        self.shared.raw.destroy_buffer(buffer.raw, None);
        self.mem_allocator
            .lock()
            .dealloc(&*self.shared, buffer.block.into_inner());
    }

    unsafe fn map_buffer(
        &self,
        buffer: &super::Buffer,
        range: crate::MemoryRange,
    ) -> Result<crate::BufferMapping, crate::DeviceError> {
        let size = range.end - range.start;
        let mut block = buffer.block.lock();
        let ptr = block.map(&*self.shared, range.start, size as usize)?;
        let is_coherent = block
            .props()
            .contains(gpu_alloc::MemoryPropertyFlags::HOST_COHERENT);
        Ok(crate::BufferMapping { ptr, is_coherent })
    }
    unsafe fn unmap_buffer(&self, buffer: &super::Buffer) -> Result<(), crate::DeviceError> {
        buffer.block.lock().unmap(&*self.shared);
        Ok(())
    }

    unsafe fn flush_mapped_ranges<I>(&self, buffer: &super::Buffer, ranges: I)
    where
        I: Iterator<Item = crate::MemoryRange>,
    {
        let block = buffer.block.lock();
        let vk_ranges = ranges.map(|range| {
            vk::MappedMemoryRange::builder()
                .memory(*block.memory())
                .offset(block.offset() + range.start)
                .size(range.end - range.start)
                .build()
        });
        inplace_or_alloc_from_iter(vk_ranges, |array| {
            self.shared.raw.flush_mapped_memory_ranges(array).unwrap()
        });
    }
    unsafe fn invalidate_mapped_ranges<I>(&self, buffer: &super::Buffer, ranges: I)
    where
        I: Iterator<Item = crate::MemoryRange>,
    {
        let block = buffer.block.lock();
        let vk_ranges = ranges.map(|range| {
            vk::MappedMemoryRange::builder()
                .memory(*block.memory())
                .offset(block.offset() + range.start)
                .size(range.end - range.start)
                .build()
        });
        inplace_or_alloc_from_iter(vk_ranges, |array| {
            self.shared
                .raw
                .invalidate_mapped_memory_ranges(array)
                .unwrap()
        });
    }

    unsafe fn create_texture(
        &self,
        desc: &crate::TextureDescriptor,
    ) -> Result<super::Texture, crate::DeviceError> {
        let (array_layer_count, vk_extent) = conv::map_extent(desc.size, desc.dimension);
        let mut flags = vk::ImageCreateFlags::empty();
        if desc.dimension == wgt::TextureDimension::D2 && desc.size.depth_or_array_layers % 6 == 0 {
            flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
        }

        let vk_info = vk::ImageCreateInfo::builder()
            .flags(flags)
            .image_type(conv::map_texture_dimension(desc.dimension))
            .format(self.shared.private_caps.map_texture_format(desc.format))
            .extent(vk_extent)
            .mip_levels(desc.mip_level_count)
            .array_layers(array_layer_count)
            .samples(vk::SampleCountFlags::from_raw(desc.sample_count))
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(conv::map_texture_usage(desc.usage))
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let raw = self.shared.raw.create_image(&vk_info, None)?;
        let req = self.shared.raw.get_image_memory_requirements(raw);

        let block = self.mem_allocator.lock().alloc(
            &*self.shared,
            gpu_alloc::Request {
                size: req.size,
                align_mask: req.alignment - 1,
                usage: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
                memory_types: req.memory_type_bits & self.valid_ash_memory_types,
            },
        )?;

        self.shared
            .raw
            .bind_image_memory(raw, *block.memory(), block.offset())?;

        if let Some(label) = desc.label {
            self.shared
                .set_object_name(vk::ObjectType::IMAGE, raw, label);
        }

        Ok(super::Texture {
            raw,
            block: Some(block),
            aspects: crate::FormatAspect::from(desc.format),
        })
    }
    unsafe fn destroy_texture(&self, texture: super::Texture) {
        self.shared.raw.destroy_image(texture.raw, None);
        self.mem_allocator
            .lock()
            .dealloc(&*self.shared, texture.block.unwrap());
    }

    unsafe fn create_texture_view(
        &self,
        texture: &super::Texture,
        desc: &crate::TextureViewDescriptor,
    ) -> Result<super::TextureView, crate::DeviceError> {
        let mut vk_info = vk::ImageViewCreateInfo::builder()
            .flags(vk::ImageViewCreateFlags::empty())
            .image(texture.raw)
            .view_type(conv::map_view_dimension(desc.dimension))
            .format(self.shared.private_caps.map_texture_format(desc.format))
            //.components(conv::map_swizzle(swizzle))
            .subresource_range(conv::map_subresource_range(&desc.range, texture.aspects));

        let mut image_view_info;
        if self.shared.private_caps.image_view_usage {
            image_view_info = vk::ImageViewUsageCreateInfo::builder()
                .usage(conv::map_texture_usage(desc.usage))
                .build();
            vk_info = vk_info.push_next(&mut image_view_info);
        }

        let raw = self.shared.raw.create_image_view(&vk_info, None)?;

        if let Some(label) = desc.label {
            self.shared
                .set_object_name(vk::ObjectType::IMAGE_VIEW, raw, label);
        }

        Ok(super::TextureView { raw })
    }
    unsafe fn destroy_texture_view(&self, view: super::TextureView) {
        self.shared.raw.destroy_image_view(view.raw, None);
    }

    unsafe fn create_sampler(
        &self,
        desc: &crate::SamplerDescriptor,
    ) -> Result<super::Sampler, crate::DeviceError> {
        let mut vk_info = vk::SamplerCreateInfo::builder()
            .flags(vk::SamplerCreateFlags::empty())
            .mag_filter(conv::map_filter_mode(desc.mag_filter))
            .min_filter(conv::map_filter_mode(desc.min_filter))
            .mipmap_mode(conv::map_mip_filter_mode(desc.mipmap_filter))
            .address_mode_u(conv::map_address_mode(desc.address_modes[0]))
            .address_mode_v(conv::map_address_mode(desc.address_modes[1]))
            .address_mode_w(conv::map_address_mode(desc.address_modes[2]));

        if let Some(fun) = desc.compare {
            vk_info = vk_info
                .compare_enable(true)
                .compare_op(conv::map_comparison(fun));
        }
        if let Some(ref range) = desc.lod_clamp {
            vk_info = vk_info.min_lod(range.start).max_lod(range.end);
        }
        if let Some(aniso) = desc.anisotropy_clamp {
            if self
                .shared
                .downlevel_flags
                .contains(wgt::DownlevelFlags::ANISOTROPIC_FILTERING)
            {
                vk_info = vk_info
                    .anisotropy_enable(true)
                    .max_anisotropy(aniso.get() as f32);
            }
        }
        if let Some(color) = desc.border_color {
            vk_info = vk_info.border_color(conv::map_border_color(color));
        }

        let raw = self.shared.raw.create_sampler(&vk_info, None)?;

        if let Some(label) = desc.label {
            self.shared
                .set_object_name(vk::ObjectType::SAMPLER, raw, label);
        }

        Ok(super::Sampler { raw })
    }
    unsafe fn destroy_sampler(&self, sampler: super::Sampler) {
        self.shared.raw.destroy_sampler(sampler.raw, None);
    }

    unsafe fn create_command_pool(
        &self,
        desc: &crate::CommandPoolDescriptor<super::Api>,
    ) -> DeviceResult<super::CommandPool> {
        let vk_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(desc.queue.family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .build();
        let raw = self.shared.raw.create_command_pool(&vk_info, None)?;

        Ok(super::CommandPool {
            raw,
            device: Arc::clone(&self.shared),
            free: Vec::new(),
        })
    }
    unsafe fn destroy_command_pool(&self, cmd_pool: super::CommandPool) {
        self.shared.raw.destroy_command_pool(cmd_pool.raw, None);
    }

    unsafe fn create_bind_group_layout(
        &self,
        desc: &crate::BindGroupLayoutDescriptor,
    ) -> Result<super::BindGroupLayout, crate::DeviceError> {
        let mut desc_count = gpu_descriptor::DescriptorTotalCount::default();
        for entry in desc.entries {
            let count = entry.count.map_or(1, |c| c.get());
            match entry.ty {
                wgt::BindingType::Buffer {
                    ty,
                    has_dynamic_offset,
                    ..
                } => match ty {
                    wgt::BufferBindingType::Uniform => {
                        if has_dynamic_offset {
                            desc_count.uniform_buffer_dynamic += count;
                        } else {
                            desc_count.uniform_buffer += count;
                        }
                    }
                    wgt::BufferBindingType::Storage { .. } => {
                        if has_dynamic_offset {
                            desc_count.storage_buffer_dynamic += count;
                        } else {
                            desc_count.storage_buffer += count;
                        }
                    }
                },
                wgt::BindingType::Sampler { .. } => {
                    desc_count.sampler += count;
                }
                wgt::BindingType::Texture { .. } => {
                    desc_count.sampled_image += count;
                }
                wgt::BindingType::StorageTexture { .. } => {
                    desc_count.storage_image += count;
                }
            }
        }

        //Note: not bothering with inplace_or_alloc_from_iter her as it's low frequency
        let vk_bindings = desc
            .entries
            .iter()
            .map(|entry| vk::DescriptorSetLayoutBinding {
                binding: entry.binding,
                descriptor_type: conv::map_binding_type(entry.ty),
                descriptor_count: entry.count.map_or(1, |c| c.get()),
                stage_flags: conv::map_shader_stage(entry.visibility),
                p_immutable_samplers: ptr::null(),
            })
            .collect::<Vec<_>>();

        let vk_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .flags(vk::DescriptorSetLayoutCreateFlags::empty())
            .bindings(&vk_bindings);

        let raw = self
            .shared
            .raw
            .create_descriptor_set_layout(&vk_info, None)?;

        if let Some(label) = desc.label {
            self.shared
                .set_object_name(vk::ObjectType::DESCRIPTOR_SET_LAYOUT, raw, label);
        }

        Ok(super::BindGroupLayout { raw, desc_count })
    }
    unsafe fn destroy_bind_group_layout(&self, bg_layout: super::BindGroupLayout) {
        self.shared
            .raw
            .destroy_descriptor_set_layout(bg_layout.raw, None);
    }

    unsafe fn create_pipeline_layout(
        &self,
        desc: &crate::PipelineLayoutDescriptor<super::Api>,
    ) -> Result<super::PipelineLayout, crate::DeviceError> {
        //Note: not bothering with inplace_or_alloc_from_iter her as it's low frequency
        let vk_set_layouts = desc
            .bind_group_layouts
            .iter()
            .map(|bgl| bgl.raw)
            .collect::<Vec<_>>();
        let vk_push_constant_ranges = desc
            .push_constant_ranges
            .iter()
            .map(|pcr| vk::PushConstantRange {
                stage_flags: conv::map_shader_stage(pcr.stages),
                offset: pcr.range.start,
                size: pcr.range.end - pcr.range.start,
            })
            .collect::<Vec<_>>();

        let vk_info = vk::PipelineLayoutCreateInfo::builder()
            .flags(vk::PipelineLayoutCreateFlags::empty())
            .set_layouts(&vk_set_layouts)
            .push_constant_ranges(&vk_push_constant_ranges);

        let raw = self.shared.raw.create_pipeline_layout(&vk_info, None)?;

        if let Some(label) = desc.label {
            self.shared
                .set_object_name(vk::ObjectType::PIPELINE_LAYOUT, raw, label);
        }

        Ok(super::PipelineLayout { raw })
    }
    unsafe fn destroy_pipeline_layout(&self, pipeline_layout: super::PipelineLayout) {
        self.shared
            .raw
            .destroy_pipeline_layout(pipeline_layout.raw, None);
    }

    unsafe fn create_bind_group(
        &self,
        desc: &crate::BindGroupDescriptor<super::Api>,
    ) -> Result<super::BindGroup, crate::DeviceError> {
        let mut vk_sets = self.desc_allocator.lock().allocate(
            &*self.shared,
            &desc.layout.raw,
            gpu_descriptor::DescriptorSetLayoutCreateFlags::empty(),
            &desc.layout.desc_count,
            1,
        )?;
        Ok(super::BindGroup {
            raw: vk_sets.pop().unwrap(),
        })
    }
    unsafe fn destroy_bind_group(&self, group: super::BindGroup) {
        self.desc_allocator
            .lock()
            .free(&*self.shared, Some(group.raw));
    }

    unsafe fn create_shader_module(
        &self,
        desc: &crate::ShaderModuleDescriptor,
        shader: crate::NagaShader,
    ) -> Result<Resource, crate::ShaderError> {
        Ok(Resource)
    }
    unsafe fn destroy_shader_module(&self, module: Resource) {}
    unsafe fn create_render_pipeline(
        &self,
        desc: &crate::RenderPipelineDescriptor<super::Api>,
    ) -> Result<Resource, crate::PipelineError> {
        Ok(Resource)
    }
    unsafe fn destroy_render_pipeline(&self, pipeline: Resource) {}
    unsafe fn create_compute_pipeline(
        &self,
        desc: &crate::ComputePipelineDescriptor<super::Api>,
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

impl From<gpu_alloc::AllocationError> for crate::DeviceError {
    fn from(error: gpu_alloc::AllocationError) -> Self {
        use gpu_alloc::AllocationError as Ae;
        match error {
            Ae::OutOfDeviceMemory | Ae::OutOfHostMemory => Self::OutOfMemory,
            _ => {
                log::error!("memory allocation: {:?}", error);
                Self::Lost
            }
        }
    }
}
impl From<gpu_alloc::MapError> for crate::DeviceError {
    fn from(error: gpu_alloc::MapError) -> Self {
        use gpu_alloc::MapError as Me;
        match error {
            Me::OutOfDeviceMemory | Me::OutOfHostMemory => Self::OutOfMemory,
            _ => {
                log::error!("memory mapping: {:?}", error);
                Self::Lost
            }
        }
    }
}
impl From<gpu_descriptor::AllocationError> for crate::DeviceError {
    fn from(error: gpu_descriptor::AllocationError) -> Self {
        log::error!("descriptor allocation: {:?}", error);
        Self::OutOfMemory
    }
}
