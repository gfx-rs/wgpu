use super::conv;

use arrayvec::ArrayVec;
use ash::{extensions::khr, vk};
use parking_lot::Mutex;

use std::{
    borrow::Cow,
    collections::{hash_map::Entry, BTreeMap},
    ffi::{CStr, CString},
    num::NonZeroU32,
    ptr,
    sync::Arc,
};

impl super::DeviceShared {
    pub(super) unsafe fn set_object_name(
        &self,
        object_type: vk::ObjectType,
        object: impl vk::Handle,
        name: &str,
    ) {
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
            buffer[..name.len()].copy_from_slice(name.as_bytes());
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

        let name = unsafe { CStr::from_bytes_with_nul_unchecked(name_bytes) };

        let _result = unsafe {
            extension.set_debug_utils_object_name(
                self.raw.handle(),
                &vk::DebugUtilsObjectNameInfoEXT::builder()
                    .object_type(object_type)
                    .object_handle(object.as_raw())
                    .object_name(name),
            )
        };
    }

    pub fn make_render_pass(
        &self,
        key: super::RenderPassKey,
    ) -> Result<vk::RenderPass, crate::DeviceError> {
        Ok(match self.render_passes.lock().entry(key) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let mut vk_attachments = Vec::new();
                let mut color_refs = Vec::with_capacity(e.key().colors.len());
                let mut resolve_refs = Vec::with_capacity(color_refs.capacity());
                let mut ds_ref = None;
                let samples = vk::SampleCountFlags::from_raw(e.key().sample_count);
                let unused = vk::AttachmentReference {
                    attachment: vk::ATTACHMENT_UNUSED,
                    layout: vk::ImageLayout::UNDEFINED,
                };
                for cat in e.key().colors.iter() {
                    let (color_ref, resolve_ref) = if let Some(cat) = cat.as_ref() {
                        let color_ref = vk::AttachmentReference {
                            attachment: vk_attachments.len() as u32,
                            layout: cat.base.layout,
                        };
                        vk_attachments.push({
                            let (load_op, store_op) = conv::map_attachment_ops(cat.base.ops);
                            vk::AttachmentDescription::builder()
                                .format(cat.base.format)
                                .samples(samples)
                                .load_op(load_op)
                                .store_op(store_op)
                                .initial_layout(cat.base.layout)
                                .final_layout(cat.base.layout)
                                .build()
                        });
                        let resolve_ref = if let Some(ref rat) = cat.resolve {
                            let (load_op, store_op) = conv::map_attachment_ops(rat.ops);
                            let vk_attachment = vk::AttachmentDescription::builder()
                                .format(rat.format)
                                .samples(vk::SampleCountFlags::TYPE_1)
                                .load_op(load_op)
                                .store_op(store_op)
                                .initial_layout(rat.layout)
                                .final_layout(rat.layout)
                                .build();
                            vk_attachments.push(vk_attachment);

                            vk::AttachmentReference {
                                attachment: vk_attachments.len() as u32 - 1,
                                layout: rat.layout,
                            }
                        } else {
                            unused
                        };

                        (color_ref, resolve_ref)
                    } else {
                        (unused, unused)
                    };

                    color_refs.push(color_ref);
                    resolve_refs.push(resolve_ref);
                }

                if let Some(ref ds) = e.key().depth_stencil {
                    ds_ref = Some(vk::AttachmentReference {
                        attachment: vk_attachments.len() as u32,
                        layout: ds.base.layout,
                    });
                    let (load_op, store_op) = conv::map_attachment_ops(ds.base.ops);
                    let (stencil_load_op, stencil_store_op) =
                        conv::map_attachment_ops(ds.stencil_ops);
                    let vk_attachment = vk::AttachmentDescription::builder()
                        .format(ds.base.format)
                        .samples(samples)
                        .load_op(load_op)
                        .store_op(store_op)
                        .stencil_load_op(stencil_load_op)
                        .stencil_store_op(stencil_store_op)
                        .initial_layout(ds.base.layout)
                        .final_layout(ds.base.layout)
                        .build();
                    vk_attachments.push(vk_attachment);
                }

                let vk_subpasses = [{
                    let mut vk_subpass = vk::SubpassDescription::builder()
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                        .color_attachments(&color_refs)
                        .resolve_attachments(&resolve_refs);

                    if self
                        .workarounds
                        .contains(super::Workarounds::EMPTY_RESOLVE_ATTACHMENT_LISTS)
                        && resolve_refs.is_empty()
                    {
                        vk_subpass.p_resolve_attachments = ptr::null();
                    }

                    if let Some(ref reference) = ds_ref {
                        vk_subpass = vk_subpass.depth_stencil_attachment(reference)
                    }
                    vk_subpass.build()
                }];

                let mut vk_info = vk::RenderPassCreateInfo::builder()
                    .attachments(&vk_attachments)
                    .subpasses(&vk_subpasses);

                let mut multiview_info;
                let mask;
                if let Some(multiview) = e.key().multiview {
                    // Sanity checks, better to panic here than cause a driver crash
                    assert!(multiview.get() <= 8);
                    assert!(multiview.get() > 1);

                    // Right now we enable all bits on the view masks and correlation masks.
                    // This means we're rendering to all views in the subpass, and that all views
                    // can be rendered concurrently.
                    mask = [(1 << multiview.get()) - 1];

                    // On Vulkan 1.1 or later, this is an alias for core functionality
                    multiview_info = vk::RenderPassMultiviewCreateInfoKHR::builder()
                        .view_masks(&mask)
                        .correlation_masks(&mask)
                        .build();
                    vk_info = vk_info.push_next(&mut multiview_info);
                }

                let raw = unsafe { self.raw.create_render_pass(&vk_info, None)? };

                *e.insert(raw)
            }
        })
    }

    pub fn make_framebuffer(
        &self,
        key: super::FramebufferKey,
        raw_pass: vk::RenderPass,
        pass_label: crate::Label,
    ) -> Result<vk::Framebuffer, crate::DeviceError> {
        Ok(match self.framebuffers.lock().entry(key) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let vk_views = e
                    .key()
                    .attachments
                    .iter()
                    .map(|at| at.raw)
                    .collect::<ArrayVec<_, { super::MAX_TOTAL_ATTACHMENTS }>>();
                let vk_view_formats = e
                    .key()
                    .attachments
                    .iter()
                    .map(|at| self.private_caps.map_texture_format(at.view_format))
                    .collect::<ArrayVec<_, { super::MAX_TOTAL_ATTACHMENTS }>>();
                let vk_view_formats_list = e
                    .key()
                    .attachments
                    .iter()
                    .map(|at| at.raw_view_formats.clone())
                    .collect::<ArrayVec<_, { super::MAX_TOTAL_ATTACHMENTS }>>();

                let vk_image_infos = e
                    .key()
                    .attachments
                    .iter()
                    .enumerate()
                    .map(|(i, at)| {
                        let mut info = vk::FramebufferAttachmentImageInfo::builder()
                            .usage(conv::map_texture_usage(at.view_usage))
                            .flags(at.raw_image_flags)
                            .width(e.key().extent.width)
                            .height(e.key().extent.height)
                            .layer_count(e.key().extent.depth_or_array_layers);
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkRenderPassBeginInfo.html#VUID-VkRenderPassBeginInfo-framebuffer-03214
                        if vk_view_formats_list[i].is_empty() {
                            info = info.view_formats(&vk_view_formats[i..i + 1]);
                        } else {
                            info = info.view_formats(&vk_view_formats_list[i]);
                        };
                        info.build()
                    })
                    .collect::<ArrayVec<_, { super::MAX_TOTAL_ATTACHMENTS }>>();

                let mut vk_attachment_info = vk::FramebufferAttachmentsCreateInfo::builder()
                    .attachment_image_infos(&vk_image_infos)
                    .build();
                let mut vk_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(raw_pass)
                    .width(e.key().extent.width)
                    .height(e.key().extent.height)
                    .layers(e.key().extent.depth_or_array_layers);

                if self.private_caps.imageless_framebuffers {
                    //TODO: https://github.com/MaikKlein/ash/issues/450
                    vk_info = vk_info
                        .flags(vk::FramebufferCreateFlags::IMAGELESS_KHR)
                        .push_next(&mut vk_attachment_info);
                    vk_info.attachment_count = e.key().attachments.len() as u32;
                } else {
                    vk_info = vk_info.attachments(&vk_views);
                }

                *e.insert(unsafe {
                    let raw = self.raw.create_framebuffer(&vk_info, None).unwrap();
                    if let Some(label) = pass_label {
                        self.set_object_name(vk::ObjectType::FRAMEBUFFER, raw, label);
                    }
                    raw
                })
            }
        })
    }

    fn make_memory_ranges<'a, I: 'a + Iterator<Item = crate::MemoryRange>>(
        &self,
        buffer: &'a super::Buffer,
        ranges: I,
    ) -> impl 'a + Iterator<Item = vk::MappedMemoryRange> {
        let block = buffer.block.lock();
        let mask = self.private_caps.non_coherent_map_mask;
        ranges.map(move |range| {
            vk::MappedMemoryRange::builder()
                .memory(*block.memory())
                .offset((block.offset() + range.start) & !mask)
                .size((range.end - range.start + mask) & !mask)
                .build()
        })
    }

    unsafe fn free_resources(&self) {
        for &raw in self.render_passes.lock().values() {
            unsafe { self.raw.destroy_render_pass(raw, None) };
        }
        for &raw in self.framebuffers.lock().values() {
            unsafe { self.raw.destroy_framebuffer(raw, None) };
        }
        if self.handle_is_owned {
            unsafe { self.raw.destroy_device(None) };
        }
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

        match unsafe { self.raw.allocate_memory(&info, None) } {
            Ok(memory) => Ok(memory),
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                Err(gpu_alloc::OutOfMemory::OutOfDeviceMemory)
            }
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => {
                Err(gpu_alloc::OutOfMemory::OutOfHostMemory)
            }
            Err(vk::Result::ERROR_TOO_MANY_OBJECTS) => panic!("Too many objects"),
            Err(err) => panic!("Unexpected Vulkan error: `{err}`"),
        }
    }

    unsafe fn deallocate_memory(&self, memory: vk::DeviceMemory) {
        unsafe { self.raw.free_memory(memory, None) };
    }

    unsafe fn map_memory(
        &self,
        memory: &mut vk::DeviceMemory,
        offset: u64,
        size: u64,
    ) -> Result<ptr::NonNull<u8>, gpu_alloc::DeviceMapError> {
        match unsafe {
            self.raw
                .map_memory(*memory, offset, size, vk::MemoryMapFlags::empty())
        } {
            Ok(ptr) => Ok(ptr::NonNull::new(ptr as *mut u8)
                .expect("Pointer to memory mapping must not be null")),
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                Err(gpu_alloc::DeviceMapError::OutOfDeviceMemory)
            }
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => {
                Err(gpu_alloc::DeviceMapError::OutOfHostMemory)
            }
            Err(vk::Result::ERROR_MEMORY_MAP_FAILED) => Err(gpu_alloc::DeviceMapError::MapFailed),
            Err(err) => panic!("Unexpected Vulkan error: `{err}`"),
        }
    }

    unsafe fn unmap_memory(&self, memory: &mut vk::DeviceMemory) {
        unsafe { self.raw.unmap_memory(*memory) };
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
            .collect::<ArrayVec<_, 8>>();

        let mut vk_flags =
            if flags.contains(gpu_descriptor::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND) {
                vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND
            } else {
                vk::DescriptorPoolCreateFlags::empty()
            };
        if flags.contains(gpu_descriptor::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET) {
            vk_flags |= vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET;
        }
        let vk_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(max_sets)
            .flags(vk_flags)
            .pool_sizes(&filtered_counts)
            .build();

        match unsafe { self.raw.create_descriptor_pool(&vk_info, None) } {
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
        unsafe { self.raw.destroy_descriptor_pool(pool, None) }
    }

    unsafe fn alloc_descriptor_sets<'a>(
        &self,
        pool: &mut vk::DescriptorPool,
        layouts: impl ExactSizeIterator<Item = &'a vk::DescriptorSetLayout>,
        sets: &mut impl Extend<vk::DescriptorSet>,
    ) -> Result<(), gpu_descriptor::DeviceAllocationError> {
        let result = unsafe {
            self.raw.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*pool)
                    .set_layouts(
                        &smallvec::SmallVec::<[vk::DescriptorSetLayout; 32]>::from_iter(
                            layouts.cloned(),
                        ),
                    )
                    .build(),
            )
        };

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
        let result = unsafe {
            self.raw.free_descriptor_sets(
                *pool,
                &smallvec::SmallVec::<[vk::DescriptorSet; 32]>::from_iter(sets),
            )
        };
        match result {
            Ok(()) => {}
            Err(err) => log::error!("free_descriptor_sets: {:?}", err),
        }
    }
}

struct CompiledStage {
    create_info: vk::PipelineShaderStageCreateInfo,
    _entry_point: CString,
    temp_raw_module: Option<vk::ShaderModule>,
}

impl super::Device {
    pub(super) unsafe fn create_swapchain(
        &self,
        surface: &mut super::Surface,
        config: &crate::SurfaceConfiguration,
        provided_old_swapchain: Option<super::Swapchain>,
    ) -> Result<super::Swapchain, crate::SurfaceError> {
        profiling::scope!("Device::create_swapchain");
        let functor = khr::Swapchain::new(&surface.instance.raw, &self.shared.raw);

        let old_swapchain = match provided_old_swapchain {
            Some(osc) => osc.raw,
            None => vk::SwapchainKHR::null(),
        };

        let color_space = if config.format == wgt::TextureFormat::Rgba16Float {
            // Enable wide color gamut mode
            // Vulkan swapchain for Android only supports DISPLAY_P3_NONLINEAR_EXT and EXTENDED_SRGB_LINEAR_EXT
            vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT
        } else {
            vk::ColorSpaceKHR::SRGB_NONLINEAR
        };

        let original_format = self.shared.private_caps.map_texture_format(config.format);
        let mut raw_flags = vk::SwapchainCreateFlagsKHR::empty();
        let mut raw_view_formats: Vec<vk::Format> = vec![];
        let mut wgt_view_formats = vec![];
        if !config.view_formats.is_empty() {
            raw_flags |= vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT;
            raw_view_formats = config
                .view_formats
                .iter()
                .map(|f| self.shared.private_caps.map_texture_format(*f))
                .collect();
            raw_view_formats.push(original_format);

            wgt_view_formats = config.view_formats.clone();
            wgt_view_formats.push(config.format);
        }

        let mut info = vk::SwapchainCreateInfoKHR::builder()
            .flags(raw_flags)
            .surface(surface.raw)
            .min_image_count(config.swap_chain_size)
            .image_format(original_format)
            .image_color_space(color_space)
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

        let mut format_list_info = vk::ImageFormatListCreateInfo::builder();
        if !raw_view_formats.is_empty() {
            format_list_info = format_list_info.view_formats(&raw_view_formats);
            info = info.push_next(&mut format_list_info);
        }

        let result = {
            profiling::scope!("vkCreateSwapchainKHR");
            unsafe { functor.create_swapchain(&info, None) }
        };

        // doing this before bailing out with error
        if old_swapchain != vk::SwapchainKHR::null() {
            unsafe { functor.destroy_swapchain(old_swapchain, None) }
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

        let images =
            unsafe { functor.get_swapchain_images(raw) }.map_err(crate::DeviceError::from)?;

        let vk_info = vk::FenceCreateInfo::builder().build();
        let fence = unsafe { self.shared.raw.create_fence(&vk_info, None) }
            .map_err(crate::DeviceError::from)?;

        Ok(super::Swapchain {
            raw,
            raw_flags,
            functor,
            device: Arc::clone(&self.shared),
            fence,
            images,
            config: config.clone(),
            view_formats: wgt_view_formats,
        })
    }

    /// # Safety
    ///
    /// - `vk_image` must be created respecting `desc`
    /// - If `drop_guard` is `Some`, the application must manually destroy the image handle. This
    ///   can be done inside the `Drop` impl of `drop_guard`.
    /// - If the `ImageCreateFlags` does not contain `MUTABLE_FORMAT`, the `view_formats` of `desc` must be empty.
    pub unsafe fn texture_from_raw(
        vk_image: vk::Image,
        desc: &crate::TextureDescriptor,
        drop_guard: Option<crate::DropGuard>,
    ) -> super::Texture {
        let mut raw_flags = vk::ImageCreateFlags::empty();
        let mut view_formats = vec![];
        for tf in desc.view_formats.iter() {
            if *tf == desc.format {
                continue;
            }
            view_formats.push(*tf);
        }
        if !view_formats.is_empty() {
            raw_flags |=
                vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE;
            view_formats.push(desc.format)
        }

        super::Texture {
            raw: vk_image,
            drop_guard,
            block: None,
            usage: desc.usage,
            format: desc.format,
            raw_flags: vk::ImageCreateFlags::empty(),
            copy_size: desc.copy_extent(),
            view_formats,
        }
    }

    fn create_shader_module_impl(
        &self,
        spv: &[u32],
    ) -> Result<vk::ShaderModule, crate::DeviceError> {
        let vk_info = vk::ShaderModuleCreateInfo::builder()
            .flags(vk::ShaderModuleCreateFlags::empty())
            .code(spv);

        let raw = unsafe {
            profiling::scope!("vkCreateShaderModule");
            self.shared.raw.create_shader_module(&vk_info, None)?
        };
        Ok(raw)
    }

    fn compile_stage(
        &self,
        stage: &crate::ProgrammableStage<super::Api>,
        naga_stage: naga::ShaderStage,
        binding_map: &naga::back::spv::BindingMap,
    ) -> Result<CompiledStage, crate::PipelineError> {
        let stage_flags = crate::auxil::map_naga_stage(naga_stage);
        let vk_module = match *stage.module {
            super::ShaderModule::Raw(raw) => raw,
            super::ShaderModule::Intermediate {
                ref naga_shader,
                runtime_checks,
            } => {
                let pipeline_options = naga::back::spv::PipelineOptions {
                    entry_point: stage.entry_point.to_string(),
                    shader_stage: naga_stage,
                };
                let needs_temp_options = !runtime_checks || !binding_map.is_empty();
                let mut temp_options;
                let options = if needs_temp_options {
                    temp_options = self.naga_options.clone();
                    if !runtime_checks {
                        temp_options.bounds_check_policies = naga::proc::BoundsCheckPolicies {
                            index: naga::proc::BoundsCheckPolicy::Unchecked,
                            buffer: naga::proc::BoundsCheckPolicy::Unchecked,
                            image: naga::proc::BoundsCheckPolicy::Unchecked,
                            binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
                        };
                    }
                    if !binding_map.is_empty() {
                        temp_options.binding_map = binding_map.clone();
                    }
                    &temp_options
                } else {
                    &self.naga_options
                };
                let spv = {
                    profiling::scope!("naga::spv::write_vec");
                    naga::back::spv::write_vec(
                        &naga_shader.module,
                        &naga_shader.info,
                        options,
                        Some(&pipeline_options),
                    )
                }
                .map_err(|e| crate::PipelineError::Linkage(stage_flags, format!("{e}")))?;
                self.create_shader_module_impl(&spv)?
            }
        };

        let entry_point = CString::new(stage.entry_point).unwrap();
        let create_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(conv::map_shader_stage(stage_flags))
            .module(vk_module)
            .name(&entry_point)
            .build();

        Ok(CompiledStage {
            create_info,
            _entry_point: entry_point,
            temp_raw_module: match *stage.module {
                super::ShaderModule::Raw(_) => None,
                super::ShaderModule::Intermediate { .. } => Some(vk_module),
            },
        })
    }

    /// Returns the queue family index of the device's internal queue.
    ///
    /// This is useful for constructing memory barriers needed for queue family ownership transfer when
    /// external memory is involved (from/to `VK_QUEUE_FAMILY_EXTERNAL_KHR` and `VK_QUEUE_FAMILY_FOREIGN_EXT`
    /// for example).
    pub fn queue_family_index(&self) -> u32 {
        self.shared.family_index
    }

    pub fn queue_index(&self) -> u32 {
        self.shared.queue_index
    }

    pub fn raw_device(&self) -> &ash::Device {
        &self.shared.raw
    }

    pub fn raw_physical_device(&self) -> ash::vk::PhysicalDevice {
        self.shared.physical_device
    }

    pub fn raw_queue(&self) -> ash::vk::Queue {
        self.shared.raw_queue
    }

    pub fn enabled_device_extensions(&self) -> &[&'static CStr] {
        &self.shared.enabled_extensions
    }

    pub fn shared_instance(&self) -> &super::InstanceShared {
        &self.shared.instance
    }
}

impl crate::Device<super::Api> for super::Device {
    unsafe fn exit(self, queue: super::Queue) {
        unsafe { self.mem_allocator.into_inner().cleanup(&*self.shared) };
        unsafe { self.desc_allocator.into_inner().cleanup(&*self.shared) };
        for &sem in queue.relay_semaphores.iter() {
            unsafe { self.shared.raw.destroy_semaphore(sem, None) };
        }
        unsafe { self.shared.free_resources() };
    }

    unsafe fn create_buffer(
        &self,
        desc: &crate::BufferDescriptor,
    ) -> Result<super::Buffer, crate::DeviceError> {
        let vk_info = vk::BufferCreateInfo::builder()
            .size(desc.size)
            .usage(conv::map_buffer_usage(desc.usage))
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let raw = unsafe { self.shared.raw.create_buffer(&vk_info, None)? };
        let req = unsafe { self.shared.raw.get_buffer_memory_requirements(raw) };

        let mut alloc_usage = if desc
            .usage
            .intersects(crate::BufferUses::MAP_READ | crate::BufferUses::MAP_WRITE)
        {
            let mut flags = gpu_alloc::UsageFlags::HOST_ACCESS;
            //TODO: find a way to use `crate::MemoryFlags::PREFER_COHERENT`
            flags.set(
                gpu_alloc::UsageFlags::DOWNLOAD,
                desc.usage.contains(crate::BufferUses::MAP_READ),
            );
            flags.set(
                gpu_alloc::UsageFlags::UPLOAD,
                desc.usage.contains(crate::BufferUses::MAP_WRITE),
            );
            flags
        } else {
            gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS
        };
        alloc_usage.set(
            gpu_alloc::UsageFlags::TRANSIENT,
            desc.memory_flags.contains(crate::MemoryFlags::TRANSIENT),
        );

        let block = unsafe {
            self.mem_allocator.lock().alloc(
                &*self.shared,
                gpu_alloc::Request {
                    size: req.size,
                    align_mask: req.alignment - 1,
                    usage: alloc_usage,
                    memory_types: req.memory_type_bits & self.valid_ash_memory_types,
                },
            )?
        };

        unsafe {
            self.shared
                .raw
                .bind_buffer_memory(raw, *block.memory(), block.offset())?
        };

        if let Some(label) = desc.label {
            unsafe {
                self.shared
                    .set_object_name(vk::ObjectType::BUFFER, raw, label)
            };
        }

        Ok(super::Buffer {
            raw,
            block: Mutex::new(block),
        })
    }
    unsafe fn destroy_buffer(&self, buffer: super::Buffer) {
        unsafe { self.shared.raw.destroy_buffer(buffer.raw, None) };
        unsafe {
            self.mem_allocator
                .lock()
                .dealloc(&*self.shared, buffer.block.into_inner())
        };
    }

    unsafe fn map_buffer(
        &self,
        buffer: &super::Buffer,
        range: crate::MemoryRange,
    ) -> Result<crate::BufferMapping, crate::DeviceError> {
        let size = range.end - range.start;
        let mut block = buffer.block.lock();
        let ptr = unsafe { block.map(&*self.shared, range.start, size as usize)? };
        let is_coherent = block
            .props()
            .contains(gpu_alloc::MemoryPropertyFlags::HOST_COHERENT);
        Ok(crate::BufferMapping { ptr, is_coherent })
    }
    unsafe fn unmap_buffer(&self, buffer: &super::Buffer) -> Result<(), crate::DeviceError> {
        unsafe { buffer.block.lock().unmap(&*self.shared) };
        Ok(())
    }

    unsafe fn flush_mapped_ranges<I>(&self, buffer: &super::Buffer, ranges: I)
    where
        I: Iterator<Item = crate::MemoryRange>,
    {
        let vk_ranges = self.shared.make_memory_ranges(buffer, ranges);

        unsafe {
            self.shared
                .raw
                .flush_mapped_memory_ranges(
                    &smallvec::SmallVec::<[vk::MappedMemoryRange; 32]>::from_iter(vk_ranges),
                )
        }
        .unwrap();
    }
    unsafe fn invalidate_mapped_ranges<I>(&self, buffer: &super::Buffer, ranges: I)
    where
        I: Iterator<Item = crate::MemoryRange>,
    {
        let vk_ranges = self.shared.make_memory_ranges(buffer, ranges);

        unsafe {
            self.shared
                .raw
                .invalidate_mapped_memory_ranges(
                    &smallvec::SmallVec::<[vk::MappedMemoryRange; 32]>::from_iter(vk_ranges),
                )
        }
        .unwrap();
    }

    unsafe fn create_texture(
        &self,
        desc: &crate::TextureDescriptor,
    ) -> Result<super::Texture, crate::DeviceError> {
        let copy_size = desc.copy_extent();

        let mut raw_flags = vk::ImageCreateFlags::empty();
        if desc.is_cube_compatible() {
            raw_flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
        }

        let original_format = self.shared.private_caps.map_texture_format(desc.format);
        let mut vk_view_formats = vec![];
        let mut wgt_view_formats = vec![];
        if !desc.view_formats.is_empty() {
            raw_flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
            wgt_view_formats = desc.view_formats.clone();
            wgt_view_formats.push(desc.format);

            if self.shared_instance().driver_api_version >= vk::API_VERSION_1_2
                || self
                    .enabled_device_extensions()
                    .contains(&vk::KhrImageFormatListFn::name())
            {
                vk_view_formats = desc
                    .view_formats
                    .iter()
                    .map(|f| self.shared.private_caps.map_texture_format(*f))
                    .collect();
                vk_view_formats.push(original_format)
            }
        }

        let mut vk_info = vk::ImageCreateInfo::builder()
            .flags(raw_flags)
            .image_type(conv::map_texture_dimension(desc.dimension))
            .format(original_format)
            .extent(conv::map_copy_extent(&copy_size))
            .mip_levels(desc.mip_level_count)
            .array_layers(desc.array_layer_count())
            .samples(vk::SampleCountFlags::from_raw(desc.sample_count))
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(conv::map_texture_usage(desc.usage))
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let mut format_list_info = vk::ImageFormatListCreateInfo::builder();
        if !vk_view_formats.is_empty() {
            format_list_info = format_list_info.view_formats(&vk_view_formats);
            vk_info = vk_info.push_next(&mut format_list_info);
        }

        let raw = unsafe { self.shared.raw.create_image(&vk_info, None)? };
        let req = unsafe { self.shared.raw.get_image_memory_requirements(raw) };

        let block = unsafe {
            self.mem_allocator.lock().alloc(
                &*self.shared,
                gpu_alloc::Request {
                    size: req.size,
                    align_mask: req.alignment - 1,
                    usage: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
                    memory_types: req.memory_type_bits & self.valid_ash_memory_types,
                },
            )?
        };

        unsafe {
            self.shared
                .raw
                .bind_image_memory(raw, *block.memory(), block.offset())?
        };

        if let Some(label) = desc.label {
            unsafe {
                self.shared
                    .set_object_name(vk::ObjectType::IMAGE, raw, label)
            };
        }

        Ok(super::Texture {
            raw,
            drop_guard: None,
            block: Some(block),
            usage: desc.usage,
            format: desc.format,
            raw_flags,
            copy_size,
            view_formats: wgt_view_formats,
        })
    }
    unsafe fn destroy_texture(&self, texture: super::Texture) {
        if texture.drop_guard.is_none() {
            unsafe { self.shared.raw.destroy_image(texture.raw, None) };
        }
        if let Some(block) = texture.block {
            unsafe { self.mem_allocator.lock().dealloc(&*self.shared, block) };
        }
    }

    unsafe fn create_texture_view(
        &self,
        texture: &super::Texture,
        desc: &crate::TextureViewDescriptor,
    ) -> Result<super::TextureView, crate::DeviceError> {
        let subresource_range = conv::map_subresource_range(&desc.range, desc.format);
        let mut vk_info = vk::ImageViewCreateInfo::builder()
            .flags(vk::ImageViewCreateFlags::empty())
            .image(texture.raw)
            .view_type(conv::map_view_dimension(desc.dimension))
            .format(self.shared.private_caps.map_texture_format(desc.format))
            .subresource_range(subresource_range);
        let layers =
            NonZeroU32::new(subresource_range.layer_count).expect("Unexpected zero layer count");

        let mut image_view_info;
        let view_usage = if self.shared.private_caps.image_view_usage && !desc.usage.is_empty() {
            image_view_info = vk::ImageViewUsageCreateInfo::builder()
                .usage(conv::map_texture_usage(desc.usage))
                .build();
            vk_info = vk_info.push_next(&mut image_view_info);
            desc.usage
        } else {
            texture.usage
        };

        let raw = unsafe { self.shared.raw.create_image_view(&vk_info, None) }?;

        if let Some(label) = desc.label {
            unsafe {
                self.shared
                    .set_object_name(vk::ObjectType::IMAGE_VIEW, raw, label)
            };
        }

        let attachment = super::FramebufferAttachment {
            raw: if self.shared.private_caps.imageless_framebuffers {
                vk::ImageView::null()
            } else {
                raw
            },
            raw_image_flags: texture.raw_flags,
            view_usage,
            view_format: desc.format,
            raw_view_formats: texture
                .view_formats
                .iter()
                .map(|tf| self.shared.private_caps.map_texture_format(*tf))
                .collect(),
        };

        Ok(super::TextureView {
            raw,
            layers,
            attachment,
        })
    }
    unsafe fn destroy_texture_view(&self, view: super::TextureView) {
        if !self.shared.private_caps.imageless_framebuffers {
            let mut fbuf_lock = self.shared.framebuffers.lock();
            for (key, &raw_fbuf) in fbuf_lock.iter() {
                if key.attachments.iter().any(|at| at.raw == view.raw) {
                    unsafe { self.shared.raw.destroy_framebuffer(raw_fbuf, None) };
                }
            }
            fbuf_lock.retain(|key, _| !key.attachments.iter().any(|at| at.raw == view.raw));
        }
        unsafe { self.shared.raw.destroy_image_view(view.raw, None) };
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
            .address_mode_w(conv::map_address_mode(desc.address_modes[2]))
            .min_lod(desc.lod_clamp.start)
            .max_lod(desc.lod_clamp.end);

        if let Some(fun) = desc.compare {
            vk_info = vk_info
                .compare_enable(true)
                .compare_op(conv::map_comparison(fun));
        }

        if desc.anisotropy_clamp != 1 {
            // We only enable anisotropy if it is supported, and wgpu-hal interface guarentees
            // the clamp is in the range [1, 16] which is always supported if anisotropy is.
            vk_info = vk_info
                .anisotropy_enable(true)
                .max_anisotropy(desc.anisotropy_clamp as f32);
        }

        if let Some(color) = desc.border_color {
            vk_info = vk_info.border_color(conv::map_border_color(color));
        }

        let raw = unsafe { self.shared.raw.create_sampler(&vk_info, None)? };

        if let Some(label) = desc.label {
            unsafe {
                self.shared
                    .set_object_name(vk::ObjectType::SAMPLER, raw, label)
            };
        }

        Ok(super::Sampler { raw })
    }
    unsafe fn destroy_sampler(&self, sampler: super::Sampler) {
        unsafe { self.shared.raw.destroy_sampler(sampler.raw, None) };
    }

    unsafe fn create_command_encoder(
        &self,
        desc: &crate::CommandEncoderDescriptor<super::Api>,
    ) -> Result<super::CommandEncoder, crate::DeviceError> {
        let vk_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(desc.queue.family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .build();
        let raw = unsafe { self.shared.raw.create_command_pool(&vk_info, None)? };

        Ok(super::CommandEncoder {
            raw,
            device: Arc::clone(&self.shared),
            active: vk::CommandBuffer::null(),
            bind_point: vk::PipelineBindPoint::default(),
            temp: super::Temp::default(),
            free: Vec::new(),
            discarded: Vec::new(),
            rpass_debug_marker_active: false,
        })
    }
    unsafe fn destroy_command_encoder(&self, cmd_encoder: super::CommandEncoder) {
        unsafe {
            if !cmd_encoder.free.is_empty() {
                self.shared
                    .raw
                    .free_command_buffers(cmd_encoder.raw, &cmd_encoder.free)
            }
            if !cmd_encoder.discarded.is_empty() {
                self.shared
                    .raw
                    .free_command_buffers(cmd_encoder.raw, &cmd_encoder.discarded)
            }
            self.shared.raw.destroy_command_pool(cmd_encoder.raw, None);
        }
    }

    unsafe fn create_bind_group_layout(
        &self,
        desc: &crate::BindGroupLayoutDescriptor,
    ) -> Result<super::BindGroupLayout, crate::DeviceError> {
        let mut desc_count = gpu_descriptor::DescriptorTotalCount::default();
        let mut types = Vec::new();
        for entry in desc.entries {
            let count = entry.count.map_or(1, |c| c.get());
            if entry.binding as usize >= types.len() {
                types.resize(
                    entry.binding as usize + 1,
                    (vk::DescriptorType::INPUT_ATTACHMENT, 0),
                );
            }
            types[entry.binding as usize] = (
                conv::map_binding_type(entry.ty),
                entry.count.map_or(1, |c| c.get()),
            );

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

        //Note: not bothering with on stack array here as it's low frequency
        let vk_bindings = desc
            .entries
            .iter()
            .map(|entry| vk::DescriptorSetLayoutBinding {
                binding: entry.binding,
                descriptor_type: types[entry.binding as usize].0,
                descriptor_count: types[entry.binding as usize].1,
                stage_flags: conv::map_shader_stage(entry.visibility),
                p_immutable_samplers: ptr::null(),
            })
            .collect::<Vec<_>>();

        let vk_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&vk_bindings);

        let binding_arrays = desc
            .entries
            .iter()
            .enumerate()
            .filter_map(|(idx, entry)| entry.count.map(|count| (idx as u32, count)))
            .collect();

        let mut binding_flag_info;
        let binding_flag_vec;

        let partially_bound = desc
            .flags
            .contains(crate::BindGroupLayoutFlags::PARTIALLY_BOUND);

        let vk_info = if partially_bound {
            binding_flag_vec = desc
                .entries
                .iter()
                .map(|entry| {
                    let mut flags = vk::DescriptorBindingFlags::empty();

                    if partially_bound && entry.count.is_some() {
                        flags |= vk::DescriptorBindingFlags::PARTIALLY_BOUND;
                    }

                    flags
                })
                .collect::<Vec<_>>();

            binding_flag_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                .binding_flags(&binding_flag_vec);

            vk_info.push_next(&mut binding_flag_info)
        } else {
            vk_info
        };

        let raw = unsafe {
            self.shared
                .raw
                .create_descriptor_set_layout(&vk_info, None)?
        };

        if let Some(label) = desc.label {
            unsafe {
                self.shared
                    .set_object_name(vk::ObjectType::DESCRIPTOR_SET_LAYOUT, raw, label)
            };
        }

        Ok(super::BindGroupLayout {
            raw,
            desc_count,
            types: types.into_boxed_slice(),
            binding_arrays,
        })
    }
    unsafe fn destroy_bind_group_layout(&self, bg_layout: super::BindGroupLayout) {
        unsafe {
            self.shared
                .raw
                .destroy_descriptor_set_layout(bg_layout.raw, None)
        };
    }

    unsafe fn create_pipeline_layout(
        &self,
        desc: &crate::PipelineLayoutDescriptor<super::Api>,
    ) -> Result<super::PipelineLayout, crate::DeviceError> {
        //Note: not bothering with on stack array here as it's low frequency
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

        let raw = {
            profiling::scope!("vkCreatePipelineLayout");
            unsafe { self.shared.raw.create_pipeline_layout(&vk_info, None)? }
        };

        if let Some(label) = desc.label {
            unsafe {
                self.shared
                    .set_object_name(vk::ObjectType::PIPELINE_LAYOUT, raw, label)
            };
        }

        let mut binding_arrays = BTreeMap::new();
        for (group, &layout) in desc.bind_group_layouts.iter().enumerate() {
            for &(binding, binding_array_size) in &layout.binding_arrays {
                binding_arrays.insert(
                    naga::ResourceBinding {
                        group: group as u32,
                        binding,
                    },
                    naga::back::spv::BindingInfo {
                        binding_array_size: Some(binding_array_size.get()),
                    },
                );
            }
        }

        Ok(super::PipelineLayout {
            raw,
            binding_arrays,
        })
    }
    unsafe fn destroy_pipeline_layout(&self, pipeline_layout: super::PipelineLayout) {
        unsafe {
            self.shared
                .raw
                .destroy_pipeline_layout(pipeline_layout.raw, None)
        };
    }

    unsafe fn create_bind_group(
        &self,
        desc: &crate::BindGroupDescriptor<super::Api>,
    ) -> Result<super::BindGroup, crate::DeviceError> {
        let mut vk_sets = unsafe {
            self.desc_allocator.lock().allocate(
                &*self.shared,
                &desc.layout.raw,
                gpu_descriptor::DescriptorSetLayoutCreateFlags::empty(),
                &desc.layout.desc_count,
                1,
            )?
        };

        let set = vk_sets.pop().unwrap();
        if let Some(label) = desc.label {
            unsafe {
                self.shared
                    .set_object_name(vk::ObjectType::DESCRIPTOR_SET, *set.raw(), label)
            };
        }

        let mut writes = Vec::with_capacity(desc.entries.len());
        let mut buffer_infos = Vec::with_capacity(desc.buffers.len());
        let mut sampler_infos = Vec::with_capacity(desc.samplers.len());
        let mut image_infos = Vec::with_capacity(desc.textures.len());
        for entry in desc.entries {
            let (ty, size) = desc.layout.types[entry.binding as usize];
            if size == 0 {
                continue; // empty slot
            }
            let mut write = vk::WriteDescriptorSet::builder()
                .dst_set(*set.raw())
                .dst_binding(entry.binding)
                .descriptor_type(ty);
            write = match ty {
                vk::DescriptorType::SAMPLER => {
                    let index = sampler_infos.len();
                    let start = entry.resource_index;
                    let end = start + entry.count;
                    sampler_infos.extend(desc.samplers[start as usize..end as usize].iter().map(
                        |binding| {
                            vk::DescriptorImageInfo::builder()
                                .sampler(binding.raw)
                                .build()
                        },
                    ));
                    write.image_info(&sampler_infos[index..])
                }
                vk::DescriptorType::SAMPLED_IMAGE | vk::DescriptorType::STORAGE_IMAGE => {
                    let index = image_infos.len();
                    let start = entry.resource_index;
                    let end = start + entry.count;
                    image_infos.extend(desc.textures[start as usize..end as usize].iter().map(
                        |binding| {
                            let layout = conv::derive_image_layout(
                                binding.usage,
                                binding.view.attachment.view_format,
                            );
                            vk::DescriptorImageInfo::builder()
                                .image_view(binding.view.raw)
                                .image_layout(layout)
                                .build()
                        },
                    ));
                    write.image_info(&image_infos[index..])
                }
                vk::DescriptorType::UNIFORM_BUFFER
                | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                | vk::DescriptorType::STORAGE_BUFFER
                | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                    let index = buffer_infos.len();
                    let start = entry.resource_index;
                    let end = start + entry.count;
                    buffer_infos.extend(desc.buffers[start as usize..end as usize].iter().map(
                        |binding| {
                            vk::DescriptorBufferInfo::builder()
                                .buffer(binding.buffer.raw)
                                .offset(binding.offset)
                                .range(binding.size.map_or(vk::WHOLE_SIZE, wgt::BufferSize::get))
                                .build()
                        },
                    ));
                    write.buffer_info(&buffer_infos[index..])
                }
                _ => unreachable!(),
            };
            writes.push(write.build());
        }

        unsafe { self.shared.raw.update_descriptor_sets(&writes, &[]) };
        Ok(super::BindGroup { set })
    }
    unsafe fn destroy_bind_group(&self, group: super::BindGroup) {
        unsafe {
            self.desc_allocator
                .lock()
                .free(&*self.shared, Some(group.set))
        };
    }

    unsafe fn create_shader_module(
        &self,
        desc: &crate::ShaderModuleDescriptor,
        shader: crate::ShaderInput,
    ) -> Result<super::ShaderModule, crate::ShaderError> {
        let spv = match shader {
            crate::ShaderInput::Naga(naga_shader) => {
                if self
                    .shared
                    .workarounds
                    .contains(super::Workarounds::SEPARATE_ENTRY_POINTS)
                {
                    return Ok(super::ShaderModule::Intermediate {
                        naga_shader,
                        runtime_checks: desc.runtime_checks,
                    });
                }
                let mut naga_options = self.naga_options.clone();
                if !desc.runtime_checks {
                    naga_options.bounds_check_policies = naga::proc::BoundsCheckPolicies {
                        index: naga::proc::BoundsCheckPolicy::Unchecked,
                        buffer: naga::proc::BoundsCheckPolicy::Unchecked,
                        image: naga::proc::BoundsCheckPolicy::Unchecked,
                        binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
                    };
                }
                Cow::Owned(
                    naga::back::spv::write_vec(
                        &naga_shader.module,
                        &naga_shader.info,
                        &naga_options,
                        None,
                    )
                    .map_err(|e| crate::ShaderError::Compilation(format!("{e}")))?,
                )
            }
            crate::ShaderInput::SpirV(spv) => Cow::Borrowed(spv),
        };

        let raw = self.create_shader_module_impl(&spv)?;

        if let Some(label) = desc.label {
            unsafe {
                self.shared
                    .set_object_name(vk::ObjectType::SHADER_MODULE, raw, label)
            };
        }

        Ok(super::ShaderModule::Raw(raw))
    }
    unsafe fn destroy_shader_module(&self, module: super::ShaderModule) {
        match module {
            super::ShaderModule::Raw(raw) => {
                unsafe { self.shared.raw.destroy_shader_module(raw, None) };
            }
            super::ShaderModule::Intermediate { .. } => {}
        }
    }

    unsafe fn create_render_pipeline(
        &self,
        desc: &crate::RenderPipelineDescriptor<super::Api>,
    ) -> Result<super::RenderPipeline, crate::PipelineError> {
        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::BLEND_CONSTANTS,
            vk::DynamicState::STENCIL_REFERENCE,
        ];
        let mut compatible_rp_key = super::RenderPassKey {
            sample_count: desc.multisample.count,
            multiview: desc.multiview,
            ..Default::default()
        };
        let mut stages = ArrayVec::<_, 2>::new();
        let mut vertex_buffers = Vec::with_capacity(desc.vertex_buffers.len());
        let mut vertex_attributes = Vec::new();

        for (i, vb) in desc.vertex_buffers.iter().enumerate() {
            vertex_buffers.push(vk::VertexInputBindingDescription {
                binding: i as u32,
                stride: vb.array_stride as u32,
                input_rate: match vb.step_mode {
                    wgt::VertexStepMode::Vertex => vk::VertexInputRate::VERTEX,
                    wgt::VertexStepMode::Instance => vk::VertexInputRate::INSTANCE,
                },
            });
            for at in vb.attributes {
                vertex_attributes.push(vk::VertexInputAttributeDescription {
                    location: at.shader_location,
                    binding: i as u32,
                    format: conv::map_vertex_format(at.format),
                    offset: at.offset as u32,
                });
            }
        }

        let vk_vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&vertex_buffers)
            .vertex_attribute_descriptions(&vertex_attributes)
            .build();

        let vk_input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(conv::map_topology(desc.primitive.topology))
            .primitive_restart_enable(desc.primitive.strip_index_format.is_some())
            .build();

        let compiled_vs = self.compile_stage(
            &desc.vertex_stage,
            naga::ShaderStage::Vertex,
            &desc.layout.binding_arrays,
        )?;
        stages.push(compiled_vs.create_info);
        let compiled_fs = match desc.fragment_stage {
            Some(ref stage) => {
                let compiled = self.compile_stage(
                    stage,
                    naga::ShaderStage::Fragment,
                    &desc.layout.binding_arrays,
                )?;
                stages.push(compiled.create_info);
                Some(compiled)
            }
            None => None,
        };

        let mut vk_rasterization = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(conv::map_polygon_mode(desc.primitive.polygon_mode))
            .front_face(conv::map_front_face(desc.primitive.front_face))
            .line_width(1.0);
        if let Some(face) = desc.primitive.cull_mode {
            vk_rasterization = vk_rasterization.cull_mode(conv::map_cull_face(face))
        }
        let mut vk_rasterization_conservative_state =
            vk::PipelineRasterizationConservativeStateCreateInfoEXT::builder()
                .conservative_rasterization_mode(vk::ConservativeRasterizationModeEXT::OVERESTIMATE)
                .build();
        if desc.primitive.conservative {
            vk_rasterization = vk_rasterization.push_next(&mut vk_rasterization_conservative_state);
        }
        let mut vk_depth_clip_state =
            vk::PipelineRasterizationDepthClipStateCreateInfoEXT::builder()
                .depth_clip_enable(false)
                .build();
        if desc.primitive.unclipped_depth {
            vk_rasterization = vk_rasterization.push_next(&mut vk_depth_clip_state);
        }

        let mut vk_depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder();
        if let Some(ref ds) = desc.depth_stencil {
            let vk_format = self.shared.private_caps.map_texture_format(ds.format);
            let vk_layout = if ds.is_read_only(desc.primitive.cull_mode) {
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
            } else {
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            };
            compatible_rp_key.depth_stencil = Some(super::DepthStencilAttachmentKey {
                base: super::AttachmentKey::compatible(vk_format, vk_layout),
                stencil_ops: crate::AttachmentOps::all(),
            });

            if ds.is_depth_enabled() {
                vk_depth_stencil = vk_depth_stencil
                    .depth_test_enable(true)
                    .depth_write_enable(ds.depth_write_enabled)
                    .depth_compare_op(conv::map_comparison(ds.depth_compare));
            }
            if ds.stencil.is_enabled() {
                let s = &ds.stencil;
                let front = conv::map_stencil_face(&s.front, s.read_mask, s.write_mask);
                let back = conv::map_stencil_face(&s.back, s.read_mask, s.write_mask);
                vk_depth_stencil = vk_depth_stencil
                    .stencil_test_enable(true)
                    .front(front)
                    .back(back);
            }

            if ds.bias.is_enabled() {
                vk_rasterization = vk_rasterization
                    .depth_bias_enable(true)
                    .depth_bias_constant_factor(ds.bias.constant as f32)
                    .depth_bias_clamp(ds.bias.clamp)
                    .depth_bias_slope_factor(ds.bias.slope_scale);
            }
        }

        let vk_viewport = vk::PipelineViewportStateCreateInfo::builder()
            .flags(vk::PipelineViewportStateCreateFlags::empty())
            .scissor_count(1)
            .viewport_count(1)
            .build();

        let vk_sample_mask = [
            desc.multisample.mask as u32,
            (desc.multisample.mask >> 32) as u32,
        ];
        let vk_multisample = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::from_raw(desc.multisample.count))
            .alpha_to_coverage_enable(desc.multisample.alpha_to_coverage_enabled)
            .sample_mask(&vk_sample_mask)
            .build();

        let mut vk_attachments = Vec::with_capacity(desc.color_targets.len());
        for cat in desc.color_targets {
            let (key, attarchment) = if let Some(cat) = cat.as_ref() {
                let mut vk_attachment = vk::PipelineColorBlendAttachmentState::builder()
                    .color_write_mask(vk::ColorComponentFlags::from_raw(cat.write_mask.bits()));
                if let Some(ref blend) = cat.blend {
                    let (color_op, color_src, color_dst) = conv::map_blend_component(&blend.color);
                    let (alpha_op, alpha_src, alpha_dst) = conv::map_blend_component(&blend.alpha);
                    vk_attachment = vk_attachment
                        .blend_enable(true)
                        .color_blend_op(color_op)
                        .src_color_blend_factor(color_src)
                        .dst_color_blend_factor(color_dst)
                        .alpha_blend_op(alpha_op)
                        .src_alpha_blend_factor(alpha_src)
                        .dst_alpha_blend_factor(alpha_dst);
                }

                let vk_format = self.shared.private_caps.map_texture_format(cat.format);
                (
                    Some(super::ColorAttachmentKey {
                        base: super::AttachmentKey::compatible(
                            vk_format,
                            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        ),
                        resolve: None,
                    }),
                    vk_attachment.build(),
                )
            } else {
                (None, vk::PipelineColorBlendAttachmentState::default())
            };

            compatible_rp_key.colors.push(key);
            vk_attachments.push(attarchment);
        }

        let vk_color_blend = vk::PipelineColorBlendStateCreateInfo::builder()
            .attachments(&vk_attachments)
            .build();

        let vk_dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&dynamic_states)
            .build();

        let raw_pass = self
            .shared
            .make_render_pass(compatible_rp_key)
            .map_err(crate::DeviceError::from)?;

        let vk_infos = [{
            vk::GraphicsPipelineCreateInfo::builder()
                .layout(desc.layout.raw)
                .stages(&stages)
                .vertex_input_state(&vk_vertex_input)
                .input_assembly_state(&vk_input_assembly)
                .rasterization_state(&vk_rasterization)
                .viewport_state(&vk_viewport)
                .multisample_state(&vk_multisample)
                .depth_stencil_state(&vk_depth_stencil)
                .color_blend_state(&vk_color_blend)
                .dynamic_state(&vk_dynamic_state)
                .render_pass(raw_pass)
                .build()
        }];

        let mut raw_vec = {
            profiling::scope!("vkCreateGraphicsPipelines");
            unsafe {
                self.shared
                    .raw
                    .create_graphics_pipelines(vk::PipelineCache::null(), &vk_infos, None)
                    .map_err(|(_, e)| crate::DeviceError::from(e))
            }?
        };

        let raw = raw_vec.pop().unwrap();
        if let Some(label) = desc.label {
            unsafe {
                self.shared
                    .set_object_name(vk::ObjectType::PIPELINE, raw, label)
            };
        }

        if let Some(raw_module) = compiled_vs.temp_raw_module {
            unsafe { self.shared.raw.destroy_shader_module(raw_module, None) };
        }
        if let Some(CompiledStage {
            temp_raw_module: Some(raw_module),
            ..
        }) = compiled_fs
        {
            unsafe { self.shared.raw.destroy_shader_module(raw_module, None) };
        }

        Ok(super::RenderPipeline { raw })
    }
    unsafe fn destroy_render_pipeline(&self, pipeline: super::RenderPipeline) {
        unsafe { self.shared.raw.destroy_pipeline(pipeline.raw, None) };
    }

    unsafe fn create_compute_pipeline(
        &self,
        desc: &crate::ComputePipelineDescriptor<super::Api>,
    ) -> Result<super::ComputePipeline, crate::PipelineError> {
        let compiled = self.compile_stage(
            &desc.stage,
            naga::ShaderStage::Compute,
            &desc.layout.binding_arrays,
        )?;

        let vk_infos = [{
            vk::ComputePipelineCreateInfo::builder()
                .layout(desc.layout.raw)
                .stage(compiled.create_info)
                .build()
        }];

        let mut raw_vec = {
            profiling::scope!("vkCreateComputePipelines");
            unsafe {
                self.shared
                    .raw
                    .create_compute_pipelines(vk::PipelineCache::null(), &vk_infos, None)
                    .map_err(|(_, e)| crate::DeviceError::from(e))
            }?
        };

        let raw = raw_vec.pop().unwrap();
        if let Some(label) = desc.label {
            unsafe {
                self.shared
                    .set_object_name(vk::ObjectType::PIPELINE, raw, label)
            };
        }

        if let Some(raw_module) = compiled.temp_raw_module {
            unsafe { self.shared.raw.destroy_shader_module(raw_module, None) };
        }

        Ok(super::ComputePipeline { raw })
    }
    unsafe fn destroy_compute_pipeline(&self, pipeline: super::ComputePipeline) {
        unsafe { self.shared.raw.destroy_pipeline(pipeline.raw, None) };
    }

    unsafe fn create_query_set(
        &self,
        desc: &wgt::QuerySetDescriptor<crate::Label>,
    ) -> Result<super::QuerySet, crate::DeviceError> {
        let (vk_type, pipeline_statistics) = match desc.ty {
            wgt::QueryType::Occlusion => (
                vk::QueryType::OCCLUSION,
                vk::QueryPipelineStatisticFlags::empty(),
            ),
            wgt::QueryType::PipelineStatistics(statistics) => (
                vk::QueryType::PIPELINE_STATISTICS,
                conv::map_pipeline_statistics(statistics),
            ),
            wgt::QueryType::Timestamp => (
                vk::QueryType::TIMESTAMP,
                vk::QueryPipelineStatisticFlags::empty(),
            ),
        };

        let vk_info = vk::QueryPoolCreateInfo::builder()
            .query_type(vk_type)
            .query_count(desc.count)
            .pipeline_statistics(pipeline_statistics)
            .build();

        let raw = unsafe { self.shared.raw.create_query_pool(&vk_info, None) }?;
        if let Some(label) = desc.label {
            unsafe {
                self.shared
                    .set_object_name(vk::ObjectType::QUERY_POOL, raw, label)
            };
        }

        Ok(super::QuerySet { raw })
    }
    unsafe fn destroy_query_set(&self, set: super::QuerySet) {
        unsafe { self.shared.raw.destroy_query_pool(set.raw, None) };
    }

    unsafe fn create_fence(&self) -> Result<super::Fence, crate::DeviceError> {
        Ok(if self.shared.private_caps.timeline_semaphores {
            let mut sem_type_info =
                vk::SemaphoreTypeCreateInfo::builder().semaphore_type(vk::SemaphoreType::TIMELINE);
            let vk_info = vk::SemaphoreCreateInfo::builder().push_next(&mut sem_type_info);
            let raw = unsafe { self.shared.raw.create_semaphore(&vk_info, None) }?;
            super::Fence::TimelineSemaphore(raw)
        } else {
            super::Fence::FencePool {
                last_completed: 0,
                active: Vec::new(),
                free: Vec::new(),
            }
        })
    }
    unsafe fn destroy_fence(&self, fence: super::Fence) {
        match fence {
            super::Fence::TimelineSemaphore(raw) => {
                unsafe { self.shared.raw.destroy_semaphore(raw, None) };
            }
            super::Fence::FencePool {
                active,
                free,
                last_completed: _,
            } => {
                for (_, raw) in active {
                    unsafe { self.shared.raw.destroy_fence(raw, None) };
                }
                for raw in free {
                    unsafe { self.shared.raw.destroy_fence(raw, None) };
                }
            }
        }
    }
    unsafe fn get_fence_value(
        &self,
        fence: &super::Fence,
    ) -> Result<crate::FenceValue, crate::DeviceError> {
        fence.get_latest(
            &self.shared.raw,
            self.shared.extension_fns.timeline_semaphore.as_ref(),
        )
    }
    unsafe fn wait(
        &self,
        fence: &super::Fence,
        wait_value: crate::FenceValue,
        timeout_ms: u32,
    ) -> Result<bool, crate::DeviceError> {
        let timeout_ns = timeout_ms as u64 * super::MILLIS_TO_NANOS;
        match *fence {
            super::Fence::TimelineSemaphore(raw) => {
                let semaphores = [raw];
                let values = [wait_value];
                let vk_info = vk::SemaphoreWaitInfo::builder()
                    .semaphores(&semaphores)
                    .values(&values);
                let result = match self.shared.extension_fns.timeline_semaphore {
                    Some(super::ExtensionFn::Extension(ref ext)) => unsafe {
                        ext.wait_semaphores(&vk_info, timeout_ns)
                    },
                    Some(super::ExtensionFn::Promoted) => unsafe {
                        self.shared.raw.wait_semaphores(&vk_info, timeout_ns)
                    },
                    None => unreachable!(),
                };
                match result {
                    Ok(()) => Ok(true),
                    Err(vk::Result::TIMEOUT) => Ok(false),
                    Err(other) => Err(other.into()),
                }
            }
            super::Fence::FencePool {
                last_completed,
                ref active,
                free: _,
            } => {
                if wait_value <= last_completed {
                    Ok(true)
                } else {
                    match active.iter().find(|&&(value, _)| value >= wait_value) {
                        Some(&(_, raw)) => {
                            match unsafe {
                                self.shared.raw.wait_for_fences(&[raw], true, timeout_ns)
                            } {
                                Ok(()) => Ok(true),
                                Err(vk::Result::TIMEOUT) => Ok(false),
                                Err(other) => Err(other.into()),
                            }
                        }
                        None => {
                            log::error!("No signals reached value {}", wait_value);
                            Err(crate::DeviceError::Lost)
                        }
                    }
                }
            }
        }
    }

    unsafe fn start_capture(&self) -> bool {
        #[cfg(feature = "renderdoc")]
        {
            // Renderdoc requires us to give us the pointer that vkInstance _points to_.
            let raw_vk_instance =
                ash::vk::Handle::as_raw(self.shared.instance.raw.handle()) as *mut *mut _;
            let raw_vk_instance_dispatch_table = unsafe { *raw_vk_instance };
            unsafe {
                self.render_doc
                    .start_frame_capture(raw_vk_instance_dispatch_table, ptr::null_mut())
            }
        }
        #[cfg(not(feature = "renderdoc"))]
        false
    }
    unsafe fn stop_capture(&self) {
        #[cfg(feature = "renderdoc")]
        {
            // Renderdoc requires us to give us the pointer that vkInstance _points to_.
            let raw_vk_instance =
                ash::vk::Handle::as_raw(self.shared.instance.raw.handle()) as *mut *mut _;
            let raw_vk_instance_dispatch_table = unsafe { *raw_vk_instance };

            unsafe {
                self.render_doc
                    .end_frame_capture(raw_vk_instance_dispatch_table, ptr::null_mut())
            }
        }
    }
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
