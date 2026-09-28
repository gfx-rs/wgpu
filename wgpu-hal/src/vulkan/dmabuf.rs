use alloc::vec::Vec;
use ash::{ext, vk};

/// Importable, single-plane DMA-BUF modifiers for a requested texture format and usage.
#[derive(Clone, Debug)]
pub struct DmabufFormat {
    /// The requested wgpu format; mapping it to a DRM FourCC is the caller's responsibility.
    pub format: wgt::TextureFormat,
    /// Modifiers accepted for this format and usage, possibly without linear sampling support.
    pub modifiers: Vec<DmabufModifier>,
}

/// Driver-reported properties of an importable DRM modifier for the queried image usage.
#[derive(Clone, Copy, Debug)]
pub struct DmabufModifier {
    /// Explicit DRM memory-layout modifier required by the Vulkan importer.
    pub modifier: u64,
    /// Number of memory planes; currently always one in query results.
    pub plane_count: u32,
    /// Format features for this modifier, independent of the requested image usage.
    pub tiling_features: vk::FormatFeatureFlags,
    /// DMA-BUF external-memory features for this modifier and image usage.
    pub external_memory_features: vk::ExternalMemoryFeatureFlags,
}

impl DmabufModifier {
    /// Whether the queried image can import DMA-BUF memory; always true in query results.
    pub fn is_importable(&self) -> bool {
        self.external_memory_features
            .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
    }

    /// Whether the modifier supports sampled images with linear filtering.
    /// This does not check whether sampling was included in the queried usage.
    pub fn supports_linear_sampling(&self) -> bool {
        self.tiling_features.contains(
            vk::FormatFeatureFlags::SAMPLED_IMAGE
                | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
        )
    }
}

impl super::CommandEncoder {
    /// Records a foreign-to-device ownership and layout transition for sampling.
    /// The barrier runs when the command buffer is submitted; this call does not wait.
    /// Returns an error if `VK_EXT_queue_family_foreign` was not enabled.
    /// Higher-level resource-state tracking is not updated by this raw barrier.
    ///
    /// # Safety
    /// Recording must be active outside a render pass. The texture must be a
    /// single-layer/mip RGB DMA-BUF from this device. Its producer must have
    /// finished and released it in GENERAL before this command executes.
    /// External producer completion must be synchronized separately.
    pub unsafe fn acquire_dmabuf_texture(
        &mut self,
        texture: &super::Texture,
    ) -> Result<(), crate::DeviceError> {
        unsafe { self.transfer_dmabuf(texture, true) }
    }

    /// Records a device-to-foreign ownership transition to GENERAL.
    /// The barrier runs when the command buffer is submitted; this call does not wait.
    /// Returns an error if `VK_EXT_queue_family_foreign` was not enabled.
    /// Higher-level resource-state tracking is not updated by this raw barrier.
    ///
    /// # Safety
    /// Recording must be active outside a render pass. The texture must have
    /// been acquired on this device; all sampling must precede this barrier.
    /// Keep the texture alive until submission completes, and do not let the
    /// producer reuse it before completion of this command.
    pub unsafe fn release_dmabuf_texture(
        &mut self,
        texture: &super::Texture,
    ) -> Result<(), crate::DeviceError> {
        unsafe { self.transfer_dmabuf(texture, false) }
    }

    unsafe fn transfer_dmabuf(
        &mut self,
        texture: &super::Texture,
        acquire: bool,
    ) -> Result<(), crate::DeviceError> {
        if !self
            .device
            .enabled_extensions
            .contains(&ext::queue_family_foreign::NAME)
        {
            return Err(crate::DeviceError::Unexpected);
        }
        let barrier = ownership_barrier(texture.raw, self.device.family_index, acquire);
        unsafe {
            self.device.raw.cmd_pipeline_barrier(
                self.active,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
        Ok(())
    }
}

fn ownership_barrier(
    image: vk::Image,
    family: u32,
    acquire: bool,
) -> vk::ImageMemoryBarrier<'static> {
    let (source, destination, old, new, read, write) = if acquire {
        (
            vk::QUEUE_FAMILY_FOREIGN_EXT,
            family,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_READ,
        )
    } else {
        (
            family,
            vk::QUEUE_FAMILY_FOREIGN_EXT,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::empty(),
        )
    };
    vk::ImageMemoryBarrier::default()
        .image(image)
        .src_queue_family_index(source)
        .dst_queue_family_index(destination)
        .old_layout(old)
        .new_layout(new)
        .src_access_mask(read)
        .dst_access_mask(write)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
        )
}

pub(super) unsafe fn query_modifiers(
    properties: &ash::khr::get_physical_device_properties2::Instance,
    physical_device: vk::PhysicalDevice,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    modifiers: &mut Vec<vk::DrmFormatModifierPropertiesEXT>,
) -> Result<Vec<DmabufModifier>, crate::DeviceError> {
    let mut supported = Vec::new();
    let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
    let mut props = vk::FormatProperties2::default().push_next(&mut list);
    unsafe {
        properties.get_physical_device_format_properties2(physical_device, format, &mut props)
    };
    if list.drm_format_modifier_count == 0 {
        return Ok(supported);
    }
    modifiers.resize(
        list.drm_format_modifier_count as usize,
        vk::DrmFormatModifierPropertiesEXT::default(),
    );
    list =
        vk::DrmFormatModifierPropertiesListEXT::default().drm_format_modifier_properties(modifiers);
    let mut props = vk::FormatProperties2::default().push_next(&mut list);
    unsafe {
        properties.get_physical_device_format_properties2(physical_device, format, &mut props)
    };
    let count = list.drm_format_modifier_count as usize;
    for modifier in modifiers.iter().take(count) {
        if modifier.drm_format_modifier_plane_count != 1 {
            continue;
        }
        let mut external = vk::PhysicalDeviceExternalImageFormatInfo::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let mut drm = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
            .drm_format_modifier(modifier.drm_format_modifier)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(format)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(usage)
            .push_next(&mut external)
            .push_next(&mut drm);
        let mut external_props = vk::ExternalImageFormatProperties::default();
        let mut props = vk::ImageFormatProperties2::default().push_next(&mut external_props);
        match unsafe {
            properties.get_physical_device_image_format_properties2(
                physical_device,
                &info,
                &mut props,
            )
        } {
            Ok(()) => {}
            Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => continue,
            Err(error) => return Err(super::map_host_device_oom_and_lost_err(error)),
        }
        if !external_props
            .external_memory_properties
            .external_memory_features
            .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
        {
            continue;
        }
        supported.push(DmabufModifier {
            modifier: modifier.drm_format_modifier,
            plane_count: modifier.drm_format_modifier_plane_count,
            tiling_features: modifier.drm_format_modifier_tiling_features,
            external_memory_features: external_props
                .external_memory_properties
                .external_memory_features,
        });
    }
    Ok(supported)
}
