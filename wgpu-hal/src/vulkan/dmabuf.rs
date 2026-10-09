use alloc::vec::Vec;
use ash::vk;

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
