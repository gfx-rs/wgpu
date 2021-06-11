use ash::vk;

impl super::PrivateCapabilities {
    pub fn map_texture_format(&self, format: wgt::TextureFormat) -> vk::Format {
        use ash::vk::Format as F;
        use wgt::TextureFormat as Tf;
        match format {
            Tf::R8Unorm => F::R8_UNORM,
            Tf::R8Snorm => F::R8_SNORM,
            Tf::R8Uint => F::R8_UINT,
            Tf::R8Sint => F::R8_SINT,
            Tf::R16Uint => F::R16_UINT,
            Tf::R16Sint => F::R16_SINT,
            Tf::R16Float => F::R16_SFLOAT,
            Tf::Rg8Unorm => F::R8G8_UNORM,
            Tf::Rg8Snorm => F::R8G8_SNORM,
            Tf::Rg8Uint => F::R8G8_UINT,
            Tf::Rg8Sint => F::R8G8_SINT,
            Tf::R32Uint => F::R32_UINT,
            Tf::R32Sint => F::R32_SINT,
            Tf::R32Float => F::R32_SFLOAT,
            Tf::Rg16Uint => F::R16G16_UINT,
            Tf::Rg16Sint => F::R16G16_SINT,
            Tf::Rg16Float => F::R16G16_SFLOAT,
            Tf::Rgba8Unorm => F::R8G8B8A8_UNORM,
            Tf::Rgba8UnormSrgb => F::R8G8B8A8_SRGB,
            Tf::Bgra8UnormSrgb => F::B8G8R8A8_UNORM,
            Tf::Rgba8Snorm => F::R8G8B8A8_SNORM,
            Tf::Bgra8Unorm => F::B8G8R8A8_SNORM,
            Tf::Rgba8Uint => F::R8G8B8A8_UINT,
            Tf::Rgba8Sint => F::R8G8B8A8_SINT,
            Tf::Rgb10a2Unorm => F::A2B10G10R10_UNORM_PACK32,
            Tf::Rg11b10Float => F::B10G11R11_UFLOAT_PACK32,
            Tf::Rg32Uint => F::R32G32_UINT,
            Tf::Rg32Sint => F::R32G32_SINT,
            Tf::Rg32Float => F::R32G32_SFLOAT,
            Tf::Rgba16Uint => F::R16G16B16A16_UINT,
            Tf::Rgba16Sint => F::R16G16B16A16_SINT,
            Tf::Rgba16Float => F::R16G16B16A16_SFLOAT,
            Tf::Rgba32Uint => F::R32G32B32A32_UINT,
            Tf::Rgba32Sint => F::R32G32B32A32_SINT,
            Tf::Rgba32Float => F::R32G32B32A32_SFLOAT,
            Tf::Depth32Float => F::D32_SFLOAT,
            Tf::Depth24Plus => {
                if self.texture_d24 {
                    F::X8_D24_UNORM_PACK32
                } else {
                    F::D32_SFLOAT
                }
            }
            Tf::Depth24PlusStencil8 => {
                if self.texture_d24_s8 {
                    F::D24_UNORM_S8_UINT
                } else {
                    F::D32_SFLOAT_S8_UINT
                }
            }
            Tf::Bc1RgbaUnorm => F::BC1_RGBA_UNORM_BLOCK,
            Tf::Bc1RgbaUnormSrgb => F::BC1_RGBA_SRGB_BLOCK,
            Tf::Bc2RgbaUnorm => F::BC2_UNORM_BLOCK,
            Tf::Bc2RgbaUnormSrgb => F::BC2_SRGB_BLOCK,
            Tf::Bc3RgbaUnorm => F::BC3_UNORM_BLOCK,
            Tf::Bc3RgbaUnormSrgb => F::BC3_SRGB_BLOCK,
            Tf::Bc4RUnorm => F::BC4_UNORM_BLOCK,
            Tf::Bc4RSnorm => F::BC4_SNORM_BLOCK,
            Tf::Bc5RgUnorm => F::BC5_UNORM_BLOCK,
            Tf::Bc5RgSnorm => F::BC5_SNORM_BLOCK,
            Tf::Bc6hRgbSfloat => F::BC6H_SFLOAT_BLOCK,
            Tf::Bc6hRgbUfloat => F::BC6H_UFLOAT_BLOCK,
            Tf::Bc7RgbaUnorm => F::BC7_UNORM_BLOCK,
            Tf::Bc7RgbaUnormSrgb => F::BC7_SRGB_BLOCK,
            Tf::Etc2RgbUnorm => F::ETC2_R8G8B8_UNORM_BLOCK,
            Tf::Etc2RgbUnormSrgb => F::ETC2_R8G8B8_SRGB_BLOCK,
            Tf::Etc2RgbA1Unorm => F::ETC2_R8G8B8A1_UNORM_BLOCK,
            Tf::Etc2RgbA1UnormSrgb => F::ETC2_R8G8B8A1_SRGB_BLOCK,
            Tf::EacRUnorm => F::EAC_R11_UNORM_BLOCK,
            Tf::EacRSnorm => F::EAC_R11_SNORM_BLOCK,
            Tf::EtcRgUnorm => F::EAC_R11G11_UNORM_BLOCK,
            Tf::EtcRgSnorm => F::EAC_R11G11_SNORM_BLOCK,
            Tf::Astc4x4RgbaUnorm => F::ASTC_4X4_UNORM_BLOCK,
            Tf::Astc4x4RgbaUnormSrgb => F::ASTC_4X4_SRGB_BLOCK,
            Tf::Astc5x4RgbaUnorm => F::ASTC_5X4_UNORM_BLOCK,
            Tf::Astc5x4RgbaUnormSrgb => F::ASTC_5X4_SRGB_BLOCK,
            Tf::Astc5x5RgbaUnorm => F::ASTC_5X5_UNORM_BLOCK,
            Tf::Astc5x5RgbaUnormSrgb => F::ASTC_5X5_SRGB_BLOCK,
            Tf::Astc6x5RgbaUnorm => F::ASTC_6X5_UNORM_BLOCK,
            Tf::Astc6x5RgbaUnormSrgb => F::ASTC_6X5_SRGB_BLOCK,
            Tf::Astc6x6RgbaUnorm => F::ASTC_6X6_UNORM_BLOCK,
            Tf::Astc6x6RgbaUnormSrgb => F::ASTC_6X6_SRGB_BLOCK,
            Tf::Astc8x5RgbaUnorm => F::ASTC_8X5_UNORM_BLOCK,
            Tf::Astc8x5RgbaUnormSrgb => F::ASTC_8X5_SRGB_BLOCK,
            Tf::Astc8x6RgbaUnorm => F::ASTC_8X6_UNORM_BLOCK,
            Tf::Astc8x6RgbaUnormSrgb => F::ASTC_8X6_SRGB_BLOCK,
            Tf::Astc10x5RgbaUnorm => F::ASTC_8X8_UNORM_BLOCK,
            Tf::Astc10x5RgbaUnormSrgb => F::ASTC_8X8_SRGB_BLOCK,
            Tf::Astc10x6RgbaUnorm => F::ASTC_10X5_UNORM_BLOCK,
            Tf::Astc10x6RgbaUnormSrgb => F::ASTC_10X5_SRGB_BLOCK,
            Tf::Astc8x8RgbaUnorm => F::ASTC_10X6_UNORM_BLOCK,
            Tf::Astc8x8RgbaUnormSrgb => F::ASTC_10X6_SRGB_BLOCK,
            Tf::Astc10x8RgbaUnorm => F::ASTC_10X8_UNORM_BLOCK,
            Tf::Astc10x8RgbaUnormSrgb => F::ASTC_10X8_SRGB_BLOCK,
            Tf::Astc10x10RgbaUnorm => F::ASTC_10X10_UNORM_BLOCK,
            Tf::Astc10x10RgbaUnormSrgb => F::ASTC_10X10_SRGB_BLOCK,
            Tf::Astc12x10RgbaUnorm => F::ASTC_12X10_UNORM_BLOCK,
            Tf::Astc12x10RgbaUnormSrgb => F::ASTC_12X10_SRGB_BLOCK,
            Tf::Astc12x12RgbaUnorm => F::ASTC_12X12_UNORM_BLOCK,
            Tf::Astc12x12RgbaUnormSrgb => F::ASTC_12X12_SRGB_BLOCK,
        }
    }
}

pub fn map_texture_usage(usage: crate::TextureUse) -> vk::ImageUsageFlags {
    let mut flags = vk::ImageUsageFlags::empty();
    if usage.contains(crate::TextureUse::COPY_SRC) {
        flags |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    if usage.contains(crate::TextureUse::COPY_DST) {
        flags |= vk::ImageUsageFlags::TRANSFER_DST;
    }
    if usage.contains(crate::TextureUse::SAMPLED) {
        flags |= vk::ImageUsageFlags::SAMPLED;
    }
    if usage.contains(crate::TextureUse::COLOR_TARGET) {
        flags |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
    }
    if usage
        .intersects(crate::TextureUse::DEPTH_STENCIL_READ | crate::TextureUse::DEPTH_STENCIL_WRITE)
    {
        flags |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
    }
    if usage.intersects(crate::TextureUse::STORAGE_LOAD | crate::TextureUse::STORAGE_STORE) {
        flags |= vk::ImageUsageFlags::STORAGE;
    }
    flags
}

pub fn map_vk_image_usage(usage: vk::ImageUsageFlags) -> crate::TextureUse {
    let mut bits = crate::TextureUse::empty();
    if usage.contains(vk::ImageUsageFlags::TRANSFER_SRC) {
        bits |= crate::TextureUse::COPY_SRC;
    }
    if usage.contains(vk::ImageUsageFlags::TRANSFER_DST) {
        bits |= crate::TextureUse::COPY_DST;
    }
    if usage.contains(vk::ImageUsageFlags::SAMPLED) {
        bits |= crate::TextureUse::SAMPLED;
    }
    if usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
        bits |= crate::TextureUse::COLOR_TARGET;
    }
    if usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
        bits |= crate::TextureUse::DEPTH_STENCIL_READ | crate::TextureUse::DEPTH_STENCIL_WRITE;
    }
    if usage.contains(vk::ImageUsageFlags::STORAGE) {
        bits |= crate::TextureUse::STORAGE_LOAD | crate::TextureUse::STORAGE_STORE;
    }
    bits
}

pub fn map_index_format(index_format: wgt::IndexFormat) -> vk::IndexType {
    match index_format {
        wgt::IndexFormat::Uint16 => vk::IndexType::UINT16,
        wgt::IndexFormat::Uint32 => vk::IndexType::UINT32,
    }
}

pub fn map_aspects(aspects: crate::FormatAspect) -> vk::ImageAspectFlags {
    let mut flags = vk::ImageAspectFlags::empty();
    if aspects.contains(crate::FormatAspect::COLOR) {
        flags |= vk::ImageAspectFlags::COLOR;
    }
    if aspects.contains(crate::FormatAspect::DEPTH) {
        flags |= vk::ImageAspectFlags::DEPTH;
    }
    if aspects.contains(crate::FormatAspect::STENCIL) {
        flags |= vk::ImageAspectFlags::STENCIL;
    }
    flags
}

pub fn map_origin(origin: wgt::Origin3d, texture_dim: wgt::TextureDimension) -> vk::Offset3D {
    vk::Offset3D {
        x: origin.x as i32,
        y: origin.y as i32,
        z: match texture_dim {
            wgt::TextureDimension::D3 => origin.z as i32,
            _ => 0,
        },
    }
}

pub fn map_extent(extent: wgt::Extent3d, texture_dim: wgt::TextureDimension) -> vk::Extent3D {
    vk::Extent3D {
        width: extent.width,
        height: extent.height,
        depth: match texture_dim {
            wgt::TextureDimension::D3 => extent.depth_or_array_layers,
            _ => 1,
        },
    }
}

pub fn map_attachment_ops(
    op: crate::AttachmentOp,
) -> (vk::AttachmentLoadOp, vk::AttachmentStoreOp) {
    let load_op = if op.contains(crate::AttachmentOp::LOAD) {
        vk::AttachmentLoadOp::LOAD
    } else {
        vk::AttachmentLoadOp::CLEAR
    };
    let store_op = if op.contains(crate::AttachmentOp::STORE) {
        vk::AttachmentStoreOp::STORE
    } else {
        vk::AttachmentStoreOp::DONT_CARE
    };
    (load_op, store_op)
}

pub fn map_present_mode(mode: wgt::PresentMode) -> vk::PresentModeKHR {
    match mode {
        wgt::PresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
        wgt::PresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
        wgt::PresentMode::Fifo => vk::PresentModeKHR::FIFO,
        //wgt::PresentMode::Relaxed => vk::PresentModeKHR::FIFO_RELAXED,
    }
}

pub fn map_vk_present_mode(mode: vk::PresentModeKHR) -> Option<wgt::PresentMode> {
    if mode == vk::PresentModeKHR::IMMEDIATE {
        Some(wgt::PresentMode::Immediate)
    } else if mode == vk::PresentModeKHR::MAILBOX {
        Some(wgt::PresentMode::Mailbox)
    } else if mode == vk::PresentModeKHR::FIFO {
        Some(wgt::PresentMode::Fifo)
    } else if mode == vk::PresentModeKHR::FIFO_RELAXED {
        //Some(wgt::PresentMode::Relaxed)
        None
    } else {
        log::warn!("Unrecognized present mode {:?}", mode);
        None
    }
}

pub fn map_composite_alpha_mode(mode: crate::CompositeAlphaMode) -> vk::CompositeAlphaFlagsKHR {
    match mode {
        crate::CompositeAlphaMode::Opaque => vk::CompositeAlphaFlagsKHR::OPAQUE,
        crate::CompositeAlphaMode::PostMultiplied => vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        crate::CompositeAlphaMode::PreMultiplied => vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
    }
}

pub fn map_vk_composite_alpha(flags: vk::CompositeAlphaFlagsKHR) -> Vec<crate::CompositeAlphaMode> {
    let mut modes = Vec::new();
    if flags.contains(vk::CompositeAlphaFlagsKHR::OPAQUE) {
        modes.push(crate::CompositeAlphaMode::Opaque);
    }
    if flags.contains(vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED) {
        modes.push(crate::CompositeAlphaMode::PostMultiplied);
    }
    if flags.contains(vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED) {
        modes.push(crate::CompositeAlphaMode::PreMultiplied);
    }
    modes
}
