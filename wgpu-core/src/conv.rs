use wgt::TextureFormatFeatures;

use crate::resource::{self, TextureDescriptor};

pub fn is_power_of_two_u16(val: u16) -> bool {
    val != 0 && (val & (val - 1)) == 0
}

pub fn is_power_of_two_u32(val: u32) -> bool {
    val != 0 && (val & (val - 1)) == 0
}

pub fn is_valid_copy_src_texture_format(
    format: wgt::TextureFormat,
    aspect: wgt::TextureAspect,
) -> bool {
    use wgt::TextureAspect as Ta;
    use wgt::TextureFormat as Tf;
    match (format, aspect) {
        (Tf::Depth24Plus, _) | (Tf::Depth24PlusStencil8, Ta::DepthOnly) => false,
        _ => true,
    }
}

pub fn is_valid_copy_dst_texture_format(
    format: wgt::TextureFormat,
    aspect: wgt::TextureAspect,
) -> bool {
    use wgt::TextureAspect as Ta;
    use wgt::TextureFormat as Tf;
    match (format, aspect) {
        (Tf::Depth24Plus | Tf::Depth32Float, _)
        | (Tf::Depth24PlusStencil8 | Tf::Depth32FloatStencil8, Ta::DepthOnly) => false,
        _ => true,
    }
}

#[cfg_attr(
    any(not(target_arch = "wasm32"), target_os = "emscripten"),
    allow(unused)
)]
pub fn is_valid_external_image_copy_dst_texture_format(format: wgt::TextureFormat) -> bool {
    use wgt::TextureFormat as Tf;
    match format {
        Tf::R8Unorm
        | Tf::R16Float
        | Tf::R32Float
        | Tf::Rg8Unorm
        | Tf::Rg16Float
        | Tf::Rg32Float
        | Tf::Rgba8Unorm
        | Tf::Rgba8UnormSrgb
        | Tf::Bgra8Unorm
        | Tf::Bgra8UnormSrgb
        | Tf::Rgb10a2Unorm
        | Tf::Rgba16Float
        | Tf::Rgba32Float => true,
        _ => false,
    }
}

pub fn map_buffer_usage(usage: wgt::BufferUsages) -> hal::BufferUses {
    let mut u = hal::BufferUses::empty();
    u.set(
        hal::BufferUses::MAP_READ,
        usage.contains(wgt::BufferUsages::MAP_READ),
    );
    u.set(
        hal::BufferUses::MAP_WRITE,
        usage.contains(wgt::BufferUsages::MAP_WRITE),
    );
    u.set(
        hal::BufferUses::COPY_SRC,
        usage.contains(wgt::BufferUsages::COPY_SRC),
    );
    u.set(
        hal::BufferUses::COPY_DST,
        usage.contains(wgt::BufferUsages::COPY_DST),
    );
    u.set(
        hal::BufferUses::INDEX,
        usage.contains(wgt::BufferUsages::INDEX),
    );
    u.set(
        hal::BufferUses::VERTEX,
        usage.contains(wgt::BufferUsages::VERTEX),
    );
    u.set(
        hal::BufferUses::UNIFORM,
        usage.contains(wgt::BufferUsages::UNIFORM),
    );
    u.set(
        hal::BufferUses::STORAGE_READ | hal::BufferUses::STORAGE_READ_WRITE,
        usage.contains(wgt::BufferUsages::STORAGE),
    );
    u.set(
        hal::BufferUses::INDIRECT,
        usage.contains(wgt::BufferUsages::INDIRECT),
    );
    u.set(
        hal::BufferUses::QUERY_RESOLVE,
        usage.contains(wgt::BufferUsages::QUERY_RESOLVE),
    );
    u
}

pub fn map_texture_usage(
    usage: wgt::TextureUsages,
    aspect: hal::FormatAspects,
) -> hal::TextureUses {
    let mut u = hal::TextureUses::empty();
    u.set(
        hal::TextureUses::COPY_SRC,
        usage.contains(wgt::TextureUsages::COPY_SRC),
    );
    u.set(
        hal::TextureUses::COPY_DST,
        usage.contains(wgt::TextureUsages::COPY_DST),
    );
    u.set(
        hal::TextureUses::RESOURCE,
        usage.contains(wgt::TextureUsages::TEXTURE_BINDING),
    );
    u.set(
        hal::TextureUses::STORAGE_READ | hal::TextureUses::STORAGE_READ_WRITE,
        usage.contains(wgt::TextureUsages::STORAGE_BINDING),
    );
    let is_color = aspect.contains(hal::FormatAspects::COLOR);
    u.set(
        hal::TextureUses::COLOR_TARGET,
        usage.contains(wgt::TextureUsages::RENDER_ATTACHMENT) && is_color,
    );
    u.set(
        hal::TextureUses::DEPTH_STENCIL_READ | hal::TextureUses::DEPTH_STENCIL_WRITE,
        usage.contains(wgt::TextureUsages::RENDER_ATTACHMENT) && !is_color,
    );
    u
}

pub fn map_texture_usage_for_texture(
    desc: &TextureDescriptor,
    format_features: &TextureFormatFeatures,
) -> hal::TextureUses {
    // Enforce having COPY_DST/DEPTH_STENCIL_WRITE/COLOR_TARGET otherwise we
    // wouldn't be able to initialize the texture.
    map_texture_usage(desc.usage, desc.format.into())
        | if desc.format.is_depth_stencil_format() {
            hal::TextureUses::DEPTH_STENCIL_WRITE
        } else if desc.usage.contains(wgt::TextureUsages::COPY_DST) {
            hal::TextureUses::COPY_DST // (set already)
        } else {
            // Use COPY_DST only if we can't use COLOR_TARGET
            if format_features
                .allowed_usages
                .contains(wgt::TextureUsages::RENDER_ATTACHMENT)
                && desc.dimension == wgt::TextureDimension::D2
            // Render targets dimension must be 2d
            {
                hal::TextureUses::COLOR_TARGET
            } else {
                hal::TextureUses::COPY_DST
            }
        }
}

pub fn map_texture_usage_from_hal(uses: hal::TextureUses) -> wgt::TextureUsages {
    let mut u = wgt::TextureUsages::empty();
    u.set(
        wgt::TextureUsages::COPY_SRC,
        uses.contains(hal::TextureUses::COPY_SRC),
    );
    u.set(
        wgt::TextureUsages::COPY_DST,
        uses.contains(hal::TextureUses::COPY_DST),
    );
    u.set(
        wgt::TextureUsages::TEXTURE_BINDING,
        uses.contains(hal::TextureUses::RESOURCE),
    );
    u.set(
        wgt::TextureUsages::STORAGE_BINDING,
        uses.contains(hal::TextureUses::STORAGE_READ | hal::TextureUses::STORAGE_READ_WRITE),
    );
    u.set(
        wgt::TextureUsages::RENDER_ATTACHMENT,
        uses.contains(hal::TextureUses::COLOR_TARGET),
    );
    u
}

pub fn check_texture_dimension_size(
    dimension: wgt::TextureDimension,
    wgt::Extent3d {
        width,
        height,
        depth_or_array_layers,
    }: wgt::Extent3d,
    sample_size: u32,
    limits: &wgt::Limits,
) -> Result<(), resource::TextureDimensionError> {
    use resource::{TextureDimensionError as Tde, TextureErrorDimension as Ted};
    use wgt::TextureDimension::*;

    let (extent_limits, sample_limit) = match dimension {
        D1 => ([limits.max_texture_dimension_1d, 1, 1], 1),
        D2 => (
            [
                limits.max_texture_dimension_2d,
                limits.max_texture_dimension_2d,
                limits.max_texture_array_layers,
            ],
            32,
        ),
        D3 => (
            [
                limits.max_texture_dimension_3d,
                limits.max_texture_dimension_3d,
                limits.max_texture_dimension_3d,
            ],
            1,
        ),
    };

    for (&dim, (&given, &limit)) in [Ted::X, Ted::Y, Ted::Z].iter().zip(
        [width, height, depth_or_array_layers]
            .iter()
            .zip(extent_limits.iter()),
    ) {
        if given == 0 {
            return Err(Tde::Zero(dim));
        }
        if given > limit {
            return Err(Tde::LimitExceeded { dim, given, limit });
        }
    }
    if sample_size == 0 || sample_size > sample_limit || !is_power_of_two_u32(sample_size) {
        return Err(Tde::InvalidSampleCount(sample_size));
    }

    Ok(())
}

pub fn bind_group_layout_flags(features: wgt::Features) -> hal::BindGroupLayoutFlags {
    let mut flags = hal::BindGroupLayoutFlags::empty();
    flags.set(
        hal::BindGroupLayoutFlags::PARTIALLY_BOUND,
        features.contains(wgt::Features::PARTIALLY_BOUND_BINDING_ARRAY),
    );
    flags
}
