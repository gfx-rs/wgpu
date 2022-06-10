use crate::resource;

pub fn is_power_of_two_u16(val: u16) -> bool {
    val != 0 && (val & (val - 1)) == 0
}

pub fn is_power_of_two_u32(val: u32) -> bool {
    val != 0 && (val & (val - 1)) == 0
}

pub fn is_valid_copy_src_texture_format(format: wgt::TextureFormat) -> bool {
    use wgt::TextureFormat as Tf;
    match format {
        Tf::Depth24Plus | Tf::Depth24PlusStencil8 => false,
        _ => true,
    }
}

pub fn is_valid_copy_dst_texture_format(format: wgt::TextureFormat) -> bool {
    use wgt::TextureFormat as Tf;
    match format {
        Tf::Depth32Float
        | Tf::Depth32FloatStencil8
        | Tf::Depth24Plus
        | Tf::Depth24PlusStencil8
        | Tf::Depth24UnormStencil8 => false,
        _ => true,
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
