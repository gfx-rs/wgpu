pub fn map_texture_usage(usage: crate::TextureUse) -> mtl::MTLTextureUsage {
    use crate::TextureUse as Tu;

    let mut mtl_usage = mtl::MTLTextureUsage::Unknown;

    mtl_usage.set(
        mtl::MTLTextureUsage::RenderTarget,
        usage.intersects(Tu::COLOR_TARGET | Tu::DEPTH_STENCIL_READ | Tu::DEPTH_STENCIL_WRITE),
    );
    mtl_usage.set(
        mtl::MTLTextureUsage::ShaderRead,
        usage.intersects(Tu::SAMPLED | Tu::DEPTH_STENCIL_READ | Tu::STORAGE_LOAD),
    );
    mtl_usage.set(
        mtl::MTLTextureUsage::ShaderWrite,
        usage.intersects(Tu::STORAGE_STORE),
    );

    mtl_usage
}

pub fn map_texture_view_dimension(dim: wgt::TextureViewDimension) -> mtl::MTLTextureType {
    use mtl::MTLTextureType::*;
    use wgt::TextureViewDimension as Tvd;
    match dim {
        Tvd::D1 => D1,
        Tvd::D2 => D2,
        Tvd::D2Array => D2Array,
        Tvd::D3 => D3,
        Tvd::Cube => Cube,
        Tvd::CubeArray => CubeArray,
    }
}

pub fn map_compare_function(fun: wgt::CompareFunction) -> mtl::MTLCompareFunction {
    use mtl::MTLCompareFunction::*;
    use wgt::CompareFunction as Cf;
    match fun {
        Cf::Never => Never,
        Cf::Less => Less,
        Cf::LessEqual => LessEqual,
        Cf::Equal => Equal,
        Cf::GreaterEqual => GreaterEqual,
        Cf::Greater => Greater,
        Cf::NotEqual => NotEqual,
        Cf::Always => Always,
    }
}

pub fn map_filter_mode(filter: wgt::FilterMode) -> mtl::MTLSamplerMinMagFilter {
    use mtl::MTLSamplerMinMagFilter::*;
    match filter {
        wgt::FilterMode::Nearest => Nearest,
        wgt::FilterMode::Linear => Linear,
    }
}

pub fn map_address_mode(address: wgt::AddressMode) -> mtl::MTLSamplerAddressMode {
    use mtl::MTLSamplerAddressMode::*;
    use wgt::AddressMode as Fm;
    match address {
        Fm::Repeat => Repeat,
        Fm::MirrorRepeat => MirrorRepeat,
        Fm::ClampToEdge => ClampToEdge,
        Fm::ClampToBorder => ClampToBorderColor,
        //Fm::MirrorClamp => MirrorClampToEdge,
    }
}

pub fn map_border_color(border_color: wgt::SamplerBorderColor) -> mtl::MTLSamplerBorderColor {
    use mtl::MTLSamplerBorderColor::*;
    match border_color {
        wgt::SamplerBorderColor::TransparentBlack => TransparentBlack,
        wgt::SamplerBorderColor::OpaqueBlack => OpaqueBlack,
        wgt::SamplerBorderColor::OpaqueWhite => OpaqueWhite,
    }
}
