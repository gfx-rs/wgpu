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
    use wgt::TextureViewDimension as Tvd;
    match dim {
        Tvd::D1 => mtl::MTLTextureType::D1,
        Tvd::D2 => mtl::MTLTextureType::D2,
        Tvd::D2Array => mtl::MTLTextureType::D2Array,
        Tvd::D3 => mtl::MTLTextureType::D3,
        Tvd::Cube => mtl::MTLTextureType::Cube,
        Tvd::CubeArray => mtl::MTLTextureType::CubeArray,
    }
}
