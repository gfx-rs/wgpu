use openxr as xr;
use wgt::{
    openxr::{OpenXRHandles, OpenXROptions},
    Backend, BackendBit,
};

#[derive(Debug)]
pub struct WGPUOpenXR {
    #[cfg(vulkan)]
    pub vulkan: Option<gfx_backend_vulkan::OpenXR>,
    /*
    #[cfg(metal)]
    pub metal: Option<gfx_backend_metal::OpenXR>,
    #[cfg(dx12)]
    pub dx12: Option<gfx_backend_dx12::OpenXR>,
    #[cfg(dx11)]
    pub dx11: Option<gfx_backend_dx11::OpenXR>,
    #[cfg(gl)]
    pub gl: Option<gfx_backend_gl::OpenXR>,
     */
}

impl WGPUOpenXR {
    pub fn configure(
        backends: BackendBit,
        instance: xr::Instance,
        _options: OpenXROptions, // FIXME use these
    ) -> WGPUOpenXR {
        backends_map! {
            let map = |(backend, openxr_configure)| {
                if backends.contains(backend.into()) {
                    openxr_configure(instance).ok() // FIXME options
                } else {
                    None
                }
            };
            Self {
                #[cfg(vulkan)]
                vulkan: map((Backend::Vulkan, gfx_backend_vulkan::OpenXR::configure)),
                /*
                #[cfg(metal)]
                metal: map((Backend::Metal, gfx_backend_metal::OpenXR::configure)),
                #[cfg(dx12)]
                dx12: map((Backend::Dx12, gfx_backend_dx12::OpenXR::configure)),
                #[cfg(dx11)]
                dx11: map((Backend::Dx11, gfx_backend_dx11::OpenXR::configure)),
                #[cfg(gl)]
                gl: map((Backend::Gl, gfx_backend_gl::OpenXR::configure)),
                 */
            }
        }
    }

    /*
    // FIXME add correct error
    pub fn texture_from_raw_image<B, C>(
        &self,
        device: Arc<C>,
        raw_image: u64,
        resolution: Extent3d
    ) -> Result<resource::Texture<B>, resource::CreateTextureError> where B: hal::Backend {
        let texture = self.texture_from_raw_image_inner(raw_image, resolution);

        Ok(resource::Texture {
            raw: Option<(B::Image, MemoryBlock<B>)>,
            device_id: Stored<DeviceId>,
            usage: wgt::TextureUsage,
            aspects: hal::format::Aspects,
            dimension: wgt::TextureDimension,
            kind: hal::image::Kind,
            format: wgt::TextureFormat,
            format_features: wgt::TextureFormatFeatures,
            framebuffer_attachment: hal::image::FramebufferAttachment,
            full_range: TextureSelector,
            life_guard: LifeGuard,
        })
    }

    fn texture_from_raw_image_inner<B>(
        &self,
        raw_image: u64,
        resolution: Extent3d
    ) -> Result<B::Image, ()> where B: hal::Backend {
        backends_map! {
            let map = |(is_backend, texture_from_raw_image)| {
                if is_backend {
                    texture_from_raw_image(
                        raw_image,
                        hal::image::Kind::D2(resolution.width, resolution.height, 1, 1),
                        hal::image::ViewCapabilities::KIND_2D_ARRAY
                    )
                } else {
                    Err(())
                }
            };

            #[cfg(vulkan)]
            map((self.vulkan.is_some(), gfx_backend_vulkan::OpenXR::texture_from_raw_image)),
        }
    }
    */

    pub fn get_session_handles(&self) -> Option<OpenXRHandles> {
        backends_map! {
            let map = |(is_backend, get_session_handles)| {
                if is_backend {
                    let ret = get_session_handles();
                    Some(OpenXRHandles {
                        session: ret.0,
                        frame_waiter: ret.1,
                        frame_stream: ret.2,
                        space: ret.3,
                        system: ret.4,
                    })
                } else {
                    None
                }
            };

            #[cfg(vulkan)]
            map((self.vulkan.is_some(), gfx_backend_vulkan::OpenXR::get_session_handles)),
        }
    }
}
