use alloc::borrow::ToOwned as _;

use objc2::{
    available,
    rc::{autoreleasepool, Retained},
    runtime::ProtocolObject,
    ClassType, Message,
};
use objc2_core_foundation::{CFString, CGSize};
use objc2_core_graphics::CGColorSpace;
use objc2_foundation::NSObjectProtocol;
use objc2_metal::MTLTextureType;
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
use parking_lot::{Mutex, RwLock};

use super::OsFeatures;

/// Walks up from `start` to the `NSWindow` hosting this layer: the first ancestor
/// layer with a delegate is the backing `NSView`, and we return its `window`.
/// `None` if no ancestor has a delegate.
#[cfg(target_os = "macos")]
fn hosting_window(
    start: Retained<objc2_quartz_core::CALayer>,
) -> Option<Retained<objc2::runtime::NSObject>> {
    let mut current = Some(start);
    while let Some(layer) = current {
        if let Some(delegate) = layer.delegate() {
            return unsafe { objc2::msg_send![&*delegate, window] };
        }
        current = layer.superlayer();
    }
    None
}

impl super::Surface {
    pub fn new(layer: Retained<CAMetalLayer>) -> Self {
        Self {
            render_layer: Mutex::new(layer),
            swapchain_format: RwLock::new(None),
            extent: RwLock::new(wgt::Extent3d::default()),
        }
    }

    pub fn from_layer(layer: &CAMetalLayer) -> Self {
        assert!(layer.isKindOfClass(CAMetalLayer::class()));
        Self::new(layer.retain())
    }

    pub fn render_layer(&self) -> &Mutex<Retained<CAMetalLayer>> {
        &self.render_layer
    }

    /// Gets the current dimensions of the `Surface`.
    ///
    /// This function is safe to call off of the main thread. However, note that
    /// `bounds` and `contentsScale` may be modified by the main thread while
    /// this function is running, possibly resulting in the two values being out
    /// of sync. This is sound, as these properties are accessed atomically.
    /// See: <https://github.com/gfx-rs/wgpu/pull/7692>
    pub(super) fn dimensions(&self) -> wgt::Extent3d {
        let (size, scale) = {
            let render_layer = self.render_layer.lock();
            let bounds = render_layer.bounds();
            let contents_scale = render_layer.contentsScale();
            (bounds.size, contents_scale)
        };

        wgt::Extent3d {
            width: (size.width * scale) as u32,
            height: (size.height * scale) as u32,
            depth_or_array_layers: 1,
        }
    }
}

impl crate::Surface for super::Surface {
    type A = super::Api;

    unsafe fn configure(
        &self,
        device: &super::Device,
        config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        log::debug!("build swapchain {config:?}");

        let caps = &device.shared.private_texture_format_caps;
        *self.swapchain_format.write() = Some(config.format);
        *self.extent.write() = config.extent;

        let render_layer = self.render_layer.lock();
        let framebuffer_only = config.usage == wgt::TextureUses::COLOR_TARGET;
        let display_sync = match config.present_mode {
            wgt::PresentMode::Fifo => true,
            wgt::PresentMode::Immediate => false,
            m => unreachable!("Unsupported present mode: {m:?}"),
        };
        // CGFloat is f64 on 64-bit, f32 on 32-bit (arm64_32/ILP32)
        let drawable_size = CGSize::new(config.extent.width as _, config.extent.height as _);

        match config.composite_alpha_mode {
            wgt::CompositeAlphaMode::Opaque => render_layer.setOpaque(true),
            wgt::CompositeAlphaMode::PostMultiplied => render_layer.setOpaque(false),
            _ => (),
        }

        let device_raw = &device.shared.device;
        render_layer.setDevice(Some(device_raw));
        render_layer.setPixelFormat(caps.map_format(config.format));
        render_layer.setFramebufferOnly(framebuffer_only);
        // Opt into Metal EDR for the HDR color spaces (more display power, memory,
        // and bandwidth). The HDR spaces are exactly those `is_hdr()` classifies.
        let wants_edr = config.color_space.is_hdr();
        if wants_edr != render_layer.wantsExtendedDynamicRangeContent() {
            render_layer.setWantsExtendedDynamicRangeContent(wants_edr);
        }

        let colorspace_name: Option<&'static CFString> = match config.color_space {
            wgt::SurfaceColorSpace::Auto => {
                unreachable!("wgpu-core resolves `Auto` before configuring the surface")
            }
            // Reset to the layer's default, which treats contents as sRGB.
            wgt::SurfaceColorSpace::Srgb => None,
            wgt::SurfaceColorSpace::ExtendedSrgbLinear => {
                Some(unsafe { objc2_core_graphics::kCGColorSpaceExtendedLinearSRGB })
            }
            wgt::SurfaceColorSpace::ExtendedSrgb => {
                Some(unsafe { objc2_core_graphics::kCGColorSpaceExtendedSRGB })
            }
            wgt::SurfaceColorSpace::ExtendedDisplayP3 => {
                Some(unsafe { objc2_core_graphics::kCGColorSpaceExtendedDisplayP3 })
            }
            wgt::SurfaceColorSpace::DisplayP3 => {
                Some(unsafe { objc2_core_graphics::kCGColorSpaceDisplayP3 })
            }
            wgt::SurfaceColorSpace::Bt2100Pq | wgt::SurfaceColorSpace::Bt2100Hlg => {
                // The ITUR_2100 color space constants require macOS 11.0/iOS 14.0;
                // `surface_capabilities` only reports BT.2100 PQ/HLG on those OS versions.
                if !available!(macos = 11.0, ios = 14.0, tvos = 14.0, visionos = 1.0) {
                    unreachable!("BT.2100 PQ/HLG color spaces are only reported on macOS 11.0+/iOS 14.0+/tvOS 14.0+");
                }
                Some(if config.color_space == wgt::SurfaceColorSpace::Bt2100Pq {
                    unsafe { objc2_core_graphics::kCGColorSpaceITUR_2100_PQ }
                } else {
                    unsafe { objc2_core_graphics::kCGColorSpaceITUR_2100_HLG }
                })
            }
        };
        let colorspace = colorspace_name.and_then(|name| CGColorSpace::with_name(Some(name)));
        render_layer.setColorspace(colorspace.as_deref());

        // this gets ignored on iOS for certain OS/device combinations (iphone5s iOS 10.3)
        render_layer.setMaximumDrawableCount(config.maximum_frame_latency as usize + 1);
        render_layer.setDrawableSize(drawable_size);
        // https://developer.apple.com/documentation/quartzcore/cametallayer/allowsnextdrawabletimeout
        if available!(macos = 10.13, ios = 11.0, tvos = 11.0, visionos = 1.0) {
            render_layer.setAllowsNextDrawableTimeout(false);
        }
        if OsFeatures::display_sync() {
            render_layer.setDisplaySyncEnabled(display_sync);
        }

        Ok(())
    }

    unsafe fn unconfigure(&self, _device: &super::Device) {
        *self.swapchain_format.write() = None;
    }

    unsafe fn acquire_texture(
        &self,
        _timeout: Option<core::time::Duration>, // TODO
        _fence: &super::Fence,
    ) -> Result<crate::AcquiredSurfaceTexture<super::Api>, crate::SurfaceError> {
        let render_layer = self.render_layer.lock();

        #[cfg(target_os = "macos")]
        {
            // Workaround for https://github.com/gfx-rs/wgpu/issues/8309
            // When the window is occluded on macOS, presented drawables get stuck waiting
            // for vsync. Check the window's occlusion state and skip acquisition if
            // the window is not visible - this avoids a 1-second hang in nextDrawable().
            use objc2::rc::Retained;

            // The CAMetalLayer is typically a sublayer; find the hosting window
            // and skip acquisition while it is occluded.
            if let Some(window) = hosting_window(Retained::into_super(render_layer.clone())) {
                const NS_WINDOW_OCCLUSION_STATE_VISIBLE: usize = 1 << 1;
                let occlusion_state: usize = unsafe { objc2::msg_send![&*window, occlusionState] };
                if occlusion_state & NS_WINDOW_OCCLUSION_STATE_VISIBLE == 0 {
                    return Err(crate::SurfaceError::Occluded);
                }
            }
        }

        let (drawable, texture) = match autoreleasepool(|_| {
            render_layer
                .nextDrawable()
                .map(|drawable| (drawable.to_owned(), drawable.texture().to_owned()))
        }) {
            Some(pair) => pair,
            None => return Err(crate::SurfaceError::Timeout),
        };

        let swapchain_format = self.swapchain_format.read().unwrap();
        let extent = self.extent.read();
        let suf_texture = super::SurfaceTexture {
            texture: super::Texture {
                raw: texture,
                format: swapchain_format,
                raw_type: MTLTextureType::Type2D,
                array_layers: 1,
                mip_levels: 1,
                copy_size: crate::CopyExtent {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
                _drop_guard: None,
            },
            drawable: ProtocolObject::from_retained(drawable),
            present_with_transaction: render_layer.presentsWithTransaction(),
        };

        Ok(crate::AcquiredSurfaceTexture {
            texture: suf_texture,
            suboptimal: false,
        })
    }

    unsafe fn discard_texture(&self, _texture: super::SurfaceTexture) {}
}
