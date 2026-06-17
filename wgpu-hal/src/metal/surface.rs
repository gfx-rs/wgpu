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

    /// Reads the EDR headroom of the screen currently hosting this surface's
    /// layer, as a [`wgt::DisplayHdrInfo`].
    ///
    /// Apple exposes a *relative* EDR multiplier (`1.0` == SDR white), no
    /// absolute nits, and no discrete HDR-mode flag — so this fills the
    /// `headroom` frame and *derives* `hdr_active` from the live `current`
    /// multiplier (`> 1.0`), not from `potential` (which is `> 1.0` on nearly
    /// every Apple display and would conflate "capable" with "active").
    ///
    /// macOS only for now: it returns `None` on iOS/tvOS/visionOS (their
    /// `UIScreen.currentEDRHeadroom` path is a follow-up) and whenever the
    /// hosting screen cannot be resolved.
    ///
    /// # Thread safety
    ///
    /// `NSScreen` / `NSWindow` are main-thread-affine; reading them off the main
    /// thread is undefined behavior. This gates on `pthread_main_np()` and
    /// returns `None` — with *no* Objective-C message send — when not on the
    /// main thread, rather than risk a crash or a `dispatch_sync` deadlock.
    pub(super) fn display_hdr_info(&self) -> Option<wgt::DisplayHdrInfo> {
        #[cfg(target_os = "macos")]
        {
            use objc2::rc::Retained;
            use objc2::runtime::NSObject;
            use objc2_quartz_core::CALayer;

            // SAFETY: `pthread_main_np` is always safe to call. Bail before any
            // message send if we are not on the main thread.
            if unsafe { libc::pthread_main_np() } == 0 {
                return None;
            }

            // Take an owned reference to the layer and drop the lock *before*
            // climbing to the window/screen, so a main-thread AppKit callback
            // can never deadlock against this lock.
            let render_layer = {
                let guard = self.render_layer.lock();
                guard.clone()
            };

            // Resolve the hosting `NSScreen` by climbing
            // layer → NSView (the delegate) → NSWindow → NSScreen, mirroring the
            // occlusion walk in `acquire_texture`. `NSWindow.screen` is nil when
            // the window is off-screen, so this can legitimately yield `None`.
            let screen: Retained<NSObject> = autoreleasepool(|_| {
                let mut current_layer: Option<Retained<CALayer>> =
                    Some(Retained::into_super(render_layer));
                while let Some(layer) = current_layer {
                    if let Some(delegate) = layer.delegate() {
                        let window: Option<Retained<NSObject>> =
                            unsafe { objc2::msg_send![&*delegate, window] };
                        return window
                            .and_then(|window| unsafe { objc2::msg_send![&*window, screen] });
                    }
                    current_layer = layer.superlayer();
                }
                None
            })?;

            // AppKit documents these EDR properties as finite multipliers
            // (`1.0` == SDR white), but guard against a non-finite read anyway so
            // the advisory values stay finite and `hdr_active` reports *unknown*
            // (`None`) rather than a false `Some(false)` — mirroring the
            // `is_finite` discipline in [`wgt::DisplayHdrInfo::tone_map_headroom`].
            // The EDR properties return `CGFloat` (`f64` on 64-bit macOS).
            let finite = |v: f64| v.is_finite().then_some(v as f32);

            // `maximumExtendedDynamicRangeColorComponentValue` is macOS 10.11+, so
            // it is safe at our 10.13 minimum.
            let current: f64 = unsafe {
                objc2::msg_send![&*screen, maximumExtendedDynamicRangeColorComponentValue]
            };

            // Apple exposes no discrete HDR-mode flag; derive it from the live
            // `current` EDR multiplier (`> 1.0`), *not* `potential` (which is
            // `> 1.0` on nearly every Apple display and would conflate "capable"
            // with "active").
            let hdr_active = current.is_finite().then_some(current > 1.0);

            let mut headroom = wgt::DisplayHeadroom::default();
            headroom.current = finite(current);
            // `maximumPotential…`/`maximumReference…` are macOS 10.15+, below which
            // sending them would raise an unrecognized-selector exception — so
            // only message them where available; otherwise leave them `None`.
            if available!(macos = 10.15) {
                let potential: f64 = unsafe {
                    objc2::msg_send![
                        &*screen,
                        maximumPotentialExtendedDynamicRangeColorComponentValue
                    ]
                };
                let reference: f64 = unsafe {
                    objc2::msg_send![
                        &*screen,
                        maximumReferenceExtendedDynamicRangeColorComponentValue
                    ]
                };
                headroom.potential = finite(potential);
                // AppKit reports `0.0` when there is no reference value; treat that
                // (and any non-finite read) as "unknown" rather than a real `0.0`.
                headroom.reference = finite(reference).filter(|&v| v > 0.0);
            }

            let mut coarse = wgt::DisplayCoarseRange::default();
            coarse.high_dynamic_range = hdr_active;
            // `NSScreen.colorSpace` returns generic names on HDR panels
            // post-Monterey, so deriving a gamut bucket from it would lie — leave
            // `coarse.gamut` as `None` on macOS.

            let mut info = wgt::DisplayHdrInfo::default();
            info.hdr_active = hdr_active;
            info.headroom = Some(headroom);
            info.coarse = Some(coarse);
            // Apple exposes no absolute nits, CIE-xy primaries, or panel bit
            // depth, so `luminance` / `chromaticity` / `bits_per_color` stay
            // `None`.
            Some(info)
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
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
        // opt-in to Metal EDR
        // EDR potentially more power used in display and more bandwidth, memory footprint.
        let wants_edr = matches!(
            config.color_space,
            wgt::SurfaceColorSpace::ExtendedSrgbLinear
                | wgt::SurfaceColorSpace::ExtendedSrgb
                | wgt::SurfaceColorSpace::ExtendedDisplayP3
                | wgt::SurfaceColorSpace::Hdr10
                | wgt::SurfaceColorSpace::Hlg
        );
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
            wgt::SurfaceColorSpace::Hdr10 | wgt::SurfaceColorSpace::Hlg => {
                // The ITUR_2100 color space constants require macOS 11.0/iOS 14.0;
                // `surface_capabilities` only reports HDR10/HLG on those OS versions.
                if !available!(macos = 11.0, ios = 14.0, tvos = 14.0, visionos = 1.0) {
                    unreachable!("HDR10/HLG color spaces are only reported on macOS 11.0+/iOS 14.0+/tvOS 14.0+");
                }
                Some(if config.color_space == wgt::SurfaceColorSpace::Hdr10 {
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
            use objc2::runtime::NSObject;
            use objc2_quartz_core::CALayer;

            // The CAMetalLayer is typically a sublayer, so we need to traverse up
            // to find the root layer whose delegate is the NSView.
            let mut current_layer: Option<Retained<CALayer>> =
                Some(Retained::into_super(render_layer.clone()));

            while let Some(layer) = current_layer {
                if let Some(delegate) = layer.delegate() {
                    // Found a layer with a delegate - this should be the NSView
                    let window: Option<Retained<NSObject>> =
                        unsafe { objc2::msg_send![&*delegate, window] };

                    if let Some(window) = window {
                        const NS_WINDOW_OCCLUSION_STATE_VISIBLE: usize = 1 << 1;
                        let occlusion_state: usize =
                            unsafe { objc2::msg_send![&*window, occlusionState] };
                        let is_visible = (occlusion_state & NS_WINDOW_OCCLUSION_STATE_VISIBLE) != 0;

                        if !is_visible {
                            return Err(crate::SurfaceError::Occluded);
                        }
                    }
                    break;
                }
                current_layer = layer.superlayer();
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
