#![allow(clippy::let_unit_value)] // `let () =` being used to constrain result type

use std::thread;

use core_graphics_types::{
    base::CGFloat,
    geometry::{CGRect, CGSize},
};
use objc::{
    class, msg_send,
    rc::autoreleasepool,
    runtime::{BOOL, YES},
    sel, sel_impl,
};
use parking_lot::{Mutex, RwLock};

impl super::Surface {
    pub fn new(layer: metal::MetalLayer) -> Self {
        Self {
            render_layer: Mutex::new(layer),
            swapchain_format: RwLock::new(None),
            extent: RwLock::new(wgt::Extent3d::default()),
            main_thread_id: thread::current().id(),
            present_with_transaction: false,
        }
    }

    pub fn from_layer(layer: &metal::MetalLayerRef) -> Self {
        let class = class!(CAMetalLayer);
        let proper_kind: BOOL = unsafe { msg_send![layer, isKindOfClass: class] };
        assert_eq!(proper_kind, YES);
        Self::new(layer.to_owned())
    }

    pub(super) fn dimensions(&self) -> wgt::Extent3d {
        let (size, scale): (CGSize, CGFloat) = unsafe {
            let render_layer_borrow = self.render_layer.lock();
            let render_layer = render_layer_borrow.as_ref();
            let bounds: CGRect = msg_send![render_layer, bounds];
            let contents_scale: CGFloat = msg_send![render_layer, contentsScale];
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
        log::debug!("build swapchain {:?}", config);

        let caps = &device.shared.private_caps;
        *self.swapchain_format.write() = Some(config.format);
        *self.extent.write() = config.extent;

        let render_layer = self.render_layer.lock();
        let framebuffer_only = config.usage == crate::TextureUses::COLOR_TARGET;
        let display_sync = match config.present_mode {
            wgt::PresentMode::Fifo => true,
            wgt::PresentMode::Immediate => false,
            m => unreachable!("Unsupported present mode: {m:?}"),
        };
        let drawable_size = CGSize::new(config.extent.width as f64, config.extent.height as f64);

        match config.composite_alpha_mode {
            wgt::CompositeAlphaMode::Opaque => render_layer.set_opaque(true),
            wgt::CompositeAlphaMode::PostMultiplied => render_layer.set_opaque(false),
            _ => (),
        }

        let device_raw = device.shared.device.lock();
        render_layer.set_device(&device_raw);
        render_layer.set_pixel_format(caps.map_format(config.format));
        render_layer.set_framebuffer_only(framebuffer_only);
        render_layer.set_presents_with_transaction(self.present_with_transaction);
        // opt-in to Metal EDR
        // EDR potentially more power used in display and more bandwidth, memory footprint.
        let wants_edr = config.format == wgt::TextureFormat::Rgba16Float;
        if wants_edr != render_layer.wants_extended_dynamic_range_content() {
            render_layer.set_wants_extended_dynamic_range_content(wants_edr);
        }

        // this gets ignored on iOS for certain OS/device combinations (iphone5s iOS 10.3)
        render_layer.set_maximum_drawable_count(config.maximum_frame_latency as u64 + 1);
        render_layer.set_drawable_size(drawable_size);
        if caps.can_set_next_drawable_timeout {
            let () = msg_send![*render_layer, setAllowsNextDrawableTimeout:false];
        }
        if caps.can_set_display_sync {
            let () = msg_send![*render_layer, setDisplaySyncEnabled: display_sync];
        }

        Ok(())
    }

    unsafe fn unconfigure(&self, _device: &super::Device) {
        *self.swapchain_format.write() = None;
    }

    unsafe fn acquire_texture(
        &self,
        _timeout_ms: Option<std::time::Duration>, //TODO
        _fence: &super::Fence,
    ) -> Result<Option<crate::AcquiredSurfaceTexture<super::Api>>, crate::SurfaceError> {
        let render_layer = self.render_layer.lock();
        let (drawable, texture) = match autoreleasepool(|| {
            render_layer
                .next_drawable()
                .map(|drawable| (drawable.to_owned(), drawable.texture().to_owned()))
        }) {
            Some(pair) => pair,
            None => return Ok(None),
        };

        let swapchain_format = self.swapchain_format.read().unwrap();
        let extent = self.extent.read();
        let suf_texture = super::SurfaceTexture {
            texture: super::Texture {
                raw: texture,
                format: swapchain_format,
                raw_type: metal::MTLTextureType::D2,
                array_layers: 1,
                mip_levels: 1,
                copy_size: crate::CopyExtent {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
            },
            drawable,
            present_with_transaction: self.present_with_transaction,
        };

        Ok(Some(crate::AcquiredSurfaceTexture {
            texture: suf_texture,
            suboptimal: false,
        }))
    }

    unsafe fn discard_texture(&self, _texture: super::SurfaceTexture) {}
}
