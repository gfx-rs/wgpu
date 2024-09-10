#![allow(clippy::let_unit_value)] // `let () =` being used to constrain result type

use std::ptr::NonNull;
use std::sync::Once;
use std::thread;

use objc2::{
    class,
    declare::ClassBuilder,
    msg_send, msg_send_id,
    rc::{autoreleasepool, Retained},
    runtime::{AnyClass, AnyObject, Bool, ProtocolObject, Sel},
    sel, ClassType,
};
use objc2_foundation::{CGFloat, CGRect, CGSize, MainThreadMarker, NSObject, NSObjectProtocol};
use objc2_metal::MTLTextureType;
use objc2_quartz_core::{
    kCAGravityResize, CAAutoresizingMask, CALayer, CAMetalDrawable, CAMetalLayer,
};
use parking_lot::{Mutex, RwLock};

extern "C" fn layer_should_inherit_contents_scale_from_window(
    _: &AnyClass,
    _: Sel,
    _layer: *mut AnyObject,
    _new_scale: CGFloat,
    _from_window: *mut AnyObject,
) -> Bool {
    Bool::YES
}

static CAML_DELEGATE_REGISTER: Once = Once::new();

#[derive(Debug)]
pub struct HalManagedMetalLayerDelegate(&'static AnyClass);

impl HalManagedMetalLayerDelegate {
    pub fn new() -> Self {
        let class_name = format!("HalManagedMetalLayerDelegate@{:p}", &CAML_DELEGATE_REGISTER);

        CAML_DELEGATE_REGISTER.call_once(|| {
            let mut decl = ClassBuilder::new(&class_name, class!(NSObject)).unwrap();
            #[allow(trivial_casts)] // false positive
            unsafe {
                // <https://developer.apple.com/documentation/appkit/nsviewlayercontentscaledelegate/3005294-layer?language=objc>
                decl.add_class_method::<extern "C" fn(_, _, _, _, _) -> _>(
                    sel!(layer:shouldInheritContentsScale:fromWindow:),
                    layer_should_inherit_contents_scale_from_window,
                );
            }
            decl.register();
        });
        Self(AnyClass::get(&class_name).unwrap())
    }
}

impl super::Surface {
    fn new(layer: Retained<CAMetalLayer>) -> Self {
        Self {
            render_layer: Mutex::new(layer),
            swapchain_format: RwLock::new(None),
            extent: RwLock::new(wgt::Extent3d::default()),
            main_thread_id: thread::current().id(),
            present_with_transaction: false,
        }
    }

    /// If not called on the main thread, this will panic.
    pub unsafe fn from_view(view: NonNull<NSObject>) -> Self {
        let layer = unsafe { Self::get_metal_layer(view) };
        Self::new(layer)
    }

    pub unsafe fn from_layer(layer: &CAMetalLayer) -> Self {
        assert!(layer.isKindOfClass(CAMetalLayer::class()));
        Self::new(layer.retain())
    }

    /// Get or create a new `CAMetalLayer` associated with the given `NSView`
    /// or `UIView`.
    ///
    /// # Panics
    ///
    /// If called from a thread that is not the main thread, this will panic.
    ///
    /// # Safety
    ///
    /// The `view` must be a valid instance of `NSView` or `UIView`.
    pub(crate) unsafe fn get_metal_layer(view: NonNull<NSObject>) -> Retained<CAMetalLayer> {
        let Some(_mtm) = MainThreadMarker::new() else {
            panic!("get_metal_layer cannot be called in non-ui thread.");
        };

        // Ensure that the view is layer-backed.
        // Views are always layer-backed in UIKit.
        #[cfg(target_os = "macos")]
        let () = msg_send![view.as_ptr(), setWantsLayer: true];

        let root_layer: Option<Retained<CALayer>> = msg_send_id![view.as_ptr(), layer];
        // `-[NSView layer]` can return `NULL`, while `-[UIView layer]` should
        // always be available.
        let root_layer = root_layer.expect("failed making the view layer-backed");

        // NOTE: We explicitly do not touch properties such as
        // `layerContentsPlacement`, `needsDisplayOnBoundsChange` and
        // `contentsGravity` etc. on the root layer, both since we would like
        // to give the user full control over them, and because the default
        // values suit us pretty well (especially the contents placement being
        // `NSViewLayerContentsRedrawDuringViewResize`, which allows the view
        // to receive `drawRect:`/`updateLayer` calls).

        if root_layer.isKindOfClass(CAMetalLayer::class()) {
            // The view has a `CAMetalLayer` as the root layer, which can
            // happen for example if user overwrote `-[NSView layerClass]` or
            // the view is `MTKView`.
            //
            // This is easily handled: We take "ownership" over the layer, and
            // render directly into that; after all, the user passed a view
            // with an explicit Metal layer to us, so this is very likely what
            // they expect us to do.
            unsafe { Retained::cast(root_layer) }
        } else {
            // The view does not have a `CAMetalLayer` as the root layer (this
            // is the default for most views).
            //
            // This case is trickier! We cannot use the existing layer with
            // Metal, so we must do something else. There are a few options:
            //
            // 1. Panic here, and require the user to pass a view with a
            //    `CAMetalLayer` layer.
            //
            //    While this would "work", it doesn't solve the problem, and
            //    instead passes the ball onwards to the user and ecosystem to
            //    figure it out.
            //
            // 2. Override the existing layer with a newly created layer.
            //
            //    If we overlook that this does not work in UIKit since
            //    `UIView`'s `layer` is `readonly`, and that as such we will
            //    need to do something different there anyhow, this is
            //    actually a fairly good solution, and was what the original
            //    implementation did.
            //
            //    It has some problems though, due to:
            //
            //    a. `wgpu` in our API design choosing not to register a
            //       callback with `-[CALayerDelegate displayLayer:]`, but
            //       instead leaves it up to the user to figure out when to
            //       redraw. That is, we rely on other libraries' callbacks
            //       telling us when to render.
            //
            //       (If this were an API only for Metal, we would probably
            //       make the user provide a `render` closure that we'd call
            //       in the right situations. But alas, we have to be
            //       cross-platform here).
            //
            //    b. Overwriting the `layer` on `NSView` makes the view
            //       "layer-hosting", see [wantsLayer], which disables drawing
            //       functionality on the view like `drawRect:`/`updateLayer`.
            //
            //    These two in combination makes it basically impossible for
            //    crates like Winit to provide a robust rendering callback
            //    that integrates with the system's built-in mechanisms for
            //    redrawing, exactly because overwriting the layer would be
            //    implicitly disabling those mechanisms!
            //
            //    [wantsLayer]: https://developer.apple.com/documentation/appkit/nsview/1483695-wantslayer?language=objc
            //
            // 3. Create a sublayer.
            //
            //    `CALayer` has the concept of "sublayers", which we can use
            //    instead of overriding the layer.
            //
            //    This is also the recommended solution on UIKit, so it's nice
            //    that we can use (almost) the same implementation for these.
            //
            //    It _might_, however, perform ever so slightly worse than
            //    overriding the layer directly.
            //
            // 4. Create a new `MTKView` (or a custom view), and add it as a
            //    subview.
            //
            //    Similar to creating a sublayer (see above), but also
            //    provides a bunch of event handling that we don't need.
            //
            // Option 3 seems like the most robust solution, so this is what
            // we're going to do.

            // Create a new sublayer.
            let new_layer = CAMetalLayer::new();
            root_layer.addSublayer(&new_layer);

            // Automatically resize the sublayer's frame to match the
            // superlayer's bounds.
            //
            // Note that there is a somewhat hidden design decision in this:
            // We define the `width` and `height` in `configure` to control
            // the `drawableSize` of the layer, while `bounds` and `frame` are
            // outside of the user's direct control - instead, though, they
            // can control the size of the view (or root layer), and get the
            // desired effect that way.
            //
            // We _could_ also let `configure` set the `bounds` size, however
            // that would be inconsistent with using the root layer directly
            // (as we may do, see above).
            new_layer.setAutoresizingMask(
                CAAutoresizingMask::kCALayerWidthSizable
                    | CAAutoresizingMask::kCALayerHeightSizable,
            );

            // Specify the relative size that the auto resizing mask above
            // will keep (i.e. tell it to fill out its superlayer).
            let frame = root_layer.bounds();
            new_layer.setFrame(frame);

            // The gravity to use when the layer's `drawableSize` isn't the
            // same as the bounds rectangle.
            //
            // The desired content gravity is `kCAGravityResize`, because it
            // masks / alleviates issues with resizing when
            // `present_with_transaction` is disabled, and behaves better when
            // moving the window between monitors.
            //
            // Unfortunately, it also makes it harder to see changes to
            // `width` and `height` in `configure`. When debugging resize
            // issues, swap this for `kCAGravityTopLeft` instead.
            new_layer.setContentsGravity(unsafe { kCAGravityResize });

            // Set initial scale factor of the layer. This is kept in sync by
            // `configure` (on UIKit), and the delegate below (on AppKit).
            let scale_factor = root_layer.contentsScale();
            new_layer.setContentsScale(scale_factor);

            let delegate = HalManagedMetalLayerDelegate::new();
            new_layer.setDelegate(std::mem::transmute(delegate.0));

            new_layer
        }
    }

    pub(super) fn dimensions(&self) -> wgt::Extent3d {
        let (size, scale): (CGSize, CGFloat) = {
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
            wgt::CompositeAlphaMode::Opaque => render_layer.setOpaque(true),
            wgt::CompositeAlphaMode::PostMultiplied => render_layer.setOpaque(false),
            _ => (),
        }

        // AppKit / UIKit automatically sets the correct scale factor for
        // layers attached to a view. Our layer, however, may not be directly
        // attached to a view; in those cases, we need to set the scale
        // factor ourselves.
        //
        // For AppKit, we do so by adding a delegate on the layer with the
        // `layer:shouldInheritContentsScale:fromWindow:` method returning
        // `true` - this tells the system to automatically update the scale
        // factor when it changes.
        //
        // For UIKit, we manually update the scale factor from the super layer
        // here, if there is one.
        //
        // TODO: Is there a way that we could listen to such changes instead?
        #[cfg(not(target_os = "macos"))]
        {
            if let Some(superlayer) = render_layer.superlayer() {
                let scale_factor = superlayer.contentsScale();
                render_layer.setContentsScale(scale_factor);
            }
        }

        let device_raw = device.shared.device.lock();
        render_layer.setDevice(Some(&device_raw));
        render_layer.setPixelFormat(caps.map_format(config.format));
        render_layer.setFramebufferOnly(framebuffer_only);
        render_layer.setPresentsWithTransaction(self.present_with_transaction);
        // opt-in to Metal EDR
        // EDR potentially more power used in display and more bandwidth, memory footprint.
        let wants_edr = config.format == wgt::TextureFormat::Rgba16Float;
        if wants_edr != render_layer.wantsExtendedDynamicRangeContent() {
            render_layer.setWantsExtendedDynamicRangeContent(wants_edr);
        }

        // this gets ignored on iOS for certain OS/device combinations (iphone5s iOS 10.3)
        render_layer.setMaximumDrawableCount(config.maximum_frame_latency as usize + 1);
        render_layer.setDrawableSize(drawable_size);
        if caps.can_set_next_drawable_timeout {
            render_layer.setAllowsNextDrawableTimeout(false);
        }
        if caps.can_set_display_sync {
            render_layer.setDisplaySyncEnabled(display_sync);
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
        let (drawable, texture) = match autoreleasepool(|_| {
            render_layer
                .nextDrawable()
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
                raw_type: MTLTextureType::MTLTextureType2D,
                array_layers: 1,
                mip_levels: 1,
                copy_size: crate::CopyExtent {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
            },
            drawable: ProtocolObject::from_retained(drawable),
            present_with_transaction: self.present_with_transaction,
        };

        Ok(Some(crate::AcquiredSurfaceTexture {
            texture: suf_texture,
            suboptimal: false,
        }))
    }

    unsafe fn discard_texture(&self, _texture: super::SurfaceTexture) {}
}
