#![allow(clippy::let_unit_value)] // `let () =` being used to constrain result type

use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::sync::Once;
use std::thread;

use core_graphics_types::{
    base::CGFloat,
    geometry::{CGRect, CGSize},
};
use metal::foreign_types::ForeignType;
use objc::{
    class,
    declare::ClassDecl,
    msg_send,
    rc::{autoreleasepool, StrongPtr},
    runtime::{Class, Object, Sel, BOOL, NO, YES},
    sel, sel_impl,
};
use parking_lot::{Mutex, RwLock};

#[link(name = "QuartzCore", kind = "framework")]
extern "C" {
    #[allow(non_upper_case_globals)]
    static kCAGravityResize: *mut Object;
}

extern "C" fn layer_should_inherit_contents_scale_from_window(
    _: &Class,
    _: Sel,
    _layer: *mut Object,
    _new_scale: CGFloat,
    _from_window: *mut Object,
) -> BOOL {
    YES
}

static CAML_DELEGATE_REGISTER: Once = Once::new();

#[derive(Debug)]
pub struct HalManagedMetalLayerDelegate(&'static Class);

impl HalManagedMetalLayerDelegate {
    pub fn new() -> Self {
        let class_name = format!("HalManagedMetalLayerDelegate@{:p}", &CAML_DELEGATE_REGISTER);

        CAML_DELEGATE_REGISTER.call_once(|| {
            type Fun = extern "C" fn(&Class, Sel, *mut Object, CGFloat, *mut Object) -> BOOL;
            let mut decl = ClassDecl::new(&class_name, class!(NSObject)).unwrap();
            unsafe {
                // <https://developer.apple.com/documentation/appkit/nsviewlayercontentscaledelegate/3005294-layer?language=objc>
                decl.add_class_method::<Fun>(
                    sel!(layer:shouldInheritContentsScale:fromWindow:),
                    layer_should_inherit_contents_scale_from_window,
                );
            }
            decl.register();
        });
        Self(Class::get(&class_name).unwrap())
    }
}

impl super::Surface {
    fn new(layer: metal::MetalLayer) -> Self {
        Self {
            render_layer: Mutex::new(layer),
            swapchain_format: RwLock::new(None),
            extent: RwLock::new(wgt::Extent3d::default()),
            main_thread_id: thread::current().id(),
            present_with_transaction: false,
        }
    }

    /// If not called on the main thread, this will panic.
    #[allow(clippy::transmute_ptr_to_ref)]
    pub unsafe fn from_view(view: NonNull<Object>) -> Self {
        let layer = unsafe { Self::get_metal_layer(view) };
        let layer = ManuallyDrop::new(layer);
        // SAFETY: The layer is an initialized instance of `CAMetalLayer`, and
        // we transfer the retain count to `MetalLayer` using `ManuallyDrop`.
        let layer = unsafe { metal::MetalLayer::from_ptr(layer.cast()) };
        Self::new(layer)
    }

    pub unsafe fn from_layer(layer: &metal::MetalLayerRef) -> Self {
        let class = class!(CAMetalLayer);
        let proper_kind: BOOL = msg_send![layer, isKindOfClass: class];
        assert_eq!(proper_kind, YES);
        Self::new(layer.to_owned())
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
    pub(crate) unsafe fn get_metal_layer(view: NonNull<Object>) -> StrongPtr {
        let is_main_thread: BOOL = msg_send![class!(NSThread), isMainThread];
        if is_main_thread == NO {
            panic!("get_metal_layer cannot be called in non-ui thread.");
        }

        // Ensure that the view is layer-backed.
        // Views are always layer-backed in UIKit.
        #[cfg(target_os = "macos")]
        let () = msg_send![view.as_ptr(), setWantsLayer: YES];

        let root_layer: *mut Object = msg_send![view.as_ptr(), layer];
        // `-[NSView layer]` can return `NULL`, while `-[UIView layer]` should
        // always be available.
        assert!(!root_layer.is_null(), "failed making the view layer-backed");

        // NOTE: We explicitly do not touch properties such as
        // `layerContentsPlacement`, `needsDisplayOnBoundsChange` and
        // `contentsGravity` etc. on the root layer, both since we would like
        // to give the user full control over them, and because the default
        // values suit us pretty well (especially the contents placement being
        // `NSViewLayerContentsRedrawDuringViewResize`, which allows the view
        // to receive `drawRect:`/`updateLayer` calls).

        let is_metal_layer: BOOL = msg_send![root_layer, isKindOfClass: class!(CAMetalLayer)];
        if is_metal_layer == YES {
            // The view has a `CAMetalLayer` as the root layer, which can
            // happen for example if user overwrote `-[NSView layerClass]` or
            // the view is `MTKView`.
            //
            // This is easily handled: We take "ownership" over the layer, and
            // render directly into that; after all, the user passed a view
            // with an explicit Metal layer to us, so this is very likely what
            // they expect us to do.
            unsafe { StrongPtr::retain(root_layer) }
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
            let new_layer: *mut Object = msg_send![class!(CAMetalLayer), new];
            let () = msg_send![root_layer, addSublayer: new_layer];

            #[cfg(target_os = "macos")]
            {
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
                let width_sizable = 1 << 1; // kCALayerWidthSizable
                let height_sizable = 1 << 4; // kCALayerHeightSizable
                let mask: std::ffi::c_uint = width_sizable | height_sizable;
                let () = msg_send![new_layer, setAutoresizingMask: mask];
            }

            // Specify the relative size that the auto resizing mask above
            // will keep (i.e. tell it to fill out its superlayer).
            let frame: CGRect = msg_send![root_layer, bounds];
            let () = msg_send![new_layer, setFrame: frame];

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
            let _: () = msg_send![new_layer, setContentsGravity: unsafe { kCAGravityResize }];

            // Set initial scale factor of the layer. This is kept in sync by
            // `configure` (on UIKit), and the delegate below (on AppKit).
            let scale_factor: CGFloat = msg_send![root_layer, contentsScale];
            let () = msg_send![new_layer, setContentsScale: scale_factor];

            let delegate = HalManagedMetalLayerDelegate::new();
            let () = msg_send![new_layer, setDelegate: delegate.0];

            unsafe { StrongPtr::new(new_layer) }
        }
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
            let superlayer: *mut Object = msg_send![render_layer.as_ptr(), superlayer];
            if !superlayer.is_null() {
                let scale_factor: CGFloat = msg_send![superlayer, contentsScale];
                let () = msg_send![render_layer.as_ptr(), setContentsScale: scale_factor];
            }
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
