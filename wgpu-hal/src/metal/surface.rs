#![allow(clippy::let_unit_value)] // `let () =` being used to constrain result type

use std::{mem, os::raw::c_void, ptr::NonNull, sync::Once, thread};

use core_graphics_types::{
    base::CGFloat,
    geometry::{CGRect, CGSize},
};
use objc::{
    class,
    declare::ClassDecl,
    msg_send,
    rc::autoreleasepool,
    runtime::{Class, Object, Sel, BOOL, NO, YES},
    sel, sel_impl,
};
use parking_lot::{Mutex, RwLock};

#[cfg(target_os = "macos")]
#[cfg_attr(feature = "link", link(name = "QuartzCore", kind = "framework"))]
extern "C" {
    #[allow(non_upper_case_globals)]
    static kCAGravityTopLeft: *mut Object;
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
            #[allow(trivial_casts)] // false positive
            unsafe {
                decl.add_class_method(
                    sel!(layer:shouldInheritContentsScale:fromWindow:),
                    layer_should_inherit_contents_scale_from_window as Fun,
                );
            }
            decl.register();
        });
        Self(Class::get(&class_name).unwrap())
    }
}

impl super::Surface {
    fn new(view: Option<NonNull<Object>>, layer: metal::MetalLayer) -> Self {
        Self {
            view,
            render_layer: Mutex::new(layer),
            swapchain_format: RwLock::new(None),
            extent: RwLock::new(wgt::Extent3d::default()),
            main_thread_id: thread::current().id(),
            present_with_transaction: false,
        }
    }

    pub unsafe fn dispose(self) {
        if let Some(view) = self.view {
            let () = msg_send![view.as_ptr(), release];
        }
    }

    /// If not called on the main thread, this will panic.
    #[allow(clippy::transmute_ptr_to_ref)]
    pub unsafe fn from_view(
        view: *mut c_void,
        delegate: Option<&HalManagedMetalLayerDelegate>,
    ) -> Self {
        let view = view as *mut Object;
        let render_layer = {
            let layer = unsafe { Self::get_metal_layer(view, delegate) };
            unsafe { mem::transmute::<_, &metal::MetalLayerRef>(layer) }
        }
        .to_owned();
        let _: *mut c_void = msg_send![view, retain];
        Self::new(NonNull::new(view), render_layer)
    }

    pub unsafe fn from_layer(layer: &metal::MetalLayerRef) -> Self {
        let class = class!(CAMetalLayer);
        let proper_kind: BOOL = msg_send![layer, isKindOfClass: class];
        assert_eq!(proper_kind, YES);
        Self::new(None, layer.to_owned())
    }

    /// If not called on the main thread, this will panic.
    pub(crate) unsafe fn get_metal_layer(
        view: *mut Object,
        delegate: Option<&HalManagedMetalLayerDelegate>,
    ) -> *mut Object {
        if view.is_null() {
            panic!("window does not have a valid contentView");
        }

        let is_main_thread: BOOL = msg_send![class!(NSThread), isMainThread];
        if is_main_thread == NO {
            panic!("get_metal_layer cannot be called in non-ui thread.");
        }

        let main_layer: *mut Object = msg_send![view, layer];
        let class = class!(CAMetalLayer);
        let is_valid_layer: BOOL = msg_send![main_layer, isKindOfClass: class];

        if is_valid_layer == YES {
            main_layer
        } else {
            // If the main layer is not a CAMetalLayer, we create a CAMetalLayer and use it.
            let new_layer: *mut Object = msg_send![class, new];
            let frame: CGRect = msg_send![main_layer, bounds];
            let () = msg_send![new_layer, setFrame: frame];
            #[cfg(target_os = "ios")]
            {
                // Unlike NSView, UIView does not allow to replace main layer.
                let () = msg_send![main_layer, addSublayer: new_layer];
                // On iOS, "from_view" may be called before the application initialization is complete,
                // `msg_send![view, window]` and `msg_send![window, screen]` will get null.
                let screen: *mut Object = msg_send![class!(UIScreen), mainScreen];
                let scale_factor: CGFloat = msg_send![screen, nativeScale];
                let () = msg_send![view, setContentScaleFactor: scale_factor];
            };
            #[cfg(target_os = "macos")]
            {
                let () = msg_send![view, setLayer: new_layer];
                let () = msg_send![view, setWantsLayer: YES];
                let () = msg_send![new_layer, setContentsGravity: unsafe { kCAGravityTopLeft }];
                let window: *mut Object = msg_send![view, window];
                if !window.is_null() {
                    let scale_factor: CGFloat = msg_send![window, backingScaleFactor];
                    let () = msg_send![new_layer, setContentsScale: scale_factor];
                }
            };
            if let Some(delegate) = delegate {
                let () = msg_send![new_layer, setDelegate: delegate.0];
            }
            new_layer
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

impl crate::Surface<super::Api> for super::Surface {
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
        // On iOS, unless the user supplies a view with a CAMetalLayer, we
        // create one as a sublayer. However, when the view changes size,
        // its sublayers are not automatically resized, and we must resize
        // it here. The drawable size and the layer size don't correlate
        #[cfg(target_os = "ios")]
        {
            if let Some(view) = self.view {
                let main_layer: *mut Object = msg_send![view.as_ptr(), layer];
                let bounds: CGRect = msg_send![main_layer, bounds];
                let () = msg_send![*render_layer, setFrame: bounds];
            }
        }
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
