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
    runtime::{Class, Object, Sel, BOOL, YES},
    sel, sel_impl,
};
use parking_lot::Mutex;

#[cfg(target_os = "macos")]
#[link(name = "QuartzCore", kind = "framework")]
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
    fn new(view: Option<NonNull<Object>>, layer: mtl::MetalLayer) -> Self {
        Self {
            view,
            render_layer: Mutex::new(layer),
            raw_swapchain_format: mtl::MTLPixelFormat::Invalid,
            extent: wgt::Extent3d::default(),
            main_thread_id: thread::current().id(),
            present_with_transaction: false,
        }
    }

    pub unsafe fn dispose(self) {
        if let Some(view) = self.view {
            let () = msg_send![view.as_ptr(), release];
        }
    }

    #[cfg(target_os = "ios")]
    #[allow(clippy::transmute_ptr_to_ref)]
    pub unsafe fn from_uiview(uiview: *mut c_void) -> Self {
        let view = uiview as *mut Object;
        if view.is_null() {
            panic!("window does not have a valid contentView");
        }

        let main_layer: *mut Object = msg_send![view, layer];
        let class = class!(CAMetalLayer);
        let is_valid_layer: BOOL = msg_send![main_layer, isKindOfClass: class];
        let render_layer = if is_valid_layer == YES {
            mem::transmute::<_, &mtl::MetalLayerRef>(main_layer).to_owned()
        } else {
            // If the main layer is not a CAMetalLayer, we create a CAMetalLayer sublayer and use it instead.
            // Unlike on macOS, we cannot replace the main view as UIView does not allow it (when NSView does).
            let new_layer: mtl::MetalLayer = msg_send![class, new];
            let bounds: CGRect = msg_send![main_layer, bounds];
            let () = msg_send![new_layer.as_ref(), setFrame: bounds];
            let () = msg_send![main_layer, addSublayer: new_layer.as_ref()];
            new_layer
        };

        let window: *mut Object = msg_send![view, window];
        if !window.is_null() {
            let screen: *mut Object = msg_send![window, screen];
            assert!(!screen.is_null(), "window is not attached to a screen");

            let scale_factor: CGFloat = msg_send![screen, nativeScale];
            let () = msg_send![view, setContentScaleFactor: scale_factor];
        }

        let _: *mut c_void = msg_send![view, retain];
        Self::new(NonNull::new(view), render_layer)
    }

    #[cfg(target_os = "macos")]
    #[allow(clippy::transmute_ptr_to_ref)]
    pub unsafe fn from_nsview(
        nsview: *mut c_void,
        delegate: &HalManagedMetalLayerDelegate,
    ) -> Self {
        let view = nsview as *mut Object;
        if view.is_null() {
            panic!("window does not have a valid contentView");
        }

        let class = class!(CAMetalLayer);
        // Deprecated! Clients should use `create_surface_from_layer` instead.
        let is_actually_layer: BOOL = msg_send![view, isKindOfClass: class];
        if is_actually_layer == YES {
            return Self::from_layer(mem::transmute(view));
        }

        let existing: *mut Object = msg_send![view, layer];
        let use_current = if existing.is_null() {
            false
        } else {
            let result: BOOL = msg_send![existing, isKindOfClass: class];
            result == YES
        };

        let render_layer: mtl::MetalLayer = if use_current {
            mem::transmute::<_, &mtl::MetalLayerRef>(existing).to_owned()
        } else {
            let layer: mtl::MetalLayer = msg_send![class, new];
            let () = msg_send![view, setLayer: layer.as_ref()];
            let () = msg_send![view, setWantsLayer: YES];
            let bounds: CGRect = msg_send![view, bounds];
            let () = msg_send![layer.as_ref(), setBounds: bounds];

            let window: *mut Object = msg_send![view, window];
            if !window.is_null() {
                let scale_factor: CGFloat = msg_send![window, backingScaleFactor];
                let () = msg_send![layer, setContentsScale: scale_factor];
            }
            let () = msg_send![layer, setDelegate: delegate.0];
            layer
        };

        let () = msg_send![render_layer, setContentsGravity: kCAGravityTopLeft];

        let _: *mut c_void = msg_send![view, retain];
        Self::new(NonNull::new(view), render_layer)
    }

    pub unsafe fn from_layer(layer: &mtl::MetalLayerRef) -> Self {
        let class = class!(CAMetalLayer);
        let proper_kind: BOOL = msg_send![layer, isKindOfClass: class];
        assert_eq!(proper_kind, YES);
        Self::new(None, layer.to_owned())
    }

    pub(super) fn dimensions(&self) -> wgt::Extent3d {
        let (size, scale): (CGSize, CGFloat) = match self.view {
            Some(view) if !cfg!(target_os = "macos") => unsafe {
                let bounds: CGRect = msg_send![view.as_ptr(), bounds];
                let window: Option<NonNull<Object>> = msg_send![view.as_ptr(), window];
                let screen = window.and_then(|window| -> Option<NonNull<Object>> {
                    msg_send![window.as_ptr(), screen]
                });
                match screen {
                    Some(screen) => {
                        let screen_space: *mut Object = msg_send![screen.as_ptr(), coordinateSpace];
                        let rect: CGRect = msg_send![view.as_ptr(), convertRect:bounds toCoordinateSpace:screen_space];
                        let scale_factor: CGFloat = msg_send![screen.as_ptr(), nativeScale];
                        (rect.size, scale_factor)
                    }
                    None => (bounds.size, 1.0),
                }
            },
            _ => unsafe {
                let render_layer_borrow = self.render_layer.lock();
                let render_layer = render_layer_borrow.as_ref();
                let bounds: CGRect = msg_send![render_layer, bounds];
                let contents_scale: CGFloat = msg_send![render_layer, contentsScale];
                (bounds.size, contents_scale)
            },
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
        &mut self,
        device: &super::Device,
        config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        log::info!("build swapchain {:?}", config);

        let caps = &device.shared.private_caps;
        self.raw_swapchain_format = caps.map_format(config.format);
        self.extent = config.extent;

        let render_layer = self.render_layer.lock();
        let framebuffer_only = config.usage == crate::TextureUses::COLOR_TARGET;
        let display_sync = config.present_mode != wgt::PresentMode::Immediate;
        let drawable_size = CGSize::new(config.extent.width as f64, config.extent.height as f64);

        match config.composite_alpha_mode {
            crate::CompositeAlphaMode::Opaque => render_layer.set_opaque(true),
            crate::CompositeAlphaMode::PostMultiplied => render_layer.set_opaque(false),
            crate::CompositeAlphaMode::PreMultiplied => (),
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
        render_layer.set_device(&*device_raw);
        render_layer.set_pixel_format(self.raw_swapchain_format);
        render_layer.set_framebuffer_only(framebuffer_only);
        render_layer.set_presents_with_transaction(self.present_with_transaction);

        // this gets ignored on iOS for certain OS/device combinations (iphone5s iOS 10.3)
        let () = msg_send![*render_layer, setMaximumDrawableCount: config.swap_chain_size as u64];

        render_layer.set_drawable_size(drawable_size);
        if caps.can_set_next_drawable_timeout {
            let () = msg_send![*render_layer, setAllowsNextDrawableTimeout:false];
        }
        if caps.can_set_display_sync {
            let () = msg_send![*render_layer, setDisplaySyncEnabled: display_sync];
        }

        Ok(())
    }

    unsafe fn unconfigure(&mut self, _device: &super::Device) {
        self.raw_swapchain_format = mtl::MTLPixelFormat::Invalid;
    }

    unsafe fn acquire_texture(
        &mut self,
        _timeout_ms: u32, //TODO
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

        let suf_texture = super::SurfaceTexture {
            texture: super::Texture {
                raw: texture,
                raw_format: self.raw_swapchain_format,
                raw_type: mtl::MTLTextureType::D2,
                array_layers: 1,
                mip_levels: 1,
                copy_size: crate::CopyExtent {
                    width: self.extent.width,
                    height: self.extent.height,
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

    unsafe fn discard_texture(&mut self, _texture: super::SurfaceTexture) {}
}
