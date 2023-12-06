use wgt::Backend;

/// The `AnySurface` type: a `Arc` of a `HalSurface<A>` for any backend `A`.
use crate::hal_api::HalApi;
use crate::instance::HalSurface;

use std::any::Any;
use std::fmt;
use std::sync::Arc;

/// A `Arc` of a `HalSurface<A>`, for any backend `A`.
///
/// Any `AnySurface` is just like an `Arc<HalSurface<A>>`, except that the
/// `A` type parameter is erased. To access the `Surface`, you must
/// downcast to a particular backend with the \[`downcast_ref`\] or
/// \[`take`\] methods.
pub struct AnySurface(Arc<dyn Any + 'static>);

impl AnySurface {
    /// Return an `AnySurface` that holds an owning `Arc` to `HalSurface`.
    pub fn new<A: HalApi>(surface: HalSurface<A>) -> AnySurface {
        AnySurface(Arc::new(surface))
    }

    pub fn backend(&self) -> Backend {
        #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
        if self.downcast_ref::<hal::api::Vulkan>().is_some() {
            return Backend::Vulkan;
        }
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        if self.downcast_ref::<hal::api::Metal>().is_some() {
            return Backend::Metal;
        }
        #[cfg(all(feature = "dx12", windows))]
        if self.downcast_ref::<hal::api::Dx12>().is_some() {
            return Backend::Dx12;
        }
        #[cfg(feature = "gles")]
        if self.downcast_ref::<hal::api::Gles>().is_some() {
            return Backend::Gl;
        }
        Backend::Empty
    }

    /// If `self` is an `Arc<HalSurface<A>>`, return a reference to the
    /// HalSurface.
    pub fn downcast_ref<A: HalApi>(&self) -> Option<&HalSurface<A>> {
        self.0.downcast_ref::<HalSurface<A>>()
    }

    /// If `self` is an `Arc<HalSurface<A>>`, returns that.
    pub fn take<A: HalApi>(self) -> Option<Arc<HalSurface<A>>> {
        // `Arc::downcast` returns `Arc<T>`, but requires that `T` be `Sync` and
        // `Send`, and this is not the case for `HalSurface` in wasm builds.
        //
        // But as far as I can see, `Arc::downcast` has no particular reason to
        // require that `T` be `Sync` and `Send`; the steps used here are sound.
        if (self.0).is::<HalSurface<A>>() {
            // Turn the `Arc`, which is a pointer to an `ArcInner` struct, into
            // a pointer to the `ArcInner`'s `data` field. Carry along the
            // vtable from the original `Arc`.
            let raw_erased: *const (dyn Any + 'static) = Arc::into_raw(self.0);
            // Remove the vtable, and supply the concrete type of the `data`.
            let raw_typed: *const HalSurface<A> = raw_erased.cast::<HalSurface<A>>();
            // Convert the pointer to the `data` field back into a pointer to
            // the `ArcInner`, and restore reference-counting behavior.
            let arc_typed: Arc<HalSurface<A>> = unsafe {
                // Safety:
                // - We checked that the `dyn Any` was indeed a `HalSurface<A>` above.
                // - We're calling `Arc::from_raw` on the same pointer returned
                //   by `Arc::into_raw`, except that we stripped off the vtable
                //   pointer.
                // - The pointer must still be live, because we've borrowed `self`,
                //   which holds another reference to it.
                // - The format of a `ArcInner<dyn Any>` must be the same as
                //   that of an `ArcInner<HalSurface<A>>`, or else `AnyHalSurface::new`
                //   wouldn't be possible.
                Arc::from_raw(raw_typed)
            };
            Some(arc_typed)
        } else {
            None
        }
    }
}

impl fmt::Debug for AnySurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AnySurface")
    }
}

#[cfg(any(
    not(target_arch = "wasm32"),
    all(
        feature = "fragile-send-sync-non-atomic-wasm",
        not(target_feature = "atomics")
    )
))]
unsafe impl Send for AnySurface {}
#[cfg(any(
    not(target_arch = "wasm32"),
    all(
        feature = "fragile-send-sync-non-atomic-wasm",
        not(target_feature = "atomics")
    )
))]
unsafe impl Sync for AnySurface {}
