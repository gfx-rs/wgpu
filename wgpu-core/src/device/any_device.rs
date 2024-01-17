use super::Device;
/// The `AnyDevice` type: a pointer to a `Device<A>` for any backend `A`.
use crate::hal_api::HalApi;

use std::any::Any;
use std::fmt;
use std::sync::Arc;

/// A pointer to a `Device<A>`, for any backend `A`.
///
/// Any `AnyDevice` is just like an `Arc<Device<A>>`, except that the
/// `A` type parameter is erased. To access the `Device`, you must
/// downcast to a particular backend with the \[`downcast_ref`\] or
/// \[`downcast_clone`\] methods.
pub struct AnyDevice(Arc<dyn Any + 'static>);

impl AnyDevice {
    /// Return an `AnyDevice` that holds an owning `Arc` pointer to `device`.
    pub fn new<A: HalApi>(device: Arc<Device<A>>) -> AnyDevice {
        AnyDevice(device)
    }

    /// If `self` is an `Arc<Device<A>>`, return a reference to the
    /// device.
    pub fn downcast_ref<A: HalApi>(&self) -> Option<&Device<A>> {
        self.0.downcast_ref::<Device<A>>()
    }

    /// If `self` is an `Arc<Device<A>>`, return a clone of that.
    pub fn downcast_clone<A: HalApi>(&self) -> Option<Arc<Device<A>>> {
        // `Arc::downcast` returns `Arc<T>`, but requires that `T` be `Sync` and
        // `Send`, and this is not the case for `Device` in wasm builds.
        //
        // But as far as I can see, `Arc::downcast` has no particular reason to
        // require that `T` be `Sync` and `Send`; the steps used here are sound.
        if (self.0).is::<Device<A>>() {
            // Get an owned Arc.
            let clone = self.0.clone();
            // Turn the `Arc`, which is a pointer to an `ArcInner` struct, into
            // a pointer to the `ArcInner`'s `data` field. Carry along the
            // vtable from the original `Arc`.
            let raw_erased: *const (dyn Any + 'static) = Arc::into_raw(clone);
            // Remove the vtable, and supply the concrete type of the `data`.
            let raw_typed: *const Device<A> = raw_erased.cast::<Device<A>>();
            // Convert the pointer to the `data` field back into a pointer to
            // the `ArcInner`, and restore reference-counting behavior.
            let arc_typed: Arc<Device<A>> = unsafe {
                // Safety:
                // - We checked that the `dyn Any` was indeed a `Device<A>` above.
                // - We're calling `Arc::from_raw` on the same pointer returned
                //   by `Arc::into_raw`, except that we stripped off the vtable
                //   pointer.
                // - The pointer must still be live, because we've borrowed `self`,
                //   which holds another reference to it.
                // - The format of a `ArcInner<dyn Any>` must be the same as
                //   that of an `ArcInner<Device<A>>`, or else `AnyDevice::new`
                //   wouldn't be possible.
                Arc::from_raw(raw_typed)
            };
            Some(arc_typed)
        } else {
            None
        }
    }
}

impl fmt::Debug for AnyDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AnyDevice")
    }
}

#[cfg(send_sync)]
unsafe impl Send for AnyDevice {}
#[cfg(send_sync)]
unsafe impl Sync for AnyDevice {}
