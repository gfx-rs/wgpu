use wgt::Backend;

use super::Device;
/// The `AnyDevice` type: a pointer to a `Device<A>` for any backend `A`.
use crate::hal_api::HalApi;

use std::fmt;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::sync::Arc;

struct AnyDeviceVtable {
    // We oppurtunistically store the backend here, since we now it will be used
    // with backend selection and it can be stored in static memory.
    backend: Backend,
    // Drop glue which knows how to drop the stored data.
    drop: unsafe fn(*mut ()),
}

/// A pointer to a `Device<A>`, for any backend `A`.
///
/// Any `AnyDevice` is just like an `Arc<Device<A>>`, except that the `A` type
/// parameter is erased. To access the `Device`, you must downcast to a
/// particular backend with the \[`downcast_ref`\] or \[`downcast_clone`\]
/// methods.
pub struct AnyDevice {
    data: NonNull<()>,
    vtable: &'static AnyDeviceVtable,
}

impl AnyDevice {
    /// Return an `AnyDevice` that holds an owning `Arc` pointer to `device`.
    pub fn new<A: HalApi>(device: Arc<Device<A>>) -> AnyDevice {
        unsafe fn drop_glue<A: HalApi>(ptr: *mut ()) {
            // Drop the arc this instance is holding.
            unsafe {
                _ = Arc::from_raw(ptr.cast::<A::Device>());
            }
        }

        // SAFETY: The pointer returned by Arc::into_raw is guaranteed to be
        // non-null.
        let data = unsafe { NonNull::new_unchecked(Arc::into_raw(device).cast_mut()) };

        AnyDevice {
            data: data.cast(),
            vtable: &AnyDeviceVtable {
                backend: A::VARIANT,
                drop: drop_glue::<A>,
            },
        }
    }

    /// If `self` is an `Arc<Device<A>>`, return a reference to the
    /// device.
    pub fn downcast_ref<A: HalApi>(&self) -> Option<&Device<A>> {
        if self.vtable.backend != A::VARIANT {
            return None;
        }

        // SAFETY: We just checked the instance above implicitly by the backend
        // that it was statically constructed through.
        Some(unsafe { &*(self.data.as_ptr().cast::<Device<A>>()) })
    }

    /// If `self` is an `Arc<Device<A>>`, return a clone of that.
    pub fn downcast_clone<A: HalApi>(&self) -> Option<Arc<Device<A>>> {
        if self.vtable.backend != A::VARIANT {
            return None;
        }

        // We need to prevent the destructor of the arc from running, since it
        // refers to the instance held by this object. Dropping it would
        // invalidate this object.
        //
        // SAFETY: We just checked the instance above implicitly by the backend
        // that it was statically constructed through.
        let this =
            ManuallyDrop::new(unsafe { Arc::from_raw(self.data.as_ptr().cast::<Device<A>>()) });

        // Cloning it increases the reference count, and we return a new arc
        // instance.
        Some((*this).clone())
    }
}

impl Drop for AnyDevice {
    fn drop(&mut self) {
        unsafe { (self.vtable.drop)(self.data.as_ptr()) }
    }
}

impl fmt::Debug for AnyDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AnyDevice<{}>", self.vtable.backend)
    }
}

#[cfg(send_sync)]
unsafe impl Send for AnyDevice {}
#[cfg(send_sync)]
unsafe impl Sync for AnyDevice {}
