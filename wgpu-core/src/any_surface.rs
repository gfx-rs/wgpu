use wgt::Backend;

/// The `AnySurface` type: a `Arc` of a `A::Surface` for any backend `A`.
use crate::hal_api::HalApi;

use std::fmt;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

struct AnySurfaceVtable {
    // We oppurtunistically store the backend here, since we now it will be used
    // with backend selection and it can be stored in static memory.
    backend: Backend,
    // Drop glue which knows how to drop the stored data.
    drop: unsafe fn(*mut ()),
}

/// An `A::Surface`, for any backend `A`.
///
/// Any `AnySurface` is just like an `A::Surface`, except that the `A` type
/// parameter is erased. To access the `Surface`, you must downcast to a
/// particular backend with the \[`downcast_ref`\] or \[`take`\] methods.
pub struct AnySurface {
    data: NonNull<()>,
    vtable: &'static AnySurfaceVtable,
}

impl AnySurface {
    /// Construct an `AnySurface` that owns an `A::Surface`.
    pub fn new<A: HalApi>(surface: A::Surface) -> AnySurface {
        unsafe fn drop_glue<A: HalApi>(ptr: *mut ()) {
            unsafe {
                _ = Box::from_raw(ptr.cast::<A::Surface>());
            }
        }

        let data = NonNull::from(Box::leak(Box::new(surface)));

        AnySurface {
            data: data.cast(),
            vtable: &AnySurfaceVtable {
                backend: A::VARIANT,
                drop: drop_glue::<A>,
            },
        }
    }

    /// Get the backend this surface was created through.
    pub fn backend(&self) -> Backend {
        self.vtable.backend
    }

    /// If `self` refers to an `A::Surface`, returns a reference to it.
    pub fn downcast_ref<A: HalApi>(&self) -> Option<&A::Surface> {
        if A::VARIANT != self.vtable.backend {
            return None;
        }

        // SAFETY: We just checked the instance above implicitly by the backend
        // that it was statically constructed through.
        Some(unsafe { &*self.data.as_ptr().cast::<A::Surface>() })
    }

    /// If `self` is an `Arc<A::Surface>`, returns that.
    pub fn take<A: HalApi>(self) -> Option<A::Surface> {
        if A::VARIANT != self.vtable.backend {
            return None;
        }

        // Disable drop glue, since we're returning the owned surface. The
        // caller will be responsible for dropping it.
        let this = ManuallyDrop::new(self);

        // SAFETY: We just checked the instance above implicitly by the backend
        // that it was statically constructed through.
        Some(unsafe { *Box::from_raw(this.data.as_ptr().cast::<A::Surface>()) })
    }
}

impl Drop for AnySurface {
    fn drop(&mut self) {
        unsafe { (self.vtable.drop)(self.data.as_ptr()) }
    }
}

impl fmt::Debug for AnySurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AnySurface<{}>", self.vtable.backend)
    }
}

#[cfg(send_sync)]
unsafe impl Send for AnySurface {}
#[cfg(send_sync)]
unsafe impl Sync for AnySurface {}
