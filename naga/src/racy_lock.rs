#![cfg_attr(
    not(any(hlsl_out, msl_out, wgsl_out, glsl_out)),
    allow(
        dead_code,
        reason = "RacyLock is only required for the above configurations"
    )
)]

use alloc::boxed::Box;
use core::sync::atomic::{AtomicPtr, Ordering};

/// An alternative to [`LazyLock`] which will race to initialize rather than blocking.
/// This makes it suitable for `no_std` environments, at the expense of possibly leaking
/// memory during initialization.
///
/// [`LazyLock`]: https://doc.rust-lang.org/stable/std/sync/struct.LazyLock.html
pub struct RacyLock<T: 'static> {
    inner: AtomicPtr<T>,
    init: fn() -> T,
}

impl<T: 'static> RacyLock<T> {
    /// Creates a new [`RacyLock`], which will initialize using the provided `init` function.
    pub const fn new(init: fn() -> T) -> Self {
        Self {
            inner: AtomicPtr::new(core::ptr::null_mut()),
            init,
        }
    }

    /// Attempts to load the internal value, returning [`None`] if it is not yet initialized.
    pub fn try_get(&self) -> Option<&T> {
        let ptr = self.inner.load(Ordering::Acquire);

        if ptr.is_null() {
            None
        } else {
            // SAFETY: ptr can only ever be null, or a static-valid value from Box::leak,
            // as it is private.
            // The above check ensures ptr is not null, so it must be a valid pointer.
            unsafe { Some(&*ptr) }
        }
    }

    /// Loads the internal value, initializing it if required.
    pub fn get(&self) -> &T {
        self.try_get().unwrap_or_else(|| {
            let value = (self.init)();

            // Refresh the static value just before leaking to minimize leaked memory.
            let ptr = self.inner.load(Ordering::Acquire);

            if ptr.is_null() {
                // Explicit type used to assert the returned reference is 'static.
                let ptr: &'static mut T = Box::leak(Box::new(value));

                self.inner.store(ptr, Ordering::Release);

                ptr
            } else {
                // SAFETY: ptr can only ever be null, or a static-valid value from Box::leak,
                // as it is private.
                // The above check ensures ptr is not null, so it must be a valid pointer.
                unsafe { &*ptr }
            }
        })
    }
}

impl<T: 'static> core::ops::Deref for RacyLock<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}
