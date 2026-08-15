cfg_if::cfg_if! {
    if #[cfg(feature = "std")] {
        type RawMutexInner = parking_lot::RawMutex;
    } else {
        type RawMutexInner = core::cell::Cell<bool>;

        /// When a `no_std` locking primitive is under contention, the "correct" way to
        /// handle it would be to spin until the lock is available. This is because
        /// without `std` there is no standard way to yield/block the current thread.
        /// However, since we only support `no_std` locks that aren't `Sync`, we know
        /// that only one thread can access the lock at a time. Therefore, we know this
        /// is actually a deadlock and will never resolve. We choose to panic in these
        /// cases to highlight what is almost certainly an internal bug.
        fn deadlock() -> ! {
            panic!("a locking primitive in wgpu is currently deadlocked");
        }
    }
}

/// Raw implementation for a [`lock_api::Mutex`].
///
/// This will delegate to [`parking_lot`] if the `std` feature is enabled (which
/// it is by default). Otherwise, it will provide a `!Sync` implementation
/// similar to [`RefCell`].
///
/// [`parking_lot`]: https://docs.rs/parking_lot/
/// [`RefCell`]: core::cell::RefCell
pub struct RawMutex(RawMutexInner);

impl RawMutex {
    /// Constructs a new [`RawMutex`].
    pub const fn new() -> Self {
        Self({
            cfg_if::cfg_if! {
                if #[cfg(feature = "std")] {
                    lock_api::RawMutex::INIT
                } else {
                    RawMutexInner::new(false)
                }
            }
        })
    }
}

impl Default for RawMutex {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for RawMutex {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("RawMutex").finish_non_exhaustive()
    }
}

// SAFETY:
//
// # With `std`
//
// This implementation directly delegates to an existing implementation of
// `RawMutex`, and is therefore safe.
//
// # Without `std`
//
// This implementation tracks the state of the mutex in a boolean, where `false`
// indicates it is unlocked, and `true` indicates it is locked. `is_locked`
// directly returns this state, and only `try_lock` and `unlock` are able to
// modify it. Both methods ensure the state of the lock is sound.
unsafe impl lock_api::RawMutex for RawMutex {
    type GuardMarker = lock_api::GuardNoSend;

    const INIT: RawMutex = RawMutex::new();

    #[inline]
    fn lock(&self) {
        cfg_if::cfg_if! {
            if #[cfg(feature = "std")] {
                lock_api::RawMutex::lock(&self.0)
            } else {
                if !self.try_lock() {
                    // Since this "mutex" is `!Sync`, any attempt to lock it twice
                    // must be from the same thread, which means a deadlock.
                    deadlock()
                }
            }
        }
    }

    #[inline]
    fn try_lock(&self) -> bool {
        cfg_if::cfg_if! {
            if #[cfg(feature = "std")] {
                lock_api::RawMutex::try_lock(&self.0)
            } else {
                !self.0.replace(true)
            }
        }
    }

    #[inline]
    unsafe fn unlock(&self) {
        cfg_if::cfg_if! {
            if #[cfg(feature = "std")] {
                // SAFETY: directly delegating to an accepted implementation
                unsafe { lock_api::RawMutex::unlock(&self.0) }
            } else {
                self.0.set(false);
            }
        }
    }

    #[inline]
    fn is_locked(&self) -> bool {
        cfg_if::cfg_if! {
            if #[cfg(feature = "std")] {
                lock_api::RawMutex::is_locked(&self.0)
            } else {
                self.0.get()
            }
        }
    }
}

// SAFETY:
//
// # With `std`
//
// This implementation directly delegates to an existing implementation of
// `RawMutexTimed`, and is therefore safe.
#[cfg(feature = "std")]
unsafe impl lock_api::RawMutexTimed for RawMutex {
    type Duration = core::time::Duration;
    type Instant = <RawMutexInner as lock_api::RawMutexTimed>::Instant;

    fn try_lock_for(&self, timeout: Self::Duration) -> bool {
        lock_api::RawMutexTimed::try_lock_for(&self.0, timeout)
    }

    fn try_lock_until(&self, timeout: Self::Instant) -> bool {
        lock_api::RawMutexTimed::try_lock_until(&self.0, timeout)
    }
}
