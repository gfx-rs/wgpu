cfg_if::cfg_if! {
    if #[cfg(feature = "std")] {
        type RawRwLockInner = parking_lot::RawRwLock;
    } else {
        type RawRwLockInner = core::cell::Cell<isize>;

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

        #[repr(isize)]
        enum BorrowCount {
            LockedExclusive = -1,
            Unlocked = 0,
            SingleLockedShared = 1,
        }
    }
}

/// Raw implementation for a [`lock_api::RwLock`].
///
/// This will delegate to [`parking_lot`] if the `std` feature is enabled (which
/// it is by default). Otherwise, it will provide a `!Sync` implementation
/// similar to [`RefCell`].
///
/// [`parking_lot`]: https://docs.rs/parking_lot/
/// [`RefCell`]: core::cell::RefCell
pub struct RawRwLock(RawRwLockInner);

impl RawRwLock {
    /// Constructs a new [`RawRwLock`].
    pub const fn new() -> Self {
        Self({
            cfg_if::cfg_if! {
                if #[cfg(feature = "std")] {
                    lock_api::RawRwLock::INIT
                } else {
                    RawRwLockInner::new(BorrowCount::Unlocked as _)
                }
            }
        })
    }
}

impl Default for RawRwLock {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for RawRwLock {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("RawRwLock").finish_non_exhaustive()
    }
}

// SAFETY:
//
// # With `std`
//
// This implementation directly delegates to an existing implementation of
// `RawRwLock`, and is therefore safe.
//
// # Without `std`
//
// This implementation tracks the number of borrows using an `isize`, where `-1`
// indicates a single _exclusive_ lock, and a positive number represents that many
// shared locks.
unsafe impl lock_api::RawRwLock for RawRwLock {
    type GuardMarker = lock_api::GuardNoSend;

    const INIT: RawRwLock = RawRwLock::new();

    #[inline]
    fn lock_exclusive(&self) {
        cfg_if::cfg_if! {
            if #[cfg(feature = "std")] {
                lock_api::RawRwLock::lock_exclusive(&self.0)
            } else {
                if !self.try_lock_exclusive() {
                    // Since this "lock" is `!Sync`, any failing attempt to lock it
                    // must be from the same thread, which means a deadlock.
                    deadlock()
                }
            }
        }
    }

    #[inline]
    fn try_lock_exclusive(&self) -> bool {
        cfg_if::cfg_if! {
            if #[cfg(feature = "std")] {
                lock_api::RawRwLock::try_lock_exclusive(&self.0)
            } else {
                if self.0.get() != BorrowCount::Unlocked as _ {
                    false
                } else {
                    self.0.set(BorrowCount::LockedExclusive as _);
                    true
                }
            }
        }
    }

    #[inline]
    unsafe fn unlock_exclusive(&self) {
        cfg_if::cfg_if! {
            if #[cfg(feature = "std")] {
                // SAFETY: directly delegating to an accepted implementation
                unsafe { lock_api::RawRwLock::unlock_exclusive(&self.0) }
            } else {
                self.0.set(BorrowCount::Unlocked as _);
            }
        }
    }

    #[inline]
    fn lock_shared(&self) {
        cfg_if::cfg_if! {
            if #[cfg(feature = "std")] {
                lock_api::RawRwLock::lock_shared(&self.0)
            } else {
                if !self.try_lock_shared() {
                    // Since this "lock" is `!Sync`, any failing attempt to lock it
                    // must be from the same thread, which means a deadlock.
                    deadlock()
                }
            }
        }
    }

    #[inline]
    fn try_lock_shared(&self) -> bool {
        cfg_if::cfg_if! {
            if #[cfg(feature = "std")] {
                lock_api::RawRwLock::try_lock_shared(&self.0)
            } else {
                if self.0.get() == BorrowCount::LockedExclusive as _ {
                    false
                } else {
                    match self.0.get().checked_add(1) {
                        Some(value) => {
                            self.0.set(value);
                            true
                        }
                        None => {
                            // Instead of panicking, we can simply fail to lock,
                            // preventing the count for overflowing.
                            false
                        }
                    }
                }
            }
        }
    }

    #[inline]
    unsafe fn unlock_shared(&self) {
        cfg_if::cfg_if! {
            if #[cfg(feature = "std")] {
                // SAFETY: directly delegating to an accepted implementation
                unsafe { lock_api::RawRwLock::unlock_shared(&self.0) }
            } else {
                match self.0.get().checked_sub(1) {
                    Some(value) => {
                        // It is a safety condition of `RawRwLock::unlock_shared` that the caller
                        // has already determined the lock is held in the shared state (`> 0`).
                        debug_assert!(!value.is_negative(), "caller violated safety condition");
                        self.0.set(value);
                    }
                    None => {
                        unreachable!("lock state should never underflow");
                    }
                }
            }
        }
    }

    #[inline]
    fn is_locked(&self) -> bool {
        cfg_if::cfg_if! {
            if #[cfg(feature = "std")] {
                lock_api::RawRwLock::is_locked(&self.0)
            } else {
                self.0.get() != BorrowCount::Unlocked as _
            }
        }
    }

    #[inline]
    fn is_locked_exclusive(&self) -> bool {
        cfg_if::cfg_if! {
            if #[cfg(feature = "std")] {
                lock_api::RawRwLock::is_locked_exclusive(&self.0)
            } else {
                self.0.get() == BorrowCount::LockedExclusive as _
            }
        }
    }
}

// SAFETY:
//
// # With `std`
//
// This implementation directly delegates to an existing implementation of
// `RawRwLockDowngrade`, and is therefore safe.
//
// # Without `std`
//
// It's a safety condition on the caller of `downgrade` that they already have an
// exclusive lock, so it is sufficient to set the count to `1` to change the state
// of the reader-writer lock to shared with one reader.
unsafe impl lock_api::RawRwLockDowngrade for RawRwLock {
    unsafe fn downgrade(&self) {
        cfg_if::cfg_if! {
            if #[cfg(feature = "std")] {
                // SAFETY: directly delegating to an accepted implementation
                unsafe { lock_api::RawRwLockDowngrade::downgrade(&self.0) }
            } else {
                self.0.set(BorrowCount::SingleLockedShared as _);
            }
        }
    }
}
