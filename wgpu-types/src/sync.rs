//! Provides [`Mutex`] and [`RwLock`] types with an appropriate implementation chosen
//! from:
//!
//! 1. [`parking_lot`] (default)
//! 2. [`std`]
//! 3. [`spin`]
//! 4. [`RefCell`](core::cell::RefCell) (fallback)
//!
//! These are ordered by priority.
//! For example if `parking_lot` and `std` are both enabled, `parking_lot` will
//! be used as the implementation.
//!
//! Generally you should use `parking_lot` for the optimal performance, at the
//! expense of reduced target compatibility.
//! In contrast, `spin` provides the best compatibility (e.g., `no_std`) in exchange
//! for potentially worse performance.
//! If no implementation is chosen, [`RefCell`](core::cell::RefCell) will be used
//! as a fallback.
//! Note that the fallback implementation is _not_ [`Sync`] and will [spin](core::hint::spin_loop)
//! when a lock is contested.
//!
//! [`parking_lot`]: https://docs.rs/parking_lot/
//! [`std`]: https://docs.rs/std/
//! [`spin`]: https://docs.rs/std/

use core::{fmt, ops};

cfg_if::cfg_if! {
    if #[cfg(feature = "parking_lot")] {
        use parking_lot as implementation;
    } else if #[cfg(feature = "std")] {
        use std::sync as implementation;
    } else if #[cfg(feature = "spin")] {
        use spin as implementation;
    } else {
        mod implementation {
            pub(super) use core::cell::RefCell as Mutex;
            pub(super) use core::cell::RefMut as MutexGuard;

            pub(super) use core::cell::RefCell as RwLock;
            pub(super) use core::cell::Ref as RwLockReadGuard;
            pub(super) use core::cell::RefMut as RwLockWriteGuard;

            /// Repeatedly invoke `f` until [`Option::Some`] is returned.
            /// This method [spins](core::hint::spin_loop), busy-waiting the current
            /// thread.
            pub(super) fn spin_unwrap<T>(mut f: impl FnMut() -> Option<T>) -> T {
                'spin: loop {
                    match (f)() {
                        Some(value) => break 'spin value,
                        None => core::hint::spin_loop(),
                    }
                }
            }
        }
    }
}

/// A plain wrapper around [`implementation::Mutex`].
///
/// This is just like [`implementation::Mutex`], but slight inconsistencies
/// between the different implementation APIs are smoothed-over.
pub struct Mutex<T>(implementation::Mutex<T>);

/// A guard produced by locking [`Mutex`].
///
/// This is just a wrapper around a [`implementation::MutexGuard`].
pub struct MutexGuard<'a, T>(implementation::MutexGuard<'a, T>);

impl<T> Mutex<T> {
    /// Create a new [`Mutex`].
    pub fn new(value: T) -> Mutex<T> {
        Mutex(implementation::Mutex::new(value))
    }

    /// Lock the provided [`Mutex`], allowing reading and/or writing.
    pub fn lock(&self) -> MutexGuard<T> {
        let lock;

        cfg_if::cfg_if! {
            if #[cfg(feature = "parking_lot")] {
                lock = self.0.lock();
            } else if #[cfg(feature = "std")] {
                lock = self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            } else if #[cfg(feature = "spin")] {
                lock = self.0.lock();
            } else {
                lock = implementation::spin_unwrap(|| self.0.try_borrow_mut().ok());
            }
        }

        MutexGuard(lock)
    }

    /// Consume the provided [`Mutex`], returning the inner value.
    pub fn into_inner(self) -> T {
        let inner = self.0.into_inner();

        #[cfg(all(feature = "std", not(feature = "parking_lot")))]
        let inner = inner.unwrap_or_else(std::sync::PoisonError::into_inner);

        inner
    }
}

impl<'a, T> ops::Deref for MutexGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl<'a, T> ops::DerefMut for MutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.deref_mut()
    }
}

impl<T: fmt::Debug> fmt::Debug for Mutex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A plain wrapper around [`implementation::RwLock`].
///
/// This is just like [`implementation::RwLock`], but slight inconsistencies
/// between the different implementation APIs are smoothed-over.
pub struct RwLock<T>(implementation::RwLock<T>);

/// A read guard produced by locking [`RwLock`] as a reader.
///
/// This is just a wrapper around a [`implementation::RwLockReadGuard`].
pub struct RwLockReadGuard<'a, T> {
    guard: implementation::RwLockReadGuard<'a, T>,
}

/// A write guard produced by locking [`RwLock`] as a writer.
///
/// This is just a wrapper around a [`implementation::RwLockWriteGuard`].
pub struct RwLockWriteGuard<'a, T> {
    guard: implementation::RwLockWriteGuard<'a, T>,
    /// Allows for a safe `downgrade` method without `parking_lot`
    #[cfg(not(feature = "parking_lot"))]
    lock: &'a RwLock<T>,
}

impl<T> RwLock<T> {
    /// Create a new [`RwLock`].
    pub fn new(value: T) -> RwLock<T> {
        RwLock(implementation::RwLock::new(value))
    }

    /// Read from the provided [`RwLock`].
    pub fn read(&self) -> RwLockReadGuard<T> {
        let guard;

        cfg_if::cfg_if! {
            if #[cfg(feature = "parking_lot")] {
                guard = self.0.read();
            } else if #[cfg(feature = "std")] {
                guard = self.0.read().unwrap_or_else(std::sync::PoisonError::into_inner);
            } else if #[cfg(feature = "spin")] {
                guard = self.0.read();
            } else {
                guard = implementation::spin_unwrap(|| self.0.try_borrow().ok());
            }
        }

        RwLockReadGuard { guard }
    }

    /// Write to the provided [`RwLock`].
    pub fn write(&self) -> RwLockWriteGuard<T> {
        let guard;

        cfg_if::cfg_if! {
            if #[cfg(feature = "parking_lot")] {
                guard = self.0.write();
            } else if #[cfg(feature = "std")] {
                guard = self.0.write().unwrap_or_else(std::sync::PoisonError::into_inner);
            } else if #[cfg(feature = "spin")] {
                guard = self.0.write();
            } else {
                guard = implementation::spin_unwrap(|| self.0.try_borrow_mut().ok());
            }
        }

        RwLockWriteGuard {
            guard,
            #[cfg(not(feature = "parking_lot"))]
            lock: self,
        }
    }
}

impl<'a, T> RwLockWriteGuard<'a, T> {
    /// Downgrade a [write guard](RwLockWriteGuard) into a [read guard](RwLockReadGuard).
    pub fn downgrade(this: Self) -> RwLockReadGuard<'a, T> {
        cfg_if::cfg_if! {
            if #[cfg(feature = "parking_lot")] {
                RwLockReadGuard { guard: implementation::RwLockWriteGuard::downgrade(this.guard) }
            } else {
                let RwLockWriteGuard { guard, lock } = this;

                // FIXME(https://github.com/rust-lang/rust/issues/128203): Replace with `RwLockWriteGuard::downgrade` once stable.
                // This implementation allows for a different thread to "steal" the lock in-between the drop and the read.
                // Ideally, `downgrade` should hold the lock the entire time, maintaining uninterrupted custody.
                drop(guard);
                lock.read()
            }
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for RwLock<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<'a, T> ops::Deref for RwLockReadGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

impl<'a, T> ops::Deref for RwLockWriteGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

impl<'a, T> ops::DerefMut for RwLockWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard.deref_mut()
    }
}
