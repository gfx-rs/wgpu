//! Plain, uninstrumented wrappers around a particular implementation of lock types.
//! These definitions are used when no particular lock instrumentation
//! Cargo feature is selected.
//! The exact implementation used depends on the following features:
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
        }
    }
}

/// A plain wrapper around [`implementation::Mutex`].
///
/// This is just like [`implementation::Mutex`], except that our [`new`]
/// method takes a rank, indicating where the new mutex should sit in
/// `wgpu-core`'s lock ordering. The rank is ignored.
///
/// See the [`lock`] module documentation for other wrappers.
///
/// [`new`]: Mutex::new
/// [`lock`]: crate::lock
pub struct Mutex<T>(implementation::Mutex<T>);

/// A guard produced by locking [`Mutex`].
///
/// This is just a wrapper around a [`implementation::MutexGuard`].
pub struct MutexGuard<'a, T>(implementation::MutexGuard<'a, T>);

impl<T> Mutex<T> {
    pub fn new(_rank: super::rank::LockRank, value: T) -> Mutex<T> {
        Mutex(implementation::Mutex::new(value))
    }

    pub fn lock(&self) -> MutexGuard<T> {
        cfg_if::cfg_if! {
            if #[cfg(feature = "parking_lot")] {
                let lock = self.0.lock();
            } else if #[cfg(feature = "std")] {
                let lock = self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            } else if #[cfg(feature = "spin")] {
                let lock = self.0.lock();
            } else {
                let lock = loop {
                    if let Ok(lock) = self.0.try_borrow_mut() {
                        break lock;
                    }
                    core::hint::spin_loop();
                };
            }
        }

        MutexGuard(lock)
    }

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
/// This is just like [`implementation::RwLock`], except that our [`new`]
/// method takes a rank, indicating where the new mutex should sit in
/// `wgpu-core`'s lock ordering. The rank is ignored.
///
/// See the [`lock`] module documentation for other wrappers.
///
/// [`new`]: RwLock::new
/// [`lock`]: crate::lock
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
    pub fn new(_rank: super::rank::LockRank, value: T) -> RwLock<T> {
        RwLock(implementation::RwLock::new(value))
    }

    pub fn read(&self) -> RwLockReadGuard<T> {
        cfg_if::cfg_if! {
            if #[cfg(feature = "parking_lot")] {
                let guard = self.0.read();
            } else if #[cfg(feature = "std")] {
                let guard = self.0.read().unwrap_or_else(std::sync::PoisonError::into_inner);
            } else if #[cfg(feature = "spin")] {
                let guard = self.0.read();
            } else {
                let guard = loop {
                    if let Ok(guard) = self.0.try_borrow() {
                        break guard;
                    }
                    core::hint::spin_loop();
                };
            }
        }

        RwLockReadGuard { guard }
    }

    pub fn write(&self) -> RwLockWriteGuard<T> {
        cfg_if::cfg_if! {
            if #[cfg(feature = "parking_lot")] {
                let guard = self.0.write();
            } else if #[cfg(feature = "std")] {
                let guard = self.0.write().unwrap_or_else(std::sync::PoisonError::into_inner);
            } else if #[cfg(feature = "spin")] {
                let guard = self.0.write();
            } else {
                let guard = loop {
                    if let Ok(guard) = self.0.try_borrow_mut() {
                        break guard;
                    }
                    core::hint::spin_loop();
                };
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
