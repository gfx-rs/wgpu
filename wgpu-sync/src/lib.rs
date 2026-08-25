#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(
    clippy::ptr_as_ptr,
    missing_docs,
    unsafe_op_in_unsafe_fn,
    unused_qualifications
)]
#![no_std]

//! Provides [`Mutex`] and [`RwLock`] types with an appropriate implementation.

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod mutex;
mod rwlock;

pub use mutex::RawMutex;
pub use rwlock::RawRwLock;

// FIXME:
// * `Condvar` is only available through `parking_lot` and not through `lock_api`.
// * `Condvar` only works with the specific `RawMutex` implementation from `parking_lot`.
#[cfg(feature = "std")]
pub use parking_lot::{Condvar, Mutex as CondvarMutex};

pub use once_cell::race::{OnceBool, OnceBox, OnceNonZeroUsize, OnceRef};

cfg_if::cfg_if! {
    if #[cfg(feature = "std")] {
        pub use once_cell::sync::{Lazy, OnceCell};
    } else {
        pub use once_cell::unsync::{Lazy, OnceCell};
    }
}

/// A [`Mutex`](lock_api::Mutex) using [`RawMutex`] for its backing implementation.
pub type Mutex<T> = lock_api::Mutex<RawMutex, T>;

/// A [`MutexGuard`](lock_api::MutexGuard) using [`RawMutex`] for its backing implementation.
pub type MutexGuard<'a, T> = lock_api::MutexGuard<'a, RawMutex, T>;

/// A [`MappedMutexGuard`](lock_api::MappedMutexGuard) using [`RawMutex`] for its backing implementation.
pub type MappedMutexGuard<'a, T> = lock_api::MappedMutexGuard<'a, RawMutex, T>;

/// A [`RwLock`](lock_api::RwLock) using [`RawRwLock`] for its backing implementation.
pub type RwLock<T> = lock_api::RwLock<RawRwLock, T>;

/// A [`RwLockReadGuard`](lock_api::RwLockReadGuard) using [`RawRwLock`] for its backing implementation.
pub type RwLockReadGuard<'a, T> = lock_api::RwLockReadGuard<'a, RawRwLock, T>;

/// A [`RwLockWriteGuard`](lock_api::RwLockWriteGuard) using [`RawRwLock`] for its backing implementation.
pub type RwLockWriteGuard<'a, T> = lock_api::RwLockWriteGuard<'a, RawRwLock, T>;

/// A [`RwLockUpgradableReadGuard`](lock_api::RwLockUpgradableReadGuard) using [`RawRwLock`] for its backing implementation.
pub type RwLockUpgradableReadGuard<'a, T> = lock_api::RwLockUpgradableReadGuard<'a, RawRwLock, T>;
