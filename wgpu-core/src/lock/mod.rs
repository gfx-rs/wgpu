//! Instrumented lock types.
//!
//! This module defines a set of instrumented wrappers for the lock
//! types used in `wgpu-core` ([`Mutex`], [`RwLock`], and
//! [`SnatchLock`]) that help us understand and validate `wgpu-core`
//! synchronization.
//!
//! - The [`ranked`] module defines lock types that perform run-time
//!   checks to ensure that each thread acquires locks only in a
//!   specific order, to prevent deadlocks.
//!
//! - The [`vanilla`] module defines lock types that are
//!   uninstrumented, no-overhead wrappers around the standard lock
//!   types.
//!
//! (We plan to add more wrappers in the future.)
//!
//! If the `wgpu_validate_locks` config is set (for example, with
//! `RUSTFLAGS='--cfg wgpu_validate_locks'`), `wgpu-core` uses the
//! [`ranked`] module's locks. We hope to make this the default for
//! debug builds soon.
//!
//! Otherwise, `wgpu-core` uses the [`vanilla`] module's locks.
//!
//! [`Mutex`]: parking_lot::Mutex
//! [`RwLock`]: parking_lot::RwLock
//! [`SnatchLock`]: crate::snatch::SnatchLock

pub mod rank;

#[cfg_attr(not(wgpu_validate_locks), allow(dead_code))]
mod ranked;

#[cfg_attr(wgpu_validate_locks, allow(dead_code))]
mod vanilla;

#[cfg(wgpu_validate_locks)]
pub use ranked::{Mutex, MutexGuard};

#[cfg(not(wgpu_validate_locks))]
pub use vanilla::{Mutex, MutexGuard};
