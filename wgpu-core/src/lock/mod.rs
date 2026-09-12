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
//! - The [`observing`] module defines lock types that record
//!   `wgpu-core`'s lock acquisition activity to disk, for later
//!   analysis by the `lock-analyzer` binary.
//!
//! - The [`vanilla`] module defines lock types that are
//!   uninstrumented, no-overhead wrappers around the standard lock
//!   types.
//!
//! If the `wgpu_validate_locks` config is set (either directly, i.e. with
//! `RUSTFLAGS='--cfg wgpu_validate_locks'`, or via the
//! `wgpu_validate_locks_debug` `cfg` alias in `build.rs` that conditionally
//! enables it when debug assertions are also enabled), `wgpu-core` uses the
//! [`ranked`] module's locks. `.config/cargo.toml` in the `wgpu` tree specifies
//! `wgpu_validate_locks_debug`, so lock validation is active by default in
//! `wgpu` CI and `wgpu` development trees, but not in `wgpu`-using applications
//! unless specifically requested.
//!
//! If the `observe_locks` feature is enabled, `wgpu-core` uses the
//! [`observing`] module's locks.
//!
//! Otherwise, `wgpu-core` uses the [`vanilla`] module's locks.
//!
//! [`Mutex`]: wgpu_sync::Mutex
//! [`RwLock`]: wgpu_sync::RwLock
//! [`SnatchLock`]: crate::snatch::SnatchLock

pub mod rank;

#[cfg(feature = "std")] // requires thread-locals to work
#[cfg_attr(not(wgpu_validate_locks), allow(dead_code))]
mod ranked;

#[cfg(feature = "observe_locks")]
#[cfg_attr(wgpu_validate_locks, allow(dead_code))]
mod observing;

#[cfg_attr(any(wgpu_validate_locks, feature = "observe_locks"), allow(dead_code))]
mod vanilla;

#[cfg(wgpu_validate_locks)]
use ranked as chosen;

#[cfg(all(not(wgpu_validate_locks), feature = "observe_locks"))]
use observing as chosen;

#[cfg(not(any(wgpu_validate_locks, feature = "observe_locks")))]
use vanilla as chosen;

pub use chosen::{Mutex, MutexGuard, RankData, RwLock, RwLockReadGuard, RwLockWriteGuard};
