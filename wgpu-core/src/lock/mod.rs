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
//! To get `ranked`, build with the `wgpu_validate_locks` config set, perhaps
//! with `RUSTFLAGS='--cfg wgpu_validate_locks'`. The `"std"` feature must also
//! be enabled, as `ranked` needs thread-local storage.
//!
//! To get `ranked` only when debug assertions are also enabled, build with the
//! `wgpu_validate_locks_debug` config set. The `wgpu` crate selects this via
//! its `.cargo/config.toml` file, so lock validation is active by default in
//! `wgpu` CI and `wgpu` development trees, but not in `wgpu`-using applications
//! unless specifically requested.
//!
//! To get `observing`, enable the `"observe_locks"` feature. This also enables
//! `"std"`. You must also ensure that `wgpu_validate_locks` is not set, as that
//! takes precedence --- and it's enabled automatically in debug builds for
//! `wgpu`, as explained above.
//!
//! Otherwise, `wgpu-core` uses the [`vanilla`] module's locks.
//!
//! [`Mutex`]: wgpu_sync::Mutex
//! [`RwLock`]: wgpu_sync::RwLock
//! [`SnatchLock`]: crate::snatch::SnatchLock

pub mod rank;

// Include all the implementations in the build whenever they can be
// compiled, even if they're not selected, to catch errors and help
// `rust-analyzer` find references.

#[cfg(feature = "std")] // requires thread-locals to work
#[cfg_attr(not(wgpu_validate_locks), allow(dead_code))]
mod ranked;

// The `observing` module requires the `serde` and `ron` dependencies
// and the "std" feature for thread-locals. All are enabled by the
// `"observe_locks"` feature.
#[cfg(feature = "observe_locks")]
#[cfg_attr(wgpu_validate_locks, allow(dead_code))]
mod observing;

// If "std" is not enabled, then none of the other modules can be built.
#[cfg_attr(
    all(feature = "std", any(wgpu_validate_locks, feature = "observe_locks")),
    allow(dead_code)
)]
mod vanilla;

#[cfg(all(wgpu_validate_locks, feature = "std"))]
use ranked as chosen;

#[cfg(all(not(wgpu_validate_locks), feature = "observe_locks"))]
use observing as chosen;

// We must use `vanilla`:
// - whenever `"std"` is not selected, and
// - whenever neither of the instrumented locks are enabled.
#[cfg(any(
    not(feature = "std"),
    not(any(wgpu_validate_locks, feature = "observe_locks"))
))]
use vanilla as chosen;

pub use chosen::{Mutex, MutexGuard, RankData, RwLock, RwLockReadGuard, RwLockWriteGuard};
