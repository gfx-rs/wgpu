//! Helpers for tracing internal wgpu locks in multithreaded code.

#[cfg(feature = "trace")]
pub use unlock::*;

#[cfg(not(feature = "trace"))]
pub use parking_lot::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
#[cfg(not(feature = "trace"))]
pub use unlock::{capture, drain};
