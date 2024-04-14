//! Plain, uninstrumented wrappers around [`parking_lot`] lock types.
//!
//! These definitions are used when no particular lock instrumentation
//! Cargo feature is selected.

/// A plain wrapper around [`parking_lot::Mutex`].
///
/// This is just like [`parking_lot::Mutex`], except that our [`new`]
/// method takes a rank, indicating where the new mutex should sitc in
/// `wgpu-core`'s lock ordering. The rank is ignored.
///
/// See the [`lock`] module documentation for other wrappers.
///
/// [`new`]: Mutex::new
/// [`lock`]: crate::lock
pub struct Mutex<T>(parking_lot::Mutex<T>);

/// A guard produced by locking [`Mutex`].
///
/// This is just a wrapper around a [`parking_lot::MutexGuard`].
pub struct MutexGuard<'a, T>(parking_lot::MutexGuard<'a, T>);

impl<T> Mutex<T> {
    #[inline]
    pub fn new(_rank: super::rank::LockRank, value: T) -> Mutex<T> {
        Mutex(parking_lot::Mutex::new(value))
    }

    #[inline]
    pub fn lock(&self) -> MutexGuard<T> {
        MutexGuard(self.0.lock())
    }
}

impl<'a, T> std::ops::Deref for MutexGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl<'a, T> std::ops::DerefMut for MutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.deref_mut()
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Mutex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
