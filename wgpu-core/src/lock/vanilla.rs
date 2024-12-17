//! Plain, uninstrumented wrappers around [`parking_lot`] lock types.
//!
//! These definitions are used when no particular lock instrumentation
//! Cargo feature is selected.

/// A plain wrapper around [`parking_lot::Mutex`].
///
/// This is just like [`parking_lot::Mutex`], except that our [`new`]
/// method takes a rank, indicating where the new mutex should sit in
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
    pub fn new(_rank: super::rank::LockRank, value: T) -> Mutex<T> {
        Mutex(parking_lot::Mutex::new(value))
    }

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

/// A plain wrapper around [`parking_lot::RwLock`].
///
/// This is just like [`parking_lot::RwLock`], except that our [`new`]
/// method takes a rank, indicating where the new mutex should sit in
/// `wgpu-core`'s lock ordering. The rank is ignored.
///
/// See the [`lock`] module documentation for other wrappers.
///
/// [`new`]: RwLock::new
/// [`lock`]: crate::lock
pub struct RwLock<T>(parking_lot::RwLock<T>);

/// A read guard produced by locking [`RwLock`] as a reader.
///
/// This is just a wrapper around a [`parking_lot::RwLockReadGuard`].
pub struct RwLockReadGuard<'a, T>(parking_lot::RwLockReadGuard<'a, T>);

/// A write guard produced by locking [`RwLock`] as a writer.
///
/// This is just a wrapper around a [`parking_lot::RwLockWriteGuard`].
pub struct RwLockWriteGuard<'a, T>(parking_lot::RwLockWriteGuard<'a, T>);

impl<T> RwLock<T> {
    pub fn new(_rank: super::rank::LockRank, value: T) -> RwLock<T> {
        RwLock(parking_lot::RwLock::new(value))
    }

    pub fn read(&self) -> RwLockReadGuard<T> {
        RwLockReadGuard(self.0.read())
    }

    pub fn write(&self) -> RwLockWriteGuard<T> {
        RwLockWriteGuard(self.0.write())
    }
}

impl<'a, T> RwLockWriteGuard<'a, T> {
    pub fn downgrade(this: Self) -> RwLockReadGuard<'a, T> {
        RwLockReadGuard(parking_lot::RwLockWriteGuard::downgrade(this.0))
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for RwLock<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<'a, T> std::ops::Deref for RwLockReadGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl<'a, T> std::ops::Deref for RwLockWriteGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

impl<'a, T> std::ops::DerefMut for RwLockWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.deref_mut()
    }
}
