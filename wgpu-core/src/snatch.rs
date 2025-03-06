use core::{cell::UnsafeCell, fmt};

use crate::lock::{rank, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// A guard that provides read access to snatchable data.
pub struct SnatchGuard<'a>(#[expect(dead_code)] RwLockReadGuard<'a, ()>);
/// A guard that allows snatching the snatchable data.
pub struct ExclusiveSnatchGuard<'a>(#[expect(dead_code)] RwLockWriteGuard<'a, ()>);

/// A value that is mostly immutable but can be "snatched" if we need to destroy
/// it early.
///
/// In order to safely access the underlying data, the device's global snatchable
/// lock must be taken. To guarantee it, methods take a read or write guard of that
/// special lock.
pub struct Snatchable<T> {
    value: UnsafeCell<Option<T>>,
}

impl<T> Snatchable<T> {
    pub fn new(val: T) -> Self {
        Snatchable {
            value: UnsafeCell::new(Some(val)),
        }
    }

    #[allow(dead_code)]
    pub fn empty() -> Self {
        Snatchable {
            value: UnsafeCell::new(None),
        }
    }

    /// Get read access to the value. Requires a the snatchable lock's read guard.
    pub fn get<'a>(&'a self, _guard: &'a SnatchGuard) -> Option<&'a T> {
        unsafe { (*self.value.get()).as_ref() }
    }

    /// Take the value. Requires a the snatchable lock's write guard.
    pub fn snatch(&self, _guard: &mut ExclusiveSnatchGuard) -> Option<T> {
        unsafe { (*self.value.get()).take() }
    }

    /// Take the value without a guard. This can only be used with exclusive access
    /// to self, so it does not require locking.
    ///
    /// Typically useful in a drop implementation.
    pub fn take(&mut self) -> Option<T> {
        self.value.get_mut().take()
    }
}

// Can't safely print the contents of a snatchable object without holding
// the lock.
impl<T> fmt::Debug for Snatchable<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<snatchable>")
    }
}

unsafe impl<T> Sync for Snatchable<T> {}

use trace::LockTrace;
#[cfg(all(debug_assertions, feature = "std"))]
mod trace {
    use core::{cell::Cell, fmt, panic::Location};
    use std::{backtrace::Backtrace, thread};

    pub(super) struct LockTrace {
        purpose: &'static str,
        caller: &'static Location<'static>,
        backtrace: Backtrace,
    }

    impl fmt::Display for LockTrace {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "a {} lock at {}\n{}",
                self.purpose, self.caller, self.backtrace
            )
        }
    }

    impl LockTrace {
        #[track_caller]
        pub(super) fn enter(purpose: &'static str) {
            let new = LockTrace {
                purpose,
                caller: Location::caller(),
                backtrace: Backtrace::capture(),
            };

            if let Some(prev) = SNATCH_LOCK_TRACE.take() {
                let current = thread::current();
                let name = current.name().unwrap_or("<unnamed>");
                panic!(
                    "thread '{name}' attempted to acquire a snatch lock recursively.\n\
                 - Currently trying to acquire {new}\n\
                 - Previously acquired {prev}",
                );
            } else {
                SNATCH_LOCK_TRACE.set(Some(new));
            }
        }

        pub(super) fn exit() {
            SNATCH_LOCK_TRACE.take();
        }
    }

    std::thread_local! {
        static SNATCH_LOCK_TRACE: Cell<Option<LockTrace>> = const { Cell::new(None) };
    }
}
#[cfg(not(all(debug_assertions, feature = "std")))]
mod trace {
    pub(super) struct LockTrace {
        _private: (),
    }

    impl LockTrace {
        pub(super) fn enter(_purpose: &'static str) {}
        pub(super) fn exit() {}
    }
}

/// A Device-global lock for all snatchable data.
pub struct SnatchLock {
    lock: RwLock<()>,
}

impl SnatchLock {
    /// The safety of `Snatchable::get` and `Snatchable::snatch` rely on their using of the
    /// right SnatchLock (the one associated to the same device). This method is unsafe
    /// to force force sers to think twice about creating a SnatchLock. The only place this
    /// method should be called is when creating the device.
    pub unsafe fn new(rank: rank::LockRank) -> Self {
        SnatchLock {
            lock: RwLock::new(rank, ()),
        }
    }

    /// Request read access to snatchable resources.
    #[track_caller]
    pub fn read(&self) -> SnatchGuard {
        LockTrace::enter("read");
        SnatchGuard(self.lock.read())
    }

    /// Request write access to snatchable resources.
    ///
    /// This should only be called when a resource needs to be snatched. This has
    /// a high risk of causing lock contention if called concurrently with other
    /// wgpu work.
    #[track_caller]
    pub fn write(&self) -> ExclusiveSnatchGuard {
        LockTrace::enter("write");
        ExclusiveSnatchGuard(self.lock.write())
    }
}

impl Drop for SnatchGuard<'_> {
    fn drop(&mut self) {
        LockTrace::exit();
    }
}

impl Drop for ExclusiveSnatchGuard<'_> {
    fn drop(&mut self) {
        LockTrace::exit();
    }
}
