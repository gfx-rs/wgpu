//! Provides a [`Mutex`] for internal use based on what features are available.

cfg_if::cfg_if! {
    if #[cfg(feature = "parking_lot")] {
        use parking_lot::Mutex as MutexInner;
    } else if #[cfg(feature = "std")] {
        use std::sync::Mutex as MutexInner;
    } else {
        use core::cell::RefCell as MutexInner;
    }
}

pub(crate) struct Mutex<T: ?Sized> {
    inner: MutexInner<T>,
}

impl<T: ?Sized + core::fmt::Debug> core::fmt::Debug for Mutex<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        <MutexInner<T> as core::fmt::Debug>::fmt(&self.inner, f)
    }
}

impl<T> Mutex<T> {
    pub const fn new(value: T) -> Self {
        Self {
            inner: MutexInner::new(value),
        }
    }
}

impl<T: ?Sized> Mutex<T> {
    pub fn lock(&self) -> impl core::ops::DerefMut<Target = T> + '_ {
        cfg_if::cfg_if! {
            if #[cfg(feature = "parking_lot")] {
                return self.inner.lock();
            } else if #[cfg(feature = "std")] {
                return self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            } else {
                loop {
                    let Ok(lock) = self.inner.try_borrow_mut() else {
                        // Without `std` all we can do is spin until the current lock is released
                        core::hint::spin_loop();
                        continue;
                    };

                    return lock;
                }
            }
        }
    }
}
