use std::ops::{Deref, DerefMut};
#[cfg(not(debug_assertions))]
use std::{cell::UnsafeCell, mem::MaybeUninit};

use parking_lot::{RwLock};

/// Unsafe cell that's actually an RW lock in debug.
pub struct DebugUnsafeCell<T> {
    #[cfg(debug_assertions)]
    inner: RwLock<T>,
    #[cfg(not(debug_assertions))]
    inner: UnsafeCell<T>,
}

#[cfg(debug_assertions)]
impl<T> DebugUnsafeCell<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: RwLock::new(value),
        }
    }

    pub unsafe fn get(&self) -> impl Deref<Target = T> + '_ {
        self.inner
            .try_read()
            .expect("DebugUnsafeCell detected read-while-write-locked")
    }

    pub unsafe fn get_debug_unchecked(&self) -> &T {
        &*self.inner.data_ptr()
    }

    pub unsafe fn get_mut(&self) -> impl DerefMut<Target = T> + '_ {
        self.inner
            .try_write()
            .expect("DebugUnsafeCell detected write-while-locked")
    }

    pub unsafe fn get_debug_unchecked_mut(&self) -> &mut T {
        &mut *self.inner.data_ptr()
    }
}

#[cfg(not(debug_assertions))]
impl<T> DebugUnsafeCell<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: UnsafeCell::new(value),
        }
    }

    pub unsafe fn get(&self) -> impl Deref<Target = T> + '_ {
        &*self.inner.get();
    }

    pub unsafe fn get_debug_unchecked(&self) -> &T {
        &*self.inner.get()
    }

    pub unsafe fn get_mut(&self) -> impl DerefMut<Target = T> + '_ {
        &mut *self.inner.get();
    }

    pub unsafe fn get_debug_unchecked_mut(&self) -> &mut T {
        &mut *self.inner.get();
    }
}

/// Maybe uninit that's actually an Option in debug.
pub struct DebugMaybeUninit<T> {
    #[cfg(debug_assertions)]
    inner: Option<T>,
    #[cfg(not(debug_assertions))]
    inner: MaybeUninit<T>,
}

#[cfg(debug_assertions)]
impl<T> DebugMaybeUninit<T> {
    pub fn uninit() -> Self {
        Self { inner: None }
    }

    pub unsafe fn write(&mut self, value: T) {
        assert!(
            self.inner.is_none(),
            "DebugMaybeUninit detected write-while-some"
        );
        self.inner = Some(value);
    }

    pub unsafe fn assume_init(self) -> T {
        self.inner
            .expect("DebugMaybeUninit detected assume-while-none")
    }

    pub unsafe fn assume_init_ref(&self) -> &T {
        self.inner
            .as_ref()
            .expect("DebugMaybeUninit detected ref-while-none")
    }

    pub unsafe fn assume_init_mut(&mut self) -> &mut T {
        self.inner
            .as_mut()
            .expect("DebugMaybeUninit detected mut-while-none")
    }

    pub unsafe fn assume_init_drop(&mut self) {
        assert!(
            self.inner.is_some(),
            "DebugMaybeUninit detected drop-while-none"
        );
        self.inner = None;
    }
}

#[cfg(not(debug_assertions))]
impl<T> DebugMaybeUninit<T> {
    pub fn uninit() -> Self {
        Self {
            inner: MaybeUninit::uninit(),
        }
    }

    pub unsafe fn write(&mut self, value: T) {
        self.inner.write(value);
    }

    pub unsafe fn assume_init(self) -> T {
        self.inner.assume_init()
    }

    pub unsafe fn assume_init_ref(&self) -> &T {
        self.inner.assume_init_ref()
    }

    pub unsafe fn assume_init_mut(&mut self) -> &mut T {
        self.inner.assume_init_mut()
    }

    pub unsafe fn assume_init_drop(&mut self) {
        self.inner.assume_init_drop()
    }
}
