use std::{cell::UnsafeCell, marker::PhantomData};

use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::{hub, id};

pub struct DeviceDestructionLock {
    inner: RwLock<()>,
    buffer_destruction_queue: Mutex<Vec<id::BufferId>>,
    texture_destruction_queue: Mutex<Vec<id::TextureId>>,
}

impl DeviceDestructionLock {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(()),
            buffer_destruction_queue: Mutex::new(Vec::new()),
            texture_destruction_queue: Mutex::new(Vec::new()),
        }
    }
    pub fn read(&self) -> ReadDestructionGuard<'_> {
        ReadDestructionGuard {
            inner: self.inner.read(),
            buffer_destruction_queue: &self.buffer_destruction_queue,
            texture_destruction_queue: &self.texture_destruction_queue,
        }
    }

    pub fn destroy_buffers(
        &self,
        buffer: id::BufferId,
    ) -> Option<WriteDestructionGuard<'_, id::BufferId>> {
        match self.inner.try_write() {
            Some(g) => Some(WriteDestructionGuard {
                inner: g,
                _phantom: PhantomData,
            }),
            None => {
                self.buffer_destruction_queue.lock().push(buffer);
                None
            }
        }
    }

    pub fn destroy_textures(
        &self,
        texture: id::TextureId,
    ) -> Option<WriteDestructionGuard<'_, id::TextureId>> {
        match self.inner.try_write() {
            Some(g) => Some(WriteDestructionGuard {
                inner: g,
                _phantom: PhantomData,
            }),
            None => {
                self.texture_destruction_queue.lock().push(texture);
                None
            }
        }
    }
}

pub struct ReadDestructionGuard<'a> {
    inner: RwLockReadGuard<'a, ()>,
    buffer_destruction_queue: &'a Mutex<Vec<id::BufferId>>,
    texture_destruction_queue: &'a Mutex<Vec<id::TextureId>>,
}

impl ReadDestructionGuard<'_> {
    pub fn run_pending_destructions(self) {
        // get a reference to the underlying lock
        let lock = RwLockReadGuard::rwlock(&self.inner);
        // drop the guard
        drop(self.inner);

        // try to grab the lock as write
        let mutable_guard = match lock.try_write() {
            Some(guard) => guard,
            None => return,
        };

        let buffer_queue_guard = self.buffer_destruction_queue.lock();

        drop(buffer_queue_guard);

        let texture_queue_guard = self.texture_destruction_queue.lock();

        drop(texture_queue_guard);

        drop(mutable_guard);
    }
}

pub struct WriteDestructionGuard<'a, T> {
    inner: RwLockWriteGuard<'a, ()>,
    _phantom: PhantomData<T>,
}

#[derive(Debug)]
pub struct DestructableResource<T>
where
    T: hub::Resource,
{
    inner: UnsafeCell<Option<T::Raw>>,
}

unsafe impl<T> Send for DestructableResource<T>
where
    T: hub::Resource,
    T::Raw: Send,
{
}
unsafe impl<T> Sync for DestructableResource<T>
where
    T: hub::Resource,
    T::Raw: Sync,
{
}

impl<T> DestructableResource<T>
where
    T: hub::Resource,
{
    pub fn new(value: T::Raw) -> Self {
        Self {
            inner: UnsafeCell::new(Some(value)),
        }
    }

    pub fn as_ref<'a>(&self, _guard: &'a ReadDestructionGuard<'_>) -> Option<&'a T::Raw> {
        // Safe as we have the read destruction guard
        let option = unsafe { &*self.inner.get() };

        option.as_ref()
    }

    pub fn destroy<'a>(&self, _guard: &'a mut WriteDestructionGuard<'_, T::Id>) -> Option<T::Raw> {
        // Safe as we have the write destruction guard
        let option = unsafe { &mut *self.inner.get() };

        option.take()
    }

    pub fn into_inner(self) -> Option<T::Raw> {
        self.inner.into_inner()
    }
}
