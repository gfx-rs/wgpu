use std::{
    cell::UnsafeCell,
    marker::PhantomData,
    mem::{self},
    sync::atomic::{AtomicU32, AtomicUsize, Ordering},
};

use parking_lot::Mutex;

use crate::{
    hub,
    sync::{DebugMaybeUninit, DebugUnsafeCell},
    Epoch,
};

pub struct Registry<A, T> {
    storage: Storage<T>,
    identity_manager: Mutex<hub::IdentityManager>,

    _phantom: PhantomData<A>,
}

impl<A, T> Registry<A, T>
where
    A: hub::HalApi,
    T: hub::Resource,
{
    pub fn new() -> Self {
        Self {
            storage: Storage::new(),
            identity_manager: Mutex::new(hub::IdentityManager::default()),
            _phantom: PhantomData,
        }
    }

    fn allocate(&self) -> hub::FutureId<'_, T> {
        let mut ident_guard = self.identity_manager.lock();
        let (index, id) = ident_guard.alloc(A::VARIANT);
        unsafe { self.storage.ensure_index(index) }
        drop(ident_guard);

        hub::FutureId {
            id,
            data: &self.storage,
        }
    }

    fn deallocate(&self, id: T::Id) {
        let mut ident_guard = self.identity_manager.lock();
        let (index, epoch) = ident_guard.free(id);
        unsafe { self.storage.free(index, epoch) }
        drop(ident_guard);
    }
}

pub struct Storage<T> {
    blocks: [DebugUnsafeCell<Option<Box<StorageBlock<512, T>>>>; 256],
    length: AtomicU32,
}
impl<T> Storage<T> {
    fn new() -> Self {
        Self {
            blocks: [(); 256].map(|_| DebugUnsafeCell::new(None)),
            length: AtomicU32::new(0),
        }
    }

    // SAFETY: Must be called inside the identity manager lock.
    unsafe fn ensure_index(&self, index: u32) {
        let length = self.length.load(Ordering::Relaxed);
        if index < length {
            return;
        }
        assert_eq!(index, length);

        let block = length / 512;
        if block >= 256 {
            panic!("Too many resources allocated");
        }

        let allocated_length = (length + 511) & !511;
        if index >= allocated_length {
            // SAFETY: we're the first to ever allocate this block, so we're the only one
            // who could get this mutable reference.
            let block = &mut *self.blocks[block as usize].get();
            *block = Some(Box::new(StorageBlock::new_uninit()));
        }
        self.length.store(length + 1, Ordering::Release);
    }

    pub unsafe fn fill(&self, index: u32, epoch: u32, value: T) {
        // We know that these are all in bounds, because it was pre-allocated for us.
        let block = (index / 512) as usize;
        let data_index = (index % 512) as usize;

        // SAFETY: We have reserved a slot in the block so this block is guarenteed to exist
        // and the slot will not be accessed by any other users.
        let block_option_ref = &*self.blocks[block].get();
        let block_ptr = block_option_ref.as_deref().unwrap();
        let data_ref = &block_ptr.data[data_index];
        let status_ref = &block_ptr.status[data_index];

        let data_mut_ref = &mut *data_ref.get();
        data_mut_ref.write(value);

        *status_ref.lock() = ElementStatus::Occupied(epoch);
    }

    // SAFETY: Must be called inside the identity manager lock.
    unsafe fn free(&self, index: u32, epoch: u32) {
        let block = (index / 512) as usize;
        let data_index = (index % 512) as usize;

        let length = self.length.load(Ordering::Relaxed);
        let allocated_length = (length + 511) & !511;
        if index >= allocated_length {
            panic!("deallocating out of bounds");
        }

        // SAFETY: We have just bounds checked the index and we're inside the lock.
        let block_option_ref = self.blocks[block].get();
        let block_ptr = block_option_ref.as_deref().unwrap();
        let data_ref = &block_ptr.data[data_index];
        let status_ref = &block_ptr.status[data_index];

        // We set this to vacant first before destroying it, so that if a stray index comes through
        // later, it will see that this is vacant and error and not try to grab a reference to
        // the actual data. We are also inside the lock, so creation can't come by and try to
        // create new data while we're still trying to drop things.
        let old_status = mem::replace(&mut *status_ref.lock(), ElementStatus::Vacant);
        assert_eq!(old_status, ElementStatus::Occupied(epoch));

        let data_mut_ref = data_ref.get_mut();
        data_mut_ref.assume_init_drop();
    }
}

struct StorageBlock<const count: usize, T> {
    data: [DebugUnsafeCell<DebugMaybeUninit<T>>; count],
    // TODO: this can be an atomic
    status: [Mutex<ElementStatus>; count],
}

impl<const count: usize, T> StorageBlock<count, T> {
    fn new_uninit() -> Self {
        Self {
            data: [(); count].map(|_| DebugUnsafeCell::new(DebugMaybeUninit::uninit())),
            status: [(); count].map(|_| Mutex::new(ElementStatus::Vacant)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ElementStatus {
    Vacant,
    Occupied(Epoch),
    Error(Epoch, String),
}
