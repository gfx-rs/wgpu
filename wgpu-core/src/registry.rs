use std::sync::Arc;

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use wgt::Backend;

use crate::{
    id,
    identity::{IdentityHandler, IdentityHandlerFactory},
    resource::Resource,
    storage::{InvalidId, Storage, StorageReport},
};

#[derive(Debug)]
pub struct Registry<I: id::TypedId, T: Resource<I>, F: IdentityHandlerFactory<I>> {
    identity: F::Filter,
    storage: RwLock<Storage<T, I>>,
    backend: Backend,
}

impl<I: id::TypedId, T: Resource<I>, F: IdentityHandlerFactory<I>> Registry<I, T, F> {
    pub(crate) fn new(backend: Backend, factory: &F) -> Self {
        Self {
            identity: factory.spawn(),
            storage: RwLock::new(Storage::new()),
            backend,
        }
    }

    pub(crate) fn without_backend(factory: &F, kind: &'static str) -> Self {
        Self {
            identity: factory.spawn(),
            storage: RwLock::new(Storage::from_kind(kind)),
            backend: Backend::Empty,
        }
    }
}

#[must_use]
pub(crate) struct FutureId<'a, I: id::TypedId, T: Resource<I>> {
    id: I,
    data: &'a RwLock<Storage<T, I>>,
}

impl<I: id::TypedId + Copy, T: Resource<I>> FutureId<'_, I, T> {
    pub fn id(&self) -> I {
        self.id
    }

    pub fn into_id(self) -> I {
        self.id
    }

    pub fn assign(self, value: T) -> (id::Valid<I>, Arc<T>) {
        value.info().set_id(self.id);
        self.data.write().insert(self.id, value);
        (
            id::Valid(self.id),
            self.data.read().get(self.id).unwrap().clone(),
        )
    }

    pub fn assign_error(self, label: &str) -> I {
        self.data.write().insert_error(self.id, label);
        self.id
    }
}

impl<I: id::TypedId + Copy, T: Resource<I>, F: IdentityHandlerFactory<I>> Registry<I, T, F> {
    pub(crate) fn prepare(
        &self,
        id_in: <F::Filter as IdentityHandler<I>>::Input,
    ) -> FutureId<I, T> {
        FutureId {
            id: self.identity.process(id_in, self.backend),
            data: &self.storage,
        }
    }
    pub(crate) fn try_get(&self, id: I) -> Result<Option<Arc<T>>, InvalidId> {
        self.storage.read().try_get(id).map(|o| o.cloned())
    }
    pub(crate) fn get(&self, id: I) -> Result<Arc<T>, InvalidId> {
        self.storage.read().get(id).map(|v| v.clone())
    }
    pub(crate) fn read<'a>(&'a self) -> RwLockReadGuard<'a, Storage<T, I>> {
        self.storage.read()
    }
    pub(crate) fn write<'a>(&'a self) -> RwLockWriteGuard<'a, Storage<T, I>> {
        self.storage.write()
    }
    pub fn unregister_locked(&self, id: I, storage: &mut Storage<T, I>) -> Option<Arc<T>> {
        let value = storage.remove(id);
        //Note: careful about the order here!
        self.identity.free(id);
        //Returning None is legal if it's an error ID
        value
    }
    pub(crate) fn unregister(&self, id: I) -> Option<Arc<T>> {
        let value = self.storage.write().remove(id);
        //Note: careful about the order here!
        self.identity.free(id);
        //Returning None is legal if it's an error ID
        value
    }

    pub fn label_for_resource(&self, id: I) -> String {
        let guard = self.storage.read();

        let type_name = guard.kind();
        match guard.get(id) {
            Ok(res) => {
                let label = res.label();
                if label.is_empty() {
                    format!("<{}-{:?}>", type_name, id.unzip())
                } else {
                    label
                }
            }
            Err(_) => format!(
                "<Invalid-{} label={}>",
                type_name,
                guard.label_for_invalid_id(id)
            ),
        }
    }

    pub(crate) fn generate_report(&self) -> StorageReport {
        self.storage.read().generate_report()
    }
}
