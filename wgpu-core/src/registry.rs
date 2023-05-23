use std::marker::PhantomData;

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use wgt::Backend;

use crate::{
    hub::{Access, Token},
    id,
    identity::{IdentityHandler, IdentityHandlerFactory},
    resource::Resource,
    storage::Storage,
};

#[derive(Debug)]
pub struct Registry<T: Resource, I: id::TypedId, F: IdentityHandlerFactory<I>> {
    identity: F::Filter,
    pub(crate) data: RwLock<Storage<T, I>>,
    backend: Backend,
}

impl<T: Resource, I: id::TypedId, F: IdentityHandlerFactory<I>> Registry<T, I, F> {
    pub(crate) fn new(backend: Backend, factory: &F) -> Self {
        Self {
            identity: factory.spawn(),
            data: RwLock::new(Storage {
                map: Vec::new(),
                kind: T::TYPE,
                _phantom: PhantomData,
            }),
            backend,
        }
    }

    pub(crate) fn without_backend(factory: &F, kind: &'static str) -> Self {
        Self {
            identity: factory.spawn(),
            data: RwLock::new(Storage {
                map: Vec::new(),
                kind,
                _phantom: PhantomData,
            }),
            backend: Backend::Empty,
        }
    }
}

#[must_use]
pub(crate) struct FutureId<'a, I: id::TypedId, T> {
    id: I,
    data: &'a RwLock<Storage<T, I>>,
}

impl<I: id::TypedId + Copy, T> FutureId<'_, I, T> {
    #[cfg(feature = "trace")]
    pub fn id(&self) -> I {
        self.id
    }

    pub fn into_id(self) -> I {
        self.id
    }

    pub fn assign<'a, A: Access<T>>(self, value: T, _: &'a mut Token<A>) -> id::Valid<I> {
        self.data.write().insert(self.id, value);
        id::Valid(self.id)
    }

    pub fn assign_error<'a, A: Access<T>>(self, label: &str, _: &'a mut Token<A>) -> I {
        self.data.write().insert_error(self.id, label);
        self.id
    }
}

impl<T: Resource, I: id::TypedId + Copy, F: IdentityHandlerFactory<I>> Registry<T, I, F> {
    pub(crate) fn prepare(
        &self,
        id_in: <F::Filter as IdentityHandler<I>>::Input,
    ) -> FutureId<I, T> {
        FutureId {
            id: self.identity.process(id_in, self.backend),
            data: &self.data,
        }
    }

    /// Acquire read access to this `Registry`'s contents.
    ///
    /// The caller must present a mutable reference to a `Token<A>`,
    /// for some type `A` that comes before this `Registry`'s resource
    /// type `T` in the lock ordering. A `Token<Root>` grants
    /// permission to lock any field; see [`Token::root`].
    ///
    /// Once the read lock is acquired, return a new `Token<T>`, along
    /// with a read guard for this `Registry`'s [`Storage`], which can
    /// be indexed by id to get at the actual resources.
    ///
    /// The borrow checker ensures that the caller cannot again access
    /// its `Token<A>` until it has dropped both the guard and the
    /// `Token<T>`.
    ///
    /// See the [`Hub`] type for more details on locking.
    ///
    /// [`Hub`]: crate::hub::Hub
    pub(crate) fn read<'a, A: Access<T>>(
        &'a self,
        _token: &'a mut Token<A>,
    ) -> (RwLockReadGuard<'a, Storage<T, I>>, Token<'a, T>) {
        (self.data.read(), Token::new())
    }

    /// Acquire write access to this `Registry`'s contents.
    ///
    /// The caller must present a mutable reference to a `Token<A>`,
    /// for some type `A` that comes before this `Registry`'s resource
    /// type `T` in the lock ordering. A `Token<Root>` grants
    /// permission to lock any field; see [`Token::root`].
    ///
    /// Once the lock is acquired, return a new `Token<T>`, along with
    /// a write guard for this `Registry`'s [`Storage`], which can be
    /// indexed by id to get at the actual resources.
    ///
    /// The borrow checker ensures that the caller cannot again access
    /// its `Token<A>` until it has dropped both the guard and the
    /// `Token<T>`.
    ///
    /// See the [`Hub`] type for more details on locking.
    ///
    /// [`Hub`]: crate::hub::Hub
    pub(crate) fn write<'a, A: Access<T>>(
        &'a self,
        _token: &'a mut Token<A>,
    ) -> (RwLockWriteGuard<'a, Storage<T, I>>, Token<'a, T>) {
        (self.data.write(), Token::new())
    }

    /// Unregister the resource at `id`.
    ///
    /// The caller must prove that it already holds a write lock for
    /// this `Registry` by passing a mutable reference to this
    /// `Registry`'s storage, obtained from the write guard returned
    /// by a previous call to [`write`], as the `guard` parameter.
    pub fn unregister_locked(&self, id: I, guard: &mut Storage<T, I>) -> Option<T> {
        let value = guard.remove(id);
        //Note: careful about the order here!
        self.identity.free(id);
        //Returning None is legal if it's an error ID
        value
    }

    /// Unregister the resource at `id` and return its value, if any.
    ///
    /// The caller must present a mutable reference to a `Token<A>`,
    /// for some type `A` that comes before this `Registry`'s resource
    /// type `T` in the lock ordering.
    ///
    /// This returns a `Token<T>`, but it's almost useless, because it
    /// doesn't return a lock guard to go with it: its only effect is
    /// to make the token you passed to this function inaccessible.
    /// However, the `Token<T>` can be used to satisfy some functions'
    /// bureacratic expectations that you will have one available.
    ///
    /// The borrow checker ensures that the caller cannot again access
    /// its `Token<A>` until it has dropped both the guard and the
    /// `Token<T>`.
    ///
    /// See the [`Hub`] type for more details on locking.
    ///
    /// [`Hub`]: crate::hub::Hub
    pub(crate) fn unregister<'a, A: Access<T>>(
        &self,
        id: I,
        _token: &'a mut Token<A>,
    ) -> (Option<T>, Token<'a, T>) {
        let value = self.data.write().remove(id);
        //Note: careful about the order here!
        self.identity.free(id);
        //Returning None is legal if it's an error ID
        (value, Token::new())
    }

    pub fn label_for_resource(&self, id: I) -> String {
        let guard = self.data.read();

        let type_name = guard.kind;
        match guard.get(id) {
            Ok(res) => {
                let label = res.label();
                if label.is_empty() {
                    format!("<{}-{:?}>", type_name, id.unzip())
                } else {
                    label.to_string()
                }
            }
            Err(_) => format!(
                "<Invalid-{} label={}>",
                type_name,
                guard.label_for_invalid_id(id)
            ),
        }
    }
}
