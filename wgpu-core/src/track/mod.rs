mod buffer;
mod range;
mod texture;

use crate::{
    hub,
    id::{self, AnyBackend, CastBackend, TypedId, Valid},
    resource, Epoch, FastHashMap, Index, RefCount,
};

use core::borrow::Borrow;
use hashbrown::hash_map::{Entry, RawEntryMut};
use std::{
    fmt, marker::PhantomData, num::NonZeroU32, ops, vec::Drain,
};
use thiserror::Error;

pub(crate) use buffer::BufferState;
pub(crate) use texture::{TextureSelector, TextureState};

/// A single unit of state tracking. It keeps an initial
/// usage as well as the last/current one, similar to `Range`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Unit<U> {
    first: Option<U>,
    last: U,
}

impl<U: Copy> Unit<U> {
    /// Create a new unit from a given usage.
    fn new(usage: U) -> Self {
        Self {
            first: None,
            last: usage,
        }
    }

    /// Return a usage to link to.
    fn port(&self) -> U {
        self.first.unwrap_or(self.last)
    }
}

/// The main trait that abstracts away the tracking logic of
/// a particular resource type, like a buffer or a texture.
pub(crate) trait ResourceState: Clone + Default {
    /// Corresponding `HUB` identifier.
    type Id: fmt::Debug/* + TypedId*/;
    type ValidId: Clone + Eq + core::hash::Hash + fmt::Debug;
    /// A type specifying the sub-resources.
    type Selector: fmt::Debug;
    /// Usage type for a `Unit` of a sub-resource.
    type Usage: fmt::Debug;
    /// Usage type used for pending transitions.
    type Pending: fmt::Debug;

    /// Check if all the selected sub-resources have the same
    /// usage, and return it.
    ///
    /// Returns `None` if no sub-resources
    /// are intersecting with the selector, or their usage
    /// isn't consistent.
    fn query(&self, selector: Self::Selector) -> Option<Self::Usage>;

    /// Change the last usage of the selected sub-resources.
    ///
    /// If `output` is specified, it's filled with the
    /// `PendingTransition` objects corresponding to smaller
    /// sub-resource transitions. The old usage is replaced by
    /// the new one.
    ///
    /// If `output` is `None`, the old usage is extended with
    /// the new usage. The error is returned if it's not possible,
    /// specifying the conflicting transition. Extension can only
    /// be done for read-only usages.
    fn change(
        &mut self,
        id: &Self::ValidId,
        selector: Self::Selector,
        usage: Self::Usage,
        output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>>;

    /// Merge the state of this resource tracked by a different instance
    /// with the current one.
    ///
    /// Same rules for `output` apply as with `change()`: last usage state
    /// is either replaced (when `output` is provided) with a
    /// `PendingTransition` pushed to this vector, or extended with the
    /// other read-only usage, unless there is a usage conflict, and
    /// the error is generated (returning the conflict).
    fn merge(
        &mut self,
        id: &Self::ValidId,
        other: &Self,
        output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>>;

    /// Try to optimize the internal representation.
    fn optimize(&mut self);
}

/// Structure wrapping the abstract tracking state with the relevant resource
/// data, such as the reference count and the epoch.
#[derive(Clone)]
struct Resource<S> {
    ref_count: RefCount,
    state: S,
    epoch: Epoch,
}

/// A structure containing all the information about a particular resource
/// transition. User code should be able to generate a pipeline barrier
/// based on the contents.
#[derive(Debug, PartialEq)]
pub(crate) struct PendingTransition<S: ResourceState> {
    pub id: Valid<S::Id>,
    pub selector: S::Selector,
    pub usage: ops::Range<S::Pending>,
}

impl PendingTransition<BufferState> {
    /// Produce the hal barrier corresponding to the transition.
    pub fn into_hal<'a, A: hal::Api>(
        self,
        buf: &'a resource::Buffer<A>,
    ) -> hal::BufferBarrier<'a, A> {
        log::trace!("\tbuffer -> {:?}", self);
        let buffer = buf.raw.as_ref().expect("Buffer is destroyed");
        hal::BufferBarrier {
            buffer,
            usage: self.usage,
        }
    }
}

impl From<PendingTransition<BufferState>> for UsageConflict {
    fn from(e: PendingTransition<BufferState>) -> Self {
        Self::Buffer {
            id: e.id.0,
            combined_use: e.usage.end,
        }
    }
}

impl PendingTransition<TextureState> {
    /// Produce the hal barrier corresponding to the transition.
    pub fn into_hal<'a, A: hal::Api>(
        self,
        tex: &'a resource::Texture<A>,
    ) -> hal::TextureBarrier<'a, A> {
        log::trace!("\ttexture -> {:?}", self);
        let texture = tex.inner.as_raw().expect("Texture is destroyed");
        hal::TextureBarrier {
            texture,
            range: wgt::ImageSubresourceRange {
                aspect: wgt::TextureAspect::All,
                base_mip_level: self.selector.levels.start,
                mip_level_count: NonZeroU32::new(
                    self.selector.levels.end - self.selector.levels.start,
                ),
                base_array_layer: self.selector.layers.start,
                array_layer_count: NonZeroU32::new(
                    self.selector.layers.end - self.selector.layers.start,
                ),
            },
            usage: self.usage,
        }
    }
}

impl From<PendingTransition<TextureState>> for UsageConflict {
    fn from(e: PendingTransition<TextureState>) -> Self {
        Self::Texture {
            id: e.id.0,
            mip_levels: e.selector.levels.start..e.selector.levels.end,
            array_layers: e.selector.layers.start..e.selector.layers.end,
            combined_use: e.usage.end,
        }
    }
}

#[derive(Clone, Debug, Error)]
pub enum UseExtendError<U: fmt::Debug> {
    #[error("resource is invalid")]
    InvalidResource,
    #[error("total usage {0:?} is not valid")]
    Conflict(U),
}

/// A tracker for all resources of a given type.
pub(crate) struct ResourceTracker<S: ResourceState> {
    /// An association of known resource indices with their tracked states.
    map: FastHashMap<Index, Resource<S>>,
    /// Temporary storage for collecting transitions.
    temp: Vec<PendingTransition<S>>,
    /// The backend variant for all the tracked resources.
    backend: wgt::Backend,
}

impl<S: ResourceState + fmt::Debug> fmt::Debug for ResourceTracker<S> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.map
            .iter()
            .map(|(&index, res)| ((index, res.epoch), &res.state))
            .collect::<FastHashMap<_, _>>()
            .fmt(formatter)
    }
}

#[allow(
    // Explicit lifetimes are easier to reason about here.
    clippy::needless_lifetimes,
)]
impl<U: fmt::Debug, S: ResourceState<Id=U, ValidId=Valid<U>>> ResourceTracker<S>
    where S::Id: Copy + TypedId
{
    /// Create a new empty tracker.
    pub fn new(backend: wgt::Backend) -> Self {
        Self {
            map: FastHashMap::default(),
            temp: Vec::new(),
            backend,
        }
    }

    /// Remove an id from the tracked map.
    pub(crate) fn remove(&mut self, id: S::ValidId) -> bool {
        let (index, epoch, backend) = id.0.unzip();
        debug_assert_eq!(backend, self.backend);
        match self.map.remove(&index) {
            Some(resource) => {
                assert_eq!(resource.epoch, epoch);
                true
            }
            None => false,
        }
    }

    /// Removes the resource from the tracker if we are holding the last reference.
    pub(crate) fn remove_abandoned(&mut self, id: S::ValidId) -> bool {
        let (index, epoch, backend) = id.0.unzip();
        debug_assert_eq!(backend, self.backend);
        match self.map.entry(index) {
            Entry::Occupied(e) => {
                if e.get().ref_count.load() == 1 {
                    let res = e.remove();
                    assert_eq!(res.epoch, epoch, "Epoch mismatch for {:?}", id);
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Try to optimize the internal representation.
    pub(crate) fn optimize(&mut self) {
        for resource in self.map.values_mut() {
            resource.state.optimize();
        }
    }

    /// Return an iterator over used resources keys.
    pub fn used<'a>(&'a self) -> impl 'a + Iterator<Item = S::ValidId> {
        let backend = self.backend;
        self.map
            .iter()
            .map(move |(&index, resource)| Valid(S::Id::zip(index, resource.epoch, backend)))
    }

    pub fn get_ref_count(&self, id: S::ValidId) -> &RefCount {
        let (index, _, _) = id.0.unzip();
        &self.map[&index].ref_count
    }

    /// Return true if there is nothing here.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Clear the tracked contents.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Initialize a resource to be used.
    ///
    /// Returns false if the resource is already registered.
    pub(crate) fn init(
        &mut self,
        id: S::ValidId,
        ref_count: RefCount,
        state: S,
    ) -> Result<(), &S> {
        let (index, epoch, backend) = id.0.unzip();
        debug_assert_eq!(backend, self.backend);
        match self.map.entry(index) {
            Entry::Vacant(e) => {
                e.insert(Resource {
                    ref_count,
                    state,
                    epoch,
                });
                Ok(())
            }
            Entry::Occupied(e) => Err(&e.into_mut().state),
        }
    }

    /// Query the usage of a resource selector.
    ///
    /// Returns `Some(Usage)` only if this usage is consistent
    /// across the given selector.
    #[allow(unused)] // TODO: figure out if this needs to be removed
    pub fn query(&self, id: S::ValidId, selector: S::Selector) -> Option<S::Usage> {
        let (index, epoch, backend) = id.0.unzip();
        debug_assert_eq!(backend, self.backend);
        let res = self.map.get(&index)?;
        assert_eq!(res.epoch, epoch);
        res.state.query(selector)
    }

    /// Make sure that a resource is tracked, and return a mutable
    /// reference to it.
    fn get_or_insert<'a>(
        self_backend: wgt::Backend,
        map: &'a mut FastHashMap<Index, Resource<S>>,
        id: S::ValidId,
        ref_count: &RefCount,
    ) -> &'a mut Resource<S> {
        let (index, epoch, backend) = id.0.unzip();
        debug_assert_eq!(self_backend, backend);
        match map.entry(index) {
            Entry::Vacant(e) => e.insert(Resource {
                ref_count: ref_count.clone(),
                state: S::default(),
                epoch,
            }),
            Entry::Occupied(e) => {
                assert_eq!(e.get().epoch, epoch);
                e.into_mut()
            }
        }
    }

    fn get<'a>(
        self_backend: wgt::Backend,
        map: &'a mut FastHashMap<Index, Resource<S>>,
        id: S::ValidId,
    ) -> &'a mut Resource<S> {
        let (index, epoch, backend) = id.0.unzip();
        debug_assert_eq!(self_backend, backend);
        let e = map.get_mut(&index).unwrap();
        assert_eq!(e.epoch, epoch);
        e
    }

    /// Extend the usage of a specified resource.
    ///
    /// Returns conflicting transition as an error.
    pub(crate) fn change_extend(
        &mut self,
        id: S::ValidId,
        ref_count: &RefCount,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Result<(), PendingTransition<S>> {
        Self::get_or_insert(self.backend, &mut self.map, id, ref_count)
            .state
            .change(&id, selector, usage, None)
    }

    /// Replace the usage of a specified resource.
    pub(crate) fn change_replace(
        &mut self,
        id: S::ValidId,
        ref_count: &RefCount,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Drain<PendingTransition<S>> {
        let res = Self::get_or_insert(self.backend, &mut self.map, id, ref_count);
        res.state
            .change(&id, selector, usage, Some(&mut self.temp))
            .ok(); //TODO: unwrap?
        self.temp.drain(..)
    }

    /// Replace the usage of a specified already tracked resource.
    /// (panics if the resource is not yet tracked)
    pub(crate) fn change_replace_tracked(
        &mut self,
        id: S::ValidId,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Drain<PendingTransition<S>> {
        let res = Self::get(self.backend, &mut self.map, id);
        res.state
            .change(&id, selector, usage, Some(&mut self.temp))
            .ok();
        self.temp.drain(..)
    }

    /// Merge another tracker into `self` by extending the current states
    /// without any transitions.
    pub(crate) fn merge_extend(&mut self, other: &Self) -> Result<(), PendingTransition<S>> {
        debug_assert_eq!(self.backend, other.backend);
        for (&index, new) in other.map.iter() {
            match self.map.entry(index) {
                Entry::Vacant(e) => {
                    e.insert(new.clone());
                }
                Entry::Occupied(e) => {
                    assert_eq!(
                        e.get().epoch,
                        new.epoch,
                        "ID {:?} wasn't properly removed",
                        S::Id::zip(index, e.get().epoch, self.backend)
                    );
                    let id = Valid(S::Id::zip(index, new.epoch, self.backend));
                    e.into_mut().state.merge(&id, &new.state, None)?;
                }
            }
        }
        Ok(())
    }

    /// Merge another tracker, adding it's transitions to `self`.
    /// Transitions the current usage to the new one.
    pub(crate) fn merge_replace<'a>(&'a mut self, other: &'a Self) -> Drain<PendingTransition<S>> {
        for (&index, new) in other.map.iter() {
            match self.map.entry(index) {
                Entry::Vacant(e) => {
                    e.insert(new.clone());
                }
                Entry::Occupied(e) => {
                    assert_eq!(
                        e.get().epoch,
                        new.epoch,
                        "ID {:?} wasn't properly removed",
                        S::Id::zip(index, e.get().epoch, self.backend)
                    );
                    let id = Valid(S::Id::zip(index, new.epoch, self.backend));
                    e.into_mut()
                        .state
                        .merge(&id, &new.state, Some(&mut self.temp))
                        .ok(); //TODO: unwrap?
                }
            }
        }
        self.temp.drain(..)
    }

    /// Use a given resource provided by an `Id` with the specified usage.
    /// Combines storage access by 'Id' with the transition that extends
    /// the last read-only usage, if possible.
    ///
    /// Returns the old usage as an error if there is a conflict.
    pub(crate) fn use_extend<'a, T: 'a + hub::Resource>(
        &mut self,
        storage: &'a hub::Storage<T, S::Id>,
        id: S::Id,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Result<&'a T, UseExtendError<S::Pending>> {
        let item = storage
            .get(id)
            .map_err(|_| UseExtendError::InvalidResource)?;
        self.change_extend(
            Valid(id),
            item.life_guard().ref_count.as_ref().unwrap(),
            selector,
            usage,
        )
        .map(|()| item)
        .map_err(|pending| UseExtendError::Conflict(pending.usage.end))
    }

    /// Use a given resource provided by an `Id` with the specified usage.
    /// Combines storage access by 'Id' with the transition that replaces
    /// the last usage with a new one, returning an iterator over these
    /// transitions.
    pub(crate) fn use_replace<'a, T: 'a + hub::Resource>(
        &mut self,
        storage: &'a hub::Storage<T, S::Id>,
        id: S::Id,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Result<(&'a T, Drain<PendingTransition<S>>), S::Id> {
        let item = storage.get(id).map_err(|_| id)?;
        let drain = self.change_replace(
            Valid(id),
            item.life_guard().ref_count.as_ref().unwrap(),
            selector,
            usage,
        );
        Ok((item, drain))
    }
}

impl<I> ResourceState for PhantomData<id::Id<I>> {
    type Id = id::Id<I>;
    type ValidId = Valid<id::Id<I>>;
    type Selector = ();
    type Usage = ();
    type Pending = core::convert::Infallible;

    fn query(&self, _selector: Self::Selector) -> Option<Self::Usage> {
        Some(())
    }

    fn change(
        &mut self,
        _id: &Self::ValidId,
        _selector: Self::Selector,
        _usage: Self::Usage,
        _output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        Ok(())
    }

    fn merge(
        &mut self,
        _id: &Self::ValidId,
        _other: &Self,
        _output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        Ok(())
    }

    fn optimize(&mut self) {}
}

#[derive(Clone, Debug, Error)]
pub enum UseExtendError2<U: fmt::Debug> {
    #[error("total usage {0:?} is not valid")]
    Conflict(U),
}

/// A tracker for all resources of a given type.
pub(crate) struct ResourceTracker2<S: ResourceState> {
    /// An association of known resource indices with their tracked states.
    map: FastHashMap</*id::ValidId2<S::Id>*/S::ValidId, S>,
    /// Temporary storage for collecting transitions.
    temp: Vec<PendingTransition<S>>,
    /* /// The backend variant for all the tracked resources.
    backend: wgt::Backend, */
}

impl<S: ResourceState + fmt::Debug> fmt::Debug for ResourceTracker2<S> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.map
            .iter()
            // .map(|(index, res)| (/*(index, res.epoch)*/index, res/*.state*/))
            .collect::<FastHashMap<_, _>>()
            .fmt(formatter)
    }
}

#[allow(
    // Explicit lifetimes are easier to reason about here.
    clippy::needless_lifetimes,
)]
impl</*T, */T, S: ResourceState<Id=id::ValidId2<T>, ValidId=id::ValidId2<T>>/*<Id=id::ValidId2<T>, ValidId=id::ValidId2<T>>*/> ResourceTracker2<S> {
    /// Create a new empty tracker.
    pub fn new(/*backend: wgt::Backend*/) -> Self {
        Self {
            map: FastHashMap::default(),
            temp: Vec::new(),
            // backend,
        }
    }

    /// Remove an id from the tracked map.
    pub(crate) fn _remove(&mut self, index: &S::ValidId) -> bool {
        // let (index, epoch, backend) = id.0.unzip();
        // debug_assert_eq!(backend, self.backend);
        self.map.remove(&index).is_some()
    }

    /* /// Removes the resource from the tracker if we are holding the last reference.
    pub(crate) fn remove_abandoned(&mut self, index: S::ValidId) -> bool {
        // let (index, epoch, backend) = id.0.unzip();
        // debug_assert_eq!(backend, self.backend);
        match self.map.entry(index) {
            Entry::Occupied(e) => {
                if index.borrow().ref_count.load() == 1 {
                    let res = e.remove();
                    // assert_eq!(res.epoch, epoch, "Epoch mismatch for {:?}", id);
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    } */

    /// Try to optimize the internal representation.
    pub(crate) fn optimize(&mut self) {
        for state in self.map.values_mut() {
            state.optimize();
        }
    }

    /// Return an iterator over used resources keys.
    pub fn used<'a>(&'a self) -> impl 'a + Iterator<Item = &'a S::ValidId>
        where T: 'a
    {
        // let backend = self.backend;
        self.map
            .keys()
        // self.map.into_iter().map(|(key, _)| key)
    }

    /// Return true if there is nothing here.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Clear the tracked contents.
    fn clear(&mut self) {
        self.map.clear();
    }

    /* /// Initialize a resource to be used.
    ///
    /// Returns the new state if the resource is already registered.
    pub(crate) fn init(
        &mut self,
        index: S::ValidId,
        state: S,
    ) -> Result<(), S> {
        // debug_assert_eq!(backend, self.backend);
        // self.map.try_insert(index, state).map_err(|e| e.value)
        match self.map.entry(index) {
            Entry::Vacant(e) => {
                e.insert(state);
                Ok(())
            }
            Entry::Occupied(_) => Err(state),
        }
    } */

    /// Query the usage of a resource selector.
    ///
    /// Returns `Some(Usage)` only if this usage is consistent
    /// across the given selector.
    #[allow(unused)] // TODO: figure out if this needs to be removed
    pub fn query(&self, index: &S::ValidId, selector: S::Selector) -> Option<S::Usage> {
        // let (index, epoch, backend) = id.0.unzip();
        // debug_assert_eq!(backend, self.backend);
        let state = self.map.get(index)?;
        // assert_eq!(res.epoch, epoch);
        state.query(selector)
    }

    /// Make sure that a resource is tracked, and return a mutable
    /// reference to it.
    fn get_or_insert<'a>(
        // self_backend: wgt::Backend,
        map: &'a mut FastHashMap<S::ValidId, S>,
        index: &S::ValidId,
        // id: Valid<S::Id>,
        // ref_count: &RefCount,
    ) -> &'a mut S {
        // let (index, epoch, backend) = id.0.unzip();
        // debug_assert_eq!(self_backend, backend);

        map.raw_entry_mut()
            .from_key(index)
            .or_insert_with(|| (index.clone(), Default::default()))
            .1
    }

    fn get<'a>(
        // self_backend: wgt::Backend,
        map: &'a mut FastHashMap<S::ValidId, S>,
        index: &S::ValidId,
    ) -> &'a mut S {
        // let (index, epoch, backend) = id.0.unzip();
        // debug_assert_eq!(self_backend, backend);
        let e = map.get_mut(index).unwrap();
        // assert_eq!(e.epoch, epoch);
        e
    }

    /// Extend the usage of a specified resource.
    ///
    /// Returns conflicting transition as an error.
    pub(crate) fn change_extend(
        &mut self,
        id: &S::ValidId,
        // ref_count: &RefCount,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Result<(), PendingTransition<S>> {
        Self::get_or_insert(/*self.backend, */&mut self.map, id/*, ref_count*/)
            // .state
            .change(id, selector, usage, None)
    }

    /// Replace the usage of a specified resource.
    pub(crate) fn change_replace(
        &mut self,
        id: &S::ValidId,
        // ref_count: &RefCount,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Drain<PendingTransition<S>> {
        let state = Self::get_or_insert(/*self.backend, */&mut self.map, id/*, ref_count*/);
        state.change(id, selector, usage, Some(&mut self.temp)).ok(); //TODO: unwrap?
        self.temp.drain(..)
    }

    /// Replace the usage of a specified already tracked resource.
    /// (panics if the resource is not yet tracked)
    pub(crate) fn change_replace_tracked(
        &mut self,
        id: &S::ValidId,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Drain<PendingTransition<S>> {
        let state = Self::get(/*self.backend, */&mut self.map, id);
        state.change(id, selector, usage, Some(&mut self.temp)).ok();
        self.temp.drain(..)
    }

    /// Merge another tracker into `self` by extending the current states
    /// without any transitions.
    pub(crate) fn merge_extend(&mut self, other: &Self) -> Result<(), PendingTransition<S>> {
        // debug_assert_eq!(self.backend, other.backend);
        for (index, new) in other.map.iter() {
            match self.map.raw_entry_mut().from_key(index) {
                RawEntryMut::Vacant(e) => {
                    e.insert(index.clone(), new.clone());
                },
                RawEntryMut::Occupied(e) => {
                    /* assert_eq!(
                        e.get().epoch,
                        new.epoch,
                        "ID {:?} wasn't properly removed",
                        S::Id::zip(index, e.get().epoch, self.backend)
                    ); */
                    // let id = Valid(S::Id::zip(index, new.epoch, self.backend));
                    e.into_mut().merge(index, new, None)?;
                }
            }
        }
        Ok(())
    }

    /// Merge another tracker, adding it's transitions to `self`.
    /// Transitions the current usage to the new one.
    pub(crate) fn merge_replace<'a>(&'a mut self, other: &'a Self) -> Drain<PendingTransition<S>> {
        for (index, new) in other.map.iter() {
            match self.map.raw_entry_mut().from_key(index) {
                RawEntryMut::Vacant(e) => {
                    e.insert(index.clone(), new.clone());
                }
                RawEntryMut::Occupied(e) => {
                    /* assert_eq!(
                        e.get().epoch,
                        new.epoch,
                        "ID {:?} wasn't properly removed",
                        S::Id::zip(index, e.get().epoch, self.backend)
                    );
                    let id = Valid(S::Id::zip(index, new.epoch, self.backend)); */
                    e.into_mut()
                        .merge(index, new, Some(&mut self.temp))
                        .ok(); //TODO: unwrap?
                }
            }
        }
        self.temp.drain(..)
    }

    /// Use a given resource provided by an `Id` with the specified usage.
    /// Combines storage access by 'Id' with the transition that extends
    /// the last read-only usage, if possible.
    ///
    /// Returns the old usage as an error if there is a conflict.
    pub(crate) fn use_extend<'a/*, T: 'a + hub::Resource*/, A, U: CastBackend<A, Output=T>>(
        &mut self,
        // storage: &'a hub::Storage<T, S::Id>,
        item: id::IdGuard<'a, A, U>,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Result<id::IdGuard<'a, A, U>, UseExtendError2<S::Pending>> {
        /* let item = storage
            .get(id)
            .map_err(|_| UseExtendError::InvalidResource)?; */

        // FIXME: Add this
        // if item.device() != device {
        //    return Err(UseExtendError::InvalidResource);
        // }

        self.change_extend(
            /*Valid(id)*/item.borrow(),
            // item.life_guard().ref_count.as_ref().unwrap(),
            selector,
            usage,
        )
        .map(|()| item)
        .map_err(|pending| UseExtendError2::Conflict(pending.usage.end))
    }

    /// Use a given resource provided by an `Id` with the specified usage.
    /// Combines storage access by 'Id' with the transition that replaces
    /// the last usage with a new one, returning an iterator over these
    /// transitions.
    pub(crate) fn use_replace<'a/*, T: 'a + hub::Resource*/, A, U: CastBackend<A, Output=T>>(
        &mut self,
        // storage: &'a hub::Storage<T, S::Id>,
        id: &'a S::ValidId,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Result<(id::IdGuard<'a, A, U>, Drain<PendingTransition<S>>), S::Id> {
        // let item = storage.get(id).map_err(|_| id)?;
        let item = id.borrow();
        // FIXME: Add this
        // if item.device() != device {
        //    return Err(UseExtendError::InvalidResource);
        // }

        let drain = self.change_replace(
            id,
            // item.life_guard().ref_count.as_ref().unwrap(),
            selector,
            usage,
        );
        Ok((item, drain))
    }
}

impl<I> ResourceState for PhantomData<id::ValidId2<I>> {
    type Id = id::ValidId2<I>;
    type ValidId = id::ValidId2<I>;
    type Selector = ();
    type Usage = ();
    type Pending = core::convert::Infallible;

    fn query(&self, _selector: Self::Selector) -> Option<Self::Usage> {
        Some(())
    }

    fn change(
        &mut self,
        _id: &Self::ValidId,
        _selector: Self::Selector,
        _usage: Self::Usage,
        _output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        Ok(())
    }

    fn merge(
        &mut self,
        _id: &Self::ValidId,
        _other: &Self,
        _output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        Ok(())
    }

    fn optimize(&mut self) {}
}


pub const DUMMY_SELECTOR: () = ();

#[derive(Clone, Debug, Error)]
pub enum UsageConflict {
    #[error(
        "Attempted to use buffer {id:?} as a combination of {combined_use:?} within a usage scope."
    )]
    Buffer {
        id: id::BufferId,
        combined_use: hal::BufferUses,
    },
    #[error("Attempted to use texture {id:?} mips {mip_levels:?} layers {array_layers:?} as a combination of {combined_use:?} within a usage scope.")]
    Texture {
        id: id::TextureId,
        mip_levels: ops::Range<u32>,
        array_layers: ops::Range<u32>,
        combined_use: hal::TextureUses,
    },
}

/// A set of trackers for all relevant resources.
#[derive(Debug)]
pub(crate) struct TrackerSet<A: hal::Api> {
    pub buffers: ResourceTracker<BufferState>,
    pub textures: ResourceTracker<TextureState>,
    pub views: ResourceTracker<PhantomData<id::TextureViewId>>,
    pub bind_groups: ResourceTracker2<PhantomData<id::ValidId2<crate::binding_model::BindGroup<A>>>>,
    pub samplers: ResourceTracker2<PhantomData<id::ValidId2<crate::resource::Sampler<A>>>>,
    pub compute_pipes: ResourceTracker2<PhantomData<id::ValidId2<crate::pipeline::ComputePipeline<A>>>>,
    pub render_pipes: ResourceTracker2<PhantomData<id::ValidId2<crate::pipeline::RenderPipeline<A>>>>,
    pub bundles: ResourceTracker2<PhantomData<id::ValidId2<crate::command::RenderBundle<A>>>>,
    pub query_sets: ResourceTracker2<PhantomData<id::ValidId2<crate::resource::QuerySet<A>>>>,
}

impl<A: hal::Api> TrackerSet<A> {
    /// Create an empty set.
    pub fn new(backend: wgt::Backend) -> Self {
        Self {
            buffers: ResourceTracker::new(backend),
            textures: ResourceTracker::new(backend),
            views: ResourceTracker::new(backend),
            bind_groups: ResourceTracker2::new(),
            samplers: ResourceTracker2::new(),
            compute_pipes: ResourceTracker2::new(),
            render_pipes: ResourceTracker2::new(),
            bundles: ResourceTracker2::new(),
            query_sets: ResourceTracker2::new(),
        }
    }

    /// Clear all the trackers.
    pub fn _clear(&mut self) {
        self.buffers.clear();
        self.textures.clear();
        self.views.clear();
        self.bind_groups.clear();
        self.samplers.clear();
        self.compute_pipes.clear();
        self.render_pipes.clear();
        self.bundles.clear();
        self.query_sets.clear();
    }

    /// Try to optimize the tracking representation.
    pub fn optimize(&mut self) {
        self.buffers.optimize();
        self.textures.optimize();
        self.views.optimize();
        self.bind_groups.optimize();
        self.samplers.optimize();
        self.compute_pipes.optimize();
        self.render_pipes.optimize();
        self.bundles.optimize();
        self.query_sets.optimize();
    }

    /// Merge only the stateful trackers of another instance by extending
    /// the usage. Returns a conflict if any.
    pub fn merge_extend_stateful(&mut self, other: &Self) -> Result<(), UsageConflict> {
        self.buffers.merge_extend(&other.buffers)?;
        self.textures.merge_extend(&other.textures)?;
        Ok(())
    }

    pub fn backend(&self) -> wgt::Backend {
        self.buffers.backend
    }

    #[inline]
    pub fn trace_resources<'a, 'b: 'a, E>(
        &self,
        mut f: impl for<'c> FnMut(id::Cached<A, id::IdGuardCon<'c>>,
    ) -> Result<(), E>) -> Result<(), E>
        where A: crate::hub::HalApi + 'b,
    {
        // FIXME: Uncomment each line as it becomes available.
        // self.buffers.used().try_for_each(|id| f(id::Cached::upcast(id)))?;
        // self.textures.used().try_for_each(|id| f(id::Cached::upcast(id)))?;
        // self.views.used().try_for_each(|id| f(id::Cached::upcast(id)))?
        self.bind_groups.used().try_for_each(|id| f(crate::binding_model::BindGroup::upcast(id.borrow())))?;
        self.samplers.used().try_for_each(|id| f(crate::resource::Sampler::upcast(id.borrow())))?;
        self.compute_pipes.used().try_for_each(|id| f(crate::pipeline::ComputePipeline::upcast(id.borrow())))?;
        self.render_pipes.used().try_for_each(|id| f(crate::pipeline::RenderPipeline::upcast(id.borrow())))?;
        self.bundles.used().try_for_each(|id| f(crate::command::RenderBundle::upcast(id.borrow())))?;
        self.query_sets.used().try_for_each(|id| f(crate::resource::QuerySet::upcast(id.borrow())))
    }
}

#[derive(Debug)]
pub(crate) struct StatefulTrackerSubset {
    pub buffers: ResourceTracker<BufferState>,
    pub textures: ResourceTracker<TextureState>,
}

impl StatefulTrackerSubset {
    /// Create an empty set.
    pub fn new(backend: wgt::Backend) -> Self {
        Self {
            buffers: ResourceTracker::new(backend),
            textures: ResourceTracker::new(backend),
        }
    }

    /// Clear all the trackers.
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.textures.clear();
    }

    /// Merge all the trackers of another tracker the usage.
    pub fn merge_extend<A: hal::Api>(&mut self, other: &TrackerSet<A>) -> Result<(), UsageConflict> {
        self.buffers.merge_extend(&other.buffers)?;
        self.textures.merge_extend(&other.textures)?;
        Ok(())
    }
}
