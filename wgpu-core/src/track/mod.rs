/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

mod buffer;
mod range;
mod texture;

use crate::{
    conv,
    hub::Storage,
    id::{self, TypedId},
    resource, Epoch, FastHashMap, Index, RefCount,
};

use std::{
    borrow::Borrow, collections::hash_map::Entry, fmt, marker::PhantomData, ops, vec::Drain,
};

pub use buffer::BufferState;
pub use texture::TextureState;

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
        Unit {
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
pub trait ResourceState: Clone + Default {
    /// Corresponding `HUB` identifier.
    type Id: Copy + fmt::Debug + TypedId;
    /// A type specifying the sub-resources.
    type Selector: fmt::Debug;
    /// Usage type for a `Unit` of a sub-resource.
    type Usage: fmt::Debug;

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
        id: Self::Id,
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
        id: Self::Id,
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
pub struct PendingTransition<S: ResourceState> {
    pub id: S::Id,
    pub selector: S::Selector,
    pub usage: ops::Range<S::Usage>,
}

impl PendingTransition<BufferState> {
    /// Produce the gfx-hal barrier corresponding to the transition.
    pub fn into_hal<'a, B: hal::Backend>(
        self,
        buf: &'a resource::Buffer<B>,
    ) -> hal::memory::Barrier<'a, B> {
        log::trace!("\tbuffer -> {:?}", self);
        hal::memory::Barrier::Buffer {
            states: conv::map_buffer_state(self.usage.start)
                ..conv::map_buffer_state(self.usage.end),
            target: &buf.raw,
            range: hal::buffer::SubRange::WHOLE,
            families: None,
        }
    }
}

impl PendingTransition<TextureState> {
    /// Produce the gfx-hal barrier corresponding to the transition.
    pub fn into_hal<'a, B: hal::Backend>(
        self,
        tex: &'a resource::Texture<B>,
    ) -> hal::memory::Barrier<'a, B> {
        log::trace!("\ttexture -> {:?}", self);
        let aspects = tex.full_range.aspects;
        hal::memory::Barrier::Image {
            states: conv::map_texture_state(self.usage.start, aspects)
                ..conv::map_texture_state(self.usage.end, aspects),
            target: &tex.raw,
            range: hal::image::SubresourceRange {
                aspects,
                ..self.selector
            },
            families: None,
        }
    }
}

/// A tracker for all resources of a given type.
pub struct ResourceTracker<S: ResourceState> {
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

impl<S: ResourceState> ResourceTracker<S> {
    /// Create a new empty tracker.
    pub fn new(backend: wgt::Backend) -> Self {
        ResourceTracker {
            map: FastHashMap::default(),
            temp: Vec::new(),
            backend,
        }
    }

    /// Remove an id from the tracked map.
    pub fn remove(&mut self, id: S::Id) -> bool {
        let (index, epoch, backend) = id.unzip();
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
    pub fn remove_abandoned(&mut self, id: S::Id) -> bool {
        let (index, epoch, backend) = id.unzip();
        debug_assert_eq!(backend, self.backend);
        match self.map.entry(index) {
            Entry::Occupied(e) => {
                if e.get().ref_count.load() == 1 {
                    let res = e.remove();
                    assert_eq!(res.epoch, epoch);
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Try to optimize the internal representation.
    pub fn optimize(&mut self) {
        for resource in self.map.values_mut() {
            resource.state.optimize();
        }
    }

    /// Return an iterator over used resources keys.
    pub fn used<'a>(&'a self) -> impl 'a + Iterator<Item = S::Id> {
        let backend = self.backend;
        self.map
            .iter()
            .map(move |(&index, resource)| S::Id::zip(index, resource.epoch, backend))
    }

    /// Clear the tracked contents.
    fn clear(&mut self) {
        self.map.clear();
    }

    /// Returns true if the tracker is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Initialize a resource to be used.
    ///
    /// Returns false if the resource is already registered.
    pub fn init(&mut self, id: S::Id, ref_count: RefCount, state: S) -> Result<(), &S> {
        let (index, epoch, backend) = id.unzip();
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
    pub fn query(&self, id: S::Id, selector: S::Selector) -> Option<S::Usage> {
        let (index, epoch, backend) = id.unzip();
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
        id: S::Id,
        ref_count: &RefCount,
    ) -> &'a mut Resource<S> {
        let (index, epoch, backend) = id.unzip();
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

    /// Extend the usage of a specified resource.
    ///
    /// Returns conflicting transition as an error.
    pub fn change_extend(
        &mut self,
        id: S::Id,
        ref_count: &RefCount,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Result<(), PendingTransition<S>> {
        Self::get_or_insert(self.backend, &mut self.map, id, ref_count)
            .state
            .change(id, selector, usage, None)
    }

    /// Replace the usage of a specified resource.
    pub fn change_replace(
        &mut self,
        id: S::Id,
        ref_count: &RefCount,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Drain<PendingTransition<S>> {
        let res = Self::get_or_insert(self.backend, &mut self.map, id, ref_count);
        res.state
            .change(id, selector, usage, Some(&mut self.temp))
            .ok(); //TODO: unwrap?
        self.temp.drain(..)
    }

    /// Merge another tracker into `self` by extending the current states
    /// without any transitions.
    pub fn merge_extend(&mut self, other: &Self) -> Result<(), PendingTransition<S>> {
        debug_assert_eq!(self.backend, other.backend);
        for (&index, new) in other.map.iter() {
            match self.map.entry(index) {
                Entry::Vacant(e) => {
                    e.insert(new.clone());
                }
                Entry::Occupied(e) => {
                    assert_eq!(e.get().epoch, new.epoch);
                    let id = S::Id::zip(index, new.epoch, self.backend);
                    e.into_mut().state.merge(id, &new.state, None)?;
                }
            }
        }
        Ok(())
    }

    /// Merge another tracker, adding it's transitions to `self`.
    /// Transitions the current usage to the new one.
    pub fn merge_replace<'a>(&'a mut self, other: &'a Self) -> Drain<PendingTransition<S>> {
        for (&index, new) in other.map.iter() {
            match self.map.entry(index) {
                Entry::Vacant(e) => {
                    e.insert(new.clone());
                }
                Entry::Occupied(e) => {
                    assert_eq!(e.get().epoch, new.epoch);
                    let id = S::Id::zip(index, new.epoch, self.backend);
                    e.into_mut()
                        .state
                        .merge(id, &new.state, Some(&mut self.temp))
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
    pub fn use_extend<'a, T: 'a + Borrow<RefCount>>(
        &mut self,
        storage: &'a Storage<T, S::Id>,
        id: S::Id,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Result<&'a T, S::Usage> {
        let item = &storage[id];
        self.change_extend(id, item.borrow(), selector, usage)
            .map(|()| item)
            .map_err(|pending| pending.usage.start)
    }

    /// Use a given resource provided by an `Id` with the specified usage.
    /// Combines storage access by 'Id' with the transition that replaces
    /// the last usage with a new one, returning an iterator over these
    /// transitions.
    pub fn use_replace<'a, T: 'a + Borrow<RefCount>>(
        &mut self,
        storage: &'a Storage<T, S::Id>,
        id: S::Id,
        selector: S::Selector,
        usage: S::Usage,
    ) -> (&'a T, Drain<PendingTransition<S>>) {
        let item = &storage[id];
        let drain = self.change_replace(id, item.borrow(), selector, usage);
        (item, drain)
    }
}

impl<I: Copy + fmt::Debug + TypedId> ResourceState for PhantomData<I> {
    type Id = I;
    type Selector = ();
    type Usage = ();

    fn query(&self, _selector: Self::Selector) -> Option<Self::Usage> {
        Some(())
    }

    fn change(
        &mut self,
        _id: Self::Id,
        _selector: Self::Selector,
        _usage: Self::Usage,
        _output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        Ok(())
    }

    fn merge(
        &mut self,
        _id: Self::Id,
        _other: &Self,
        _output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        Ok(())
    }

    fn optimize(&mut self) {}
}

pub const DUMMY_SELECTOR: () = ();

/// A set of trackers for all relevant resources.
#[derive(Debug)]
pub struct TrackerSet {
    pub buffers: ResourceTracker<BufferState>,
    pub textures: ResourceTracker<TextureState>,
    pub views: ResourceTracker<PhantomData<id::TextureViewId>>,
    pub bind_groups: ResourceTracker<PhantomData<id::BindGroupId>>,
    pub samplers: ResourceTracker<PhantomData<id::SamplerId>>,
    pub compute_pipes: ResourceTracker<PhantomData<id::ComputePipelineId>>,
    pub render_pipes: ResourceTracker<PhantomData<id::RenderPipelineId>>,
}

impl TrackerSet {
    /// Create an empty set.
    pub fn new(backend: wgt::Backend) -> Self {
        TrackerSet {
            buffers: ResourceTracker::new(backend),
            textures: ResourceTracker::new(backend),
            views: ResourceTracker::new(backend),
            bind_groups: ResourceTracker::new(backend),
            samplers: ResourceTracker::new(backend),
            compute_pipes: ResourceTracker::new(backend),
            render_pipes: ResourceTracker::new(backend),
        }
    }

    /// Clear all the trackers.
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.textures.clear();
        self.views.clear();
        self.bind_groups.clear();
        self.samplers.clear();
        self.compute_pipes.clear();
        self.render_pipes.clear();
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
    }

    /// Merge all the trackers of another instance by extending
    /// the usage. Panics on a conflict.
    pub fn merge_extend(&mut self, other: &Self) {
        self.buffers.merge_extend(&other.buffers).unwrap();
        self.textures.merge_extend(&other.textures).unwrap();
        self.views.merge_extend(&other.views).unwrap();
        self.bind_groups.merge_extend(&other.bind_groups).unwrap();
        self.samplers.merge_extend(&other.samplers).unwrap();
        self.compute_pipes
            .merge_extend(&other.compute_pipes)
            .unwrap();
        self.render_pipes.merge_extend(&other.render_pipes).unwrap();
    }

    pub fn backend(&self) -> wgt::Backend {
        self.buffers.backend
    }
}
