/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

mod buffer;
mod range;
mod texture;

use crate::{
    conv, hub,
    id::{self, TypedId, Valid},
    resource, Epoch, FastHashMap, Index, RefCount,
};

use std::{collections::hash_map::Entry, fmt, marker::PhantomData, ops, vec::Drain};
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
        id: Valid<Self::Id>,
        selector: Self::Selector,
        usage: Self::Usage,
        output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>>;

    /// Sets up the first usage of the selected sub-resources.
    fn prepend(
        &mut self,
        id: Valid<Self::Id>,
        selector: Self::Selector,
        usage: Self::Usage,
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
        id: Valid<Self::Id>,
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
    pub usage: ops::Range<S::Usage>,
}

impl PendingTransition<BufferState> {
    /// Produce the gfx-hal barrier corresponding to the transition.
    pub fn into_hal<'a, B: hal::Backend>(
        self,
        buf: &'a resource::Buffer<B>,
    ) -> hal::memory::Barrier<'a, B> {
        log::trace!("\tbuffer -> {:?}", self);
        let &(ref target, _) = buf.raw.as_ref().expect("Buffer is destroyed");
        hal::memory::Barrier::Buffer {
            states: conv::map_buffer_state(self.usage.start)
                ..conv::map_buffer_state(self.usage.end),
            target,
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
        let &(ref target, _) = tex.raw.as_ref().expect("Texture is destroyed");
        let aspects = tex.aspects;
        hal::memory::Barrier::Image {
            states: conv::map_texture_state(self.usage.start, aspects)
                ..conv::map_texture_state(self.usage.end, aspects),
            target,
            range: hal::image::SubresourceRange {
                aspects,
                level_start: self.selector.levels.start,
                level_count: Some(self.selector.levels.end - self.selector.levels.start),
                layer_start: self.selector.layers.start,
                layer_count: Some(self.selector.layers.end - self.selector.layers.start),
            },
            families: None,
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
impl<S: ResourceState> ResourceTracker<S> {
    /// Create a new empty tracker.
    pub fn new(backend: wgt::Backend) -> Self {
        Self {
            map: FastHashMap::default(),
            temp: Vec::new(),
            backend,
        }
    }

    /// Remove an id from the tracked map.
    pub(crate) fn remove(&mut self, id: Valid<S::Id>) -> bool {
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
    pub(crate) fn remove_abandoned(&mut self, id: Valid<S::Id>) -> bool {
        let (index, epoch, backend) = id.0.unzip();
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
    pub(crate) fn optimize(&mut self) {
        for resource in self.map.values_mut() {
            resource.state.optimize();
        }
    }

    /// Return an iterator over used resources keys.
    pub fn used<'a>(&'a self) -> impl 'a + Iterator<Item = Valid<S::Id>> {
        let backend = self.backend;
        self.map
            .iter()
            .map(move |(&index, resource)| Valid(S::Id::zip(index, resource.epoch, backend)))
    }

    /// Clear the tracked contents.
    fn clear(&mut self) {
        self.map.clear();
    }

    /// Initialize a resource to be used.
    ///
    /// Returns false if the resource is already registered.
    pub(crate) fn init(
        &mut self,
        id: Valid<S::Id>,
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
    pub fn query(&self, id: Valid<S::Id>, selector: S::Selector) -> Option<S::Usage> {
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
        id: Valid<S::Id>,
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
        id: Valid<S::Id>,
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
        id: Valid<S::Id>,
        ref_count: &RefCount,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Result<(), PendingTransition<S>> {
        Self::get_or_insert(self.backend, &mut self.map, id, ref_count)
            .state
            .change(id, selector, usage, None)
    }

    /// Replace the usage of a specified resource.
    pub(crate) fn change_replace(
        &mut self,
        id: Valid<S::Id>,
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

    /// Replace the usage of a specified already tracked resource.
    /// (panics if the resource is not yet tracked)
    pub(crate) fn change_replace_tracked(
        &mut self,
        id: Valid<S::Id>,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Drain<PendingTransition<S>> {
        let res = Self::get(self.backend, &mut self.map, id);
        res.state
            .change(id, selector, usage, Some(&mut self.temp))
            .ok();
        self.temp.drain(..)
    }

    /// Turn the tracking from the "expand" mode into the "replace" one,
    /// installing the selected usage as the "first".
    /// This is a special operation only used by the render pass attachments.
    pub(crate) fn prepend(
        &mut self,
        id: Valid<S::Id>,
        ref_count: &RefCount,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Result<(), PendingTransition<S>> {
        Self::get_or_insert(self.backend, &mut self.map, id, ref_count)
            .state
            .prepend(id, selector, usage)
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
                    e.into_mut().state.merge(id, &new.state, None)?;
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
    pub(crate) fn use_extend<'a, T: 'a + hub::Resource>(
        &mut self,
        storage: &'a hub::Storage<T, S::Id>,
        id: S::Id,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Result<&'a T, UseExtendError<S::Usage>> {
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

impl<I: Copy + fmt::Debug + TypedId> ResourceState for PhantomData<I> {
    type Id = I;
    type Selector = ();
    type Usage = ();

    fn query(&self, _selector: Self::Selector) -> Option<Self::Usage> {
        Some(())
    }

    fn change(
        &mut self,
        _id: Valid<Self::Id>,
        _selector: Self::Selector,
        _usage: Self::Usage,
        _output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        Ok(())
    }

    fn prepend(
        &mut self,
        _id: Valid<Self::Id>,
        _selector: Self::Selector,
        _usage: Self::Usage,
    ) -> Result<(), PendingTransition<Self>> {
        Ok(())
    }

    fn merge(
        &mut self,
        _id: Valid<Self::Id>,
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
        combined_use: resource::BufferUse,
    },
    #[error("Attempted to use texture {id:?} mips {mip_levels:?} layers {array_layers:?} as a combination of {combined_use:?} within a usage scope.")]
    Texture {
        id: id::TextureId,
        mip_levels: ops::Range<u32>,
        array_layers: ops::Range<u32>,
        combined_use: resource::TextureUse,
    },
}

/// A set of trackers for all relevant resources.
#[derive(Debug)]
pub(crate) struct TrackerSet {
    pub buffers: ResourceTracker<BufferState>,
    pub textures: ResourceTracker<TextureState>,
    pub views: ResourceTracker<PhantomData<id::TextureViewId>>,
    pub bind_groups: ResourceTracker<PhantomData<id::BindGroupId>>,
    pub samplers: ResourceTracker<PhantomData<id::SamplerId>>,
    pub compute_pipes: ResourceTracker<PhantomData<id::ComputePipelineId>>,
    pub render_pipes: ResourceTracker<PhantomData<id::RenderPipelineId>>,
    pub bundles: ResourceTracker<PhantomData<id::RenderBundleId>>,
    pub query_sets: ResourceTracker<PhantomData<id::QuerySetId>>,
}

impl TrackerSet {
    /// Create an empty set.
    pub fn new(backend: wgt::Backend) -> Self {
        Self {
            buffers: ResourceTracker::new(backend),
            textures: ResourceTracker::new(backend),
            views: ResourceTracker::new(backend),
            bind_groups: ResourceTracker::new(backend),
            samplers: ResourceTracker::new(backend),
            compute_pipes: ResourceTracker::new(backend),
            render_pipes: ResourceTracker::new(backend),
            bundles: ResourceTracker::new(backend),
            query_sets: ResourceTracker::new(backend),
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

    /// Merge all the trackers of another instance by extending
    /// the usage. Panics on a conflict.
    pub fn merge_extend(&mut self, other: &Self) -> Result<(), UsageConflict> {
        self.buffers
            .merge_extend(&other.buffers)
            .map_err(|e| UsageConflict::Buffer {
                id: e.id.0,
                combined_use: e.usage.end,
            })?;
        self.textures
            .merge_extend(&other.textures)
            .map_err(|e| UsageConflict::Texture {
                id: e.id.0,
                mip_levels: e.selector.levels.start as u32..e.selector.levels.end as u32,
                array_layers: e.selector.layers.start as u32..e.selector.layers.end as u32,
                combined_use: e.usage.end,
            })?;
        self.views.merge_extend(&other.views).unwrap();
        self.bind_groups.merge_extend(&other.bind_groups).unwrap();
        self.samplers.merge_extend(&other.samplers).unwrap();
        self.compute_pipes
            .merge_extend(&other.compute_pipes)
            .unwrap();
        self.render_pipes.merge_extend(&other.render_pipes).unwrap();
        self.bundles.merge_extend(&other.bundles).unwrap();
        self.query_sets.merge_extend(&other.query_sets).unwrap();
        Ok(())
    }

    pub fn backend(&self) -> wgt::Backend {
        self.buffers.backend
    }
}
