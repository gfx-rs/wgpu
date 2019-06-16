mod buffer;
mod range;
mod texture;

use crate::{
    hub::Storage,
    Epoch,
    Index,
    RefCount,
    TextureViewId,
    TypedId,
    BindGroupId,
};

use hal::backend::FastHashMap;

use std::{
    borrow::Borrow,
    collections::hash_map::Entry,
    fmt::Debug,
    marker::PhantomData,
    ops::Range,
    vec::Drain,
};

use buffer::BufferState;
use texture::TextureState;


/// A single unit of state tracking. It keeps an initial
/// usage as well as the last/current one, similar to `Range`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Unit<U> {
    init: U,
    last: U,
}

impl<U: Copy> Unit<U> {
    /// Create a new unit from a given usage.
    fn new(usage: U) -> Self {
        Unit {
            init: usage,
            last: usage,
        }
    }

    /// Select one of the ends of the usage, based on the
    /// given `Stitch`.
    ///
    /// In some scenarios, when merging two trackers
    /// A and B for a resource, we want to connect A to the initial state
    /// of B. In other scenarios, we want to reach the last state of B.
    fn select(&self, stitch: Stitch) -> U {
        match stitch {
            Stitch::Init => self.init,
            Stitch::Last => self.last,
        }
    }
}

/// Mode of stitching to states together.
#[derive(Clone, Copy, Debug)]
pub enum Stitch {
    /// Stitch to the init state of the other resource.
    Init,
    /// Stitch to the last state of the other resource.
    Last,
}

/// The main trait that abstracts away the tracking logic of
/// a particular resource type, like a buffer or a texture.
pub trait ResourceState: Clone + Default {
    /// Corresponding `HUB` identifier.
    type Id: Copy + Debug + TypedId;
    /// A type specifying the sub-resources.
    type Selector: Debug;
    /// Usage type for a `Unit` of a sub-resource.
    type Usage: Debug;

    /// Check if all the selected sub-resources have the same
    /// usage, and return it.
    ///
    /// Returns `None` if no sub-resources
    /// are intersecting with the selector, or their usage
    /// isn't consistent.
    fn query(
        &self,
        selector: Self::Selector,
    ) -> Option<Self::Usage>;

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
    ///
    /// `stitch` only defines the end points of generated transitions.
    /// Last states of `self` are nevertheless updated to the *last* states
    /// of `other`, if `output` is provided.
    fn merge(
        &mut self,
        id: Self::Id,
        other: &Self,
        stitch: Stitch,
        output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>>;

    /// Try to optimize the internal representation.
    fn optimize(&mut self);
}

/// Structure wrapping the abstract tracking state with the relevant resource
/// data, such as the reference count and the epoch.
#[derive(Clone)]
#[cfg_attr(debug_assertions, derive(Debug))]
struct Resource<S> {
    ref_count: RefCount,
    state: S,
    epoch: Epoch,
}

/// A structure containing all the information about a particular resource
/// transition. User code should be able to generate a pipeline barrier
/// based on the contents.
#[derive(Debug)]
pub struct PendingTransition<S: ResourceState> {
    pub id: S::Id,
    pub selector: S::Selector,
    pub usage: Range<S::Usage>,
}

/// A tracker for all resources of a given type.
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct ResourceTracker<S: ResourceState> {
    /// An association of known resource indices with their tracked states.
    map: FastHashMap<Index, Resource<S>>,
    /// Temporary storage for collecting transitions.
    temp: Vec<PendingTransition<S>>,
}

impl<S: ResourceState> ResourceTracker<S> {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        ResourceTracker {
            map: FastHashMap::default(),
            temp: Vec::new(),
        }
    }

    /// Remove an id from the tracked map.
    pub fn remove(&mut self, id: S::Id) -> bool {
        match self.map.remove(&id.index()) {
            Some(resource) => {
                assert_eq!(resource.epoch, id.epoch());
                true
            }
            None => false,
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
        self.map
            .iter()
            .map(|(&index, resource)| S::Id::new(index, resource.epoch))
    }

    /// Clear the tracked contents.
    fn clear(&mut self) {
        self.map.clear();
    }

    /// Initialize a resource to be used.
    ///
    /// Returns `false` if the resource is already tracked.
    pub fn init(
        &mut self,
        id: S::Id,
        ref_count: &RefCount,
        selector: S::Selector,
        default: S::Usage,
    ) -> bool {
        let mut state = S::default();
        let _ = state.change(
            id,
            selector,
            default,
            None,
        );
        self.map
            .insert(id.index(), Resource {
                ref_count: ref_count.clone(),
                state,
                epoch: id.epoch(),
            })
            .is_none()
    }

    /// Query the usage of a resource selector.
    ///
    /// Returns `Some(Usage)` only if this usage is consistent
    /// across the given selector.
    pub fn query(
        &mut self,
        id: S::Id,
        selector: S::Selector,
    ) -> Option<S::Usage> {
        let res = self.map.get(&id.index())?;
        assert_eq!(res.epoch, id.epoch());
        res.state.query(selector)
    }

    /// Make sure that a resource is tracked, and return a mutable
    /// reference to it.
    fn get_or_insert<'a>(
        map: &'a mut FastHashMap<Index, Resource<S>>,
        id: S::Id,
        ref_count: &RefCount,
    ) -> &'a mut Resource<S> {
        match map.entry(id.index()) {
            Entry::Vacant(e) => {
                e.insert(Resource {
                    ref_count: ref_count.clone(),
                    state: S::default(),
                    epoch: id.epoch(),
                })
            }
            Entry::Occupied(e) => {
                assert_eq!(e.get().epoch, id.epoch());
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
        Self::get_or_insert(&mut self.map, id, ref_count)
            .state.change(id, selector, usage, None)
    }

    /// Replace the usage of a specified resource.
    pub fn change_replace(
        &mut self,
        id: S::Id,
        ref_count: &RefCount,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Drain<PendingTransition<S>> {
        let res = Self::get_or_insert(&mut self.map, id, ref_count);
        res.state.change(id, selector, usage, Some(&mut self.temp))
            .ok(); //TODO: unwrap?
        self.temp.drain(..)
    }

    /// Merge another tracker into `self` by extending the current states
    /// without any transitions.
    pub fn merge_extend(
        &mut self, other: &Self,
    ) -> Result<(), PendingTransition<S>> {
        for (&index, new) in other.map.iter() {
            match self.map.entry(index) {
                Entry::Vacant(e) => {
                    e.insert(new.clone());
                }
                Entry::Occupied(e) => {
                    assert_eq!(e.get().epoch, new.epoch);
                    let id = S::Id::new(index, new.epoch);
                    e.into_mut().state.merge(id, &new.state, Stitch::Last, None)?;
                }
            }
        }
        Ok(())
    }

    /// Merge another tracker, adding it's transitions to `self`.
    /// Transitions the current usage to the new one.
    pub fn merge_replace<'a>(
        &'a mut self,
        other: &'a Self,
        stitch: Stitch,
    ) -> Drain<PendingTransition<S>> {
        for (&index, new) in other.map.iter() {
            match self.map.entry(index) {
                Entry::Vacant(e) => {
                    e.insert(new.clone());
                }
                Entry::Occupied(e) => {
                    assert_eq!(e.get().epoch, new.epoch);
                    let id = S::Id::new(index, new.epoch);
                    e.into_mut().state
                        .merge(id, &new.state, stitch, Some(&mut self.temp))
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


impl<I: Copy + Debug + TypedId> ResourceState for PhantomData<I> {
    type Id = I;
    type Selector = ();
    type Usage = ();

    fn query(
        &self,
        _selector: Self::Selector,
    ) -> Option<Self::Usage> {
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
        _stitch: Stitch,
        _output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        Ok(())
    }

    fn optimize(&mut self) {
    }
}


/// A set of trackers for all relevant resources.
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct TrackerSet {
    pub buffers: ResourceTracker<BufferState>,
    pub textures: ResourceTracker<TextureState>,
    pub views: ResourceTracker<PhantomData<TextureViewId>>,
    pub bind_groups: ResourceTracker<PhantomData<BindGroupId>>,
    //TODO: samplers
}

impl TrackerSet {
    /// Create an empty set.
    pub fn new() -> Self {
        TrackerSet {
            buffers: ResourceTracker::new(),
            textures: ResourceTracker::new(),
            views: ResourceTracker::new(),
            bind_groups: ResourceTracker::new(),
        }
    }

    /// Clear all the trackers.
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.textures.clear();
        self.views.clear();
        self.bind_groups.clear();
    }

    /// Try to optimize the tracking representation.
    pub fn optimize(&mut self) {
        self.buffers.optimize();
        self.textures.optimize();
        self.views.optimize();
        self.bind_groups.optimize();
    }

    /// Merge all the trackers of another instance by extending
    /// the usage. Panics on a conflict.
    pub fn merge_extend(&mut self, other: &Self) {
        self.buffers.merge_extend(&other.buffers).unwrap();
        self.textures.merge_extend(&other.textures).unwrap();
        self.views.merge_extend(&other.views).unwrap();
        self.bind_groups.merge_extend(&other.bind_groups).unwrap();
    }
}
