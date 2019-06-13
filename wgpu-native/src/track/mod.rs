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
    marker::PhantomData,
    ops::Range,
    vec::Drain,
};

use buffer::BufferState;
use texture::TextureStates;


/// A single unit of state tracking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Unit<U> {
    init: U,
    last: U,
}

impl<U: Copy> Unit<U> {
    fn new(usage: U) -> Self {
        Unit {
            init: usage,
            last: usage,
        }
    }

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

pub trait ResourceState: Clone + Default {
    type Id: Copy + TypedId;
    type Selector;
    type Usage;

    fn query(
        &self,
        selector: Self::Selector,
    ) -> Option<Self::Usage>;

    fn change(
        &mut self,
        id: Self::Id,
        selector: Self::Selector,
        usage: Self::Usage,
        output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>>;

    fn merge(
        &mut self,
        id: Self::Id,
        other: &Self,
        stitch: Stitch,
        output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>>;
}

#[derive(Clone, Debug)]
struct Resource<S> {
    ref_count: RefCount,
    state: S,
    epoch: Epoch,
}

#[derive(Debug)]
pub struct PendingTransition<S: ResourceState> {
    pub id: S::Id,
    pub selector: S::Selector,
    pub usage: Range<S::Usage>,
}

pub struct ResourceTracker<S: ResourceState> {
    /// An association of known resource indices with their tracked states.
    map: FastHashMap<Index, Resource<S>>,
    /// Temporary storage for collecting transitions.
    temp: Vec<PendingTransition<S>>,
}

impl<S: ResourceState> ResourceTracker<S> {
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

    /// Return an iterator over used resources keys.
    pub fn used<'a>(&'a self) -> impl 'a + Iterator<Item = S::Id> {
        self.map
            .iter()
            .map(|(&index, resource)| S::Id::new(index, resource.epoch))
    }

    fn clear(&mut self) {
        self.map.clear();
    }

    /// Initialize a resource to be used.
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

    /// Query a resource selector. Returns `Some(Usage)` only if
    /// this usage is consistent across the given selector.
    pub fn query(
        &mut self,
        id: S::Id,
        selector: S::Selector,
    ) -> Option<S::Usage> {
        let res = self.map.get(&id.index())?;
        assert_eq!(res.epoch, id.epoch());
        res.state.query(selector)
    }

    fn grab<'a>(
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
    pub fn change_extend(
        &mut self,
        id: S::Id,
        ref_count: &RefCount,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Result<(), PendingTransition<S>> {
        Self::grab(&mut self.map, id, ref_count)
            .state.change(id, selector, usage, None)
    }

    /// Replace the usage of a specified resource.
    pub fn change_replace(
        &mut self,
        id: S::Id,
        ref_count: &RefCount,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Result<Drain<PendingTransition<S>>, PendingTransition<S>> {
        let res = Self::grab(&mut self.map, id, ref_count);
        res.state.change(id, selector, usage, Some(&mut self.temp))?;
        Ok(self.temp.drain(..))
    }

    /// Merge another tacker into `self` by extending the current states
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

    /// Merge another tacker, adding it's transitions to `self`.
    /// Transitions the current usage to the new one.
    pub fn merge_replace<'a>(
        &'a mut self,
        other: &'a Self,
        stitch: Stitch,
    ) -> Result<Drain<PendingTransition<S>>, PendingTransition<S>> {
        for (&index, new) in other.map.iter() {
            match self.map.entry(index) {
                Entry::Vacant(e) => {
                    e.insert(new.clone());
                }
                Entry::Occupied(e) => {
                    assert_eq!(e.get().epoch, new.epoch);
                    let id = S::Id::new(index, new.epoch);
                    e.into_mut().state.merge(id, &new.state, stitch, Some(&mut self.temp))?;
                }
            }
        }
        Ok(self.temp.drain(..))
    }

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

    pub fn use_replace<'a, T: 'a + Borrow<RefCount>>(
        &mut self,
        storage: &'a Storage<T, S::Id>,
        id: S::Id,
        selector: S::Selector,
        usage: S::Usage,
    ) -> Result<(&'a T, Drain<PendingTransition<S>>), S::Usage> {
        let item = &storage[id];
        self.change_replace(id, item.borrow(), selector, usage)
            .map(|drain| (item, drain))
            .map_err(|pending| pending.usage.start)
    }
}


impl<I: Copy + TypedId> ResourceState for PhantomData<I> {
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
}


pub struct TrackerSet {
    pub buffers: ResourceTracker<BufferState>,
    pub textures: ResourceTracker<TextureStates>,
    pub views: ResourceTracker<PhantomData<TextureViewId>>,
    pub bind_groups: ResourceTracker<PhantomData<BindGroupId>>,
    //TODO: samplers
}

impl TrackerSet {
    pub fn new() -> Self {
        TrackerSet {
            buffers: ResourceTracker::new(),
            textures: ResourceTracker::new(),
            views: ResourceTracker::new(),
            bind_groups: ResourceTracker::new(),
        }
    }

    pub fn clear(&mut self) {
        self.buffers.clear();
        self.textures.clear();
        self.views.clear();
        self.bind_groups.clear();
    }

    pub fn merge_extend(&mut self, other: &Self) {
        self.buffers.merge_extend(&other.buffers).unwrap();
        self.textures.merge_extend(&other.textures).unwrap();
        self.views.merge_extend(&other.views).unwrap();
        self.bind_groups.merge_extend(&other.bind_groups).unwrap();
    }
}
