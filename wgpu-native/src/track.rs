use crate::{
    hub::Storage,
    resource::{BufferUsage, TextureUsage},
    BufferId,
    Epoch,
    Index,
    RefCount,
    TextureId,
    TextureViewId,
    TypedId,
    BindGroupId,
};

use bitflags::bitflags;
use hal::backend::FastHashMap;

use std::{
    borrow::Borrow,
    collections::hash_map::{Entry, Iter},
    marker::PhantomData,
    mem,
    ops::{BitOr, Range},
};

#[derive(Clone, Debug)]
pub struct RangedStates<I, T> {
    ranges: Vec<(Range<I>, T)>,
}

pub type TextureLayerStates = RangedStates<hal::image::Layer, TextureUsage>;
pub type TextureStates = RangedStates<hal::image::Level, TextureLayerStates>;

impl<I: Copy + PartialOrd, T: Clone> RangedStates<I, T> {
    fn isolate(&mut self, index: Range<I>) -> &mut T {
        let mut pos = self.ranges
            .iter()
            .position(|&(ref range, _)| index.start >= range.start)
            .unwrap();
        let base_range = self.ranges[pos].0.clone();
        assert!(index.end <= base_range.end);
        if base_range.start < index.start {
            let value = ((base_range.start .. index.start), self.ranges[pos].1.clone());
            self.ranges.insert(pos, value);
            pos += 1;
            self.ranges[pos].0.start = index.start;
        }
        if base_range.end > index.end {
            let value = ((index.end .. base_range.end), self.ranges[pos].1.clone());
            self.ranges.insert(pos + 1, value);
            self.ranges[pos].0.end = index.end;
        }
        &mut self.ranges[pos].1
    }
}

impl TextureStates {
    fn change_state(
        &mut self, level: hal::image::Level, layer: hal::image::Layer, usage: TextureUsage
    ) -> Option<TextureUsage> {
        let layer_states = self.isolate(level .. level + 1);
        let cur_usage = layer_states.isolate(layer .. layer + 1);
        if *cur_usage != usage {
            Some(mem::replace(cur_usage, usage))
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[allow(unused)]
pub enum Tracktion<T> {
    Init,
    Keep,
    Extend { old: T },
    Replace { old: T },
}

impl<T> Tracktion<T> {
    pub fn into_source(self) -> Option<T> {
        match self {
            Tracktion::Init | Tracktion::Keep => None,
            Tracktion::Extend { old } | Tracktion::Replace { old } => Some(old),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Query<T> {
    pub usage: T,
    pub initialized: bool,
}

bitflags! {
    pub struct TrackPermit: u32 {
        /// Allow extension of the current usage. This is useful during render pass
        /// recording, where the usage has to stay constant, but we can defer the
        /// decision on what it is until the end of the pass.
        const EXTEND = 1;
        /// Allow replacing the current usage with the new one. This is useful when
        /// recording a command buffer live, and the current usage is already been set.
        const REPLACE = 2;
    }
}

pub trait GenericUsage {
    fn is_exclusive(&self) -> bool;
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DummyUsage;
impl BitOr for DummyUsage {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        other
    }
}

impl GenericUsage for BufferUsage {
    fn is_exclusive(&self) -> bool {
        BufferUsage::WRITE_ALL.intersects(*self)
    }
}
impl GenericUsage for TextureUsage {
    fn is_exclusive(&self) -> bool {
        TextureUsage::WRITE_ALL.intersects(*self)
    }
}
impl GenericUsage for DummyUsage {
    fn is_exclusive(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug)]
struct Track<U> {
    ref_count: RefCount,
    init: U,
    last: U,
    epoch: Epoch,
}

//TODO: consider having `I` as an associated type of `U`?
#[derive(Debug)]
pub struct Tracker<I, U> {
    map: FastHashMap<Index, Track<U>>,
    _phantom: PhantomData<I>,
}
pub type BufferTracker = Tracker<BufferId, BufferUsage>;
pub type TextureTracker = Tracker<TextureId, TextureUsage>;
pub type TextureViewTracker = Tracker<TextureViewId, DummyUsage>;
pub type BindGroupTracker = Tracker<BindGroupId, DummyUsage>;

//TODO: make this a generic parameter.
/// Mode of stitching to states together.
#[derive(Clone, Copy, Debug)]
pub enum Stitch {
    /// Stitch to the init state of the other resource.
    Init,
    /// Stitch to the last state of the other resource.
    Last,
}

//TODO: consider rewriting this without any iterators that have side effects.
#[derive(Debug)]
pub struct ConsumeIterator<'a, I: TypedId, U: Copy + PartialEq> {
    src: Iter<'a, Index, Track<U>>,
    dst: &'a mut FastHashMap<Index, Track<U>>,
    stitch: Stitch,
    _marker: PhantomData<I>,
}

impl<'a, I: TypedId, U: Copy + PartialEq> Iterator for ConsumeIterator<'a, I, U> {
    type Item = (I, Range<U>);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (&index, new) = self.src.next()?;
            match self.dst.entry(index) {
                Entry::Vacant(e) => {
                    e.insert(new.clone());
                }
                Entry::Occupied(mut e) => {
                    assert_eq!(e.get().epoch, new.epoch);
                    let old = mem::replace(&mut e.get_mut().last, new.last);
                    if old != new.init {
                        let state = match self.stitch {
                            Stitch::Init => new.init,
                            Stitch::Last => new.last,
                        };
                        return Some((I::new(index, new.epoch), old .. state))
                    }
                }
            }
        }
    }
}

// Make sure to finish all side effects on drop
impl<'a, I: TypedId, U: Copy + PartialEq> Drop for ConsumeIterator<'a, I, U> {
    fn drop(&mut self) {
        self.for_each(drop)
    }
}

#[derive(Debug)]
pub struct TrackerSet {
    pub buffers: BufferTracker,
    pub textures: TextureTracker,
    pub views: TextureViewTracker,
    pub bind_groups: BindGroupTracker,
    //TODO: samplers
}

impl TrackerSet {
    pub fn new() -> Self {
        TrackerSet {
            buffers: BufferTracker::new(),
            textures: TextureTracker::new(),
            views: TextureViewTracker::new(),
            bind_groups: BindGroupTracker::new(),
        }
    }

    pub fn clear(&mut self) {
        self.buffers.clear();
        self.textures.clear();
        self.views.clear();
        self.bind_groups.clear();
    }

    pub fn consume_by_extend(&mut self, other: &Self) {
        self.buffers.consume_by_extend(&other.buffers).unwrap();
        self.textures.consume_by_extend(&other.textures).unwrap();
        self.views.consume_by_extend(&other.views).unwrap();
        self.bind_groups.consume_by_extend(&other.bind_groups).unwrap();
    }
}

impl<I: TypedId, U: Copy + GenericUsage + BitOr<Output = U> + PartialEq> Tracker<I, U> {
    pub fn new() -> Self {
        Tracker {
            map: FastHashMap::default(),
            _phantom: PhantomData,
        }
    }

    /// Remove an id from the tracked map.
    pub(crate) fn remove(&mut self, id: I) -> bool {
        match self.map.remove(&id.index()) {
            Some(track) => {
                assert_eq!(track.epoch, id.epoch());
                true
            }
            None => false,
        }
    }

    /// Get the last usage on a resource.
    pub(crate) fn query(&mut self, id: I, ref_count: &RefCount, default: U) -> Query<U> {
        match self.map.entry(id.index()) {
            Entry::Vacant(e) => {
                e.insert(Track {
                    ref_count: ref_count.clone(),
                    init: default,
                    last: default,
                    epoch: id.epoch(),
                });
                Query {
                    usage: default,
                    initialized: true,
                }
            }
            Entry::Occupied(e) => {
                assert_eq!(e.get().epoch, id.epoch());
                Query {
                    usage: e.get().last,
                    initialized: false,
                }
            }
        }
    }

    /// Transit a specified resource into a different usage.
    pub(crate) fn transit(
        &mut self,
        id: I,
        ref_count: &RefCount,
        usage: U,
        permit: TrackPermit,
    ) -> Result<Tracktion<U>, U> {
        match self.map.entry(id.index()) {
            Entry::Vacant(e) => {
                e.insert(Track {
                    ref_count: ref_count.clone(),
                    init: usage,
                    last: usage,
                    epoch: id.epoch(),
                });
                Ok(Tracktion::Init)
            }
            Entry::Occupied(mut e) => {
                assert_eq!(e.get().epoch, id.epoch());
                let old = e.get().last;
                if usage == old {
                    Ok(Tracktion::Keep)
                } else if permit.contains(TrackPermit::EXTEND) && !(old | usage).is_exclusive() {
                    e.get_mut().last = old | usage;
                    Ok(Tracktion::Extend { old })
                } else if permit.contains(TrackPermit::REPLACE) {
                    e.get_mut().last = usage;
                    Ok(Tracktion::Replace { old })
                } else {
                    Err(old)
                }
            }
        }
    }

    /// Consume another tacker, adding it's transitions to `self`.
    /// Transitions the current usage to the new one.
    pub fn consume_by_replace<'a>(
        &'a mut self,
        other: &'a Self,
        stitch: Stitch,
    ) -> ConsumeIterator<'a, I, U> {
        ConsumeIterator {
            src: other.map.iter(),
            dst: &mut self.map,
            stitch,
            _marker: PhantomData,
        }
    }

    /// Consume another tacker, adding it's transitions to `self`.
    /// Extends the current usage without doing any transitions.
    pub fn consume_by_extend<'a>(&'a mut self, other: &'a Self) -> Result<(), (I, Range<U>)> {
        for (&index, new) in other.map.iter() {
            match self.map.entry(index) {
                Entry::Vacant(e) => {
                    e.insert(new.clone());
                }
                Entry::Occupied(mut e) => {
                    assert_eq!(e.get().epoch, new.epoch);
                    let old = e.get().last;
                    if old != new.last {
                        let extended = old | new.last;
                        if extended.is_exclusive() {
                            let id = I::new(index, new.epoch);
                            return Err((id, old .. new.last));
                        }
                        e.get_mut().last = extended;
                    }
                }
            }
        }
        Ok(())
    }

    /// Return an iterator over used resources keys.
    pub fn used<'a>(&'a self) -> impl 'a + Iterator<Item = I> {
        self.map
            .iter()
            .map(|(&index, track)| I::new(index, track.epoch))
    }
}

impl<I: TypedId + Copy, U: Copy + GenericUsage + BitOr<Output = U> + PartialEq> Tracker<I, U> {
    fn clear(&mut self) {
        self.map.clear();
    }

    fn _get_with_usage<'a, T: 'a + Borrow<RefCount>>(
        &mut self,
        storage: &'a Storage<T, I>,
        id: I,
        usage: U,
        permit: TrackPermit,
    ) -> Result<(&'a T, Tracktion<U>), U> {
        let item = &storage[id];
        self.transit(id, item.borrow(), usage, permit)
            .map(|tracktion| (item, tracktion))
    }

    pub(crate) fn get_with_extended_usage<'a, T: 'a + Borrow<RefCount>>(
        &mut self,
        storage: &'a Storage<T, I>,
        id: I,
        usage: U,
    ) -> Result<&'a T, U> {
        let item = &storage[id];
        self.transit(id, item.borrow(), usage, TrackPermit::EXTEND)
            .map(|_tracktion| item)
    }

    pub(crate) fn get_with_replaced_usage<'a, T: 'a + Borrow<RefCount>>(
        &mut self,
        storage: &'a Storage<T, I>,
        id: I,
        usage: U,
    ) -> Result<(&'a T, Option<U>), U> {
        let item = &storage[id];
        self.transit(id, item.borrow(), usage, TrackPermit::REPLACE)
            .map(|tracktion| {
                (
                    item,
                    match tracktion {
                        Tracktion::Init | Tracktion::Keep => None,
                        Tracktion::Extend { .. } => unreachable!(),
                        Tracktion::Replace { old } => Some(old),
                    },
                )
            })
    }
}
