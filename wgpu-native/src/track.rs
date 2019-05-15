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
};

use bitflags::bitflags;
use hal::backend::FastHashMap;

use std::{
    borrow::Borrow,
    collections::hash_map::Entry,
    marker::PhantomData,
    mem,
    ops::{BitOr, Range},
};

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

#[derive(Clone)]
struct Track<U> {
    ref_count: RefCount,
    init: U,
    last: U,
    epoch: Epoch,
}

//TODO: consider having `I` as an associated type of `U`?
pub struct Tracker<I, U> {
    map: FastHashMap<Index, Track<U>>,
    _phantom: PhantomData<I>,
}
pub type BufferTracker = Tracker<BufferId, BufferUsage>;
pub type TextureTracker = Tracker<TextureId, TextureUsage>;
pub type TextureViewTracker = Tracker<TextureViewId, DummyUsage>;

//TODO: make this a generic parameter.
/// Mode of stitching to states together.
#[derive(Clone, Copy, Debug)]
pub enum Stitch {
    /// Stitch to the init state of the other resource.
    Init,
    /// Stitch to the last sttate of the other resource.
    Last,
}

pub struct TrackerSet {
    pub buffers: BufferTracker,
    pub textures: TextureTracker,
    pub views: TextureViewTracker,
    //TODO: samplers
}

impl TrackerSet {
    pub fn new() -> Self {
        TrackerSet {
            buffers: BufferTracker::new(),
            textures: TextureTracker::new(),
            views: TextureViewTracker::new(),
        }
    }

    pub fn consume_by_extend(&mut self, other: &Self) {
        self.buffers.consume_by_extend(&other.buffers).unwrap();
        self.textures.consume_by_extend(&other.textures).unwrap();
        self.views.consume_by_extend(&other.views).unwrap();
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
    ) -> impl 'a + Iterator<Item = (I, Range<U>)> {
        other
            .map
            .iter()
            .flat_map(move |(&index, new)| match self.map.entry(index) {
                Entry::Vacant(e) => {
                    e.insert(new.clone());
                    None
                }
                Entry::Occupied(mut e) => {
                    assert_eq!(e.get().epoch, new.epoch);
                    let old = mem::replace(&mut e.get_mut().last, new.last);
                    if old == new.init {
                        None
                    } else {
                        let state = match stitch {
                            Stitch::Init => new.init,
                            Stitch::Last => new.last,
                        };
                        Some((I::new(index, new.epoch), old .. state))
                    }
                }
            })
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
