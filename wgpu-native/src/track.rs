use crate::{
    hub::Storage,
    device::MAX_MIP_LEVELS,
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

use arrayvec::ArrayVec;
use bitflags::bitflags;
use hal::backend::FastHashMap;

use std::{
    borrow::Borrow,
    collections::hash_map::{Entry, Iter},
    marker::PhantomData,
    mem,
    ops::{BitOr, Range},
    vec::Drain,
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

/// A single unit of state tracking.
#[derive(Clone, Copy, Debug)]
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

impl<U: Copy + BitOr<Output=U> + PartialEq + GenericUsage> Unit<U> {
    fn transit(&mut self, usage: U, permit: TrackPermit) -> Result<Tracktion<U>, U> {
        let old = self.last;
        if usage == old {
            Ok(Tracktion::Keep)
        } else if permit.contains(TrackPermit::EXTEND) && !(old | usage).is_exclusive() {
            self.last = old | usage;
            Ok(Tracktion::Extend { old })
        } else if permit.contains(TrackPermit::REPLACE) {
            self.last = usage;
            Ok(Tracktion::Replace { old })
        } else {
            Err(old)
        }
    }
}

//TODO: consider having `I` as an associated type of `S`?
#[derive(Debug)]
pub struct Tracker<I, S> {
    /// An association of known resource indices with their tracked states.
    map: FastHashMap<Index, Resource<S>>,
    _phantom: PhantomData<I>,
}

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
    src: Iter<'a, Index, Resource<Unit<U>>>,
    dst: &'a mut FastHashMap<Index, Resource<Unit<U>>>,
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
                    let old = mem::replace(&mut e.get_mut().state.last, new.state.last);
                    if old != new.state.init {
                        let states = old .. new.state.select(self.stitch);
                        return Some((I::new(index, new.epoch), states))
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

impl<I: TypedId, S> Tracker<I, S> {
    pub fn new() -> Self {
        Tracker {
            map: FastHashMap::default(),
            _phantom: PhantomData,
        }
    }

    /// Remove an id from the tracked map.
    pub fn remove(&mut self, id: I) -> bool {
        match self.map.remove(&id.index()) {
            Some(resource) => {
                assert_eq!(resource.epoch, id.epoch());
                true
            }
            None => false,
        }
    }

    /// Return an iterator over used resources keys.
    pub fn used<'a>(&'a self) -> impl 'a + Iterator<Item = I> {
        self.map
            .iter()
            .map(|(&index, resource)| I::new(index, resource.epoch))
    }

    fn clear(&mut self) {
        self.map.clear();
    }
}

impl<I: Copy + TypedId, U: Copy + GenericUsage + BitOr<Output = U> + PartialEq> Tracker<I, Unit<U>> {
    /// Get the last usage on a resource.
    pub(crate) fn query(&mut self, id: I, ref_count: &RefCount, default: U) -> Query<U> {
        match self.map.entry(id.index()) {
            Entry::Vacant(e) => {
                e.insert(Resource {
                    ref_count: ref_count.clone(),
                    state: Unit::new(default),
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
                    usage: e.get().state.last,
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
                e.insert(Resource {
                    ref_count: ref_count.clone(),
                    state: Unit::new(usage),
                    epoch: id.epoch(),
                });
                Ok(Tracktion::Init)
            }
            Entry::Occupied(mut e) => {
                assert_eq!(e.get().epoch, id.epoch());
                e.get_mut().state.transit(usage, permit)
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
                    let old = e.get().state.last;
                    if old != new.state.last {
                        let extended = old | new.state.last;
                        if extended.is_exclusive() {
                            let id = I::new(index, new.epoch);
                            return Err((id, old .. new.state.last));
                        }
                        e.get_mut().state.last = extended;
                    }
                }
            }
        }
        Ok(())
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

pub type BufferTracker = Tracker<BufferId, Unit<BufferUsage>>;
pub type TextureTracker = Tracker<TextureId, Unit<TextureUsage>>;
pub type TextureViewTracker = Tracker<TextureViewId, Unit<DummyUsage>>;
pub type BindGroupTracker = Tracker<BindGroupId, Unit<DummyUsage>>;

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

pub struct PendingTransition<S: ResourceState> {
    pub id: S::Id,
    pub selector: S::Selector,
    pub usage: Range<S::Usage>,
}

struct ResourceTracker<S: ResourceState> {
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
        &mut self, other: &Self
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

pub type BufferState = Unit<BufferUsage>;

impl Default for BufferState {
    fn default() -> Self {
        BufferState {
            init: BufferUsage::empty(),
            last: BufferUsage::empty(),
        }
    }
}

impl ResourceState for BufferState {
    type Id = BufferId;
    type Selector = ();
    type Usage = BufferUsage;

    fn query(
        &self,
        _selector: Self::Selector,
    ) -> Option<Self::Usage> {
        Some(self.last)
    }

    fn change(
        &mut self,
        id: Self::Id,
        _selector: Self::Selector,
        usage: Self::Usage,
        output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        let old = self.last;
        if usage != old {
            let pending = PendingTransition {
                id,
                selector: (),
                usage: old .. usage,
            };
            self.last = match output {
                Some(transitions) => {
                    transitions.push(pending);
                    usage
                }
                None =>  {
                    if !old.is_empty() && BufferUsage::WRITE_ALL.intersects(old | usage) {
                        return Err(pending);
                    }
                    old | usage
                }
            };
        }
        Ok(())
    }

    fn merge(
        &mut self,
        id: Self::Id,
        other: &Self,
        stitch: Stitch,
        output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        let usage = other.select(stitch);
        self.change(id, (), usage, output)
    }
}


#[derive(Clone, Debug)]
pub struct RangedStates<I, T> {
    ranges: Vec<(Range<I>, T)>,
}

impl<I, T> Default for RangedStates<I, T> {
    fn default() -> Self {
        RangedStates {
            ranges: Vec::new(),
        }
    }
}

impl<I: Copy + PartialOrd, T: Copy> RangedStates<I, T> {
    fn isolate(&mut self, index: &Range<I>, default: T) -> &mut [(Range<I>, T)] {
        let start_pos = match self.ranges
            .iter()
            .position(|pair| pair.0.end > index.start)
        {
            Some(pos) => pos,
            None => {
                let pos = self.ranges.len();
                self.ranges.push((index.clone(), default));
                return &mut self.ranges[pos ..];
            }
        };

        let mut pos = start_pos;
        let mut range_pos = index.start;
        loop {
            let (range, unit) = self.ranges[pos].clone();
            if range.start >= index.end {
                self.ranges.insert(pos, (range_pos .. index.end, default));
                pos += 1;
                break;
            }
            if range.start > range_pos {
                self.ranges.insert(pos, (range_pos .. range.start, default));
                pos += 1;
                range_pos = range.start;
            }
            if range.end >= index.end {
                self.ranges[pos].0.start = index.end;
                self.ranges.insert(pos, (range_pos .. index.end, unit));
                pos += 1;
                break;
            }
            pos += 1;
            range_pos = range.end;
            if pos == self.ranges.len() {
                self.ranges.push((range_pos .. index.end, default));
                pos += 1;
                break;
            }
        }

        &mut self.ranges[start_pos .. pos]
    }
}

type PlaneStates<T> = RangedStates<hal::image::Layer, T>;

#[derive(Clone)]
struct DepthStencilState {
    depth: Unit<TextureUsage>,
    stencil: Unit<TextureUsage>,
}

#[derive(Clone, Default)]
struct TextureStates {
    color_mips: ArrayVec<[PlaneStates<Unit<TextureUsage>>; MAX_MIP_LEVELS]>,
    depth_stencil: PlaneStates<DepthStencilState>,
}

impl ResourceState for TextureStates {
    type Id = TextureId;
    type Selector = hal::image::SubresourceRange;
    type Usage = TextureUsage;

    fn query(
        &self,
        selector: Self::Selector,
    ) -> Option<Self::Usage> {
        let mut usage = None;
        if selector.aspects.contains(hal::format::Aspects::COLOR) {
            let num_levels = self.color_mips.len();
            let layer_start = num_levels.min(selector.levels.start as usize);
            let layer_end = num_levels.min(selector.levels.end as usize);
            for layer in self.color_mips[layer_start .. layer_end].iter() {
                for &(ref range, ref unit) in layer.ranges.iter() {
                    if range.end > selector.layers.start && range.start < selector.layers.end {
                        let old = usage.replace(unit.last);
                        if old.is_some() && old != usage {
                            return None
                        }
                    }
                }
            }
        }
        if selector.aspects.intersects(hal::format::Aspects::DEPTH | hal::format::Aspects::STENCIL) {
            for &(ref range, ref ds) in self.depth_stencil.ranges.iter() {
                if range.end > selector.layers.start && range.start < selector.layers.end {
                    if selector.aspects.contains(hal::format::Aspects::DEPTH) {
                        let old = usage.replace(ds.depth.last);
                        if old.is_some() && old != usage {
                            return None
                        }
                    }
                    if selector.aspects.contains(hal::format::Aspects::STENCIL) {
                        let old = usage.replace(ds.stencil.last);
                        if old.is_some() && old != usage {
                            return None
                        }
                    }
                }
            }
        }
        usage
    }

    fn change(
        &mut self,
        id: Self::Id,
        selector: Self::Selector,
        usage: Self::Usage,
        mut output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        if selector.aspects.contains(hal::format::Aspects::COLOR) {
            while self.color_mips.len() < selector.levels.end as usize {
                self.color_mips.push(PlaneStates::default());
            }
            for level in selector.levels.clone() {
                let layers = self
                    .color_mips[level as usize]
                    .isolate(&selector.layers, Unit::new(usage));
                for &mut (ref range, ref mut unit) in layers {
                    let old = unit.last;
                    if old == usage {
                        continue
                    }
                    let pending = PendingTransition {
                        id,
                        selector: hal::image::SubresourceRange {
                            aspects: hal::format::Aspects::COLOR,
                            levels: level .. level + 1,
                            layers: range.clone(),
                        },
                        usage: old .. usage,
                    };
                    unit.last = match output.as_mut() {
                        Some(out) => {
                            out.push(pending);
                            usage
                        }
                        None => {
                            if !old.is_empty() && TextureUsage::WRITE_ALL.intersects(old | usage) {
                                return Err(pending);
                            }
                            old | usage
                        }
                    };
                }
            }
        }
        if selector.aspects.intersects(hal::format::Aspects::DEPTH | hal::format::Aspects::STENCIL) {
            unimplemented!() //TODO
        }
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


#[derive(Clone, Debug, Default)]
pub struct TextureViewState;

impl ResourceState for TextureViewState {
    type Id = TextureViewId;
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
