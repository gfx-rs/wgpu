use crate::{
    conv,
    device::MAX_MIP_LEVELS,
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

use arrayvec::ArrayVec;
use hal::backend::FastHashMap;

use std::{
    borrow::Borrow,
    cmp::Ordering,
    collections::hash_map::Entry,
    iter::Peekable,
    ops::Range,
    slice,
    vec::Drain,
};


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

//TODO: store `hal::buffer::State` here to avoid extra conversions
pub type BufferState = Unit<BufferUsage>;

impl PendingTransition<BufferState> {
    pub fn to_states(&self) -> Range<hal::buffer::State> {
        conv::map_buffer_state(self.usage.start) ..
        conv::map_buffer_state(self.usage.end)
    }
}

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

impl<I: Copy + PartialOrd, T: Copy + PartialEq> RangedStates<I, T> {
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

    fn _coalesce(&mut self) {
        let mut num_removed = 0;
        let mut iter = self.ranges.iter_mut();
        let mut cur = match iter.next() {
            Some(elem) => elem,
            None => return,
        };
        while let Some(next) = iter.next() {
            if cur.0.end == next.0.start && cur.1 == next.1 {
                num_removed += 1;
                cur.0.end = next.0.end;
                next.0.end = next.0.start;
            } else {
                cur = next;
            }
        }
        if num_removed != 0 {
            self.ranges.retain(|pair| pair.0.start != pair.0.end);
        }
    }
}

struct Merge<'a, I, T> {
    base: I,
    sa: Peekable<slice::Iter<'a, (Range<I>, T)>>,
    sb: Peekable<slice::Iter<'a, (Range<I>, T)>>,
}

impl<'a, I: Copy + Ord, T: Copy> Iterator for Merge<'a, I, T> {
    type Item = (Range<I>, Range<T>);
    fn next(&mut self) -> Option<Self::Item> {
        match (self.sa.peek(), self.sb.peek()) {
            // we have both streams
            (Some(&(ref ra, va)), Some(&(ref rb, vb))) => {
                let (range, usage) = if ra.start < self.base { // in the middle of the left stream
                    if self.base == rb.start { // right stream is starting
                        debug_assert!(self.base < ra.end);
                        (self.base .. ra.end.min(rb.end), *va .. *vb)
                    } else { // right hasn't started yet
                        debug_assert!(self.base < rb.start);
                        (self.base .. rb.start, *va .. *va)
                    }
                } else if rb.start < self.base { // in the middle of the right stream
                    if self.base == ra.start { // left stream is starting
                        debug_assert!(self.base < rb.end);
                        (self.base .. ra.end.min(rb.end), *va .. *vb)
                    } else { // left hasn't started yet
                        debug_assert!(self.base < ra.start);
                        (self.base .. ra.start, *vb .. *vb)
                    }
                } else { // no active streams
                    match ra.start.cmp(&rb.start) {
                        // both are starting
                        Ordering::Equal => (ra.start .. ra.end.min(rb.end), *va .. *vb),
                        // only left is starting
                        Ordering::Less => (ra.start .. rb.start, *va .. *va),
                        // only right is starting
                        Ordering::Greater => (rb.start .. ra.start, *vb .. *vb),
                    }
                };
                self.base = range.end;
                if ra.end == range.end {
                    let _ = self.sa.next();
                }
                if rb.end == range.end {
                    let _ = self.sb.next();
                }
                Some((range, usage))
            }
            // only right stream
            (None, Some(&(ref rb, vb))) => {
                let range = self.base.max(rb.start) .. rb.end;
                self.base = rb.end;
                let _ = self.sb.next();
                Some((range, *vb .. *vb))
            }
            // only left stream
            (Some(&(ref ra, va)), None) => {
                let range = self.base.max(ra.start) .. ra.end;
                self.base = ra.end;
                let _ = self.sa.next();
                Some((range, *va .. *va))
            }
            // done
            (None, None) => None,
        }
    }
}

type PlaneStates<T> = RangedStates<hal::image::Layer, T>;

//TODO: store `hal::image::State` here to avoid extra conversions
#[derive(Clone, Copy, Debug, PartialEq)]
struct DepthStencilState {
    depth: Unit<TextureUsage>,
    stencil: Unit<TextureUsage>,
}

#[derive(Clone, Debug, Default)]
pub struct TextureStates {
    color_mips: ArrayVec<[PlaneStates<Unit<TextureUsage>>; MAX_MIP_LEVELS]>,
    depth_stencil: PlaneStates<DepthStencilState>,
}

impl PendingTransition<TextureStates> {
    pub fn to_states(&self) -> Range<hal::image::State> {
        conv::map_texture_state(self.usage.start, self.selector.aspects) ..
        conv::map_texture_state(self.usage.end, self.selector.aspects)
    }
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
            for level in selector.levels.clone() {
                let ds_state = DepthStencilState {
                    depth: Unit::new(usage),
                    stencil: Unit::new(usage),
                };
                for &mut (ref range, ref mut unit) in self.depth_stencil
                    .isolate(&selector.layers, ds_state)
                {
                    //TODO: check if anything needs to be done when only one of the depth/stencil
                    // is selected?
                    if unit.depth.last != usage && selector.aspects.contains(hal::format::Aspects::DEPTH) {
                        let old = unit.depth.last;
                        let pending = PendingTransition {
                            id,
                            selector: hal::image::SubresourceRange {
                                aspects: hal::format::Aspects::DEPTH,
                                levels: level .. level + 1,
                                layers: range.clone(),
                            },
                            usage: old .. usage,
                        };
                        unit.depth.last = match output.as_mut() {
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
                    if unit.stencil.last != usage && selector.aspects.contains(hal::format::Aspects::STENCIL) {
                        let old = unit.stencil.last;
                        let pending = PendingTransition {
                            id,
                            selector: hal::image::SubresourceRange {
                                aspects: hal::format::Aspects::STENCIL,
                                levels: level .. level + 1,
                                layers: range.clone(),
                            },
                            usage: old .. usage,
                        };
                        unit.stencil.last = match output.as_mut() {
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
        }
        Ok(())
    }

    fn merge(
        &mut self,
        id: Self::Id,
        other: &Self,
        stitch: Stitch,
        mut output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        let mut temp_color = Vec::new();
        while self.color_mips.len() < other.color_mips.len() {
            self.color_mips.push(PlaneStates::default());
        }
        for (mip_id, (mip_self, mip_other)) in self.color_mips
            .iter_mut()
            .zip(&other.color_mips)
            .enumerate()
        {
            temp_color.extend(Merge {
                base: 0,
                sa: mip_self.ranges.iter().peekable(),
                sb: mip_other.ranges.iter().peekable(),
            });
            mip_self.ranges.clear();
            for (layers, states) in temp_color.drain(..) {
                let color_usage = states.start.last .. states.end.select(stitch);
                if let Some(out) = output.as_mut() {
                    if color_usage.start != color_usage.end {
                        let level = mip_id as hal::image::Level;
                        out.push(PendingTransition {
                            id,
                            selector: hal::image::SubresourceRange {
                                aspects: hal::format::Aspects::COLOR,
                                levels: level .. level + 1,
                                layers: layers.clone(),
                            },
                            usage: color_usage.clone(),
                        });
                    }
                }
                mip_self.ranges.push((layers, Unit {
                    init: states.start.init,
                    last: color_usage.end,
                }));
            }
        }

        let mut temp_ds = Vec::new();
        temp_ds.extend(Merge {
            base: 0,
            sa: self.depth_stencil.ranges.iter().peekable(),
            sb: other.depth_stencil.ranges.iter().peekable(),
        });
        self.depth_stencil.ranges.clear();
        for (layers, states) in temp_ds.drain(..) {
            let usage_depth = states.start.depth.last .. states.end.depth.select(stitch);
            let usage_stencil = states.start.stencil.last .. states.end.stencil.select(stitch);
            if let Some(out) = output.as_mut() {
                if usage_depth.start != usage_depth.end {
                    out.push(PendingTransition {
                        id,
                        selector: hal::image::SubresourceRange {
                            aspects: hal::format::Aspects::DEPTH,
                            levels: 0 .. 1,
                            layers: layers.clone(),
                        },
                        usage: usage_depth.clone(),
                    });
                }
                if usage_stencil.start != usage_stencil.end {
                    out.push(PendingTransition {
                        id,
                        selector: hal::image::SubresourceRange {
                            aspects: hal::format::Aspects::STENCIL,
                            levels: 0 .. 1,
                            layers: layers.clone(),
                        },
                        usage: usage_stencil.clone(),
                    });
                }
            }
            self.depth_stencil.ranges.push((layers, DepthStencilState {
                depth: Unit {
                    init: states.start.depth.init,
                    last: usage_depth.end,
                },
                stencil: Unit {
                    init: states.start.stencil.init,
                    last: usage_stencil.end,
                },
            }));
        }

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

#[derive(Clone, Debug, Default)]
pub struct BindGroupState;

impl ResourceState for BindGroupState {
    type Id = BindGroupId;
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
    pub views: ResourceTracker<TextureViewState>,
    pub bind_groups: ResourceTracker<BindGroupState>,
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
