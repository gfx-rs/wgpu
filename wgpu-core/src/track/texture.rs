use super::{range::RangedStates, PendingTransition};
use crate::{
    hub,
    id::{TextureId, TypedId, Valid},
    resource::Texture,
    track::{
        invalid_resource_state, iterate_bitvec_indices, resize_bitvec, skip_barrier, ResourceUses,
        UsageConflict,
    },
    Epoch, RefCount,
};
use bit_vec::BitVec;
use hal::TextureUses;

use arrayvec::ArrayVec;
use naga::FastHashMap;

use std::{iter, marker::PhantomData, ops::Range, vec::Drain};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureSelector {
    pub mips: Range<u32>,
    pub layers: Range<u32>,
}

impl ResourceUses for TextureUses {
    const EXCLUSIVE: Self = Self::EXCLUSIVE;

    type Id = TextureId;
    type Selector = TextureSelector;

    fn bits(self) -> u16 {
        Self::bits(&self)
    }

    fn all_ordered(self) -> bool {
        self.contains(Self::ORDERED)
    }

    fn any_exclusive(self) -> bool {
        self.intersects(Self::EXCLUSIVE)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct ComplexTextureState {
    mips: ArrayVec<RangedStates<u32, TextureUses>, { hal::MAX_MIP_LEVELS as usize }>,
}

impl ComplexTextureState {
    fn new(mip_level_count: u32, array_layer_count: u32) -> Self {
        Self {
            mips: iter::repeat_with(|| {
                RangedStates::from_range(0..array_layer_count, TextureUses::UNINITIALIZED)
            })
            .take(mip_level_count as usize)
            .collect(),
        }
    }
}

// TODO: This representation could be optimized in a couple ways, but keep it simple for now.
pub(crate) struct TextureBindGroupState<A: hub::HalApi> {
    textures: Vec<(
        Valid<TextureId>,
        Option<TextureSelector>,
        RefCount,
        TextureUses,
    )>,

    _phantom: PhantomData<A>,
}
impl<A: hub::HalApi> TextureBindGroupState<A> {
    pub fn new() -> Self {
        Self {
            textures: Vec::new(),

            _phantom: PhantomData,
        }
    }

    pub fn used(&self) -> impl Iterator<Item = Valid<TextureId>> + '_ {
        self.textures.iter().map(|&(id, _, _, _)| id)
    }

    pub fn extend_with_refcount<'a>(
        &mut self,
        storage: &'a hub::Storage<Texture<A>, TextureId>,
        id: TextureId,
        ref_count: RefCount,
        selector: Option<TextureSelector>,
        state: TextureUses,
    ) -> Option<&'a Texture<A>> {
        let value = storage.get(id).ok()?;

        self.textures.push((Valid(id), selector, ref_count, state));

        Some(value)
    }
}

#[derive(Debug)]
pub(crate) struct TextureStateSet {
    simple: Vec<TextureUses>,
    complex: FastHashMap<u32, ComplexTextureState>,
}
impl TextureStateSet {
    pub fn new() -> Self {
        Self {
            simple: Vec::new(),
            complex: FastHashMap::default(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct TextureUsageScope<A: hub::HalApi> {
    set: TextureStateSet,

    ref_counts: Vec<Option<RefCount>>,
    epochs: Vec<Epoch>,

    owned: BitVec<usize>,

    _phantom: PhantomData<A>,
}

impl<A: hub::HalApi> TextureUsageScope<A> {
    pub fn new() -> Self {
        Self {
            set: TextureStateSet::new(),

            epochs: Vec::new(),
            ref_counts: Vec::new(),

            owned: BitVec::default(),

            _phantom: PhantomData,
        }
    }

    fn debug_assert_in_bounds(&self, index: usize) {
        debug_assert!(index < self.set.simple.len());
        debug_assert!(index < self.owned.len());
        debug_assert!(index < self.ref_counts.len());
        debug_assert!(index < self.epochs.len());

        debug_assert!(if self.owned.get(index).unwrap() {
            self.ref_counts[index].is_some()
        } else {
            true
        });

        debug_assert!(if self.owned.get(index).unwrap()
            && self.set.simple[index] == TextureUses::COMPLEX
        {
            self.set.complex.contains_key(&(index as u32))
        } else {
            true
        });
    }

    pub fn set_max_index(&mut self, size: usize) {
        self.set.simple.resize(size, TextureUses::UNINITIALIZED);
        self.ref_counts.resize(size, None);
        self.epochs.resize(size, u32::MAX);

        resize_bitvec(&mut self.owned, size);
    }

    pub fn used(&self) -> impl Iterator<Item = Valid<TextureId>> + '_ {
        self.debug_assert_in_bounds(self.owned.len() - 1);
        iterate_bitvec_indices(&self.owned).map(move |index| {
            let epoch = unsafe { *self.epochs.get_unchecked(index) };
            Valid(TextureId::zip(index as u32, epoch, A::VARIANT))
        })
    }

    pub(crate) fn is_empty(&self) -> bool {
        !self.owned.any()
    }

    pub fn extend_from_scope(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        scope: &Self,
    ) -> Result<(), UsageConflict> {
        let incoming_size = scope.set.simple.len();
        if incoming_size > self.set.simple.len() {
            self.set_max_index(incoming_size);
        }

        for index in iterate_bitvec_indices(&scope.owned) {
            let _ = (storage, index);
            todo!()
        }

        Ok(())
    }

    pub unsafe fn extend_from_bind_group(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        bind_group: &TextureBindGroupState<A>,
    ) -> Result<(), UsageConflict> {
        for (id, selector, ref_count, state) in &bind_group.textures {
            self.extend_refcount(storage, *id, selector.clone(), ref_count, *state)?;
        }

        Ok(())
    }

    /// # Safety
    ///
    /// `id` must be a valid ID and have an ID value less than the last call to set_max_index.
    pub unsafe fn extend<'a>(
        &mut self,
        storage: &'a hub::Storage<Texture<A>, TextureId>,
        id: TextureId,
        selector: Option<TextureSelector>,
        new_state: TextureUses,
    ) -> Result<&'a Texture<A>, UsageConflict> {
        let tex = storage
            .get(id)
            .map_err(|_| UsageConflict::TextureInvalid { id })?;

        self.extend_refcount(
            storage,
            Valid(id),
            selector,
            tex.life_guard.ref_count.as_ref().unwrap(),
            new_state,
        )?;

        Ok(tex)
    }

    /// # Safety
    ///
    /// `id` must be a valid ID and have an ID value less than the last call to set_max_index.
    pub unsafe fn extend_refcount(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        id: Valid<TextureId>,
        selector: Option<TextureSelector>,
        ref_count: &RefCount,
        new_state: TextureUses,
    ) -> Result<(), UsageConflict> {
        let (index32, epoch, _) = id.0.unzip();
        let index = index32 as usize;

        self.debug_assert_in_bounds(index);

        let currently_active = self.owned.get(index).unwrap_unchecked();
        if currently_active {
            let current_state = *self.set.simple.get_unchecked(index);
            match (current_state == hal::TextureUses::COMPLEX, selector) {
                // Both our usages are simple
                (false, None) => {
                    let merged_state = current_state | new_state;

                    if invalid_resource_state(merged_state) {
                        return Err(UsageConflict::from_texture(
                            storage,
                            id,
                            None,
                            current_state,
                            new_state,
                        ));
                    }

                    *self.set.simple.get_unchecked_mut(index) = merged_state;

                    return Ok(());
                }
                // The old usage is complex.
                (true, selector) => {
                    return self.extend_complex(storage, id, index32, selector, new_state)
                }

                // The old usage is simple, so demote it to a complex one.
                (false, Some(selector)) => {
                    *self.set.simple.get_unchecked_mut(index) = hal::TextureUses::COMPLEX;

                    // Demote our simple state to a complex one.
                    self.extend_complex(storage, id, index32, None, current_state)?;

                    // Extend that complex state with our new complex state.
                    return self.extend_complex(storage, id, index32, Some(selector), new_state);
                }
            }
        }

        // We're the first to use this resource, let's add it.
        *self.ref_counts.get_unchecked_mut(index) = Some(ref_count.clone());
        *self.epochs.get_unchecked_mut(index) = epoch;
        self.owned.set(index, true);

        if let Some(selector) = selector {
            *self.set.simple.get_unchecked_mut(index) = hal::TextureUses::COMPLEX;
            self.extend_complex(storage, id, index32, Some(selector), new_state)?;
        } else {
            *self.set.simple.get_unchecked_mut(index) = new_state;
        }

        Ok(())
    }

    #[cold]
    unsafe fn extend_complex(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        id: Valid<TextureId>,
        index32: u32,
        selector: Option<TextureSelector>,
        new_state: TextureUses,
    ) -> Result<(), UsageConflict> {
        let texture = storage.get_unchecked(index32);

        // Create the complex entry for this texture.
        let complex = self.set.complex.entry(index32).or_insert_with(|| {
            ComplexTextureState::new(
                texture.desc.mip_level_count,
                texture.desc.array_layer_count(),
            )
        });

        let mips;
        let layers;
        match selector {
            Some(ref selector) => {
                mips = selector.mips.clone();
                layers = selector.layers.clone();
            }
            None => {
                mips = texture.full_range.mips.clone();
                layers = texture.full_range.layers.clone();
            }
        }

        // Go through all relevant mip levels
        for mip in mips {
            let mip_state = &mut complex.mips[mip as usize];

            // Set our state.
            for (_, current_state) in mip_state.isolate(&layers, new_state) {
                let merged = *current_state | new_state;
                if invalid_resource_state(merged) {
                    return Err(UsageConflict::from_texture(
                        storage,
                        id,
                        selector,
                        *current_state,
                        new_state,
                    ));
                }
                *current_state = merged;
            }
        }

        Ok(())
    }
}

pub(crate) struct TextureTracker<A: hub::HalApi> {
    start_set: TextureStateSet,
    end_set: TextureStateSet,

    epochs: Vec<u32>,
    ref_counts: Vec<Option<RefCount>>,
    owned: BitVec<usize>,

    /// Temporary storage for collecting transitions.
    temp: Vec<PendingTransition<TextureUses>>,

    _phantom: PhantomData<A>,
}
impl<A: hub::HalApi> TextureTracker<A> {
    pub fn new() -> Self {
        Self {
            start_set: TextureStateSet::new(),
            end_set: TextureStateSet::new(),

            epochs: Vec::new(),
            ref_counts: Vec::new(),
            owned: BitVec::default(),

            temp: Vec::new(),

            _phantom: PhantomData,
        }
    }

    fn debug_assert_in_bounds(&self, index: usize) {
        debug_assert!(index < self.start_set.simple.len());
        debug_assert!(index < self.end_set.simple.len());
        debug_assert!(index < self.owned.len());
        debug_assert!(index < self.ref_counts.len());
        debug_assert!(index < self.epochs.len());

        debug_assert!(if self.owned.get(index).unwrap() {
            self.ref_counts[index].is_some()
        } else {
            true
        });

        debug_assert!(if self.owned.get(index).unwrap()
            && self.start_set.simple[index] == TextureUses::COMPLEX
        {
            self.start_set.complex.contains_key(&(index as u32))
        } else {
            true
        });
        debug_assert!(if self.owned.get(index).unwrap()
            && self.end_set.simple[index] == TextureUses::COMPLEX
        {
            self.end_set.complex.contains_key(&(index as u32))
        } else {
            true
        });
    }

    fn set_max_index(&mut self, size: usize) {
        self.start_set
            .simple
            .resize(size, TextureUses::UNINITIALIZED);
        self.end_set.simple.resize(size, TextureUses::UNINITIALIZED);
        self.ref_counts.resize(size, None);
        self.epochs.resize(size, u32::MAX);

        resize_bitvec(&mut self.owned, size);
    }

    pub fn used(&self) -> impl Iterator<Item = Valid<TextureId>> + '_ {
        self.debug_assert_in_bounds(self.owned.len() - 1);
        iterate_bitvec_indices(&self.owned).map(move |index| {
            let epoch = unsafe { *self.epochs.get_unchecked(index) };
            Valid(TextureId::zip(index as u32, epoch, A::VARIANT))
        })
    }

    pub fn drain(&mut self) -> Drain<PendingTransition<TextureUses>> {
        self.temp.drain(..)
    }

    pub fn get_ref_count(&self, id: Valid<TextureId>) -> &RefCount {
        let (index32, _, _) = id.0.unzip();
        let index = index32 as usize;

        self.ref_counts[index].as_ref().unwrap()
    }

    pub unsafe fn init(&mut self, id: TextureId, ref_count: RefCount, usage: TextureUses) {
        let (index32, epoch, _) = id.unzip();
        let index = index32 as usize;

        self.debug_assert_in_bounds(index);

        *self.start_set.simple.get_unchecked_mut(index) = usage;
        *self.end_set.simple.get_unchecked_mut(index) = usage;

        *self.epochs.get_unchecked_mut(index) = epoch;
        *self.ref_counts.get_unchecked_mut(index) = Some(ref_count);
        self.owned.set(index, true);
    }

    pub unsafe fn change_state<'a>(
        &mut self,
        storage: &'a hub::Storage<Texture<A>, TextureId>,
        id: TextureId,
        selector: TextureSelector,
        new_usage: TextureUses,
    ) -> Option<(&'a Texture<A>, Option<PendingTransition<TextureUses>>)> {
        let _ = (storage, id, selector, new_usage);
        todo!()
    }

    pub fn change_states_tracker(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        tracker: &Self,
    ) {
        let incoming_size = tracker.start_set.simple.len();
        if incoming_size > self.start_set.simple.len() {
            self.set_max_index(incoming_size);
        }

        for index in iterate_bitvec_indices(&tracker.owned) {
            tracker.debug_assert_in_bounds(index);
            unsafe {
                self.transition(
                    storage,
                    &tracker.start_set,
                    &tracker.ref_counts,
                    &tracker.epochs,
                    index,
                )
            };
        }
    }

    pub fn change_states_scope(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        scope: &TextureUsageScope<A>,
    ) {
        let incoming_size = scope.set.simple.len();
        if incoming_size > self.start_set.simple.len() {
            self.set_max_index(incoming_size);
        }

        for index in iterate_bitvec_indices(&scope.owned) {
            scope.debug_assert_in_bounds(index);
            unsafe {
                self.transition(storage, &scope.set, &scope.ref_counts, &scope.epochs, index)
            };
        }
    }

    pub unsafe fn change_states_bind_group(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        scope: &mut TextureUsageScope<A>,
        bind_group_state: &TextureBindGroupState<A>,
    ) {
        let incoming_size = scope.set.simple.len();
        if incoming_size > self.start_set.simple.len() {
            self.set_max_index(incoming_size);
        }

        for &(id, _, _, _) in bind_group_state.textures.iter() {
            let (index32, _, _) = id.0.unzip();
            let index = index32 as usize;
            scope.debug_assert_in_bounds(index);

            if !scope.owned.get(index).unwrap_unchecked() {
                continue;
            }
            self.transition(storage, &scope.set, &scope.ref_counts, &scope.epochs, index);

            *scope.ref_counts.get_unchecked_mut(index) = None;
            *scope.epochs.get_unchecked_mut(index) = u32::MAX;
            scope.owned.set(index, false);
        }
    }

    unsafe fn transition(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        incoming_set: &TextureStateSet,
        incoming_ref_counts: &[Option<RefCount>],
        incoming_epochs: &[Epoch],
        index: usize,
    ) {
        // Note: both callees of this function call scope.debug_assert_in_bounds.
        self.debug_assert_in_bounds(index);

        let old_tracked = self.owned.get(index).unwrap_unchecked();
        let old_state = *self.end_set.simple.get_unchecked(index);
        let new_state = *incoming_set.simple.get_unchecked(index);

        match (
            old_tracked,
            old_state == TextureUses::COMPLEX,
            new_state == TextureUses::COMPLEX,
        ) {
            (false, _, false) => {
                *self.start_set.simple.get_unchecked_mut(index) = new_state;
                *self.end_set.simple.get_unchecked_mut(index) = new_state;

                self.owned.set(index, true);

                let ref_count = incoming_ref_counts
                    .get_unchecked(index)
                    .clone()
                    .unwrap_unchecked();
                *self.ref_counts.get_unchecked_mut(index) = Some(ref_count);

                let epoch = *incoming_epochs.get_unchecked(index);
                *self.epochs.get_unchecked_mut(index) = epoch;
            }
            (false, _, true) => {
                *self.start_set.simple.get_unchecked_mut(index) = TextureUses::COMPLEX;
                *self.end_set.simple.get_unchecked_mut(index) = TextureUses::COMPLEX;

                let complex_state = incoming_set.complex.get(&(index as u32)).unwrap_unchecked();
                self.start_set
                    .complex
                    .insert(index as u32, complex_state.clone());
                self.end_set
                    .complex
                    .insert(index as u32, complex_state.clone());

                self.owned.set(index, true);

                let ref_count = incoming_ref_counts
                    .get_unchecked(index)
                    .clone()
                    .unwrap_unchecked();
                *self.ref_counts.get_unchecked_mut(index) = Some(ref_count);

                let epoch = *incoming_epochs.get_unchecked(index);
                *self.epochs.get_unchecked_mut(index) = epoch;
            }
            (true, false, false) => {
                if skip_barrier(old_state, new_state) {
                    return;
                }

                self.temp.push(PendingTransition {
                    id: index as u32,
                    selector: storage.get_unchecked(index as u32).full_range.clone(),
                    usage: old_state..new_state,
                });

                *self.end_set.simple.get_unchecked_mut(index) = new_state;
            }
            (true, true, true) => {
                self.transition_complex_to_complex(incoming_set, index);
            }
            (true, true, false) => {
                self.transition_complex_to_simple(index, new_state);
            }
            (true, false, true) => {
                self.transition_simple_to_complex(incoming_set, index, old_state);
            }
        }
    }

    unsafe fn transition_complex_to_complex(
        &mut self,
        incoming_set: &TextureStateSet,
        index: usize,
    ) {
        let old_complex = self
            .end_set
            .complex
            .get_mut(&(index as u32))
            .unwrap_unchecked();
        let new_complex = incoming_set.complex.get(&(index as u32)).unwrap_unchecked();

        let mut temp = Vec::new();
        debug_assert!(old_complex.mips.len() >= new_complex.mips.len());

        for (mip_id, (mip_self, mip_other)) in old_complex
            .mips
            .iter_mut()
            .zip(&new_complex.mips)
            .enumerate()
        {
            let level = mip_id as u32;
            temp.extend(mip_self.merge(mip_other, 0));

            for (layers, states) in temp.drain(..) {
                match states {
                    Range {
                        start: Some(start),
                        end: Some(end),
                    } => {
                        if skip_barrier(start, end) {
                            return;
                        }
                        // TODO: Can't satisfy clippy here unless we modify
                        // `TextureSelector` to use `std::ops::RangeBounds`.
                        #[allow(clippy::range_plus_one)]
                        let pending = PendingTransition {
                            id: index as u32,
                            selector: TextureSelector {
                                mips: level..level + 1,
                                layers: layers.clone(),
                            },
                            usage: start..end,
                        };

                        self.temp.push(pending);

                        for (_, state) in mip_self.isolate(&layers, end) {
                            *state = end;
                        }
                    }
                    _ => unreachable!(),
                };
            }
        }
    }

    unsafe fn transition_complex_to_simple(&mut self, index: usize, new_state: TextureUses) {
        let old_complex = self
            .end_set
            .complex
            .remove(&(index as u32))
            .unwrap_unchecked();

        for (mip_index, mips) in old_complex.mips.into_iter().enumerate() {
            let mip_index = mip_index as u32;
            for (layer, old_state) in mips.into_iter() {
                if skip_barrier(old_state, new_state) {
                    continue;
                }

                #[allow(clippy::range_plus_one)]
                self.temp.push(PendingTransition {
                    id: index as u32,
                    selector: TextureSelector {
                        mips: mip_index..mip_index + 1,
                        layers: layer,
                    },
                    usage: old_state..new_state,
                })
            }
        }

        *self.end_set.simple.get_unchecked_mut(index) = new_state;
    }

    unsafe fn transition_simple_to_complex(
        &mut self,
        incoming_set: &TextureStateSet,
        index: usize,
        old_state: TextureUses,
    ) {
        let new_complex = incoming_set.complex.get(&(index as u32)).unwrap_unchecked();

        for (mip_index, mips) in new_complex.mips.iter().enumerate() {
            let mip_index = mip_index as u32;
            for &(ref layer, new_state) in mips.iter() {
                if skip_barrier(old_state, new_state) {
                    continue;
                }

                #[allow(clippy::range_plus_one)]
                self.temp.push(PendingTransition {
                    id: index as u32,
                    selector: TextureSelector {
                        mips: mip_index..mip_index + 1,
                        layers: layer.clone(),
                    },
                    usage: old_state..new_state,
                })
            }
        }

        self.end_set
            .complex
            .insert(index as u32, new_complex.clone());
        *self.end_set.simple.get_unchecked_mut(index) = TextureUses::COMPLEX;
    }

    pub fn remove(&mut self, id: Valid<TextureId>) -> bool {
        self.remove_inner(id, true)
    }

    pub fn remove_abandoned(&mut self, id: Valid<TextureId>) -> bool {
        self.remove_inner(id, false)
    }

    fn remove_inner(&mut self, id: Valid<TextureId>, force: bool) -> bool {
        let (index32, epoch, _) = id.0.unzip();
        let index = index32 as usize;

        if index > self.owned.len() {
            return false;
        }

        self.debug_assert_in_bounds(index);

        unsafe {
            if self.owned.get(index).unwrap_unchecked() {
                let existing_epoch = self.epochs.get_unchecked_mut(index);
                let existing_ref_count = self.ref_counts.get_unchecked_mut(index);

                if *existing_epoch == epoch
                    && existing_ref_count.as_mut().unwrap_unchecked().load() == 1
                {
                    self.owned.set(index, false);
                    *existing_epoch = u32::MAX;
                    *existing_ref_count = None;

                    return true;
                } else if force {
                    assert_eq!(*existing_epoch, epoch);
                }
            }
        }

        false
    }
}
