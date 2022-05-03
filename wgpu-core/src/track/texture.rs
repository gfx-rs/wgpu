use super::{range::RangedStates, PendingTransition};
use crate::{
    hub,
    id::{TextureId, TypedId, Valid},
    resource::Texture,
    track::{
        invalid_resource_state, iterate_bitvec, resize_bitvec, skip_barrier, ResourceUses,
        UsageConflict,
    },
};
use bit_vec::BitVec;
use hal::TextureUses;

use arrayvec::ArrayVec;
use naga::FastHashMap;

use std::{iter, ops::Range, vec::Drain};

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
        self.bits()
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
pub struct TextureBindGroupState {
    textures: Vec<(Valid<TextureId>, Option<TextureSelector>, TextureUses)>,
}

pub struct TextureStateSet {
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

pub struct TextureUsageScope {
    set: TextureStateSet,
    owned: BitVec<usize>,
}

impl TextureUsageScope {
    pub fn new() -> Self {
        Self {
            set: TextureStateSet::new(),
            owned: BitVec::default(),
        }
    }

    pub fn set_max_index(&mut self, size: usize) {
        self.set.simple.resize(size, TextureUses::UNINITIALIZED);

        resize_bitvec(&mut self.owned, size);
    }

    pub fn extend_from_scope<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        scope: &Self,
    ) -> Drain<PendingTransition<TextureUses>> {
        let incoming_size = scope.set.simple.len();
        if incoming_size > self.set.simple.len() {
            self.set_max_index(incoming_size);
        }

        iterate_bitvec(&scope.set.simple, |index| {
            todo!()
        });

        self.temp.drain(..)
    }

    pub unsafe fn extend_from_bind_group<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        bind_group: &TextureBindGroupState,
    ) -> Result<(), UsageConflict> {
        for (id, selector, state) in &bind_group.textures {
            self.extend(storage, *id, selector.clone(), *state)?;
        }

        Ok(())
    }

    /// # Safety
    ///
    /// `id` must be a valid ID and have an ID value less than the last call to set_max_index.
    pub unsafe fn extend<'a, A: hal::Api>(
        &mut self,
        storage: &'a hub::Storage<Texture<A>, TextureId>,
        id: TextureId,
        selector: Option<TextureSelector>,
        new_state: TextureUses,
    ) -> Result<&'a Texture<A>, UsageConflict> {
        let (index, _, _) = id.unzip();
        self.extend_inner(storage, index, selector, new_state)?;
        Ok(storage
            .get(id)
            .map_err(|_| UsageConflict::TextureInvalid { id: index })?)
    }

    /// # Safety
    ///
    /// `id` must be a valid ID and have an ID value less than the last call to set_max_index.
    unsafe fn extend_inner<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        index: u32,
        selector: Option<TextureSelector>,
        new_state: TextureUses,
    ) -> Result<(), UsageConflict> {
        let index = index as usize;

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
                            index,
                            selector,
                            current_state,
                            new_state,
                        ));
                    }

                    *self.set.simple.get_unchecked_mut(index) = merged_state;

                    return Ok(());
                }
                // The old usage is complex.
                (true, selector) => {
                    return self.extend_complex(storage, index, selector, new_state)
                }

                // The old usage is simple, so demote it to a complex one.
                (false, Some(selector)) => {
                    *self.set.simple.get_unchecked_mut(index) = hal::TextureUses::COMPLEX;

                    // Demote our simple state to a complex one.
                    self.extend_complex(storage, index, None, current_state)?;

                    // Extend that complex state with our new complex state.
                    return self.extend_complex(storage, index, Some(selector), new_state);
                }
            }
        }

        // We're the first to use this resource, let's add it.
        self.owned.set(index, true);

        if let Some(selector) = selector {
            *self.set.simple.get_unchecked_mut(index) = hal::TextureUses::COMPLEX;
            self.extend_complex(storage, index, Some(selector), new_state)?;
        } else {
            *self.set.simple.get_unchecked_mut(index) = new_state;
        }

        Ok(())
    }

    #[cold]
    unsafe fn extend_complex<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        index: u32,
        selector: Option<TextureSelector>,
        new_state: TextureUses,
    ) -> Result<(), UsageConflict> {
        let texture = storage.get_unchecked(index);

        // Create the complex entry for this texture.
        let complex = self.set.complex.entry(index).or_insert_with(|| {
            ComplexTextureState::new(
                texture.desc.mip_level_count,
                texture.desc.array_layer_count(),
            )
        });

        let mips;
        let layers;
        match selector {
            Some(selector) => {
                mips = selector.mips.clone();
                layers = selector.layers.clone();
            }
            None => {
                mips = texture.full_range.mips;
                layers = texture.full_range.layers;
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
                        index,
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

pub(crate) struct TextureTracker {
    start_set: TextureStateSet,
    end_set: TextureStateSet,
    /// Temporary storage for collecting transitions.
    temp: Vec<PendingTransition<TextureUses>>,
    owned: BitVec<usize>,
}
impl TextureTracker {
    pub fn new() -> Self {
        Self {
            start_set: TextureStateSet::new(),
            end_set: TextureStateSet::new(),
            temp: Vec::new(),
            owned: BitVec::default(),
        }
    }

    fn set_max_index(&mut self, size: usize) {
        self.start_set
            .simple
            .resize(size, TextureUses::UNINITIALIZED);
        self.end_set.simple.resize(size, TextureUses::UNINITIALIZED);

        resize_bitvec(&mut self.owned, size);
    }

    pub fn change_states<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        incoming_set: &TextureStateSet,
        incoming_ownership: &BitVec<usize>,
    ) -> Drain<PendingTransition<TextureUses>> {
        let incoming_size = incoming_set.simple.len();
        if incoming_size > self.start_set.simple.len() {
            self.set_max_index(incoming_size);
        }

        iterate_bitvec(incoming_ownership, |index| {
            unsafe { self.transition(storage, incoming_set, index) };
        });

        self.temp.drain(..)
    }

    pub unsafe fn change_states_bind_group<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        incoming_set: &TextureStateSet,
        incoming_ownership: &mut BitVec<usize>,
        bind_group_state: &TextureBindGroupState,
    ) -> Drain<PendingTransition<TextureUses>> {
        let incoming_size = incoming_set.simple.len();
        if incoming_size > self.start_set.simple.len() {
            self.set_max_index(incoming_size);
        }

        for &(index, _, _) in bind_group_state.textures.iter() {
            let index = index.0.unzip().0 as usize;
            if !incoming_ownership.get(index).unwrap_unchecked() {
                continue;
            }
            self.transition(storage, incoming_set, index);
            incoming_ownership.set(index, false);
        }

        self.temp.drain(..)
    }

    unsafe fn transition<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        incoming_set: &TextureStateSet,
        index: usize,
    ) {
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
            }
            (true, false, false) => {
                if skip_barrier(old_state, new_state) {
                    return;
                }

                self.temp.push(PendingTransition {
                    id: index as u32,
                    selector: storage.get_unchecked(index as u32).unwrap().full_range,
                    usage: old_state..new_state,
                });

                *self.end_set.simple.get_unchecked_mut(index) = new_state;
            }
            (true, true, true) => {
                self.transition_complex_to_complex(storage, incoming_set, index);
            }
            (true, true, false) => {
                self.transition_complex_to_simple(storage, incoming_set, index, new_state);
            }
            (true, false, true) => {
                self.transition_simple_to_complex(storage, incoming_set, index, old_state);
            }
        }
    }

    #[cold]
    unsafe fn transition_complex_to_complex<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        incoming_set: &TextureStateSet,
        index: usize,
    ) {
        let old_complex = self.end_set.complex.get(&(index as u32)).unwrap_unchecked();
        let new_complex = self.end_set.complex.get(&(index as u32)).unwrap_unchecked();

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

    #[cold]
    unsafe fn transition_complex_to_simple<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        incoming_set: &TextureStateSet,
        index: usize,
        new_state: TextureUses,
    ) {
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

    #[cold]
    unsafe fn transition_simple_to_complex<A: hal::Api>(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        incoming_set: &TextureStateSet,
        index: usize,
        old_state: TextureUses,
    ) {
        let new_complex = incoming_set.complex.get(&(index as u32)).unwrap_unchecked();

        for (mip_index, mips) in new_complex.mips.iter().enumerate() {
            let mip_index = mip_index as u32;
            for (layer, new_state) in mips.into_iter() {
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

        self.end_set
            .complex
            .insert(index as u32, new_complex.clone());
        *self.end_set.simple.get_unchecked_mut(index) = TextureUses::COMPLEX;
    }
}
