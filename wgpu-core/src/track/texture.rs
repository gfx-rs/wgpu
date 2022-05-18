use super::{range::RangedStates, PendingTransition};
use crate::{
    hub,
    id::{TextureId, TypedId, Valid},
    resource::Texture,
    track::{
        invalid_resource_state, iterate_bitvec_indices, skip_barrier, ResourceMetadata,
        ResourceMetadataProvider, ResourceUses, UsageConflict,
    },
    LifeGuard, RefCount,
};
use hal::TextureUses;

use arrayvec::ArrayVec;
use naga::FastHashMap;

use std::{borrow::Cow, iter, marker::PhantomData, ops::Range, vec::Drain};

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
        Self::ORDERED.contains(self)
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
                RangedStates::from_range(0..array_layer_count, TextureUses::UNKNOWN)
            })
            .take(mip_level_count as usize)
            .collect(),
        }
    }

    fn from_selector_state_iter(
        full_range: TextureSelector,
        state_iter: impl Iterator<Item = (TextureSelector, TextureUses)>,
    ) -> Self {
        let mut complex =
            ComplexTextureState::new(full_range.mips.len() as u32, full_range.layers.len() as u32);
        for (selector, desired_state) in state_iter {
            // This should only ever happen with a wgpu bug, but let's just double
            // check that resource states don't have any conflicts.
            debug_assert_eq!(invalid_resource_state(desired_state), false);

            let mips = selector.mips.start as usize..selector.mips.end as usize;
            for mip in &mut complex.mips[mips] {
                for &mut (_, ref mut state) in mip.isolate(&selector.layers, TextureUses::UNKNOWN) {
                    *state = desired_state;
                }
            }
        }
        complex
    }

    fn to_selector_state_iter(
        &self,
    ) -> impl Iterator<Item = (TextureSelector, TextureUses)> + Clone + '_ {
        self.mips.iter().enumerate().flat_map(|(mip, inner)| {
            let mip = mip as u32;
            {
                inner.iter().map(move |&(ref layers, inner)| {
                    (
                        TextureSelector {
                            mips: mip..mip + 1,
                            layers: layers.clone(),
                        },
                        inner,
                    )
                })
            }
        })
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

    pub(crate) fn optimize(&mut self) {
        self.textures
            .sort_unstable_by_key(|&(id, _, _, _)| id.0.unzip().0);
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
    fn new() -> Self {
        Self {
            simple: Vec::new(),
            complex: FastHashMap::default(),
        }
    }

    fn set_size(&mut self, size: usize) {
        self.simple.resize(size, TextureUses::UNINITIALIZED);
    }
}

#[derive(Debug)]
pub(crate) struct TextureUsageScope<A: hub::HalApi> {
    set: TextureStateSet,

    metadata: ResourceMetadata<A>,
}

impl<A: hub::HalApi> TextureUsageScope<A> {
    pub fn new() -> Self {
        Self {
            set: TextureStateSet::new(),

            metadata: ResourceMetadata::new(),
        }
    }

    fn debug_assert_in_bounds(&self, index: usize) {
        self.metadata.debug_assert_in_bounds(index);

        debug_assert!(index < self.set.simple.len());

        debug_assert!(if self.metadata.owned.get(index).unwrap()
            && self.set.simple[index] == TextureUses::COMPLEX
        {
            self.set.complex.contains_key(&(index as u32))
        } else {
            true
        });
    }

    pub fn set_size(&mut self, size: usize) {
        self.set.set_size(size);
        self.metadata.set_size(size);
    }

    pub fn used(&self) -> impl Iterator<Item = Valid<TextureId>> + '_ {
        self.metadata.used()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    pub fn extend_from_scope(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        scope: &Self,
    ) -> Result<(), UsageConflict> {
        let incoming_size = scope.set.simple.len();
        if incoming_size > self.set.simple.len() {
            self.set_size(incoming_size);
        }

        for index in iterate_bitvec_indices(&scope.metadata.owned) {
            let index32 = index as u32;

            unsafe {
                insert_or_merge(
                    texture_data_from_texture(storage, index32),
                    &mut self.set,
                    &mut self.metadata,
                    index32,
                    index,
                    LayeredStateProvider::TextureSet { set: &scope.set },
                    ResourceMetadataProvider::Indirect {
                        metadata: &scope.metadata,
                    },
                )?
            };
        }

        Ok(())
    }

    pub unsafe fn extend_from_bind_group(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        bind_group: &TextureBindGroupState<A>,
    ) -> Result<(), UsageConflict> {
        for &(id, ref selector, ref ref_count, state) in &bind_group.textures {
            self.extend_refcount(storage, id, selector.clone(), ref_count, state)?;
        }

        Ok(())
    }

    /// # Safety
    ///
    /// `id` must be a valid ID and have an ID value less than the last call to set_size.
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

        insert_or_merge(
            texture_data_from_texture(storage, index32),
            &mut self.set,
            &mut self.metadata,
            index32,
            index,
            LayeredStateProvider::from_option(selector, new_state),
            ResourceMetadataProvider::Direct {
                epoch,
                ref_count: Cow::Borrowed(ref_count),
            },
        )?;

        Ok(())
    }
}

pub(crate) struct TextureTracker<A: hub::HalApi> {
    start_set: TextureStateSet,
    end_set: TextureStateSet,

    metadata: ResourceMetadata<A>,

    /// Temporary storage for collecting transitions.
    temp: Vec<PendingTransition<TextureUses>>,

    _phantom: PhantomData<A>,
}
impl<A: hub::HalApi> TextureTracker<A> {
    pub fn new() -> Self {
        Self {
            start_set: TextureStateSet::new(),
            end_set: TextureStateSet::new(),

            metadata: ResourceMetadata::new(),

            temp: Vec::new(),

            _phantom: PhantomData,
        }
    }

    fn debug_assert_in_bounds(&self, index: usize) {
        self.metadata.debug_assert_in_bounds(index);

        debug_assert!(index < self.start_set.simple.len());
        debug_assert!(index < self.end_set.simple.len());

        debug_assert!(if self.metadata.owned.get(index).unwrap()
            && self.start_set.simple[index] == TextureUses::COMPLEX
        {
            self.start_set.complex.contains_key(&(index as u32))
        } else {
            true
        });
        debug_assert!(if self.metadata.owned.get(index).unwrap()
            && self.end_set.simple[index] == TextureUses::COMPLEX
        {
            self.end_set.complex.contains_key(&(index as u32))
        } else {
            true
        });
    }

    pub fn set_size(&mut self, size: usize) {
        self.start_set.set_size(size);
        self.end_set.set_size(size);

        self.metadata.set_size(size);
    }

    fn allow_index(&mut self, index: usize) {
        if index >= self.start_set.simple.len() {
            self.set_size(index + 1);
        }
    }

    pub fn used(&self) -> impl Iterator<Item = Valid<TextureId>> + '_ {
        self.metadata.used()
    }

    pub fn drain(&mut self) -> Drain<PendingTransition<TextureUses>> {
        self.temp.drain(..)
    }

    pub fn get_ref_count(&self, id: Valid<TextureId>) -> &RefCount {
        let (index32, _, _) = id.0.unzip();
        let index = index32 as usize;

        self.metadata.ref_counts[index].as_ref().unwrap()
    }

    pub fn init(&mut self, id: TextureId, ref_count: RefCount, usage: TextureUses) {
        let (index32, epoch, _) = id.unzip();
        let index = index32 as usize;

        self.allow_index(index);

        self.debug_assert_in_bounds(index);

        unsafe {
            let currently_owned = self.metadata.owned.get(index).unwrap_unchecked();

            if currently_owned {
                panic!("Tried to init resource already tracked");
            }

            insert(
                None,
                Some(&mut self.start_set),
                &mut self.end_set,
                &mut self.metadata,
                index32,
                index,
                LayeredStateProvider::KnownSingle { state: usage },
                None,
                ResourceMetadataProvider::Direct {
                    epoch,
                    ref_count: Cow::Owned(ref_count),
                },
            )
        };
    }

    pub fn change_state<'a>(
        &mut self,
        storage: &'a hub::Storage<Texture<A>, TextureId>,
        id: TextureId,
        selector: TextureSelector,
        new_state: TextureUses,
    ) -> Option<(&'a Texture<A>, Drain<PendingTransition<TextureUses>>)> {
        let texture = storage.get(id).ok()?;

        let (index32, epoch, _) = id.unzip();
        let index = index32 as usize;

        self.allow_index(index);

        self.debug_assert_in_bounds(index);

        unsafe {
            insert_or_barrier_update(
                texture_data_from_texture(storage, index32),
                Some(&mut self.start_set),
                &mut self.end_set,
                &mut self.metadata,
                index32,
                index,
                LayeredStateProvider::Selector {
                    selector,
                    state: new_state,
                },
                None,
                ResourceMetadataProvider::Resource { epoch },
                &mut self.temp,
            )
        }

        Some((texture, self.temp.drain(..)))
    }

    pub fn change_states_tracker(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        tracker: &Self,
    ) {
        let incoming_size = tracker.start_set.simple.len();
        if incoming_size > self.start_set.simple.len() {
            self.set_size(incoming_size);
        }

        for index in iterate_bitvec_indices(&tracker.metadata.owned) {
            let index32 = index as u32;

            self.debug_assert_in_bounds(index);
            tracker.debug_assert_in_bounds(index);
            unsafe {
                insert_or_barrier_update(
                    texture_data_from_texture(storage, index32),
                    Some(&mut self.start_set),
                    &mut self.end_set,
                    &mut self.metadata,
                    index32,
                    index,
                    LayeredStateProvider::TextureSet {
                        set: &tracker.start_set,
                    },
                    Some(LayeredStateProvider::TextureSet {
                        set: &tracker.end_set,
                    }),
                    ResourceMetadataProvider::Indirect {
                        metadata: &tracker.metadata,
                    },
                    &mut self.temp,
                );
            }
        }
    }

    pub fn change_states_scope(
        &mut self,
        storage: &hub::Storage<Texture<A>, TextureId>,
        scope: &TextureUsageScope<A>,
    ) {
        let incoming_size = scope.set.simple.len();
        if incoming_size > self.start_set.simple.len() {
            self.set_size(incoming_size);
        }

        for index in iterate_bitvec_indices(&scope.metadata.owned) {
            let index32 = index as u32;

            self.debug_assert_in_bounds(index);
            scope.debug_assert_in_bounds(index);
            unsafe {
                insert_or_barrier_update(
                    texture_data_from_texture(storage, index32),
                    Some(&mut self.start_set),
                    &mut self.end_set,
                    &mut self.metadata,
                    index32,
                    index,
                    LayeredStateProvider::TextureSet { set: &scope.set },
                    None,
                    ResourceMetadataProvider::Indirect {
                        metadata: &scope.metadata,
                    },
                    &mut self.temp,
                );
            }
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
            self.set_size(incoming_size);
        }

        for &(id, _, _, _) in bind_group_state.textures.iter() {
            let (index32, _, _) = id.0.unzip();
            let index = index32 as usize;
            scope.debug_assert_in_bounds(index);

            if !scope.metadata.owned.get(index).unwrap_unchecked() {
                continue;
            }
            insert_or_barrier_update(
                texture_data_from_texture(storage, index32),
                Some(&mut self.start_set),
                &mut self.end_set,
                &mut self.metadata,
                index32,
                index,
                LayeredStateProvider::TextureSet { set: &scope.set },
                None,
                ResourceMetadataProvider::Indirect {
                    metadata: &scope.metadata,
                },
                &mut self.temp,
            );

            scope.metadata.reset(index);
        }
    }

    pub fn remove(&mut self, id: Valid<TextureId>) -> bool {
        let (index32, epoch, _) = id.0.unzip();
        let index = index32 as usize;

        if index > self.metadata.owned.len() {
            return false;
        }

        self.debug_assert_in_bounds(index);

        unsafe {
            if self.metadata.owned.get(index).unwrap_unchecked() {
                let existing_epoch = *self.metadata.epochs.get_unchecked_mut(index);
                assert_eq!(existing_epoch, epoch);

                self.start_set.complex.remove(&index32);
                self.end_set.complex.remove(&index32);

                self.metadata.reset(index);

                return true;
            }
        }

        false
    }

    pub fn remove_abandoned(&mut self, id: Valid<TextureId>) -> bool {
        let (index32, epoch, _) = id.0.unzip();
        let index = index32 as usize;

        if index > self.metadata.owned.len() {
            return false;
        }

        self.debug_assert_in_bounds(index);

        unsafe {
            if self.metadata.owned.get(index).unwrap_unchecked() {
                let existing_epoch = self.metadata.epochs.get_unchecked_mut(index);
                let existing_ref_count = self.metadata.ref_counts.get_unchecked_mut(index);

                if *existing_epoch == epoch
                    && existing_ref_count.as_mut().unwrap_unchecked().load() == 1
                {
                    self.start_set.complex.remove(&index32);
                    self.end_set.complex.remove(&index32);

                    self.metadata.reset(index);

                    return true;
                }
            }
        }

        false
    }
}

#[derive(Clone)]
enum Either<L, R> {
    Left(L),
    Right(R),
}

impl<L, R, D> Iterator for Either<L, R>
where
    L: Iterator<Item = D>,
    R: Iterator<Item = D>,
{
    type Item = D;

    fn next(&mut self) -> Option<Self::Item> {
        match *self {
            Either::Left(ref mut inner) => inner.next(),
            Either::Right(ref mut inner) => inner.next(),
        }
    }
}

#[derive(Debug, Clone)]
enum SingleOrManyStates<S, M> {
    Single(S),
    Many(M),
}

#[derive(Clone)]
enum LayeredStateProvider<'a> {
    KnownSingle {
        state: TextureUses,
    },
    Selector {
        selector: TextureSelector,
        state: TextureUses,
    },
    TextureSet {
        set: &'a TextureStateSet,
    },
}
impl<'a> LayeredStateProvider<'a> {
    fn from_option(selector: Option<TextureSelector>, state: TextureUses) -> Self {
        match selector {
            Some(selector) => Self::Selector { selector, state },
            None => Self::KnownSingle { state },
        }
    }

    #[inline(always)]
    unsafe fn get_layers(
        self,
        texture_data: Option<(&LifeGuard, &TextureSelector)>,
        index32: u32,
        index: usize,
    ) -> SingleOrManyStates<
        TextureUses,
        impl Iterator<Item = (TextureSelector, TextureUses)> + Clone + 'a,
    > {
        match self {
            LayeredStateProvider::KnownSingle { state } => SingleOrManyStates::Single(state),
            LayeredStateProvider::Selector { selector, state } => {
                if *texture_data.unwrap().1 == selector {
                    SingleOrManyStates::Single(state)
                } else {
                    SingleOrManyStates::Many(Either::Left(iter::once((selector, state))))
                }
            }
            LayeredStateProvider::TextureSet { set } => {
                let new_state = *set.simple.get_unchecked(index);

                if new_state == TextureUses::COMPLEX {
                    let new_complex = set.complex.get(&index32).unwrap_unchecked();

                    SingleOrManyStates::Many(Either::Right(new_complex.to_selector_state_iter()))
                } else {
                    SingleOrManyStates::Single(new_state)
                }
            }
        }
    }
}

#[inline(always)]
unsafe fn texture_data_from_texture<A: hub::HalApi>(
    storage: &hub::Storage<Texture<A>, TextureId>,
    index32: u32,
) -> (&LifeGuard, &TextureSelector) {
    let texture = storage.get_unchecked(index32);
    (&texture.life_guard, &texture.full_range)
}

// I think the trick to the barriers is that there are three different possible operations:

// merge from a single state
// barrier from a single state into a double sided tracker
// update from a single state into a double sided tracker

// recording rpasses is a merge(rpass, resource)
// cmd_buf <- rpass is a barrier(cmd_buf, rpass) + update(cmd_buf, rpass)
// device <- cmd_buf is a barrier(device, cmd_buff.start) + update(cmd_buf, cmd_buf.end)

#[inline(always)]
unsafe fn insert_or_merge<A: hub::HalApi>(
    texture_data: (&LifeGuard, &TextureSelector),
    current_state_set: &mut TextureStateSet,
    resource_metadata: &mut ResourceMetadata<A>,
    index32: u32,
    index: usize,
    state_provider: LayeredStateProvider<'_>,
    metadata_provider: ResourceMetadataProvider<'_, A>,
) -> Result<(), UsageConflict> {
    let currently_owned = resource_metadata.owned.get(index).unwrap_unchecked();

    if !currently_owned {
        insert(
            Some(texture_data),
            None,
            current_state_set,
            resource_metadata,
            index32,
            index,
            state_provider,
            None,
            metadata_provider,
        );
        return Ok(());
    }

    merge(
        texture_data,
        current_state_set,
        index32,
        index,
        state_provider,
        metadata_provider,
    )
}

#[inline(always)]
unsafe fn insert_or_barrier_update<A: hub::HalApi>(
    texture_data: (&LifeGuard, &TextureSelector),
    start_state: Option<&mut TextureStateSet>,
    current_state_set: &mut TextureStateSet,
    resource_metadata: &mut ResourceMetadata<A>,
    index32: u32,
    index: usize,
    start_state_provider: LayeredStateProvider<'_>,
    end_state_provider: Option<LayeredStateProvider<'_>>,
    metadata_provider: ResourceMetadataProvider<'_, A>,
    barriers: &mut Vec<PendingTransition<TextureUses>>,
) {
    let currently_owned = resource_metadata.owned.get(index).unwrap_unchecked();

    if !currently_owned {
        insert(
            Some(texture_data),
            start_state,
            current_state_set,
            resource_metadata,
            index32,
            index,
            start_state_provider,
            end_state_provider,
            metadata_provider,
        );
        return;
    }

    let update_state_provider = end_state_provider.unwrap_or_else(|| start_state_provider.clone());
    barrier(
        texture_data,
        current_state_set,
        index32,
        index,
        start_state_provider,
        barriers,
    );

    let start_state_set = start_state.unwrap();
    update(
        texture_data,
        start_state_set,
        current_state_set,
        index32,
        index,
        update_state_provider,
    );
}

#[inline(always)]
unsafe fn insert<A: hub::HalApi>(
    texture_data: Option<(&LifeGuard, &TextureSelector)>,
    start_state: Option<&mut TextureStateSet>,
    end_state: &mut TextureStateSet,
    resource_metadata: &mut ResourceMetadata<A>,
    index32: u32,
    index: usize,
    start_state_provider: LayeredStateProvider<'_>,
    end_state_provider: Option<LayeredStateProvider<'_>>,
    metadata_provider: ResourceMetadataProvider<'_, A>,
) {
    let start_layers = start_state_provider.get_layers(texture_data, index32, index);
    match start_layers {
        SingleOrManyStates::Single(state) => {
            // This should only ever happen with a wgpu bug, but let's just double
            // check that resource states don't have any conflicts.
            debug_assert_eq!(invalid_resource_state(state), false);

            if let Some(start_state) = start_state {
                *start_state.simple.get_unchecked_mut(index) = state;
            }
            if end_state_provider.is_none() {
                *end_state.simple.get_unchecked_mut(index) = state;
            }
        }
        SingleOrManyStates::Many(state_iter) => {
            let full_range = texture_data.unwrap().1.clone();

            let complex = ComplexTextureState::from_selector_state_iter(full_range, state_iter);

            if let Some(start_state) = start_state {
                *start_state.simple.get_unchecked_mut(index) = TextureUses::COMPLEX;
                start_state.complex.insert(index32, complex.clone());
            }

            if end_state_provider.is_none() {
                *end_state.simple.get_unchecked_mut(index) = TextureUses::COMPLEX;
                end_state.complex.insert(index32, complex);
            }
        }
    }

    if let Some(end_state_provider) = end_state_provider {
        match end_state_provider.get_layers(texture_data, index32, index) {
            SingleOrManyStates::Single(state) => {
                // This should only ever happen with a wgpu bug, but let's just double
                // check that resource states don't have any conflicts.
                debug_assert_eq!(invalid_resource_state(state), false);

                *end_state.simple.get_unchecked_mut(index) = state;
            }
            SingleOrManyStates::Many(state_iter) => {
                let full_range = texture_data.unwrap().1.clone();

                let complex = ComplexTextureState::from_selector_state_iter(full_range, state_iter);

                *end_state.simple.get_unchecked_mut(index) = TextureUses::COMPLEX;
                end_state.complex.insert(index32, complex);
            }
        }
    }

    let (epoch, ref_count) =
        metadata_provider.get_own(texture_data.map(|(life_guard, _)| life_guard), index);

    resource_metadata.owned.set(index, true);
    *resource_metadata.epochs.get_unchecked_mut(index) = epoch;
    *resource_metadata.ref_counts.get_unchecked_mut(index) = Some(ref_count);
}

#[inline(always)]
unsafe fn merge<A: hub::HalApi>(
    texture_data: (&LifeGuard, &TextureSelector),
    current_state_set: &mut TextureStateSet,
    index32: u32,
    index: usize,
    state_provider: LayeredStateProvider<'_>,
    metadata_provider: ResourceMetadataProvider<'_, A>,
) -> Result<(), UsageConflict> {
    let current_simple = current_state_set.simple.get_unchecked_mut(index);
    let current_state = if *current_simple == TextureUses::COMPLEX {
        SingleOrManyStates::Many(
            current_state_set
                .complex
                .get_mut(&index32)
                .unwrap_unchecked(),
        )
    } else {
        SingleOrManyStates::Single(current_simple)
    };

    let new_state = state_provider.get_layers(Some(texture_data), index32, index);

    match (current_state, new_state) {
        (SingleOrManyStates::Single(current_simple), SingleOrManyStates::Single(new_simple)) => {
            let merged_state = *current_simple | new_simple;

            if invalid_resource_state(merged_state) {
                return Err(UsageConflict::from_texture(
                    TextureId::zip(index32, metadata_provider.get_epoch(index), A::VARIANT),
                    texture_data.1.clone(),
                    *current_simple,
                    new_simple,
                ));
            }

            *current_simple = merged_state;
        }
        (SingleOrManyStates::Single(current_simple), SingleOrManyStates::Many(new_many)) => {
            let mut new_complex = ComplexTextureState::from_selector_state_iter(
                texture_data.1.clone(),
                iter::once((texture_data.1.clone(), *current_simple)),
            );

            for (selector, new_state) in new_many {
                let merged_state = *current_simple | new_state;

                if invalid_resource_state(merged_state) {
                    return Err(UsageConflict::from_texture(
                        TextureId::zip(index32, metadata_provider.get_epoch(index), A::VARIANT),
                        selector,
                        *current_simple,
                        new_state,
                    ));
                }

                for mip in
                    &mut new_complex.mips[selector.mips.start as usize..selector.mips.end as usize]
                {
                    for &mut (_, ref mut current_layer_state) in
                        mip.isolate(&selector.layers, TextureUses::UNKNOWN)
                    {
                        *current_layer_state = merged_state;
                    }

                    mip.coalesce();
                }
            }

            *current_simple = TextureUses::COMPLEX;
            current_state_set.complex.insert(index32, new_complex);
        }
        (SingleOrManyStates::Many(current_complex), SingleOrManyStates::Single(new_simple)) => {
            for (mip_id, mip) in current_complex.mips.iter_mut().enumerate() {
                let mip_id = mip_id as u32;

                for &mut (ref layers, ref mut current_layer_state) in mip.iter_mut() {
                    let merged_state = *current_layer_state | new_simple;

                    // Once we remove unknown, this will never be empty, as simple states are never unknown.
                    let merged_state = merged_state - TextureUses::UNKNOWN;

                    if invalid_resource_state(merged_state) {
                        return Err(UsageConflict::from_texture(
                            TextureId::zip(index32, metadata_provider.get_epoch(index), A::VARIANT),
                            TextureSelector {
                                mips: mip_id..mip_id + 1,
                                layers: layers.clone(),
                            },
                            *current_layer_state,
                            new_simple,
                        ));
                    }

                    *current_layer_state = merged_state;
                }

                mip.coalesce();
            }
        }
        (SingleOrManyStates::Many(current_complex), SingleOrManyStates::Many(new_many)) => {
            for (selector, new_state) in new_many {
                for mip_id in selector.mips {
                    debug_assert!((mip_id as usize) < current_complex.mips.len());

                    let mip = current_complex.mips.get_unchecked_mut(mip_id as usize);

                    for &mut (ref layers, ref mut current_layer_state) in
                        mip.isolate(&selector.layers, TextureUses::UNKNOWN)
                    {
                        let merged_state = *current_layer_state | new_state;
                        let merged_state = merged_state - TextureUses::UNKNOWN;

                        if merged_state.is_empty() {
                            // We know nothing about this state, lets just move on.
                            continue;
                        }

                        if invalid_resource_state(merged_state) {
                            return Err(UsageConflict::from_texture(
                                TextureId::zip(
                                    index32,
                                    metadata_provider.get_epoch(index),
                                    A::VARIANT,
                                ),
                                TextureSelector {
                                    mips: mip_id..mip_id + 1,
                                    layers: layers.clone(),
                                },
                                *current_layer_state,
                                new_state,
                            ));
                        }
                        *current_layer_state = merged_state;
                    }

                    mip.coalesce();
                }
            }
        }
    }
    Ok(())
}

#[inline(always)]
unsafe fn barrier(
    texture_data: (&LifeGuard, &TextureSelector),
    current_state_set: &TextureStateSet,
    index32: u32,
    index: usize,
    state_provider: LayeredStateProvider<'_>,
    barriers: &mut Vec<PendingTransition<TextureUses>>,
) {
    let current_simple = *current_state_set.simple.get_unchecked(index);
    let current_state = if current_simple == TextureUses::COMPLEX {
        SingleOrManyStates::Many(current_state_set.complex.get(&index32).unwrap_unchecked())
    } else {
        SingleOrManyStates::Single(current_simple)
    };

    let new_state = state_provider.get_layers(Some(texture_data), index32, index);

    match (current_state, new_state) {
        (SingleOrManyStates::Single(current_simple), SingleOrManyStates::Single(new_simple)) => {
            if skip_barrier(current_simple, new_simple) {
                return;
            }

            barriers.push(PendingTransition {
                id: index32,
                selector: texture_data.1.clone(),
                usage: current_simple..new_simple,
            })
        }
        (SingleOrManyStates::Single(current_simple), SingleOrManyStates::Many(new_many)) => {
            for (selector, new_state) in new_many {
                if new_state == TextureUses::UNKNOWN {
                    continue;
                }

                if skip_barrier(current_simple, new_state) {
                    continue;
                }

                barriers.push(PendingTransition {
                    id: index32,
                    selector,
                    usage: current_simple..new_state,
                })
            }
        }
        (SingleOrManyStates::Many(current_complex), SingleOrManyStates::Single(new_simple)) => {
            for (mip_id, mip) in current_complex.mips.iter().enumerate() {
                let mip_id = mip_id as u32;

                for &(ref layers, current_layer_state) in mip.iter() {
                    if current_layer_state == TextureUses::UNKNOWN {
                        continue;
                    }

                    if skip_barrier(current_layer_state, new_simple) {
                        continue;
                    }

                    barriers.push(PendingTransition {
                        id: index32,
                        selector: TextureSelector {
                            mips: mip_id..mip_id + 1,
                            layers: layers.clone(),
                        },
                        usage: current_layer_state..new_simple,
                    });
                }
            }
        }
        (SingleOrManyStates::Many(current_complex), SingleOrManyStates::Many(new_many)) => {
            for (selector, new_state) in new_many {
                for mip_id in selector.mips {
                    debug_assert!((mip_id as usize) < current_complex.mips.len());

                    let mip = current_complex.mips.get_unchecked(mip_id as usize);

                    for (layers, current_layer_state) in mip.iter_filter(&selector.layers) {
                        if *current_layer_state == TextureUses::UNKNOWN
                            || new_state == TextureUses::UNKNOWN
                        {
                            continue;
                        }

                        if skip_barrier(*current_layer_state, new_state) {
                            continue;
                        }

                        barriers.push(PendingTransition {
                            id: index32,
                            selector: TextureSelector {
                                mips: mip_id..mip_id + 1,
                                layers,
                            },
                            usage: *current_layer_state..new_state,
                        });
                    }
                }
            }
        }
    }
}

#[inline(always)]
unsafe fn update(
    texture_data: (&LifeGuard, &TextureSelector),
    start_state_set: &mut TextureStateSet,
    current_state_set: &mut TextureStateSet,
    index32: u32,
    index: usize,
    state_provider: LayeredStateProvider<'_>,
) {
    let start_simple = *start_state_set.simple.get_unchecked(index);
    let mut start_complex = None;
    if start_simple == TextureUses::COMPLEX {
        start_complex = Some(start_state_set.complex.get_mut(&index32).unwrap_unchecked());
    }

    let current_simple = current_state_set.simple.get_unchecked_mut(index);
    let current_state = if *current_simple == TextureUses::COMPLEX {
        SingleOrManyStates::Many(
            current_state_set
                .complex
                .get_mut(&index32)
                .unwrap_unchecked(),
        )
    } else {
        SingleOrManyStates::Single(current_simple)
    };

    let new_state = state_provider.get_layers(Some(texture_data), index32, index);

    match (current_state, new_state) {
        (SingleOrManyStates::Single(current_simple), SingleOrManyStates::Single(new_simple)) => {
            *current_simple = new_simple;
        }
        (SingleOrManyStates::Single(current_simple), SingleOrManyStates::Many(new_many)) => {
            let mut new_complex = ComplexTextureState::from_selector_state_iter(
                texture_data.1.clone(),
                iter::once((texture_data.1.clone(), *current_simple)),
            );

            for (selector, mut new_state) in new_many {
                if new_state == TextureUses::UNKNOWN {
                    new_state = *current_simple;
                }
                for mip in
                    &mut new_complex.mips[selector.mips.start as usize..selector.mips.end as usize]
                {
                    for &mut (_, ref mut current_layer_state) in
                        mip.isolate(&selector.layers, TextureUses::UNKNOWN)
                    {
                        *current_layer_state = new_state;
                    }

                    mip.coalesce();
                }
            }

            *current_simple = TextureUses::COMPLEX;
            current_state_set.complex.insert(index32, new_complex);
        }
        (SingleOrManyStates::Many(current_complex), SingleOrManyStates::Single(new_single)) => {
            for (mip_id, mip) in current_complex.mips.iter().enumerate() {
                for &(ref layers, current_layer_state) in mip.iter() {
                    // If this state is unknown, that means that the start is _also_ unknown.
                    if current_layer_state == TextureUses::UNKNOWN {
                        if let Some(&mut ref mut start_complex) = start_complex {
                            debug_assert!(mip_id < start_complex.mips.len());

                            let start_mip = start_complex.mips.get_unchecked_mut(mip_id);

                            for &mut (_, ref mut current_start_state) in
                                start_mip.isolate(layers, TextureUses::UNKNOWN)
                            {
                                debug_assert_eq!(*current_start_state, TextureUses::UNKNOWN);
                                *current_start_state = new_single;
                            }

                            start_mip.coalesce();
                        }
                    }
                }
            }

            *current_state_set.simple.get_unchecked_mut(index) = new_single;
            current_state_set
                .complex
                .remove(&index32)
                .unwrap_unchecked();
        }
        (SingleOrManyStates::Many(current_complex), SingleOrManyStates::Many(new_many)) => {
            for (selector, new_state) in new_many {
                if new_state == TextureUses::UNKNOWN {
                    // We know nothing new
                    continue;
                }

                for mip_id in selector.mips {
                    let mip_id = mip_id as usize;
                    debug_assert!(mip_id < current_complex.mips.len());

                    let mip = current_complex.mips.get_unchecked_mut(mip_id);

                    for &mut (ref layers, ref mut current_layer_state) in
                        mip.isolate(&selector.layers, TextureUses::UNKNOWN)
                    {
                        if *current_layer_state == TextureUses::UNKNOWN
                            && new_state != TextureUses::UNKNOWN
                        {
                            if let Some(&mut ref mut start_complex) = start_complex {
                                debug_assert!(mip_id < start_complex.mips.len());

                                let start_mip = start_complex.mips.get_unchecked_mut(mip_id);

                                for &mut (_, ref mut current_start_state) in
                                    start_mip.isolate(layers, TextureUses::UNKNOWN)
                                {
                                    debug_assert_eq!(*current_start_state, TextureUses::UNKNOWN);
                                    *current_start_state = new_state;
                                }

                                start_mip.coalesce();
                            }
                        }

                        *current_layer_state = new_state;
                    }

                    mip.coalesce();
                }
            }
        }
    }
}
