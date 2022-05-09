use super::{range::RangedStates, PendingTransition};
use crate::{
    hub,
    id::{TextureId, TypedId, Valid},
    resource::Texture,
    track::{
        invalid_resource_state, iterate_bitvec_indices, skip_barrier,
        ResourceMetadata, ResourceUses, UsageConflict,
    },
    Epoch, RefCount,
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

    fn from_selector_state_iter(
        full_range: TextureSelector,
        state_iter: impl Iterator<Item = (TextureSelector, TextureUses)>,
    ) -> Self {
        let mut complex =
            ComplexTextureState::new(full_range.mips.len() as u32, full_range.layers.len() as u32);
        for (selector, desired_state) in state_iter {
            let mips = selector.mips.start as usize..selector.mips.end as usize;
            for mip in &mut complex.mips[mips] {
                for (_, state) in mip.isolate(&selector.layers, TextureUses::UNINITIALIZED) {
                    *state = desired_state;
                }
            }
        }
        complex
    }

    fn into_selector_state_iter(
        &self,
    ) -> impl Iterator<Item = (TextureSelector, TextureUses)> + '_ {
        self.mips.iter().enumerate().flat_map(|(mip, inner)| {
            let mip = mip as u32;
            {
                inner.iter().map(move |(layers, inner)| {
                    (
                        TextureSelector {
                            mips: mip..mip + 1,
                            layers: layers.clone(),
                        },
                        *inner,
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
        self.set.simple.resize(size, TextureUses::UNINITIALIZED);
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
                state_combine(
                    Some(storage),
                    None,
                    &mut self.set,
                    &mut self.metadata,
                    index32,
                    index,
                    LayeredStateProvider::TextureSet { set: &scope.set },
                    ResourceMetadataProvider::Indirect {
                        metadata: &scope.metadata,
                    },
                    None,
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
        for (id, selector, ref_count, state) in &bind_group.textures {
            self.extend_refcount(storage, *id, selector.clone(), ref_count, *state)?;
        }

        Ok(())
    }

    /// # Safety
    ///
    /// `id` must be a valid ID and have an ID value less than the last call to set_size.
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

        state_combine(
            Some(storage),
            None,
            &mut self.set,
            &mut self.metadata,
            index32,
            index,
            LayeredStateProvider::from_option(selector, new_state),
            ResourceMetadataProvider::Direct {
                epoch: epoch,
                ref_count: Cow::Borrowed(ref_count),
            },
            None,
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
        self.start_set
            .simple
            .resize(size, TextureUses::UNINITIALIZED);
        self.end_set.simple.resize(size, TextureUses::UNINITIALIZED);

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
            state_combine(
                None,
                Some(&mut self.start_set),
                &mut self.end_set,
                &mut self.metadata,
                index32,
                index,
                LayeredStateProvider::KnownSingle { state: usage },
                ResourceMetadataProvider::Direct {
                    epoch: epoch,
                    ref_count: Cow::Owned(ref_count),
                },
                None,
            )
            .unwrap();
        }
    }

    pub fn change_state<'a>(
        &mut self,
        storage: &'a hub::Storage<Texture<A>, TextureId>,
        id: TextureId,
        selector: TextureSelector,
        new_state: TextureUses,
    ) -> Option<(&'a Texture<A>, Option<PendingTransition<TextureUses>>)> {
        let texture = storage.get(id).ok()?;

        let (index32, epoch, _) = id.unzip();
        let index = index32 as usize;

        self.allow_index(index);

        self.debug_assert_in_bounds(index);

        unsafe {
            state_combine(
                Some(storage),
                Some(&mut self.start_set),
                &mut self.end_set,
                &mut self.metadata,
                index32,
                index,
                LayeredStateProvider::Selector {
                    selector,
                    state: new_state,
                },
                ResourceMetadataProvider::Resource { epoch },
                Some(&mut self.temp),
            )
            .unwrap();
        }

        Some((texture, self.temp.pop()))
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
                state_combine(
                    Some(storage),
                    Some(&mut self.start_set),
                    &mut self.end_set,
                    &mut self.metadata,
                    index32,
                    index,
                    LayeredStateProvider::TextureSet {
                        set: &tracker.end_set,
                    },
                    ResourceMetadataProvider::Indirect {
                        metadata: &tracker.metadata,
                    },
                    Some(&mut self.temp),
                )
                .unwrap();
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
                state_combine(
                    Some(storage),
                    Some(&mut self.start_set),
                    &mut self.end_set,
                    &mut self.metadata,
                    index32,
                    index,
                    LayeredStateProvider::TextureSet { set: &scope.set },
                    ResourceMetadataProvider::Indirect {
                        metadata: &scope.metadata,
                    },
                    Some(&mut self.temp),
                )
                .unwrap();
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
            unsafe {
                state_combine(
                    Some(storage),
                    Some(&mut self.start_set),
                    &mut self.end_set,
                    &mut self.metadata,
                    index32,
                    index,
                    LayeredStateProvider::TextureSet { set: &scope.set },
                    ResourceMetadataProvider::Indirect {
                        metadata: &scope.metadata,
                    },
                    Some(&mut self.temp),
                )
                .unwrap();
            }

            scope.metadata.reset(index);
        }
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
                } else if force {
                    assert_eq!(*existing_epoch, epoch);
                }
            }
        }

        false
    }
}

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
        match self {
            Either::Left(inner) => inner.next(),
            Either::Right(inner) => inner.next(),
        }
    }
}

enum SingleOrManyStates<S, M> {
    Single(S),
    Many(M),
}

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
            Some(selector) => Self::Selector {
                selector: selector,
                state,
            },
            None => Self::KnownSingle { state },
        }
    }

    unsafe fn get_layers<A: hub::HalApi>(
        self,
        texture: Option<&Texture<A>>,
        index32: u32,
        index: usize,
    ) -> SingleOrManyStates<TextureUses, impl Iterator<Item = (TextureSelector, TextureUses)> + 'a>
    {
        match self {
            LayeredStateProvider::KnownSingle { state } => SingleOrManyStates::Single(state),
            LayeredStateProvider::Selector { selector, state } => {
                if texture.unwrap().full_range == selector {
                    SingleOrManyStates::Single(state)
                } else {
                    SingleOrManyStates::Many(Either::Left(iter::once((selector, state))))
                }
            }
            LayeredStateProvider::TextureSet { set } => {
                let new_state = *set.simple.get_unchecked(index);

                if new_state == TextureUses::COMPLEX {
                    let new_complex = set.complex.get(&index32).unwrap_unchecked();

                    SingleOrManyStates::Many(Either::Right(new_complex.into_selector_state_iter()))
                } else {
                    SingleOrManyStates::Single(new_state)
                }
            }
        }
    }
}

enum ResourceMetadataProvider<'a, A: hub::HalApi> {
    Direct {
        epoch: Epoch,
        ref_count: Cow<'a, RefCount>,
    },
    Indirect {
        metadata: &'a ResourceMetadata<A>,
    },
    Resource {
        epoch: Epoch,
    },
}
impl<A: hub::HalApi> ResourceMetadataProvider<'_, A> {
    unsafe fn get_own(self, texture: Option<&Texture<A>>, index: usize) -> (Epoch, RefCount) {
        match self {
            ResourceMetadataProvider::Direct { epoch, ref_count } => {
                (epoch, ref_count.into_owned())
            }
            ResourceMetadataProvider::Indirect { metadata } => (
                *metadata.epochs.get_unchecked(index),
                metadata
                    .ref_counts
                    .get_unchecked(index)
                    .clone()
                    .unwrap_unchecked(),
            ),
            ResourceMetadataProvider::Resource { epoch } => {
                (epoch, texture.unwrap().life_guard.add_ref())
            }
        }
    }
    unsafe fn get_epoch(self, index: usize) -> Epoch {
        match self {
            ResourceMetadataProvider::Direct { epoch, .. }
            | ResourceMetadataProvider::Resource { epoch, .. } => epoch,
            ResourceMetadataProvider::Indirect { metadata } => {
                *metadata.epochs.get_unchecked(index)
            }
        }
    }
}

unsafe fn state_combine<A: hub::HalApi>(
    storage: Option<&hub::Storage<Texture<A>, TextureId>>,
    start_state: Option<&mut TextureStateSet>,
    end_state: &mut TextureStateSet,
    resource_metadata: &mut ResourceMetadata<A>,
    index32: u32,
    index: usize,
    state_provider: LayeredStateProvider<'_>,
    metadata_provider: ResourceMetadataProvider<'_, A>,
    barriers: Option<&mut Vec<PendingTransition<TextureUses>>>,
) -> Result<(), UsageConflict> {
    let texture = storage.map(|s| s.get_unchecked(index32));

    let currently_owned = resource_metadata.owned.get(index).unwrap_unchecked();

    if currently_owned {
        let old_simple = end_state.simple.get_unchecked_mut(index);

        let old_state = if *old_simple == TextureUses::COMPLEX {
            SingleOrManyStates::Many(end_state.complex.remove(&index32).unwrap_unchecked())
        } else {
            SingleOrManyStates::Single(*old_simple)
        };

        let new_state = set_state(
            texture,
            start_state,
            end_state,
            index32,
            index,
            state_provider,
            barriers.is_none(),
        );

        merge_or_transition_state(texture.unwrap(), old_state, new_state, index32, barriers)
            .map_err(|partial| {
                let epoch = metadata_provider.get_epoch(index);
                UsageConflict::from_texture(
                    TextureId::zip(index32, epoch, A::VARIANT),
                    partial.selector,
                    partial.current_state,
                    partial.new_state,
                )
            })?;
    } else {
        set_state(
            texture,
            start_state,
            end_state,
            index32,
            index,
            state_provider,
            false,
        );

        let (epoch, ref_count) = metadata_provider.get_own(texture, index);

        resource_metadata.owned.set(index, true);
        *resource_metadata.epochs.get_unchecked_mut(index) = epoch;
        *resource_metadata.ref_counts.get_unchecked_mut(index) = Some(ref_count);
    }

    Ok(())
}

unsafe fn set_state<'a, A: hub::HalApi>(
    texture: Option<&Texture<A>>,
    start_state: Option<&mut TextureStateSet>,
    end_state: &'a mut TextureStateSet,
    index32: u32,
    index: usize,
    state_provider: LayeredStateProvider,
    merging_and_existing: bool,
) -> SingleOrManyStates<&'a mut TextureUses, &'a mut ComplexTextureState> {
    match state_provider.get_layers(texture, index32, index) {
        SingleOrManyStates::Single(state) => {
            let reference = end_state.simple.get_unchecked_mut(index);

            if merging_and_existing && *reference == TextureUses::COMPLEX {
                let full_range = texture.unwrap().full_range.clone();

                let complex = ComplexTextureState::from_selector_state_iter(
                    full_range.clone(),
                    iter::once((full_range, state)),
                );

                *reference = state;
                let new_state = end_state
                    .complex
                    .entry(index32)
                    .insert_entry(complex)
                    .into_mut();

                return SingleOrManyStates::Many(new_state);
            }

            if let Some(start_state) = start_state {
                *start_state.simple.get_unchecked_mut(index) = state;
            }
            *reference = state;
            SingleOrManyStates::Single(reference)
        }
        SingleOrManyStates::Many(state_iter) => {
            let full_range = texture.unwrap().full_range.clone();

            let complex = ComplexTextureState::from_selector_state_iter(full_range, state_iter);

            if let Some(start_state) = start_state {
                *start_state.simple.get_unchecked_mut(index) = TextureUses::COMPLEX;
                start_state.complex.insert(index32, complex.clone());
            }
            *end_state.simple.get_unchecked_mut(index) = TextureUses::COMPLEX;
            let new_state = end_state
                .complex
                .entry(index32)
                .insert_entry(complex)
                .into_mut();

            SingleOrManyStates::Many(new_state)
        }
    }
}

struct PartialUsageConflict {
    selector: TextureSelector,
    current_state: TextureUses,
    new_state: TextureUses,
}

fn merge_or_transition_state<A: hub::HalApi>(
    texture: &Texture<A>,
    old_state: SingleOrManyStates<TextureUses, ComplexTextureState>,
    new_state: SingleOrManyStates<&mut TextureUses, &mut ComplexTextureState>,
    index32: u32,
    barriers: Option<&mut Vec<PendingTransition<TextureUses>>>,
) -> Result<(), PartialUsageConflict> {
    match (old_state, new_state) {
        (SingleOrManyStates::Single(old_simple), SingleOrManyStates::Single(new_simple)) => {
            match barriers {
                Some(barriers) => {
                    if skip_barrier(old_simple, *new_simple) {
                        return Ok(());
                    }

                    #[allow(clippy::range_plus_one)]
                    barriers.push(PendingTransition {
                        id: index32,
                        selector: texture.full_range.clone(),
                        usage: old_simple..*new_simple,
                    })
                }
                None => {
                    let merged_state = old_simple | *new_simple;

                    if invalid_resource_state(merged_state) {
                        return Err(PartialUsageConflict {
                            selector: texture.full_range.clone(),
                            current_state: old_simple,
                            new_state: *new_simple,
                        });
                    }

                    *new_simple = merged_state;
                }
            }

            Ok(())
        }
        (SingleOrManyStates::Single(old_simple), SingleOrManyStates::Many(new_many)) => {
            for (mip_index, mips) in new_many.mips.iter_mut().enumerate() {
                let mip_index = mip_index as u32;
                for (layer, new_state) in mips.iter_mut() {
                    match barriers {
                        Some(&mut ref mut barriers) => {
                            if skip_barrier(old_simple, *new_state) {
                                return Ok(());
                            }

                            #[allow(clippy::range_plus_one)]
                            barriers.push(PendingTransition {
                                id: index32,
                                selector: TextureSelector {
                                    mips: mip_index..mip_index + 1,
                                    layers: layer.clone(),
                                },
                                usage: old_simple..*new_state,
                            })
                        }
                        None => {
                            let merged_state = old_simple | *new_state;

                            if invalid_resource_state(merged_state) {
                                return Err(PartialUsageConflict {
                                    selector: TextureSelector {
                                        mips: mip_index..mip_index + 1,
                                        layers: layer.clone(),
                                    },
                                    current_state: old_simple,
                                    new_state: *new_state,
                                });
                            }

                            *new_state = merged_state;
                        }
                    }
                }

                mips.coalesce();
            }

            Ok(())
        }
        (SingleOrManyStates::Many(old_many), SingleOrManyStates::Single(new_single)) => {
            for (mip_index, mips) in old_many.mips.iter().enumerate() {
                let mip_index = mip_index as u32;
                for &(ref layer, old_state) in mips.iter() {
                    match barriers {
                        Some(&mut ref mut barriers) => {
                            if skip_barrier(old_state, *new_single) {
                                return Ok(());
                            }

                            #[allow(clippy::range_plus_one)]
                            barriers.push(PendingTransition {
                                id: index32,
                                selector: TextureSelector {
                                    mips: mip_index..mip_index + 1,
                                    layers: layer.clone(),
                                },
                                usage: old_state..*new_single,
                            })
                        }
                        None => {
                            unreachable!();
                        }
                    }
                }
            }

            Ok(())
        }
        (SingleOrManyStates::Many(mut old_complex), SingleOrManyStates::Many(new_complex)) => {
            let mut temp = Vec::new();
            debug_assert!(old_complex.mips.len() >= new_complex.mips.len());

            for (mip_id, (mip_old, mip_new)) in old_complex
                .mips
                .iter_mut()
                .zip(&mut new_complex.mips)
                .enumerate()
            {
                let level = mip_id as u32;
                temp.extend(mip_old.merge(mip_new, 0));

                for (layers, states) in temp.drain(..) {
                    match states {
                        Range {
                            start: Some(start),
                            end: Some(end),
                        } => {
                            match barriers {
                                Some(&mut ref mut barriers) => {
                                    if skip_barrier(start, end) {
                                        return Ok(());
                                    }
                                    // TODO: Can't satisfy clippy here unless we modify
                                    // `TextureSelector` to use `std::ops::RangeBounds`.
                                    #[allow(clippy::range_plus_one)]
                                    let pending = PendingTransition {
                                        id: index32,
                                        selector: TextureSelector {
                                            mips: level..level + 1,
                                            layers: layers.clone(),
                                        },
                                        usage: start..end,
                                    };

                                    barriers.push(pending);
                                }
                                None => {
                                    let merged_state = start | end;

                                    if invalid_resource_state(merged_state) {
                                        return Err(PartialUsageConflict {
                                            selector: TextureSelector {
                                                mips: level..level + 1,
                                                layers: layers.clone(),
                                            },
                                            current_state: start,
                                            new_state: end,
                                        });
                                    }

                                    for (_, state) in mip_new.isolate(&layers, end) {
                                        *state = merged_state;
                                    }
                                }
                            }
                        }
                        _ => unreachable!(),
                    };
                }

                mip_new.coalesce();
            }

            Ok(())
        }
    }
}
