mod buffer;
mod range;
mod stateless;
mod texture;

use crate::{
    binding_model, command, conv, hub,
    id::{self, TypedId},
    pipeline, resource, Epoch, RefCount,
};

use bit_vec::BitVec;
use std::{fmt, marker::PhantomData, mem, num::NonZeroU32, ops};
use thiserror::Error;

pub(crate) use buffer::{BufferBindGroupState, BufferTracker, BufferUsageScope};
pub(crate) use stateless::{StatelessBindGroupSate, StatelessTracker};
pub(crate) use texture::{
    TextureBindGroupState, TextureSelector, TextureTracker, TextureUsageScope,
};

/// A structure containing all the information about a particular resource
/// transition. User code should be able to generate a pipeline barrier
/// based on the contents.
#[derive(Debug, PartialEq)]
pub(crate) struct PendingTransition<S: ResourceUses> {
    pub id: u32,
    pub selector: S::Selector,
    pub usage: ops::Range<S>,
}

impl PendingTransition<hal::BufferUses> {
    /// Produce the hal barrier corresponding to the transition.
    pub fn into_hal<'a, A: hal::Api>(
        self,
        buf: &'a resource::Buffer<A>,
    ) -> hal::BufferBarrier<'a, A> {
        log::trace!("\tbuffer -> {:?}", self);
        let buffer = buf.raw.as_ref().expect("Buffer is destroyed");
        hal::BufferBarrier {
            buffer,
            usage: self.usage,
        }
    }
}

impl PendingTransition<hal::TextureUses> {
    /// Produce the hal barrier corresponding to the transition.
    pub fn into_hal<'a, A: hal::Api>(
        self,
        tex: &'a resource::Texture<A>,
    ) -> hal::TextureBarrier<'a, A> {
        log::trace!("\ttexture -> {:?}", self);
        let texture = tex.inner.as_raw().expect("Texture is destroyed");

        // These showing up in a barrier is always a bug
        debug_assert_ne!(self.usage.start, hal::TextureUses::UNKNOWN);
        debug_assert_ne!(self.usage.end, hal::TextureUses::UNKNOWN);

        hal::TextureBarrier {
            texture,
            range: wgt::ImageSubresourceRange {
                aspect: wgt::TextureAspect::All,
                base_mip_level: self.selector.mips.start,
                mip_level_count: NonZeroU32::new(self.selector.mips.end - self.selector.mips.start),
                base_array_layer: self.selector.layers.start,
                array_layer_count: NonZeroU32::new(
                    self.selector.layers.end - self.selector.layers.start,
                ),
            },
            usage: self.usage,
        }
    }
}

pub trait ResourceUses:
    fmt::Debug + ops::BitAnd<Output = Self> + ops::BitOr<Output = Self> + PartialEq + Sized + Copy
{
    const EXCLUSIVE: Self;

    type Id: Copy + fmt::Debug + TypedId;
    type Selector: fmt::Debug;

    fn bits(self) -> u16;
    fn all_ordered(self) -> bool;
    fn any_exclusive(self) -> bool;
    fn uninit(self) -> bool;
}

fn invalid_resource_state<T: ResourceUses>(state: T) -> bool {
    // Is power of two also means "is one bit set". We check for this as if
    // we're in any exclusive state, we must only be in a single state.
    state.any_exclusive() && !conv::is_power_of_two_u16(state.bits())
}

fn skip_barrier<T: ResourceUses>(old_state: T, new_state: T) -> bool {
    // If the state didn't change and all the usages are ordered, the hardware
    // will guarentee the order of accesses, so we do not need to issue a barrier at all
    old_state == new_state && old_state.all_ordered()
}

fn resize_bitvec<B: bit_vec::BitBlock>(vec: &mut BitVec<B>, size: usize) {
    let owned_size_to_grow = size.checked_sub(vec.len());
    if let Some(delta) = owned_size_to_grow {
        if delta != 0 {
            vec.grow(delta, false);
        }
    } else {
        vec.truncate(size);
    }
}

fn iterate_bitvec_indices(ownership: &BitVec<usize>) -> impl Iterator<Item = usize> + '_ {
    const BITS_PER_BLOCK: usize = mem::size_of::<usize>() * 8;

    let size = ownership.len();

    ownership
        .blocks()
        .enumerate()
        .filter(|(_, word)| *word != 0)
        .flat_map(move |(word_index, mut word)| {
            let bit_start = word_index * BITS_PER_BLOCK;
            let bit_end = (bit_start + BITS_PER_BLOCK).min(size);

            (bit_start..bit_end).filter(move |_| {
                let active = word & 0b1 != 0;
                word >>= 1;

                active
            })
        })
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum UsageConflict {
    #[error("Attempted to use buffer {id:?} which is invalid.")]
    BufferInvalid { id: id::BufferId },
    #[error("Attempted to use texture {id:?} which is invalid.")]
    TextureInvalid { id: id::TextureId },
    #[error("Attempted to use buffer {id:?} with {invalid_use}.")]
    Buffer {
        id: id::BufferId,
        invalid_use: InvalidUse<hal::BufferUses>,
    },
    #[error("Attempted to use a texture {id:?} mips {mip_levels:?} layers {array_layers:?} with {invalid_use}.")]
    Texture {
        id: id::TextureId,
        mip_levels: ops::Range<u32>,
        array_layers: ops::Range<u32>,
        invalid_use: InvalidUse<hal::TextureUses>,
    },
}
impl UsageConflict {
    fn from_buffer(
        id: id::BufferId,
        current_state: hal::BufferUses,
        new_state: hal::BufferUses,
    ) -> Self {
        Self::Buffer {
            id,
            invalid_use: InvalidUse {
                current_state,
                new_state,
            },
        }
    }

    fn from_texture(
        id: id::TextureId,
        selector: TextureSelector,
        current_state: hal::TextureUses,
        new_state: hal::TextureUses,
    ) -> Self {
        Self::Texture {
            id: id,
            mip_levels: selector.mips,
            array_layers: selector.layers,
            invalid_use: InvalidUse {
                current_state,
                new_state,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InvalidUse<T> {
    current_state: T,
    new_state: T,
}

impl<T: ResourceUses> fmt::Display for InvalidUse<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let current = self.current_state;
        let new = self.new_state;

        let current_exclusive = current & T::EXCLUSIVE;
        let new_exclusive = new & T::EXCLUSIVE;

        let exclusive = current_exclusive | new_exclusive;

        write!(
            f,
            "conflicting usages. Current usage {current:?} and new usage {new:?}. \
            {exclusive:?} is an exclusive usage and cannot be used with any other\
            usages within the usage scope (renderpass or compute dispatch)"
        )
    }
}

#[derive(Debug)]
pub(crate) struct ResourceMetadata<A: hub::HalApi> {
    owned: BitVec<usize>,
    ref_counts: Vec<Option<RefCount>>,
    epochs: Vec<Epoch>,

    _phantom: PhantomData<A>,
}
impl<A: hub::HalApi> ResourceMetadata<A> {
    pub fn new() -> Self {
        Self {
            owned: BitVec::default(),
            ref_counts: Vec::new(),
            epochs: Vec::new(),

            _phantom: PhantomData,
        }
    }

    pub fn set_size(&mut self, size: usize) {
        self.ref_counts.resize(size, None);
        self.epochs.resize(size, u32::MAX);

        resize_bitvec(&mut self.owned, size);
    }

    fn debug_assert_in_bounds(&self, index: usize) {
        debug_assert!(index < self.owned.len());
        debug_assert!(index < self.ref_counts.len());
        debug_assert!(index < self.epochs.len());

        debug_assert!(if self.owned.get(index).unwrap() {
            self.ref_counts[index].is_some()
        } else {
            true
        });
    }

    fn is_empty(&self) -> bool {
        !self.owned.any()
    }

    fn used<Id: TypedId>(&self) -> impl Iterator<Item = id::Valid<Id>> + '_ {
        if !self.owned.is_empty() {
            self.debug_assert_in_bounds(self.owned.len() - 1)
        };
        iterate_bitvec_indices(&self.owned).map(move |index| {
            let epoch = unsafe { *self.epochs.get_unchecked(index) };
            id::Valid(Id::zip(index as u32, epoch, A::VARIANT))
        })
    }

    unsafe fn reset(&mut self, index: usize) {
        *self.ref_counts.get_unchecked_mut(index) = None;
        *self.epochs.get_unchecked_mut(index) = u32::MAX;
        self.owned.set(index, false);
    }
}

pub(crate) struct BindGroupStates<A: hub::HalApi> {
    pub buffers: BufferBindGroupState<A>,
    pub textures: TextureBindGroupState<A>,
    pub views: StatelessBindGroupSate<resource::TextureView<A>, id::TextureViewId>,
    pub samplers: StatelessBindGroupSate<resource::Sampler<A>, id::SamplerId>,
}

impl<A: hub::HalApi> BindGroupStates<A> {
    pub fn new() -> Self {
        Self {
            buffers: BufferBindGroupState::new(),
            textures: TextureBindGroupState::new(),
            views: StatelessBindGroupSate::new(),
            samplers: StatelessBindGroupSate::new(),
        }
    }

    pub fn optimize(&mut self) {
        self.buffers.optimize();
        self.textures.optimize();
        self.views.optimize();
        self.samplers.optimize();
    }
}

pub(crate) struct RenderBundleScope<A: hub::HalApi> {
    pub buffers: BufferUsageScope<A>,
    pub textures: TextureUsageScope<A>,
    // Don't need to track views and samplers, they are never used directly, only by bind groups.
    pub bind_groups: StatelessTracker<A, binding_model::BindGroup<A>, id::BindGroupId>,
    pub render_pipelines: StatelessTracker<A, pipeline::RenderPipeline<A>, id::RenderPipelineId>,
    pub query_sets: StatelessTracker<A, resource::QuerySet<A>, id::QuerySetId>,
}

impl<A: hub::HalApi> RenderBundleScope<A> {
    pub fn new(
        buffers: &hub::Storage<resource::Buffer<A>, id::BufferId>,
        textures: &hub::Storage<resource::Texture<A>, id::TextureId>,
        bind_groups: &hub::Storage<binding_model::BindGroup<A>, id::BindGroupId>,
        render_pipelines: &hub::Storage<pipeline::RenderPipeline<A>, id::RenderPipelineId>,
        query_sets: &hub::Storage<resource::QuerySet<A>, id::QuerySetId>,
    ) -> Self {
        let mut value = Self {
            buffers: BufferUsageScope::new(),
            textures: TextureUsageScope::new(),
            bind_groups: StatelessTracker::new(),
            render_pipelines: StatelessTracker::new(),
            query_sets: StatelessTracker::new(),
        };

        value.buffers.set_size(buffers.len());
        value.textures.set_size(textures.len());
        value.bind_groups.set_size(bind_groups.len());
        value.render_pipelines.set_size(render_pipelines.len());
        value.query_sets.set_size(query_sets.len());

        value
    }

    pub unsafe fn extend_from_bind_group(
        &mut self,
        textures: &hub::Storage<resource::Texture<A>, id::TextureId>,
        bind_group: &BindGroupStates<A>,
    ) -> Result<(), UsageConflict> {
        self.buffers.extend_from_bind_group(&bind_group.buffers)?;
        self.textures
            .extend_from_bind_group(textures, &bind_group.textures)?;

        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct UsageScope<A: hub::HalApi> {
    pub buffers: BufferUsageScope<A>,
    pub textures: TextureUsageScope<A>,
}

impl<A: hub::HalApi> UsageScope<A> {
    pub fn new(
        buffers: &hub::Storage<resource::Buffer<A>, id::BufferId>,
        textures: &hub::Storage<resource::Texture<A>, id::TextureId>,
    ) -> Self {
        let mut value = Self {
            buffers: BufferUsageScope::new(),
            textures: TextureUsageScope::new(),
        };

        value.buffers.set_size(buffers.len());
        value.textures.set_size(textures.len());

        value
    }

    pub unsafe fn extend_from_bind_group(
        &mut self,
        textures: &hub::Storage<resource::Texture<A>, id::TextureId>,
        bind_group: &BindGroupStates<A>,
    ) -> Result<(), UsageConflict> {
        self.buffers.extend_from_bind_group(&bind_group.buffers)?;
        self.textures
            .extend_from_bind_group(textures, &bind_group.textures)?;

        Ok(())
    }

    pub unsafe fn extend_from_render_bundle(
        &mut self,
        textures: &hub::Storage<resource::Texture<A>, id::TextureId>,
        render_bundle: &RenderBundleScope<A>,
    ) -> Result<(), UsageConflict> {
        self.buffers.extend_from_scope(&render_bundle.buffers)?;
        self.textures
            .extend_from_scope(textures, &render_bundle.textures)?;

        Ok(())
    }
}

pub(crate) struct Tracker<A: hub::HalApi> {
    pub buffers: BufferTracker<A>,
    pub textures: TextureTracker<A>,
    pub views: StatelessTracker<A, resource::TextureView<A>, id::TextureViewId>,
    pub samplers: StatelessTracker<A, resource::Sampler<A>, id::SamplerId>,
    pub bind_groups: StatelessTracker<A, binding_model::BindGroup<A>, id::BindGroupId>,
    pub compute_pipelines: StatelessTracker<A, pipeline::ComputePipeline<A>, id::ComputePipelineId>,
    pub render_pipelines: StatelessTracker<A, pipeline::RenderPipeline<A>, id::RenderPipelineId>,
    pub bundles: StatelessTracker<A, command::RenderBundle<A>, id::RenderBundleId>,
    pub query_sets: StatelessTracker<A, resource::QuerySet<A>, id::QuerySetId>,
}

impl<A: hub::HalApi> Tracker<A> {
    pub fn new() -> Self {
        Self {
            buffers: BufferTracker::new(),
            textures: TextureTracker::new(),
            views: StatelessTracker::new(),
            samplers: StatelessTracker::new(),
            bind_groups: StatelessTracker::new(),
            compute_pipelines: StatelessTracker::new(),
            render_pipelines: StatelessTracker::new(),
            bundles: StatelessTracker::new(),
            query_sets: StatelessTracker::new(),
        }
    }

    pub fn set_size(
        &mut self,
        buffers: Option<&hub::Storage<resource::Buffer<A>, id::BufferId>>,
        textures: Option<&hub::Storage<resource::Texture<A>, id::TextureId>>,
        views: Option<&hub::Storage<resource::TextureView<A>, id::TextureViewId>>,
        samplers: Option<&hub::Storage<resource::Sampler<A>, id::SamplerId>>,
        bind_groups: Option<&hub::Storage<binding_model::BindGroup<A>, id::BindGroupId>>,
        compute_pipelines: Option<
            &hub::Storage<pipeline::ComputePipeline<A>, id::ComputePipelineId>,
        >,
        render_pipelines: Option<&hub::Storage<pipeline::RenderPipeline<A>, id::RenderPipelineId>>,
        bundles: Option<&hub::Storage<command::RenderBundle<A>, id::RenderBundleId>>,
        query_sets: Option<&hub::Storage<resource::QuerySet<A>, id::QuerySetId>>,
    ) {
        if let Some(buffers) = buffers {
            self.buffers.set_size(buffers.len());
        };
        if let Some(textures) = textures {
            self.textures.set_size(textures.len());
        };
        if let Some(views) = views {
            self.views.set_size(views.len());
        };
        if let Some(samplers) = samplers {
            self.samplers.set_size(samplers.len());
        };
        if let Some(bind_groups) = bind_groups {
            self.bind_groups.set_size(bind_groups.len());
        };
        if let Some(compute_pipelines) = compute_pipelines {
            self.compute_pipelines.set_size(compute_pipelines.len());
        }
        if let Some(render_pipelines) = render_pipelines {
            self.render_pipelines.set_size(render_pipelines.len());
        };
        if let Some(bundles) = bundles {
            self.bundles.set_size(bundles.len());
        };
        if let Some(query_sets) = query_sets {
            self.query_sets.set_size(query_sets.len());
        };
    }

    pub unsafe fn extend_from_bind_group(
        &mut self,
        textures: &hub::Storage<resource::Texture<A>, id::TextureId>,
        scope: &mut UsageScope<A>,
        bind_group: &BindGroupStates<A>,
    ) {
        self.buffers
            .change_states_bind_group(&mut scope.buffers, &bind_group.buffers);
        self.textures
            .change_states_bind_group(textures, &mut scope.textures, &bind_group.textures);
    }

    pub unsafe fn extend_from_render_bundle(
        &mut self,
        render_bundle: &RenderBundleScope<A>,
    ) -> Result<(), UsageConflict> {
        self.bind_groups
            .extend_from_tracker(&render_bundle.bind_groups);
        self.render_pipelines
            .extend_from_tracker(&render_bundle.render_pipelines);
        self.query_sets
            .extend_from_tracker(&render_bundle.query_sets);

        Ok(())
    }
}
