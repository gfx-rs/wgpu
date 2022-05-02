mod buffer;
mod range;
mod texture;

use crate::{
    conv, hub,
    id::{self, TypedId, Valid},
    resource, Epoch, FastHashMap, Index, RefCount,
};

use bit_vec::BitVec;
use std::{
    collections::hash_map::Entry, fmt, marker::PhantomData, num::NonZeroU32, ops, vec::Drain,
};
use thiserror::Error;

pub(crate) use buffer::BufferState;
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

trait ResourceUses:
    fmt::Debug + ops::BitAnd<Output = Self> + ops::BitOr<Output = Self> + PartialEq + Sized
{
    const EXCLUSIVE: Self;

    type Id: Copy + fmt::Debug + TypedId;
    type Selector: fmt::Debug;

    fn bits(self) -> u16;
    fn all_ordered(self) -> bool;
    fn any_exclusive(self) -> bool;
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

#[derive(Clone, Debug, Error)]
pub enum UsageConflict {
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
        id: Valid<id::BufferId>,
        current_state: hal::BufferUses,
        new_state: hal::BufferUses,
    ) -> Self {
        Self::Buffer {
            id: id.0,
            invalid_use: InvalidUse {
                current_state,
                new_state,
            },
        }
    }

    fn from_texture<A: hal::Api>(
        storage: &hub::Storage<resource::Texture<A>, id::TextureId>,
        id: Valid<id::TextureId>,
        selector: Option<TextureSelector>,
        current_state: hal::TextureUses,
        new_state: hal::TextureUses,
    ) -> Self {
        let texture = &storage[id];

        let mips;
        let layers;

        match selector {
            Some(selector) => {
                mips = selector.mips;
                layers = selector.layers;
            }
            None => {
                mips = texture.full_range.mips;
                layers = texture.full_range.layers;
            }
        }

        Self::Texture {
            id: id.0,
            mip_levels: mips,
            array_layers: layers,
            invalid_use: InvalidUse {
                current_state,
                new_state,
            },
        }
    }
}

#[derive(Clone, Debug)]
struct InvalidUse<T> {
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

pub(crate) struct BindGroupStates {
    pub textures: TextureBindGroupState,
}

/// A set of trackers for all relevant resources.
///
/// `Device` uses this to track all resources allocated from that device.
/// Resources like `BindGroup`, `CommandBuffer`, and so on that may own a
/// variety of other resources also use a value of this type to keep track of
/// everything they're depending on.
#[derive(Debug)]
pub(crate) struct TrackerSet {
    pub buffers: ResourceTracker<BufferState>,
    pub textures: ResourceTracker<OldTextureState>,
    pub views: ResourceTracker<PhantomData<id::TextureViewId>>,
    pub bind_groups: ResourceTracker<PhantomData<id::BindGroupId>>,
    pub samplers: ResourceTracker<PhantomData<id::SamplerId>>,
    pub compute_pipes: ResourceTracker<PhantomData<id::ComputePipelineId>>,
    pub render_pipes: ResourceTracker<PhantomData<id::RenderPipelineId>>,
    pub bundles: ResourceTracker<PhantomData<id::RenderBundleId>>,
    pub query_sets: ResourceTracker<PhantomData<id::QuerySetId>>,
}

impl TrackerSet {
    /// Create an empty set.
    pub fn new(backend: wgt::Backend) -> Self {
        Self {
            buffers: ResourceTracker::new(backend),
            textures: ResourceTracker::new(backend),
            views: ResourceTracker::new(backend),
            bind_groups: ResourceTracker::new(backend),
            samplers: ResourceTracker::new(backend),
            compute_pipes: ResourceTracker::new(backend),
            render_pipes: ResourceTracker::new(backend),
            bundles: ResourceTracker::new(backend),
            query_sets: ResourceTracker::new(backend),
        }
    }

    /// Clear all the trackers.
    pub fn _clear(&mut self) {
        self.buffers.clear();
        self.textures.clear();
        self.views.clear();
        self.bind_groups.clear();
        self.samplers.clear();
        self.compute_pipes.clear();
        self.render_pipes.clear();
        self.bundles.clear();
        self.query_sets.clear();
    }

    /// Try to optimize the tracking representation.
    pub fn optimize(&mut self) {
        self.buffers.optimize();
        self.textures.optimize();
        self.views.optimize();
        self.bind_groups.optimize();
        self.samplers.optimize();
        self.compute_pipes.optimize();
        self.render_pipes.optimize();
        self.bundles.optimize();
        self.query_sets.optimize();
    }

    /// Merge only the stateful trackers of another instance by extending
    /// the usage. Returns a conflict if any.
    pub fn merge_extend_stateful(&mut self, other: &Self) -> Result<(), UsageConflict> {
        self.buffers.merge_extend(&other.buffers)?;
        self.textures.merge_extend(&other.textures)?;
        Ok(())
    }

    pub fn backend(&self) -> wgt::Backend {
        self.buffers.backend
    }
}

#[derive(Debug)]
pub(crate) struct StatefulTrackerSubset {
    pub buffers: ResourceTracker<BufferState>,
    pub textures: ResourceTracker<OldTextureState>,
}

impl StatefulTrackerSubset {
    /// Create an empty set.
    pub fn new(backend: wgt::Backend) -> Self {
        Self {
            buffers: ResourceTracker::new(backend),
            textures: ResourceTracker::new(backend),
        }
    }

    /// Clear all the trackers.
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.textures.clear();
    }

    /// Merge all the trackers of another tracker the usage.
    pub fn merge_extend(&mut self, other: &TrackerSet) -> Result<(), UsageConflict> {
        self.buffers.merge_extend(&other.buffers)?;
        self.textures.merge_extend(&other.textures)?;
        Ok(())
    }
}
