/*! Resource State and Lifetime Trackers

These structures are responsible for keeping track of resource state,
generating barriers where needed, and making sure resources are kept
alive until the trackers die.

## General Architecture

Tracking is some of the hottest code in the entire codebase, so the trackers
are designed to be as cache efficient as possible. They store resource state
in flat vectors, storing metadata SOA style, one vector per type of metadata.

A lot of the tracker code is deeply unsafe, using unchecked accesses all over
to make performance as good as possible. However, for all unsafe accesses, there
is a corresponding debug assert the checks if that access is valid. This helps
get bugs caught fast, while still letting users not need to pay for the bounds
checks.

In wgpu, resource IDs are allocated and re-used, so will always be as low
as reasonably possible. This allows us to use the ID as an index into an array.

## Statefulness

There are two main types of trackers, stateful and stateless.

Stateful trackers are for buffers and textures. They both have
resource state attached to them which needs to be used to generate
automatic synchronization. Because of the different requirements of
buffers and textures, they have two separate tracking structures.

Stateless trackers only store metadata and own the given resource.

## Use Case

Within each type of tracker, the trackers are further split into 3 different
use cases, Bind Group, Usage Scope, and a full Tracker.

Bind Group trackers are just a list of different resources, their refcount,
and how they are used. Textures are used via a selector and a usage type.
Buffers by just a usage type. Stateless resources don't have a usage type.

Usage Scope trackers are only for stateful resources. These trackers represent
a single [`UsageScope`] in the spec. When a use is added to a usage scope,
it is merged with all other uses of that resource in that scope. If there
is a usage conflict, merging will fail and an error will be reported.

Full trackers represent a before and after state of a resource. These
are used for tracking on the device and on command buffers. The before
state represents the state the resource is first used as in the command buffer,
the after state is the state the command buffer leaves the resource in.
These double ended buffers can then be used to generate the needed transitions
between command buffers.

## Dense Datastructure with Sparse Data

This tracking system is based on having completely dense data, but trackers do
not always contain every resource. Some resources (or even most resources) go
unused in any given command buffer. So to help speed up the process of iterating
through possibly thousands of resources, we use a bit vector to represent if
a resource is in the buffer or not. This allows us extremely efficient memory
utilization, as well as being able to bail out of whole blocks of 32-64 resources
with a single usize comparison with zero. In practice this means that merging
partially resident buffers is extremely quick.

The main advantage of this dense datastructure is that we can do merging
of trackers in an extremely efficient fashion that results in us doing linear
scans down a couple of buffers. CPUs and their caches absolutely eat this up.

## Stateful Resource Operations

All operations on stateful trackers boil down to one of four operations:
- `insert(tracker, new_state)` adds a resource with a given state to the tracker
  for the first time.
- `merge(tracker, new_state)` merges this new state with the previous state, checking
  for usage conflicts.
- `barrier(tracker, new_state)` compares the given state to the existing state and
  generates the needed barriers.
- `update(tracker, new_state)` takes the given new state and overrides the old state.

This allows us to compose the operations to form the various kinds of tracker merges
that need to happen in the codebase. For each resource in the given merger, the following
operation applies:

```text
UsageScope <- Resource = insert(scope, usage) OR merge(scope, usage)
UsageScope <- UsageScope = insert(scope, scope) OR merge(scope, scope)
CommandBuffer <- UsageScope = insert(buffer.start, buffer.end, scope)
                              OR barrier(buffer.end, scope) + update(buffer.end, scope)
Device <- CommandBuffer = insert(device.start, device.end, buffer.start, buffer.end)
                          OR barrier(device.end, buffer.start) + update(device.end, buffer.end)
```

[`UsageScope`]: https://gpuweb.github.io/gpuweb/#programming-model-synchronization
*/

mod buffer;
mod metadata;
mod range;
mod stateless;
mod texture;

use crate::{
    binding_model, command, conv, hub,
    id::{self, TypedId},
    pipeline, resource,
};

use std::{fmt, ops};
use thiserror::Error;

pub(crate) use buffer::{BufferBindGroupState, BufferTracker, BufferUsageScope};
use metadata::{ResourceMetadata, ResourceMetadataProvider};
pub(crate) use stateless::{StatelessBindGroupSate, StatelessTracker};
pub(crate) use texture::{
    TextureBindGroupState, TextureSelector, TextureTracker, TextureUsageScope,
};
use wgt::strict_assert_ne;

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
        let texture = tex.inner.as_raw().expect("Texture is destroyed");

        // These showing up in a barrier is always a bug
        strict_assert_ne!(self.usage.start, hal::TextureUses::UNKNOWN);
        strict_assert_ne!(self.usage.end, hal::TextureUses::UNKNOWN);

        let mip_count = self.selector.mips.end - self.selector.mips.start;
        strict_assert_ne!(mip_count, 0);
        let layer_count = self.selector.layers.end - self.selector.layers.start;
        strict_assert_ne!(layer_count, 0);

        hal::TextureBarrier {
            texture,
            range: wgt::ImageSubresourceRange {
                aspect: wgt::TextureAspect::All,
                base_mip_level: self.selector.mips.start,
                mip_level_count: Some(mip_count),
                base_array_layer: self.selector.layers.start,
                array_layer_count: Some(layer_count),
            },
            usage: self.usage,
        }
    }
}

/// The uses that a resource or subresource can be in.
pub(crate) trait ResourceUses:
    fmt::Debug + ops::BitAnd<Output = Self> + ops::BitOr<Output = Self> + PartialEq + Sized + Copy
{
    /// All flags that are exclusive.
    const EXCLUSIVE: Self;

    /// The relevant resource ID type.
    type Id: Copy + fmt::Debug + TypedId;
    /// The selector used by this resource.
    type Selector: fmt::Debug;

    /// Turn the resource into a pile of bits.
    fn bits(self) -> u16;
    /// Returns true if the all the uses are ordered.
    fn all_ordered(self) -> bool;
    /// Returns true if any of the uses are exclusive.
    fn any_exclusive(self) -> bool;
}

/// Returns true if the given states violates the usage scope rule
/// of any(inclusive) XOR one(exclusive)
fn invalid_resource_state<T: ResourceUses>(state: T) -> bool {
    // Is power of two also means "is one bit set". We check for this as if
    // we're in any exclusive state, we must only be in a single state.
    state.any_exclusive() && !conv::is_power_of_two_u16(state.bits())
}

/// Returns true if the transition from one state to another does not require
/// a barrier.
fn skip_barrier<T: ResourceUses>(old_state: T, new_state: T) -> bool {
    // If the state didn't change and all the usages are ordered, the hardware
    // will guarentee the order of accesses, so we do not need to issue a barrier at all
    old_state == new_state && old_state.all_ordered()
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum UsageConflict {
    #[error("Attempted to use invalid buffer")]
    BufferInvalid { id: id::BufferId },
    #[error("Attempted to use invalid texture")]
    TextureInvalid { id: id::TextureId },
    #[error("Attempted to use buffer with {invalid_use}.")]
    Buffer {
        id: id::BufferId,
        invalid_use: InvalidUse<hal::BufferUses>,
    },
    #[error("Attempted to use a texture (mips {mip_levels:?} layers {array_layers:?}) with {invalid_use}.")]
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
            id,
            mip_levels: selector.mips,
            array_layers: selector.layers,
            invalid_use: InvalidUse {
                current_state,
                new_state,
            },
        }
    }
}

impl crate::error::PrettyError for UsageConflict {
    fn fmt_pretty(&self, fmt: &mut crate::error::ErrorFormatter) {
        fmt.error(self);
        match *self {
            Self::BufferInvalid { id } => {
                fmt.buffer_label(&id);
            }
            Self::TextureInvalid { id } => {
                fmt.texture_label(&id);
            }
            Self::Buffer { id, .. } => {
                fmt.buffer_label(&id);
            }
            Self::Texture { id, .. } => {
                fmt.texture_label(&id);
            }
        }
    }
}

/// Pretty print helper that shows helpful descriptions of a conflicting usage.
#[derive(Clone, Debug, Eq, PartialEq)]
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

        // The text starts with "tried to use X resource with {self}"
        write!(
            f,
            "conflicting usages. Current usage {current:?} and new usage {new:?}. \
            {exclusive:?} is an exclusive usage and cannot be used with any other \
            usages within the usage scope (renderpass or compute dispatch)"
        )
    }
}

/// All the usages that a bind group contains. The uses are not deduplicated in any way
/// and may include conflicting uses. This is fully compliant by the WebGPU spec.
///
/// All bind group states are sorted by their ID so that when adding to a tracker,
/// they are added in the most efficient order possible (assending order).
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

    /// Optimize the bind group states by sorting them by ID.
    ///
    /// When this list of states is merged into a tracker, the memory
    /// accesses will be in a constant assending order.
    pub fn optimize(&mut self) {
        self.buffers.optimize();
        self.textures.optimize();
        self.views.optimize();
        self.samplers.optimize();
    }
}

/// This is a render bundle specific usage scope. It includes stateless resources
/// that are not normally included in a usage scope, but are used by render bundles
/// and need to be owned by the render bundles.
pub(crate) struct RenderBundleScope<A: hub::HalApi> {
    pub buffers: BufferUsageScope<A>,
    pub textures: TextureUsageScope<A>,
    // Don't need to track views and samplers, they are never used directly, only by bind groups.
    pub bind_groups: StatelessTracker<A, binding_model::BindGroup<A>, id::BindGroupId>,
    pub render_pipelines: StatelessTracker<A, pipeline::RenderPipeline<A>, id::RenderPipelineId>,
    pub query_sets: StatelessTracker<A, resource::QuerySet<A>, id::QuerySetId>,
}

impl<A: hub::HalApi> RenderBundleScope<A> {
    /// Create the render bundle scope and pull the maximum IDs from the hubs.
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

    /// Merge the inner contents of a bind group into the render bundle tracker.
    ///
    /// Only stateful things are merged in here, all other resources are owned
    /// indirectly by the bind group.
    ///
    /// # Safety
    ///
    /// The maximum ID given by each bind group resource must be less than the
    /// length of the storage given at the call to `new`.
    pub unsafe fn merge_bind_group(
        &mut self,
        textures: &hub::Storage<resource::Texture<A>, id::TextureId>,
        bind_group: &BindGroupStates<A>,
    ) -> Result<(), UsageConflict> {
        unsafe { self.buffers.merge_bind_group(&bind_group.buffers)? };
        unsafe {
            self.textures
                .merge_bind_group(textures, &bind_group.textures)?
        };

        Ok(())
    }
}

/// A usage scope tracker. Only needs to store stateful resources as stateless
/// resources cannot possibly have a usage conflict.
#[derive(Debug)]
pub(crate) struct UsageScope<A: hub::HalApi> {
    pub buffers: BufferUsageScope<A>,
    pub textures: TextureUsageScope<A>,
}

impl<A: hub::HalApi> UsageScope<A> {
    /// Create the render bundle scope and pull the maximum IDs from the hubs.
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

    /// Merge the inner contents of a bind group into the usage scope.
    ///
    /// Only stateful things are merged in here, all other resources are owned
    /// indirectly by the bind group.
    ///
    /// # Safety
    ///
    /// The maximum ID given by each bind group resource must be less than the
    /// length of the storage given at the call to `new`.
    pub unsafe fn merge_bind_group(
        &mut self,
        textures: &hub::Storage<resource::Texture<A>, id::TextureId>,
        bind_group: &BindGroupStates<A>,
    ) -> Result<(), UsageConflict> {
        unsafe {
            self.buffers.merge_bind_group(&bind_group.buffers)?;
            self.textures
                .merge_bind_group(textures, &bind_group.textures)?;
        }

        Ok(())
    }

    /// Merge the inner contents of a bind group into the usage scope.
    ///
    /// Only stateful things are merged in here, all other resources are owned
    /// indirectly by a bind group or are merged directly into the command buffer tracker.
    ///
    /// # Safety
    ///
    /// The maximum ID given by each bind group resource must be less than the
    /// length of the storage given at the call to `new`.
    pub unsafe fn merge_render_bundle(
        &mut self,
        textures: &hub::Storage<resource::Texture<A>, id::TextureId>,
        render_bundle: &RenderBundleScope<A>,
    ) -> Result<(), UsageConflict> {
        self.buffers.merge_usage_scope(&render_bundle.buffers)?;
        self.textures
            .merge_usage_scope(textures, &render_bundle.textures)?;

        Ok(())
    }
}

/// A full double sided tracker used by CommandBuffers and the Device.
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

    /// Pull the maximum IDs from the hubs.
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

    /// Iterates through all resources in the given bind group and adopts
    /// the state given for those resources in the UsageScope. It also
    /// removes all touched resources from the usage scope.
    ///
    /// If a transition is needed to get the resources into the needed
    /// state, those transitions are stored within the tracker. A
    /// subsequent call to [`BufferTracker::drain`] or
    /// [`TextureTracker::drain`] is needed to get those transitions.
    ///
    /// This is a really funky method used by Compute Passes to generate
    /// barriers after a call to dispatch without needing to iterate
    /// over all elements in the usage scope. We use each the
    /// bind group as a source of which IDs to look at. The bind groups
    /// must have first been added to the usage scope.
    ///
    /// Only stateful things are merged in here, all other resources are owned
    /// indirectly by the bind group.
    ///
    /// # Safety
    ///
    /// The maximum ID given by each bind group resource must be less than the
    /// value given to `set_size`
    pub unsafe fn set_and_remove_from_usage_scope_sparse(
        &mut self,
        textures: &hub::Storage<resource::Texture<A>, id::TextureId>,
        scope: &mut UsageScope<A>,
        bind_group: &BindGroupStates<A>,
    ) {
        unsafe {
            self.buffers.set_and_remove_from_usage_scope_sparse(
                &mut scope.buffers,
                bind_group.buffers.used(),
            )
        };
        unsafe {
            self.textures.set_and_remove_from_usage_scope_sparse(
                textures,
                &mut scope.textures,
                &bind_group.textures,
            )
        };
    }

    /// Tracks the stateless resources from the given renderbundle. It is expected
    /// that the stateful resources will get merged into a usage scope first.
    ///
    /// # Safety
    ///
    /// The maximum ID given by each bind group resource must be less than the
    /// value given to `set_size`
    pub unsafe fn add_from_render_bundle(
        &mut self,
        render_bundle: &RenderBundleScope<A>,
    ) -> Result<(), UsageConflict> {
        self.bind_groups
            .add_from_tracker(&render_bundle.bind_groups);
        self.render_pipelines
            .add_from_tracker(&render_bundle.render_pipelines);
        self.query_sets.add_from_tracker(&render_bundle.query_sets);

        Ok(())
    }
}
