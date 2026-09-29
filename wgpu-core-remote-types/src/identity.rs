use alloc::vec::Vec;
use core::cell::RefCell;
use core::{fmt::Debug, marker::PhantomData};

use crate::{id::markers, id::Id, id::Marker};
use crate::{Epoch, Index};

/// A simple structure to allocate [`Id`] identifiers.
///
/// Calling [`alloc`] returns a fresh, never-before-seen id. Calling [`release`]
/// marks an id as dead; it will never be returned again by `alloc`.
///
/// `IdentityValues` returns `Id`s whose index values are suitable for use as
/// indices into a `Vec<T>` that holds those ids' referents:
///
/// - Every live id has a distinct index value. Every live id's index
///   selects a distinct element in the vector.
///
/// - `IdentityValues` prefers low index numbers. If you size your vector to
///   accommodate the indices produced here, the vector's length will reflect
///   the highwater mark of actual occupancy.
///
/// - `IdentityValues` reuses the index values of freed ids before returning
///   ids with new index values. Freed vector entries get reused.
///
/// - The non-reuse property is achieved by storing an `epoch` alongside the
///   index in an `Id`. Index values are reused, but only with a different
///   epoch.
///
/// [`Id`]: crate::id::Id
/// [`alloc`]: IdentityValues::alloc
/// [`release`]: IdentityValues::release
#[derive(Debug)]
pub(super) struct IdentityValues {
    free: Vec<(Index, Epoch)>,
    next_index: Index,
    count: usize,
}

impl IdentityValues {
    /// Allocate a fresh, never-before-seen ID.
    pub fn alloc<T: Marker>(&mut self) -> Id<T> {
        self.count += 1;
        match self.free.pop() {
            Some((index, epoch)) => Id::zip(index, epoch + 1),
            None => {
                let index = self.next_index;
                self.next_index += 1;
                let epoch = 1;
                Id::zip(index, epoch)
            }
        }
    }

    /// Free `id` and/or decrement the count of used IDs.
    ///
    /// Freed IDs will never be returned from `alloc` again.
    pub fn release<T: Marker>(&mut self, id: Id<T>) {
        let (index, epoch) = id.unzip();
        self.free.push((index, epoch));
        self.count -= 1;
    }
}

#[derive(Debug)]
pub struct IdentityManager<T: Marker> {
    pub(super) values: RefCell<IdentityValues>,
    _phantom: PhantomData<T>,
}

impl<T: Marker> IdentityManager<T> {
    pub fn process(&self) -> Id<T> {
        self.values.borrow_mut().alloc()
    }

    pub fn free(&self, id: Id<T>) {
        self.values.borrow_mut().release(id)
    }
}

impl<T: Marker> IdentityManager<T> {
    pub fn new() -> Self {
        Self {
            values: RefCell::new(IdentityValues {
                free: Vec::new(),
                next_index: 0,
                count: 0,
            }),
            _phantom: PhantomData,
        }
    }
}

impl<T: Marker> Default for IdentityManager<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A collection of identity managers for all resource types.
///
/// This is to be used in content process for ID generation.
/// IDs are then sent to the GPU process for resource creation in `Hub`.
#[derive(Debug, Default)]
pub struct IdentityHub {
    pub adapters: IdentityManager<markers::Adapter>,
    pub devices: IdentityManager<markers::Device>,
    pub queues: IdentityManager<markers::Queue>,
    pub pipeline_layouts: IdentityManager<markers::PipelineLayout>,
    pub shader_modules: IdentityManager<markers::ShaderModule>,
    pub bind_group_layouts: IdentityManager<markers::BindGroupLayout>,
    pub bind_groups: IdentityManager<markers::BindGroup>,
    pub command_encoders: IdentityManager<markers::CommandEncoder>,
    pub command_buffers: IdentityManager<markers::CommandBuffer>,
    pub render_bundles: IdentityManager<markers::RenderBundle>,
    pub render_pipelines: IdentityManager<markers::RenderPipeline>,
    pub compute_pipelines: IdentityManager<markers::ComputePipeline>,
    pub pipeline_caches: IdentityManager<markers::PipelineCache>,
    pub query_sets: IdentityManager<markers::QuerySet>,
    pub buffers: IdentityManager<markers::Buffer>,
    pub textures: IdentityManager<markers::Texture>,
    pub texture_views: IdentityManager<markers::TextureView>,
    pub external_textures: IdentityManager<markers::ExternalTexture>,
    pub samplers: IdentityManager<markers::Sampler>,
    pub render_passes: IdentityManager<markers::RenderPassEncoder>,
    pub compute_passes: IdentityManager<markers::ComputePassEncoder>,
    pub render_bundle_encoders: IdentityManager<markers::RenderBundleEncoder>,
}

impl IdentityHub {
    pub fn new() -> Self {
        Self::default()
    }
}

#[test]
fn test_epoch_end_of_life() {
    let man = IdentityManager::<markers::Buffer>::new();
    let id1 = man.process();
    assert_eq!(id1.unzip(), (0, 1));
    man.free(id1);
    let id2 = man.process();
    // confirm that the epoch 1 is no longer re-used
    assert_eq!(id2.unzip(), (0, 2));
}
