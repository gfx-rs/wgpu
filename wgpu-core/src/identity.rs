use parking_lot::Mutex;
use wgt::Backend;

use crate::{id, Epoch, Index};
use std::fmt::Debug;

/// A simple structure to allocate [`Id`] identifiers.
///
/// Calling [`alloc`] returns a fresh, never-before-seen id. Calling [`free`]
/// marks an id as dead; it will never be returned again by `alloc`.
///
/// Use `IdentityManager::default` to construct new instances.
///
/// `IdentityManager` returns `Id`s whose index values are suitable for use as
/// indices into a `Storage<T>` that holds those ids' referents:
///
/// - Every live id has a distinct index value. Each live id's index selects a
///   distinct element in the vector.
///
/// - `IdentityManager` prefers low index numbers. If you size your vector to
///   accommodate the indices produced here, the vector's length will reflect
///   the highwater mark of actual occupancy.
///
/// - `IdentityManager` reuses the index values of freed ids before returning
///   ids with new index values. Freed vector entries get reused.
///
/// See the module-level documentation for an overview of how this
/// fits together.
///
/// [`Id`]: crate::id::Id
/// [`Backend`]: wgt::Backend;
/// [`alloc`]: IdentityManager::alloc
/// [`free`]: IdentityManager::free
#[derive(Debug, Default)]
pub struct IdentityManager {
    /// Available index values. If empty, then `epochs.len()` is the next index
    /// to allocate.
    free: Vec<Index>,

    /// The next or currently-live epoch value associated with each `Id` index.
    ///
    /// If there is a live id with index `i`, then `epochs[i]` is its epoch; any
    /// id with the same index but an older epoch is dead.
    ///
    /// If index `i` is currently unused, `epochs[i]` is the epoch to use in its
    /// next `Id`.
    epochs: Vec<Epoch>,
}

impl IdentityManager {
    /// Allocate a fresh, never-before-seen id with the given `backend`.
    ///
    /// The backend is incorporated into the id, so that ids allocated with
    /// different `backend` values are always distinct.
    pub fn alloc<I: id::TypedId>(&mut self, backend: Backend) -> I {
        match self.free.pop() {
            Some(index) => I::zip(index, self.epochs[index as usize], backend),
            None => {
                let epoch = 1;
                let id = I::zip(self.epochs.len() as Index, epoch, backend);
                self.epochs.push(epoch);
                id
            }
        }
    }

    /// Free `id`. It will never be returned from `alloc` again.
    pub fn free<I: id::TypedId + Debug>(&mut self, id: I) {
        let (index, epoch, _backend) = id.unzip();
        let pe = &mut self.epochs[index as usize];
        assert_eq!(*pe, epoch);
        // If the epoch reaches EOL, the index doesn't go
        // into the free list, will never be reused again.
        if epoch < id::EPOCH_MASK {
            *pe = epoch + 1;
            self.free.push(index);
        }
    }
}

/// A type that can build true ids from proto-ids, and free true ids.
///
/// For some implementations, the true id is based on the proto-id.
/// The caller is responsible for providing well-allocated proto-ids.
///
/// For other implementations, the proto-id carries no information
/// (it's `()`, say), and this `IdentityHandler` type takes care of
/// allocating a fresh true id.
///
/// See the module-level documentation for details.
pub trait IdentityHandler<I>: Debug {
    /// The type of proto-id consumed by this filter, to produce a true id.
    type Input: Clone + Debug;

    /// Given a proto-id value `id`, return a true id for `backend`.
    fn process(&self, id: Self::Input, backend: Backend) -> I;

    /// Free the true id `id`.
    fn free(&self, id: I);
}

impl<I: id::TypedId + Debug> IdentityHandler<I> for Mutex<IdentityManager> {
    type Input = ();
    fn process(&self, _id: Self::Input, backend: Backend) -> I {
        self.lock().alloc(backend)
    }
    fn free(&self, id: I) {
        self.lock().free(id)
    }
}

/// A type that can produce [`IdentityHandler`] filters for ids of type `I`.
///
/// See the module-level documentation for details.
pub trait IdentityHandlerFactory<I> {
    /// The type of filter this factory constructs.
    ///
    /// "Filter" and "handler" seem to both mean the same thing here:
    /// something that can produce true ids from proto-ids.
    type Filter: IdentityHandler<I>;

    /// Create an [`IdentityHandler<I>`] implementation that can
    /// transform proto-ids into ids of type `I`.
    ///
    /// [`IdentityHandler<I>`]: IdentityHandler
    fn spawn(&self) -> Self::Filter;
}

/// A global identity handler factory based on [`IdentityManager`].
///
/// Each of this type's `IdentityHandlerFactory<I>::spawn` methods
/// returns a `Mutex<IdentityManager<I>>`, which allocates fresh `I`
/// ids itself, and takes `()` as its proto-id type.
#[derive(Debug)]
pub struct IdentityManagerFactory;

impl<I: id::TypedId + Debug> IdentityHandlerFactory<I> for IdentityManagerFactory {
    type Filter = Mutex<IdentityManager>;
    fn spawn(&self) -> Self::Filter {
        Mutex::new(IdentityManager::default())
    }
}

/// A factory that can build [`IdentityHandler`]s for all resource
/// types.
pub trait GlobalIdentityHandlerFactory:
    IdentityHandlerFactory<id::AdapterId>
    + IdentityHandlerFactory<id::DeviceId>
    + IdentityHandlerFactory<id::PipelineLayoutId>
    + IdentityHandlerFactory<id::ShaderModuleId>
    + IdentityHandlerFactory<id::BindGroupLayoutId>
    + IdentityHandlerFactory<id::BindGroupId>
    + IdentityHandlerFactory<id::CommandBufferId>
    + IdentityHandlerFactory<id::RenderBundleId>
    + IdentityHandlerFactory<id::RenderPipelineId>
    + IdentityHandlerFactory<id::ComputePipelineId>
    + IdentityHandlerFactory<id::QuerySetId>
    + IdentityHandlerFactory<id::BufferId>
    + IdentityHandlerFactory<id::StagingBufferId>
    + IdentityHandlerFactory<id::TextureId>
    + IdentityHandlerFactory<id::TextureViewId>
    + IdentityHandlerFactory<id::SamplerId>
    + IdentityHandlerFactory<id::SurfaceId>
{
    fn ids_are_generated_in_wgpu() -> bool;
}

impl GlobalIdentityHandlerFactory for IdentityManagerFactory {
    fn ids_are_generated_in_wgpu() -> bool {
        true
    }
}

pub type Input<G, I> = <<G as IdentityHandlerFactory<I>>::Filter as IdentityHandler<I>>::Input;

#[test]
fn test_epoch_end_of_life() {
    use id::TypedId as _;
    let mut man = IdentityManager::default();
    man.epochs.push(id::EPOCH_MASK);
    man.free.push(0);
    let id1 = man.alloc::<id::BufferId>(Backend::Empty);
    assert_eq!(id1.unzip().0, 0);
    man.free(id1);
    let id2 = man.alloc::<id::BufferId>(Backend::Empty);
    // confirm that the index 0 is no longer re-used
    assert_eq!(id2.unzip().0, 1);
}
