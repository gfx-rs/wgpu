use parking_lot::Mutex;
use wgt::Backend;

use crate::{id, Epoch, Index};
use std::{fmt::Debug, marker::PhantomData, sync::Arc};

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
pub(super) struct IdentityValues {
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

    count: usize,
}

impl IdentityValues {
    /// Allocate a fresh, never-before-seen id with the given `backend`.
    ///
    /// The backend is incorporated into the id, so that ids allocated with
    /// different `backend` values are always distinct.
    pub fn alloc<I: id::TypedId>(&mut self, backend: Backend) -> I {
        self.count += 1;
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
    pub fn release<I: id::TypedId>(&mut self, id: I) {
        let (index, epoch, _backend) = id.unzip();
        let pe = &mut self.epochs[index as usize];
        assert_eq!(*pe, epoch);
        // If the epoch reaches EOL, the index doesn't go
        // into the free list, will never be reused again.
        if epoch < id::EPOCH_MASK {
            *pe = epoch + 1;
            self.free.push(index);
            self.count -= 1;
        }
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

#[derive(Debug)]
pub struct IdentityManager<I: id::TypedId> {
    pub(super) values: Mutex<IdentityValues>,
    _phantom: PhantomData<I>,
}

impl<I: id::TypedId> IdentityManager<I> {
    pub fn process(&self, backend: Backend) -> I {
        self.values.lock().alloc(backend)
    }
    pub fn free(&self, id: I) {
        self.values.lock().release(id)
    }
}

impl<I: id::TypedId> IdentityManager<I> {
    pub fn new() -> Self {
        Self {
            values: Mutex::new(IdentityValues::default()),
            _phantom: PhantomData,
        }
    }
}

/// A type that can produce [`IdentityHandler`] filters for ids of type `I`.
///
/// See the module-level documentation for details.
pub trait IdentityHandlerFactory<I: id::TypedId> {
    type Input: Copy;
    /// Create an [`IdentityHandler<I>`] implementation that can
    /// transform proto-ids into ids of type `I`.
    ///
    /// [`IdentityHandler<I>`]: IdentityHandler
    fn spawn(&self) -> Option<Arc<IdentityManager<I>>>;

    fn input_to_id(id_in: Self::Input) -> I;
}

/// A global identity handler factory based on [`IdentityManager`].
///
/// Each of this type's `IdentityHandlerFactory<I>::spawn` methods
/// returns a `Mutex<IdentityManager<I>>`, which allocates fresh `I`
/// ids itself, and takes `()` as its proto-id type.
#[derive(Debug)]
pub struct IdentityManagerFactory;

impl<I: id::TypedId> IdentityHandlerFactory<I> for IdentityManagerFactory {
    type Input = ();

    fn spawn(&self) -> Option<Arc<IdentityManager<I>>> {
        Some(Arc::new(IdentityManager::new()))
    }

    fn input_to_id(_id_in: Self::Input) -> I {
        unreachable!("It should not be called")
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
}

impl GlobalIdentityHandlerFactory for IdentityManagerFactory {}

pub type Input<G, I> = <G as IdentityHandlerFactory<I>>::Input;

#[test]
fn test_epoch_end_of_life() {
    use id::TypedId as _;
    let man = IdentityManager::<id::BufferId>::new();
    man.values.lock().epochs.push(id::EPOCH_MASK);
    man.values.lock().free.push(0);
    let id1 = man.values.lock().alloc::<id::BufferId>(Backend::Empty);
    assert_eq!(id1.unzip().0, 0);
    man.values.lock().release(id1);
    let id2 = man.values.lock().alloc::<id::BufferId>(Backend::Empty);
    // confirm that the index 0 is no longer re-used
    assert_eq!(id2.unzip().0, 1);
}
