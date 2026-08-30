use alloc::string::String;

use crate::*;

/// Handle to a resource table.
///
/// A `ResourceTable` is a device-timeline-mutable, sparse array of resource bindings
/// (currently sampled/depth textures; see [`Features::EXPERIMENTAL_HETEROGENEOUS_RESOURCE_TABLE`]
/// for planned samplers, storage textures, and buffers). It is bound as pass-level encoder state
/// (see [`ComputePass::set_resource_table`] and [`RenderPassDescriptor::resource_table`]) and
/// accessed from WGSL shaders via `enable resource_table;` and `getResource<T>(index)`.
///
/// It can be created with [`Device::create_resource_table`].
///
/// This mirrors the [`GPUResourceTable`] proposal for WebGPU; it is a native-only, experimental
/// feature gated by [`Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE`].
///
/// [`GPUResourceTable`]: https://github.com/gpuweb/gpuweb/issues/5372
#[derive(Debug, Clone)]
pub struct ResourceTable {
    pub(crate) inner: dispatch::DispatchResourceTable,
    pub(crate) size: u32,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(ResourceTable: Send, Sync);

crate::cmp::impl_eq_ord_hash_proxy!(ResourceTable => .inner);

impl ResourceTable {
    /// Destroy the associated native resources as soon as possible.
    pub fn destroy(&self) {
        self.inner.destroy();
    }

    /// Returns the number of slots in this resource table.
    ///
    /// This is always equal to the `size` that was specified when creating the table.
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Bind `texture_view` into `slot`, replacing whatever was previously bound there.
    ///
    /// This is a device-timeline mutation of the table's contents. In the current
    /// (M0) implementation only sampled/depth texture views are accepted; support
    /// for samplers, storage textures, and buffers arrives with
    /// [`Features::EXPERIMENTAL_HETEROGENEOUS_RESOURCE_TABLE`].
    ///
    /// # Errors
    ///
    /// Returns [`ResourceTableError::SlotInUse`] if a submission that may still
    /// dynamically read the slot has not yet completed (retry once it has),
    /// [`ResourceTableError::SlotOutOfBounds`] if `slot >= self.size()`, or
    /// [`ResourceTableError::Other`] for any other validation failure.
    pub fn update(&self, slot: u32, texture_view: &TextureView) -> Result<(), ResourceTableError> {
        self.inner.update(slot, &texture_view.inner)
    }

    /// Bind `texture_view` into the lowest currently-empty slot and return its index.
    ///
    /// See [`update`](Self::update) for the accepted resource kinds and the
    /// device-timeline semantics.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceTableError::NoAvailableSlot`] if every slot is occupied,
    /// [`ResourceTableError::SlotInUse`] if the chosen empty slot is still gated by
    /// an in-flight submission, or [`ResourceTableError::Other`] for any other
    /// validation failure.
    pub fn insert_binding(&self, texture_view: &TextureView) -> Result<u32, ResourceTableError> {
        self.inner.insert_binding(&texture_view.inner)
    }

    /// Clear the binding in `slot`.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceTableError::SlotInUse`] if a submission that may still
    /// dynamically read the slot has not yet completed,
    /// [`ResourceTableError::SlotOutOfBounds`] if `slot >= self.size()`, or
    /// [`ResourceTableError::Other`] for any other validation failure.
    pub fn remove_binding(&self, slot: u32) -> Result<(), ResourceTableError> {
        self.inner.remove_binding(slot)
    }

    /// Returns the inner hal [`A::ResourceTable`].
    ///
    /// Returns a guard which dereferences to the inner hal resource table.
    ///
    /// # Errors
    ///
    /// This method will return `None` if:
    /// - The resource table is not from the backend `A`.
    /// - The resource table is from the `webgpu` or `custom` backend.
    ///
    /// # Safety
    ///
    /// - The returned resource must not be destroyed unless the guard
    ///   is the last reference to it and it is not in use by the GPU.
    ///   The guard and handle may be dropped at any time however.
    /// - All the safety requirements of wgpu-hal must be upheld.
    ///
    /// [`A::ResourceTable`]: hal::Api::ResourceTable
    #[cfg(wgpu_core)]
    pub unsafe fn as_hal<A: hal::Api>(
        &self,
    ) -> Option<impl core::ops::Deref<Target = A::ResourceTable> + WasmNotSendSync> {
        let table = self.inner.as_core_opt()?;
        unsafe { table.context.resource_table_as_hal::<A>(table) }
    }

    #[cfg(custom)]
    /// Returns custom implementation of ResourceTable (if custom backend and is internally T)
    pub fn as_custom<T: custom::ResourceTableInterface>(&self) -> Option<&T> {
        self.inner.as_custom()
    }
}

/// Error produced by the [`ResourceTable`] slot-update methods
/// ([`ResourceTable::update`], [`ResourceTable::insert_binding`], and
/// [`ResourceTable::remove_binding`]).
///
/// Unlike most `wgpu` validation errors — which surface asynchronously through
/// error scopes and [`Device::on_uncaptured_error`] — these device-timeline
/// mutations report failures synchronously through `Result`, mirroring the
/// `GPUResourceTable` proposal. This lets callers react to the recoverable
/// [`SlotInUse`](ResourceTableError::SlotInUse) condition at the call site.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ResourceTableError {
    /// The slot cannot be rewritten yet: a submission that may still dynamically
    /// read it has not completed. Retry once the submission at index
    /// `available_after` (or later) has finished — for example after a
    /// [`Device::poll`] confirms its completion.
    SlotInUse {
        /// The submission index that must complete before the slot becomes free.
        available_after: SubmissionIndex,
    },
    /// `slot` is greater than or equal to the table's [`size`](ResourceTable::size).
    SlotOutOfBounds {
        /// The offending slot index.
        slot: u32,
        /// The table's size.
        size: u32,
    },
    /// [`ResourceTable::insert_binding`] found no empty slot: the table is full.
    NoAvailableSlot,
    /// Any other failure: an invalid or destroyed resource, a missing feature or
    /// texture usage, or a device error. Rendered as a human-readable message.
    Other(String),
}

impl core::fmt::Display for ResourceTableError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SlotInUse { available_after } => write!(
                f,
                "resource table slot cannot be updated yet: it may still be used by a submission \
                 that completes at index {}",
                available_after.index
            ),
            Self::SlotOutOfBounds { slot, size } => write!(
                f,
                "resource table slot {slot} is out of bounds for a table with {size} slots"
            ),
            Self::NoAvailableSlot => {
                f.write_str("resource table has no available slot: all slots are occupied")
            }
            Self::Other(message) => f.write_str(message),
        }
    }
}

impl core::error::Error for ResourceTableError {}

/// Describes a [`ResourceTable`].
///
/// For use with [`Device::create_resource_table`].
pub type ResourceTableDescriptor<'a> = wgt::ResourceTableDescriptor<Label<'a>>;
#[cfg(send_sync)]
static_assertions::assert_impl_all!(ResourceTableDescriptor<'_>: Send, Sync);
