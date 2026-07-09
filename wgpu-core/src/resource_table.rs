//! The wgpu-core [`ResourceTable`] object: the device-timeline-mutable, sparse
//! descriptor array of the WebGPU bindless proposal (see `docs/bindless.md` and
//! `plans/resource-table.md`).
//!
//! This module implements work item 0.6 of the resource-table feature: the
//! wgpu-core object itself — creation, destruction, registry/lifetime plumbing,
//! the slot-reuse gate (Invariant 2), and the error types. Encoder integration
//! (0.7), queue/submit machinery (0.8), and the public `wgpu` API (0.11) are
//! implemented separately and build on the shapes defined here.

use alloc::{borrow::ToOwned as _, boxed::Box, string::String, sync::Arc, vec::Vec};
use core::sync::atomic::Ordering;
use core::{fmt, mem::ManuallyDrop};

use smallvec::SmallVec;
use thiserror::Error;
use wgt::error::{ErrorType, WebGpuError};

use crate::{
    api_log,
    device::{queue::TempResource, Device, DeviceError, MissingFeatures},
    lock::{rank, Mutex},
    resource::{
        DestroyedResourceError, InvalidResourceError, Labeled as _, MissingTextureUsageError,
        ParentDevice as _, RawResourceAccess, ResourceState, Texture, TextureView, Trackable as _,
        TrackingData,
    },
    resource_log,
    snatch::{SnatchGuard, Snatchable},
    track::TrackerIndex,
    FastHashMap, Label, LabelHelpers as _, SubmissionIndex,
};

/// The maximum number of slots a [`ResourceTable`] may have.
///
/// Fixed by the bindless proposal (`docs/bindless.md`): `size` must be
/// `<= 65536`.
pub const MAX_RESOURCE_TABLE_SIZE: u32 = 65536;

/// cbindgen:ignore
pub type ResourceTableDescriptor<'a> = wgt::ResourceTableDescriptor<Label<'a>>;

/// A single slot of a [`ResourceTable`].
///
/// M0 stores only the slot-reuse gate. Later work items grow this to the full
/// `Slot` model from `plans/resource-table.md`
/// (`{ resource: Option<Arc<..>>, flavor: TypeClass, available_after }`),
/// populated by `insert_binding`/`update`.
#[derive(Debug)]
#[allow(
    dead_code,
    reason = "`available_after` is only read by the slot-reuse gate, which is consumed by work items 0.7/0.8"
)]
struct Slot {
    /// The submission index after which this slot may be rewritten with a CPU
    /// descriptor write (Invariant 2 in `plans/resource-table.md`).
    ///
    /// It records the most recent submission that could *dynamically* reach the
    /// slot through its table. A rewrite is legal only once the device's
    /// completed submission index reaches this value. `0` means the slot has
    /// never been used by any submission and is always available.
    ///
    /// In M0 nothing sets this — the setter ([`ResourceTableSlots::mark_in_use`])
    /// exists for work item 0.8 (queue submit) to call.
    available_after: hal::AtomicFenceValue,
}

/// The slot storage of a [`ResourceTable`], plus the slot-reuse gate logic.
///
/// This is deliberately independent of [`Device`] so that the gate is unit
/// testable without a queue: the "completed submission index" is passed in
/// rather than read from the device (see [`ResourceTable::check_slot_available`]
/// for the wiring used in production).
#[derive(Debug)]
pub(crate) struct ResourceTableSlots {
    slots: Box<[Slot]>,
}

#[allow(
    dead_code,
    reason = "the slot-reuse gate is scaffolding consumed by work items 0.7 (update) and 0.8 (submit)"
)]
impl ResourceTableSlots {
    fn new(size: u32) -> Self {
        let slots = (0..size)
            .map(|_| Slot {
                available_after: hal::AtomicFenceValue::new(0),
            })
            .collect();
        Self { slots }
    }

    fn len(&self) -> u32 {
        self.slots.len() as u32
    }

    /// Check whether `slot` may be rewritten, given the most recently
    /// `completed` submission index.
    ///
    /// Returns [`UpdateResourceTableError::SlotOutOfBounds`] if `slot` is not in
    /// the table, or [`UpdateResourceTableError::SlotInUse`] if an in-flight
    /// submission may still dynamically reach it (Invariant 2).
    fn check_available(
        &self,
        slot: u32,
        completed: SubmissionIndex,
    ) -> Result<(), UpdateResourceTableError> {
        let slot_state =
            self.slots
                .get(slot as usize)
                .ok_or(UpdateResourceTableError::SlotOutOfBounds {
                    slot,
                    size: self.len(),
                })?;
        let available_after = slot_state.available_after.load(Ordering::Acquire);
        if available_after > completed {
            return Err(UpdateResourceTableError::SlotInUse { available_after });
        }
        Ok(())
    }

    /// Record that submission `submission_index` may dynamically reach `slot`
    /// through its table, gating any later rewrite of the slot until that
    /// submission completes (Invariant 2).
    ///
    /// Out-of-bounds slots are ignored. The gate is monotonic: marking with an
    /// older submission index never lowers an existing gate.
    fn mark_in_use(&self, slot: u32, submission_index: SubmissionIndex) {
        if let Some(slot_state) = self.slots.get(slot as usize) {
            slot_state
                .available_after
                .fetch_max(submission_index, Ordering::AcqRel);
        }
    }
}

/// The host-side, mutable contents of a [`ResourceTable`]: the resources
/// currently bound in each slot, plus the reverse `resource → slots` map.
///
/// This is separate from the per-slot [`available_after`] gate in
/// [`ResourceTableSlots`] (which is lock-free and atomic). Guarding the
/// contents with a [`Mutex`] and holding it across both a host update's
/// gate-check-and-descriptor-write and queue-submit's slot marking is what
/// serializes the two against each other, upholding Invariant 1 in
/// `plans/resource-table.md`: a CPU descriptor write never races a submission
/// that could dynamically reach the slot.
///
/// [`available_after`]: Slot::available_after
#[derive(Debug)]
struct ResourceTableContents {
    /// The resource bound in each slot, keeping it alive for as long as the
    /// slot references it (Invariant 5). `None` for empty slots.
    ///
    /// M0 supports only sampled/depth texture views (deviation 3 in
    /// `plans/m0-notes.md`); later milestones widen this to the full
    /// `BindingResource` set.
    slots: Box<[Option<Arc<TextureView>>]>,

    /// Maps a bound texture (keyed by its tracker index) to the set of slots
    /// currently holding a view of it.
    ///
    /// This is the `resource → slots` multimap from the plan's object model.
    /// It powers destroy-driven metadata zeroing (M1) and usage-state updates
    /// (M2). In M0 nothing reads it yet — the [`slots`] `Arc`s alone are
    /// load-bearing for keep-alive — but it is maintained here so those later
    /// milestones can build on it.
    ///
    /// [`slots`]: ResourceTableContents::slots
    texture_to_slots: FastHashMap<TrackerIndex, SmallVec<[u32; 1]>>,
}

impl ResourceTableContents {
    fn new(size: u32) -> Self {
        Self {
            slots: (0..size).map(|_| None).collect(),
            texture_to_slots: FastHashMap::default(),
        }
    }

    /// Remove `slot` from the reverse map entry of whatever texture (if any)
    /// currently occupies it.
    fn unlink_slot(&mut self, slot: u32) {
        if let Some(old) = self.slots[slot as usize].take() {
            let index = old.parent.tracker_index();
            if let Some(slots) = self.texture_to_slots.get_mut(&index) {
                slots.retain(|s| *s != slot);
                if slots.is_empty() {
                    self.texture_to_slots.remove(&index);
                }
            }
        }
    }

    /// Bind `view` into `slot`, updating the reverse map. `slot` must be in
    /// bounds (checked by the caller's gate).
    fn set_slot(&mut self, slot: u32, view: Arc<TextureView>) {
        self.unlink_slot(slot);
        self.texture_to_slots
            .entry(view.parent.tracker_index())
            .or_default()
            .push(slot);
        self.slots[slot as usize] = Some(view);
    }

    /// Clear `slot`, updating the reverse map. `slot` must be in bounds.
    fn clear_slot(&mut self, slot: u32) {
        self.unlink_slot(slot);
    }

    /// The lowest slot index that currently has no binding, or `None` if the
    /// table is full. Used by `insert_binding` (D8: lowest-available-first).
    fn lowest_empty_slot(&self) -> Option<u32> {
        self.slots
            .iter()
            .position(Option::is_none)
            .map(|i| i as u32)
    }
}

#[derive(Debug)]
pub(crate) struct ResourceTableState {
    raw: Snatchable<Box<dyn hal::DynResourceTable>>,
}

/// A resource table: a device-wide, sparse, mutable array of resource bindings
/// addressed by slot index and accessed from shaders via `getResource<T>`.
///
/// See the module docs and `plans/resource-table.md` for the design.
#[derive(Debug)]
pub struct ResourceTable {
    pub(crate) state: ResourceState<ResourceTableState>,
    pub(crate) device: Arc<Device>,
    /// Number of slots, as requested at creation.
    pub(crate) size: u32,
    /// The `label` from the descriptor used to create the resource.
    pub(crate) label: String,
    pub(crate) tracking_data: TrackingData,
    /// Per-slot reuse gate (Invariant 2): lock-free, atomic.
    slots: ResourceTableSlots,
    /// Host-side bound resources and reverse map (see [`ResourceTableContents`]).
    contents: Mutex<ResourceTableContents>,
}

impl Drop for ResourceTable {
    #[allow(trivial_casts)]
    fn drop(&mut self) {
        profiling::scope!("ResourceTable::drop");
        api_log!("ResourceTable::drop {:?}", self as *const _);
        #[cfg(feature = "trace")]
        if let Some(trace) = self.device.trace.lock().as_mut() {
            use crate::device::trace::{to_trace, Action};
            trace.add(Action::DropResourceTable(unsafe { to_trace(self) }));
        }
        resource_log!("Destroy raw {}", self.error_ident());
        // SAFETY: We are in the Drop impl and we don't use `state.raw` anymore
        // after this point.
        if let ResourceState::Valid(state) = &mut self.state {
            if let Some(raw) = state.raw.take() {
                unsafe {
                    self.device.raw().destroy_resource_table(raw);
                }
            }
        }
    }
}

impl RawResourceAccess for ResourceTable {
    type DynResource = dyn hal::DynResourceTable;

    fn raw<'a>(&'a self, guard: &'a SnatchGuard) -> Option<&'a Self::DynResource> {
        self.state().ok()?.raw.get(guard).map(|it| it.as_ref())
    }
}

impl ResourceTable {
    pub(crate) fn state(&self) -> Result<&ResourceTableState, InvalidResourceError> {
        match &self.state {
            ResourceState::Valid(state) => Ok(state),
            ResourceState::Invalid => Err(InvalidResourceError(self.error_ident())),
        }
    }

    /// Consumed by work item 0.7 when a table is bound as encoder state
    /// (`set_resource_table`), and by 0.7's host-side update/insert flows.
    pub(crate) fn check_is_valid(&self) -> Result<(), InvalidResourceError> {
        self.state().map(|_| ())
    }

    /// The number of slots in the table.
    pub fn size(&self) -> u32 {
        self.size
    }

    pub(crate) fn invalid(device: Arc<Device>, desc: &ResourceTableDescriptor) -> Arc<Self> {
        Arc::new(ResourceTable {
            state: ResourceState::Invalid,
            size: desc.size,
            label: desc.label.to_string(),
            tracking_data: TrackingData::new(device.tracker_indices.resource_tables.clone()),
            // An invalid table has no backing, so it needs no slot storage: its
            // slots are never consulted (all uses are gated by `check_is_valid`),
            // and `desc.size` may be arbitrarily large after a failed creation.
            slots: ResourceTableSlots::new(0),
            contents: Mutex::new(rank::RESOURCE_TABLE_CONTENTS, ResourceTableContents::new(0)),
            device,
        })
    }

    /// Check whether `slot` may currently be rewritten with a CPU descriptor
    /// write, per the slot-reuse gate (Invariant 2 in `plans/resource-table.md`).
    ///
    /// Compares the slot's stored `available_after` submission index against the
    /// device's cached completed submission index (a fast, hal-round-trip-free
    /// check). Returns [`UpdateResourceTableError::SlotInUse`] with the
    /// `available_after` index when the slot may still be reached by an
    /// in-flight submission, or [`UpdateResourceTableError::SlotOutOfBounds`]
    /// when `slot >= size`.
    ///
    /// Consumed by the host update flows (`update_slot`/`insert_binding`).
    pub(crate) fn check_slot_available(&self, slot: u32) -> Result<(), UpdateResourceTableError> {
        let completed = self
            .device
            .last_completed_submission_index
            .load(Ordering::Acquire);
        self.slots.check_available(slot, completed)
    }

    /// Mark **every** slot of this table as reachable by submission
    /// `submission_index`, gating any later host rewrite until it completes.
    ///
    /// This is the M0 marking rule (deviation: unchecked mode ⇒ a shader may
    /// dynamically index any slot, so the whole table is conservatively marked;
    /// `plans/resource-table.md` Invariant 2). Called once per table referenced
    /// by a submission (from queue submit).
    ///
    /// The contents lock is held across the marking so that it is mutually
    /// exclusive with a concurrent host [`update_slot`]/[`remove_binding`],
    /// which check the reuse gate under the same lock. This is what makes the
    /// gate race-free (Invariant 1): a host descriptor write and this marking
    /// can never interleave such that a write lands on a slot a just-submitted
    /// submission can reach.
    ///
    /// [`update_slot`]: ResourceTable::update_slot
    /// [`remove_binding`]: ResourceTable::remove_binding
    pub(crate) fn mark_all_slots_in_use(&self, submission_index: SubmissionIndex) {
        let _contents = self.contents.lock();
        for slot in 0..self.slots.len() {
            self.slots.mark_in_use(slot, submission_index);
        }
    }

    /// Append the parent [`Texture`] of every currently-bound slot to `out`
    /// (with duplicates: a texture bound in multiple slots, or a slot bound to
    /// a view of an already-listed texture, appears more than once).
    ///
    /// Called at submit time to drive the pass-start layout transitions
    /// (D2/D11). Reads the *current* bindings, so bindings added after
    /// `finish()` but before submit are included (the "add-to-table-after-finish"
    /// case).
    pub(crate) fn collect_bound_textures(&self, out: &mut Vec<Arc<Texture>>) {
        let contents = self.contents.lock();
        out.extend(
            contents
                .slots
                .iter()
                .flatten()
                .map(|view| view.parent.clone()),
        );
    }

    /// Whether a view of the texture with the given tracker index is currently
    /// bound in any slot of this table.
    ///
    /// Consulted at texture-destroy time (through the lifetime tracker) so that a
    /// texture reachable *only* through this table — the core bindless case,
    /// which never enters a per-submission texture tracker — still has its hal
    /// teardown deferred until every in-flight submission that references the
    /// table has completed (Invariant 5, finding C2).
    ///
    /// This reads the table's *live* membership, which is sound because the
    /// slot-reuse gate (Invariant 2) forbids removing or overwriting a slot while
    /// any submission that marked it is still in flight: a member texture that an
    /// incomplete submission can reach through the table therefore cannot have
    /// left the table's contents, so it is always found here.
    pub(crate) fn contains_texture(&self, texture_index: TrackerIndex) -> bool {
        self.contents
            .lock()
            .texture_to_slots
            .contains_key(&texture_index)
    }

    /// Validate a candidate texture view for binding into this table in M0:
    /// same device, still valid, and sampleable (`TEXTURE_BINDING`). Storage or
    /// other non-sampled views fail here, pointing at the heterogeneous
    /// milestone (deviation 3 in `plans/m0-notes.md`).
    fn validate_texture_view_update(
        &self,
        view: &Arc<TextureView>,
    ) -> Result<(), UpdateResourceTableError> {
        self.check_is_valid()?;
        view.same_device(&self.device)?;
        view.check_valid()?;
        view.check_usage(wgt::TextureUsages::TEXTURE_BINDING)?;
        Ok(())
    }

    /// Gate, perform the hal descriptor write, and record the binding for
    /// `slot`. The caller holds `contents` and `snatch_guard`, and has already
    /// validated `view`.
    fn write_slot_locked(
        &self,
        contents: &mut ResourceTableContents,
        snatch_guard: &SnatchGuard,
        slot: u32,
        view: &Arc<TextureView>,
    ) -> Result<(), UpdateResourceTableError> {
        // Slot-reuse gate (Invariant 2) and bounds check. Held under
        // `contents`, which serializes against submit-time marking.
        self.check_slot_available(slot)?;

        let raw_table = self.try_raw(snatch_guard)?;
        let raw_view = view.try_raw(snatch_guard)?;

        // SAFETY: `slot` is in bounds (checked above); the slot-reuse gate
        // guarantees no in-flight submission can dynamically reach it, so this
        // CPU descriptor write is legal (Invariant 1; see the safety docs on
        // `hal::Device::update_table_slot`).
        unsafe {
            self.device.raw().update_table_slot(
                raw_table,
                slot,
                hal::ResourceTableUpdate::SampledTextureView(raw_view),
            );
        }

        contents.set_slot(slot, view.clone());
        Ok(())
    }

    /// Bind `view` into `slot`, per the plan's `table.update(slot, ..)`.
    ///
    /// Public so trace replay (`player`) can drive it directly on a resolved
    /// table, mirroring [`destroy`](ResourceTable::destroy).
    pub fn update_slot(
        self: &Arc<Self>,
        slot: u32,
        view: &Arc<TextureView>,
    ) -> Result<(), UpdateResourceTableError> {
        profiling::scope!("ResourceTable::update");
        api_log!("ResourceTable::update {:?} slot {slot}", Arc::as_ptr(self));

        #[cfg(feature = "trace")]
        if let Some(trace) = self.device.trace.lock().as_mut() {
            use crate::device::trace::{Action, IntoTrace as _};
            trace.add(Action::UpdateResourceTableSlot {
                id: self.to_trace(),
                slot,
                texture_view: view.to_trace(),
            });
        }

        self.validate_texture_view_update(view)?;
        let snatch_guard = self.device.snatchable_lock.read();
        let mut contents = self.contents.lock();
        self.write_slot_locked(&mut contents, &snatch_guard, slot, view)
    }

    /// Bind `view` into the lowest currently-empty slot and return its index,
    /// per the plan's `table.insert_binding(..)` (D8: lowest-available-first).
    ///
    /// Fails with [`UpdateResourceTableError::NoAvailableSlot`] if every slot is
    /// occupied, or with [`UpdateResourceTableError::SlotInUse`] if the chosen
    /// empty slot is still gated by an in-flight submission (in M0 every slot of
    /// a referenced table is marked, so freshly-submitted tables gate all their
    /// empty slots until the submission completes).
    pub fn insert_binding(
        self: &Arc<Self>,
        view: &Arc<TextureView>,
    ) -> Result<u32, UpdateResourceTableError> {
        profiling::scope!("ResourceTable::insert_binding");
        api_log!("ResourceTable::insert_binding {:?}", Arc::as_ptr(self));

        self.validate_texture_view_update(view)?;
        let snatch_guard = self.device.snatchable_lock.read();
        let mut contents = self.contents.lock();
        let slot = contents
            .lowest_empty_slot()
            .ok_or(UpdateResourceTableError::NoAvailableSlot)?;
        self.write_slot_locked(&mut contents, &snatch_guard, slot, view)?;

        // Recorded as an update to the concrete slot chosen, so replay is
        // deterministic (the slot picked depends on runtime table state).
        #[cfg(feature = "trace")]
        if let Some(trace) = self.device.trace.lock().as_mut() {
            use crate::device::trace::{Action, IntoTrace as _};
            trace.add(Action::UpdateResourceTableSlot {
                id: self.to_trace(),
                slot,
                texture_view: view.to_trace(),
            });
        }

        Ok(slot)
    }

    /// Clear the binding in `slot`, per the plan's `table.remove_binding(slot)`.
    ///
    /// Gated by the slot-reuse gate (Invariant 2): a slot still reachable by an
    /// in-flight submission cannot be cleared, because dropping our `Arc` could
    /// otherwise let the backing resource's deferred teardown proceed while the
    /// stale descriptor is still live for that submission.
    pub fn remove_binding(self: &Arc<Self>, slot: u32) -> Result<(), UpdateResourceTableError> {
        profiling::scope!("ResourceTable::remove_binding");
        api_log!(
            "ResourceTable::remove_binding {:?} slot {slot}",
            Arc::as_ptr(self)
        );

        #[cfg(feature = "trace")]
        if let Some(trace) = self.device.trace.lock().as_mut() {
            use crate::device::trace::{Action, IntoTrace as _};
            trace.add(Action::RemoveResourceTableBinding {
                id: self.to_trace(),
                slot,
            });
        }

        self.check_is_valid()?;
        let mut contents = self.contents.lock();
        self.check_slot_available(slot)?;
        contents.clear_slot(slot);
        Ok(())
    }

    pub fn destroy(self: &Arc<Self>) {
        profiling::scope!("ResourceTable::destroy");
        api_log!("ResourceTable::destroy {:?}", Arc::as_ptr(self));

        let device = &self.device;

        #[cfg(feature = "trace")]
        if let Some(trace) = device.trace.lock().as_mut() {
            use crate::device::trace::{Action, IntoTrace as _};
            trace.add(Action::DestroyResourceTable(self.to_trace()));
        }

        let ResourceState::Valid(state) = &self.state else {
            return;
        };

        let temp = {
            let mut snatch_guard = device.snatchable_lock.write();

            let raw = match state.raw.snatch(&mut snatch_guard) {
                Some(raw) => raw,
                None => {
                    // Per spec, it is valid to call `destroy` multiple times.
                    return;
                }
            };

            drop(snatch_guard);

            TempResource::DestroyedResourceTable(DestroyedResourceTable {
                raw: ManuallyDrop::new(raw),
                device: Arc::clone(&self.device),
                label: self.label().to_owned(),
            })
        };

        // Route hal teardown through the deferred-destruction machinery, exactly
        // as buffers/textures/query sets do. In later milestones a table can be
        // referenced by in-flight submissions; work item 0.8 records that so
        // that the lookup below finds the submission and teardown waits for it.
        // In M0 nothing references a table from a submission, so the lookup
        // returns `None` and `temp` is torn down promptly when it drops here.
        let Some(queue) = device.get_queue() else {
            return;
        };

        let mut life_lock = queue.lock_life();
        let last_submit_index = life_lock.get_resource_table_latest_submission_index(self);
        if let Some(last_submit_index) = last_submit_index {
            life_lock.schedule_resource_destruction(temp, last_submit_index);
        }
    }
}

crate::impl_resource_type!(ResourceTable);
crate::impl_labeled!(ResourceTable);
crate::impl_parent_device!(ResourceTable);
crate::impl_storage_item!(ResourceTable);
crate::impl_trackable!(ResourceTable);

/// A resource table that has been marked as destroyed and is staged for actual
/// deletion once no in-flight submission references it.
#[derive(Debug)]
pub struct DestroyedResourceTable {
    raw: ManuallyDrop<Box<dyn hal::DynResourceTable>>,
    device: Arc<Device>,
    label: String,
}

impl DestroyedResourceTable {
    pub fn label(&self) -> &dyn fmt::Debug {
        &self.label
    }
}

impl Drop for DestroyedResourceTable {
    fn drop(&mut self) {
        resource_log!("Destroy raw ResourceTable (destroyed) {:?}", self.label());
        // SAFETY: We are in the Drop impl and we don't use `self.raw` anymore
        // after this point.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        unsafe {
            hal::DynDevice::destroy_resource_table(self.device.raw(), raw);
        }
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateResourceTableError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error("Resource tables cannot be created with zero slots")]
    ZeroSize,
    #[error("Resource table size {size} exceeds the maximum of {max} slots")]
    TooManySlots { size: u32, max: u32 },
}

impl WebGpuError for CreateResourceTableError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Self::Device(e) => e.webgpu_error_type(),
            Self::MissingFeatures(e) => e.webgpu_error_type(),
            Self::ZeroSize | Self::TooManySlots { .. } => ErrorType::Validation,
        }
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum UpdateResourceTableError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),
    #[error(transparent)]
    MissingTextureUsage(#[from] MissingTextureUsageError),
    #[error("Resource table slot {slot} is out of bounds for a table with {size} slots")]
    SlotOutOfBounds { slot: u32, size: u32 },
    #[error(
        "Resource table slot cannot be updated yet: it may still be used by a submission that \
         completes at index {available_after}"
    )]
    SlotInUse { available_after: SubmissionIndex },
    #[error("Resource table has no available slot for `insert_binding`: all slots are occupied")]
    NoAvailableSlot,
}

impl WebGpuError for UpdateResourceTableError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Self::Device(e) => e.webgpu_error_type(),
            Self::MissingFeatures(e) => e.webgpu_error_type(),
            Self::InvalidResource(e) => e.webgpu_error_type(),
            Self::DestroyedResource(e) => e.webgpu_error_type(),
            Self::MissingTextureUsage(e) => e.webgpu_error_type(),
            Self::SlotOutOfBounds { .. } | Self::SlotInUse { .. } | Self::NoAvailableSlot => {
                ErrorType::Validation
            }
        }
    }
}

impl Device {
    pub fn create_resource_table(
        self: &Arc<Self>,
        desc: &ResourceTableDescriptor,
    ) -> (Arc<ResourceTable>, Option<CreateResourceTableError>) {
        profiling::scope!("Device::create_resource_table");

        let (table, error) = match self.create_resource_table_inner(desc) {
            Ok(table) => (table, None),
            Err(error) => (ResourceTable::invalid(self.clone(), desc), Some(error)),
        };

        #[cfg(feature = "trace")]
        if let Some(trace) = self.trace.lock().as_mut() {
            use crate::device::trace::{Action, IntoTrace as _};
            trace.add(Action::CreateResourceTable {
                id: table.to_trace(),
                desc: desc.clone(),
            });
        }

        api_log!("Device::create_resource_table -> {:?}", Arc::as_ptr(&table));
        (table, error)
    }

    pub(crate) fn create_resource_table_inner(
        self: &Arc<Self>,
        desc: &ResourceTableDescriptor,
    ) -> Result<Arc<ResourceTable>, CreateResourceTableError> {
        self.check_is_valid()?;
        self.require_features(wgt::Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE)?;

        if desc.size == 0 {
            return Err(CreateResourceTableError::ZeroSize);
        }
        if desc.size > MAX_RESOURCE_TABLE_SIZE {
            return Err(CreateResourceTableError::TooManySlots {
                size: desc.size,
                max: MAX_RESOURCE_TABLE_SIZE,
            });
        }

        let hal_desc = hal::ResourceTableDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            size: desc.size,
        };

        let raw = unsafe { self.raw().create_resource_table(&hal_desc) }
            .map_err(|e| self.handle_hal_error_with_nonfatal_oom(e))?;

        Ok(Arc::new(ResourceTable {
            state: ResourceState::Valid(ResourceTableState {
                raw: Snatchable::new(raw),
            }),
            device: self.clone(),
            size: desc.size,
            label: desc.label.to_string(),
            tracking_data: TrackingData::new(self.tracker_indices.resource_tables.clone()),
            slots: ResourceTableSlots::new(desc.size),
            contents: Mutex::new(
                rank::RESOURCE_TABLE_CONTENTS,
                ResourceTableContents::new(desc.size),
            ),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The slot-reuse gate (Invariant 2). Exercised directly on
    /// [`ResourceTableSlots`] so no device or queue is required — the completed
    /// submission index is supplied by the test.
    #[test]
    fn slot_reuse_gate() {
        let slots = ResourceTableSlots::new(4);

        // Fresh slots have never been used, so they are always available.
        assert!(slots.check_available(0, 0).is_ok());
        assert!(slots.check_available(3, 0).is_ok());

        // Out-of-bounds slot.
        assert!(matches!(
            slots.check_available(4, 100),
            Err(UpdateResourceTableError::SlotOutOfBounds { slot: 4, size: 4 })
        ));

        // Mark slot 1 as used by submission 5.
        slots.mark_in_use(1, 5);

        // Gated while the completed index is below 5.
        assert!(matches!(
            slots.check_available(1, 0),
            Err(UpdateResourceTableError::SlotInUse { available_after: 5 })
        ));
        assert!(matches!(
            slots.check_available(1, 4),
            Err(UpdateResourceTableError::SlotInUse { available_after: 5 })
        ));

        // Available once the completing submission reaches the stored index
        // (`available_after <= completed`).
        assert!(slots.check_available(1, 5).is_ok());
        assert!(slots.check_available(1, 6).is_ok());

        // Other slots are unaffected.
        assert!(slots.check_available(0, 0).is_ok());

        // The gate is monotonic: marking with an older index does not lower it.
        slots.mark_in_use(1, 3);
        assert!(matches!(
            slots.check_available(1, 4),
            Err(UpdateResourceTableError::SlotInUse { available_after: 5 })
        ));

        // Marking out-of-bounds slots is a no-op (does not panic).
        slots.mark_in_use(99, 10);
    }

    #[test]
    fn slot_count() {
        assert_eq!(ResourceTableSlots::new(0).len(), 0);
        assert_eq!(ResourceTableSlots::new(7).len(), 7);
    }
}
