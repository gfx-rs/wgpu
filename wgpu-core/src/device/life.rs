use crate::{
    device::{
        queue::{EncoderInFlight, SubmittedWorkDoneClosure, TempResource},
        DeviceError, DeviceLostClosure,
    },
    resource::{self, Buffer, Texture, Trackable},
    snatch::SnatchGuard,
    SubmissionIndex,
};
use smallvec::SmallVec;

use std::sync::Arc;
use thiserror::Error;

/// A command submitted to the GPU for execution.
///
/// ## Keeping resources alive while the GPU is using them
///
/// [`wgpu_hal`] requires that, when a command is submitted to a queue, all the
/// resources it uses must remain alive until it has finished executing.
///
/// [`wgpu_hal`]: hal
/// [`ResourceInfo::submission_index`]: crate::resource::ResourceInfo
struct ActiveSubmission {
    /// The index of the submission we track.
    ///
    /// When `Device::fence`'s value is greater than or equal to this, our queue
    /// submission has completed.
    index: SubmissionIndex,

    /// Temporary resources to be freed once this queue submission has completed.
    temp_resources: Vec<TempResource>,

    /// Buffers to be mapped once this submission has completed.
    mapped: Vec<Arc<Buffer>>,

    /// Command buffers used by this submission, and the encoder that owns them.
    ///
    /// [`wgpu_hal::Queue::submit`] requires the submitted command buffers to
    /// remain alive until the submission has completed execution. Command
    /// encoders double as allocation pools for command buffers, so holding them
    /// here and cleaning them up in [`LifetimeTracker::triage_submissions`]
    /// satisfies that requirement.
    ///
    /// Once this submission has completed, the command buffers are reset and
    /// the command encoder is recycled.
    ///
    /// [`wgpu_hal::Queue::submit`]: hal::Queue::submit
    encoders: Vec<EncoderInFlight>,

    /// List of queue "on_submitted_work_done" closures to be called once this
    /// submission has completed.
    work_done_closures: SmallVec<[SubmittedWorkDoneClosure; 1]>,
}

impl ActiveSubmission {
    /// Returns true if this submission contains the given buffer.
    ///
    /// This only uses constant-time operations.
    pub fn contains_buffer(&self, buffer: &Buffer) -> bool {
        for encoder in &self.encoders {
            // The ownership location of buffers depends on where the command encoder
            // came from. If it is the staging command encoder on the queue, it is
            // in the pending buffer list. If it came from a user command encoder,
            // it is in the tracker.

            if encoder.trackers.buffers.contains(buffer) {
                return true;
            }

            if encoder
                .pending_buffers
                .contains_key(&buffer.tracker_index())
            {
                return true;
            }
        }

        false
    }

    /// Returns true if this submission contains the given texture.
    ///
    /// This only uses constant-time operations.
    pub fn contains_texture(&self, texture: &Texture) -> bool {
        for encoder in &self.encoders {
            // The ownership location of textures depends on where the command encoder
            // came from. If it is the staging command encoder on the queue, it is
            // in the pending buffer list. If it came from a user command encoder,
            // it is in the tracker.

            if encoder.trackers.textures.contains(texture) {
                return true;
            }

            if encoder
                .pending_textures
                .contains_key(&texture.tracker_index())
            {
                return true;
            }
        }

        false
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum WaitIdleError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Tried to wait using a submission index ({0}) that has not been returned by a successful submission (last successful submission: {1})")]
    WrongSubmissionIndex(SubmissionIndex, SubmissionIndex),
    #[error("GPU got stuck :(")]
    StuckGpu,
}

/// Resource tracking for a device.
///
/// ## Host mapping buffers
///
/// A buffer cannot be mapped until all active queue submissions that use it
/// have completed. To that end:
///
/// -   Each buffer's `ResourceInfo::submission_index` records the index of the
///     most recent queue submission that uses that buffer.
///
/// -   Calling `Global::buffer_map_async` adds the buffer to
///     `self.mapped`, and changes `Buffer::map_state` to prevent it
///     from being used in any new submissions.
///
/// -   When the device is polled, the following `LifetimeTracker` methods decide
///     what should happen next:
///
///     1)  `triage_mapped` drains `self.mapped`, checking the submission index
///         of each buffer against the queue submissions that have finished
///         execution. Buffers used by submissions still in flight go in
///         `self.active[index].mapped`, and the rest go into
///         `self.ready_to_map`.
///
///     2)  `triage_submissions` moves entries in `self.active[i]` for completed
///         submissions to `self.ready_to_map`.  At this point, both
///         `self.active` and `self.ready_to_map` are up to date with the given
///         submission index.
///
///     3)  `handle_mapping` drains `self.ready_to_map` and actually maps the
///         buffers, collecting a list of notification closures to call.
///
/// Only calling `Global::buffer_map_async` clones a new `Arc` for the
/// buffer. This new `Arc` is only dropped by `handle_mapping`.
pub(crate) struct LifetimeTracker {
    /// Buffers for which a call to [`Buffer::map_async`] has succeeded, but
    /// which haven't been examined by `triage_mapped` yet to decide when they
    /// can be mapped.
    mapped: Vec<Arc<Buffer>>,

    /// Resources used by queue submissions still in flight. One entry per
    /// submission, with older submissions appearing before younger.
    ///
    /// Entries are added by `track_submission` and drained by
    /// `LifetimeTracker::triage_submissions`. Lots of methods contribute data
    /// to particular entries.
    active: Vec<ActiveSubmission>,

    /// Buffers the user has asked us to map, and which are not used by any
    /// queue submission still in flight.
    ready_to_map: Vec<Arc<Buffer>>,

    /// Queue "on_submitted_work_done" closures that were initiated for while there is no
    /// currently pending submissions. These cannot be immediately invoked as they
    /// must happen _after_ all mapped buffer callbacks are mapped, so we defer them
    /// here until the next time the device is maintained.
    work_done_closures: SmallVec<[SubmittedWorkDoneClosure; 1]>,

    /// Closure to be called on "lose the device". This is invoked directly by
    /// device.lose or by the UserCallbacks returned from maintain when the device
    /// has been destroyed and its queues are empty.
    pub device_lost_closure: Option<DeviceLostClosure>,
}

impl LifetimeTracker {
    pub fn new() -> Self {
        Self {
            mapped: Vec::new(),
            active: Vec::new(),
            ready_to_map: Vec::new(),
            work_done_closures: SmallVec::new(),
            device_lost_closure: None,
        }
    }

    /// Return true if there are no queue submissions still in flight.
    pub fn queue_empty(&self) -> bool {
        self.active.is_empty()
    }

    /// Start tracking resources associated with a new queue submission.
    pub fn track_submission(
        &mut self,
        index: SubmissionIndex,
        temp_resources: impl Iterator<Item = TempResource>,
        encoders: Vec<EncoderInFlight>,
    ) {
        self.active.push(ActiveSubmission {
            index,
            temp_resources: temp_resources.collect(),
            mapped: Vec::new(),
            encoders,
            work_done_closures: SmallVec::new(),
        });
    }

    pub(crate) fn map(&mut self, value: &Arc<Buffer>) {
        self.mapped.push(value.clone());
    }

    /// Returns the submission index of the most recent submission that uses the
    /// given buffer.
    pub fn get_buffer_latest_submission_index(&self, buffer: &Buffer) -> Option<SubmissionIndex> {
        // We iterate in reverse order, so that we can bail out early as soon
        // as we find a hit.
        self.active.iter().rev().find_map(|submission| {
            if submission.contains_buffer(buffer) {
                Some(submission.index)
            } else {
                None
            }
        })
    }

    /// Returns the submission index of the most recent submission that uses the
    /// given texture.
    pub fn get_texture_latest_submission_index(
        &self,
        texture: &Texture,
    ) -> Option<SubmissionIndex> {
        // We iterate in reverse order, so that we can bail out early as soon
        // as we find a hit.
        self.active.iter().rev().find_map(|submission| {
            if submission.contains_texture(texture) {
                Some(submission.index)
            } else {
                None
            }
        })
    }

    /// Sort out the consequences of completed submissions.
    ///
    /// Assume that all submissions up through `last_done` have completed.
    ///
    /// -   Buffers used by those submissions are now ready to map, if requested.
    ///     Add any buffers in the submission's [`mapped`] list to
    ///     [`self.ready_to_map`], where [`LifetimeTracker::handle_mapping`]
    ///     will find them.
    ///
    /// Return a list of [`SubmittedWorkDoneClosure`]s to run.
    ///
    /// [`mapped`]: ActiveSubmission::mapped
    /// [`self.ready_to_map`]: LifetimeTracker::ready_to_map
    /// [`SubmittedWorkDoneClosure`]: crate::device::queue::SubmittedWorkDoneClosure
    #[must_use]
    pub fn triage_submissions(
        &mut self,
        last_done: SubmissionIndex,
        command_allocator: &crate::command::CommandAllocator,
    ) -> SmallVec<[SubmittedWorkDoneClosure; 1]> {
        profiling::scope!("triage_submissions");

        //TODO: enable when `is_sorted_by_key` is stable
        //debug_assert!(self.active.is_sorted_by_key(|a| a.index));
        let done_count = self
            .active
            .iter()
            .position(|a| a.index > last_done)
            .unwrap_or(self.active.len());

        let mut work_done_closures: SmallVec<_> = self.work_done_closures.drain(..).collect();
        for a in self.active.drain(..done_count) {
            self.ready_to_map.extend(a.mapped);
            for encoder in a.encoders {
                let raw = unsafe { encoder.land() };
                command_allocator.release_encoder(raw);
            }
            drop(a.temp_resources);
            work_done_closures.extend(a.work_done_closures);
        }
        work_done_closures
    }

    pub fn schedule_resource_destruction(
        &mut self,
        temp_resource: TempResource,
        last_submit_index: SubmissionIndex,
    ) {
        let resources = self
            .active
            .iter_mut()
            .find(|a| a.index == last_submit_index)
            .map(|a| &mut a.temp_resources);
        if let Some(resources) = resources {
            resources.push(temp_resource);
        }
    }

    pub fn add_work_done_closure(&mut self, closure: SubmittedWorkDoneClosure) {
        match self.active.last_mut() {
            Some(active) => {
                active.work_done_closures.push(closure);
            }
            // We must defer the closure until all previously occurring map_async closures
            // have fired. This is required by the spec.
            None => {
                self.work_done_closures.push(closure);
            }
        }
    }

    /// Determine which buffers are ready to map, and which must wait for the
    /// GPU.
    ///
    /// See the documentation for [`LifetimeTracker`] for details.
    pub(crate) fn triage_mapped(&mut self) {
        if self.mapped.is_empty() {
            return;
        }

        for buffer in self.mapped.drain(..) {
            let submission = self
                .active
                .iter_mut()
                .rev()
                .find(|a| a.contains_buffer(&buffer));

            submission
                .map_or(&mut self.ready_to_map, |a| &mut a.mapped)
                .push(buffer);
        }
    }

    /// Map the buffers in `self.ready_to_map`.
    ///
    /// Return a list of mapping notifications to send.
    ///
    /// See the documentation for [`LifetimeTracker`] for details.
    #[must_use]
    pub(crate) fn handle_mapping(
        &mut self,
        raw: &dyn hal::DynDevice,
        snatch_guard: &SnatchGuard,
    ) -> Vec<super::BufferMapPendingClosure> {
        if self.ready_to_map.is_empty() {
            return Vec::new();
        }
        let mut pending_callbacks: Vec<super::BufferMapPendingClosure> =
            Vec::with_capacity(self.ready_to_map.len());

        for buffer in self.ready_to_map.drain(..) {
            // This _cannot_ be inlined into the match. If it is, the lock will be held
            // open through the whole match, resulting in a deadlock when we try to re-lock
            // the buffer back to active.
            let mapping = std::mem::replace(
                &mut *buffer.map_state.lock(),
                resource::BufferMapState::Idle,
            );
            let pending_mapping = match mapping {
                resource::BufferMapState::Waiting(pending_mapping) => pending_mapping,
                // Mapping cancelled
                resource::BufferMapState::Idle => continue,
                // Mapping queued at least twice by map -> unmap -> map
                // and was already successfully mapped below
                resource::BufferMapState::Active { .. } => {
                    *buffer.map_state.lock() = mapping;
                    continue;
                }
                _ => panic!("No pending mapping."),
            };
            let status = if pending_mapping.range.start != pending_mapping.range.end {
                let host = pending_mapping.op.host;
                let size = pending_mapping.range.end - pending_mapping.range.start;
                match super::map_buffer(
                    raw,
                    &buffer,
                    pending_mapping.range.start,
                    size,
                    host,
                    snatch_guard,
                ) {
                    Ok(mapping) => {
                        *buffer.map_state.lock() = resource::BufferMapState::Active {
                            mapping,
                            range: pending_mapping.range.clone(),
                            host,
                        };
                        Ok(())
                    }
                    Err(e) => {
                        log::error!("Mapping failed: {e}");
                        Err(e)
                    }
                }
            } else {
                *buffer.map_state.lock() = resource::BufferMapState::Active {
                    mapping: hal::BufferMapping {
                        ptr: std::ptr::NonNull::dangling(),
                        is_coherent: true,
                    },
                    range: pending_mapping.range,
                    host: pending_mapping.op.host,
                };
                Ok(())
            };
            pending_callbacks.push((pending_mapping.op, status));
        }
        pending_callbacks
    }
}
