#[cfg(feature = "trace")]
use crate::device::trace;
use crate::{
    binding_model::{BindGroup, BindGroupLayout, PipelineLayout},
    command::RenderBundle,
    device::{
        queue::{EncoderInFlight, SubmittedWorkDoneClosure, TempResource},
        DeviceError,
    },
    hal_api::HalApi,
    hub::Hub,
    id,
    identity::GlobalIdentityHandlerFactory,
    pipeline::{ComputePipeline, RenderPipeline},
    resource::{self, Buffer, QuerySet, Resource, Sampler, Texture, TextureView},
    track::{BindGroupStates, RenderBundleScope, Tracker},
    SubmissionIndex,
};
use smallvec::SmallVec;

use parking_lot::Mutex;
use thiserror::Error;

use std::{mem, sync::Arc};

/// A struct that keeps lists of resources that are no longer needed by the user.
pub(crate) struct SuspectedResources<A: HalApi> {
    pub(crate) buffers: Vec<Arc<Buffer<A>>>,
    pub(crate) textures: Vec<Arc<Texture<A>>>,
    pub(crate) texture_views: Vec<Arc<TextureView<A>>>,
    pub(crate) samplers: Vec<Arc<Sampler<A>>>,
    pub(crate) bind_groups: Vec<Arc<BindGroup<A>>>,
    pub(crate) compute_pipelines: Vec<Arc<ComputePipeline<A>>>,
    pub(crate) render_pipelines: Vec<Arc<RenderPipeline<A>>>,
    pub(crate) bind_group_layouts: Vec<Arc<BindGroupLayout<A>>>,
    pub(crate) pipeline_layouts: Vec<Arc<PipelineLayout<A>>>,
    pub(crate) render_bundles: Vec<Arc<RenderBundle<A>>>,
    pub(crate) query_sets: Vec<Arc<QuerySet<A>>>,
}

impl<A: HalApi> SuspectedResources<A> {
    pub(crate) fn new() -> Self {
        Self {
            buffers: Vec::new(),
            textures: Vec::new(),
            texture_views: Vec::new(),
            samplers: Vec::new(),
            bind_groups: Vec::new(),
            compute_pipelines: Vec::new(),
            render_pipelines: Vec::new(),
            bind_group_layouts: Vec::new(),
            pipeline_layouts: Vec::new(),
            render_bundles: Vec::new(),
            query_sets: Vec::new(),
        }
    }
    pub(crate) fn clear(&mut self) {
        self.buffers.clear();
        self.textures.clear();
        self.texture_views.clear();
        self.samplers.clear();
        self.bind_groups.clear();
        self.compute_pipelines.clear();
        self.render_pipelines.clear();
        self.bind_group_layouts.clear();
        self.pipeline_layouts.clear();
        self.render_bundles.clear();
        self.query_sets.clear();
    }

    pub(crate) fn extend(&mut self, other: &Self) {
        self.buffers.extend_from_slice(&other.buffers);
        self.textures.extend_from_slice(&other.textures);
        self.texture_views.extend_from_slice(&other.texture_views);
        self.samplers.extend_from_slice(&other.samplers);
        self.bind_groups.extend_from_slice(&other.bind_groups);
        self.compute_pipelines
            .extend_from_slice(&other.compute_pipelines);
        self.render_pipelines
            .extend_from_slice(&other.render_pipelines);
        self.bind_group_layouts
            .extend_from_slice(&other.bind_group_layouts);
        self.pipeline_layouts
            .extend_from_slice(&other.pipeline_layouts);
        self.render_bundles.extend_from_slice(&other.render_bundles);
        self.query_sets.extend_from_slice(&other.query_sets);
    }

    pub(crate) fn add_render_bundle_scope(&mut self, trackers: &RenderBundleScope<A>) {
        self.buffers
            .extend(trackers.buffers.used_resources().cloned());
        self.textures
            .extend(trackers.textures.used_resources().cloned());
        self.bind_groups
            .extend(trackers.bind_groups.used_resources().cloned());
        self.render_pipelines
            .extend(trackers.render_pipelines.used_resources().cloned());
        self.query_sets
            .extend(trackers.query_sets.used_resources().cloned());
    }

    pub(crate) fn add_bind_group_states(&mut self, trackers: &BindGroupStates<A>) {
        self.buffers
            .extend(trackers.buffers.used_resources().cloned());
        self.textures
            .extend(trackers.textures.used_resources().cloned());
        self.texture_views.extend(trackers.views.used_resources());
        self.samplers.extend(trackers.samplers.used_resources());
    }
}

/// Raw backend resources that should be freed shortly.
#[derive(Debug)]
struct NonReferencedResources<A: HalApi> {
    buffers: Vec<Arc<Buffer<A>>>,
    textures: Vec<Arc<Texture<A>>>,
    texture_views: Vec<Arc<TextureView<A>>>,
    samplers: Vec<Arc<Sampler<A>>>,
    bind_groups: Vec<Arc<BindGroup<A>>>,
    compute_pipes: Vec<Arc<ComputePipeline<A>>>,
    render_pipes: Vec<Arc<RenderPipeline<A>>>,
    bind_group_layouts: Vec<Arc<BindGroupLayout<A>>>,
    pipeline_layouts: Vec<Arc<PipelineLayout<A>>>,
    query_sets: Vec<Arc<QuerySet<A>>>,
}

impl<A: HalApi> NonReferencedResources<A> {
    fn new() -> Self {
        Self {
            buffers: Vec::new(),
            textures: Vec::new(),
            texture_views: Vec::new(),
            samplers: Vec::new(),
            bind_groups: Vec::new(),
            compute_pipes: Vec::new(),
            render_pipes: Vec::new(),
            bind_group_layouts: Vec::new(),
            pipeline_layouts: Vec::new(),
            query_sets: Vec::new(),
        }
    }

    fn extend(&mut self, other: Self) {
        self.buffers.extend(other.buffers);
        self.textures.extend(other.textures);
        self.texture_views.extend(other.texture_views);
        self.samplers.extend(other.samplers);
        self.bind_groups.extend(other.bind_groups);
        self.compute_pipes.extend(other.compute_pipes);
        self.render_pipes.extend(other.render_pipes);
        self.query_sets.extend(other.query_sets);
        assert!(other.bind_group_layouts.is_empty());
        assert!(other.pipeline_layouts.is_empty());
    }

    unsafe fn clean(&mut self) {
        if !self.buffers.is_empty() {
            profiling::scope!("destroy_buffers");
            self.buffers.clear();
        }
        if !self.textures.is_empty() {
            profiling::scope!("destroy_textures");
            self.textures.clear();
        }
        if !self.texture_views.is_empty() {
            profiling::scope!("destroy_texture_views");
            self.texture_views.clear();
        }
        if !self.samplers.is_empty() {
            profiling::scope!("destroy_samplers");
            self.samplers.clear();
        }
        if !self.bind_groups.is_empty() {
            profiling::scope!("destroy_bind_groups");
            self.bind_groups.clear();
        }
        if !self.compute_pipes.is_empty() {
            profiling::scope!("destroy_compute_pipelines");
            self.compute_pipes.clear();
        }
        if !self.render_pipes.is_empty() {
            profiling::scope!("destroy_render_pipelines");
            self.render_pipes.clear();
        }
        if !self.bind_group_layouts.is_empty() {
            profiling::scope!("destroy_bind_group_layouts");
            self.bind_group_layouts.clear();
        }
        if !self.pipeline_layouts.is_empty() {
            profiling::scope!("destroy_pipeline_layouts");
            self.pipeline_layouts.clear();
        }
        if !self.query_sets.is_empty() {
            profiling::scope!("destroy_query_sets");
            self.query_sets.clear();
        }
    }
}

/// Resources used by a queue submission, and work to be done once it completes.
struct ActiveSubmission<A: HalApi> {
    /// The index of the submission we track.
    ///
    /// When `Device::fence`'s value is greater than or equal to this, our queue
    /// submission has completed.
    index: SubmissionIndex,

    /// Resources to be freed once this queue submission has completed.
    ///
    /// When the device is polled, for completed submissions,
    /// `triage_submissions` merges these into
    /// `LifetimeTracker::free_resources`. From there,
    /// `LifetimeTracker::cleanup` passes them to the hal to be freed.
    ///
    /// This includes things like temporary resources and resources that are
    /// used by submitted commands but have been dropped by the user (meaning that
    /// this submission is their last reference.)
    last_resources: NonReferencedResources<A>,

    /// Buffers to be mapped once this submission has completed.
    mapped: Vec<Arc<Buffer<A>>>,

    encoders: Vec<EncoderInFlight<A>>,
    work_done_closures: SmallVec<[SubmittedWorkDoneClosure; 1]>,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum WaitIdleError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Tried to wait using a submission index from the wrong device. Submission index is from device {0:?}. Called poll on device {1:?}.")]
    WrongSubmissionIndex(id::QueueId, id::DeviceId),
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
/// -   Each buffer's `LifeGuard::submission_index` records the index of the
///     most recent queue submission that uses that buffer.
///
/// -   Calling `map_async` adds the buffer to `self.mapped`, and changes
///     `Buffer::map_state` to prevent it from being used in any new
///     submissions.
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
///         buffers, collecting a list of notification closures to call. But any
///         buffers that were dropped by the user get moved to
///         `self.free_resources`.
///
///     4)  `cleanup` frees everything in `free_resources`.
///
/// Only `self.mapped` holds a `RefCount` for the buffer; it is dropped by
/// `triage_mapped`.
pub(crate) struct LifetimeTracker<A: HalApi> {
    /// Resources that the user has requested be mapped, but which are used by
    /// queue submissions still in flight.
    mapped: Vec<Arc<Buffer<A>>>,

    /// Buffers can be used in a submission that is yet to be made, by the
    /// means of `write_buffer()`, so we have a special place for them.
    pub future_suspected_buffers: Vec<Arc<Buffer<A>>>,

    /// Textures can be used in the upcoming submission by `write_texture`.
    pub future_suspected_textures: Vec<Arc<Texture<A>>>,

    /// Resources whose user handle has died (i.e. drop/destroy has been called)
    /// and will likely be ready for destruction soon.
    pub suspected_resources: SuspectedResources<A>,

    /// Resources used by queue submissions still in flight. One entry per
    /// submission, with older submissions appearing before younger.
    ///
    /// Entries are added by `track_submission` and drained by
    /// `LifetimeTracker::triage_submissions`. Lots of methods contribute data
    /// to particular entries.
    active: Vec<ActiveSubmission<A>>,

    /// Raw backend resources that are neither referenced nor used.
    ///
    /// These are freed by `LifeTracker::cleanup`, which is called from periodic
    /// maintenance functions like `Global::device_poll`, and when a device is
    /// destroyed.
    free_resources: NonReferencedResources<A>,

    /// Buffers the user has asked us to map, and which are not used by any
    /// queue submission still in flight.
    ready_to_map: Vec<Arc<Buffer<A>>>,
}

impl<A: HalApi> LifetimeTracker<A> {
    pub fn new() -> Self {
        Self {
            mapped: Vec::new(),
            future_suspected_buffers: Vec::new(),
            future_suspected_textures: Vec::new(),
            suspected_resources: SuspectedResources::new(),
            active: Vec::new(),
            free_resources: NonReferencedResources::new(),
            ready_to_map: Vec::new(),
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
        temp_resources: impl Iterator<Item = TempResource<A>>,
        encoders: Vec<EncoderInFlight<A>>,
    ) {
        let mut last_resources = NonReferencedResources::new();
        for res in temp_resources {
            match res {
                TempResource::Buffer(raw) => last_resources.buffers.push(raw),
                TempResource::Texture(raw, views) => {
                    last_resources.textures.push(raw);
                    last_resources.texture_views.extend(views);
                }
            }
        }

        self.active.push(ActiveSubmission {
            index,
            last_resources,
            mapped: Vec::new(),
            encoders,
            work_done_closures: SmallVec::new(),
        });
    }

    pub fn post_submit(&mut self) {
        self.suspected_resources
            .buffers
            .append(&mut self.future_suspected_buffers);
        self.suspected_resources
            .textures
            .append(&mut self.future_suspected_textures);
    }

    pub(crate) fn map(&mut self, value: &Arc<Buffer<A>>) {
        self.mapped.push(value.clone());
    }

    /// Sort out the consequences of completed submissions.
    ///
    /// Assume that all submissions up through `last_done` have completed.
    ///
    /// -   Buffers used by those submissions are now ready to map, if
    ///     requested. Add any buffers in the submission's [`mapped`] list to
    ///     [`self.ready_to_map`], where [`LifetimeTracker::handle_mapping`] will find
    ///     them.
    ///
    /// -   Resources whose final use was in those submissions are now ready to
    ///     free. Add any resources in the submission's [`last_resources`] table
    ///     to [`self.free_resources`], where [`LifetimeTracker::cleanup`] will find
    ///     them.
    ///
    /// Return a list of [`SubmittedWorkDoneClosure`]s to run.
    ///
    /// [`mapped`]: ActiveSubmission::mapped
    /// [`self.ready_to_map`]: LifetimeTracker::ready_to_map
    /// [`last_resources`]: ActiveSubmission::last_resources
    /// [`self.free_resources`]: LifetimeTracker::free_resources
    /// [`SubmittedWorkDoneClosure`]: crate::device::queue::SubmittedWorkDoneClosure
    #[must_use]
    pub fn triage_submissions(
        &mut self,
        last_done: SubmissionIndex,
        command_allocator: &mut super::CommandAllocator<A>,
    ) -> SmallVec<[SubmittedWorkDoneClosure; 1]> {
        profiling::scope!("triage_submissions");

        //TODO: enable when `is_sorted_by_key` is stable
        //debug_assert!(self.active.is_sorted_by_key(|a| a.index));
        let done_count = self
            .active
            .iter()
            .position(|a| a.index > last_done)
            .unwrap_or(self.active.len());

        let mut work_done_closures = SmallVec::new();
        for a in self.active.drain(..done_count) {
            log::trace!("Active submission {} is done", a.index);
            self.free_resources.extend(a.last_resources);
            self.ready_to_map.extend(a.mapped);
            for encoder in a.encoders {
                let raw = unsafe { encoder.land() };
                command_allocator.release_encoder(raw);
            }
            work_done_closures.extend(a.work_done_closures);
        }
        work_done_closures
    }

    pub fn cleanup(&mut self) {
        profiling::scope!("LifetimeTracker::cleanup");
        unsafe {
            self.free_resources.clean();
        }
    }

    pub fn schedule_resource_destruction(
        &mut self,
        temp_resource: TempResource<A>,
        last_submit_index: SubmissionIndex,
    ) {
        let resources = self
            .active
            .iter_mut()
            .find(|a| a.index == last_submit_index)
            .map_or(&mut self.free_resources, |a| &mut a.last_resources);
        match temp_resource {
            TempResource::Buffer(raw) => resources.buffers.push(raw),
            TempResource::Texture(raw, views) => {
                resources.texture_views.extend(views);
                resources.textures.push(raw);
            }
        }
    }

    pub fn add_work_done_closure(
        &mut self,
        closure: SubmittedWorkDoneClosure,
    ) -> Option<SubmittedWorkDoneClosure> {
        match self.active.last_mut() {
            Some(active) => {
                active.work_done_closures.push(closure);
                None
            }
            // Note: we can't immediately invoke the closure, since it assumes
            // nothing is currently locked in the hubs.
            None => Some(closure),
        }
    }
}

impl<A: HalApi> LifetimeTracker<A> {
    /// Identify resources to free, according to `trackers` and `self.suspected_resources`.
    ///
    /// Given `trackers`, the [`Tracker`] belonging to same [`Device`] as
    /// `self`, and `hub`, the [`Hub`] to which that `Device` belongs:
    ///
    /// Remove from `trackers` each resource mentioned in
    /// [`self.suspected_resources`]. If `trackers` held the final reference to
    /// that resource, add it to the appropriate free list, to be destroyed by
    /// the hal:
    ///
    /// -   Add resources used by queue submissions still in flight to the
    ///     [`last_resources`] table of the last such submission's entry in
    ///     [`self.active`]. When that submission has finished execution. the
    ///     [`triage_submissions`] method will move them to
    ///     [`self.free_resources`].
    ///
    /// -   Add resources that can be freed right now to [`self.free_resources`]
    ///     directly. [`LifetimeTracker::cleanup`] will take care of them as
    ///     part of this poll.
    ///
    /// ## Entrained resources
    ///
    /// This function finds resources that are used only by other resources
    /// ready to be freed, and adds those to the free lists as well. For
    /// example, if there's some texture `T` used only by some texture view
    /// `TV`, then if `TV` can be freed, `T` gets added to the free lists too.
    ///
    /// Since `wgpu-core` resource ownership patterns are acyclic, we can visit
    /// each type that can be owned after all types that could possibly own
    /// it. This way, we can detect all free-able objects in a single pass,
    /// simply by starting with types that are roots of the ownership DAG (like
    /// render bundles) and working our way towards leaf types (like buffers).
    ///
    /// [`Device`]: super::Device
    /// [`self.suspected_resources`]: LifetimeTracker::suspected_resources
    /// [`last_resources`]: ActiveSubmission::last_resources
    /// [`self.active`]: LifetimeTracker::active
    /// [`triage_submissions`]: LifetimeTracker::triage_submissions
    /// [`self.free_resources`]: LifetimeTracker::free_resources
    pub(crate) fn triage_suspected<G: GlobalIdentityHandlerFactory>(
        &mut self,
        hub: &Hub<A, G>,
        trackers: &Mutex<Tracker<A>>,
        #[cfg(feature = "trace")] mut trace: Option<&mut trace::Trace>,
    ) {
        profiling::scope!("triage_suspected");

        if !self.suspected_resources.render_bundles.is_empty() {
            let mut trackers = trackers.lock();

            while let Some(bundle) = self.suspected_resources.render_bundles.pop() {
                let id = bundle.info.id();
                if trackers.bundles.remove_abandoned(id) {
                    log::debug!("Bundle {:?} will be destroyed", id);
                    #[cfg(feature = "trace")]
                    if let Some(ref mut t) = trace {
                        t.add(trace::Action::DestroyRenderBundle(id.0));
                    }

                    if let Some(res) = hub.render_bundles.unregister(id.0) {
                        self.suspected_resources.add_render_bundle_scope(&res.used);
                    }
                }
            }
        }

        if !self.suspected_resources.bind_groups.is_empty() {
            let mut trackers = trackers.lock();

            while let Some(resource) = self.suspected_resources.bind_groups.pop() {
                let id = resource.info.id();
                if trackers.bind_groups.remove_abandoned(id) {
                    log::debug!("Bind group {:?} will be destroyed", id);
                    #[cfg(feature = "trace")]
                    if let Some(ref mut t) = trace {
                        t.add(trace::Action::DestroyBindGroup(id.0));
                    }

                    if let Some(res) = hub.bind_groups.unregister(id.0) {
                        self.suspected_resources.add_bind_group_states(&res.used);
                        let bind_group_layout =
                            hub.bind_group_layouts.get(res.layout_id.0).unwrap();
                        self.suspected_resources
                            .bind_group_layouts
                            .push(bind_group_layout);

                        let submit_index = res.info.submission_index();
                        self.active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                            .bind_groups
                            .push(res);
                    }
                }
            }
        }

        if !self.suspected_resources.texture_views.is_empty() {
            let mut trackers = trackers.lock();

            let mut list = mem::take(&mut self.suspected_resources.texture_views);
            for texture_view in list.drain(..) {
                let id = texture_view.info.id();
                if trackers.views.remove_abandoned(id) {
                    log::debug!("Texture view {:?} will be destroyed", id);
                    #[cfg(feature = "trace")]
                    if let Some(ref mut t) = trace {
                        t.add(trace::Action::DestroyTextureView(id.0));
                    }

                    if let Some(res) = hub.texture_views.unregister(id.0) {
                        if let Some(parent_texture) = res.parent.as_ref() {
                            self.suspected_resources
                                .textures
                                .push(parent_texture.clone());
                        }
                        let submit_index = res.info.submission_index();
                        self.active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                            .texture_views
                            .push(res);
                    }
                }
            }
            self.suspected_resources.texture_views = list;
        }

        if !self.suspected_resources.textures.is_empty() {
            let mut trackers = trackers.lock();

            for texture in self.suspected_resources.textures.drain(..) {
                let id = texture.info.id();
                if trackers.textures.remove_abandoned(id) {
                    log::debug!("Texture {:?} will be destroyed", id);
                    #[cfg(feature = "trace")]
                    if let Some(ref mut t) = trace {
                        t.add(trace::Action::DestroyTexture(id.0));
                    }

                    if let Some(res) = hub.textures.unregister(id.0) {
                        let submit_index = res.info.submission_index();
                        let non_referenced_resources = self
                            .active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources);

                        if let &resource::TextureClearMode::RenderPass {
                            ref clear_views, ..
                        } = &*res.clear_mode.read()
                        {
                            non_referenced_resources
                                .texture_views
                                .extend(clear_views.iter().cloned().into_iter());
                        }
                        non_referenced_resources.textures.push(res);
                    }
                }
            }
        }

        if !self.suspected_resources.samplers.is_empty() {
            let mut trackers = trackers.lock();

            for sampler in self.suspected_resources.samplers.drain(..) {
                let id = sampler.info.id();
                if trackers.samplers.remove_abandoned(id) {
                    log::debug!("Sampler {:?} will be destroyed", id);
                    #[cfg(feature = "trace")]
                    if let Some(ref mut t) = trace {
                        t.add(trace::Action::DestroySampler(id.0));
                    }

                    if let Some(res) = hub.samplers.unregister(id.0) {
                        let submit_index = res.info.submission_index();
                        self.active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                            .samplers
                            .push(res);
                    }
                }
            }
        }

        if !self.suspected_resources.buffers.is_empty() {
            let mut trackers = trackers.lock();

            for buffer in self.suspected_resources.buffers.drain(..) {
                let id = buffer.info.id();
                if trackers.buffers.remove_abandoned(id) {
                    log::debug!("Buffer {:?} will be destroyed", id);
                    #[cfg(feature = "trace")]
                    if let Some(ref mut t) = trace {
                        t.add(trace::Action::DestroyBuffer(id.0));
                    }

                    if let Some(res) = hub.buffers.unregister(id.0) {
                        let submit_index = res.info.submission_index();
                        if let resource::BufferMapState::Init {
                            ref stage_buffer, ..
                        } = *res.map_state.lock()
                        {
                            self.free_resources.buffers.push(stage_buffer.clone());
                        }
                        self.active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                            .buffers
                            .push(res);
                    }
                }
            }
        }

        if !self.suspected_resources.compute_pipelines.is_empty() {
            let mut trackers = trackers.lock();

            for compute_pipeline in self.suspected_resources.compute_pipelines.drain(..) {
                let id = compute_pipeline.info.id();
                if trackers.compute_pipelines.remove_abandoned(id) {
                    log::debug!("Compute pipeline {:?} will be destroyed", id);
                    #[cfg(feature = "trace")]
                    if let Some(ref mut t) = trace {
                        t.add(trace::Action::DestroyComputePipeline(id.0));
                    }

                    if let Some(res) = hub.compute_pipelines.unregister(id.0) {
                        let submit_index = res.info.submission_index();
                        self.active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                            .compute_pipes
                            .push(res);
                    }
                }
            }
        }

        if !self.suspected_resources.render_pipelines.is_empty() {
            let mut trackers = trackers.lock();

            for render_pipeline in self.suspected_resources.render_pipelines.drain(..) {
                let id = render_pipeline.info.id();
                if trackers.render_pipelines.remove_abandoned(id) {
                    log::debug!("Render pipeline {:?} will be destroyed", id);
                    #[cfg(feature = "trace")]
                    if let Some(ref mut t) = trace {
                        t.add(trace::Action::DestroyRenderPipeline(id.0));
                    }

                    if let Some(res) = hub.render_pipelines.unregister(id.0) {
                        let submit_index = res.info.submission_index();
                        self.active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                            .render_pipes
                            .push(res);
                    }
                }
            }
        }

        if !self.suspected_resources.pipeline_layouts.is_empty() {
            let mut pipeline_layouts_locked = hub.pipeline_layouts.write();

            for pipeline_layout in self.suspected_resources.pipeline_layouts.drain(..) {
                let id = pipeline_layout.info.id();
                //Note: this has to happen after all the suspected pipelines are destroyed
                if pipeline_layouts_locked.is_unique(id.0).unwrap() {
                    log::debug!("Pipeline layout {:?} will be destroyed", id);
                    #[cfg(feature = "trace")]
                    if let Some(ref mut t) = trace {
                        t.add(trace::Action::DestroyPipelineLayout(id.0));
                    }

                    if let Some(lay) = hub
                        .pipeline_layouts
                        .unregister_locked(id.0, &mut *pipeline_layouts_locked)
                    {
                        for bgl_id in &lay.bind_group_layout_ids {
                            let bgl = hub.bind_group_layouts.get(bgl_id.0).unwrap();
                            self.suspected_resources.bind_group_layouts.push(bgl);
                        }
                        self.free_resources.pipeline_layouts.push(lay);
                    }
                }
            }
        }

        if !self.suspected_resources.bind_group_layouts.is_empty() {
            let mut bind_group_layouts_locked = hub.bind_group_layouts.write();

            for bgl in self.suspected_resources.bind_group_layouts.drain(..) {
                let id = bgl.info().id();
                //Note: this has to happen after all the suspected pipelines are destroyed
                //Note: nothing else can bump the refcount since the guard is locked exclusively
                //Note: same BGL can appear multiple times in the list, but only the last
                // encounter could drop the refcount to 0.
                if bind_group_layouts_locked.is_unique(id.0).unwrap() {
                    log::debug!("Bind group layout {:?} will be destroyed", id);
                    #[cfg(feature = "trace")]
                    if let Some(ref mut t) = trace {
                        t.add(trace::Action::DestroyBindGroupLayout(id.0));
                    }
                    if let Some(lay) = hub
                        .bind_group_layouts
                        .unregister_locked(id.0, &mut *bind_group_layouts_locked)
                    {
                        self.free_resources.bind_group_layouts.push(lay);
                    }
                }
            }
        }

        if !self.suspected_resources.query_sets.is_empty() {
            let mut trackers = trackers.lock();

            for query_set in self.suspected_resources.query_sets.drain(..) {
                let id = query_set.info.id();
                if trackers.query_sets.remove_abandoned(id) {
                    log::debug!("Query set {:?} will be destroyed", id);
                    // #[cfg(feature = "trace")]
                    // trace.map(|t| t.add(trace::Action::DestroyComputePipeline(id.0)));
                    if let Some(res) = hub.query_sets.unregister(id.0) {
                        let submit_index = res.info.submission_index();
                        self.active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                            .query_sets
                            .push(res);
                    }
                }
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
            let submit_index = buffer.info.submission_index();
            log::trace!(
                "Mapping of {:?} at submission {:?} gets assigned to active {:?}",
                buffer.info.id(),
                submit_index,
                self.active.iter().position(|a| a.index == submit_index)
            );

            self.active
                .iter_mut()
                .find(|a| a.index == submit_index)
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
    pub(crate) fn handle_mapping<G: GlobalIdentityHandlerFactory>(
        &mut self,
        hub: &Hub<A, G>,
        raw: &A::Device,
        trackers: &Mutex<Tracker<A>>,
    ) -> Vec<super::BufferMapPendingClosure> {
        if self.ready_to_map.is_empty() {
            return Vec::new();
        }
        let mut pending_callbacks: Vec<super::BufferMapPendingClosure> =
            Vec::with_capacity(self.ready_to_map.len());
        let mut trackers = trackers.lock();
        for buffer in self.ready_to_map.drain(..) {
            let buffer_id = buffer.info.id();
            if trackers.buffers.remove_abandoned(buffer_id) {
                *buffer.map_state.lock() = resource::BufferMapState::Idle;
                log::debug!("Mapping request is dropped because the buffer is destroyed.");
                if let Some(buf) = hub.buffers.unregister(buffer_id.0) {
                    self.free_resources.buffers.push(buf);
                }
            } else {
                let mapping = match std::mem::replace(
                    &mut *buffer.map_state.lock(),
                    resource::BufferMapState::Idle,
                ) {
                    resource::BufferMapState::Waiting(pending_mapping) => pending_mapping,
                    // Mapping cancelled
                    resource::BufferMapState::Idle => continue,
                    // Mapping queued at least twice by map -> unmap -> map
                    // and was already successfully mapped below
                    active @ resource::BufferMapState::Active { .. } => {
                        *buffer.map_state.lock() = active;
                        continue;
                    }
                    _ => panic!("No pending mapping."),
                };
                let status = if mapping.range.start != mapping.range.end {
                    log::debug!("Buffer {:?} map state -> Active", buffer_id);
                    let host = mapping.op.host;
                    let size = mapping.range.end - mapping.range.start;
                    match super::map_buffer(raw, &buffer, mapping.range.start, size, host) {
                        Ok(ptr) => {
                            *buffer.map_state.lock() = resource::BufferMapState::Active {
                                ptr,
                                range: mapping.range.start..mapping.range.start + size,
                                host,
                            };
                            Ok(())
                        }
                        Err(e) => {
                            log::error!("Mapping failed {:?}", e);
                            Err(e)
                        }
                    }
                } else {
                    *buffer.map_state.lock() = resource::BufferMapState::Active {
                        ptr: std::ptr::NonNull::dangling(),
                        range: mapping.range,
                        host: mapping.op.host,
                    };
                    Ok(())
                };
                pending_callbacks.push((mapping.op, status));
            }
        }
        pending_callbacks
    }
}
