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
    id::{self},
    pipeline::{ComputePipeline, RenderPipeline},
    resource::{self, Buffer, QuerySet, Resource, Sampler, StagingBuffer, Texture, TextureView},
    track::Tracker,
    SubmissionIndex,
};
use smallvec::SmallVec;

use parking_lot::Mutex;
use thiserror::Error;

use std::{collections::HashMap, sync::Arc};

/// A struct that keeps lists of resources that are no longer needed by the user.
pub(crate) struct ResourceMaps<A: HalApi> {
    pub(crate) buffers: HashMap<id::BufferId, Arc<Buffer<A>>>,
    pub(crate) staging_buffers: HashMap<id::StagingBufferId, Arc<StagingBuffer<A>>>,
    pub(crate) textures: HashMap<id::TextureId, Arc<Texture<A>>>,
    pub(crate) texture_views: HashMap<id::TextureViewId, Arc<TextureView<A>>>,
    pub(crate) samplers: HashMap<id::SamplerId, Arc<Sampler<A>>>,
    pub(crate) bind_groups: HashMap<id::BindGroupId, Arc<BindGroup<A>>>,
    pub(crate) compute_pipelines: HashMap<id::ComputePipelineId, Arc<ComputePipeline<A>>>,
    pub(crate) render_pipelines: HashMap<id::RenderPipelineId, Arc<RenderPipeline<A>>>,
    pub(crate) bind_group_layouts: HashMap<id::BindGroupLayoutId, Arc<BindGroupLayout<A>>>,
    pub(crate) pipeline_layouts: HashMap<id::PipelineLayoutId, Arc<PipelineLayout<A>>>,
    pub(crate) render_bundles: HashMap<id::RenderBundleId, Arc<RenderBundle<A>>>,
    pub(crate) query_sets: HashMap<id::QuerySetId, Arc<QuerySet<A>>>,
}

impl<A: HalApi> ResourceMaps<A> {
    pub(crate) fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            staging_buffers: HashMap::new(),
            textures: HashMap::new(),
            texture_views: HashMap::new(),
            samplers: HashMap::new(),
            bind_groups: HashMap::new(),
            compute_pipelines: HashMap::new(),
            render_pipelines: HashMap::new(),
            bind_group_layouts: HashMap::new(),
            pipeline_layouts: HashMap::new(),
            render_bundles: HashMap::new(),
            query_sets: HashMap::new(),
        }
    }
    pub(crate) fn clear(&mut self) {
        self.buffers.clear();
        self.staging_buffers.clear();
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

    pub(crate) fn extend(&mut self, other: Self) {
        self.buffers.extend(other.buffers);
        self.staging_buffers.extend(other.staging_buffers);
        self.textures.extend(other.textures);
        self.texture_views.extend(other.texture_views);
        self.samplers.extend(other.samplers);
        self.bind_groups.extend(other.bind_groups);
        self.compute_pipelines.extend(other.compute_pipelines);
        self.render_pipelines.extend(other.render_pipelines);
        self.bind_group_layouts.extend(other.bind_group_layouts);
        self.pipeline_layouts.extend(other.pipeline_layouts);
        self.query_sets.extend(other.query_sets);
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
    last_resources: ResourceMaps<A>,

    /// Buffers to be mapped once this submission has completed.
    mapped: Vec<Arc<Buffer<A>>>,

    encoders: Vec<EncoderInFlight<A>>,

    /// List of queue "on_submitted_work_done" closures to be called once this
    /// submission has completed.
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
///         buffers, collecting a list of notification closures to call. But any
///         buffers that were dropped by the user get moved to
///         `self.free_resources`.
///
///     4)  `cleanup` frees everything in `free_resources`.
///
/// Only calling `Global::buffer_map_async` clones a new `Arc` for the
/// buffer. This new `Arc` is only dropped by `handle_mapping`.
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
    pub suspected_resources: ResourceMaps<A>,

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
    free_resources: ResourceMaps<A>,

    /// Buffers the user has asked us to map, and which are not used by any
    /// queue submission still in flight.
    ready_to_map: Vec<Arc<Buffer<A>>>,

    /// Queue "on_submitted_work_done" closures that were initiated for while there is no
    /// currently pending submissions. These cannot be immeidately invoked as they
    /// must happen _after_ all mapped buffer callbacks are mapped, so we defer them
    /// here until the next time the device is maintained.
    work_done_closures: SmallVec<[SubmittedWorkDoneClosure; 1]>,
}

impl<A: HalApi> LifetimeTracker<A> {
    pub fn new() -> Self {
        Self {
            mapped: Vec::new(),
            future_suspected_buffers: Vec::new(),
            future_suspected_textures: Vec::new(),
            suspected_resources: ResourceMaps::new(),
            active: Vec::new(),
            free_resources: ResourceMaps::new(),
            ready_to_map: Vec::new(),
            work_done_closures: SmallVec::new(),
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
        let mut last_resources = ResourceMaps::new();
        for res in temp_resources {
            match res {
                TempResource::Buffer(raw) => {
                    last_resources.buffers.insert(raw.as_info().id(), raw);
                }
                TempResource::StagingBuffer(raw) => {
                    last_resources
                        .staging_buffers
                        .insert(raw.as_info().id(), raw);
                }
                TempResource::Texture(raw, views) => {
                    last_resources.textures.insert(raw.as_info().id(), raw);
                    views.into_iter().for_each(|v| {
                        last_resources.texture_views.insert(v.as_info().id(), v);
                    });
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
        for v in self.future_suspected_buffers.drain(..).take(1) {
            self.suspected_resources.buffers.insert(v.as_info().id(), v);
        }
        for v in self.future_suspected_textures.drain(..).take(1) {
            self.suspected_resources
                .textures
                .insert(v.as_info().id(), v);
        }
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

        let mut work_done_closures: SmallVec<_> = self.work_done_closures.drain(..).collect();
        for a in self.active.drain(..done_count) {
            log::info!("Active submission {} is done", a.index);
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
        self.free_resources.clear();
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
            TempResource::Buffer(raw) => {
                resources.buffers.insert(raw.as_info().id(), raw);
            }
            TempResource::StagingBuffer(raw) => {
                resources.staging_buffers.insert(raw.as_info().id(), raw);
            }
            TempResource::Texture(raw, views) => {
                views.into_iter().for_each(|v| {
                    resources.texture_views.insert(v.as_info().id(), v);
                });
                resources.textures.insert(raw.as_info().id(), raw);
            }
        }
    }

    pub fn add_work_done_closure(&mut self, closure: SubmittedWorkDoneClosure) {
        match self.active.last_mut() {
            Some(active) => {
                active.work_done_closures.push(closure);
            }
            // We must defer the closure until all previously occuring map_async closures
            // have fired. This is required by the spec.
            None => {
                self.work_done_closures.push(closure);
            }
        }
    }
}

impl<A: HalApi> LifetimeTracker<A> {
    fn triage_suspected_render_bundles<F>(
        &mut self,
        hub: &Hub<A>,
        trackers: &Mutex<Tracker<A>>,
        mut f: F,
    ) -> &mut Self
    where
        F: FnMut(&id::RenderBundleId),
    {
        self.suspected_resources
            .render_bundles
            .retain(|&bundle_id, bundle| {
                let is_removed = {
                    let mut trackers = trackers.lock();
                    trackers
                        .bundles
                        .remove_abandoned(bundle_id, hub.render_bundles.contains(bundle_id))
                };
                if is_removed {
                    log::info!("Bundle {:?} is not tracked anymore", bundle_id);
                    f(&bundle_id);

                    for v in bundle.used.buffers.used_resources() {
                        self.suspected_resources
                            .buffers
                            .insert(v.as_info().id(), v.clone());
                    }
                    for v in bundle.used.textures.used_resources() {
                        self.suspected_resources
                            .textures
                            .insert(v.as_info().id(), v.clone());
                    }
                    for v in bundle.used.bind_groups.used_resources() {
                        self.suspected_resources
                            .bind_groups
                            .insert(v.as_info().id(), v.clone());
                    }
                    for v in bundle.used.render_pipelines.used_resources() {
                        self.suspected_resources
                            .render_pipelines
                            .insert(v.as_info().id(), v.clone());
                    }
                    for v in bundle.used.query_sets.used_resources() {
                        self.suspected_resources
                            .query_sets
                            .insert(v.as_info().id(), v.clone());
                    }
                }
                !is_removed
            });
        self
    }

    fn triage_suspected_bind_groups<F>(
        &mut self,
        hub: &Hub<A>,
        trackers: &Mutex<Tracker<A>>,
        mut f: F,
    ) -> Vec<u64>
    where
        F: FnMut(&id::BindGroupId),
    {
        let mut submit_indices = Vec::new();
        self.suspected_resources
            .bind_groups
            .retain(|&bind_group_id, bind_group| {
                let is_removed = {
                    let mut trackers = trackers.lock();
                    trackers
                        .bind_groups
                        .remove_abandoned(bind_group_id, hub.bind_groups.contains(bind_group_id))
                };
                if is_removed {
                    log::info!("BindGroup {:?} is not tracked anymore", bind_group_id);
                    f(&bind_group_id);

                    for v in bind_group.used.buffers.used_resources() {
                        self.suspected_resources
                            .buffers
                            .insert(v.as_info().id(), v.clone());
                    }
                    for v in bind_group.used.textures.used_resources() {
                        self.suspected_resources
                            .textures
                            .insert(v.as_info().id(), v.clone());
                    }
                    for v in bind_group.used.views.used_resources() {
                        self.suspected_resources
                            .texture_views
                            .insert(v.as_info().id(), v.clone());
                    }
                    for v in bind_group.used.samplers.used_resources() {
                        self.suspected_resources
                            .samplers
                            .insert(v.as_info().id(), v.clone());
                    }

                    self.suspected_resources
                        .bind_group_layouts
                        .insert(bind_group.layout.as_info().id(), bind_group.layout.clone());

                    let submit_index = bind_group.info.submission_index();
                    if !submit_indices.contains(&submit_index) {
                        submit_indices.push(submit_index);
                    }
                    self.active
                        .iter_mut()
                        .find(|a| a.index == submit_index)
                        .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                        .bind_groups
                        .insert(bind_group_id, bind_group.clone());
                }
                !is_removed
            });
        submit_indices
    }

    fn triage_suspected_texture_views<F>(
        &mut self,
        hub: &Hub<A>,
        trackers: &Mutex<Tracker<A>>,
        mut f: F,
    ) -> Vec<u64>
    where
        F: FnMut(&id::TextureViewId),
    {
        let mut submit_indices = Vec::new();
        self.suspected_resources
            .texture_views
            .retain(|&view_id, view| {
                let is_removed = {
                    let mut trackers = trackers.lock();
                    trackers
                        .views
                        .remove_abandoned(view_id, hub.texture_views.contains(view_id))
                };
                if is_removed {
                    log::info!("TextureView {:?} is not tracked anymore", view_id);
                    f(&view_id);

                    if let Some(parent_texture) = view.parent.as_ref() {
                        self.suspected_resources
                            .textures
                            .insert(parent_texture.as_info().id(), parent_texture.clone());
                    }
                    let submit_index = view.info.submission_index();
                    if !submit_indices.contains(&submit_index) {
                        submit_indices.push(submit_index);
                    }
                    self.active
                        .iter_mut()
                        .find(|a| a.index == submit_index)
                        .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                        .texture_views
                        .insert(view_id, view.clone());
                }
                !is_removed
            });
        submit_indices
    }

    fn triage_suspected_textures<F>(
        &mut self,
        hub: &Hub<A>,
        trackers: &Mutex<Tracker<A>>,
        mut f: F,
    ) -> &mut Self
    where
        F: FnMut(&id::TextureId),
    {
        self.suspected_resources
            .textures
            .retain(|&texture_id, texture| {
                let is_removed = {
                    let mut trackers = trackers.lock();
                    trackers
                        .textures
                        .remove_abandoned(texture_id, hub.textures.contains(texture_id))
                };
                if is_removed {
                    log::info!("Texture {:?} is not tracked anymore", texture_id);
                    f(&texture_id);

                    let submit_index = texture.info.submission_index();
                    let non_referenced_resources = self
                        .active
                        .iter_mut()
                        .find(|a| a.index == submit_index)
                        .map_or(&mut self.free_resources, |a| &mut a.last_resources);

                    if let &resource::TextureClearMode::RenderPass {
                        ref clear_views, ..
                    } = &*texture.clear_mode.read()
                    {
                        clear_views.into_iter().for_each(|v| {
                            non_referenced_resources
                                .texture_views
                                .insert(v.as_info().id(), v.clone());
                        });
                    }
                    non_referenced_resources
                        .textures
                        .insert(texture_id, texture.clone());
                }
                !is_removed
            });
        self
    }

    fn triage_suspected_samplers<F>(
        &mut self,
        hub: &Hub<A>,
        trackers: &Mutex<Tracker<A>>,
        mut f: F,
    ) -> Vec<u64>
    where
        F: FnMut(&id::SamplerId),
    {
        let mut submit_indices = Vec::new();
        self.suspected_resources
            .samplers
            .retain(|&sampler_id, sampler| {
                let is_removed = {
                    let mut trackers = trackers.lock();
                    trackers
                        .samplers
                        .remove_abandoned(sampler_id, hub.samplers.contains(sampler_id))
                };
                if is_removed {
                    log::info!("Sampler {:?} is not tracked anymore", sampler_id);
                    f(&sampler_id);

                    let submit_index = sampler.info.submission_index();
                    if !submit_indices.contains(&submit_index) {
                        submit_indices.push(submit_index);
                    }
                    self.active
                        .iter_mut()
                        .find(|a| a.index == submit_index)
                        .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                        .samplers
                        .insert(sampler_id, sampler.clone());
                }
                !is_removed
            });
        submit_indices
    }

    fn triage_suspected_buffers<F>(
        &mut self,
        hub: &Hub<A>,
        trackers: &Mutex<Tracker<A>>,
        mut f: F,
    ) -> Vec<u64>
    where
        F: FnMut(&id::BufferId),
    {
        let mut submit_indices = Vec::new();
        self.suspected_resources
            .buffers
            .retain(|&buffer_id, buffer| {
                let is_removed = {
                    let mut trackers = trackers.lock();
                    trackers
                        .buffers
                        .remove_abandoned(buffer_id, hub.buffers.contains(buffer_id))
                };
                if is_removed {
                    log::info!("Buffer {:?} is not tracked anymore", buffer_id);
                    f(&buffer_id);

                    let submit_index = buffer.info.submission_index();
                    if !submit_indices.contains(&submit_index) {
                        submit_indices.push(submit_index);
                    }
                    if let resource::BufferMapState::Init {
                        ref stage_buffer, ..
                    } = *buffer.map_state.lock()
                    {
                        self.free_resources
                            .buffers
                            .insert(stage_buffer.as_info().id(), stage_buffer.clone());
                    }
                    self.active
                        .iter_mut()
                        .find(|a| a.index == submit_index)
                        .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                        .buffers
                        .insert(buffer_id, buffer.clone());
                }
                !is_removed
            });
        submit_indices
    }

    fn triage_suspected_compute_pipelines<F>(
        &mut self,
        hub: &Hub<A>,
        trackers: &Mutex<Tracker<A>>,
        mut f: F,
    ) -> Vec<u64>
    where
        F: FnMut(&id::ComputePipelineId),
    {
        let mut submit_indices = Vec::new();
        self.suspected_resources.compute_pipelines.retain(
            |&compute_pipeline_id, compute_pipeline| {
                let is_removed = {
                    let mut trackers = trackers.lock();
                    trackers.compute_pipelines.remove_abandoned(
                        compute_pipeline_id,
                        hub.compute_pipelines.contains(compute_pipeline_id),
                    )
                };
                if is_removed {
                    log::info!(
                        "ComputePipeline {:?} is not tracked anymore",
                        compute_pipeline_id
                    );
                    f(&compute_pipeline_id);

                    self.suspected_resources.pipeline_layouts.insert(
                        compute_pipeline.layout.as_info().id(),
                        compute_pipeline.layout.clone(),
                    );

                    let submit_index = compute_pipeline.info.submission_index();
                    if !submit_indices.contains(&submit_index) {
                        submit_indices.push(submit_index);
                    }
                    self.active
                        .iter_mut()
                        .find(|a| a.index == submit_index)
                        .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                        .compute_pipelines
                        .insert(compute_pipeline_id, compute_pipeline.clone());
                }
                !is_removed
            },
        );
        submit_indices
    }

    fn triage_suspected_render_pipelines<F>(
        &mut self,
        hub: &Hub<A>,
        trackers: &Mutex<Tracker<A>>,
        mut f: F,
    ) -> Vec<u64>
    where
        F: FnMut(&id::RenderPipelineId),
    {
        let mut submit_indices = Vec::new();
        self.suspected_resources
            .render_pipelines
            .retain(|&render_pipeline_id, render_pipeline| {
                let is_removed = {
                    let mut trackers = trackers.lock();
                    trackers.render_pipelines.remove_abandoned(
                        render_pipeline_id,
                        hub.render_pipelines.contains(render_pipeline_id),
                    )
                };
                if is_removed {
                    log::info!(
                        "RenderPipeline {:?} is not tracked anymore",
                        render_pipeline_id
                    );
                    f(&render_pipeline_id);

                    self.suspected_resources.pipeline_layouts.insert(
                        render_pipeline.layout.as_info().id(),
                        render_pipeline.layout.clone(),
                    );

                    let submit_index = render_pipeline.info.submission_index();
                    if !submit_indices.contains(&submit_index) {
                        submit_indices.push(submit_index);
                    }
                    self.active
                        .iter_mut()
                        .find(|a| a.index == submit_index)
                        .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                        .render_pipelines
                        .insert(render_pipeline_id, render_pipeline.clone());
                }
                !is_removed
            });
        submit_indices
    }

    fn triage_suspected_pipeline_layouts<F>(
        &mut self,
        pipeline_submit_indices: &[u64],
        mut f: F,
    ) -> &mut Self
    where
        F: FnMut(&id::PipelineLayoutId),
    {
        self.suspected_resources
            .pipeline_layouts
            .retain(|pipeline_layout_id, pipeline_layout| {
                //Note: this has to happen after all the suspected pipelines are destroyed

                let mut num_ref_in_nonreferenced_resources = 0;
                pipeline_submit_indices.iter().for_each(|submit_index| {
                    let resources = self
                        .active
                        .iter()
                        .find(|a| a.index == *submit_index)
                        .map_or(&self.free_resources, |a| &a.last_resources);

                    resources.compute_pipelines.iter().for_each(|(_id, p)| {
                        if p.layout.as_info().id() == *pipeline_layout_id {
                            num_ref_in_nonreferenced_resources += 1;
                        }
                    });
                    resources.render_pipelines.iter().for_each(|(_id, p)| {
                        if p.layout.as_info().id() == *pipeline_layout_id {
                            num_ref_in_nonreferenced_resources += 1;
                        }
                    });
                });

                if pipeline_layout.ref_count() == (1 + num_ref_in_nonreferenced_resources) {
                    log::debug!(
                        "PipelineLayout {:?} is not tracked anymore",
                        pipeline_layout_id
                    );

                    f(pipeline_layout_id);

                    for bgl in &pipeline_layout.bind_group_layouts {
                        self.suspected_resources
                            .bind_group_layouts
                            .insert(bgl.as_info().id(), bgl.clone());
                    }
                    self.free_resources
                        .pipeline_layouts
                        .insert(*pipeline_layout_id, pipeline_layout.clone());

                    return false;
                } else {
                    log::info!(
                        "PipelineLayout {:?} is still referenced from {}",
                        pipeline_layout_id,
                        pipeline_layout.ref_count()
                    );
                }
                true
            });
        self
    }

    fn triage_suspected_bind_group_layouts<F>(
        &mut self,
        bind_group_submit_indices: &[u64],
        pipeline_submit_indices: &[u64],
        mut f: F,
    ) -> &mut Self
    where
        F: FnMut(&id::BindGroupLayoutId),
    {
        self.suspected_resources.bind_group_layouts.retain(
            |bind_group_layout_id, bind_group_layout| {
                //Note: this has to happen after all the suspected pipelines are destroyed
                //Note: nothing else can bump the refcount since the guard is locked exclusively
                //Note: same BGL can appear multiple times in the list, but only the last
                // encounter could drop the refcount to 0.
                let mut num_ref_in_nonreferenced_resources = 0;
                bind_group_submit_indices.iter().for_each(|submit_index| {
                    let resources = self
                        .active
                        .iter()
                        .find(|a| a.index == *submit_index)
                        .map_or(&self.free_resources, |a| &a.last_resources);

                    resources.bind_groups.iter().for_each(|(_id, b)| {
                        if b.layout.as_info().id() == *bind_group_layout_id {
                            num_ref_in_nonreferenced_resources += 1;
                        }
                    });
                    resources.bind_group_layouts.iter().for_each(|(id, _b)| {
                        if id == bind_group_layout_id {
                            num_ref_in_nonreferenced_resources += 1;
                        }
                    });
                });
                pipeline_submit_indices.iter().for_each(|submit_index| {
                    let resources = self
                        .active
                        .iter()
                        .find(|a| a.index == *submit_index)
                        .map_or(&self.free_resources, |a| &a.last_resources);

                    resources.compute_pipelines.iter().for_each(|(_id, p)| {
                        p.layout.bind_group_layouts.iter().for_each(|b| {
                            if b.as_info().id() == *bind_group_layout_id {
                                num_ref_in_nonreferenced_resources += 1;
                            }
                        });
                    });
                    resources.render_pipelines.iter().for_each(|(_id, p)| {
                        p.layout.bind_group_layouts.iter().for_each(|b| {
                            if b.as_info().id() == *bind_group_layout_id {
                                num_ref_in_nonreferenced_resources += 1;
                            }
                        });
                    });
                    resources.pipeline_layouts.iter().for_each(|(_id, p)| {
                        p.bind_group_layouts.iter().for_each(|b| {
                            if b.as_info().id() == *bind_group_layout_id {
                                num_ref_in_nonreferenced_resources += 1;
                            }
                        });
                    });
                });

                //Note: this has to happen after all the suspected pipelines are destroyed
                if bind_group_layout.ref_count() == (1 + num_ref_in_nonreferenced_resources) {
                    // If This layout points to a compatible one, go over the latter
                    // to decrement the ref count and potentially destroy it.
                    //bgl_to_check = bind_group_layout.compatible_layout;

                    log::debug!(
                        "BindGroupLayout {:?} is not tracked anymore",
                        bind_group_layout_id
                    );
                    f(bind_group_layout_id);

                    self.free_resources
                        .bind_group_layouts
                        .insert(*bind_group_layout_id, bind_group_layout.clone());

                    return false;
                } else {
                    log::info!(
                        "BindGroupLayout {:?} is still referenced from {}",
                        bind_group_layout_id,
                        bind_group_layout.ref_count()
                    );
                }
                true
            },
        );
        self
    }

    fn triage_suspected_query_sets(
        &mut self,
        hub: &Hub<A>,
        trackers: &Mutex<Tracker<A>>,
    ) -> Vec<u64> {
        let mut submit_indices = Vec::new();
        self.suspected_resources
            .query_sets
            .retain(|&query_set_id, query_set| {
                let is_removed = {
                    let mut trackers = trackers.lock();
                    trackers
                        .query_sets
                        .remove_abandoned(query_set_id, hub.query_sets.contains(query_set_id))
                };
                if is_removed {
                    log::info!("QuerySet {:?} is not tracked anymore", query_set_id);
                    // #[cfg(feature = "trace")]
                    // trace.map(|t| t.add(trace::Action::DestroyComputePipeline(id)));

                    let submit_index = query_set.info.submission_index();
                    if !submit_indices.contains(&submit_index) {
                        submit_indices.push(submit_index);
                    }
                    self.active
                        .iter_mut()
                        .find(|a| a.index == submit_index)
                        .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                        .query_sets
                        .insert(query_set_id, query_set.clone());
                }
                !is_removed
            });
        submit_indices
    }

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
    pub(crate) fn triage_suspected(
        &mut self,
        hub: &Hub<A>,
        trackers: &Mutex<Tracker<A>>,
        #[cfg(feature = "trace")] mut trace: Option<&mut trace::Trace>,
    ) {
        profiling::scope!("triage_suspected");

        self.triage_suspected_render_bundles(hub, trackers, |_id| {
            #[cfg(feature = "trace")]
            if let Some(ref mut t) = trace {
                t.add(trace::Action::DestroyRenderBundle(*_id));
            }
        });
        let compute_pipeline_indices =
            self.triage_suspected_compute_pipelines(hub, trackers, |_id| {
                #[cfg(feature = "trace")]
                if let Some(ref mut t) = trace {
                    t.add(trace::Action::DestroyComputePipeline(*_id));
                }
            });
        let render_pipeline_indices =
            self.triage_suspected_render_pipelines(hub, trackers, |_id| {
                #[cfg(feature = "trace")]
                if let Some(ref mut t) = trace {
                    t.add(trace::Action::DestroyRenderPipeline(*_id));
                }
            });
        let mut pipeline_submit_indices = Vec::new();
        pipeline_submit_indices.extend(compute_pipeline_indices);
        pipeline_submit_indices.extend(render_pipeline_indices);
        let bind_group_submit_indices = self.triage_suspected_bind_groups(hub, trackers, |_id| {
            #[cfg(feature = "trace")]
            if let Some(ref mut t) = trace {
                t.add(trace::Action::DestroyBindGroup(*_id));
            }
        });
        self.triage_suspected_pipeline_layouts(&pipeline_submit_indices, |_id| {
            #[cfg(feature = "trace")]
            if let Some(ref mut t) = trace {
                t.add(trace::Action::DestroyPipelineLayout(*_id));
            }
        });
        self.triage_suspected_bind_group_layouts(
            &bind_group_submit_indices,
            &pipeline_submit_indices,
            |_id| {
                #[cfg(feature = "trace")]
                if let Some(ref mut t) = trace {
                    t.add(trace::Action::DestroyBindGroupLayout(*_id));
                }
            },
        );
        self.triage_suspected_samplers(hub, trackers, |_id| {
            #[cfg(feature = "trace")]
            if let Some(ref mut t) = trace {
                t.add(trace::Action::DestroySampler(*_id));
            }
        });
        self.triage_suspected_buffers(hub, trackers, |_id| {
            #[cfg(feature = "trace")]
            if let Some(ref mut t) = trace {
                t.add(trace::Action::DestroyBuffer(*_id));
            }
        });
        self.triage_suspected_texture_views(hub, trackers, |_id| {
            #[cfg(feature = "trace")]
            if let Some(ref mut t) = trace {
                t.add(trace::Action::DestroyTextureView(*_id));
            }
        });
        self.triage_suspected_textures(hub, trackers, |_id| {
            #[cfg(feature = "trace")]
            if let Some(ref mut t) = trace {
                t.add(trace::Action::DestroyTexture(*_id));
            }
        });
        self.triage_suspected_query_sets(hub, trackers);
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
    pub(crate) fn handle_mapping(
        &mut self,
        hub: &Hub<A>,
        raw: &A::Device,
        trackers: &Mutex<Tracker<A>>,
    ) -> Vec<super::BufferMapPendingClosure> {
        if self.ready_to_map.is_empty() {
            return Vec::new();
        }
        let mut pending_callbacks: Vec<super::BufferMapPendingClosure> =
            Vec::with_capacity(self.ready_to_map.len());

        for buffer in self.ready_to_map.drain(..) {
            let buffer_id = buffer.info.id();
            let is_removed = {
                let mut trackers = trackers.lock();
                trackers
                    .buffers
                    .remove_abandoned(buffer_id, hub.buffers.contains(buffer_id))
            };
            if is_removed {
                *buffer.map_state.lock() = resource::BufferMapState::Idle;
                log::info!("Buffer {:?} is not tracked anymore", buffer_id);
                self.free_resources
                    .buffers
                    .insert(buffer_id, buffer.clone());
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
