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
    identity::GlobalIdentityHandlerFactory,
    pipeline::{ComputePipeline, RenderPipeline},
    resource::{self, Buffer, QuerySet, Resource, Sampler, Texture, TextureView},
    track::Tracker,
    SubmissionIndex,
};
use smallvec::SmallVec;

use parking_lot::Mutex;
use thiserror::Error;

use std::{collections::HashMap, sync::Arc};

/// A struct that keeps lists of resources that are no longer needed by the user.
pub(crate) struct SuspectedResources<A: HalApi> {
    pub(crate) buffers: HashMap<id::BufferId, Arc<Buffer<A>>>,
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

impl<A: HalApi> SuspectedResources<A> {
    pub(crate) fn new() -> Self {
        Self {
            buffers: HashMap::new(),
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
        other.buffers.iter().for_each(|(id, v)| {
            self.buffers.insert(*id, v.clone());
        });
        other.textures.iter().for_each(|(id, v)| {
            self.textures.insert(*id, v.clone());
        });
        other.texture_views.iter().for_each(|(id, v)| {
            self.texture_views.insert(*id, v.clone());
        });
        other.samplers.iter().for_each(|(id, v)| {
            self.samplers.insert(*id, v.clone());
        });
        other.bind_groups.iter().for_each(|(id, v)| {
            self.bind_groups.insert(*id, v.clone());
        });
        other.compute_pipelines.iter().for_each(|(id, v)| {
            self.compute_pipelines.insert(*id, v.clone());
        });
        other.render_pipelines.iter().for_each(|(id, v)| {
            self.render_pipelines.insert(*id, v.clone());
        });
        other.bind_group_layouts.iter().for_each(|(id, v)| {
            self.bind_group_layouts.insert(*id, v.clone());
        });
        other.pipeline_layouts.iter().for_each(|(id, v)| {
            self.pipeline_layouts.insert(*id, v.clone());
        });
        other.render_bundles.iter().for_each(|(id, v)| {
            self.render_bundles.insert(*id, v.clone());
        });
        other.query_sets.iter().for_each(|(id, v)| {
            self.query_sets.insert(*id, v.clone());
        });
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
            suspected_resources: SuspectedResources::new(),
            active: Vec::new(),
            free_resources: NonReferencedResources::new(),
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
        for v in self.future_suspected_buffers.drain(..).take(1) {
            self.suspected_resources
                .buffers
                .insert(v.as_info().id().0, v);
        }
        for v in self.future_suspected_textures.drain(..).take(1) {
            self.suspected_resources
                .textures
                .insert(v.as_info().id().0, v);
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
    fn triage_suspected_render_bundles<G, F>(
        &mut self,
        hub: &Hub<A, G>,
        trackers: &Mutex<Tracker<A>>,
        mut f: F,
    ) -> &mut Self
    where
        G: GlobalIdentityHandlerFactory,
        F: FnMut(&id::RenderBundleId),
    {
        self.suspected_resources
            .render_bundles
            .retain(|bundle_id, bundle| {
                let id = bundle.info.id();
                let is_removed = {
                    let mut trackers = trackers.lock();
                    trackers.bundles.remove_abandoned(id)
                };
                if is_removed {
                    log::info!("Bundle {:?} is removed from registry", id);
                    f(bundle_id);

                    if let Some(res) = hub.render_bundles.unregister(id.0) {
                        for v in res.used.buffers.used_resources() {
                            self.suspected_resources
                                .buffers
                                .insert(v.as_info().id().0, v.clone());
                        }
                        for v in res.used.textures.used_resources() {
                            self.suspected_resources
                                .textures
                                .insert(v.as_info().id().0, v.clone());
                        }
                        for v in res.used.bind_groups.used_resources() {
                            self.suspected_resources
                                .bind_groups
                                .insert(v.as_info().id().0, v.clone());
                        }
                        for v in res.used.render_pipelines.used_resources() {
                            self.suspected_resources
                                .render_pipelines
                                .insert(v.as_info().id().0, v.clone());
                        }
                        for v in res.used.query_sets.used_resources() {
                            self.suspected_resources
                                .query_sets
                                .insert(v.as_info().id().0, v.clone());
                        }
                    }
                }
                !is_removed
            });
        self
    }

    fn triage_suspected_bind_groups<G, F>(
        &mut self,
        hub: &Hub<A, G>,
        trackers: &Mutex<Tracker<A>>,
        mut f: F,
    ) -> &mut Self
    where
        G: GlobalIdentityHandlerFactory,
        F: FnMut(&id::BindGroupId),
    {
        self.suspected_resources
            .bind_groups
            .retain(|bind_group_id, bind_group| {
                let id = bind_group.info.id();
                let is_removed = {
                    let mut trackers = trackers.lock();
                    trackers.bind_groups.remove_abandoned(id)
                };
                if is_removed {
                    log::info!("BindGroup {:?} is removed from registry", id);
                    f(bind_group_id);

                    if let Some(res) = hub.bind_groups.unregister(id.0) {
                        for v in res.used.buffers.used_resources() {
                            self.suspected_resources
                                .buffers
                                .insert(v.as_info().id().0, v.clone());
                        }
                        for v in res.used.textures.used_resources() {
                            self.suspected_resources
                                .textures
                                .insert(v.as_info().id().0, v.clone());
                        }
                        for v in res.used.views.used_resources() {
                            self.suspected_resources
                                .texture_views
                                .insert(v.as_info().id().0, v.clone());
                        }
                        for v in res.used.samplers.used_resources() {
                            self.suspected_resources
                                .samplers
                                .insert(v.as_info().id().0, v.clone());
                        }

                        self.suspected_resources
                            .bind_group_layouts
                            .insert(res.layout.as_info().id().0, res.layout.clone());

                        let submit_index = res.info.submission_index();
                        self.active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                            .bind_groups
                            .push(res);
                    }
                }
                !is_removed
            });
        self
    }

    fn triage_suspected_texture_views<G, F>(
        &mut self,
        hub: &Hub<A, G>,
        trackers: &Mutex<Tracker<A>>,
        mut f: F,
    ) -> &mut Self
    where
        G: GlobalIdentityHandlerFactory,
        F: FnMut(&id::TextureViewId),
    {
        self.suspected_resources
            .texture_views
            .retain(|view_id, view| {
                let id = view.info.id();
                let is_removed = {
                    let mut trackers = trackers.lock();
                    trackers.views.remove_abandoned(id)
                };
                if is_removed {
                    log::info!("TextureView {:?} is removed from registry", id);
                    f(view_id);

                    if let Some(res) = hub.texture_views.unregister(id.0) {
                        if let Some(parent_texture) = res.parent.as_ref() {
                            self.suspected_resources
                                .textures
                                .insert(parent_texture.as_info().id().0, parent_texture.clone());
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
                !is_removed
            });
        self
    }

    fn triage_suspected_textures<G, F>(
        &mut self,
        hub: &Hub<A, G>,
        trackers: &Mutex<Tracker<A>>,
        mut f: F,
    ) -> &mut Self
    where
        G: GlobalIdentityHandlerFactory,
        F: FnMut(&id::TextureId),
    {
        self.suspected_resources
            .textures
            .retain(|texture_id, texture| {
                let id = texture.info.id();
                let is_removed = {
                    let mut trackers = trackers.lock();
                    trackers.textures.remove_abandoned(id)
                };
                if is_removed {
                    log::info!("Texture {:?} is removed from registry", id);
                    f(texture_id);

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
                                .extend(clear_views.iter().cloned());
                        }
                        non_referenced_resources.textures.push(res);
                    }
                }
                !is_removed
            });
        self
    }

    fn triage_suspected_samplers<G, F>(
        &mut self,
        hub: &Hub<A, G>,
        trackers: &Mutex<Tracker<A>>,
        mut f: F,
    ) -> &mut Self
    where
        G: GlobalIdentityHandlerFactory,
        F: FnMut(&id::SamplerId),
    {
        self.suspected_resources
            .samplers
            .retain(|sampler_id, sampler| {
                let id = sampler.info.id();
                let is_removed = {
                    let mut trackers = trackers.lock();
                    trackers.samplers.remove_abandoned(id)
                };
                if is_removed {
                    log::info!("Sampler {:?} is removed from registry", id);
                    f(sampler_id);

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
                !is_removed
            });
        self
    }

    fn triage_suspected_buffers<G, F>(
        &mut self,
        hub: &Hub<A, G>,
        trackers: &Mutex<Tracker<A>>,
        mut f: F,
    ) -> &mut Self
    where
        G: GlobalIdentityHandlerFactory,
        F: FnMut(&id::BufferId),
    {
        self.suspected_resources
            .buffers
            .retain(|buffer_id, buffer| {
                let id = buffer.info.id();
                let is_removed = {
                    let mut trackers = trackers.lock();
                    trackers.buffers.remove_abandoned(id)
                };
                if is_removed {
                    log::info!("Buffer {:?} is removed from registry", id);
                    f(buffer_id);

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
                !is_removed
            });
        self
    }

    fn triage_suspected_compute_pipelines<G, F>(
        &mut self,
        hub: &Hub<A, G>,
        trackers: &Mutex<Tracker<A>>,
        mut f: F,
    ) -> &mut Self
    where
        G: GlobalIdentityHandlerFactory,
        F: FnMut(&id::ComputePipelineId),
    {
        self.suspected_resources.compute_pipelines.retain(
            |compute_pipeline_id, compute_pipeline| {
                let id = compute_pipeline.info.id();
                let is_removed = {
                    let mut trackers = trackers.lock();
                    trackers.compute_pipelines.remove_abandoned(id)
                };
                if is_removed {
                    log::info!("ComputePipeline {:?} is removed from registry", id);
                    f(compute_pipeline_id);

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
                !is_removed
            },
        );
        self
    }

    fn triage_suspected_render_pipelines<G, F>(
        &mut self,
        hub: &Hub<A, G>,
        trackers: &Mutex<Tracker<A>>,
        mut f: F,
    ) -> &mut Self
    where
        G: GlobalIdentityHandlerFactory,
        F: FnMut(&id::RenderPipelineId),
    {
        self.suspected_resources
            .render_pipelines
            .retain(|render_pipeline_id, render_pipeline| {
                let id = render_pipeline.info.id();
                let is_removed = {
                    let mut trackers = trackers.lock();
                    trackers.render_pipelines.remove_abandoned(id)
                };
                if is_removed {
                    log::info!("RenderPipeline {:?} is removed from registry", id);
                    f(render_pipeline_id);

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
                !is_removed
            });
        self
    }

    fn triage_suspected_pipeline_layouts<G, F>(&mut self, hub: &Hub<A, G>, mut f: F) -> &mut Self
    where
        G: GlobalIdentityHandlerFactory,
        F: FnMut(&id::PipelineLayoutId),
    {
        let mut pipeline_layouts_locked = hub.pipeline_layouts.write();
        self.suspected_resources
            .pipeline_layouts
            .retain(|pipeline_layout_id, pipeline_layout| {
                let id = pipeline_layout.info.id();
                //Note: this has to happen after all the suspected pipelines are destroyed
                if pipeline_layouts_locked.is_unique(id.0).unwrap() {
                    log::debug!("PipelineLayout {:?} will be removed from registry", id);
                    f(pipeline_layout_id);

                    if let Some(lay) = hub
                        .pipeline_layouts
                        .unregister_locked(id.0, &mut *pipeline_layouts_locked)
                    {
                        for bgl in &lay.bind_group_layouts {
                            self.suspected_resources
                                .bind_group_layouts
                                .insert(bgl.as_info().id().0, bgl.clone());
                        }
                        self.free_resources.pipeline_layouts.push(lay);
                    }
                    return false;
                }
                true
            });
        self
    }

    fn triage_suspected_bind_group_layouts<G, F>(&mut self, hub: &Hub<A, G>, mut f: F) -> &mut Self
    where
        G: GlobalIdentityHandlerFactory,
        F: FnMut(&id::BindGroupLayoutId),
    {
        let mut bind_group_layouts_locked = hub.bind_group_layouts.write();

        self.suspected_resources.bind_group_layouts.retain(
            |bind_group_layout_id, bind_group_layout| {
                let id = bind_group_layout.info.id();
                //Note: this has to happen after all the suspected pipelines are destroyed
                //Note: nothing else can bump the refcount since the guard is locked exclusively
                //Note: same BGL can appear multiple times in the list, but only the last
                // encounter could drop the refcount to 0.

                //Note: this has to happen after all the suspected pipelines are destroyed
                if bind_group_layouts_locked.is_unique(id.0).unwrap() {
                    // If This layout points to a compatible one, go over the latter
                    // to decrement the ref count and potentially destroy it.
                    //bgl_to_check = bind_group_layout.compatible_layout;

                    log::debug!(
                        "BindGroupLayout {:?} will be removed from registry",
                        bind_group_layout_id
                    );
                    f(bind_group_layout_id);

                    if let Some(lay) = hub
                        .bind_group_layouts
                        .unregister_locked(*bind_group_layout_id, &mut *bind_group_layouts_locked)
                    {
                        self.free_resources.bind_group_layouts.push(lay);
                    }
                    return false;
                }
                true
            },
        );
        self
    }

    fn triage_suspected_query_sets<G>(
        &mut self,
        hub: &Hub<A, G>,
        trackers: &Mutex<Tracker<A>>,
    ) -> &mut Self
    where
        G: GlobalIdentityHandlerFactory,
    {
        self.suspected_resources
            .query_sets
            .retain(|_query_set_id, query_set| {
                let id = query_set.info.id();
                let is_removed = {
                    let mut trackers = trackers.lock();
                    trackers.query_sets.remove_abandoned(id)
                };
                if is_removed {
                    log::info!("QuerySet {:?} is removed from registry", id);
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
                !is_removed
            });
        self
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
    pub(crate) fn triage_suspected<G: GlobalIdentityHandlerFactory>(
        &mut self,
        hub: &Hub<A, G>,
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
        self.triage_suspected_bind_groups(hub, trackers, |_id| {
            #[cfg(feature = "trace")]
            if let Some(ref mut t) = trace {
                t.add(trace::Action::DestroyBindGroup(*_id));
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
        self.triage_suspected_compute_pipelines(hub, trackers, |_id| {
            #[cfg(feature = "trace")]
            if let Some(ref mut t) = trace {
                t.add(trace::Action::DestroyComputePipeline(*_id));
            }
        });
        self.triage_suspected_render_pipelines(hub, trackers, |_id| {
            #[cfg(feature = "trace")]
            if let Some(ref mut t) = trace {
                t.add(trace::Action::DestroyRenderPipeline(*_id));
            }
        });
        self.triage_suspected_pipeline_layouts(hub, |_id| {
            #[cfg(feature = "trace")]
            if let Some(ref mut t) = trace {
                t.add(trace::Action::DestroyPipelineLayout(*_id));
            }
        });
        self.triage_suspected_bind_group_layouts(hub, |_id| {
            #[cfg(feature = "trace")]
            if let Some(ref mut t) = trace {
                t.add(trace::Action::DestroyBindGroupLayout(*_id));
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

        for buffer in self.ready_to_map.drain(..) {
            let buffer_id = buffer.info.id();
            let is_removed = {
                let mut trackers = trackers.lock();
                trackers.buffers.remove_abandoned(buffer_id)
            };
            if is_removed {
                *buffer.map_state.lock() = resource::BufferMapState::Idle;
                log::info!("Buffer {:?} is removed from registry", buffer_id);
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
