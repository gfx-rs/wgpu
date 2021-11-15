#[cfg(feature = "trace")]
use crate::device::trace;
use crate::{
    device::{
        queue::{EncoderInFlight, SubmittedWorkDoneClosure, TempResource},
        DeviceError,
    },
    hub::{GlobalIdentityHandlerFactory, HalApi, Hub, Token},
    id, resource,
    track::TrackerSet,
    RefCount, Stored, SubmissionIndex,
};
use smallvec::SmallVec;

use copyless::VecHelper as _;
use hal::Device as _;
use parking_lot::Mutex;
use thiserror::Error;

use std::mem;

/// A struct that keeps lists of resources that are no longer needed by the user.
#[derive(Debug, Default)]
pub(super) struct SuspectedResources {
    pub(super) buffers: Vec<id::Valid<id::BufferId>>,
    pub(super) textures: Vec<id::Valid<id::TextureId>>,
    pub(super) texture_views: Vec<id::Valid<id::TextureViewId>>,
    pub(super) samplers: Vec<id::Valid<id::SamplerId>>,
    pub(super) bind_groups: Vec<id::Valid<id::BindGroupId>>,
    pub(super) compute_pipelines: Vec<id::Valid<id::ComputePipelineId>>,
    pub(super) render_pipelines: Vec<id::Valid<id::RenderPipelineId>>,
    pub(super) bind_group_layouts: Vec<id::Valid<id::BindGroupLayoutId>>,
    pub(super) pipeline_layouts: Vec<Stored<id::PipelineLayoutId>>,
    pub(super) render_bundles: Vec<id::Valid<id::RenderBundleId>>,
    pub(super) query_sets: Vec<id::Valid<id::QuerySetId>>,
}

impl SuspectedResources {
    pub(super) fn clear(&mut self) {
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

    pub(super) fn extend(&mut self, other: &Self) {
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

    pub(super) fn add_trackers(&mut self, trackers: &TrackerSet) {
        self.buffers.extend(trackers.buffers.used());
        self.textures.extend(trackers.textures.used());
        self.texture_views.extend(trackers.views.used());
        self.samplers.extend(trackers.samplers.used());
        self.bind_groups.extend(trackers.bind_groups.used());
        self.compute_pipelines.extend(trackers.compute_pipes.used());
        self.render_pipelines.extend(trackers.render_pipes.used());
        self.render_bundles.extend(trackers.bundles.used());
        self.query_sets.extend(trackers.query_sets.used());
    }
}

/// A struct that keeps lists of resources that are no longer needed.
#[derive(Debug)]
struct NonReferencedResources<A: hal::Api> {
    buffers: Vec<A::Buffer>,
    textures: Vec<A::Texture>,
    texture_views: Vec<A::TextureView>,
    samplers: Vec<A::Sampler>,
    bind_groups: Vec<A::BindGroup>,
    compute_pipes: Vec<A::ComputePipeline>,
    render_pipes: Vec<A::RenderPipeline>,
    bind_group_layouts: Vec<A::BindGroupLayout>,
    pipeline_layouts: Vec<A::PipelineLayout>,
    query_sets: Vec<A::QuerySet>,
}

impl<A: hal::Api> NonReferencedResources<A> {
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

    unsafe fn clean(&mut self, device: &A::Device) {
        if !self.buffers.is_empty() {
            profiling::scope!("destroy_buffers");
            for raw in self.buffers.drain(..) {
                device.destroy_buffer(raw);
            }
        }
        if !self.textures.is_empty() {
            profiling::scope!("destroy_textures");
            for raw in self.textures.drain(..) {
                device.destroy_texture(raw);
            }
        }
        if !self.texture_views.is_empty() {
            profiling::scope!("destroy_texture_views");
            for raw in self.texture_views.drain(..) {
                device.destroy_texture_view(raw);
            }
        }
        if !self.samplers.is_empty() {
            profiling::scope!("destroy_samplers");
            for raw in self.samplers.drain(..) {
                device.destroy_sampler(raw);
            }
        }
        if !self.bind_groups.is_empty() {
            profiling::scope!("destroy_bind_groups");
            for raw in self.bind_groups.drain(..) {
                device.destroy_bind_group(raw);
            }
        }
        if !self.compute_pipes.is_empty() {
            profiling::scope!("destroy_compute_pipelines");
            for raw in self.compute_pipes.drain(..) {
                device.destroy_compute_pipeline(raw);
            }
        }
        if !self.render_pipes.is_empty() {
            profiling::scope!("destroy_render_pipelines");
            for raw in self.render_pipes.drain(..) {
                device.destroy_render_pipeline(raw);
            }
        }
        if !self.bind_group_layouts.is_empty() {
            profiling::scope!("destroy_bind_group_layouts");
            for raw in self.bind_group_layouts.drain(..) {
                device.destroy_bind_group_layout(raw);
            }
        }
        if !self.pipeline_layouts.is_empty() {
            profiling::scope!("destroy_pipeline_layouts");
            for raw in self.pipeline_layouts.drain(..) {
                device.destroy_pipeline_layout(raw);
            }
        }
        if !self.query_sets.is_empty() {
            profiling::scope!("destroy_query_sets");
            for raw in self.query_sets.drain(..) {
                device.destroy_query_set(raw);
            }
        }
    }
}

struct ActiveSubmission<A: hal::Api> {
    index: SubmissionIndex,
    last_resources: NonReferencedResources<A>,
    mapped: Vec<id::Valid<id::BufferId>>,
    encoders: Vec<EncoderInFlight<A>>,
    work_done_closures: SmallVec<[SubmittedWorkDoneClosure; 1]>,
}

#[derive(Clone, Debug, Error)]
pub enum WaitIdleError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("GPU got stuck :(")]
    StuckGpu,
}

/// A struct responsible for tracking resource lifetimes.
///
/// Here is how host mapping is handled:
///   1. When mapping is requested we add the buffer to the life_tracker list of `mapped` buffers.
///   2. When `triage_suspected` is called, it checks the last submission index associated with each of the mapped buffer,
/// and register the buffer with either a submission in flight, or straight into `ready_to_map` vector.
///   3. When `ActiveSubmission` is retired, the mapped buffers associated with it are moved to `ready_to_map` vector.
///   4. Finally, `handle_mapping` issues all the callbacks.
pub(super) struct LifetimeTracker<A: hal::Api> {
    /// Resources that the user has requested be mapped, but are still in use.
    mapped: Vec<Stored<id::BufferId>>,
    /// Buffers can be used in a submission that is yet to be made, by the
    /// means of `write_buffer()`, so we have a special place for them.
    pub future_suspected_buffers: Vec<Stored<id::BufferId>>,
    /// Textures can be used in the upcoming submission by `write_texture`.
    pub future_suspected_textures: Vec<Stored<id::TextureId>>,
    /// Resources that are suspected for destruction.
    pub suspected_resources: SuspectedResources,
    /// Resources that are not referenced any more but still used by GPU.
    /// Grouped by submissions associated with a fence and a submission index.
    /// The active submissions have to be stored in FIFO order: oldest come first.
    active: Vec<ActiveSubmission<A>>,
    /// Resources that are neither referenced or used, just life_tracker
    /// actual deletion.
    free_resources: NonReferencedResources<A>,
    ready_to_map: Vec<id::Valid<id::BufferId>>,
}

impl<A: hal::Api> LifetimeTracker<A> {
    pub fn new() -> Self {
        Self {
            mapped: Vec::new(),
            future_suspected_buffers: Vec::new(),
            future_suspected_textures: Vec::new(),
            suspected_resources: SuspectedResources::default(),
            active: Vec::new(),
            free_resources: NonReferencedResources::new(),
            ready_to_map: Vec::new(),
        }
    }

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
                TempResource::Texture(raw) => last_resources.textures.push(raw),
            }
        }

        self.active.alloc().init(ActiveSubmission {
            index,
            last_resources,
            mapped: Vec::new(),
            encoders,
            work_done_closures: SmallVec::new(),
        });
    }

    pub fn post_submit(&mut self) {
        self.suspected_resources.buffers.extend(
            self.future_suspected_buffers
                .drain(..)
                .map(|stored| stored.value),
        );
        self.suspected_resources.textures.extend(
            self.future_suspected_textures
                .drain(..)
                .map(|stored| stored.value),
        );
    }

    pub(crate) fn map(&mut self, value: id::Valid<id::BufferId>, ref_count: RefCount) {
        self.mapped.push(Stored { value, ref_count });
    }

    /// Returns the last submission index that is done.
    #[must_use]
    pub fn triage_submissions(
        &mut self,
        last_done: SubmissionIndex,
        command_allocator: &Mutex<super::CommandAllocator<A>>,
    ) -> SmallVec<[SubmittedWorkDoneClosure; 1]> {
        profiling::scope!("triage_submissions");

        //TODO: enable when `is_sorted_by_key` is stable
        //debug_assert!(self.active.is_sorted_by_key(|a| a.index));
        let done_count = self
            .active
            .iter()
            .position(|a| a.index > last_done)
            .unwrap_or_else(|| self.active.len());

        let mut work_done_closures = SmallVec::new();
        for a in self.active.drain(..done_count) {
            log::trace!("Active submission {} is done", a.index);
            self.free_resources.extend(a.last_resources);
            self.ready_to_map.extend(a.mapped);
            for encoder in a.encoders {
                let raw = unsafe { encoder.land() };
                command_allocator.lock().release_encoder(raw);
            }
            work_done_closures.extend(a.work_done_closures);
        }
        work_done_closures
    }

    pub fn cleanup(&mut self, device: &A::Device) {
        profiling::scope!("cleanup", "LifetimeTracker");
        unsafe {
            self.free_resources.clean(device);
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
            TempResource::Texture(raw) => resources.textures.push(raw),
        }
    }

    pub fn add_work_done_closure(&mut self, closure: SubmittedWorkDoneClosure) -> bool {
        match self.active.last_mut() {
            Some(active) => {
                active.work_done_closures.push(closure);
                true
            }
            // Note: we can't immediately invoke the closure, since it assumes
            // nothing is currently locked in the hubs.
            None => false,
        }
    }
}

impl<A: HalApi> LifetimeTracker<A> {
    pub(super) fn triage_suspected<G: GlobalIdentityHandlerFactory>(
        &mut self,
        hub: &Hub<A, G>,
        trackers: &Mutex<TrackerSet>,
        #[cfg(feature = "trace")] trace: Option<&Mutex<trace::Trace>>,
        token: &mut Token<super::Device<A>>,
    ) {
        profiling::scope!("triage_suspected");

        if !self.suspected_resources.render_bundles.is_empty() {
            let (mut guard, _) = hub.render_bundles.write(token);
            let mut trackers = trackers.lock();

            while let Some(id) = self.suspected_resources.render_bundles.pop() {
                if trackers.bundles.remove_abandoned(id) {
                    #[cfg(feature = "trace")]
                    if let Some(t) = trace {
                        t.lock().add(trace::Action::DestroyRenderBundle(id.0));
                    }

                    if let Some(res) = hub.render_bundles.unregister_locked(id.0, &mut *guard) {
                        self.suspected_resources.add_trackers(&res.used);
                    }
                }
            }
        }

        if !self.suspected_resources.bind_groups.is_empty() {
            let (mut guard, _) = hub.bind_groups.write(token);
            let mut trackers = trackers.lock();

            while let Some(id) = self.suspected_resources.bind_groups.pop() {
                if trackers.bind_groups.remove_abandoned(id) {
                    #[cfg(feature = "trace")]
                    if let Some(t) = trace {
                        t.lock().add(trace::Action::DestroyBindGroup(id.0));
                    }

                    if let Some(res) = hub.bind_groups.unregister_locked(id.0, &mut *guard) {
                        self.suspected_resources.add_trackers(&res.used);

                        self.suspected_resources
                            .bind_group_layouts
                            .push(res.layout_id);

                        let submit_index = res.life_guard.life_count();
                        self.active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                            .bind_groups
                            .push(res.raw);
                    }
                }
            }
        }

        if !self.suspected_resources.texture_views.is_empty() {
            let (mut guard, _) = hub.texture_views.write(token);
            let mut trackers = trackers.lock();

            let mut list = mem::take(&mut self.suspected_resources.texture_views);
            for id in list.drain(..) {
                if trackers.views.remove_abandoned(id) {
                    #[cfg(feature = "trace")]
                    if let Some(t) = trace {
                        t.lock().add(trace::Action::DestroyTextureView(id.0));
                    }

                    if let Some(res) = hub.texture_views.unregister_locked(id.0, &mut *guard) {
                        self.suspected_resources.textures.push(res.parent_id.value);
                        let submit_index = res.life_guard.life_count();
                        self.active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                            .texture_views
                            .push(res.raw);
                    }
                }
            }
            self.suspected_resources.texture_views = list;
        }

        if !self.suspected_resources.textures.is_empty() {
            let (mut guard, _) = hub.textures.write(token);
            let mut trackers = trackers.lock();

            for id in self.suspected_resources.textures.drain(..) {
                if trackers.textures.remove_abandoned(id) {
                    #[cfg(feature = "trace")]
                    if let Some(t) = trace {
                        t.lock().add(trace::Action::DestroyTexture(id.0));
                    }

                    if let Some(res) = hub.textures.unregister_locked(id.0, &mut *guard) {
                        let submit_index = res.life_guard.life_count();
                        let raw = match res.inner {
                            resource::TextureInner::Native { raw: Some(raw) } => raw,
                            _ => continue,
                        };
                        self.active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                            .textures
                            .push(raw);
                    }
                }
            }
        }

        if !self.suspected_resources.samplers.is_empty() {
            let (mut guard, _) = hub.samplers.write(token);
            let mut trackers = trackers.lock();

            for id in self.suspected_resources.samplers.drain(..) {
                if trackers.samplers.remove_abandoned(id) {
                    #[cfg(feature = "trace")]
                    if let Some(t) = trace {
                        t.lock().add(trace::Action::DestroySampler(id.0));
                    }

                    if let Some(res) = hub.samplers.unregister_locked(id.0, &mut *guard) {
                        let submit_index = res.life_guard.life_count();
                        self.active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                            .samplers
                            .push(res.raw);
                    }
                }
            }
        }

        if !self.suspected_resources.buffers.is_empty() {
            let (mut guard, _) = hub.buffers.write(token);
            let mut trackers = trackers.lock();

            for id in self.suspected_resources.buffers.drain(..) {
                if trackers.buffers.remove_abandoned(id) {
                    #[cfg(feature = "trace")]
                    if let Some(t) = trace {
                        t.lock().add(trace::Action::DestroyBuffer(id.0));
                    }
                    log::debug!("Buffer {:?} is detached", id);

                    if let Some(res) = hub.buffers.unregister_locked(id.0, &mut *guard) {
                        let submit_index = res.life_guard.life_count();
                        if let resource::BufferMapState::Init { stage_buffer, .. } = res.map_state {
                            self.free_resources.buffers.push(stage_buffer);
                        }
                        self.active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                            .buffers
                            .extend(res.raw);
                    }
                }
            }
        }

        if !self.suspected_resources.compute_pipelines.is_empty() {
            let (mut guard, _) = hub.compute_pipelines.write(token);
            let mut trackers = trackers.lock();

            for id in self.suspected_resources.compute_pipelines.drain(..) {
                if trackers.compute_pipes.remove_abandoned(id) {
                    #[cfg(feature = "trace")]
                    if let Some(t) = trace {
                        t.lock().add(trace::Action::DestroyComputePipeline(id.0));
                    }

                    if let Some(res) = hub.compute_pipelines.unregister_locked(id.0, &mut *guard) {
                        let submit_index = res.life_guard.life_count();
                        self.active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                            .compute_pipes
                            .push(res.raw);
                    }
                }
            }
        }

        if !self.suspected_resources.render_pipelines.is_empty() {
            let (mut guard, _) = hub.render_pipelines.write(token);
            let mut trackers = trackers.lock();

            for id in self.suspected_resources.render_pipelines.drain(..) {
                if trackers.render_pipes.remove_abandoned(id) {
                    #[cfg(feature = "trace")]
                    if let Some(t) = trace {
                        t.lock().add(trace::Action::DestroyRenderPipeline(id.0));
                    }

                    if let Some(res) = hub.render_pipelines.unregister_locked(id.0, &mut *guard) {
                        let submit_index = res.life_guard.life_count();
                        self.active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                            .render_pipes
                            .push(res.raw);
                    }
                }
            }
        }

        if !self.suspected_resources.pipeline_layouts.is_empty() {
            let (mut guard, _) = hub.pipeline_layouts.write(token);

            for Stored {
                value: id,
                ref_count,
            } in self.suspected_resources.pipeline_layouts.drain(..)
            {
                //Note: this has to happen after all the suspected pipelines are destroyed
                if ref_count.load() == 1 {
                    #[cfg(feature = "trace")]
                    if let Some(t) = trace {
                        t.lock().add(trace::Action::DestroyPipelineLayout(id.0));
                    }

                    if let Some(lay) = hub.pipeline_layouts.unregister_locked(id.0, &mut *guard) {
                        self.suspected_resources
                            .bind_group_layouts
                            .extend_from_slice(&lay.bind_group_layout_ids);
                        self.free_resources.pipeline_layouts.push(lay.raw);
                    }
                }
            }
        }

        if !self.suspected_resources.bind_group_layouts.is_empty() {
            let (mut guard, _) = hub.bind_group_layouts.write(token);

            for id in self.suspected_resources.bind_group_layouts.drain(..) {
                //Note: this has to happen after all the suspected pipelines are destroyed
                //Note: nothing else can bump the refcount since the guard is locked exclusively
                //Note: same BGL can appear multiple times in the list, but only the last
                // encounter could drop the refcount to 0.
                if guard[id].multi_ref_count.dec_and_check_empty() {
                    #[cfg(feature = "trace")]
                    if let Some(t) = trace {
                        t.lock().add(trace::Action::DestroyBindGroupLayout(id.0));
                    }
                    if let Some(lay) = hub.bind_group_layouts.unregister_locked(id.0, &mut *guard) {
                        self.free_resources.bind_group_layouts.push(lay.raw);
                    }
                }
            }
        }

        if !self.suspected_resources.query_sets.is_empty() {
            let (mut guard, _) = hub.query_sets.write(token);
            let mut trackers = trackers.lock();

            for id in self.suspected_resources.query_sets.drain(..) {
                if trackers.query_sets.remove_abandoned(id) {
                    // #[cfg(feature = "trace")]
                    // trace.map(|t| t.lock().add(trace::Action::DestroyComputePipeline(id.0)));
                    if let Some(res) = hub.query_sets.unregister_locked(id.0, &mut *guard) {
                        let submit_index = res.life_guard.life_count();
                        self.active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                            .query_sets
                            .push(res.raw);
                    }
                }
            }
        }
    }

    pub(super) fn triage_mapped<G: GlobalIdentityHandlerFactory>(
        &mut self,
        hub: &Hub<A, G>,
        token: &mut Token<super::Device<A>>,
    ) {
        if self.mapped.is_empty() {
            return;
        }
        let (buffer_guard, _) = hub.buffers.read(token);

        for stored in self.mapped.drain(..) {
            let resource_id = stored.value;
            let buf = &buffer_guard[resource_id];

            let submit_index = buf.life_guard.life_count();
            log::trace!(
                "Mapping of {:?} at submission {:?} gets assigned to active {:?}",
                resource_id,
                submit_index,
                self.active.iter().position(|a| a.index == submit_index)
            );

            self.active
                .iter_mut()
                .find(|a| a.index == submit_index)
                .map_or(&mut self.ready_to_map, |a| &mut a.mapped)
                .push(resource_id);
        }
    }

    #[must_use]
    pub(super) fn handle_mapping<G: GlobalIdentityHandlerFactory>(
        &mut self,
        hub: &Hub<A, G>,
        raw: &A::Device,
        trackers: &Mutex<TrackerSet>,
        token: &mut Token<super::Device<A>>,
    ) -> Vec<super::BufferMapPendingClosure> {
        if self.ready_to_map.is_empty() {
            return Vec::new();
        }
        let (mut buffer_guard, _) = hub.buffers.write(token);
        let mut pending_callbacks: Vec<super::BufferMapPendingClosure> =
            Vec::with_capacity(self.ready_to_map.len());
        let mut trackers = trackers.lock();
        for buffer_id in self.ready_to_map.drain(..) {
            let buffer = &mut buffer_guard[buffer_id];
            if buffer.life_guard.ref_count.is_none() && trackers.buffers.remove_abandoned(buffer_id)
            {
                buffer.map_state = resource::BufferMapState::Idle;
                log::debug!("Mapping request is dropped because the buffer is destroyed.");
                if let Some(buf) = hub
                    .buffers
                    .unregister_locked(buffer_id.0, &mut *buffer_guard)
                {
                    self.free_resources.buffers.extend(buf.raw);
                }
            } else {
                let mapping = match std::mem::replace(
                    &mut buffer.map_state,
                    resource::BufferMapState::Idle,
                ) {
                    resource::BufferMapState::Waiting(pending_mapping) => pending_mapping,
                    // Mapping cancelled
                    resource::BufferMapState::Idle => continue,
                    // Mapping queued at least twice by map -> unmap -> map
                    // and was already successfully mapped below
                    active @ resource::BufferMapState::Active { .. } => {
                        buffer.map_state = active;
                        continue;
                    }
                    _ => panic!("No pending mapping."),
                };
                let status = if mapping.range.start != mapping.range.end {
                    log::debug!("Buffer {:?} map state -> Active", buffer_id);
                    let host = mapping.op.host;
                    let size = mapping.range.end - mapping.range.start;
                    match super::map_buffer(raw, buffer, mapping.range.start, size, host) {
                        Ok(ptr) => {
                            buffer.map_state = resource::BufferMapState::Active {
                                ptr,
                                range: mapping.range.start..mapping.range.start + size,
                                host,
                            };
                            resource::BufferMapAsyncStatus::Success
                        }
                        Err(e) => {
                            log::error!("Mapping failed {:?}", e);
                            resource::BufferMapAsyncStatus::Error
                        }
                    }
                } else {
                    resource::BufferMapAsyncStatus::Success
                };
                pending_callbacks.push((mapping.op, status));
            }
        }
        pending_callbacks
    }
}
