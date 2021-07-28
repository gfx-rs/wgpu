/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#[cfg(feature = "trace")]
use crate::device::trace;
use crate::{
    device::{
        alloc,
        descriptor::{DescriptorAllocator, DescriptorSet},
        queue::TempResource,
        DeviceError,
    },
    hub::{GfxBackend, GlobalIdentityHandlerFactory, Hub, Token},
    id, resource,
    track::TrackerSet,
    RefCount, Stored, SubmissionIndex,
};

use copyless::VecHelper as _;
use hal::device::Device as _;
use parking_lot::Mutex;
use thiserror::Error;

use std::sync::atomic::Ordering;

const CLEANUP_WAIT_MS: u64 = 5000;

/// A struct that keeps lists of resources that are no longer needed by the user.
#[derive(Debug, Default)]
pub(super) struct SuspectedResources {
    pub(crate) buffers: Vec<id::Valid<id::BufferId>>,
    pub(crate) textures: Vec<id::Valid<id::TextureId>>,
    pub(crate) texture_views: Vec<id::Valid<id::TextureViewId>>,
    pub(crate) samplers: Vec<id::Valid<id::SamplerId>>,
    pub(crate) bind_groups: Vec<id::Valid<id::BindGroupId>>,
    pub(crate) compute_pipelines: Vec<id::Valid<id::ComputePipelineId>>,
    pub(crate) render_pipelines: Vec<id::Valid<id::RenderPipelineId>>,
    pub(crate) bind_group_layouts: Vec<id::Valid<id::BindGroupLayoutId>>,
    pub(crate) pipeline_layouts: Vec<Stored<id::PipelineLayoutId>>,
    pub(crate) render_bundles: Vec<id::Valid<id::RenderBundleId>>,
    pub(crate) query_sets: Vec<id::Valid<id::QuerySetId>>,
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
struct NonReferencedResources<B: hal::Backend> {
    buffers: Vec<(B::Buffer, alloc::MemoryBlock<B>)>,
    images: Vec<(B::Image, alloc::MemoryBlock<B>)>,
    // Note: we keep the associated ID here in order to be able to check
    // at any point what resources are used in a submission.
    image_views: Vec<(id::Valid<id::TextureViewId>, B::ImageView)>,
    samplers: Vec<B::Sampler>,
    framebuffers: Vec<B::Framebuffer>,
    desc_sets: Vec<DescriptorSet<B>>,
    compute_pipes: Vec<B::ComputePipeline>,
    graphics_pipes: Vec<B::GraphicsPipeline>,
    descriptor_set_layouts: Vec<B::DescriptorSetLayout>,
    pipeline_layouts: Vec<B::PipelineLayout>,
    query_sets: Vec<B::QueryPool>,
}

impl<B: hal::Backend> NonReferencedResources<B> {
    fn new() -> Self {
        Self {
            buffers: Vec::new(),
            images: Vec::new(),
            image_views: Vec::new(),
            samplers: Vec::new(),
            framebuffers: Vec::new(),
            desc_sets: Vec::new(),
            compute_pipes: Vec::new(),
            graphics_pipes: Vec::new(),
            descriptor_set_layouts: Vec::new(),
            pipeline_layouts: Vec::new(),
            query_sets: Vec::new(),
        }
    }

    fn extend(&mut self, other: Self) {
        self.buffers.extend(other.buffers);
        self.images.extend(other.images);
        self.image_views.extend(other.image_views);
        self.samplers.extend(other.samplers);
        self.framebuffers.extend(other.framebuffers);
        self.desc_sets.extend(other.desc_sets);
        self.compute_pipes.extend(other.compute_pipes);
        self.graphics_pipes.extend(other.graphics_pipes);
        self.query_sets.extend(other.query_sets);
        assert!(other.descriptor_set_layouts.is_empty());
        assert!(other.pipeline_layouts.is_empty());
    }

    unsafe fn clean(
        &mut self,
        device: &B::Device,
        memory_allocator_mutex: &Mutex<alloc::MemoryAllocator<B>>,
        descriptor_allocator_mutex: &Mutex<DescriptorAllocator<B>>,
    ) {
        if !self.buffers.is_empty() || !self.images.is_empty() {
            let mut allocator = memory_allocator_mutex.lock();
            for (raw, memory) in self.buffers.drain(..) {
                log::trace!("Buffer {:?} is destroyed with memory {:?}", raw, memory);
                device.destroy_buffer(raw);
                allocator.free(device, memory);
            }
            for (raw, memory) in self.images.drain(..) {
                log::trace!("Image {:?} is destroyed with memory {:?}", raw, memory);
                device.destroy_image(raw);
                allocator.free(device, memory);
            }
        }

        for (_, raw) in self.image_views.drain(..) {
            device.destroy_image_view(raw);
        }
        for raw in self.samplers.drain(..) {
            device.destroy_sampler(raw);
        }
        for raw in self.framebuffers.drain(..) {
            device.destroy_framebuffer(raw);
        }

        if !self.desc_sets.is_empty() {
            descriptor_allocator_mutex
                .lock()
                .free(device, self.desc_sets.drain(..));
        }

        for raw in self.compute_pipes.drain(..) {
            device.destroy_compute_pipeline(raw);
        }
        for raw in self.graphics_pipes.drain(..) {
            device.destroy_graphics_pipeline(raw);
        }
        for raw in self.descriptor_set_layouts.drain(..) {
            device.destroy_descriptor_set_layout(raw);
        }
        for raw in self.pipeline_layouts.drain(..) {
            device.destroy_pipeline_layout(raw);
        }
        for raw in self.query_sets.drain(..) {
            device.destroy_query_pool(raw);
        }
    }
}

#[derive(Debug)]
struct ActiveSubmission<B: hal::Backend> {
    index: SubmissionIndex,
    fence: B::Fence,
    last_resources: NonReferencedResources<B>,
    mapped: Vec<id::Valid<id::BufferId>>,
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
#[derive(Debug)]
pub(super) struct LifetimeTracker<B: hal::Backend> {
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
    active: Vec<ActiveSubmission<B>>,
    /// Resources that are neither referenced or used, just life_tracker
    /// actual deletion.
    free_resources: NonReferencedResources<B>,
    ready_to_map: Vec<id::Valid<id::BufferId>>,
}

impl<B: hal::Backend> LifetimeTracker<B> {
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
        fence: B::Fence,
        temp_resources: impl Iterator<Item = (TempResource<B>, alloc::MemoryBlock<B>)>,
    ) {
        let mut last_resources = NonReferencedResources::new();
        for (res, memory) in temp_resources {
            match res {
                TempResource::Buffer(raw) => last_resources.buffers.push((raw, memory)),
                TempResource::Image(raw) => last_resources.images.push((raw, memory)),
            }
        }

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

        self.active.alloc().init(ActiveSubmission {
            index,
            fence,
            last_resources,
            mapped: Vec::new(),
        });
    }

    pub(crate) fn map(&mut self, value: id::Valid<id::BufferId>, ref_count: RefCount) {
        self.mapped.push(Stored { value, ref_count });
    }

    fn wait_idle(&self, device: &B::Device) -> Result<(), WaitIdleError> {
        if !self.active.is_empty() {
            log::debug!("Waiting for IDLE...");
            let status = unsafe {
                device
                    .wait_for_fences(
                        self.active.iter().map(|a| &a.fence),
                        hal::device::WaitFor::All,
                        CLEANUP_WAIT_MS * 1_000_000,
                    )
                    .map_err(DeviceError::from)?
            };
            log::debug!("...Done");

            if !status {
                // We timed out while waiting for the fences
                return Err(WaitIdleError::StuckGpu);
            }
        }
        Ok(())
    }

    /// Returns the last submission index that is done.
    pub fn triage_submissions(
        &mut self,
        device: &B::Device,
        force_wait: bool,
    ) -> Result<SubmissionIndex, WaitIdleError> {
        profiling::scope!("triage_submissions");
        if force_wait {
            self.wait_idle(device)?;
        }
        //TODO: enable when `is_sorted_by_key` is stable
        //debug_assert!(self.active.is_sorted_by_key(|a| a.index));
        let done_count = self
            .active
            .iter()
            .position(|a| unsafe { !device.get_fence_status(&a.fence).unwrap_or(false) })
            .unwrap_or_else(|| self.active.len());
        let last_done = match done_count.checked_sub(1) {
            Some(i) => self.active[i].index,
            None => return Ok(0),
        };

        for a in self.active.drain(..done_count) {
            log::trace!("Active submission {} is done", a.index);
            self.free_resources.extend(a.last_resources);
            self.ready_to_map.extend(a.mapped);
            unsafe {
                device.destroy_fence(a.fence);
            }
        }

        Ok(last_done)
    }

    pub fn cleanup(
        &mut self,
        device: &B::Device,
        memory_allocator_mutex: &Mutex<alloc::MemoryAllocator<B>>,
        descriptor_allocator_mutex: &Mutex<DescriptorAllocator<B>>,
    ) {
        profiling::scope!("cleanup");
        unsafe {
            self.free_resources
                .clean(device, memory_allocator_mutex, descriptor_allocator_mutex);
            descriptor_allocator_mutex.lock().cleanup(device);
        }
    }

    pub fn schedule_resource_destruction(
        &mut self,
        temp_resource: TempResource<B>,
        memory: alloc::MemoryBlock<B>,
        last_submit_index: SubmissionIndex,
    ) {
        let resources = self
            .active
            .iter_mut()
            .find(|a| a.index == last_submit_index)
            .map_or(&mut self.free_resources, |a| &mut a.last_resources);
        match temp_resource {
            TempResource::Buffer(raw) => resources.buffers.push((raw, memory)),
            TempResource::Image(raw) => resources.images.push((raw, memory)),
        }
    }
}

impl<B: GfxBackend> LifetimeTracker<B> {
    pub(super) fn triage_suspected<G: GlobalIdentityHandlerFactory>(
        &mut self,
        hub: &Hub<B, G>,
        trackers: &Mutex<TrackerSet>,
        #[cfg(feature = "trace")] trace: Option<&Mutex<trace::Trace>>,
        token: &mut Token<super::Device<B>>,
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

                        let submit_index = res.life_guard.submission_index.load(Ordering::Acquire);
                        self.active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                            .desc_sets
                            .push(res.raw);
                    }
                }
            }
        }

        if !self.suspected_resources.texture_views.is_empty() {
            let (mut guard, _) = hub.texture_views.write(token);
            let mut trackers = trackers.lock();

            for id in self.suspected_resources.texture_views.drain(..) {
                if trackers.views.remove_abandoned(id) {
                    #[cfg(feature = "trace")]
                    if let Some(t) = trace {
                        t.lock().add(trace::Action::DestroyTextureView(id.0));
                    }

                    if let Some(res) = hub.texture_views.unregister_locked(id.0, &mut *guard) {
                        let raw = match res.inner {
                            resource::TextureViewInner::Native { raw, source_id } => {
                                self.suspected_resources.textures.push(source_id.value);
                                raw
                            }
                            resource::TextureViewInner::SwapChain { .. } => unreachable!(),
                        };

                        let submit_index = res.life_guard.submission_index.load(Ordering::Acquire);
                        self.active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                            .image_views
                            .push((id, raw));
                    }
                }
            }
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
                        let submit_index = res.life_guard.submission_index.load(Ordering::Acquire);
                        self.active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                            .images
                            .extend(res.raw);
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
                        let submit_index = res.life_guard.submission_index.load(Ordering::Acquire);
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
                        let submit_index = res.life_guard.submission_index.load(Ordering::Acquire);
                        if let resource::BufferMapState::Init {
                            stage_buffer,
                            stage_memory,
                            ..
                        } = res.map_state
                        {
                            self.free_resources
                                .buffers
                                .push((stage_buffer, stage_memory));
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
                        let submit_index = res.life_guard.submission_index.load(Ordering::Acquire);
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
                        let submit_index = res.life_guard.submission_index.load(Ordering::Acquire);
                        self.active
                            .iter_mut()
                            .find(|a| a.index == submit_index)
                            .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                            .graphics_pipes
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
                        self.free_resources.descriptor_set_layouts.push(lay.raw);
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
                        let submit_index = res.life_guard.submission_index.load(Ordering::Acquire);
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
        hub: &Hub<B, G>,
        token: &mut Token<super::Device<B>>,
    ) {
        if self.mapped.is_empty() {
            return;
        }
        let (buffer_guard, _) = hub.buffers.read(token);

        for stored in self.mapped.drain(..) {
            let resource_id = stored.value;
            let buf = &buffer_guard[resource_id];

            let submit_index = buf.life_guard.submission_index.load(Ordering::Acquire);
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

    pub(super) fn handle_mapping<G: GlobalIdentityHandlerFactory>(
        &mut self,
        hub: &Hub<B, G>,
        raw: &B::Device,
        trackers: &Mutex<TrackerSet>,
        token: &mut Token<super::Device<B>>,
    ) -> Vec<super::BufferMapPendingCallback> {
        if self.ready_to_map.is_empty() {
            return Vec::new();
        }
        let (mut buffer_guard, _) = hub.buffers.write(token);
        let mut pending_callbacks: Vec<super::BufferMapPendingCallback> =
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
                                sub_range: hal::buffer::SubRange {
                                    offset: mapping.range.start,
                                    size: Some(size),
                                },
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
