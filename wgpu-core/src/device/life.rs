/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#[cfg(feature = "trace")]
use crate::device::trace;
use crate::{
    hub::{GfxBackend, GlobalIdentityHandlerFactory, Hub, Token},
    id, resource,
    track::TrackerSet,
    FastHashMap, RefCount, Stored, SubmissionIndex,
};

use copyless::VecHelper as _;
use gfx_descriptor::{DescriptorAllocator, DescriptorSet};
use gfx_memory::{Heaps, MemoryBlock};
use hal::device::{Device, OomOrDeviceLost};
use parking_lot::Mutex;

use std::{fmt, sync::atomic::Ordering};

const CLEANUP_WAIT_MS: u64 = 5000;

/// A struct that keeps lists of resources that are no longer needed by the user.
#[derive(Debug, Default)]
pub struct SuspectedResources {
    pub(crate) buffers: Vec<id::BufferId>,
    pub(crate) textures: Vec<id::TextureId>,
    pub(crate) texture_views: Vec<id::TextureViewId>,
    pub(crate) samplers: Vec<id::SamplerId>,
    pub(crate) bind_groups: Vec<id::BindGroupId>,
    pub(crate) compute_pipelines: Vec<id::ComputePipelineId>,
    pub(crate) render_pipelines: Vec<id::RenderPipelineId>,
    pub(crate) bind_group_layouts: Vec<Stored<id::BindGroupLayoutId>>,
    pub(crate) pipeline_layouts: Vec<Stored<id::PipelineLayoutId>>,
    pub(crate) render_bundles: Vec<id::RenderBundleId>,
}

impl SuspectedResources {
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
    }

    pub(crate) fn add_trackers(&mut self, trackers: &TrackerSet) {
        self.buffers.extend(trackers.buffers.used());
        self.textures.extend(trackers.textures.used());
        self.texture_views.extend(trackers.views.used());
        self.samplers.extend(trackers.samplers.used());
        self.bind_groups.extend(trackers.bind_groups.used());
        self.compute_pipelines.extend(trackers.compute_pipes.used());
        self.render_pipelines.extend(trackers.render_pipes.used());
        self.render_bundles.extend(trackers.bundles.used());
    }
}

/// A struct that keeps lists of resources that are no longer needed.
#[derive(Debug)]
struct NonReferencedResources<B: hal::Backend> {
    buffers: Vec<(B::Buffer, MemoryBlock<B>)>,
    images: Vec<(B::Image, MemoryBlock<B>)>,
    // Note: we keep the associated ID here in order to be able to check
    // at any point what resources are used in a submission.
    image_views: Vec<(id::TextureViewId, B::ImageView)>,
    samplers: Vec<B::Sampler>,
    framebuffers: Vec<B::Framebuffer>,
    desc_sets: Vec<DescriptorSet<B>>,
    compute_pipes: Vec<B::ComputePipeline>,
    graphics_pipes: Vec<B::GraphicsPipeline>,
    descriptor_set_layouts: Vec<B::DescriptorSetLayout>,
    pipeline_layouts: Vec<B::PipelineLayout>,
}

impl<B: hal::Backend> NonReferencedResources<B> {
    fn new() -> Self {
        NonReferencedResources {
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
        assert!(other.descriptor_set_layouts.is_empty());
        assert!(other.pipeline_layouts.is_empty());
    }

    unsafe fn clean(
        &mut self,
        device: &B::Device,
        heaps_mutex: &Mutex<Heaps<B>>,
        descriptor_allocator_mutex: &Mutex<DescriptorAllocator<B>>,
    ) {
        if !self.buffers.is_empty() {
            let mut heaps = heaps_mutex.lock();
            for (raw, memory) in self.buffers.drain(..) {
                log::trace!("Buffer {:?} is destroyed with memory {:?}", raw, memory);
                device.destroy_buffer(raw);
                heaps.free(device, memory);
            }
        }
        if !self.images.is_empty() {
            let mut heaps = heaps_mutex.lock();
            for (raw, memory) in self.images.drain(..) {
                device.destroy_image(raw);
                heaps.free(device, memory);
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
                .free(self.desc_sets.drain(..));
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
    }
}

#[derive(Debug)]
struct ActiveSubmission<B: hal::Backend> {
    index: SubmissionIndex,
    fence: B::Fence,
    last_resources: NonReferencedResources<B>,
    mapped: Vec<id::BufferId>,
}

#[derive(Clone, Debug)]
pub enum WaitIdleError {
    OomOrDeviceLost(OomOrDeviceLost),
    StuckGpu,
}

impl From<OomOrDeviceLost> for WaitIdleError {
    fn from(error: OomOrDeviceLost) -> Self {
        Self::OomOrDeviceLost(error)
    }
}

impl fmt::Display for WaitIdleError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::OomOrDeviceLost(error) => write!(f, "{}", error),
            Self::StuckGpu => write!(f, "GPU got stuck :("),
        }
    }
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
pub struct LifetimeTracker<B: hal::Backend> {
    /// Resources that the user has requested be mapped, but are still in use.
    mapped: Vec<Stored<id::BufferId>>,
    /// Buffers can be used in a submission that is yet to be made, by the
    /// means of `write_buffer()`, so we have a special place for them.
    pub future_suspected_buffers: Vec<id::BufferId>,
    /// Textures can be used in the upcoming submission by `write_texture`.
    pub future_suspected_textures: Vec<id::TextureId>,
    /// Resources that are suspected for destruction.
    pub suspected_resources: SuspectedResources,
    /// Resources that are not referenced any more but still used by GPU.
    /// Grouped by submissions associated with a fence and a submission index.
    /// The active submissions have to be stored in FIFO order: oldest come first.
    active: Vec<ActiveSubmission<B>>,
    /// Resources that are neither referenced or used, just life_tracker
    /// actual deletion.
    free_resources: NonReferencedResources<B>,
    ready_to_map: Vec<id::BufferId>,
}

impl<B: hal::Backend> LifetimeTracker<B> {
    pub fn new() -> Self {
        LifetimeTracker {
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
        new_suspects: &SuspectedResources,
        temp_buffers: impl Iterator<Item = (B::Buffer, MemoryBlock<B>)>,
    ) {
        let mut last_resources = NonReferencedResources::new();
        last_resources.buffers.extend(temp_buffers);
        self.suspected_resources
            .buffers
            .extend(self.future_suspected_buffers.drain(..));
        self.suspected_resources
            .textures
            .extend(self.future_suspected_textures.drain(..));
        self.suspected_resources.extend(new_suspects);
        self.active.alloc().init(ActiveSubmission {
            index,
            fence,
            last_resources,
            mapped: Vec::new(),
        });
    }

    pub(crate) fn map(&mut self, buffer: id::BufferId, ref_count: RefCount) {
        self.mapped.push(Stored {
            value: buffer,
            ref_count,
        });
    }

    fn wait_idle(&self, device: &B::Device) -> Result<(), WaitIdleError> {
        if !self.active.is_empty() {
            log::debug!("Waiting for IDLE...");
            let status = unsafe {
                device.wait_for_fences(
                    self.active.iter().map(|a| &a.fence),
                    hal::device::WaitFor::All,
                    CLEANUP_WAIT_MS * 1_000_000,
                )?
            };
            log::debug!("...Done");

            if status == false {
                // We timed out while waiting for the fences
                return Err(WaitIdleError::StuckGpu);
            }
        }
        Ok(())
    }

    /// Returns the last submission index that is done.
    pub fn triage_submissions(&mut self, device: &B::Device, force_wait: bool) -> SubmissionIndex {
        if force_wait {
            self.wait_idle(device).unwrap();
        }
        //TODO: enable when `is_sorted_by_key` is stable
        //debug_assert!(self.active.is_sorted_by_key(|a| a.index));
        let done_count = self
            .active
            .iter()
            .position(|a| unsafe { !device.get_fence_status(&a.fence).unwrap() })
            .unwrap_or_else(|| self.active.len());
        let last_done = if done_count != 0 {
            self.active[done_count - 1].index
        } else {
            return 0;
        };

        for a in self.active.drain(..done_count) {
            log::trace!("Active submission {} is done", a.index);
            self.free_resources.extend(a.last_resources);
            self.ready_to_map.extend(a.mapped);
            unsafe {
                device.destroy_fence(a.fence);
            }
        }

        last_done
    }

    pub fn cleanup(
        &mut self,
        device: &B::Device,
        heaps_mutex: &Mutex<Heaps<B>>,
        descriptor_allocator_mutex: &Mutex<DescriptorAllocator<B>>,
    ) {
        unsafe {
            self.free_resources
                .clean(device, heaps_mutex, descriptor_allocator_mutex);
            descriptor_allocator_mutex.lock().cleanup(device);
        }
    }
}

impl<B: GfxBackend> LifetimeTracker<B> {
    pub(crate) fn triage_suspected<G: GlobalIdentityHandlerFactory>(
        &mut self,
        hub: &Hub<B, G>,
        trackers: &Mutex<TrackerSet>,
        #[cfg(feature = "trace")] trace: Option<&Mutex<trace::Trace>>,
        token: &mut Token<super::Device<B>>,
    ) {
        if !self.suspected_resources.render_bundles.is_empty() {
            let mut trackers = trackers.lock();
            let (mut guard, _) = hub.render_bundles.write(token);

            while let Some(id) = self.suspected_resources.render_bundles.pop() {
                if trackers.bundles.remove_abandoned(id) {
                    #[cfg(feature = "trace")]
                    trace.map(|t| t.lock().add(trace::Action::DestroyRenderBundle(id)));
                    hub.render_bundles.free_id(id);
                    let res = guard.remove(id).unwrap();
                    self.suspected_resources.add_trackers(&res.used);
                }
            }
        }

        if !self.suspected_resources.bind_groups.is_empty() {
            let mut trackers = trackers.lock();
            let (mut guard, _) = hub.bind_groups.write(token);

            while let Some(id) = self.suspected_resources.bind_groups.pop() {
                if trackers.bind_groups.remove_abandoned(id) {
                    #[cfg(feature = "trace")]
                    trace.map(|t| t.lock().add(trace::Action::DestroyBindGroup(id)));
                    hub.bind_groups.free_id(id);
                    let res = guard.remove(id).unwrap();

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

        if !self.suspected_resources.texture_views.is_empty() {
            let mut trackers = trackers.lock();
            let (mut guard, _) = hub.texture_views.write(token);

            for id in self.suspected_resources.texture_views.drain(..) {
                if trackers.views.remove_abandoned(id) {
                    #[cfg(feature = "trace")]
                    trace.map(|t| t.lock().add(trace::Action::DestroyTextureView(id)));
                    hub.texture_views.free_id(id);
                    let res = guard.remove(id).unwrap();

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

        if !self.suspected_resources.textures.is_empty() {
            let mut trackers = trackers.lock();
            let (mut guard, _) = hub.textures.write(token);

            for id in self.suspected_resources.textures.drain(..) {
                if trackers.textures.remove_abandoned(id) {
                    #[cfg(feature = "trace")]
                    trace.map(|t| t.lock().add(trace::Action::DestroyTexture(id)));
                    hub.textures.free_id(id);
                    let res = guard.remove(id).unwrap();

                    let submit_index = res.life_guard.submission_index.load(Ordering::Acquire);
                    self.active
                        .iter_mut()
                        .find(|a| a.index == submit_index)
                        .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                        .images
                        .push((res.raw, res.memory));
                }
            }
        }

        if !self.suspected_resources.samplers.is_empty() {
            let mut trackers = trackers.lock();
            let (mut guard, _) = hub.samplers.write(token);

            for id in self.suspected_resources.samplers.drain(..) {
                if trackers.samplers.remove_abandoned(id) {
                    #[cfg(feature = "trace")]
                    trace.map(|t| t.lock().add(trace::Action::DestroySampler(id)));
                    hub.samplers.free_id(id);
                    let res = guard.remove(id).unwrap();

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

        if !self.suspected_resources.buffers.is_empty() {
            let mut trackers = trackers.lock();
            let (mut guard, _) = hub.buffers.write(token);

            for id in self.suspected_resources.buffers.drain(..) {
                if trackers.buffers.remove_abandoned(id) {
                    #[cfg(feature = "trace")]
                    trace.map(|t| t.lock().add(trace::Action::DestroyBuffer(id)));
                    hub.buffers.free_id(id);
                    let res = guard.remove(id).unwrap();
                    log::debug!("Buffer {:?} is detached", id);

                    let submit_index = res.life_guard.submission_index.load(Ordering::Acquire);
                    self.active
                        .iter_mut()
                        .find(|a| a.index == submit_index)
                        .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                        .buffers
                        .push((res.raw, res.memory));
                }
            }
        }

        if !self.suspected_resources.compute_pipelines.is_empty() {
            let mut trackers = trackers.lock();
            let (mut guard, _) = hub.compute_pipelines.write(token);

            for id in self.suspected_resources.compute_pipelines.drain(..) {
                if trackers.compute_pipes.remove_abandoned(id) {
                    #[cfg(feature = "trace")]
                    trace.map(|t| t.lock().add(trace::Action::DestroyComputePipeline(id)));
                    hub.compute_pipelines.free_id(id);
                    let res = guard.remove(id).unwrap();

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

        if !self.suspected_resources.render_pipelines.is_empty() {
            let mut trackers = trackers.lock();
            let (mut guard, _) = hub.render_pipelines.write(token);

            for id in self.suspected_resources.render_pipelines.drain(..) {
                if trackers.render_pipes.remove_abandoned(id) {
                    #[cfg(feature = "trace")]
                    trace.map(|t| t.lock().add(trace::Action::DestroyRenderPipeline(id)));
                    hub.render_pipelines.free_id(id);
                    let res = guard.remove(id).unwrap();

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

        if !self.suspected_resources.bind_group_layouts.is_empty() {
            let (mut guard, _) = hub.bind_group_layouts.write(token);

            for Stored {
                value: id,
                ref_count,
            } in self.suspected_resources.bind_group_layouts.drain(..)
            {
                //Note: this has to happen after all the suspected pipelines are destroyed
                if ref_count.load() == 1 {
                    #[cfg(feature = "trace")]
                    trace.map(|t| t.lock().add(trace::Action::DestroyBindGroupLayout(id)));
                    hub.bind_group_layouts.free_id(id);
                    let layout = guard.remove(id).unwrap();
                    self.free_resources.descriptor_set_layouts.push(layout.raw);
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
                    trace.map(|t| t.lock().add(trace::Action::DestroyPipelineLayout(id)));
                    hub.pipeline_layouts.free_id(id);
                    let layout = guard.remove(id).unwrap();
                    self.free_resources.pipeline_layouts.push(layout.raw);
                }
            }
        }
    }

    pub(crate) fn triage_mapped<G: GlobalIdentityHandlerFactory>(
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

    pub(crate) fn triage_framebuffers<G: GlobalIdentityHandlerFactory>(
        &mut self,
        hub: &Hub<B, G>,
        framebuffers: &mut FastHashMap<super::FramebufferKey, B::Framebuffer>,
        token: &mut Token<super::Device<B>>,
    ) {
        let (texture_view_guard, _) = hub.texture_views.read(token);
        let remove_list = framebuffers
            .keys()
            .filter_map(|key| {
                let mut last_submit = None;
                let mut needs_cleanup = false;

                // A framebuffer needs to be scheduled for cleanup, if there's at least one
                // attachment is no longer valid.

                for &at in key.all() {
                    // If this attachment is still registered, it's still valid
                    if texture_view_guard.contains(at) {
                        continue;
                    }

                    // This attachment is no longer registered, this framebuffer needs cleanup
                    needs_cleanup = true;

                    // Check if there's any active submissions that are still referring to this
                    // attachment, if there are we need to get the greatest submission index, as
                    // that's the last time this attachment is still valid
                    let mut attachment_last_submit = None;
                    for a in &self.active {
                        if a.last_resources.image_views.iter().any(|&(id, _)| id == at) {
                            let max = attachment_last_submit.unwrap_or(0).max(a.index);
                            attachment_last_submit = Some(max);
                        }
                    }

                    // Between all attachments, we need the smallest index, because that's the last
                    // time this framebuffer is still valid
                    if let Some(attachment_last_submit) = attachment_last_submit {
                        let min = last_submit
                            .unwrap_or(std::usize::MAX)
                            .min(attachment_last_submit);
                        last_submit = Some(min);
                    }
                }

                if needs_cleanup {
                    Some((key.clone(), last_submit.unwrap_or(0)))
                } else {
                    None
                }
            })
            .collect::<FastHashMap<_, _>>();

        if !remove_list.is_empty() {
            log::debug!("Free framebuffers {:?}", remove_list);
            for (ref key, submit_index) in remove_list {
                let framebuffer = framebuffers.remove(key).unwrap();
                self.active
                    .iter_mut()
                    .find(|a| a.index == submit_index)
                    .map_or(&mut self.free_resources, |a| &mut a.last_resources)
                    .framebuffers
                    .push(framebuffer);
            }
        }
    }

    pub(crate) fn handle_mapping<G: GlobalIdentityHandlerFactory>(
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
                hub.buffers.free_id(buffer_id);
                let buffer = buffer_guard.remove(buffer_id).unwrap();
                self.free_resources
                    .buffers
                    .push((buffer.raw, buffer.memory));
            } else {
                let mapping = match std::mem::replace(
                    &mut buffer.map_state,
                    resource::BufferMapState::Idle,
                ) {
                    resource::BufferMapState::Waiting(pending_mapping) => pending_mapping,
                    _ => panic!("No pending mapping."),
                };
                let status = if mapping.sub_range.size.map_or(true, |x| x != 0) {
                    log::debug!("Buffer {:?} map state -> Active", buffer_id);
                    let host = mapping.op.host;
                    match super::map_buffer(raw, buffer, mapping.sub_range.clone(), host) {
                        Ok(ptr) => {
                            buffer.map_state = resource::BufferMapState::Active {
                                ptr,
                                sub_range: mapping.sub_range,
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
