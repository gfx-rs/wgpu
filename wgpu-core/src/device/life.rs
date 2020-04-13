/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Token},
    id, resource,
    track::TrackerSet,
    FastHashMap, RefCount, Stored, SubmissionIndex,
};

use copyless::VecHelper as _;
use gfx_descriptor::{DescriptorAllocator, DescriptorSet};
use gfx_memory::{Heaps, MemoryBlock};
use hal::device::Device as _;
use parking_lot::Mutex;

use std::sync::atomic::Ordering;

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
}

impl SuspectedResources {
    pub fn clear(&mut self) {
        self.buffers.clear();
        self.textures.clear();
        self.texture_views.clear();
        self.samplers.clear();
        self.bind_groups.clear();
        self.compute_pipelines.clear();
        self.render_pipelines.clear();
    }

    pub fn extend(&mut self, other: &Self) {
        self.buffers.extend_from_slice(&other.buffers);
        self.textures.extend_from_slice(&other.textures);
        self.texture_views.extend_from_slice(&other.texture_views);
        self.samplers.extend_from_slice(&other.samplers);
        self.bind_groups.extend_from_slice(&other.bind_groups);
        self.compute_pipelines
            .extend_from_slice(&other.compute_pipelines);
        self.render_pipelines
            .extend_from_slice(&other.render_pipelines);
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
    }
}

#[derive(Debug)]
struct ActiveSubmission<B: hal::Backend> {
    index: SubmissionIndex,
    fence: B::Fence,
    last_resources: NonReferencedResources<B>,
    mapped: Vec<id::BufferId>,
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
    ) {
        self.suspected_resources.extend(new_suspects);
        self.active.alloc().init(ActiveSubmission {
            index,
            fence,
            last_resources: NonReferencedResources::new(),
            mapped: Vec::new(),
        });
    }

    pub fn map(&mut self, buffer: id::BufferId, ref_count: RefCount) {
        self.mapped.push(Stored {
            value: buffer,
            ref_count,
        });
    }

    /// Find the pending entry with the lowest active index. If none can be found that means
    /// everything in the allocator can be cleaned up, so std::usize::MAX is correct.
    pub fn lowest_active_submission(&self) -> SubmissionIndex {
        self.active
            .iter()
            .fold(std::usize::MAX, |v, active| active.index.min(v))
    }

    fn wait_idle(&self, device: &B::Device) {
        if !self.active.is_empty() {
            log::debug!("Waiting for IDLE...");
            let status = unsafe {
                device.wait_for_fences(
                    self.active.iter().map(|a| &a.fence),
                    hal::device::WaitFor::All,
                    CLEANUP_WAIT_MS * 1_000_000,
                )
            };
            log::debug!("...Done");
            assert_eq!(status, Ok(true), "GPU got stuck :(");
        }
    }

    /// Returns the last submission index that is done.
    pub fn triage_submissions(&mut self, device: &B::Device, force_wait: bool) -> SubmissionIndex {
        if force_wait {
            self.wait_idle(device);
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
        global: &Global<G>,
        trackers: &Mutex<TrackerSet>,
        token: &mut Token<super::Device<B>>,
    ) {
        let hub = B::hub(global);

        if !self.suspected_resources.bind_groups.is_empty() {
            let mut trackers = trackers.lock();
            let (mut guard, _) = hub.bind_groups.write(token);

            for id in self.suspected_resources.bind_groups.drain(..) {
                if trackers.bind_groups.remove_abandoned(id) {
                    hub.bind_groups.free_id(id);
                    let res = guard.remove(id).unwrap();

                    assert!(res.used.bind_groups.is_empty());
                    self.suspected_resources
                        .buffers
                        .extend(res.used.buffers.used());
                    self.suspected_resources
                        .textures
                        .extend(res.used.textures.used());
                    self.suspected_resources
                        .texture_views
                        .extend(res.used.views.used());
                    self.suspected_resources
                        .samplers
                        .extend(res.used.samplers.used());

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
    }

    pub(crate) fn triage_mapped<G: GlobalIdentityHandlerFactory>(
        &mut self,
        global: &Global<G>,
        token: &mut Token<super::Device<B>>,
    ) {
        if self.mapped.is_empty() {
            return;
        }
        let (buffer_guard, _) = B::hub(global).buffers.read(token);

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
        global: &Global<G>,
        framebuffers: &mut FastHashMap<super::FramebufferKey, B::Framebuffer>,
        token: &mut Token<super::Device<B>>,
    ) {
        let (texture_view_guard, _) = B::hub(global).texture_views.read(token);
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

    pub(crate) fn handle_mapping<G: GlobalIdentityHandlerFactory>(
        &mut self,
        global: &Global<G>,
        raw: &B::Device,
        trackers: &Mutex<TrackerSet>,
        token: &mut Token<super::Device<B>>,
    ) -> Vec<super::BufferMapPendingCallback> {
        if self.ready_to_map.is_empty() {
            return Vec::new();
        }
        let hub = B::hub(global);
        let (mut buffer_guard, _) = B::hub(global).buffers.write(token);
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
                    resource::BufferMapState::Active,
                ) {
                    resource::BufferMapState::Waiting(pending_mapping) => pending_mapping,
                    _ => panic!("No pending mapping."),
                };
                log::debug!("Buffer {:?} map state -> Active", buffer_id);
                let result = match mapping.op {
                    resource::BufferMapOperation::Read { .. } => {
                        super::map_buffer(raw, buffer, mapping.sub_range, super::HostMap::Read)
                    }
                    resource::BufferMapOperation::Write { .. } => {
                        super::map_buffer(raw, buffer, mapping.sub_range, super::HostMap::Write)
                    }
                };
                pending_callbacks.push((mapping.op, result));
            }
        }
        pending_callbacks
    }
}
