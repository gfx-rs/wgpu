/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#[cfg(feature = "trace")]
use crate::device::trace::Action;
use crate::{
    command::{
        extract_texture_selector, validate_linear_texture_data, validate_texture_copy_range,
        CommandBuffer, CopySide, ImageCopyTexture, TransferError,
    },
    conv,
    device::{DeviceError, WaitIdleError},
    hub::{Global, GlobalIdentityHandlerFactory, HalApi, Storage, Token},
    id,
    memory_init_tracker::{MemoryInitKind, MemoryInitTrackerAction},
    resource::{Buffer, BufferAccessError, BufferMapState},
    FastHashMap, FastHashSet,
};

use hal::{CommandBuffer as _, Device as _, Queue as _};
use smallvec::SmallVec;
use std::{iter, num::NonZeroU32, ops::Range, ptr};
use thiserror::Error;

struct StagingData<A: hal::Api> {
    buffer: A::Buffer,
    cmdbuf: A::CommandBuffer,
    is_coherent: bool,
}

impl<A: hal::Api> StagingData<A> {
    unsafe fn write(
        &self,
        device: &A::Device,
        offset: wgt::BufferAddress,
        data: &[u8],
    ) -> Result<(), hal::DeviceError> {
        let ptr = device.map_buffer(&self.buffer, offset..offset + data.len() as u64)?;
        ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr(), data.len());
        device.unmap_buffer(&self.buffer)?;
        Ok(())
    }
}

#[derive(Debug)]
pub enum TempResource<A: hal::Api> {
    Buffer(A::Buffer),
    Texture(A::Texture),
}

#[derive(Debug)]
pub(crate) struct PendingWrites<A: hal::Api> {
    pub command_buffer: Option<A::CommandBuffer>,
    pub temp_resources: Vec<TempResource<A>>,
    pub dst_buffers: FastHashSet<id::BufferId>,
    pub dst_textures: FastHashSet<id::TextureId>,
}

impl<A: hal::Api> PendingWrites<A> {
    pub fn new() -> Self {
        Self {
            command_buffer: None,
            temp_resources: Vec::new(),
            dst_buffers: FastHashSet::default(),
            dst_textures: FastHashSet::default(),
        }
    }

    pub fn dispose(self, device: &A::Device) {
        if let Some(raw) = self.command_buffer {
            unsafe {
                device.destroy_command_buffer(raw);
            }
        }
        for resource in self.temp_resources {
            match resource {
                TempResource::Buffer(buffer) => unsafe {
                    device.destroy_buffer(buffer);
                },
                TempResource::Texture(texture) => unsafe {
                    device.destroy_texture(texture);
                },
            }
        }
    }

    pub fn consume_temp(&mut self, resource: TempResource<A>) {
        self.temp_resources.push(resource);
    }

    fn consume(&mut self, stage: StagingData<A>) {
        self.temp_resources.push(TempResource::Buffer(stage.buffer));
        self.command_buffer = Some(stage.cmdbuf);
    }

    #[must_use]
    fn finish(&mut self) -> Option<A::CommandBuffer> {
        self.dst_buffers.clear();
        self.dst_textures.clear();
        self.command_buffer.take().map(|mut cmd_buf| unsafe {
            cmd_buf.finish();
            cmd_buf
        })
    }

    fn create_cmd_buf(device: &A::Device) -> A::CommandBuffer {
        unsafe {
            device
                .create_command_buffer(&hal::CommandBufferDescriptor {
                    label: Some("_PendingWrites"),
                })
                .unwrap()
        }
    }

    fn borrow_cmd_buf(&mut self, device: &A::Device) -> &mut A::CommandBuffer {
        if self.command_buffer.is_none() {
            self.command_buffer = Some(Self::create_cmd_buf(device));
        }
        self.command_buffer.as_mut().unwrap()
    }
}

#[derive(Default)]
struct RequiredBufferInits {
    map: FastHashMap<id::BufferId, Vec<Range<wgt::BufferAddress>>>,
}

impl RequiredBufferInits {
    fn add<A: hal::Api>(
        &mut self,
        buffer_memory_init_actions: &[MemoryInitTrackerAction<id::BufferId>],
        buffer_guard: &mut Storage<Buffer<A>, id::BufferId>,
    ) -> Result<(), QueueSubmitError> {
        for buffer_use in buffer_memory_init_actions.iter() {
            let buffer = buffer_guard
                .get_mut(buffer_use.id)
                .map_err(|_| QueueSubmitError::DestroyedBuffer(buffer_use.id))?;

            let uninitialized_ranges = buffer.initialization_status.drain(buffer_use.range.clone());
            match buffer_use.kind {
                MemoryInitKind::ImplicitlyInitialized => {
                    uninitialized_ranges.for_each(drop);
                }
                MemoryInitKind::NeedsInitializedMemory => {
                    self.map
                        .entry(buffer_use.id)
                        .or_default()
                        .extend(uninitialized_ranges);
                }
            }
        }
        Ok(())
    }
}

impl<A: hal::Api> super::Device<A> {
    pub fn borrow_pending_writes(&mut self) -> &mut A::CommandBuffer {
        self.pending_writes.borrow_cmd_buf(&self.raw)
    }

    fn prepare_stage(&mut self, size: wgt::BufferAddress) -> Result<StagingData<A>, DeviceError> {
        profiling::scope!("prepare_stage");
        let stage_desc = hal::BufferDescriptor {
            label: Some("_Staging"),
            size,
            usage: hal::BufferUse::MAP_WRITE | hal::BufferUse::COPY_SRC,
            memory_flags: hal::MemoryFlag::TRANSIENT,
        };
        let buffer = unsafe { self.raw.create_buffer(&stage_desc)? };

        let cmdbuf = match self.pending_writes.command_buffer.take() {
            Some(cmdbuf) => cmdbuf,
            None => PendingWrites::<A>::create_cmd_buf(&self.raw),
        };
        Ok(StagingData {
            buffer,
            cmdbuf,
            is_coherent: true, //TODO
        })
    }

    fn initialize_buffer_memory(
        &mut self,
        mut required_buffer_inits: RequiredBufferInits,
        buffer_guard: &mut Storage<Buffer<A>, id::BufferId>,
    ) -> Result<(), QueueSubmitError> {
        self.pending_writes
            .dst_buffers
            .extend(required_buffer_inits.map.keys());

        let cmd_buf = self.pending_writes.borrow_cmd_buf(&self.raw);
        let mut trackers = self.trackers.lock();

        for (buffer_id, mut ranges) in required_buffer_inits.map.drain() {
            // Collapse touching ranges. We can't do this any earlier since we only now gathered ranges from several different command buffers!
            ranges.sort_by(|a, b| a.start.cmp(&b.start));
            for i in (1..ranges.len()).rev() {
                assert!(ranges[i - 1].end <= ranges[i].start); // The memory init tracker made sure of this!
                if ranges[i].start == ranges[i - 1].end {
                    ranges[i - 1].end = ranges[i].end;
                    ranges.swap_remove(i); // Ordering not important at this point
                }
            }

            // Don't do use_replace since the buffer may already no longer have a ref_count.
            // However, we *know* that it is currently in use, so the tracker must already know about it.
            let transition = trackers.buffers.change_replace_tracked(
                id::Valid(buffer_id),
                (),
                hal::BufferUse::COPY_DST,
            );
            let buffer = buffer_guard.get(buffer_id).unwrap();
            let raw_buf = buffer
                .raw
                .as_ref()
                .ok_or(QueueSubmitError::DestroyedBuffer(buffer_id))?;
            unsafe {
                cmd_buf.transition_buffers(transition.map(|pending| pending.into_hal(buffer)));
            }

            for range in ranges {
                assert!(range.start % 4 == 0, "Buffer {:?} has an uninitialized range with a start not aligned to 4 (start was {})", raw_buf, range.start);
                assert!(range.end % 4 == 0, "Buffer {:?} has an uninitialized range with an end not aligned to 4 (end was {})", raw_buf, range.end);

                unsafe {
                    cmd_buf.fill_buffer(raw_buf, range, 0);
                }
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Error)]
#[error("queue is invalid")]
pub struct InvalidQueue;

#[derive(Clone, Debug, Error)]
pub enum QueueWriteError {
    #[error(transparent)]
    Queue(#[from] DeviceError),
    #[error(transparent)]
    Transfer(#[from] TransferError),
}

#[derive(Clone, Debug, Error)]
pub enum QueueSubmitError {
    #[error(transparent)]
    Queue(#[from] DeviceError),
    #[error("buffer {0:?} is destroyed")]
    DestroyedBuffer(id::BufferId),
    #[error("texture {0:?} is destroyed")]
    DestroyedTexture(id::TextureId),
    #[error(transparent)]
    Unmap(#[from] BufferAccessError),
    #[error("swap chain output was dropped before the command buffer got submitted")]
    SwapChainOutputDropped,
    #[error("GPU got stuck :(")]
    StuckGpu,
}

//TODO: move out common parts of write_xxx.

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn queue_write_buffer<A: HalApi>(
        &self,
        queue_id: id::QueueId,
        buffer_id: id::BufferId,
        buffer_offset: wgt::BufferAddress,
        data: &[u8],
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("write_buffer", "Queue");

        let hub = A::hub(self);
        let mut token = Token::root();
        let (mut device_guard, mut token) = hub.devices.write(&mut token);
        let device = device_guard
            .get_mut(queue_id)
            .map_err(|_| DeviceError::Invalid)?;
        let (buffer_guard, _) = hub.buffers.read(&mut token);

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            let mut trace = trace.lock();
            let data_path = trace.make_binary("bin", data);
            trace.add(Action::WriteBuffer {
                id: buffer_id,
                data: data_path,
                range: buffer_offset..buffer_offset + data.len() as wgt::BufferAddress,
                queued: true,
            });
        }

        let data_size = data.len() as wgt::BufferAddress;
        if data_size == 0 {
            log::trace!("Ignoring write_buffer of size 0");
            return Ok(());
        }

        let mut stage = device.prepare_stage(data_size)?;
        unsafe { stage.write(&device.raw, 0, data) }.map_err(DeviceError::from)?;

        let mut trackers = device.trackers.lock();
        let (dst, transition) = trackers
            .buffers
            .use_replace(&*buffer_guard, buffer_id, (), hal::BufferUse::COPY_DST)
            .map_err(TransferError::InvalidBuffer)?;
        let dst_raw = dst
            .raw
            .as_ref()
            .ok_or(TransferError::InvalidBuffer(buffer_id))?;
        if !dst.usage.contains(wgt::BufferUsage::COPY_DST) {
            return Err(TransferError::MissingCopyDstUsageFlag(Some(buffer_id), None).into());
        }
        dst.life_guard.use_at(device.active_submission_index + 1);

        if data_size % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(TransferError::UnalignedCopySize(data_size).into());
        }
        if buffer_offset % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(TransferError::UnalignedBufferOffset(buffer_offset).into());
        }
        if buffer_offset + data_size > dst.size {
            return Err(TransferError::BufferOverrun {
                start_offset: buffer_offset,
                end_offset: buffer_offset + data_size,
                buffer_size: dst.size,
                side: CopySide::Destination,
            }
            .into());
        }

        let region = wgt::BufferSize::new(data.len() as u64).map(|size| hal::BufferCopy {
            src_offset: 0,
            dst_offset: buffer_offset,
            size,
        });
        let barriers = iter::once(hal::BufferBarrier {
            buffer: &stage.buffer,
            usage: hal::BufferUse::MAP_WRITE..hal::BufferUse::COPY_SRC,
        })
        .chain(transition.map(|pending| pending.into_hal(dst)));
        unsafe {
            stage.cmdbuf.transition_buffers(barriers);
            stage
                .cmdbuf
                .copy_buffer_to_buffer(&stage.buffer, dst_raw, region.into_iter());
        }

        device.pending_writes.consume(stage);
        device.pending_writes.dst_buffers.insert(buffer_id);

        // Ensure the overwritten bytes are marked as initialized so they don't need to be nulled prior to mapping or binding.
        {
            drop(buffer_guard);
            let (mut buffer_guard, _) = hub.buffers.write(&mut token);

            let dst = buffer_guard.get_mut(buffer_id).unwrap();
            dst.initialization_status
                .clear(buffer_offset..(buffer_offset + data_size));
        }

        Ok(())
    }

    pub fn queue_write_texture<A: HalApi>(
        &self,
        queue_id: id::QueueId,
        destination: &ImageCopyTexture,
        data: &[u8],
        data_layout: &wgt::ImageDataLayout,
        size: &wgt::Extent3d,
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("write_texture", "Queue");

        let hub = A::hub(self);
        let mut token = Token::root();
        let (mut device_guard, mut token) = hub.devices.write(&mut token);
        let device = device_guard
            .get_mut(queue_id)
            .map_err(|_| DeviceError::Invalid)?;

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            let mut trace = trace.lock();
            let data_path = trace.make_binary("bin", data);
            trace.add(Action::WriteTexture {
                to: destination.clone(),
                data: data_path,
                layout: *data_layout,
                size: *size,
            });
        }

        if size.width == 0 || size.height == 0 || size.depth_or_array_layers == 0 {
            log::trace!("Ignoring write_texture of size 0");
            return Ok(());
        }

        let (texture_guard, _) = hub.textures.read(&mut token);
        let (selector, texture_base, texture_format) =
            extract_texture_selector(destination, size, &*texture_guard)?;
        let format_desc = texture_format.describe();
        validate_linear_texture_data(
            data_layout,
            texture_format,
            data.len() as wgt::BufferAddress,
            CopySide::Source,
            format_desc.block_size as wgt::BufferAddress,
            size,
            false,
        )?;

        let (block_width, block_height) = format_desc.block_dimensions;
        let block_width = block_width as u32;
        let block_height = block_height as u32;

        if !conv::is_valid_copy_dst_texture_format(texture_format) {
            return Err(TransferError::CopyToForbiddenTextureFormat(texture_format).into());
        }
        let width_blocks = size.width / block_width;
        let height_blocks = size.height / block_width;

        let texel_rows_per_image = if let Some(rows_per_image) = data_layout.rows_per_image {
            rows_per_image.get()
        } else {
            // doesn't really matter because we need this only if we copy more than one layer, and then we validate for this being not None
            size.height
        };
        let block_rows_per_image = texel_rows_per_image / block_height;

        let bytes_per_row_alignment = get_lowest_common_denom(
            device.alignments.buffer_copy_pitch.get() as u32,
            format_desc.block_size as u32,
        );
        let stage_bytes_per_row = align_to(
            format_desc.block_size as u32 * width_blocks,
            bytes_per_row_alignment,
        );

        let block_rows_in_copy =
            (size.depth_or_array_layers - 1) * block_rows_per_image + height_blocks;
        let stage_size = stage_bytes_per_row as u64 * block_rows_in_copy as u64;
        let mut stage = device.prepare_stage(stage_size)?;

        let mut trackers = device.trackers.lock();
        let (dst, transition) = trackers
            .textures
            .use_replace(
                &*texture_guard,
                destination.texture,
                selector,
                hal::TextureUse::COPY_DST,
            )
            .unwrap();
        let dst_raw = dst
            .raw
            .as_ref()
            .ok_or(TransferError::InvalidTexture(destination.texture))?;

        if !dst.desc.usage.contains(wgt::TextureUsage::COPY_DST) {
            return Err(
                TransferError::MissingCopyDstUsageFlag(None, Some(destination.texture)).into(),
            );
        }
        let max_image_extent =
            validate_texture_copy_range(destination, &dst.desc, CopySide::Destination, size)?;
        dst.life_guard.use_at(device.active_submission_index + 1);

        let bytes_per_row = if let Some(bytes_per_row) = data_layout.bytes_per_row {
            bytes_per_row.get()
        } else {
            width_blocks * format_desc.block_size as u32
        };

        let ptr = unsafe { device.raw.map_buffer(&stage.buffer, 0..stage_size) }
            .map_err(DeviceError::from)?;
        unsafe {
            profiling::scope!("copy");
            if stage_bytes_per_row == bytes_per_row {
                // Fast path if the data isalready being aligned optimally.
                ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr(), stage_size as usize);
            } else {
                // Copy row by row into the optimal alignment.
                let copy_bytes_per_row = stage_bytes_per_row.min(bytes_per_row) as usize;
                for layer in 0..size.depth_or_array_layers {
                    let rows_offset = layer * block_rows_per_image;
                    for row in 0..height_blocks {
                        ptr::copy_nonoverlapping(
                            data.as_ptr()
                                .offset((rows_offset + row) as isize * bytes_per_row as isize),
                            ptr.as_ptr().offset(
                                (rows_offset + row) as isize * stage_bytes_per_row as isize,
                            ),
                            copy_bytes_per_row,
                        );
                    }
                }
            }
        }
        unsafe {
            device
                .raw
                .unmap_buffer(&stage.buffer)
                .map_err(DeviceError::from)?;
            if !stage.is_coherent {
                device
                    .raw
                    .flush_mapped_ranges(&stage.buffer, iter::once(0..stage_size));
            }
        }

        // WebGPU uses the physical size of the texture for copies whereas vulkan uses
        // the virtual size. We have passed validation, so it's safe to use the
        // image extent data directly. We want the provided copy size to be no larger than
        // the virtual size.
        let region = hal::BufferTextureCopy {
            buffer_layout: wgt::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(stage_bytes_per_row),
                rows_per_image: NonZeroU32::new(texel_rows_per_image),
            },
            texture_base,
            size: wgt::Extent3d {
                width: size.width.min(max_image_extent.width),
                height: size.height.min(max_image_extent.height),
                depth_or_array_layers: size.depth_or_array_layers,
            },
        };

        let barrier = hal::BufferBarrier {
            buffer: &stage.buffer,
            usage: hal::BufferUse::MAP_WRITE..hal::BufferUse::COPY_SRC,
        };
        unsafe {
            stage.cmdbuf.transition_buffers(iter::once(barrier));
            stage
                .cmdbuf
                .transition_textures(transition.map(|pending| pending.into_hal(dst)));
            stage
                .cmdbuf
                .copy_buffer_to_texture(&stage.buffer, dst_raw, iter::once(region));
        }

        device.pending_writes.consume(stage);
        device
            .pending_writes
            .dst_textures
            .insert(destination.texture);

        Ok(())
    }

    pub fn queue_submit<A: HalApi>(
        &self,
        queue_id: id::QueueId,
        command_buffer_ids: &[id::CommandBufferId],
    ) -> Result<(), QueueSubmitError> {
        profiling::scope!("submit", "Queue");

        let hub = A::hub(self);
        let mut token = Token::root();

        let callbacks = {
            let (mut device_guard, mut token) = hub.devices.write(&mut token);
            let device = device_guard
                .get_mut(queue_id)
                .map_err(|_| DeviceError::Invalid)?;
            let pending_write_command_buffer = device.pending_writes.finish();
            device.temp_suspected.clear();
            device.active_submission_index += 1;
            let submit_index = device.active_submission_index;

            {
                let mut signal_swapchain_semaphores = SmallVec::<[_; 1]>::new();
                let (mut swap_chain_guard, mut token) = hub.swap_chains.write(&mut token);
                let (mut command_buffer_guard, mut token) = hub.command_buffers.write(&mut token);

                if !command_buffer_ids.is_empty() {
                    profiling::scope!("prepare");

                    let (render_bundle_guard, mut token) = hub.render_bundles.read(&mut token);
                    let (_, mut token) = hub.pipeline_layouts.read(&mut token);
                    let (bind_group_guard, mut token) = hub.bind_groups.read(&mut token);
                    let (compute_pipe_guard, mut token) = hub.compute_pipelines.read(&mut token);
                    let (render_pipe_guard, mut token) = hub.render_pipelines.read(&mut token);
                    let (mut buffer_guard, mut token) = hub.buffers.write(&mut token);
                    let (texture_guard, mut token) = hub.textures.write(&mut token);
                    let (texture_view_guard, mut token) = hub.texture_views.read(&mut token);
                    let (sampler_guard, _) = hub.samplers.read(&mut token);

                    let mut required_buffer_inits = RequiredBufferInits::default();
                    //Note: locking the trackers has to be done after the storages
                    let mut trackers = device.trackers.lock();

                    //TODO: if multiple command buffers are submitted, we can re-use the last
                    // native command buffer of the previous chain instead of always creating
                    // a temporary one, since the chains are not finished.

                    // finish all the command buffers first
                    for &cmb_id in command_buffer_ids {
                        let cmdbuf = match command_buffer_guard.get_mut(cmb_id) {
                            Ok(cmdbuf) => cmdbuf,
                            Err(_) => continue,
                        };
                        #[cfg(feature = "trace")]
                        if let Some(ref trace) = device.trace {
                            trace.lock().add(Action::Submit(
                                submit_index,
                                cmdbuf.commands.take().unwrap(),
                            ));
                        }
                        if !cmdbuf.is_finished() {
                            continue;
                        }

                        required_buffer_inits
                            .add(&cmdbuf.buffer_memory_init_actions, &mut *buffer_guard)?;
                        // optimize the tracked states
                        cmdbuf.trackers.optimize();

                        for sc_id in cmdbuf.used_swap_chains.drain(..) {
                            let sc = &mut swap_chain_guard[sc_id.value];
                            if sc.acquired_texture.is_none() {
                                return Err(QueueSubmitError::SwapChainOutputDropped);
                            }
                            if sc.active_submission_index != submit_index {
                                sc.active_submission_index = submit_index;
                                // Only add a signal if this is the first time for this swapchain
                                // to be used in the submission.
                                signal_swapchain_semaphores.push(sc_id.value);
                            }
                        }

                        // update submission IDs
                        for id in cmdbuf.trackers.buffers.used() {
                            let buffer = &mut buffer_guard[id];
                            let raw_buf = match buffer.raw {
                                Some(ref raw) => raw,
                                None => {
                                    return Err(QueueSubmitError::DestroyedBuffer(id.0));
                                }
                            };
                            if !buffer.life_guard.use_at(submit_index) {
                                if let BufferMapState::Active { .. } = buffer.map_state {
                                    log::warn!("Dropped buffer has a pending mapping.");
                                    unsafe { device.raw.unmap_buffer(raw_buf) }
                                        .map_err(DeviceError::from)?;
                                }
                                device.temp_suspected.buffers.push(id);
                            } else {
                                match buffer.map_state {
                                    BufferMapState::Idle => (),
                                    _ => panic!("Buffer {:?} is still mapped", id),
                                }
                            }
                        }
                        for id in cmdbuf.trackers.textures.used() {
                            let texture = &texture_guard[id];
                            if texture.raw.is_none() {
                                return Err(QueueSubmitError::DestroyedTexture(id.0));
                            }
                            if !texture.life_guard.use_at(submit_index) {
                                device.temp_suspected.textures.push(id);
                            }
                        }
                        for id in cmdbuf.trackers.views.used() {
                            if !texture_view_guard[id].life_guard.use_at(submit_index) {
                                device.temp_suspected.texture_views.push(id);
                            }
                        }
                        for id in cmdbuf.trackers.bind_groups.used() {
                            if !bind_group_guard[id].life_guard.use_at(submit_index) {
                                device.temp_suspected.bind_groups.push(id);
                            }
                        }
                        for id in cmdbuf.trackers.samplers.used() {
                            if !sampler_guard[id].life_guard.use_at(submit_index) {
                                device.temp_suspected.samplers.push(id);
                            }
                        }
                        for id in cmdbuf.trackers.compute_pipes.used() {
                            if !compute_pipe_guard[id].life_guard.use_at(submit_index) {
                                device.temp_suspected.compute_pipelines.push(id);
                            }
                        }
                        for id in cmdbuf.trackers.render_pipes.used() {
                            if !render_pipe_guard[id].life_guard.use_at(submit_index) {
                                device.temp_suspected.render_pipelines.push(id);
                            }
                        }
                        for id in cmdbuf.trackers.bundles.used() {
                            if !render_bundle_guard[id].life_guard.use_at(submit_index) {
                                device.temp_suspected.render_bundles.push(id);
                            }
                        }

                        unsafe {
                            // the last buffer was open, closing now
                            cmdbuf.raw.last_mut().unwrap().finish();
                        }
                        // execute resource transitions
                        let mut transit = unsafe {
                            device
                                .raw
                                .create_command_buffer(&hal::CommandBufferDescriptor {
                                    label: Some("_Transit"),
                                })
                                .map_err(DeviceError::from)?
                        };
                        log::trace!("Stitching command buffer {:?} before submission", cmb_id);
                        trackers.merge_extend_stateless(&cmdbuf.trackers);
                        CommandBuffer::insert_barriers(
                            &mut transit,
                            &mut *trackers,
                            &cmdbuf.trackers.buffers,
                            &cmdbuf.trackers.textures,
                            &*buffer_guard,
                            &*texture_guard,
                        );
                        unsafe {
                            transit.finish();
                        }
                        cmdbuf.raw.insert(0, transit);
                    }

                    log::trace!("Device after submission {}: {:#?}", submit_index, trackers);
                    drop(trackers);
                    if !required_buffer_inits.map.is_empty() {
                        device
                            .initialize_buffer_memory(required_buffer_inits, &mut *buffer_guard)?;
                    }
                }

                //Note: we could technically avoid the heap Vec here
                let mut command_buffers = Vec::new();
                command_buffers.extend(pending_write_command_buffer);
                for &cmd_buf_id in command_buffer_ids.iter() {
                    match command_buffer_guard.get_mut(cmd_buf_id) {
                        Ok(cmd_buf) if cmd_buf.is_finished() => {
                            command_buffers.extend(cmd_buf.raw.drain(..));
                        }
                        _ => {}
                    }
                }

                unsafe {
                    device.queue.submit(
                        command_buffers.into_iter(),
                        Some((&mut device.fence, submit_index)),
                    );
                }
            }

            let callbacks = match device.maintain(&hub, false, &mut token) {
                Ok(callbacks) => callbacks,
                Err(WaitIdleError::Device(err)) => return Err(QueueSubmitError::Queue(err)),
                Err(WaitIdleError::StuckGpu) => return Err(QueueSubmitError::StuckGpu),
            };

            profiling::scope!("cleanup");
            super::Device::lock_life_internal(&device.life_tracker, &mut token).track_submission(
                submit_index,
                &device.temp_suspected,
                device.pending_writes.temp_resources.drain(..),
            );

            callbacks
        };

        // the map callbacks should execute with nothing locked!
        drop(token);
        super::fire_map_callbacks(callbacks);

        Ok(())
    }

    pub fn queue_get_timestamp_period<A: HalApi>(
        &self,
        queue_id: id::QueueId,
    ) -> Result<f32, InvalidQueue> {
        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, _) = hub.devices.read(&mut token);
        match device_guard.get(queue_id) {
            Ok(_device) => Ok(1.0), //TODO?
            Err(_) => Err(InvalidQueue),
        }
    }
}

fn get_lowest_common_denom(a: u32, b: u32) -> u32 {
    let gcd = if a >= b {
        get_greatest_common_divisor(a, b)
    } else {
        get_greatest_common_divisor(b, a)
    };
    a * b / gcd
}

fn get_greatest_common_divisor(mut a: u32, mut b: u32) -> u32 {
    assert!(a >= b);
    loop {
        let c = a % b;
        if c == 0 {
            return b;
        } else {
            a = b;
            b = c;
        }
    }
}

fn align_to(value: u32, alignment: u32) -> u32 {
    match value % alignment {
        0 => value,
        other => value - other + alignment,
    }
}

#[test]
fn test_lcd() {
    assert_eq!(get_lowest_common_denom(2, 2), 2);
    assert_eq!(get_lowest_common_denom(2, 3), 6);
    assert_eq!(get_lowest_common_denom(6, 4), 12);
}

#[test]
fn test_gcd() {
    assert_eq!(get_greatest_common_divisor(5, 1), 1);
    assert_eq!(get_greatest_common_divisor(4, 2), 2);
    assert_eq!(get_greatest_common_divisor(6, 4), 2);
    assert_eq!(get_greatest_common_divisor(7, 7), 7);
}
