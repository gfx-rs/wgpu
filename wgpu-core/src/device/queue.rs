/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#[cfg(feature = "trace")]
use crate::device::trace::Action;
use crate::{
    command::{
        texture_copy_view_to_hal, validate_linear_texture_data, validate_texture_copy_range,
        CommandAllocator, CommandBuffer, CopySide, TextureCopyView, TransferError, BITS_PER_BYTE,
    },
    conv,
    device::{alloc, DeviceError, WaitIdleError},
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Token},
    id,
    resource::{BufferAccessError, BufferMapState, BufferUse, TextureUse},
    span, FastHashSet,
};

use hal::{command::CommandBuffer as _, device::Device as _, queue::CommandQueue as _};
use smallvec::SmallVec;
use std::{iter, ptr};
use thiserror::Error;

struct StagingData<B: hal::Backend> {
    buffer: B::Buffer,
    memory: alloc::MemoryBlock<B>,
    cmdbuf: B::CommandBuffer,
}

#[derive(Debug)]
pub enum TempResource<B: hal::Backend> {
    Buffer(B::Buffer),
    Image(B::Image),
}

#[derive(Debug)]
pub(crate) struct PendingWrites<B: hal::Backend> {
    pub command_buffer: Option<B::CommandBuffer>,
    pub temp_resources: Vec<(TempResource<B>, alloc::MemoryBlock<B>)>,
    pub dst_buffers: FastHashSet<id::BufferId>,
    pub dst_textures: FastHashSet<id::TextureId>,
}

impl<B: hal::Backend> PendingWrites<B> {
    pub fn new() -> Self {
        Self {
            command_buffer: None,
            temp_resources: Vec::new(),
            dst_buffers: FastHashSet::default(),
            dst_textures: FastHashSet::default(),
        }
    }

    pub fn dispose(
        self,
        device: &B::Device,
        cmd_allocator: &CommandAllocator<B>,
        mem_allocator: &mut alloc::MemoryAllocator<B>,
    ) {
        if let Some(raw) = self.command_buffer {
            cmd_allocator.discard_internal(raw);
        }
        for (resource, memory) in self.temp_resources {
            mem_allocator.free(device, memory);
            match resource {
                TempResource::Buffer(buffer) => unsafe {
                    device.destroy_buffer(buffer);
                },
                TempResource::Image(image) => unsafe {
                    device.destroy_image(image);
                },
            }
        }
    }

    pub fn consume_temp(&mut self, resource: TempResource<B>, memory: alloc::MemoryBlock<B>) {
        self.temp_resources.push((resource, memory));
    }

    fn consume(&mut self, stage: StagingData<B>) {
        self.temp_resources
            .push((TempResource::Buffer(stage.buffer), stage.memory));
        self.command_buffer = Some(stage.cmdbuf);
    }

    #[must_use]
    fn finish(&mut self) -> Option<B::CommandBuffer> {
        self.dst_buffers.clear();
        self.dst_textures.clear();
        self.command_buffer.take().map(|mut cmd_buf| unsafe {
            cmd_buf.finish();
            cmd_buf
        })
    }
}

impl<B: hal::Backend> super::Device<B> {
    pub fn borrow_pending_writes(&mut self) -> &mut B::CommandBuffer {
        if self.pending_writes.command_buffer.is_none() {
            let mut cmdbuf = self.cmd_allocator.allocate_internal();
            unsafe {
                cmdbuf.begin_primary(hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
            }
            self.pending_writes.command_buffer = Some(cmdbuf);
        }
        self.pending_writes.command_buffer.as_mut().unwrap()
    }

    fn prepare_stage(&mut self, size: wgt::BufferAddress) -> Result<StagingData<B>, DeviceError> {
        let mut buffer = unsafe {
            self.raw
                .create_buffer(size, hal::buffer::Usage::TRANSFER_SRC)
                .map_err(|err| match err {
                    hal::buffer::CreationError::OutOfMemory(_) => DeviceError::OutOfMemory,
                    _ => panic!("failed to create staging buffer: {}", err),
                })?
        };
        //TODO: do we need to transition into HOST_WRITE access first?
        let requirements = unsafe {
            self.raw.set_buffer_name(&mut buffer, "<write_buffer_temp>");
            self.raw.get_buffer_requirements(&buffer)
        };

        let block = self.mem_allocator.lock().allocate(
            &self.raw,
            requirements,
            gpu_alloc::UsageFlags::UPLOAD | gpu_alloc::UsageFlags::TRANSIENT,
        )?;
        block.bind_buffer(&self.raw, &mut buffer)?;

        let cmdbuf = match self.pending_writes.command_buffer.take() {
            Some(cmdbuf) => cmdbuf,
            None => {
                let mut cmdbuf = self.cmd_allocator.allocate_internal();
                unsafe {
                    cmdbuf.begin_primary(hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
                }
                cmdbuf
            }
        };
        Ok(StagingData {
            buffer,
            memory: block,
            cmdbuf,
        })
    }
}

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
    #[error("command buffer {0:?} is invalid")]
    InvalidCommandBuffer(id::CommandBufferId),
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
    pub fn queue_write_buffer<B: GfxBackend>(
        &self,
        queue_id: id::QueueId,
        buffer_id: id::BufferId,
        buffer_offset: wgt::BufferAddress,
        data: &[u8],
    ) -> Result<(), QueueWriteError> {
        span!(_guard, INFO, "Queue::write_buffer");

        let hub = B::hub(self);
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
            tracing::trace!("Ignoring write_buffer of size 0");
            return Ok(());
        }

        let mut stage = device.prepare_stage(data_size)?;
        stage.memory.write_bytes(&device.raw, 0, data)?;

        let mut trackers = device.trackers.lock();
        let (dst, transition) = trackers
            .buffers
            .use_replace(&*buffer_guard, buffer_id, (), BufferUse::COPY_DST)
            .map_err(TransferError::InvalidBuffer)?;
        let &(ref dst_raw, _) = dst
            .raw
            .as_ref()
            .ok_or(TransferError::InvalidBuffer(buffer_id))?;
        if !dst.usage.contains(wgt::BufferUsage::COPY_DST) {
            Err(TransferError::MissingCopyDstUsageFlag(
                Some(buffer_id),
                None,
            ))?;
        }
        dst.life_guard.use_at(device.active_submission_index + 1);

        if data_size % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            Err(TransferError::UnalignedCopySize(data_size))?
        }
        if buffer_offset % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            Err(TransferError::UnalignedBufferOffset(buffer_offset))?
        }
        if buffer_offset + data_size > dst.size {
            Err(TransferError::BufferOverrun {
                start_offset: buffer_offset,
                end_offset: buffer_offset + data_size,
                buffer_size: dst.size,
                side: CopySide::Destination,
            })?
        }

        let region = hal::command::BufferCopy {
            src: 0,
            dst: buffer_offset,
            size: data.len() as _,
        };
        unsafe {
            stage.cmdbuf.pipeline_barrier(
                super::all_buffer_stages()..hal::pso::PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                iter::once(hal::memory::Barrier::Buffer {
                    states: hal::buffer::Access::HOST_WRITE..hal::buffer::Access::TRANSFER_READ,
                    target: &stage.buffer,
                    range: hal::buffer::SubRange::WHOLE,
                    families: None,
                })
                .chain(transition.map(|pending| pending.into_hal(dst))),
            );
            stage
                .cmdbuf
                .copy_buffer(&stage.buffer, dst_raw, iter::once(region));
        }

        device.pending_writes.consume(stage);
        device.pending_writes.dst_buffers.insert(buffer_id);

        Ok(())
    }

    pub fn queue_write_texture<B: GfxBackend>(
        &self,
        queue_id: id::QueueId,
        destination: &TextureCopyView,
        data: &[u8],
        data_layout: &wgt::TextureDataLayout,
        size: &wgt::Extent3d,
    ) -> Result<(), QueueWriteError> {
        span!(_guard, INFO, "Queue::write_texture");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (mut device_guard, mut token) = hub.devices.write(&mut token);
        let device = device_guard
            .get_mut(queue_id)
            .map_err(|_| DeviceError::Invalid)?;
        let (texture_guard, _) = hub.textures.read(&mut token);
        let (image_layers, image_range, image_offset) =
            texture_copy_view_to_hal(destination, size, &*texture_guard)?;

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            let mut trace = trace.lock();
            let data_path = trace.make_binary("bin", data);
            trace.add(Action::WriteTexture {
                to: destination.clone(),
                data: data_path,
                layout: data_layout.clone(),
                size: *size,
            });
        }

        if size.width == 0 || size.height == 0 || size.depth == 0 {
            tracing::trace!("Ignoring write_texture of size 0");
            return Ok(());
        }

        let texture_format = texture_guard.get(destination.texture).unwrap().format;
        let bytes_per_block = conv::map_texture_format(texture_format, device.private_features)
            .surface_desc()
            .bits as u32
            / BITS_PER_BYTE;
        validate_linear_texture_data(
            data_layout,
            texture_format,
            data.len() as wgt::BufferAddress,
            CopySide::Source,
            bytes_per_block as wgt::BufferAddress,
            size,
        )?;
        let (block_width, block_height) = conv::texture_block_size(texture_format);
        if !conv::is_valid_copy_dst_texture_format(texture_format) {
            Err(TransferError::CopyToForbiddenTextureFormat(texture_format))?
        }
        let width_blocks = size.width / block_width;
        let height_blocks = size.height / block_width;

        let texel_rows_per_image = data_layout.rows_per_image;
        let block_rows_per_image = data_layout.rows_per_image / block_height;

        let bytes_per_row_alignment = get_lowest_common_denom(
            device.hal_limits.optimal_buffer_copy_pitch_alignment as u32,
            bytes_per_block,
        );
        let stage_bytes_per_row = align_to(bytes_per_block * width_blocks, bytes_per_row_alignment);

        let block_rows_in_copy = (size.depth - 1) * block_rows_per_image + height_blocks;
        let stage_size = stage_bytes_per_row as u64 * block_rows_in_copy as u64;
        let mut stage = device.prepare_stage(stage_size)?;

        let mut trackers = device.trackers.lock();
        let (dst, transition) = trackers
            .textures
            .use_replace(
                &*texture_guard,
                destination.texture,
                image_range,
                TextureUse::COPY_DST,
            )
            .unwrap();
        let &(ref dst_raw, _) = dst
            .raw
            .as_ref()
            .ok_or(TransferError::InvalidTexture(destination.texture))?;

        if !dst.usage.contains(wgt::TextureUsage::COPY_DST) {
            Err(TransferError::MissingCopyDstUsageFlag(
                None,
                Some(destination.texture),
            ))?
        }
        validate_texture_copy_range(
            destination,
            dst.format,
            dst.kind,
            CopySide::Destination,
            size,
        )?;
        dst.life_guard.use_at(device.active_submission_index + 1);

        let ptr = stage.memory.map(&device.raw, 0, stage_size)?;
        unsafe {
            //TODO: https://github.com/zakarumych/gpu-alloc/issues/13
            if stage_bytes_per_row == data_layout.bytes_per_row {
                // Fast path if the data isalready being aligned optimally.
                ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr(), stage_size as usize);
            } else {
                // Copy row by row into the optimal alignment.
                let copy_bytes_per_row =
                    stage_bytes_per_row.min(data_layout.bytes_per_row) as usize;
                for layer in 0..size.depth {
                    let rows_offset = layer * block_rows_per_image;
                    for row in 0..height_blocks {
                        ptr::copy_nonoverlapping(
                            data.as_ptr().offset(
                                (rows_offset + row) as isize * data_layout.bytes_per_row as isize,
                            ),
                            ptr.as_ptr().offset(
                                (rows_offset + row) as isize * stage_bytes_per_row as isize,
                            ),
                            copy_bytes_per_row,
                        );
                    }
                }
            }
        }
        stage.memory.unmap(&device.raw);
        if !stage.memory.is_coherent() {
            stage.memory.flush_range(&device.raw, 0, None)?;
        }

        let region = hal::command::BufferImageCopy {
            buffer_offset: 0,
            buffer_width: (stage_bytes_per_row / bytes_per_block) * block_width,
            buffer_height: texel_rows_per_image,
            image_layers,
            image_offset,
            image_extent: conv::map_extent(size, dst.dimension),
        };
        unsafe {
            stage.cmdbuf.pipeline_barrier(
                super::all_image_stages() | hal::pso::PipelineStage::HOST
                    ..hal::pso::PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                iter::once(hal::memory::Barrier::Buffer {
                    states: hal::buffer::Access::HOST_WRITE..hal::buffer::Access::TRANSFER_READ,
                    target: &stage.buffer,
                    range: hal::buffer::SubRange::WHOLE,
                    families: None,
                })
                .chain(transition.map(|pending| pending.into_hal(dst))),
            );
            stage.cmdbuf.copy_buffer_to_image(
                &stage.buffer,
                dst_raw,
                hal::image::Layout::TransferDstOptimal,
                iter::once(region),
            );
        }

        device.pending_writes.consume(stage);
        device
            .pending_writes
            .dst_textures
            .insert(destination.texture);

        Ok(())
    }

    pub fn queue_submit<B: GfxBackend>(
        &self,
        queue_id: id::QueueId,
        command_buffer_ids: &[id::CommandBufferId],
    ) -> Result<(), QueueSubmitError> {
        span!(_guard, INFO, "Queue::submit");

        let hub = B::hub(self);

        let callbacks = {
            let mut token = Token::root();
            let (mut device_guard, mut token) = hub.devices.write(&mut token);
            let device = device_guard
                .get_mut(queue_id)
                .map_err(|_| DeviceError::Invalid)?;
            let pending_write_command_buffer = device.pending_writes.finish();
            device.temp_suspected.clear();
            device.active_submission_index += 1;
            let submit_index = device.active_submission_index;

            let fence = {
                let mut signal_swapchain_semaphores = SmallVec::<[_; 1]>::new();
                let (mut swap_chain_guard, mut token) = hub.swap_chains.write(&mut token);
                let (mut command_buffer_guard, mut token) = hub.command_buffers.write(&mut token);

                {
                    let (bind_group_guard, mut token) = hub.bind_groups.read(&mut token);
                    let (compute_pipe_guard, mut token) = hub.compute_pipelines.read(&mut token);
                    let (render_pipe_guard, mut token) = hub.render_pipelines.read(&mut token);
                    let (mut buffer_guard, mut token) = hub.buffers.write(&mut token);
                    let (texture_guard, mut token) = hub.textures.read(&mut token);
                    let (texture_view_guard, mut token) = hub.texture_views.read(&mut token);
                    let (sampler_guard, _) = hub.samplers.read(&mut token);

                    //Note: locking the trackers has to be done after the storages
                    let mut trackers = device.trackers.lock();

                    //TODO: if multiple command buffers are submitted, we can re-use the last
                    // native command buffer of the previous chain instead of always creating
                    // a temporary one, since the chains are not finished.

                    // finish all the command buffers first
                    for &cmb_id in command_buffer_ids {
                        let cmdbuf = command_buffer_guard
                            .get_mut(cmb_id)
                            .map_err(|_| QueueSubmitError::InvalidCommandBuffer(cmb_id))?;
                        #[cfg(feature = "trace")]
                        if let Some(ref trace) = device.trace {
                            trace.lock().add(Action::Submit(
                                submit_index,
                                cmdbuf.commands.take().unwrap(),
                            ));
                        }

                        if let Some((sc_id, fbo)) = cmdbuf.used_swap_chain.take() {
                            let sc = &mut swap_chain_guard[sc_id.value];
                            sc.active_submission_index = submit_index;
                            if sc.acquired_view_id.is_none() {
                                return Err(QueueSubmitError::SwapChainOutputDropped);
                            }
                            // For each swapchain, we only want to have at most 1 signaled semaphore.
                            if sc.acquired_framebuffers.is_empty() {
                                // Only add a signal if this is the first time for this swapchain
                                // to be used in the submission.
                                signal_swapchain_semaphores.push(sc_id.value);
                            }
                            sc.acquired_framebuffers.push(fbo);
                        }

                        // optimize the tracked states
                        cmdbuf.trackers.optimize();

                        // update submission IDs
                        for id in cmdbuf.trackers.buffers.used() {
                            let buffer = &mut buffer_guard[id];
                            if buffer.raw.is_none() {
                                return Err(QueueSubmitError::DestroyedBuffer(id.0))?;
                            }
                            if !buffer.life_guard.use_at(submit_index) {
                                if let BufferMapState::Active { .. } = buffer.map_state {
                                    tracing::warn!("Dropped buffer has a pending mapping.");
                                    super::unmap_buffer(&device.raw, buffer)?;
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
                                return Err(QueueSubmitError::DestroyedTexture(id.0))?;
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

                        // execute resource transitions
                        let mut transit = device.cmd_allocator.extend(cmdbuf);
                        unsafe {
                            // the last buffer was open, closing now
                            cmdbuf.raw.last_mut().unwrap().finish();
                            transit
                                .begin_primary(hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
                        }
                        tracing::trace!("Stitching command buffer {:?} before submission", cmb_id);
                        CommandBuffer::insert_barriers(
                            &mut transit,
                            &mut *trackers,
                            &cmdbuf.trackers,
                            &*buffer_guard,
                            &*texture_guard,
                        );
                        unsafe {
                            transit.finish();
                        }
                        cmdbuf.raw.insert(0, transit);
                    }

                    tracing::trace!("Device after submission {}: {:#?}", submit_index, trackers);
                }

                // now prepare the GPU submission
                let fence = device
                    .raw
                    .create_fence(false)
                    .or(Err(DeviceError::OutOfMemory))?;
                let submission = hal::queue::Submission {
                    command_buffers: pending_write_command_buffer.as_ref().into_iter().chain(
                        command_buffer_ids
                            .iter()
                            .flat_map(|&cmb_id| &command_buffer_guard.get(cmb_id).unwrap().raw),
                    ),
                    wait_semaphores: Vec::new(),
                    signal_semaphores: signal_swapchain_semaphores
                        .into_iter()
                        .map(|sc_id| &swap_chain_guard[sc_id].semaphore),
                };

                unsafe {
                    device.queue_group.queues[0].submit(submission, Some(&fence));
                }
                fence
            };

            if let Some(comb_raw) = pending_write_command_buffer {
                device
                    .cmd_allocator
                    .after_submit_internal(comb_raw, submit_index);
            }

            let callbacks = match device.maintain(&hub, false, &mut token) {
                Ok(callbacks) => callbacks,
                Err(WaitIdleError::Device(err)) => return Err(QueueSubmitError::Queue(err)),
                Err(WaitIdleError::StuckGpu) => return Err(QueueSubmitError::StuckGpu),
            };
            super::Device::lock_life_internal(&device.life_tracker, &mut token).track_submission(
                submit_index,
                fence,
                &device.temp_suspected,
                device.pending_writes.temp_resources.drain(..),
            );

            // finally, return the command buffers to the allocator
            for &cmb_id in command_buffer_ids {
                if let (Some(cmd_buf), _) = hub.command_buffers.unregister(cmb_id, &mut token) {
                    device.cmd_allocator.after_submit(cmd_buf, submit_index);
                }
            }

            callbacks
        };

        super::fire_map_callbacks(callbacks);

        Ok(())
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
