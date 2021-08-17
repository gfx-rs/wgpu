#[cfg(feature = "trace")]
use crate::device::trace::Action;
use crate::{
    command::{
        extract_texture_selector, validate_linear_texture_data, validate_texture_copy_range,
        CommandBuffer, CopySide, ImageCopyTexture, TransferError,
    },
    conv,
    device::{Device, DeviceError, Device as Queue, WaitIdleError},
    hub::{Global, GlobalIdentityHandlerFactory, HalApi, Token},
    id::{self, Dummy},
    resource::{BufferAccessError, BufferMapState, TextureInner},
    track::{self, ResourceTracker, TrackerSet},
    FastHashSet,
};

use core::{iter, mem, num::NonZeroU32, ptr};
use hal::{CommandEncoder as _, Device as _, Queue as _};
use parking_lot::Mutex;
use thiserror::Error;

/// Number of command buffers that we generate from the same pool
/// for the write_xxx commands, before the pool is recycled.
///
/// If we don't stop at some point, the pool will grow forever,
/// without a concrete moment of when it can be cleared.
const WRITE_COMMAND_BUFFERS_PER_POOL: usize = 64;

pub type OnSubmittedWorkDoneCallback = unsafe extern "C" fn(user_data: *mut u8);
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SubmittedWorkDoneClosure {
    pub callback: OnSubmittedWorkDoneCallback,
    pub user_data: *mut u8,
}

unsafe impl Send for SubmittedWorkDoneClosure {}
unsafe impl Sync for SubmittedWorkDoneClosure {}

struct StagingData<A: hal::Api> {
    buffer: A::Buffer,
}

impl<A: hal::Api> StagingData<A> {
    unsafe fn write(
        &self,
        device: &A::Device,
        offset: wgt::BufferAddress,
        data: &[u8],
    ) -> Result<(), hal::DeviceError> {
        let mapping = device.map_buffer(&self.buffer, offset..offset + data.len() as u64)?;
        ptr::copy_nonoverlapping(data.as_ptr(), mapping.ptr.as_ptr(), data.len());
        if !mapping.is_coherent {
            device
                .flush_mapped_ranges(&self.buffer, iter::once(offset..offset + data.len() as u64));
        }
        device.unmap_buffer(&self.buffer)?;
        Ok(())
    }
}

#[derive(Debug)]
pub enum TempResource<A: hal::Api> {
    Buffer(A::Buffer),
    Texture(A::Texture),
}

/// A queue execution for a particular command encoder.
pub(super) struct EncoderInFlight<A: hal::Api> {
    raw: A::CommandEncoder,
    cmd_buffers: Vec<A::CommandBuffer>,
    /// NOTE: This is deliberately not used for anything, it's just there to be dropped.
    _trackers: TrackerSet<A>,
}

impl<A: hal::Api> EncoderInFlight<A> {
    pub(super) unsafe fn land(mut self) -> A::CommandEncoder {
        self.raw.reset_all(self.cmd_buffers.into_iter());
        // NOTE: trackers get dropped automatically.
        self.raw
    }
}

#[derive(Debug)]
pub(crate) struct PendingWrites<A: hal::Api> {
    pub command_encoder: A::CommandEncoder,
    pub is_active: bool,
    pub temp_resources: Vec<TempResource<A>>,
    pub dst_buffers: FastHashSet<id::BufferId>,
    pub dst_textures: FastHashSet<id::TextureId>,
    pub executing_command_buffers: Vec<A::CommandBuffer>,
}

impl<A: hal::Api> PendingWrites<A> {
    pub fn new(command_encoder: A::CommandEncoder) -> Self {
        Self {
            command_encoder,
            is_active: false,
            temp_resources: Vec::new(),
            dst_buffers: FastHashSet::default(),
            dst_textures: FastHashSet::default(),
            executing_command_buffers: Vec::new(),
        }
    }

    /// Safety: The device has to match the device used to construct the encoder.
    pub(super) unsafe fn dispose(mut self, device: &A::Device) {
        {
            if self.is_active {
                self.command_encoder.discard_encoding();
            }
            self.command_encoder
                .reset_all(self.executing_command_buffers.into_iter());
            // NOTE: Safe because of the requirement on `dispose` (that `self` is never accessed
            // again outside of moving or dropping it).
            device.destroy_command_encoder(self.command_encoder);
        }

        for resource in self.temp_resources.drain(..) {
            match resource {
                TempResource::Buffer(buffer) => {
                    device.destroy_buffer(buffer);
                },
                TempResource::Texture(texture) => {
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
    }

    #[must_use]
    fn pre_submit(&mut self) -> Option<&A::CommandBuffer> {
        self.dst_buffers.clear();
        self.dst_textures.clear();
        if self.is_active {
            let cmd_buf = unsafe { self.command_encoder.end_encoding().unwrap() };
            self.is_active = false;
            self.executing_command_buffers.push(cmd_buf);
            self.executing_command_buffers.last()
        } else {
            None
        }
    }
}

impl<A: HalApi> PendingWrites<A> {
    #[must_use]
    fn post_submit(
        &mut self,
        command_allocator: &Mutex<super::CommandAllocator<A>>,
        device: &A::Device,
        queue: &A::Queue,
    ) -> Option<EncoderInFlight<A>> {
        if self.executing_command_buffers.len() >= WRITE_COMMAND_BUFFERS_PER_POOL {
            let new_encoder = command_allocator
                .lock()
                .acquire_encoder(device, queue)
                .unwrap();
            Some(EncoderInFlight {
                raw: mem::replace(&mut self.command_encoder, new_encoder),
                cmd_buffers: mem::take(&mut self.executing_command_buffers),
                // TODO: add more trackers
                _trackers: TrackerSet::new(A::VARIANT)
            })
        } else {
            None
        }
    }

    pub fn activate(&mut self) -> &mut A::CommandEncoder {
        if !self.is_active {
            unsafe {
                self.command_encoder
                    .begin_encoding(Some("_PendingWrites"))
                    .unwrap();
            }
            self.is_active = true;
        }
        &mut self.command_encoder
    }
}

impl<A: hal::Api> PendingWrites<A> {
    pub fn deactivate(&mut self) {
        if self.is_active {
            unsafe {
                self.command_encoder.discard_encoding();
            }
            self.is_active = false;
        }
    }
}

impl<A: hal::Api> Queue<A> {
    fn prepare_stage(&self, size: wgt::BufferAddress) -> Result<StagingData<A>, DeviceError> {
        profiling::scope!("prepare_stage");
        let stage_desc = hal::BufferDescriptor {
            label: Some("_Staging"),
            size,
            usage: hal::BufferUses::MAP_WRITE | hal::BufferUses::COPY_SRC,
            memory_flags: hal::MemoryFlags::TRANSIENT,
        };
        let buffer = unsafe { self.raw.create_buffer(&stage_desc)? };
        Ok(StagingData { buffer })
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
    #[error("surface output was dropped before the command buffer got submitted")]
    SurfaceOutputDropped,
    #[error("surface was unconfigured before the command buffer got submitted")]
    SurfaceUnconfigured,
    #[error("GPU got stuck :(")]
    StuckGpu,
}

//TODO: move out common parts of write_xxx.

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn queue_write_buffer<A: HalApi>(
        &self,
        queue_id: /*id::QueueId*/id::IdGuard<A, Queue<Dummy>>,
        buffer_id: id::BufferId,
        buffer_offset: wgt::BufferAddress,
        data: &[u8],
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("write_buffer", "Queue");

        let hub = A::hub(self);
        let mut token_ = Token::root();
        // let (mut device_guard, mut token) = hub.devices.write(&mut token);
        let device = /*device_guard
            .get_mut(queue_id)
            .map_err(|_| DeviceError::Invalid)?*/&*queue_id;
        let (buffer_guard, mut token1) = hub.buffers.read(&mut token_);

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
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

        let stage = device.prepare_stage(data_size)?;
        unsafe { stage.write(&device.raw, 0, data) }.map_err(DeviceError::from)?;

        let (mut trackers, mut token2) = token1.lock(&device.trackers);
        let (dst, transition) = trackers
            .buffers
            .use_replace(&*buffer_guard, buffer_id, (), hal::BufferUses::COPY_DST)
            .map_err(TransferError::InvalidBuffer)?;
        let dst_raw = dst
            .raw
            .as_ref()
            .ok_or(TransferError::InvalidBuffer(buffer_id))?;
        if !dst.usage.contains(wgt::BufferUsages::COPY_DST) {
            return Err(TransferError::MissingCopyDstUsageFlag(Some(buffer_id), None).into());
        }

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
            usage: hal::BufferUses::MAP_WRITE..hal::BufferUses::COPY_SRC,
        })
        .chain(transition.map(|pending| pending.into_hal(dst)));
        {
            let (mut queue_inner_guard, _) = token2.lock(&device.queue.inner);
            // let pending_writes = token.lock(&device.queue.pending_writes).0;
            // let pending_writes = &mut queue_inner_guard.pending_writes;
            // NOTE: Must wait until the pending writes are locked to assign a submission index, to
            // make sure we don't miss the scheduled writes.
            dst.life_guard.use_at(queue_inner_guard.active_submission_index + 1);
            let encoder = queue_inner_guard.pending_writes.activate();
            unsafe {
                encoder.transition_buffers(barriers);
                encoder.copy_buffer_to_buffer(&stage.buffer, dst_raw, region.into_iter());
            }

            queue_inner_guard.pending_writes.consume(stage);
            queue_inner_guard.pending_writes.dst_buffers.insert(buffer_id);
        }

        // Ensure the overwritten bytes are marked as initialized so they don't need to be nulled prior to mapping or binding.
        {
            drop((trackers, token2));
            drop((buffer_guard, token1));
            let (mut buffer_guard, _) = hub.buffers.write(&mut token_);

            let dst = buffer_guard.get_mut(buffer_id).unwrap();
            dst.initialization_status
                .clear(buffer_offset..(buffer_offset + data_size));
        }

        Ok(())
    }

    pub fn queue_write_texture<A: HalApi>(
        &self,
        queue_id: id::IdGuard<A, Queue<Dummy>>,
        destination: &ImageCopyTexture,
        data: &[u8],
        data_layout: &wgt::ImageDataLayout,
        size: &wgt::Extent3d,
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("write_texture", "Queue");

        let hub = A::hub(self);
        let mut token = Token::root();
        // let (mut device_guard, mut token) = hub.devices.write(&mut token);
        // let (mut queue_inner_guard, mut token) = token.lock(&self.queue.inner);
        let device = /*device_guard
            .get_mut(queue_id)
            .map_err(|_| DeviceError::Invalid)?*/&*queue_id;

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
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

        let (texture_guard, mut token) = hub.textures.read(&mut token);
        let (selector, dst_base, texture_format) =
            extract_texture_selector(destination, size, &*texture_guard)?;
        let format_desc = texture_format.describe();
        let (_, bytes_per_array_layer) = validate_linear_texture_data(
            data_layout,
            texture_format,
            data.len() as wgt::BufferAddress,
            CopySide::Source,
            format_desc.block_size as wgt::BufferAddress,
            size,
            false,
        )?;

        if !conv::is_valid_copy_dst_texture_format(texture_format) {
            return Err(TransferError::CopyToForbiddenTextureFormat(texture_format).into());
        }
        let (block_width, block_height) = format_desc.block_dimensions;
        let width_blocks = size.width / block_width as u32;
        let height_blocks = size.height / block_height as u32;

        let block_rows_per_image = match data_layout.rows_per_image {
            Some(rows_per_image) => rows_per_image.get(),
            None => {
                // doesn't really matter because we need this only if we copy more than one layer, and then we validate for this being not None
                size.height
            }
        };

        let bytes_per_row_alignment = get_lowest_common_denom(
            device.adapter_id.raw.capabilities.alignments.buffer_copy_pitch.get() as u32,
            format_desc.block_size as u32,
        );
        let stage_bytes_per_row = align_to(
            format_desc.block_size as u32 * width_blocks,
            bytes_per_row_alignment,
        );

        let block_rows_in_copy =
            (size.depth_or_array_layers - 1) * block_rows_per_image + height_blocks;
        let stage_size = stage_bytes_per_row as u64 * block_rows_in_copy as u64;
        let stage = device.prepare_stage(stage_size)?;

        let (mut trackers, mut token) = token.lock(&device.trackers);
        let (dst, transition) = trackers
            .textures
            .use_replace(
                &*texture_guard,
                destination.texture,
                selector,
                hal::TextureUses::COPY_DST,
            )
            .unwrap();
        let dst_raw = dst
            .inner
            .as_raw()
            .ok_or(TransferError::InvalidTexture(destination.texture))?;

        if !dst.desc.usage.contains(wgt::TextureUsages::COPY_DST) {
            return Err(
                TransferError::MissingCopyDstUsageFlag(None, Some(destination.texture)).into(),
            );
        }
        let (hal_copy_size, array_layer_count) =
            validate_texture_copy_range(destination, &dst.desc, CopySide::Destination, size)?;

        let bytes_per_row = if let Some(bytes_per_row) = data_layout.bytes_per_row {
            bytes_per_row.get()
        } else {
            width_blocks * format_desc.block_size as u32
        };

        let mapping = unsafe { device.raw.map_buffer(&stage.buffer, 0..stage_size) }
            .map_err(DeviceError::from)?;
        unsafe {
            profiling::scope!("copy");
            if stage_bytes_per_row == bytes_per_row {
                // Fast path if the data isalready being aligned optimally.
                ptr::copy_nonoverlapping(data.as_ptr(), mapping.ptr.as_ptr(), stage_size as usize);
            } else {
                // Copy row by row into the optimal alignment.
                let copy_bytes_per_row = stage_bytes_per_row.min(bytes_per_row) as usize;
                for layer in 0..size.depth_or_array_layers {
                    let rows_offset = layer * block_rows_per_image;
                    for row in 0..height_blocks {
                        ptr::copy_nonoverlapping(
                            data.as_ptr()
                                .offset((rows_offset + row) as isize * bytes_per_row as isize),
                            mapping.ptr.as_ptr().offset(
                                (rows_offset + row) as isize * stage_bytes_per_row as isize,
                            ),
                            copy_bytes_per_row,
                        );
                    }
                }
            }
        }
        unsafe {
            if !mapping.is_coherent {
                device
                    .raw
                    .flush_mapped_ranges(&stage.buffer, iter::once(0..stage_size));
            }
            device
                .raw
                .unmap_buffer(&stage.buffer)
                .map_err(DeviceError::from)?;
        }

        let regions = (0..array_layer_count).map(|rel_array_layer| {
            let mut texture_base = dst_base.clone();
            texture_base.array_layer += rel_array_layer;
            hal::BufferTextureCopy {
                buffer_layout: wgt::ImageDataLayout {
                    offset: rel_array_layer as u64 * bytes_per_array_layer,
                    bytes_per_row: NonZeroU32::new(stage_bytes_per_row),
                    rows_per_image: NonZeroU32::new(block_rows_per_image),
                },
                texture_base,
                size: hal_copy_size,
            }
        });
        let barrier = hal::BufferBarrier {
            buffer: &stage.buffer,
            usage: hal::BufferUses::MAP_WRITE..hal::BufferUses::COPY_SRC,
        };

        {
            let (mut queue_inner_guard, _) = token.lock(&device.queue.inner);
            // let pending_writes = token.lock(&device.queue.pending_writes).0;
            // let pending_writes = &mut queue_inner_guard.pending_writes;
            // NOTE: Must wait until the pending writes are locked to assign a submission index, to
            // make sure we don't miss the scheduled writes.
            dst.life_guard.use_at(queue_inner_guard.active_submission_index + 1);
            let encoder = queue_inner_guard.pending_writes.activate();
            unsafe {
                encoder.transition_buffers(iter::once(barrier));
                encoder.transition_textures(transition.map(|pending| pending.into_hal(dst)));
                encoder.copy_buffer_to_texture(&stage.buffer, dst_raw, regions);
            }

            queue_inner_guard.pending_writes.consume(stage);
            queue_inner_guard.pending_writes
                .dst_textures
                .insert(destination.texture);
        }

        Ok(())
    }

    pub fn queue_submit<A: HalApi + core::fmt::Debug, I: Iterator<Item=id::CommandBufferId>>(
        &self,
        queue_id: /*id::QueueId*/id::IdGuard<A, Queue<Dummy>>,
        command_buffer_ids: /*&[id::CommandBufferId]*/I,
    ) -> Result<(), QueueSubmitError> {
        profiling::scope!("submit", "Queue");

        let callbacks = {
            let hub = A::hub(self);
            let mut token = Token::root();

            // let (mut device_guard, mut token) = hub.devices.write(&mut token);
            let device = /*device_guard
                .get_mut(*/&*queue_id/*)
                .map_err(|_| DeviceError::Invalid)?*/;
            let (mut temp_suspected, mut token) = token.lock(&hub.temp_suspected);
            temp_suspected.clear();

            // FIXME: I didn't bother to optimize the locks here, for the simple reason that they
            // are all (hopefully!) going to go away after the hubs refactor.  But if any are still
            // here, remember to try to drop as soon as possible to avoid blocking the queue.
            // let (mut swap_chain_guard, mut token) = hub.swap_chains.write(&mut token);
            // let (mut bgl_guard, mut token) = hub.bind_group_layouts.write(&mut token);
            let (mut buffer_guard, mut token) = hub.buffers.write(&mut token);
            let (mut texture_guard, mut token) = hub.textures.write(&mut token);
            let (mut trackers, mut token) = token.lock::<super::Trackers<A>>(&device.trackers);
            let (mut queue_inner_guard, mut token) = token.lock(&device.queue.inner);
            // let pending_writes = token.lock(&device.queue.pending_writes).0;
            let super::QueueInner {
                ref mut pending_writes,
                raw: ref mut queue,
                ref mut fence,
                ref mut active_submission_index,
            } = &mut **queue_inner_guard;

            // NOTE: Must wait until the pending writes are locked to assign a submission index, to
            // make sure we synchronize with reads of the index from concurrent writers trying to add
            // new pending operations.
            let submit_index = active_submission_index.checked_add(1)
                // TODO: Abort?  Not clear just panicking is sound here (especially if we switch
                // this to an atomic).
                .expect("SubmissionIndex overflow: please try to avoid submitting to a queue more than\
                        u64::MAX times! (in the likely case that you did not actually do this, this is\
                        probably a bug in wgpu-core)");
            *active_submission_index = submit_index;
            let mut active_executions = Vec::new();
            let mut used_surface_textures = track::ResourceTracker::new(A::VARIANT);

            {
                let (mut command_buffer_guard, _) = hub.command_buffers.write(&mut token);

                // NOTE: If the iterator lies about its upper bound, the worst thing that can
                // happen is that no commands are processed; we don't rely on this for memory
                // safety, only correctness.
                if command_buffer_ids.size_hint().1 != Some(0) {
                    profiling::scope!("prepare");

                    // let (render_bundle_guard, mut token) = hub.render_bundles.read(&mut token);
                    // let (_, mut token) = hub.pipeline_layouts.read(&mut token);
                    // let (bind_group_guard, mut token) = hub.bind_groups.read(&mut token);
                    // let (compute_pipe_guard, mut token) = hub.compute_pipelines.read(&mut token);
                    // let (render_pipe_guard, mut token) = hub.render_pipelines.read(&mut token);
                    // let (query_set_guard, mut token) = hub.query_sets.read(&mut token);
                    // let (texture_view_guard, _) = hub.texture_views.read(&mut token);
                    // let (sampler_guard, _) = hub.samplers.read(&mut token);

                    //TODO: if multiple command buffers are submitted, we can re-use the last
                    // native command buffer of the previous chain instead of always creating
                    // a temporary one, since the chains are not finished.

                    // finish all the command buffers first
                    for cmb_id in command_buffer_ids {
                        let mut cmdbuf = match hub
                            .command_buffers
                            .unregister_locked(cmb_id, &mut *command_buffer_guard)
                        {
                            Some(cmdbuf) => cmdbuf,
                            None => continue,
                        };
                        #[cfg(feature = "trace")]
                        if let Some(ref trace) = device.trace {
                            trace.add(Action::Submit(
                                submit_index,
                                cmdbuf.commands.take().unwrap(),
                            ));
                        }
                        if !cmdbuf.is_finished() {
                            /*device.*/Device::destroy_command_buffer(cmdbuf);
                            continue;
                        }

                        // optimize the tracked states
                        cmdbuf.trackers.optimize();

                        let mut baked = cmdbuf.into_baked();

                        // update submission IDs
                        for id in baked.trackers.buffers.used() {
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
                                temp_suspected.buffers.push(id);
                            } else {
                                match buffer.map_state {
                                    BufferMapState::Idle => (),
                                    _ => panic!("Buffer {:?} is still mapped", id),
                                }
                            }
                        }
                        for id in baked.trackers.textures.used() {
                            let texture = &texture_guard[id];
                            match texture.inner {
                                TextureInner::Native { raw: None } => {
                                    return Err(QueueSubmitError::DestroyedTexture(id.0));
                                }
                                TextureInner::Native { raw: Some(_) } => {}
                                TextureInner::Surface { .. } => {
                                    use track::ResourceState as _;
                                    let ref_count = baked.trackers.textures.get_ref_count(id);
                                    //TODO: better error handling here?
                                    {
                                        // first, register it in the device tracker with uninitialized,
                                        // if it wasn't used before.
                                        let mut ts = track::TextureState::default();
                                        let _ = ts.change(
                                            &id,
                                            texture.full_range.clone(),
                                            hal::TextureUses::UNINITIALIZED,
                                            None,
                                        );
                                        let _ = trackers.textures.init(
                                            id,
                                            ref_count.clone(),
                                            ts.clone(),
                                        );
                                    }
                                    {
                                        // then, register it in the temporary tracker.
                                        let mut ts = track::TextureState::default();
                                        let _ = ts.change(
                                            &id,
                                            texture.full_range.clone(),
                                            hal::TextureUses::empty(),
                                            None,
                                        );
                                        let _ =
                                            used_surface_textures.init(id, ref_count.clone(), ts);
                                    }
                                }
                            }
                            if !texture.life_guard.use_at(submit_index) {
                                temp_suspected.textures.push(id);
                            }
                        }
                        /* for id in baked.trackers.views.used() {
                            if !texture_view_guard[id].life_guard.use_at(submit_index) {
                                temp_suspected.texture_views.push(id);
                            }
                        } */
                        /* for bg in baked.trackers.bind_groups.used() {
                            /* if !bg.life_guard.use_at(submit_index) {
                                temp_suspected.bind_groups.push(id);
                            } */
                            // We need to update the submission indices for the contained
                            // state-less (!) resources as well, so that they don't get
                            // deleted too early if the parent bind group goes out of scope.
                            /* for sub_id in bg.used.views.used() {
                                texture_view_guard[sub_id].life_guard.use_at(submit_index);
                            } */
                            /* for sub_id in bg.used.samplers.used() {
                                sampler_guard[sub_id].life_guard.use_at(submit_index);
                            } */
                        } */
                        assert!(baked.trackers.samplers.is_empty());
                        /* for id in baked.trackers.compute_pipes.used() {
                            if !compute_pipe_guard[id].life_guard.use_at(submit_index) {
                                temp_suspected.compute_pipelines.push(id);
                            }
                        } */
                        /* for id in baked.trackers.render_pipes.used() {
                            if !render_pipe_guard[id].life_guard.use_at(submit_index) {
                                temp_suspected.render_pipelines.push(id);
                            }
                        } */
                        /* for id in baked.trackers.query_sets.used() {
                            if !query_set_guard[id].life_guard.use_at(submit_index) {
                                temp_suspected.query_sets.push(id);
                            }
                        } */
                        /* for id in baked.trackers.bundles.used() {
                            let bundle = &render_bundle_guard[id];
                            if !bundle.life_guard.use_at(submit_index) {
                                temp_suspected.render_bundles.push(id);
                            }
                            // We need to update the submission indices for the contained
                            // state-less (!) resources as well, excluding the bind groups.
                            // They don't get deleted too early if the bundle goes out of scope.
                            // FIXME: Until render bundles are Arc'd, these being commented out is
                            // unsound!
                            /* for sub_id in bundle.used.compute_pipes.used() {
                                compute_pipe_guard[sub_id].life_guard.use_at(submit_index);
                            } */
                            /* for sub_id in bundle.used.render_pipes.used() {
                                render_pipe_guard[sub_id].life_guard.use_at(submit_index);
                            } */
                        } */

                        // execute resource transitions
                        unsafe {
                            baked
                                .encoder
                                .begin_encoding(Some("_Transit"))
                                .map_err(DeviceError::from)?
                        };
                        log::trace!("Stitching command buffer {:?} before submission", cmb_id);
                        baked
                            .initialize_buffer_memory(&mut *trackers, &mut *buffer_guard)
                            .map_err(|err| QueueSubmitError::DestroyedBuffer(err.0))?;
                        //Note: stateless trackers are not merged:
                        // device already knows these resources exist.
                        CommandBuffer::insert_barriers(
                            &mut baked.encoder,
                            &mut *trackers,
                            &baked.trackers.buffers,
                            &baked.trackers.textures,
                            &*buffer_guard,
                            &*texture_guard,
                        );

                        // FIXME: Remove this.
                        //
                        // Creating new TrackerSet (temporarily) to hold the contents of any
                        // tracked resources not in the hub.
                        let backend = A::VARIANT;
                        let temp_tracker_set = TrackerSet {
                            buffers: ResourceTracker::new(backend),
                            textures: ResourceTracker::new(backend),
                            views: baked.trackers.views,
                            bind_groups: baked.trackers.bind_groups,
                            samplers: baked.trackers.samplers,
                            compute_pipes: baked.trackers.compute_pipes,
                            render_pipes: baked.trackers.render_pipes,
                            bundles: baked.trackers.bundles,
                            query_sets: baked.trackers.query_sets,
                        };

                        let transit = unsafe { baked.encoder.end_encoding().unwrap() };
                        baked.list.insert(0, transit);

                        // Transition surface textures into `Present` state.
                        // Note: we could technically do it after all of the command buffers,
                        // but here we have a command encoder by hand, so it's easier to use it.
                        if !used_surface_textures.is_empty() {
                            unsafe {
                                baked
                                    .encoder
                                    .begin_encoding(Some("_Present"))
                                    .map_err(DeviceError::from)?
                            };
                            let texture_barriers = trackers
                                .textures
                                .merge_replace(&used_surface_textures)
                                .map(|pending| {
                                    let tex = &texture_guard[pending.id];
                                    pending.into_hal(tex)
                                });
                            let present = unsafe {
                                baked.encoder.transition_textures(texture_barriers);
                                baked.encoder.end_encoding().unwrap()
                            };
                            baked.list.push(present);
                            used_surface_textures.clear();
                        }

                        // done
                        active_executions.push(EncoderInFlight {
                            raw: baked.encoder,
                            cmd_buffers: baked.list,
                            _trackers: temp_tracker_set,
                        });
                    }

                    log::trace!("Device after submission {}: {:#?}", submit_index, trackers);
                }

                let refs = pending_writes
                    .pre_submit()
                    .into_iter()
                    .chain(
                        active_executions
                            .iter()
                            .flat_map(|pool_execution| pool_execution.cmd_buffers.iter()),
                    )
                    .collect::<Vec<_>>();
                unsafe {
                    queue
                        .submit(&refs, Some((fence, submit_index)))
                        .map_err(DeviceError::from)?;
                }
            }

            profiling::scope!("cleanup");
            if let Some(pending_execution) = pending_writes.post_submit(
                &device.queue.command_allocator,
                &device.raw,
                queue,
            ) {
                active_executions.push(pending_execution);
            }

            // this will register the new submission to the life time tracker
            let mut pending_write_resources = mem::take(&mut pending_writes.temp_resources);
            token.lock(&device.queue.life_tracker).0.track_submission(
                submit_index,
                pending_write_resources.drain(..),
                active_executions,
            );

            // This will schedule destruction of all resources that are no longer needed
            // by the user but used in the command stream, among other things.
            let closures = match device.maintain(hub, false, &*temp_suspected, /* &mut *bgl_guard, */&mut *buffer_guard, &mut *texture_guard, &mut *trackers, &*queue_inner_guard, &mut token) {
                Ok(closures) => closures,
                Err(WaitIdleError::Device(err)) => return Err(QueueSubmitError::Queue(err)),
                Err(WaitIdleError::StuckGpu) => return Err(QueueSubmitError::StuckGpu),
            };

            queue_inner_guard.pending_writes.temp_resources = pending_write_resources;
            temp_suspected.clear();
            token.lock(&device.queue.life_tracker).0.post_submit();

            closures
        };

        // the closures should execute with nothing locked!
        unsafe {
            callbacks.fire();
        }
        Ok(())
    }

    pub fn queue_get_timestamp_period<A: HalApi>(
        &self,
        _queue_id: /*id::QueueId*/id::IdGuard<A, Queue<Dummy>>,
    ) -> /*Result<f32, InvalidQueue>*/f32 {
        /* let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, _) = hub.devices.read(&mut token);
        match device_guard.get(queue_id) {
            Ok(_device) => Ok(*/1.0/*),*/ //TODO?
            /*Err(_) => Err(InvalidQueue),
        } */
    }

    pub fn queue_on_submitted_work_done<A: HalApi>(
        // &self,
        queue_id: /*id::QueueId*/id::IdGuard<A, Queue<Dummy>>,
        closure: SubmittedWorkDoneClosure,
    ) -> Result<(), InvalidQueue> {
        //TODO: flush pending writes
        let mut token = Token::root();
        let added = {
            // let hub = A::hub(self);
            // let (device_guard, mut token) = hub.devices.read(&mut token);
            /*match device_guard.get(queue_id) {
                Ok(device) => device.lock_life(&mut token)*/token.lock(&queue_id.queue.life_tracker).0.add_work_done_closure(closure)/*,
                Err(_) => return Err(InvalidQueue),
            }*/
        };
        if !added {
            unsafe {
                (closure.callback)(closure.user_data);
            }
        }
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
