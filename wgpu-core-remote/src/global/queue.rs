use core::ptr::NonNull;

use wgpu_core::device::queue::{QueueSubmitError, QueueWriteError, SubmittedWorkDoneClosure};
use wgpu_core::SubmissionIndex;

use crate::global::Global;
use crate::id::{BufferId, CommandBufferId, QueueId, StagingBufferId, TextureId};

impl Global {
    pub fn queue_write_buffer(
        &self,
        queue_id: QueueId,
        buffer_id: BufferId,
        buffer_offset: wgt::BufferAddress,
        data: &[u8],
    ) -> Result<(), QueueWriteError> {
        let queue = self.hub.queues.get(queue_id);
        let buffer = self.hub.buffers.get(buffer_id);

        queue.write_buffer(buffer, buffer_offset, data)
    }

    pub fn queue_create_staging_buffer(
        &self,
        queue_id: QueueId,
        buffer_size: wgt::BufferSize,
        id_in: Option<StagingBufferId>,
    ) -> Result<(StagingBufferId, NonNull<u8>), QueueWriteError> {
        let queue = self.hub.queues.get(queue_id);
        let (staging_buffer, ptr) = queue.create_staging_buffer(buffer_size)?;

        let fid = self.hub.staging_buffers.prepare(id_in);
        let id = fid.assign(staging_buffer);

        Ok((id, ptr))
    }

    pub fn queue_write_staging_buffer(
        &self,
        queue_id: QueueId,
        buffer_id: BufferId,
        buffer_offset: wgt::BufferAddress,
        staging_buffer_id: StagingBufferId,
    ) -> Result<(), QueueWriteError> {
        let queue = self.hub.queues.get(queue_id);
        let buffer = self.hub.buffers.get(buffer_id);
        let staging_buffer = self.hub.staging_buffers.remove(staging_buffer_id);
        queue.write_staging_buffer(buffer, buffer_offset, staging_buffer)
    }

    pub fn queue_validate_write_buffer(
        &self,
        queue_id: QueueId,
        buffer_id: BufferId,
        buffer_offset: u64,
        buffer_size: wgt::BufferSize,
    ) -> Result<(), QueueWriteError> {
        let queue = self.hub.queues.get(queue_id);
        let buffer = self.hub.buffers.get(buffer_id);
        queue.validate_write_buffer(buffer, buffer_offset, buffer_size)
    }

    pub fn queue_write_texture(
        &self,
        queue_id: QueueId,
        destination: &wgt::TexelCopyTextureInfo<TextureId>,
        data: &[u8],
        data_layout: &wgt::TexelCopyBufferLayout,
        size: &wgt::Extent3d,
    ) -> Result<(), QueueWriteError> {
        let queue = self.hub.queues.get(queue_id);
        let texture = self.hub.textures.get(destination.texture);
        let destination = wgt::TexelCopyTextureInfo {
            texture,
            mip_level: destination.mip_level,
            origin: destination.origin,
            aspect: destination.aspect,
        };

        queue.write_texture(destination, data, data_layout, size)
    }

    pub fn queue_submit(
        &self,
        queue_id: QueueId,
        command_buffer_ids: &[CommandBufferId],
    ) -> Result<SubmissionIndex, (SubmissionIndex, QueueSubmitError)> {
        let queue = self.hub.queues.get(queue_id);
        let command_buffer_guard = self.hub.command_buffers.read();
        let command_buffers = command_buffer_ids
            .iter()
            .map(|id| command_buffer_guard.get(*id))
            .collect::<Vec<_>>();
        drop(command_buffer_guard);
        queue.submit(&command_buffers)
    }

    pub fn queue_get_timestamp_period(&self, queue_id: QueueId) -> f32 {
        let queue = self.hub.queues.get(queue_id);

        queue.get_timestamp_period()
    }

    pub fn queue_on_submitted_work_done(
        &self,
        queue_id: QueueId,
        closure: SubmittedWorkDoneClosure,
    ) -> SubmissionIndex {
        let queue = self.hub.queues.get(queue_id);
        let result = queue.on_submitted_work_done(closure);
        result.unwrap_or(0) // '0' means no wait is necessary
    }
}
