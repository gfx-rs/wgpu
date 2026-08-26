use wgpu_core_remote_types::encoders::TexelCopyTextureInfo;
use wgpu_core_remote_types::{SubmissionIndex, SubmittedWorkDoneClosure};

use crate::global::Global;
use crate::id::{BufferId, CommandBufferId, QueueId};

impl Global {
    pub fn queue_write_buffer(
        &self,
        queue_id: QueueId,
        buffer_id: BufferId,
        buffer_offset: wgt::BufferAddress,
        data: &[u8],
    ) {
        let hub = self.hub.borrow();
        let queue = hub.queues.get(queue_id);
        let buffer = hub.buffers.get(buffer_id);

        queue.write_buffer(buffer, buffer_offset, data)
    }

    pub fn queue_write_texture(
        &self,
        queue_id: QueueId,
        destination: &TexelCopyTextureInfo,
        data: &[u8],
        data_layout: &wgt::TexelCopyBufferLayout,
        size: &wgt::Extent3d,
    ) {
        let hub = self.hub.borrow();
        let queue = hub.queues.get(queue_id);
        let texture = hub.textures.get(destination.texture);
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
    ) -> SubmissionIndex {
        let hub = self.hub.borrow();
        let queue = hub.queues.get(queue_id);
        let command_buffers = command_buffer_ids
            .iter()
            .map(|id| hub.command_buffers.get(*id))
            .collect::<Vec<_>>();
        queue.submit(&command_buffers)
    }

    pub fn queue_on_submitted_work_done(
        &self,
        queue_id: QueueId,
        closure: SubmittedWorkDoneClosure,
    ) -> SubmissionIndex {
        let hub = self.hub.borrow();
        let queue = hub.queues.get(queue_id);
        let result = queue.on_submitted_work_done(closure);
        result.unwrap_or(0) // '0' means no wait is necessary
    }
}
