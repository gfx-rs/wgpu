use thiserror::Error;

use crate::{
    command::CommandBuffer,
    device::DeviceError,
    global::Global,
    id::{BufferId, CommandEncoderId, TextureId},
    resource::{InvalidResourceError, ParentDevice},
    track::ResourceUsageCompatibilityError,
};

use super::CommandEncoderError;

impl Global {
    pub fn command_encoder_transition_resources(
        &self,
        command_encoder_id: CommandEncoderId,
        buffer_transitions: impl Iterator<Item = wgt::BufferTransition<BufferId>>,
        texture_transitions: impl Iterator<Item = wgt::TextureTransition<TextureId>>,
    ) -> Result<(), TransitionResourcesError> {
        profiling::scope!("CommandEncoder::transition_resources");

        let hub = &self.hub;

        // Lock command encoder for recording
        let cmd_buf = hub
            .command_buffers
            .get(command_encoder_id.into_command_buffer_id());
        let mut cmd_buf_data = cmd_buf.data.lock();
        let mut cmd_buf_data_guard = cmd_buf_data.record()?;
        let cmd_buf_data = &mut *cmd_buf_data_guard;

        // Get and lock device
        let device = &cmd_buf.device;
        device.check_is_valid()?;
        let snatch_guard = &device.snatchable_lock.read();

        let mut usage_scope = device.new_usage_scope();
        let indices = &device.tracker_indices;
        usage_scope.buffers.set_size(indices.buffers.size());
        usage_scope.textures.set_size(indices.textures.size());

        // Process buffer transitions
        for buffer_transition in buffer_transitions {
            let buffer = hub.buffers.get(buffer_transition.buffer).get()?;
            buffer.same_device_as(cmd_buf.as_ref())?;

            usage_scope
                .buffers
                .merge_single(&buffer, buffer_transition.state)?;
        }

        // Process texture transitions
        for texture_transition in texture_transitions {
            let texture = hub.textures.get(texture_transition.texture).get()?;
            texture.same_device_as(cmd_buf.as_ref())?;

            unsafe {
                usage_scope.textures.merge_single(
                    &texture,
                    texture_transition.selector,
                    texture_transition.state,
                )
            }?;
        }

        // Record any needed barriers based on tracker data
        let cmd_buf_raw = cmd_buf_data.encoder.open()?;
        CommandBuffer::insert_barriers_from_scope(
            cmd_buf_raw,
            &mut cmd_buf_data.trackers,
            &usage_scope,
            snatch_guard,
        );
        cmd_buf_data_guard.mark_successful();

        Ok(())
    }
}

/// Error encountered while attempting to perform [`Global::command_encoder_transition_resources`].
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum TransitionResourcesError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    Encoder(#[from] CommandEncoderError),
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
    #[error(transparent)]
    ResourceUsage(#[from] ResourceUsageCompatibilityError),
}
