use thiserror::Error;
use wgt::error::{ErrorType, WebGpuError};

use crate::{
    command::{CommandEncoder, CommandEncoderError, EncoderStateError},
    device::DeviceError,
    global::Global,
    id::{BufferId, CommandEncoderId, TextureId},
    resource::{InvalidResourceError, ParentDevice},
    track::ResourceUsageCompatibilityError,
};

impl Global {
    pub fn command_encoder_transition_resources(
        &self,
        command_encoder_id: CommandEncoderId,
        buffer_transitions: impl Iterator<Item = wgt::BufferTransition<BufferId>>,
        texture_transitions: impl Iterator<Item = wgt::TextureTransition<TextureId>>,
    ) -> Result<(), EncoderStateError> {
        profiling::scope!("CommandEncoder::transition_resources");

        let hub = &self.hub;

        // Lock command encoder for recording
        let cmd_enc = hub.command_encoders.get(command_encoder_id);
        let mut cmd_buf_data = cmd_enc.data.lock();
        cmd_buf_data.record_with(|cmd_buf_data| -> Result<(), CommandEncoderError> {
            // Get and lock device
            let device = &cmd_enc.device;
            device.check_is_valid()?;
            let snatch_guard = &device.snatchable_lock.read();

            let mut usage_scope = device.new_usage_scope();
            let indices = &device.tracker_indices;
            usage_scope.buffers.set_size(indices.buffers.size());
            usage_scope.textures.set_size(indices.textures.size());

            // Process buffer transitions
            for buffer_transition in buffer_transitions {
                let buffer = hub.buffers.get(buffer_transition.buffer).get()?;
                buffer.same_device_as(cmd_enc.as_ref())?;

                usage_scope
                    .buffers
                    .merge_single(&buffer, buffer_transition.state)?;
            }

            // Process texture transitions
            for texture_transition in texture_transitions {
                let texture = hub.textures.get(texture_transition.texture).get()?;
                texture.same_device_as(cmd_enc.as_ref())?;

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
            CommandEncoder::insert_barriers_from_scope(
                cmd_buf_raw,
                &mut cmd_buf_data.trackers,
                &usage_scope,
                snatch_guard,
            );
            Ok(())
        })
    }
}

/// Error encountered while attempting to perform [`Global::command_encoder_transition_resources`].
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum TransitionResourcesError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    EncoderState(#[from] EncoderStateError),
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
    #[error(transparent)]
    ResourceUsage(#[from] ResourceUsageCompatibilityError),
}

impl WebGpuError for TransitionResourcesError {
    fn webgpu_error_type(&self) -> ErrorType {
        let e: &dyn WebGpuError = match self {
            Self::Device(e) => e,
            Self::EncoderState(e) => e,
            Self::InvalidResource(e) => e,
            Self::ResourceUsage(e) => e,
        };
        e.webgpu_error_type()
    }
}
