use alloc::{sync::Arc, vec::Vec};

use thiserror::Error;
use wgt::error::{ErrorType, WebGpuError};

use crate::{
    command::{encoder::EncodingState, ArcCommand, CommandEncoder, EncoderStateError},
    device::DeviceError,
    global::Global,
    id::{BufferId, CommandEncoderId, TextureId},
    resource::{Buffer, InvalidResourceError, ParentDevice, Texture},
    track::ResourceUsageCompatibilityError,
};

impl CommandEncoder {
    pub fn transition_resources(
        self: &Arc<Self>,
        buffer_transitions: impl Iterator<Item = wgt::BufferTransition<Arc<Buffer>>>,
        texture_transitions: impl Iterator<Item = wgt::TextureTransition<Arc<Texture>>>,
    ) -> Result<(), EncoderStateError> {
        profiling::scope!("CommandEncoder::transition_resources");

        // Lock command encoder for recording
        let mut cmd_buf_data = self.data.lock();
        cmd_buf_data.push_with(|| -> Result<_, TransitionResourcesError> {
            Ok(ArcCommand::TransitionResources {
                buffer_transitions: buffer_transitions
                    .map(|t| {
                        t.buffer.check_is_valid()?;
                        Ok(wgt::BufferTransition {
                            buffer: t.buffer,
                            state: t.state,
                        })
                    })
                    .collect::<Result<_, TransitionResourcesError>>()?,
                texture_transitions: texture_transitions
                    .map(|t| {
                        t.texture.check_valid()?;
                        Ok(wgt::TextureTransition {
                            texture: t.texture,
                            selector: t.selector,
                            state: t.state,
                        })
                    })
                    .collect::<Result<_, TransitionResourcesError>>()?,
            })
        })
    }
}

impl Global {
    pub fn command_encoder_transition_resources(
        &self,
        command_encoder_id: CommandEncoderId,
        buffer_transitions: impl Iterator<Item = wgt::BufferTransition<BufferId>>,
        texture_transitions: impl Iterator<Item = wgt::TextureTransition<TextureId>>,
    ) -> Result<(), EncoderStateError> {
        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(command_encoder_id);
        let buffer_transitions = buffer_transitions
            .map(|t| {
                let buffer = hub.buffers.get(t.buffer);
                wgt::BufferTransition {
                    buffer,
                    state: t.state,
                }
            })
            .collect::<Vec<_>>();
        let texture_transitions = texture_transitions
            .map(|t| {
                let texture = hub.textures.get(t.texture);
                wgt::TextureTransition {
                    texture,
                    selector: t.selector,
                    state: t.state,
                }
            })
            .collect::<Vec<_>>();
        cmd_enc.transition_resources(
            buffer_transitions.into_iter(),
            texture_transitions.into_iter(),
        )
    }
}

pub(crate) fn transition_resources(
    state: &mut EncodingState,
    buffer_transitions: Vec<wgt::BufferTransition<Arc<Buffer>>>,
    texture_transitions: Vec<wgt::TextureTransition<Arc<Texture>>>,
) -> Result<(), TransitionResourcesError> {
    let mut usage_scope = state.device.new_usage_scope();
    let indices = &state.device.tracker_indices;
    usage_scope.buffers.set_size(indices.buffers.size());
    usage_scope.textures.set_size(indices.textures.size());

    // Process buffer transitions
    for buffer_transition in buffer_transitions {
        buffer_transition.buffer.same_device(state.device)?;

        usage_scope
            .buffers
            .merge_single(&buffer_transition.buffer, buffer_transition.state)?;
    }

    // Process texture transitions
    for texture_transition in texture_transitions {
        texture_transition.texture.same_device(state.device)?;

        unsafe {
            usage_scope.textures.merge_single(
                &texture_transition.texture,
                texture_transition.selector,
                texture_transition.state,
            )
        }?;
    }

    // Record any needed barriers based on tracker data
    CommandEncoder::insert_barriers_from_scope(
        state.raw_encoder,
        state.tracker,
        &usage_scope,
        state.snatch_guard,
    );
    Ok(())
}

/// Error encountered while attempting to perform [`CommandEncoder::transition_resources`].
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
        match self {
            Self::Device(e) => e.webgpu_error_type(),
            Self::EncoderState(e) => e.webgpu_error_type(),
            Self::InvalidResource(e) => e.webgpu_error_type(),
            Self::ResourceUsage(e) => e.webgpu_error_type(),
        }
    }
}
