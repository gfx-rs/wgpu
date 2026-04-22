use alloc::sync::Arc;

use thiserror::Error;
use wgt::error::{ErrorType, WebGpuError};

use crate::{
    command::{encoder::EncodingState, CommandEncoder, EncoderStateError, EncodingApi},
    device::{DeviceError, MissingFeatures, TextureMapClosure},
    global::Global,
    id::{CommandEncoderId, TextureId},
    resource::{
        DestroyedResourceError, InvalidResourceError, ParentDevice as _, RawResourceAccess as _,
        TextureMapState,
    },
    track::ResourceUsageCompatibilityError,
};

impl Global {
    pub fn command_encoder_map_texture_on_completion(
        &self,
        command_encoder_id: CommandEncoderId,
        texture_id: TextureId,
        callback: Option<TextureMapClosure>,
    ) -> Result<(), EncoderStateError> {
        profiling::scope!("CommandEncoder::map_texture_on_completion");

        let hub = &self.hub;
        let cmd_enc = hub.command_encoders.get(command_encoder_id);
        let mut cmd_buf_data = cmd_enc.data.lock();
        cmd_buf_data.with_buffer(
            EncodingApi::Wgpu,
            |buf| -> Result<(), MapTextureOnCompletionError> {
                let texture = self.resolve_texture_id(texture_id)?;
                texture
                    .device
                    .require_features(wgt::Features::HOST_IMAGE_COPY)?;
                if !texture
                    .desc
                    .usage
                    .contains(wgt::TextureUsages::HOST_VISIBLE)
                {
                    return Err(MapTextureOnCompletionError::MissingHostVisibleUsage);
                }

                {
                    let mut map_state = texture.map_state.lock();
                    match &*map_state {
                        TextureMapState::Unmapped => {
                            *map_state = TextureMapState::MappingQueued(callback);
                        }
                        TextureMapState::MappingQueued(_) | TextureMapState::Mapped(_) => {
                            return Err(MapTextureOnCompletionError::AlreadyMappedOrQueued);
                        }
                    }
                }

                buf.textures_to_map_on_completion.push(texture);
                Ok(())
            },
        )
    }
}

pub(crate) fn encode_map_texture_on_completion(
    state: &mut EncodingState,
    texture: Arc<crate::resource::Texture>,
) -> Result<(), MapTextureOnCompletionError> {
    texture.same_device(state.device)?;

    let mut usage_scope = state.device.new_usage_scope();
    let indices = &state.device.tracker_indices;
    usage_scope.textures.set_size(indices.textures.size());
    unsafe {
        usage_scope
            .textures
            .merge_single(&texture, None, wgt::TextureUses::HOST_COPY)?;
    }
    CommandEncoder::insert_barriers_from_scope(
        state.raw_encoder,
        state.tracker,
        &usage_scope,
        state.snatch_guard,
    );

    let raw_texture = texture.try_raw(state.snatch_guard)?;
    unsafe { state.raw_encoder.pre_texture_map(raw_texture) };

    Ok(())
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum MapTextureOnCompletionError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    EncoderState(#[from] EncoderStateError),
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),
    #[error(transparent)]
    ResourceUsage(#[from] ResourceUsageCompatibilityError),
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error("Texture was not created with TextureUsages::HOST_VISIBLE")]
    MissingHostVisibleUsage,
    #[error("Texture is already mapped or queued for mapping")]
    AlreadyMappedOrQueued,
}

impl WebGpuError for MapTextureOnCompletionError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Self::Device(e) => e.webgpu_error_type(),
            Self::EncoderState(e) => e.webgpu_error_type(),
            Self::InvalidResource(e) => e.webgpu_error_type(),
            Self::DestroyedResource(e) => e.webgpu_error_type(),
            Self::ResourceUsage(e) => e.webgpu_error_type(),
            Self::MissingFeatures(e) => e.webgpu_error_type(),
            Self::MissingHostVisibleUsage | Self::AlreadyMappedOrQueued => ErrorType::Validation,
        }
    }
}
