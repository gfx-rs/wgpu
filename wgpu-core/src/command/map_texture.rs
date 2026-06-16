use alloc::{sync::Arc, vec::Vec};

use thiserror::Error;
use wgt::error::{ErrorType, WebGpuError};

use crate::{
    command::{encoder::EncodingState, CommandEncoder, EncoderStateError, EncodingApi},
    device::{DeviceError, MissingFeatures, TextureMapClosure, TextureMapPendingClosure},
    global::Global,
    id::{CommandEncoderId, TextureId},
    resource::{
        DestroyedResourceError, InvalidResourceError, ParentDevice as _, Texture, TextureMapState,
    },
    track::ResourceUsageCompatibilityError,
};

/// Revert textures that were queued for host-mapping by a command encoder back
/// to [`TextureMapState::Unmapped`] and return their callbacks paired with an
/// error, so the callbacks are fired rather than silently dropped.
///
/// Called when an encoder (or command buffer) carrying queued maps is dropped,
/// or its recording fails, before the mapping could be registered with the
/// queue. Without this the textures would be stranded in
/// [`TextureMapState::MappingQueued`] forever (rejected by every future submit)
/// and any caller awaiting the mapping callback would hang. This is the
/// encoder-side analogue of the device-loss cleanup
/// `LifetimeTracker::drain_pending_texture_maps`.
pub(crate) fn cancel_texture_maps(
    textures: impl IntoIterator<Item = Arc<Texture>>,
) -> Vec<TextureMapPendingClosure> {
    let mut closures = Vec::new();
    for texture in textures {
        let mut map_state = texture.map_state.lock();
        if let TextureMapState::MappingQueued(cb) = &mut *map_state {
            // `DeviceError::Lost` is the existing "this mapping will never
            // complete" signal (matching `drain_pending_texture_maps`); here the
            // device isn't necessarily lost, but the mapping has been cancelled.
            if let Some(cb) = cb.take() {
                closures.push((cb, Err(DeviceError::Lost)));
            }
            *map_state = TextureMapState::Unmapped;
        }
    }
    closures
}

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

                buf.pending_texture_maps.push(texture);
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
