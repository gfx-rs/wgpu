use alloc::{sync::Arc, vec::Vec};

use thiserror::Error;
use wgt::error::{ErrorType, WebGpuError};

use crate::{
    command::{encoder::EncodingState, CommandEncoder, EncoderStateError, EncodingApi},
    device::{DeviceError, MissingFeatures, TextureMapClosure, TextureMapPendingClosure},
    resource::{
        DestroyedResourceError, InvalidResourceError, Labeled as _, ParentDevice as _, Texture,
        TextureMapState,
    },
    track::ResourceUsageCompatibilityError,
};

/// Encoder-side analogue of `drain_pending_texture_maps`.
pub(crate) fn cancel_texture_maps(
    textures: impl IntoIterator<Item = Arc<Texture>>,
) -> Vec<TextureMapPendingClosure> {
    let mut closures = Vec::new();
    for texture in textures {
        let mut map_state = texture.map_state.lock();
        if let TextureMapState::MappingQueued(cb) = &mut *map_state {
            // `Lost` is the existing "mapping will never complete" signal; the
            // device isn't lost here, but the map is cancelled all the same.
            if let Some(cb) = cb.take() {
                closures.push((cb, Err(DeviceError::Lost)));
            }
            *map_state = TextureMapState::Unmapped;
        }
    }
    closures
}

impl CommandEncoder {
    fn map_texture_on_completion_inner(
        self: &Arc<Self>,
        texture: Arc<Texture>,
        callback: Option<TextureMapClosure>,
    ) -> Result<(), EncoderStateError> {
        profiling::scope!("CommandEncoder::map_texture_on_completion");

        let mut cmd_buf_data = self.data.lock();
        cmd_buf_data.with_buffer(
            EncodingApi::Wgpu,
            |buf| -> Result<(), MapTextureOnCompletionError> {
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

    pub fn map_texture_on_completion(
        self: &Arc<Self>,
        texture: Arc<Texture>,
        callback: Option<TextureMapClosure>,
    ) {
        if let Err(err) = self.map_texture_on_completion_inner(texture, callback) {
            self.device.handle_error(
                err,
                Some(self.label()),
                "CommandEncoder::map_texture_on_completion",
            );
        }
    }
}

pub(crate) fn encode_map_texture_on_completion(
    state: &mut EncodingState,
    texture: Arc<Texture>,
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
