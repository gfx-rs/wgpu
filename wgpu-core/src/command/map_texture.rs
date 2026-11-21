use alloc::{sync::Arc, vec::Vec};

use thiserror::Error;
use wgt::error::{ErrorType, WebGpuError};

use crate::{
    api_log,
    command::{encoder::EncodingState, CommandEncoder, EncoderStateError, EncodingApi},
    device::{DeviceError, MissingFeatures, TextureMapClosure, TextureMapError},
    resource::{
        DestroyedResourceError, InvalidResourceError, Labeled as _, ParentDevice as _,
        RawResourceAccess as _, Texture, TextureMapState,
    },
    track::ResourceUsageCompatibilityError,
};

/// A `map_texture_on_completion` request that has not been submitted yet.
///
/// These are kept outside of the encoder's recording state, so that
/// invalidating the encoder can't drop a callback without calling it.
pub(crate) struct PendingTextureMap {
    pub(crate) texture: Arc<Texture>,
    pub(crate) callback: Option<TextureMapClosure>,
}

/// Call the callbacks of maps that will never be submitted.
///
/// Must not be called while holding any lock.
pub(crate) fn abort_pending_texture_maps(maps: impl IntoIterator<Item = PendingTextureMap>) {
    for map in maps {
        if let Some(callback) = map.callback {
            callback(Err(TextureMapError::Aborted));
        }
    }
}

/// Revert textures that were marked `MappingQueued` by a submission that then
/// failed, and return their callbacks.
pub(crate) fn cancel_queued_texture_maps(
    textures: impl IntoIterator<Item = Arc<Texture>>,
) -> Vec<(TextureMapClosure, Result<(), TextureMapError>)> {
    let mut closures = Vec::new();
    for texture in textures {
        let mut map_state = texture.map_state.lock();
        if let TextureMapState::MappingQueued(cb) = &mut *map_state {
            if let Some(cb) = cb.take() {
                closures.push((cb, Err(TextureMapError::Aborted)));
            }
            *map_state = TextureMapState::Unmapped;
        }
    }
    closures
}

impl CommandEncoder {
    pub fn map_texture_on_completion(
        self: &Arc<Self>,
        texture: Arc<Texture>,
        callback: Option<TextureMapClosure>,
    ) {
        profiling::scope!("CommandEncoder::map_texture_on_completion");
        api_log!("CommandEncoder::map_texture_on_completion {texture:?}");

        let mut callback = Some(callback);
        let result = self.data.lock().with_buffer(
            EncodingApi::Wgpu,
            |_| -> Result<(), MapTextureOnCompletionError> {
                texture.same_device(&self.device)?;
                texture.check_valid()?;
                self.device
                    .require_features(wgt::Features::HOST_TEXTURE_COPY)?;
                if !texture
                    .desc
                    .usage
                    .contains(wgt::TextureUsages::HOST_VISIBLE)
                {
                    return Err(MapTextureOnCompletionError::MissingHostVisibleUsage);
                }
                let mut maps = self.texture_maps.lock();
                if maps.iter().any(|map| Arc::ptr_eq(&map.texture, &texture)) {
                    return Err(MapTextureOnCompletionError::AlreadyMappedOrQueued);
                }
                maps.push(PendingTextureMap {
                    texture: texture.clone(),
                    callback: callback.take().unwrap(),
                });
                Ok(())
            },
        );

        // The map was not recorded, so it can never complete.
        if let Some(Some(callback)) = callback {
            callback(Err(TextureMapError::Aborted));
        }

        if let Err(err) = result {
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
    texture: &Arc<Texture>,
) -> Result<(), MapTextureOnCompletionError> {
    texture.same_device(state.device)?;
    texture.try_raw(state.snatch_guard)?;

    let mut usage_scope = state.device.new_usage_scope();
    let indices = &state.device.tracker_indices;
    usage_scope.textures.set_size(indices.textures.size());
    unsafe {
        usage_scope
            .textures
            .merge_single(texture, None, wgt::TextureUses::HOST_COPY)?;
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
