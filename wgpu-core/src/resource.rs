use crate::{
    device::{DeviceError, HostMap, MissingDownlevelFlags, MissingFeatures},
    global::Global,
    hal_api::HalApi,
    hub::Token,
    id::{AdapterId, DeviceId, SurfaceId, TextureId, Valid},
    identity::GlobalIdentityHandlerFactory,
    init_tracker::{BufferInitTracker, TextureInitTracker},
    track::TextureSelector,
    validation::MissingBufferUsageError,
    Label, LifeGuard, RefCount, Stored,
};

use smallvec::SmallVec;
use thiserror::Error;

use std::{borrow::Borrow, ops::Range, ptr::NonNull};

pub trait Resource {
    const TYPE: &'static str;
    fn life_guard(&self) -> &LifeGuard;
    fn label(&self) -> &str {
        #[cfg(debug_assertions)]
        return &self.life_guard().label;
        #[cfg(not(debug_assertions))]
        return "";
    }
}

/// The status code provided to the buffer mapping callback.
///
/// This is very similar to `BufferAccessResult`, except that this is FFI-friendly.
#[repr(C)]
#[derive(Debug)]
pub enum BufferMapAsyncStatus {
    /// The Buffer is sucessfully mapped, `get_mapped_range` can be called.
    ///
    /// All other variants of this enum represent failures to map the buffer.
    Success,
    /// The buffer is already mapped.
    ///
    /// While this is treated as an error, it does not prevent mapped range from being accessed.
    AlreadyMapped,
    /// Mapping was already requested.
    MapAlreadyPending,
    /// An unknown error.
    Error,
    /// Mapping was aborted (by unmapping or destroying the buffer before mapping
    /// happened).
    Aborted,
    /// The context is Lost.
    ContextLost,
    /// The buffer is in an invalid state.
    Invalid,
    /// The range isn't fully contained in the buffer.
    InvalidRange,
    /// The range isn't properly aligned.
    InvalidAlignment,
    /// Incompatible usage flags.
    InvalidUsageFlags,
}

pub(crate) enum BufferMapState<A: hal::Api> {
    /// Mapped at creation.
    Init {
        ptr: NonNull<u8>,
        stage_buffer: A::Buffer,
        needs_flush: bool,
    },
    /// Waiting for GPU to be done before mapping
    Waiting(BufferPendingMapping),
    /// Mapped
    Active {
        ptr: NonNull<u8>,
        range: hal::MemoryRange,
        host: HostMap,
    },
    /// Not mapped
    Idle,
}

#[cfg(any(
    not(target_arch = "wasm32"),
    all(
        feature = "fragile-send-sync-non-atomic-wasm",
        not(target_feature = "atomics")
    )
))]
unsafe impl<A: hal::Api> Send for BufferMapState<A> {}
#[cfg(any(
    not(target_arch = "wasm32"),
    all(
        feature = "fragile-send-sync-non-atomic-wasm",
        not(target_feature = "atomics")
    )
))]
unsafe impl<A: hal::Api> Sync for BufferMapState<A> {}

#[repr(C)]
pub struct BufferMapCallbackC {
    pub callback: unsafe extern "C" fn(status: BufferMapAsyncStatus, user_data: *mut u8),
    pub user_data: *mut u8,
}

#[cfg(any(
    not(target_arch = "wasm32"),
    all(
        feature = "fragile-send-sync-non-atomic-wasm",
        not(target_feature = "atomics")
    )
))]
unsafe impl Send for BufferMapCallbackC {}

pub struct BufferMapCallback {
    // We wrap this so creating the enum in the C variant can be unsafe,
    // allowing our call function to be safe.
    inner: Option<BufferMapCallbackInner>,
}

#[cfg(any(
    not(target_arch = "wasm32"),
    all(
        feature = "fragile-send-sync-non-atomic-wasm",
        not(target_feature = "atomics")
    )
))]
type BufferMapCallbackCallback = Box<dyn FnOnce(BufferAccessResult) + Send + 'static>;
#[cfg(not(any(
    not(target_arch = "wasm32"),
    all(
        feature = "fragile-send-sync-non-atomic-wasm",
        not(target_feature = "atomics")
    )
)))]
type BufferMapCallbackCallback = Box<dyn FnOnce(BufferAccessResult) + 'static>;

enum BufferMapCallbackInner {
    Rust { callback: BufferMapCallbackCallback },
    C { inner: BufferMapCallbackC },
}

impl BufferMapCallback {
    pub fn from_rust(callback: BufferMapCallbackCallback) -> Self {
        Self {
            inner: Some(BufferMapCallbackInner::Rust { callback }),
        }
    }

    /// # Safety
    ///
    /// - The callback pointer must be valid to call with the provided user_data
    ///   pointer.
    ///
    /// - Both pointers must point to valid memory until the callback is
    ///   invoked, which may happen at an unspecified time.
    pub unsafe fn from_c(inner: BufferMapCallbackC) -> Self {
        Self {
            inner: Some(BufferMapCallbackInner::C { inner }),
        }
    }

    pub(crate) fn call(mut self, result: BufferAccessResult) {
        match self.inner.take() {
            Some(BufferMapCallbackInner::Rust { callback }) => {
                callback(result);
            }
            // SAFETY: the contract of the call to from_c says that this unsafe is sound.
            Some(BufferMapCallbackInner::C { inner }) => unsafe {
                let status = match result {
                    Ok(()) => BufferMapAsyncStatus::Success,
                    Err(BufferAccessError::Device(_)) => BufferMapAsyncStatus::ContextLost,
                    Err(BufferAccessError::Invalid) | Err(BufferAccessError::Destroyed) => {
                        BufferMapAsyncStatus::Invalid
                    }
                    Err(BufferAccessError::AlreadyMapped) => BufferMapAsyncStatus::AlreadyMapped,
                    Err(BufferAccessError::MapAlreadyPending) => {
                        BufferMapAsyncStatus::MapAlreadyPending
                    }
                    Err(BufferAccessError::MissingBufferUsage(_)) => {
                        BufferMapAsyncStatus::InvalidUsageFlags
                    }
                    Err(BufferAccessError::UnalignedRange)
                    | Err(BufferAccessError::UnalignedRangeSize { .. })
                    | Err(BufferAccessError::UnalignedOffset { .. }) => {
                        BufferMapAsyncStatus::InvalidAlignment
                    }
                    Err(BufferAccessError::OutOfBoundsUnderrun { .. })
                    | Err(BufferAccessError::OutOfBoundsOverrun { .. })
                    | Err(BufferAccessError::NegativeRange { .. }) => {
                        BufferMapAsyncStatus::InvalidRange
                    }
                    Err(_) => BufferMapAsyncStatus::Error,
                };

                (inner.callback)(status, inner.user_data);
            },
            None => {
                panic!("Map callback invoked twice");
            }
        }
    }
}

impl Drop for BufferMapCallback {
    fn drop(&mut self) {
        if self.inner.is_some() {
            panic!("Map callback was leaked");
        }
    }
}

pub struct BufferMapOperation {
    pub host: HostMap,
    pub callback: BufferMapCallback,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum BufferAccessError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Buffer map failed")]
    Failed,
    #[error("Buffer is invalid")]
    Invalid,
    #[error("Buffer is destroyed")]
    Destroyed,
    #[error("Buffer is already mapped")]
    AlreadyMapped,
    #[error("Buffer map is pending")]
    MapAlreadyPending,
    #[error(transparent)]
    MissingBufferUsage(#[from] MissingBufferUsageError),
    #[error("Buffer is not mapped")]
    NotMapped,
    #[error(
        "Buffer map range must start aligned to `MAP_ALIGNMENT` and end to `COPY_BUFFER_ALIGNMENT`"
    )]
    UnalignedRange,
    #[error("Buffer offset invalid: offset {offset} must be multiple of 8")]
    UnalignedOffset { offset: wgt::BufferAddress },
    #[error("Buffer range size invalid: range_size {range_size} must be multiple of 4")]
    UnalignedRangeSize { range_size: wgt::BufferAddress },
    #[error("Buffer access out of bounds: index {index} would underrun the buffer (limit: {min})")]
    OutOfBoundsUnderrun {
        index: wgt::BufferAddress,
        min: wgt::BufferAddress,
    },
    #[error(
        "Buffer access out of bounds: last index {index} would overrun the buffer (limit: {max})"
    )]
    OutOfBoundsOverrun {
        index: wgt::BufferAddress,
        max: wgt::BufferAddress,
    },
    #[error("Buffer map range start {start} is greater than end {end}")]
    NegativeRange {
        start: wgt::BufferAddress,
        end: wgt::BufferAddress,
    },
    #[error("Buffer map aborted")]
    MapAborted,
}

pub type BufferAccessResult = Result<(), BufferAccessError>;
pub(crate) struct BufferPendingMapping {
    pub range: Range<wgt::BufferAddress>,
    pub op: BufferMapOperation,
    // hold the parent alive while the mapping is active
    pub _parent_ref_count: RefCount,
}

pub type BufferDescriptor<'a> = wgt::BufferDescriptor<Label<'a>>;

pub struct Buffer<A: hal::Api> {
    pub(crate) raw: Option<A::Buffer>,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) usage: wgt::BufferUsages,
    pub(crate) size: wgt::BufferAddress,
    pub(crate) initialization_status: BufferInitTracker,
    pub(crate) sync_mapped_writes: Option<hal::MemoryRange>,
    pub(crate) life_guard: LifeGuard,
    pub(crate) map_state: BufferMapState<A>,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateBufferError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Failed to map buffer while creating: {0}")]
    AccessError(#[from] BufferAccessError),
    #[error("Buffers that are mapped at creation have to be aligned to `COPY_BUFFER_ALIGNMENT`")]
    UnalignedSize,
    #[error("Invalid usage flags {0:?}")]
    InvalidUsage(wgt::BufferUsages),
    #[error("`MAP` usage can only be combined with the opposite `COPY`, requested {0:?}")]
    UsageMismatch(wgt::BufferUsages),
    #[error("Buffer size {requested} is greater than the maximum buffer size ({maximum})")]
    MaxBufferSize { requested: u64, maximum: u64 },
    #[error(transparent)]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
}

impl<A: hal::Api> Resource for Buffer<A> {
    const TYPE: &'static str = "Buffer";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

/// A temporary buffer, consumed by the command that uses it.
///
/// A [`StagingBuffer`] is designed for one-shot uploads of data to the GPU. It
/// is always created mapped, and the command that uses it destroys the buffer
/// when it is done.
///
/// [`StagingBuffer`]s can be created with [`queue_create_staging_buffer`] and
/// used with [`queue_write_staging_buffer`]. They are also used internally by
/// operations like [`queue_write_texture`] that need to upload data to the GPU,
/// but that don't belong to any particular wgpu command buffer.
///
/// Used `StagingBuffer`s are accumulated in [`Device::pending_writes`], to be
/// freed once their associated operation's queue submission has finished
/// execution.
///
/// [`queue_create_staging_buffer`]: Global::queue_create_staging_buffer
/// [`queue_write_staging_buffer`]: Global::queue_write_staging_buffer
/// [`queue_write_texture`]: Global::queue_write_texture
/// [`Device::pending_writes`]: crate::device::Device
pub struct StagingBuffer<A: hal::Api> {
    pub(crate) raw: A::Buffer,
    pub(crate) size: wgt::BufferAddress,
    pub(crate) is_coherent: bool,
}

impl<A: hal::Api> Resource for StagingBuffer<A> {
    const TYPE: &'static str = "StagingBuffer";

    fn life_guard(&self) -> &LifeGuard {
        unreachable!()
    }

    fn label(&self) -> &str {
        "<StagingBuffer>"
    }
}

pub type TextureDescriptor<'a> = wgt::TextureDescriptor<Label<'a>, Vec<wgt::TextureFormat>>;

#[derive(Debug)]
pub(crate) enum TextureInner<A: hal::Api> {
    Native {
        raw: Option<A::Texture>,
    },
    Surface {
        raw: A::SurfaceTexture,
        parent_id: Valid<SurfaceId>,
        has_work: bool,
    },
}

impl<A: hal::Api> TextureInner<A> {
    pub fn as_raw(&self) -> Option<&A::Texture> {
        match *self {
            Self::Native { raw: Some(ref tex) } => Some(tex),
            Self::Native { raw: None } => None,
            Self::Surface { ref raw, .. } => Some(raw.borrow()),
        }
    }
}

#[derive(Debug)]
pub enum TextureClearMode<A: hal::Api> {
    BufferCopy,
    // View for clear via RenderPass for every subsurface (mip/layer/slice)
    RenderPass {
        clear_views: SmallVec<[A::TextureView; 1]>,
        is_color: bool,
    },
    // Texture can't be cleared, attempting to do so will cause panic.
    // (either because it is impossible for the type of texture or it is being destroyed)
    None,
}

impl<A: hal::Api> TextureClearMode<A> {
    pub(crate) fn destroy_clear_views(self, device: &A::Device) {
        if let TextureClearMode::RenderPass { clear_views, .. } = self {
            for clear_view in clear_views {
                unsafe {
                    hal::Device::destroy_texture_view(device, clear_view);
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct Texture<A: hal::Api> {
    pub(crate) inner: TextureInner<A>,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) desc: wgt::TextureDescriptor<(), Vec<wgt::TextureFormat>>,
    pub(crate) hal_usage: hal::TextureUses,
    pub(crate) format_features: wgt::TextureFormatFeatures,
    pub(crate) initialization_status: TextureInitTracker,
    pub(crate) full_range: TextureSelector,
    pub(crate) life_guard: LifeGuard,
    pub(crate) clear_mode: TextureClearMode<A>,
}

impl<A: hal::Api> Texture<A> {
    pub(crate) fn get_clear_view(&self, mip_level: u32, depth_or_layer: u32) -> &A::TextureView {
        match self.clear_mode {
            TextureClearMode::BufferCopy => {
                panic!("Given texture is cleared with buffer copies, not render passes")
            }
            TextureClearMode::None => {
                panic!("Given texture can't be cleared")
            }
            TextureClearMode::RenderPass {
                ref clear_views, ..
            } => {
                let index = if self.desc.dimension == wgt::TextureDimension::D3 {
                    (0..mip_level).fold(0, |acc, mip| {
                        acc + (self.desc.size.depth_or_array_layers >> mip).max(1)
                    })
                } else {
                    mip_level * self.desc.size.depth_or_array_layers
                } + depth_or_layer;
                &clear_views[index as usize]
            }
        }
    }
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    /// # Safety
    ///
    /// - The raw texture handle must not be manually destroyed
    pub unsafe fn texture_as_hal<A: HalApi, F: FnOnce(Option<&A::Texture>)>(
        &self,
        id: TextureId,
        hal_texture_callback: F,
    ) {
        profiling::scope!("Texture::as_hal");

        let hub = A::hub(self);
        let mut token = Token::root();
        let (guard, _) = hub.textures.read(&mut token);
        let texture = guard.try_get(id).ok().flatten();
        let hal_texture = texture.and_then(|tex| tex.inner.as_raw());

        hal_texture_callback(hal_texture);
    }

    /// # Safety
    ///
    /// - The raw adapter handle must not be manually destroyed
    pub unsafe fn adapter_as_hal<A: HalApi, F: FnOnce(Option<&A::Adapter>) -> R, R>(
        &self,
        id: AdapterId,
        hal_adapter_callback: F,
    ) -> R {
        profiling::scope!("Adapter::as_hal");

        let hub = A::hub(self);
        let mut token = Token::root();

        let (guard, _) = hub.adapters.read(&mut token);
        let adapter = guard.try_get(id).ok().flatten();
        let hal_adapter = adapter.map(|adapter| &adapter.raw.adapter);

        hal_adapter_callback(hal_adapter)
    }

    /// # Safety
    ///
    /// - The raw device handle must not be manually destroyed
    pub unsafe fn device_as_hal<A: HalApi, F: FnOnce(Option<&A::Device>) -> R, R>(
        &self,
        id: DeviceId,
        hal_device_callback: F,
    ) -> R {
        profiling::scope!("Device::as_hal");

        let hub = A::hub(self);
        let mut token = Token::root();
        let (guard, _) = hub.devices.read(&mut token);
        let device = guard.try_get(id).ok().flatten();
        let hal_device = device.map(|device| &device.raw);

        hal_device_callback(hal_device)
    }

    /// # Safety
    /// - The raw surface handle must not be manually destroyed
    pub unsafe fn surface_as_hal_mut<A: HalApi, F: FnOnce(Option<&mut A::Surface>) -> R, R>(
        &self,
        id: SurfaceId,
        hal_surface_callback: F,
    ) -> R {
        profiling::scope!("Surface::as_hal_mut");

        let mut token = Token::root();
        let (mut guard, _) = self.surfaces.write(&mut token);
        let surface = guard.get_mut(id).ok();
        let hal_surface = surface
            .and_then(|surface| A::get_surface_mut(surface))
            .map(|surface| &mut surface.raw);

        hal_surface_callback(hal_surface)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum TextureErrorDimension {
    X,
    Y,
    Z,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum TextureDimensionError {
    #[error("Dimension {0:?} is zero")]
    Zero(TextureErrorDimension),
    #[error("Dimension {dim:?} value {given} exceeds the limit of {limit}")]
    LimitExceeded {
        dim: TextureErrorDimension,
        given: u32,
        limit: u32,
    },
    #[error("Sample count {0} is invalid")]
    InvalidSampleCount(u32),
    #[error("Width {width} is not a multiple of {format:?}'s block width ({block_width})")]
    NotMultipleOfBlockWidth {
        width: u32,
        block_width: u32,
        format: wgt::TextureFormat,
    },
    #[error("Height {height} is not a multiple of {format:?}'s block height ({block_height})")]
    NotMultipleOfBlockHeight {
        height: u32,
        block_height: u32,
        format: wgt::TextureFormat,
    },
    #[error("Multisampled texture depth or array layers must be 1, got {0}")]
    MultisampledDepthOrArrayLayer(u32),
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateTextureError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Invalid usage flags {0:?}")]
    InvalidUsage(wgt::TextureUsages),
    #[error(transparent)]
    InvalidDimension(#[from] TextureDimensionError),
    #[error("Depth texture ({1:?}) can't be created as {0:?}")]
    InvalidDepthDimension(wgt::TextureDimension, wgt::TextureFormat),
    #[error("Compressed texture ({1:?}) can't be created as {0:?}")]
    InvalidCompressedDimension(wgt::TextureDimension, wgt::TextureFormat),
    #[error(
        "Texture descriptor mip level count {requested} is invalid, maximum allowed is {maximum}"
    )]
    InvalidMipLevelCount { requested: u32, maximum: u32 },
    #[error(
        "Texture usages {0:?} are not allowed on a texture of type {1:?}{}",
        if *.2 { " due to downlevel restrictions" } else { "" }
    )]
    InvalidFormatUsages(wgt::TextureUsages, wgt::TextureFormat, bool),
    #[error("The view format {0:?} is not compatible with texture format {1:?}, only changing srgb-ness is allowed.")]
    InvalidViewFormat(wgt::TextureFormat, wgt::TextureFormat),
    #[error("Texture usages {0:?} are not allowed on a texture of dimensions {1:?}")]
    InvalidDimensionUsages(wgt::TextureUsages, wgt::TextureDimension),
    #[error("Texture usage STORAGE_BINDING is not allowed for multisampled textures")]
    InvalidMultisampledStorageBinding,
    #[error("Format {0:?} does not support multisampling")]
    InvalidMultisampledFormat(wgt::TextureFormat),
    #[error("Sample count {0} is not supported by format {1:?} on this device. The WebGPU spec guarentees {2:?} samples are supported by this format. With the TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES feature your device supports {3:?}.")]
    InvalidSampleCount(u32, wgt::TextureFormat, Vec<u32>, Vec<u32>),
    #[error("Multisampled textures must have RENDER_ATTACHMENT usage")]
    MultisampledNotRenderAttachment,
    #[error("Texture format {0:?} can't be used due to missing features")]
    MissingFeatures(wgt::TextureFormat, #[source] MissingFeatures),
    #[error(transparent)]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
}

impl<A: hal::Api> Resource for Texture<A> {
    const TYPE: &'static str = "Texture";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

impl<A: hal::Api> Borrow<TextureSelector> for Texture<A> {
    fn borrow(&self) -> &TextureSelector {
        &self.full_range
    }
}

/// Describes a [`TextureView`].
#[derive(Clone, Debug, Default, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize), serde(default))]
pub struct TextureViewDescriptor<'a> {
    /// Debug label of the texture view.
    ///
    /// This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Format of the texture view, or `None` for the same format as the texture
    /// itself.
    ///
    /// At this time, it must be the same the underlying format of the texture.
    pub format: Option<wgt::TextureFormat>,
    /// The dimension of the texture view.
    ///
    /// - For 1D textures, this must be `D1`.
    /// - For 2D textures it must be one of `D2`, `D2Array`, `Cube`, or `CubeArray`.
    /// - For 3D textures it must be `D3`.
    pub dimension: Option<wgt::TextureViewDimension>,
    /// Range within the texture that is accessible via this view.
    pub range: wgt::ImageSubresourceRange,
}

#[derive(Debug)]
pub(crate) struct HalTextureViewDescriptor {
    pub format: wgt::TextureFormat,
    pub dimension: wgt::TextureViewDimension,
    pub range: wgt::ImageSubresourceRange,
}

impl HalTextureViewDescriptor {
    pub fn aspects(&self) -> hal::FormatAspects {
        hal::FormatAspects::new(self.format, self.range.aspect)
    }
}

#[derive(Debug, Copy, Clone, Error)]
pub enum TextureViewNotRenderableReason {
    #[error("The texture this view references doesn't include the RENDER_ATTACHMENT usage. Provided usages: {0:?}")]
    Usage(wgt::TextureUsages),
    #[error("The dimension of this texture view is not 2D. View dimension: {0:?}")]
    Dimension(wgt::TextureViewDimension),
    #[error("This texture view has more than one mipmap level. View mipmap levels: {0:?}")]
    MipLevelCount(u32),
    #[error("This texture view has more than one array layer. View array layers: {0:?}")]
    ArrayLayerCount(u32),
    #[error(
        "The aspects of this texture view are a subset of the aspects in the original texture. Aspects: {0:?}"
    )]
    Aspects(hal::FormatAspects),
}

#[derive(Debug)]
pub struct TextureView<A: hal::Api> {
    pub(crate) raw: A::TextureView,
    // The parent's refcount is held alive, but the parent may still be deleted
    // if it's a surface texture. TODO: make this cleaner.
    pub(crate) parent_id: Stored<TextureId>,
    pub(crate) device_id: Stored<DeviceId>,
    //TODO: store device_id for quick access?
    pub(crate) desc: HalTextureViewDescriptor,
    pub(crate) format_features: wgt::TextureFormatFeatures,
    /// This is `Err` only if the texture view is not renderable
    pub(crate) render_extent: Result<wgt::Extent3d, TextureViewNotRenderableReason>,
    pub(crate) samples: u32,
    pub(crate) selector: TextureSelector,
    pub(crate) life_guard: LifeGuard,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateTextureViewError {
    #[error("Parent texture is invalid or destroyed")]
    InvalidTexture,
    #[error("Not enough memory left")]
    OutOfMemory,
    #[error("Invalid texture view dimension `{view:?}` with texture of dimension `{texture:?}`")]
    InvalidTextureViewDimension {
        view: wgt::TextureViewDimension,
        texture: wgt::TextureDimension,
    },
    #[error("Invalid texture view dimension `{0:?}` of a multisampled texture")]
    InvalidMultisampledTextureViewDimension(wgt::TextureViewDimension),
    #[error("Invalid texture depth `{depth}` for texture view of dimension `Cubemap`. Cubemap views must use images of size 6.")]
    InvalidCubemapTextureDepth { depth: u32 },
    #[error("Invalid texture depth `{depth}` for texture view of dimension `CubemapArray`. Cubemap views must use images with sizes which are a multiple of 6.")]
    InvalidCubemapArrayTextureDepth { depth: u32 },
    #[error("Source texture width and height must be equal for a texture view of dimension `Cube`/`CubeArray`")]
    InvalidCubeTextureViewSize,
    #[error("Mip level count is 0")]
    ZeroMipLevelCount,
    #[error("Array layer count is 0")]
    ZeroArrayLayerCount,
    #[error(
        "TextureView mip level count + base mip level {requested} must be <= Texture mip level count {total}"
    )]
    TooManyMipLevels { requested: u32, total: u32 },
    #[error("TextureView array layer count + base array layer {requested} must be <= Texture depth/array layer count {total}")]
    TooManyArrayLayers { requested: u32, total: u32 },
    #[error("Requested array layer count {requested} is not valid for the target view dimension {dim:?}")]
    InvalidArrayLayerCount {
        requested: u32,
        dim: wgt::TextureViewDimension,
    },
    #[error("Aspect {requested_aspect:?} is not in the source texture format {texture_format:?}")]
    InvalidAspect {
        texture_format: wgt::TextureFormat,
        requested_aspect: wgt::TextureAspect,
    },
    #[error("Unable to view texture {texture:?} as {view:?}")]
    FormatReinterpretation {
        texture: wgt::TextureFormat,
        view: wgt::TextureFormat,
    },
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum TextureViewDestroyError {}

impl<A: hal::Api> Resource for TextureView<A> {
    const TYPE: &'static str = "TextureView";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

/// Describes a [`Sampler`]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct SamplerDescriptor<'a> {
    /// Debug label of the sampler.
    ///
    /// This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// How to deal with out of bounds accesses in the u (i.e. x) direction
    pub address_modes: [wgt::AddressMode; 3],
    /// How to filter the texture when it needs to be magnified (made larger)
    pub mag_filter: wgt::FilterMode,
    /// How to filter the texture when it needs to be minified (made smaller)
    pub min_filter: wgt::FilterMode,
    /// How to filter between mip map levels
    pub mipmap_filter: wgt::FilterMode,
    /// Minimum level of detail (i.e. mip level) to use
    pub lod_min_clamp: f32,
    /// Maximum level of detail (i.e. mip level) to use
    pub lod_max_clamp: f32,
    /// If this is enabled, this is a comparison sampler using the given comparison function.
    pub compare: Option<wgt::CompareFunction>,
    /// Must be at least 1. If this is not 1, all filter modes must be linear.
    pub anisotropy_clamp: u16,
    /// Border color to use when address_mode is
    /// [`AddressMode::ClampToBorder`](wgt::AddressMode::ClampToBorder)
    pub border_color: Option<wgt::SamplerBorderColor>,
}

#[derive(Debug)]
pub struct Sampler<A: hal::Api> {
    pub(crate) raw: A::Sampler,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) life_guard: LifeGuard,
    /// `true` if this is a comparison sampler
    pub(crate) comparison: bool,
    /// `true` if this is a filtering sampler
    pub(crate) filtering: bool,
}

#[derive(Copy, Clone)]
pub enum SamplerFilterErrorType {
    MagFilter,
    MinFilter,
    MipmapFilter,
}

impl std::fmt::Debug for SamplerFilterErrorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            SamplerFilterErrorType::MagFilter => write!(f, "magFilter"),
            SamplerFilterErrorType::MinFilter => write!(f, "minFilter"),
            SamplerFilterErrorType::MipmapFilter => write!(f, "mipmapFilter"),
        }
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateSamplerError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Invalid lodMinClamp: {0}. Must be greater or equal to 0.0")]
    InvalidLodMinClamp(f32),
    #[error("Invalid lodMaxClamp: {lod_max_clamp}. Must be greater or equal to lodMinClamp (which is {lod_min_clamp}).")]
    InvalidLodMaxClamp {
        lod_min_clamp: f32,
        lod_max_clamp: f32,
    },
    #[error("Invalid anisotropic clamp: {0}. Must be at least 1.")]
    InvalidAnisotropy(u16),
    #[error("Invalid filter mode for {filter_type:?}: {filter_mode:?}. When anistropic clamp is not 1 (it is {anisotropic_clamp}), all filter modes must be linear.")]
    InvalidFilterModeWithAnisotropy {
        filter_type: SamplerFilterErrorType,
        filter_mode: wgt::FilterMode,
        anisotropic_clamp: u16,
    },
    #[error("Cannot create any more samplers")]
    TooManyObjects,
    /// AddressMode::ClampToBorder requires feature ADDRESS_MODE_CLAMP_TO_BORDER.
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
}

impl<A: hal::Api> Resource for Sampler<A> {
    const TYPE: &'static str = "Sampler";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateQuerySetError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("QuerySets cannot be made with zero queries")]
    ZeroCount,
    #[error("{count} is too many queries for a single QuerySet. QuerySets cannot be made more than {maximum} queries.")]
    TooManyQueries { count: u32, maximum: u32 },
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
}

pub type QuerySetDescriptor<'a> = wgt::QuerySetDescriptor<Label<'a>>;

#[derive(Debug)]
pub struct QuerySet<A: hal::Api> {
    pub(crate) raw: A::QuerySet,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) life_guard: LifeGuard,
    pub(crate) desc: wgt::QuerySetDescriptor<()>,
}

impl<A: hal::Api> Resource for QuerySet<A> {
    const TYPE: &'static str = "QuerySet";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum DestroyError {
    #[error("Resource is invalid")]
    Invalid,
    #[error("Resource is already destroyed")]
    AlreadyDestroyed,
}
