use crate::{
    device::{DeviceError, HostMap, MissingFeatures},
    hub::{Global, GlobalIdentityHandlerFactory, HalApi, Resource, Token},
    id::{AdapterId, DeviceId, SurfaceId, TextureId, Valid},
    init_tracker::{BufferInitTracker, TextureInitTracker},
    track::TextureSelector,
    validation::MissingBufferUsageError,
    Label, LifeGuard, RefCount, Stored,
};

use smallvec::SmallVec;
use thiserror::Error;

use std::{borrow::Borrow, num::NonZeroU8, ops::Range, ptr::NonNull};

#[repr(C)]
#[derive(Debug)]
pub enum BufferMapAsyncStatus {
    Success,
    Error,
    Aborted,
    Unknown,
    ContextLost,
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

unsafe impl<A: hal::Api> Send for BufferMapState<A> {}
unsafe impl<A: hal::Api> Sync for BufferMapState<A> {}

#[repr(C)]
pub struct BufferMapCallbackC {
    pub callback: unsafe extern "C" fn(status: BufferMapAsyncStatus, user_data: *mut u8),
    pub user_data: *mut u8,
}

unsafe impl Send for BufferMapCallbackC {}

pub struct BufferMapCallback {
    // We wrap this so creating the enum in the C variant can be unsafe,
    // allowing our call function to be safe.
    inner: BufferMapCallbackInner,
}

enum BufferMapCallbackInner {
    Rust {
        callback: Box<dyn FnOnce(BufferMapAsyncStatus) + Send + 'static>,
    },
    C {
        inner: BufferMapCallbackC,
    },
}

impl BufferMapCallback {
    pub fn from_rust(callback: Box<dyn FnOnce(BufferMapAsyncStatus) + Send + 'static>) -> Self {
        Self {
            inner: BufferMapCallbackInner::Rust { callback },
        }
    }

    /// # Safety
    ///
    /// - The callback pointer must be valid to call with the provided user_data pointer.
    /// - Both pointers must point to 'static data as the callback may happen at an unspecified time.
    pub unsafe fn from_c(inner: BufferMapCallbackC) -> Self {
        Self {
            inner: BufferMapCallbackInner::C { inner },
        }
    }

    pub(crate) fn call(self, status: BufferMapAsyncStatus) {
        match self.inner {
            BufferMapCallbackInner::Rust { callback } => callback(status),
            // SAFETY: the contract of the call to from_c says that this unsafe is sound.
            BufferMapCallbackInner::C { inner } => unsafe {
                (inner.callback)(status, inner.user_data)
            },
        }
    }

    pub(crate) fn call_error(self) {
        log::error!("wgpu_buffer_map_async failed: buffer mapping is pending");
        self.call(BufferMapAsyncStatus::Error);
    }
}

pub struct BufferMapOperation {
    pub host: HostMap,
    pub callback: BufferMapCallback,
}

#[derive(Clone, Debug, Error)]
pub enum BufferAccessError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("buffer is invalid")]
    Invalid,
    #[error("buffer is destroyed")]
    Destroyed,
    #[error("buffer is already mapped")]
    AlreadyMapped,
    #[error(transparent)]
    MissingBufferUsage(#[from] MissingBufferUsageError),
    #[error("buffer is not mapped")]
    NotMapped,
    #[error(
        "buffer map range must start aligned to `MAP_ALIGNMENT` and end to `COPY_BUFFER_ALIGNMENT`"
    )]
    UnalignedRange,
    #[error("buffer offset invalid: offset {offset} must be multiple of 8")]
    UnalignedOffset { offset: wgt::BufferAddress },
    #[error("buffer range size invalid: range_size {range_size} must be multiple of 4")]
    UnalignedRangeSize { range_size: wgt::BufferAddress },
    #[error("buffer access out of bounds: index {index} would underrun the buffer (limit: {min})")]
    OutOfBoundsUnderrun {
        index: wgt::BufferAddress,
        min: wgt::BufferAddress,
    },
    #[error(
        "buffer access out of bounds: last index {index} would overrun the buffer (limit: {max})"
    )]
    OutOfBoundsOverrun {
        index: wgt::BufferAddress,
        max: wgt::BufferAddress,
    },
}

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
pub enum CreateBufferError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("failed to map buffer while creating: {0}")]
    AccessError(#[from] BufferAccessError),
    #[error("buffers that are mapped at creation have to be aligned to `COPY_BUFFER_ALIGNMENT`")]
    UnalignedSize,
    #[error("Buffers cannot have empty usage flags")]
    EmptyUsage,
    #[error("`MAP` usage can only be combined with the opposite `COPY`, requested {0:?}")]
    UsageMismatch(wgt::BufferUsages),
    #[error("Buffer size {requested} is greater than the maximum buffer size ({maximum})")]
    MaxBufferSize { requested: u64, maximum: u64 },
}

impl<A: hal::Api> Resource for Buffer<A> {
    const TYPE: &'static str = "Buffer";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

pub type TextureDescriptor<'a> = wgt::TextureDescriptor<Label<'a>>;

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

#[derive(Debug)]
pub struct Texture<A: hal::Api> {
    pub(crate) inner: TextureInner<A>,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) desc: wgt::TextureDescriptor<()>,
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
        profiling::scope!("as_hal", "Texture");

        let hub = A::hub(self);
        let mut token = Token::root();
        let (guard, _) = hub.textures.read(&mut token);
        let texture = guard.get(id).ok();
        let hal_texture = texture.map(|tex| tex.inner.as_raw().unwrap());

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
        profiling::scope!("as_hal", "Adapter");

        let hub = A::hub(self);
        let mut token = Token::root();

        let (guard, _) = hub.adapters.read(&mut token);
        let adapter = guard.get(id).ok();
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
        profiling::scope!("as_hal", "Device");

        let hub = A::hub(self);
        let mut token = Token::root();
        let (guard, _) = hub.devices.read(&mut token);
        let device = guard.get(id).ok();
        let hal_device = device.map(|device| &device.raw);

        hal_device_callback(hal_device)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum TextureErrorDimension {
    X,
    Y,
    Z,
}

#[derive(Clone, Debug, Error)]
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
pub enum CreateTextureError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Textures cannot have empty usage flags")]
    EmptyUsage,
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
    #[error("Texture usages {0:?} are not allowed on a texture of type {1:?}")]
    InvalidFormatUsages(wgt::TextureUsages, wgt::TextureFormat),
    #[error("Texture usages {0:?} are not allowed on a texture of dimensions {1:?}")]
    InvalidDimensionUsages(wgt::TextureUsages, wgt::TextureDimension),
    #[error("Texture usage STORAGE_BINDING is not allowed for multisampled textures")]
    InvalidMultisampledStorageBinding,
    #[error("Format {0:?} does not support multisampling")]
    InvalidMultisampledFormat(wgt::TextureFormat),
    #[error("Multisampled textures must have RENDER_ATTACHMENT usage")]
    MultisampledNotRenderAttachment,
    #[error("Texture format {0:?} can't be used due to missing features.")]
    MissingFeatures(wgt::TextureFormat, #[source] MissingFeatures),
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
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize), serde(default))]
pub struct TextureViewDescriptor<'a> {
    /// Debug label of the texture view. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Format of the texture view, or `None` for the same format as the texture itself.
    /// At this time, it must be the same the underlying format of the texture.
    pub format: Option<wgt::TextureFormat>,
    /// The dimension of the texture view. For 1D textures, this must be `1D`. For 2D textures it must be one of
    /// `D2`, `D2Array`, `Cube`, and `CubeArray`. For 3D textures it must be `3D`
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
        hal::FormatAspects::from(self.format) & hal::FormatAspects::from(self.range.aspect)
    }
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
    pub(crate) extent: wgt::Extent3d,
    pub(crate) samples: u32,
    pub(crate) selector: TextureSelector,
    pub(crate) life_guard: LifeGuard,
}

#[derive(Clone, Debug, Error)]
pub enum CreateTextureViewError {
    #[error("parent texture is invalid or destroyed")]
    InvalidTexture,
    #[error("not enough memory left")]
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
    /// Debug label of the sampler. This will show up in graphics debuggers for easy identification.
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
    /// Valid values: 1, 2, 4, 8, and 16.
    pub anisotropy_clamp: Option<NonZeroU8>,
    /// Border color to use when address_mode is [`AddressMode::ClampToBorder`](wgt::AddressMode::ClampToBorder)
    pub border_color: Option<wgt::SamplerBorderColor>,
}

impl Default for SamplerDescriptor<'_> {
    fn default() -> Self {
        Self {
            label: None,
            address_modes: Default::default(),
            mag_filter: Default::default(),
            min_filter: Default::default(),
            mipmap_filter: Default::default(),
            lod_min_clamp: 0.0,
            lod_max_clamp: std::f32::MAX,
            compare: None,
            anisotropy_clamp: None,
            border_color: None,
        }
    }
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

#[derive(Clone, Debug, Error)]
pub enum CreateSamplerError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("invalid anisotropic clamp {0}, must be one of 1, 2, 4, 8 or 16")]
    InvalidClamp(u8),
    #[error("cannot create any more samplers")]
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
pub enum DestroyError {
    #[error("resource is invalid")]
    Invalid,
    #[error("resource is already destroyed")]
    AlreadyDestroyed,
}
