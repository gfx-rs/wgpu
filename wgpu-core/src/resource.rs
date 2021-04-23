/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    device::{alloc::MemoryBlock, DeviceError, HostMap},
    hub::Resource,
    id::{DeviceId, SwapChainId, TextureId},
    memory_init_tracker::MemoryInitTracker,
    track::{TextureSelector, DUMMY_SELECTOR},
    validation::MissingBufferUsageError,
    Label, LifeGuard, RefCount, Stored,
};

use thiserror::Error;

use std::{
    borrow::Borrow,
    num::{NonZeroU32, NonZeroU8},
    ops::Range,
    ptr::NonNull,
};

bitflags::bitflags! {
    /// The internal enum mirrored from `BufferUsage`. The values don't have to match!
    pub struct BufferUse: u32 {
        const EMPTY = 0;
        const MAP_READ = 1;
        const MAP_WRITE = 2;
        const COPY_SRC = 4;
        const COPY_DST = 8;
        const INDEX = 16;
        const VERTEX = 32;
        const UNIFORM = 64;
        const STORAGE_LOAD = 128;
        const STORAGE_STORE = 256;
        const INDIRECT = 512;
        /// The combination of all read-only usages.
        const READ_ALL = Self::MAP_READ.bits | Self::COPY_SRC.bits |
            Self::INDEX.bits | Self::VERTEX.bits | Self::UNIFORM.bits |
            Self::STORAGE_LOAD.bits | Self::INDIRECT.bits;
        /// The combination of all write-only and read-write usages.
        const WRITE_ALL = Self::MAP_WRITE.bits | Self::COPY_DST.bits | Self::STORAGE_STORE.bits;
        /// The combination of all usages that the are guaranteed to be be ordered by the hardware.
        /// If a usage is not ordered, then even if it doesn't change between draw calls, there
        /// still need to be pipeline barriers inserted for synchronization.
        const ORDERED = Self::READ_ALL.bits | Self::MAP_WRITE.bits | Self::COPY_DST.bits;
    }
}

bitflags::bitflags! {
    /// The internal enum mirrored from `TextureUsage`. The values don't have to match!
    pub struct TextureUse: u32 {
        const EMPTY = 0;
        const COPY_SRC = 1;
        const COPY_DST = 2;
        const SAMPLED = 4;
        const ATTACHMENT_READ = 8;
        const ATTACHMENT_WRITE = 16;
        const STORAGE_LOAD = 32;
        const STORAGE_STORE = 48;
        /// The combination of all read-only usages.
        const READ_ALL = Self::COPY_SRC.bits | Self::SAMPLED.bits | Self::ATTACHMENT_READ.bits | Self::STORAGE_LOAD.bits;
        /// The combination of all write-only and read-write usages.
        const WRITE_ALL = Self::COPY_DST.bits | Self::ATTACHMENT_WRITE.bits | Self::STORAGE_STORE.bits;
        /// The combination of all usages that the are guaranteed to be be ordered by the hardware.
        /// If a usage is not ordered, then even if it doesn't change between draw calls, there
        /// still need to be pipeline barriers inserted for synchronization.
        const ORDERED = Self::READ_ALL.bits | Self::COPY_DST.bits | Self::ATTACHMENT_WRITE.bits;
        const UNINITIALIZED = 0xFFFF;
    }
}

#[repr(C)]
#[derive(Debug)]
pub enum BufferMapAsyncStatus {
    Success,
    Error,
    Aborted,
    Unknown,
    ContextLost,
}

#[derive(Debug)]
pub(crate) enum BufferMapState<B: hal::Backend> {
    /// Mapped at creation.
    Init {
        ptr: NonNull<u8>,
        stage_buffer: B::Buffer,
        stage_memory: MemoryBlock<B>,
        needs_flush: bool,
    },
    /// Waiting for GPU to be done before mapping
    Waiting(BufferPendingMapping),
    /// Mapped
    Active {
        ptr: NonNull<u8>,
        sub_range: hal::buffer::SubRange,
        host: HostMap,
    },
    /// Not mapped
    Idle,
}

unsafe impl<B: hal::Backend> Send for BufferMapState<B> {}
unsafe impl<B: hal::Backend> Sync for BufferMapState<B> {}

pub type BufferMapCallback = unsafe extern "C" fn(status: BufferMapAsyncStatus, userdata: *mut u8);

#[repr(C)]
#[derive(Debug)]
pub struct BufferMapOperation {
    pub host: HostMap,
    pub callback: BufferMapCallback,
    pub user_data: *mut u8,
}

//TODO: clarify if/why this is needed here
unsafe impl Send for BufferMapOperation {}
unsafe impl Sync for BufferMapOperation {}

impl BufferMapOperation {
    pub(crate) fn call_error(self) {
        log::error!("wgpu_buffer_map_async failed: buffer mapping is pending");
        unsafe {
            (self.callback)(BufferMapAsyncStatus::Error, self.user_data);
        }
    }
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

#[derive(Debug)]
pub(crate) struct BufferPendingMapping {
    pub range: Range<wgt::BufferAddress>,
    pub op: BufferMapOperation,
    // hold the parent alive while the mapping is active
    pub parent_ref_count: RefCount,
}

pub type BufferDescriptor<'a> = wgt::BufferDescriptor<Label<'a>>;

#[derive(Debug)]
pub struct Buffer<B: hal::Backend> {
    pub(crate) raw: Option<(B::Buffer, MemoryBlock<B>)>,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) usage: wgt::BufferUsage,
    pub(crate) size: wgt::BufferAddress,
    pub(crate) initialization_status: MemoryInitTracker,
    pub(crate) sync_mapped_writes: Option<hal::memory::Segment>,
    pub(crate) life_guard: LifeGuard,
    pub(crate) map_state: BufferMapState<B>,
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
    UsageMismatch(wgt::BufferUsage),
}

impl<B: hal::Backend> Resource for Buffer<B> {
    const TYPE: &'static str = "Buffer";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

impl<B: hal::Backend> Borrow<()> for Buffer<B> {
    fn borrow(&self) -> &() {
        &DUMMY_SELECTOR
    }
}

pub type TextureDescriptor<'a> = wgt::TextureDescriptor<Label<'a>>;

#[derive(Debug)]
pub struct Texture<B: hal::Backend> {
    pub(crate) raw: Option<(B::Image, MemoryBlock<B>)>,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) usage: wgt::TextureUsage,
    pub(crate) aspects: hal::format::Aspects,
    pub(crate) dimension: wgt::TextureDimension,
    pub(crate) kind: hal::image::Kind,
    pub(crate) format: wgt::TextureFormat,
    pub(crate) format_features: wgt::TextureFormatFeatures,
    pub(crate) framebuffer_attachment: hal::image::FramebufferAttachment,
    pub(crate) full_range: TextureSelector,
    pub(crate) life_guard: LifeGuard,
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
    #[error("Dimension {0:?} value {given} exceeds the limit of {limit}")]
    LimitExceeded {
        dim: TextureErrorDimension,
        given: u32,
        limit: u32,
    },
    #[error("sample count {0} is invalid")]
    InvalidSampleCount(u32),
}

#[derive(Clone, Debug, Error)]
pub enum CreateTextureError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("D24Plus textures cannot be copied")]
    CannotCopyD24Plus,
    #[error("Textures cannot have empty usage flags")]
    EmptyUsage,
    #[error(transparent)]
    InvalidDimension(#[from] TextureDimensionError),
    #[error("texture descriptor mip level count ({0}) is invalid")]
    InvalidMipLevelCount(u32),
    #[error("The texture usages {0:?} are not allowed on a texture of type {1:?}")]
    InvalidUsages(wgt::TextureUsage, wgt::TextureFormat),
    #[error("Feature {0:?} must be enabled to create a texture of type {1:?}")]
    MissingFeature(wgt::Features, wgt::TextureFormat),
}

impl<B: hal::Backend> Resource for Texture<B> {
    const TYPE: &'static str = "Texture";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

impl<B: hal::Backend> Borrow<TextureSelector> for Texture<B> {
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
    /// Aspect of the texture. Color textures must be [`TextureAspect::All`](wgt::TextureAspect::All).
    pub aspect: wgt::TextureAspect,
    /// Base mip level.
    pub base_mip_level: u32,
    /// Mip level count.
    /// If `Some(count)`, `base_mip_level + count` must be less or equal to underlying texture mip count.
    /// If `None`, considered to include the rest of the mipmap levels, but at least 1 in total.
    pub mip_level_count: Option<NonZeroU32>,
    /// Base array layer.
    pub base_array_layer: u32,
    /// Layer count.
    /// If `Some(count)`, `base_array_layer + count` must be less or equal to the underlying array count.
    /// If `None`, considered to include the rest of the array layers, but at least 1 in total.
    pub array_layer_count: Option<NonZeroU32>,
}

#[derive(Debug)]
pub(crate) enum TextureViewInner<B: hal::Backend> {
    Native {
        raw: B::ImageView,
        source_id: Stored<TextureId>,
    },
    SwapChain {
        image: <B::Surface as hal::window::PresentationSurface<B>>::SwapchainImage,
        source_id: Stored<SwapChainId>,
    },
}

#[derive(Debug)]
pub struct TextureView<B: hal::Backend> {
    pub(crate) inner: TextureViewInner<B>,
    //TODO: store device_id for quick access?
    pub(crate) aspects: hal::format::Aspects,
    pub(crate) format: wgt::TextureFormat,
    pub(crate) format_features: wgt::TextureFormatFeatures,
    pub(crate) dimension: wgt::TextureViewDimension,
    pub(crate) extent: wgt::Extent3d,
    pub(crate) samples: hal::image::NumSamples,
    pub(crate) framebuffer_attachment: hal::image::FramebufferAttachment,
    /// Internal use of this texture view when used as `BindingType::Texture`.
    pub(crate) sampled_internal_use: TextureUse,
    pub(crate) selector: TextureSelector,
    pub(crate) life_guard: LifeGuard,
}

#[derive(Clone, Debug, Error)]
pub enum CreateTextureViewError {
    #[error("parent texture is invalid or destroyed")]
    InvalidTexture,
    #[error("not enough memory left")]
    OutOfMemory,
    #[error("Invalid texture view dimension `{view:?}` with texture of dimension `{image:?}`")]
    InvalidTextureViewDimension {
        view: wgt::TextureViewDimension,
        image: wgt::TextureDimension,
    },
    #[error("Invalid texture depth `{depth}` for texture view of dimension `Cubemap`. Cubemap views must use images of size 6.")]
    InvalidCubemapTextureDepth { depth: u16 },
    #[error("Invalid texture depth `{depth}` for texture view of dimension `CubemapArray`. Cubemap views must use images with sizes which are a multiple of 6.")]
    InvalidCubemapArrayTextureDepth { depth: u16 },
    #[error(
        "TextureView mip level count + base mip level {requested} must be <= Texture mip level count {total}"
    )]
    TooManyMipLevels { requested: u32, total: u8 },
    #[error("TextureView array layer count + base array layer {requested} must be <= Texture depth/array layer count {total}")]
    TooManyArrayLayers { requested: u32, total: u16 },
    #[error("Requested array layer count {requested} is not valid for the target view dimension {dim:?}")]
    InvalidArrayLayerCount {
        requested: u32,
        dim: wgt::TextureViewDimension,
    },
    #[error("Aspect {requested:?} is not in the source texture ({total:?})")]
    InvalidAspect {
        requested: hal::format::Aspects,
        total: hal::format::Aspects,
    },
}

#[derive(Clone, Debug, Error)]
pub enum TextureViewDestroyError {
    #[error("cannot destroy swap chain image")]
    SwapChainImage,
}

impl<B: hal::Backend> Resource for TextureView<B> {
    const TYPE: &'static str = "TextureView";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

impl<B: hal::Backend> Borrow<()> for TextureView<B> {
    fn borrow(&self) -> &() {
        &DUMMY_SELECTOR
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
pub struct Sampler<B: hal::Backend> {
    pub(crate) raw: B::Sampler,
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
    /// AddressMode::ClampToBorder requires feature ADDRESS_MODE_CLAMP_TO_BORDER
    #[error("Feature {0:?} must be enabled")]
    MissingFeature(wgt::Features),
}

impl<B: hal::Backend> Resource for Sampler<B> {
    const TYPE: &'static str = "Sampler";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

impl<B: hal::Backend> Borrow<()> for Sampler<B> {
    fn borrow(&self) -> &() {
        &DUMMY_SELECTOR
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
    #[error("Feature {0:?} must be enabled")]
    MissingFeature(wgt::Features),
}

#[derive(Debug)]
pub struct QuerySet<B: hal::Backend> {
    pub(crate) raw: B::QueryPool,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) life_guard: LifeGuard,
    /// Amount of queries in the query set.
    pub(crate) desc: wgt::QuerySetDescriptor,
    /// Amount of numbers in each query (i.e. a pipeline statistics query for two attributes will have this number be two)
    pub(crate) elements: u32,
}

impl<B: hal::Backend> Resource for QuerySet<B> {
    const TYPE: &'static str = "QuerySet";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

impl<B: hal::Backend> Borrow<()> for QuerySet<B> {
    fn borrow(&self) -> &() {
        &DUMMY_SELECTOR
    }
}

#[derive(Clone, Debug, Error)]
pub enum DestroyError {
    #[error("resource is invalid")]
    Invalid,
    #[error("resource is already destroyed")]
    AlreadyDestroyed,
}
