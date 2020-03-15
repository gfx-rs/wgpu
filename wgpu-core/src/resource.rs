/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    id::{DeviceId, SwapChainId, TextureId},
    track::DUMMY_SELECTOR,
    Extent3d,
    LifeGuard,
    RefCount,
    Stored,
};

use wgt::{BufferAddress, BufferUsage, CompareFunction, TextureFormat, TextureUsage};
use hal;
use rendy_memory::MemoryBlock;

use std::{borrow::Borrow, fmt};

#[repr(C)]
#[derive(Debug)]
pub enum BufferMapAsyncStatus {
    Success,
    Error,
    Unknown,
    ContextLost,
}

pub enum BufferMapOperation {
    Read(Box<dyn FnOnce(BufferMapAsyncStatus, *const u8)>),
    Write(Box<dyn FnOnce(BufferMapAsyncStatus, *mut u8)>),
}

//TODO: clarify if/why this is needed here
unsafe impl Send for BufferMapOperation {}
unsafe impl Sync for BufferMapOperation {}

impl fmt::Debug for BufferMapOperation {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let op = match *self {
            BufferMapOperation::Read(_) => "read",
            BufferMapOperation::Write(_) => "write",
        };
        write!(fmt, "BufferMapOperation <{}>", op)
    }
}

impl BufferMapOperation {
    pub(crate) fn call_error(self) {
        match self {
            BufferMapOperation::Read(callback) => {
                log::error!("wgpu_buffer_map_read_async failed: buffer mapping is pending");
                callback(BufferMapAsyncStatus::Error, std::ptr::null());
            }
            BufferMapOperation::Write(callback) => {
                log::error!("wgpu_buffer_map_write_async failed: buffer mapping is pending");
                callback(BufferMapAsyncStatus::Error, std::ptr::null_mut());
            }
        }
    }
}

#[derive(Debug)]
pub struct BufferPendingMapping {
    pub range: std::ops::Range<BufferAddress>,
    pub op: BufferMapOperation,
    // hold the parent alive while the mapping is active
    pub parent_ref_count: RefCount,
}

#[derive(Debug)]
pub struct Buffer<B: hal::Backend> {
    pub(crate) raw: B::Buffer,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) usage: BufferUsage,
    pub(crate) memory: MemoryBlock<B>,
    pub(crate) size: BufferAddress,
    pub(crate) full_range: (),
    pub(crate) mapped_write_ranges: Vec<std::ops::Range<BufferAddress>>,
    pub(crate) pending_mapping: Option<BufferPendingMapping>,
    pub(crate) life_guard: LifeGuard,
}

impl<B: hal::Backend> Borrow<RefCount> for Buffer<B> {
    fn borrow(&self) -> &RefCount {
        self.life_guard.ref_count.as_ref().unwrap()
    }
}

impl<B: hal::Backend> Borrow<()> for Buffer<B> {
    fn borrow(&self) -> &() {
        &DUMMY_SELECTOR
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum TextureDimension {
    D1,
    D2,
    D3,
}

#[repr(C)]
#[derive(Debug)]
pub struct TextureDescriptor {
    pub size: Extent3d,
    pub array_layer_count: u32,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub dimension: TextureDimension,
    pub format: TextureFormat,
    pub usage: TextureUsage,
}

#[derive(Debug)]
pub struct Texture<B: hal::Backend> {
    pub(crate) raw: B::Image,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) usage: TextureUsage,
    pub(crate) kind: hal::image::Kind,
    pub(crate) format: TextureFormat,
    pub(crate) full_range: hal::image::SubresourceRange,
    pub(crate) memory: MemoryBlock<B>,
    pub(crate) life_guard: LifeGuard,
}

impl<B: hal::Backend> Borrow<RefCount> for Texture<B> {
    fn borrow(&self) -> &RefCount {
        self.life_guard.ref_count.as_ref().unwrap()
    }
}

impl<B: hal::Backend> Borrow<hal::image::SubresourceRange> for Texture<B> {
    fn borrow(&self) -> &hal::image::SubresourceRange {
        &self.full_range
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum TextureAspect {
    All,
    StencilOnly,
    DepthOnly,
}

impl Default for TextureAspect {
    fn default() -> Self {
        TextureAspect::All
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct TextureViewDescriptor {
    pub format: TextureFormat,
    pub dimension: wgt::TextureViewDimension,
    pub aspect: TextureAspect,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub array_layer_count: u32,
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
    pub(crate) format: TextureFormat,
    pub(crate) extent: hal::image::Extent,
    pub(crate) samples: hal::image::NumSamples,
    pub(crate) range: hal::image::SubresourceRange,
    pub(crate) life_guard: LifeGuard,
}

impl<B: hal::Backend> Borrow<RefCount> for TextureView<B> {
    fn borrow(&self) -> &RefCount {
        self.life_guard.ref_count.as_ref().unwrap()
    }
}

impl<B: hal::Backend> Borrow<()> for TextureView<B> {
    fn borrow(&self) -> &() {
        &DUMMY_SELECTOR
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum AddressMode {
    ClampToEdge = 0,
    Repeat = 1,
    MirrorRepeat = 2,
}

impl Default for AddressMode {
    fn default() -> Self {
        AddressMode::ClampToEdge
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum FilterMode {
    Nearest = 0,
    Linear = 1,
}

impl Default for FilterMode {
    fn default() -> Self {
        FilterMode::Nearest
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct SamplerDescriptor<'a> {
    pub address_mode_u: AddressMode,
    pub address_mode_v: AddressMode,
    pub address_mode_w: AddressMode,
    pub mag_filter: FilterMode,
    pub min_filter: FilterMode,
    pub mipmap_filter: FilterMode,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub compare: Option<&'a CompareFunction>,
}

#[derive(Debug)]
pub struct Sampler<B: hal::Backend> {
    pub(crate) raw: B::Sampler,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) life_guard: LifeGuard,
}

impl<B: hal::Backend> Borrow<RefCount> for Sampler<B> {
    fn borrow(&self) -> &RefCount {
        self.life_guard.ref_count.as_ref().unwrap()
    }
}

impl<B: hal::Backend> Borrow<()> for Sampler<B> {
    fn borrow(&self) -> &() {
        &DUMMY_SELECTOR
    }
}
