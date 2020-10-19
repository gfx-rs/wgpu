/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    id::{DeviceId, SwapChainId, TextureId},
    track::DUMMY_SELECTOR,
    LifeGuard, RefCount, Stored,
};

use gfx_memory::MemoryBlock;
use wgt::{BufferAddress, BufferUsage, TextureFormat, TextureUsage};

use std::{borrow::Borrow, ptr::NonNull};

bitflags::bitflags! {
    /// The internal enum mirrored from `BufferUsage`. The values don't have to match!
    pub (crate) struct BufferUse: u32 {
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
    pub(crate) struct TextureUse: u32 {
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
    Unknown,
    ContextLost,
}

#[derive(Debug)]
pub enum BufferMapState<B: hal::Backend> {
    /// Mapped at creation.
    Init {
        ptr: NonNull<u8>,
        stage_buffer: B::Buffer,
        stage_memory: MemoryBlock<B>,
    },
    /// Waiting for GPU to be done before mapping
    Waiting(BufferPendingMapping),
    /// Mapped
    Active {
        ptr: NonNull<u8>,
        sub_range: hal::buffer::SubRange,
        host: crate::device::HostMap,
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
    pub host: crate::device::HostMap,
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

#[derive(Debug)]
pub struct BufferPendingMapping {
    pub sub_range: hal::buffer::SubRange,
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
    pub(crate) sync_mapped_writes: Option<hal::memory::Segment>,
    pub(crate) life_guard: LifeGuard,
    pub(crate) map_state: BufferMapState<B>,
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

#[derive(Debug)]
pub struct Texture<B: hal::Backend> {
    pub(crate) raw: B::Image,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) usage: TextureUsage,
    pub(crate) dimension: wgt::TextureDimension,
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
