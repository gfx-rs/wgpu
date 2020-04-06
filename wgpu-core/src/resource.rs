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

use std::{borrow::Borrow, fmt};

#[repr(C)]
#[derive(Debug)]
pub enum BufferMapAsyncStatus {
    Success,
    Error,
    Unknown,
    ContextLost,
}

#[derive(Debug)]
pub enum BufferMapState {
    /// Waiting for GPU to be done before mapping
    Waiting(BufferPendingMapping),
    /// Mapped
    Active,
    /// Not mapped
    Idle,
}

pub enum BufferMapOperation {
    Read {
        callback: crate::device::BufferMapReadCallback,
        userdata: *mut u8,
    },
    Write {
        callback: crate::device::BufferMapWriteCallback,
        userdata: *mut u8,
    },
}

//TODO: clarify if/why this is needed here
unsafe impl Send for BufferMapOperation {}
unsafe impl Sync for BufferMapOperation {}

impl fmt::Debug for BufferMapOperation {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let op = match *self {
            BufferMapOperation::Read { .. } => "read",
            BufferMapOperation::Write { .. } => "write",
        };
        write!(fmt, "BufferMapOperation <{}>", op)
    }
}

impl BufferMapOperation {
    pub(crate) fn call_error(self) {
        match self {
            BufferMapOperation::Read { callback, userdata } => {
                log::error!("wgpu_buffer_map_read_async failed: buffer mapping is pending");
                unsafe {
                    callback(BufferMapAsyncStatus::Error, std::ptr::null(), userdata);
                }
            }
            BufferMapOperation::Write { callback, userdata } => {
                log::error!("wgpu_buffer_map_write_async failed: buffer mapping is pending");
                unsafe {
                    callback(BufferMapAsyncStatus::Error, std::ptr::null_mut(), userdata);
                }
            }
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
    pub(crate) mapped_write_segments: Vec<hal::memory::Segment>,
    pub(crate) life_guard: LifeGuard,
    pub(crate) map_state: BufferMapState,
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
