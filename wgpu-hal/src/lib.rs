/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/*! This library describes the internal unsafe graphics abstraction API.
 */

pub mod empty;

use std::{fmt, num::NonZeroU8, ops::Range, ptr::NonNull};

use bitflags::bitflags;

pub const MAX_ANISOTROPY: u8 = 16;
pub const MAX_BIND_GROUPS: usize = 8;

pub type Label<'a> = Option<&'a str>;
pub type MemoryRange = Range<wgt::BufferAddress>;
pub type MipLevel = u8;
pub type ArrayLayer = u16;

#[derive(Debug)]
pub enum Error {
    OutOfMemory,
    DeviceLost,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::OutOfMemory => write!(f, "out of memory"),
            Self::DeviceLost => write!(f, "device is lost"),
        }
    }
}
impl std::error::Error for Error {}

pub trait Api: Sized {
    type Surface: Surface<Self>;
    type Adapter: Adapter<Self>;
    type Device: Device<Self>;
    type Queue: Queue<Self>;

    type CommandBuffer: CommandBuffer<Self>;
    type RenderPass: RenderPass<Self>;
    type ComputePass: ComputePass<Self>;

    type Buffer;
    type QuerySet;
    type Texture;
    type SwapChainTexture;
    type TextureView;
    type Sampler;

    unsafe fn enumerate_adapters(&self) -> Vec<ExposedAdapter<Self>>;
}

pub trait Surface<A: Api> {}

pub trait Adapter<A: Api> {
    unsafe fn open(&self, features: wgt::Features) -> Result<OpenDevice<A>, Error>;
    unsafe fn close(&self, device: A::Device);

    /// Return the set of supported capabilities for a texture format.
    unsafe fn texture_format_capabilities(
        &self,
        format: wgt::TextureFormat,
    ) -> TextureFormatCapability;
    /// Returns the list of surface formats supported for presentation, if any.
    unsafe fn surface_formats(&self, surface: &A::Surface) -> Vec<wgt::TextureFormat>;
}

pub trait Device<A: Api> {
    unsafe fn create_buffer(&self, desc: &wgt::BufferDescriptor<Label>)
        -> Result<A::Buffer, Error>;
    unsafe fn destroy_buffer(&self, buffer: A::Buffer);
    unsafe fn map_buffer(
        &self,
        buffer: &A::Buffer,
        range: MemoryRange,
    ) -> Result<NonNull<u8>, Error>;
    unsafe fn unmap_buffer(&self, buffer: &A::Buffer);
    unsafe fn flush_mapped_ranges<I: Iterator<Item = MemoryRange>>(
        &self,
        buffer: &A::Buffer,
        ranges: I,
    );
    unsafe fn invalidate_mapped_ranges<I: Iterator<Item = MemoryRange>>(
        &self,
        buffer: &A::Buffer,
        ranges: I,
    );

    unsafe fn create_texture(
        &self,
        desc: &wgt::TextureDescriptor<Label>,
    ) -> Result<A::Texture, Error>;
    unsafe fn destroy_texture(&self, texture: A::Texture);
    unsafe fn create_texture_view(
        &self,
        texture: &A::Texture,
        desc: &TextureViewDescriptor<Label>,
    ) -> Result<A::TextureView, Error>;
    unsafe fn destroy_texture_view(&self, view: A::TextureView);
    unsafe fn create_sampler(&self, desc: &SamplerDescriptor<Label>) -> Result<A::Sampler, Error>;
    unsafe fn destroy_sampler(&self, sampler: A::Sampler);

    unsafe fn create_command_buffer(&self) -> Result<A::CommandBuffer, Error>;
}

pub trait Queue<A: Api> {
    unsafe fn submit<I: Iterator<Item = A::CommandBuffer>>(&mut self, command_buffers: I);
}

pub trait CommandBuffer<A: Api> {
    unsafe fn begin(&mut self);
    unsafe fn end(&mut self);

    unsafe fn begin_render_pass(&mut self) -> A::RenderPass;
    unsafe fn end_render_pass(&mut self, pass: A::RenderPass);
    unsafe fn begin_compute_pass(&mut self) -> A::ComputePass;
    unsafe fn end_compute_pass(&mut self, pass: A::ComputePass);
}

pub trait RenderPass<A: Api> {}
pub trait ComputePass<A: Api> {}

#[derive(Debug)]
pub struct Alignments {
    /// The alignment of the start of the buffer used as a GPU copy source.
    pub buffer_copy_offset: wgt::BufferSize,
    /// The alignment of the row pitch of the texture data stored in a buffer that is
    /// used in a GPU copy operation.
    pub buffer_copy_pitch: wgt::BufferSize,
    pub storage_buffer_offset: wgt::BufferSize,
    pub uniform_buffer_offset: wgt::BufferSize,
}

#[derive(Debug)]
pub struct Capabilities {
    pub limits: wgt::Limits,
    pub alignments: Alignments,
    pub downlevel: wgt::DownlevelCapabilities,
}

#[derive(Debug)]
pub struct ExposedAdapter<A: Api> {
    pub adapter: A::Adapter,
    pub info: wgt::AdapterInfo,
    pub features: wgt::Features,
    pub capabilities: Capabilities,
}

#[derive(Debug)]
pub struct OpenDevice<A: Api> {
    pub device: A::Device,
    pub queue: A::Queue,
}

bitflags!(
    /// Texture format capability flags.
    pub struct TextureFormatCapability: u32 {
        /// Format can be sampled.
        const SAMPLED = 0x1;
        /// Format can be sampled with a linear sampler.
        const SAMPLED_LINEAR = 0x2;
        /// Format can be sampled with a min/max reduction sampler.
        const SAMPLED_MINMAX = 0x4;

        /// Format can be used as storage with exclusive read & write access.
        const STORAGE = 0x10;
        /// Format can be used as storage with simultaneous read/write access.
        const STORAGE_READ_WRITE = 0x20;
        /// Format can be used as storage with atomics.
        const STORAGE_ATOMIC = 0x40;

        /// Format can be used as color and input attachment.
        const COLOR_ATTACHMENT = 0x100;
        /// Format can be used as color (with blending) and input attachment.
        const COLOR_ATTACHMENT_BLEND = 0x200;
        /// Format can be used as depth-stencil and input attachment.
        const DEPTH_STENCIL_ATTACHMENT = 0x400;

        /// Format can be copied from.
        const COPY_SRC = 0x1000;
        /// Format can be copied to.
        const COPY_DST = 0x2000;
    }
);

bitflags!(
    /// Texture format capability flags.
    pub struct FormatAspect: u8 {
        const COLOR = 1;
        const DEPTH = 2;
        const STENCIL = 4;
    }
);

impl From<wgt::TextureFormat> for FormatAspect {
    fn from(format: wgt::TextureFormat) -> Self {
        match format {
            wgt::TextureFormat::Depth32Float | wgt::TextureFormat::Depth24Plus => Self::DEPTH,
            wgt::TextureFormat::Depth24PlusStencil8 => Self::DEPTH | Self::STENCIL,
            _ => Self::COLOR,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TextureViewDescriptor<L> {
    pub label: L,
    pub format: wgt::TextureFormat,
    pub dimension: wgt::TextureViewDimension,
    pub range: wgt::ImageSubresourceRange,
}

impl<L> TextureViewDescriptor<L> {
    ///
    pub fn map_label<K>(&self, fun: impl FnOnce(&L) -> K) -> TextureViewDescriptor<K> {
        TextureViewDescriptor {
            label: fun(&self.label),
            format: self.format,
            dimension: self.dimension,
            range: self.range.clone(),
        }
    }
}

/// Describes a [`Sampler`]
#[derive(Clone, Debug)]
pub struct SamplerDescriptor<L> {
    pub label: L,
    pub address_modes: [wgt::AddressMode; 3],
    pub mag_filter: wgt::FilterMode,
    pub min_filter: wgt::FilterMode,
    pub mipmap_filter: wgt::FilterMode,
    pub lod_clamp: Option<Range<f32>>,
    pub compare: Option<wgt::CompareFunction>,
    pub anisotropy_clamp: Option<NonZeroU8>,
    pub border_color: Option<wgt::SamplerBorderColor>,
}

#[test]
fn test_default_limits() {
    let limits = wgt::Limits::default();
    assert!(limits.max_bind_groups <= MAX_BIND_GROUPS as u32);
}
