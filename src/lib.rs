/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use wgc::id;

pub use wgc::command::{compute_ffi::*, render_ffi::*};

pub mod client;
pub mod identity;
pub mod server;

pub use wgc::device::trace::Command as CommandEncoderAction;

use std::{borrow::Cow, mem, slice};

type RawString = *const std::os::raw::c_char;

//TODO: figure out why 'a and 'b have to be different here
//TODO: remove this
fn cow_label<'a, 'b>(raw: &'a RawString) -> Option<Cow<'b, str>> {
    if raw.is_null() {
        None
    } else {
        let cstr = unsafe { std::ffi::CStr::from_ptr(*raw) };
        cstr.to_str().ok().map(Cow::Borrowed)
    }
}

#[repr(C)]
pub struct ByteBuf {
    data: *const u8,
    len: usize,
    capacity: usize,
}

impl ByteBuf {
    fn from_vec(vec: Vec<u8>) -> Self {
        if vec.is_empty() {
            ByteBuf {
                data: std::ptr::null(),
                len: 0,
                capacity: 0,
            }
        } else {
            let bb = ByteBuf {
                data: vec.as_ptr(),
                len: vec.len(),
                capacity: vec.capacity(),
            };
            mem::forget(vec);
            bb
        }
    }

    unsafe fn as_slice(&self) -> &[u8] {
        slice::from_raw_parts(self.data, self.len)
    }
}

#[repr(C)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct AdapterInformation {
    id: id::AdapterId,
    //inner: wgt::AdapterInfo, //TODO: not C-friendly
    limits: wgt::Limits,
    features: wgt::Features,
}

#[derive(serde::Serialize, serde::Deserialize)]
enum ShaderModuleSource<'a> {
    SpirV(Cow<'a, [u32]>),
    Wgsl(Cow<'a, str>),
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ImplicitLayout<'a> {
    pipeline: id::PipelineLayoutId,
    bind_groups: Cow<'a, [id::BindGroupLayoutId]>,
}

#[derive(serde::Serialize, serde::Deserialize)]
enum DeviceAction<'a> {
    CreateBuffer(id::BufferId, wgc::resource::BufferDescriptor<'a>),
    CreateTexture(id::TextureId, wgc::resource::TextureDescriptor<'a>),
    CreateSampler(id::SamplerId, wgc::resource::SamplerDescriptor<'a>),
    CreateBindGroupLayout(
        id::BindGroupLayoutId,
        wgc::binding_model::BindGroupLayoutDescriptor<'a>,
    ),
    CreatePipelineLayout(
        id::PipelineLayoutId,
        wgc::binding_model::PipelineLayoutDescriptor<'a>,
    ),
    CreateBindGroup(id::BindGroupId, wgc::binding_model::BindGroupDescriptor<'a>),
    CreateShaderModule(
        id::ShaderModuleId,
        wgc::pipeline::ShaderModuleDescriptor<'a>,
        ShaderModuleSource<'a>,
    ),
    CreateComputePipeline(
        id::ComputePipelineId,
        wgc::pipeline::ComputePipelineDescriptor<'a>,
        Option<ImplicitLayout<'a>>,
    ),
    CreateRenderPipeline(
        id::RenderPipelineId,
        wgc::pipeline::RenderPipelineDescriptor<'a>,
        Option<ImplicitLayout<'a>>,
    ),
    CreateRenderBundle(
        id::RenderBundleId,
        wgc::command::RenderBundleEncoderDescriptor<'a>,
        wgc::command::BasePass<wgc::command::RenderCommand>,
    ),
    CreateCommandEncoder(
        id::CommandEncoderId,
        wgt::CommandEncoderDescriptor<wgc::Label<'a>>,
    ),
}

#[derive(serde::Serialize, serde::Deserialize)]
enum QueueWriteAction {
    Buffer {
        dst: id::BufferId,
        offset: wgt::BufferAddress,
    },
    Texture {
        dst: wgt::ImageCopyTexture<id::TextureId>,
        layout: wgt::ImageDataLayout,
        size: wgt::Extent3d,
    },
}

#[derive(serde::Serialize, serde::Deserialize)]
enum TextureAction<'a> {
    CreateView(id::TextureViewId, wgc::resource::TextureViewDescriptor<'a>),
}

#[repr(C)]
#[derive(serde::Serialize, serde::Deserialize)]
enum DropAction {
    Adapter(id::AdapterId),
    Device(id::DeviceId),
    ShaderModule(id::ShaderModuleId),
    PipelineLayout(id::PipelineLayoutId),
    BindGroupLayout(id::BindGroupLayoutId),
    BindGroup(id::BindGroupId),
    CommandBuffer(id::CommandBufferId),
    RenderBundle(id::RenderBundleId),
    RenderPipeline(id::RenderPipelineId),
    ComputePipeline(id::ComputePipelineId),
    Buffer(id::BufferId),
    Texture(id::TextureId),
    TextureView(id::TextureViewId),
    Sampler(id::SamplerId),
}

impl DropAction {
    // helper function to construct byte bufs
    fn to_byte_buf(&self) -> ByteBuf {
        let mut data = Vec::new();
        bincode::serialize_into(&mut data, self).unwrap();
        ByteBuf::from_vec(data)
    }
}
