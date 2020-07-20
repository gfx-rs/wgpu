/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::id;
#[cfg(feature = "trace")]
use std::io::Write as _;
use std::ops::Range;

//TODO: consider a readable Id that doesn't include the backend

type FileName = String;

pub const FILE_NAME: &str = "trace.ron";

pub type BindingResource = crate::binding_model::BindingResource<'static>;

pub type ProgrammableStageDescriptor =
    wgt::ProgrammableStageDescriptor<'static, id::ShaderModuleId>;

pub type ComputePipelineDescriptor =
    wgt::ComputePipelineDescriptor<id::PipelineLayoutId, ProgrammableStageDescriptor>;

pub type VertexBufferDescriptor = wgt::VertexBufferDescriptor<'static>;

pub type VertexStateDescriptor = wgt::VertexStateDescriptor<'static>;

pub type RenderPipelineDescriptor =
    wgt::RenderPipelineDescriptor<'static, id::PipelineLayoutId, ProgrammableStageDescriptor>;

pub type RenderBundleEncoderDescriptor = wgt::RenderBundleEncoderDescriptor<'static>;

#[cfg(feature = "trace")]
pub(crate) fn new_render_bundle_encoder_descriptor(
    label: super::Label,
    context: &super::RenderPassContext,
) -> RenderBundleEncoderDescriptor {
    RenderBundleEncoderDescriptor {
        label: Some(super::own_label(&label).into()),
        color_formats: context.attachments.colors.to_vec().into(),
        depth_stencil_format: context.attachments.depth_stencil,
        sample_count: context.sample_count as u32,
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub enum Action {
    Init {
        desc: wgt::DeviceDescriptor,
        backend: wgt::Backend,
    },
    CreateBuffer(id::BufferId, wgt::BufferDescriptor<String>),
    DestroyBuffer(id::BufferId),
    CreateTexture(id::TextureId, wgt::TextureDescriptor<String>),
    DestroyTexture(id::TextureId),
    CreateTextureView {
        id: id::TextureViewId,
        parent_id: id::TextureId,
        desc: Option<wgt::TextureViewDescriptor<String>>,
    },
    DestroyTextureView(id::TextureViewId),
    CreateSampler(id::SamplerId, wgt::SamplerDescriptor<String>),
    DestroySampler(id::SamplerId),
    CreateSwapChain(id::SwapChainId, wgt::SwapChainDescriptor),
    GetSwapChainTexture {
        id: Option<id::TextureViewId>,
        parent_id: id::SwapChainId,
    },
    PresentSwapChain(id::SwapChainId),
    CreateBindGroupLayout(
        id::BindGroupLayoutId,
        wgt::BindGroupLayoutDescriptor<'static>,
    ),
    DestroyBindGroupLayout(id::BindGroupLayoutId),
    CreatePipelineLayout(
        id::PipelineLayoutId,
        wgt::PipelineLayoutDescriptor<'static, id::BindGroupLayoutId>,
    ),
    DestroyPipelineLayout(id::PipelineLayoutId),
    CreateBindGroup(
        id::BindGroupId,
        wgt::BindGroupDescriptor<
            'static,
            id::BindGroupLayoutId,
            wgt::BindGroupEntry<BindingResource>,
        >,
    ),
    DestroyBindGroup(id::BindGroupId),
    CreateShaderModule {
        id: id::ShaderModuleId,
        data: FileName,
    },
    DestroyShaderModule(id::ShaderModuleId),
    CreateComputePipeline(id::ComputePipelineId, ComputePipelineDescriptor),
    DestroyComputePipeline(id::ComputePipelineId),
    CreateRenderPipeline(id::RenderPipelineId, RenderPipelineDescriptor),
    DestroyRenderPipeline(id::RenderPipelineId),
    CreateRenderBundle {
        id: id::RenderBundleId,
        desc: RenderBundleEncoderDescriptor,
        base: crate::command::BasePass<crate::command::RenderCommand>,
    },
    DestroyRenderBundle(id::RenderBundleId),
    WriteBuffer {
        id: id::BufferId,
        data: FileName,
        range: Range<wgt::BufferAddress>,
        queued: bool,
    },
    WriteTexture {
        to: crate::command::TextureCopyView,
        data: FileName,
        layout: wgt::TextureDataLayout,
        size: wgt::Extent3d,
    },
    Submit(crate::SubmissionIndex, Vec<Command>),
}

#[derive(Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub enum Command {
    CopyBufferToBuffer {
        src: id::BufferId,
        src_offset: wgt::BufferAddress,
        dst: id::BufferId,
        dst_offset: wgt::BufferAddress,
        size: wgt::BufferAddress,
    },
    CopyBufferToTexture {
        src: crate::command::BufferCopyView,
        dst: crate::command::TextureCopyView,
        size: wgt::Extent3d,
    },
    CopyTextureToBuffer {
        src: crate::command::TextureCopyView,
        dst: crate::command::BufferCopyView,
        size: wgt::Extent3d,
    },
    CopyTextureToTexture {
        src: crate::command::TextureCopyView,
        dst: crate::command::TextureCopyView,
        size: wgt::Extent3d,
    },
    RunComputePass {
        base: crate::command::BasePass<crate::command::ComputeCommand>,
    },
    RunRenderPass {
        base: crate::command::BasePass<crate::command::RenderCommand>,
        target_colors: Vec<crate::command::ColorAttachmentDescriptor>,
        target_depth_stencil: Option<crate::command::DepthStencilAttachmentDescriptor>,
    },
}

#[cfg(feature = "trace")]
#[derive(Debug)]
pub struct Trace {
    path: std::path::PathBuf,
    file: std::fs::File,
    config: ron::ser::PrettyConfig,
    binary_id: usize,
}

#[cfg(feature = "trace")]
impl Trace {
    pub fn new(path: &std::path::Path) -> Result<Self, std::io::Error> {
        log::info!("Tracing into '{:?}'", path);
        let mut file = std::fs::File::create(path.join(FILE_NAME))?;
        file.write_all(b"[\n")?;
        Ok(Trace {
            path: path.to_path_buf(),
            file,
            config: ron::ser::PrettyConfig::default(),
            binary_id: 0,
        })
    }

    pub fn make_binary(&mut self, kind: &str, data: &[u8]) -> String {
        self.binary_id += 1;
        let name = format!("data{}.{}", self.binary_id, kind);
        let _ = std::fs::write(self.path.join(&name), data);
        name
    }

    pub(crate) fn add(&mut self, action: Action) {
        match ron::ser::to_string_pretty(&action, self.config.clone()) {
            Ok(string) => {
                let _ = writeln!(self.file, "{},", string);
            }
            Err(e) => {
                log::warn!("RON serialization failure: {:?}", e);
            }
        }
    }
}

#[cfg(feature = "trace")]
impl Drop for Trace {
    fn drop(&mut self) {
        let _ = self.file.write_all(b"]");
    }
}
