/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    command::{BufferCopyView, TextureCopyView},
    id,
};
use std::{io::Write as _, ops::Range};

//TODO: consider a readable Id that doesn't include the backend
type FileName = String;

#[derive(serde::Serialize)]
pub enum BindingResource {
    Buffer {
        id: id::BufferId,
        offset: wgt::BufferAddress,
        size: wgt::BufferAddress,
    },
    Sampler(id::SamplerId),
    TextureView(id::TextureViewId),
}

#[derive(serde::Serialize)]
pub(crate) enum Action {
    Init {
        limits: wgt::Limits,
    },
    CreateBuffer {
        id: id::BufferId,
        desc: wgt::BufferDescriptor<String>,
    },
    DestroyBuffer(id::BufferId),
    CreateTexture {
        id: id::TextureId,
        desc: wgt::TextureDescriptor<String>,
    },
    DestroyTexture(id::TextureId),
    CreateTextureView {
        id: id::TextureViewId,
        parent_id: id::TextureId,
        desc: Option<wgt::TextureViewDescriptor<String>>,
    },
    DestroyTextureView(id::TextureViewId),
    CreateSampler {
        id: id::SamplerId,
        desc: wgt::SamplerDescriptor<String>,
    },
    DestroySampler(id::SamplerId),
    CreateSwapChain {
        id: id::SwapChainId,
        desc: wgt::SwapChainDescriptor,
    },
    GetSwapChainTexture {
        id: id::TextureViewId,
        parent_id: id::SwapChainId,
    },
    PresentSwapChain(id::SwapChainId),
    CreateBindGroupLayout {
        id: id::BindGroupLayoutId,
        label: String,
        entries: Vec<crate::binding_model::BindGroupLayoutEntry>,
    },
    DestroyBindGroupLayout(id::BindGroupLayoutId),
    CreatePipelineLayout {
        id: id::PipelineLayoutId,
        bind_group_layouts: Vec<id::BindGroupLayoutId>,
    },
    DestroyPipelineLayout(id::PipelineLayoutId),
    CreateBindGroup {
        id: id::BindGroupId,
        label: String,
        layout_id: id::BindGroupLayoutId,
        entries: std::collections::BTreeMap<u32, BindingResource>,
    },
    DestroyBindGroup(id::BindGroupId),
    CreateShaderModule {
        id: id::ShaderModuleId,
        data: FileName,
    },
    DestroyShaderModule(id::ShaderModuleId),
    WriteBuffer {
        id: id::BufferId,
        data: FileName,
        range: Range<wgt::BufferAddress>,
    },
    Submit(Vec<Command>),
}

#[derive(Debug, serde::Serialize)]
pub(crate) enum Command {
    CopyBufferToBuffer {
        src: id::BufferId,
        src_offset: wgt::BufferAddress,
        dst: id::BufferId,
        dst_offset: wgt::BufferAddress,
        size: wgt::BufferAddress,
    },
    CopyBufferToTexture {
        src: BufferCopyView,
        dst: TextureCopyView,
        size: wgt::Extent3d,
    },
    CopyTextureToBuffer {
        src: TextureCopyView,
        dst: BufferCopyView,
        size: wgt::Extent3d,
    },
    CopyTextureToTexture {
        src: TextureCopyView,
        dst: TextureCopyView,
        size: wgt::Extent3d,
    },
    RunComputePass {
        commands: Vec<crate::command::ComputeCommand>,
        dynamic_offsets: Vec<wgt::DynamicOffset>,
    },
    RunRenderPass {
        target_colors: Vec<crate::command::RenderPassColorAttachmentDescriptor>,
        target_depth_stencil: Option<crate::command::RenderPassDepthStencilAttachmentDescriptor>,
        commands: Vec<crate::command::RenderCommand>,
        dynamic_offsets: Vec<wgt::DynamicOffset>,
    },
}

#[derive(Debug)]
pub struct Trace {
    path: std::path::PathBuf,
    file: std::fs::File,
    config: ron::ser::PrettyConfig,
    binary_id: usize,
}

impl Trace {
    pub fn new(path: &std::path::Path) -> Result<Self, std::io::Error> {
        log::info!("Tracing into '{:?}'", path);
        let mut file = std::fs::File::create(path.join("trace.ron"))?;
        file.write(b"[\n")?;
        Ok(Trace {
            path: path.to_path_buf(),
            file,
            config: ron::ser::PrettyConfig::default(),
            binary_id: 0,
        })
    }

    pub fn make_binary(&mut self, kind: &str, data: &[u8]) -> String {
        self.binary_id += 1;
        let name = format!("{}{}.bin", kind, self.binary_id);
        let _ = std::fs::write(self.path.join(&name), data);
        name
    }

    pub(crate) fn add(&mut self, action: Action) {
        match ron::ser::to_string_pretty(&action, self.config.clone()) {
            Ok(string) => {
                let _ = write!(self.file, "{},\n", string);
            }
            Err(e) => {
                log::warn!("RON serialization failure: {:?}", e);
            }
        }
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        let _ = self.file.write(b"]");
    }
}
