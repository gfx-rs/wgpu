#[cfg(feature = "trace")]
mod record;

use core::{convert::Infallible, ops::Range};

use alloc::{string::String, vec::Vec};
use macro_rules_attribute::apply;

use crate::{
    command::{serde_object_reference_struct, BasePass, Command, ReferenceType, RenderCommand},
    id,
};

#[cfg(feature = "trace")]
pub use record::*;

type FileName = String;

pub const FILE_NAME: &str = "trace.ron";

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
#[apply(serde_object_reference_struct)]
pub enum Action<'a, R: ReferenceType> {
    Init {
        desc: crate::device::DeviceDescriptor<'a>,
        backend: wgt::Backend,
    },
    ConfigureSurface(
        R::Surface,
        wgt::SurfaceConfiguration<Vec<wgt::TextureFormat>>,
    ),
    CreateBuffer(R::Buffer, crate::resource::BufferDescriptor<'a>),
    FreeBuffer(R::Buffer),
    DestroyBuffer(R::Buffer),
    CreateTexture(R::Texture, crate::resource::TextureDescriptor<'a>),
    FreeTexture(R::Texture),
    DestroyTexture(R::Texture),
    CreateTextureView {
        id: R::TextureView,
        parent: R::Texture,
        desc: crate::resource::TextureViewDescriptor<'a>,
    },
    DestroyTextureView(R::TextureView),
    CreateExternalTexture {
        id: id::ExternalTextureId,
        desc: crate::resource::ExternalTextureDescriptor<'a>,
        planes: alloc::boxed::Box<[id::TextureViewId]>,
    },
    FreeExternalTexture(id::ExternalTextureId),
    DestroyExternalTexture(id::ExternalTextureId),
    CreateSampler(id::SamplerId, crate::resource::SamplerDescriptor<'a>),
    DestroySampler(id::SamplerId),
    GetSurfaceTexture {
        id: R::Texture,
        parent: R::Surface,
    },
    Present(R::Surface),
    DiscardSurfaceTexture(R::Surface),
    CreateBindGroupLayout(
        id::BindGroupLayoutId,
        crate::binding_model::BindGroupLayoutDescriptor<'a>,
    ),
    DestroyBindGroupLayout(id::BindGroupLayoutId),
    CreatePipelineLayout(
        id::PipelineLayoutId,
        crate::binding_model::PipelineLayoutDescriptor<'a>,
    ),
    DestroyPipelineLayout(id::PipelineLayoutId),
    CreateBindGroup(
        id::BindGroupId,
        crate::binding_model::BindGroupDescriptor<'a>,
    ),
    DestroyBindGroup(id::BindGroupId),
    CreateShaderModule {
        id: id::ShaderModuleId,
        desc: crate::pipeline::ShaderModuleDescriptor<'a>,
        data: FileName,
    },
    CreateShaderModulePassthrough {
        id: id::ShaderModuleId,
        data: Vec<FileName>,

        entry_point: String,
        label: crate::Label<'a>,
        num_workgroups: (u32, u32, u32),
        runtime_checks: wgt::ShaderRuntimeChecks,
    },
    DestroyShaderModule(id::ShaderModuleId),
    CreateComputePipeline {
        id: id::ComputePipelineId,
        desc: crate::pipeline::ComputePipelineDescriptor<'a>,
    },
    DestroyComputePipeline(id::ComputePipelineId),
    CreateRenderPipeline {
        id: id::RenderPipelineId,
        desc: crate::pipeline::RenderPipelineDescriptor<'a>,
    },
    CreateMeshPipeline {
        id: id::RenderPipelineId,
        desc: crate::pipeline::MeshPipelineDescriptor<'a>,
    },
    DestroyRenderPipeline(id::RenderPipelineId),
    CreatePipelineCache {
        id: id::PipelineCacheId,
        desc: crate::pipeline::PipelineCacheDescriptor<'a>,
    },
    DestroyPipelineCache(id::PipelineCacheId),
    CreateRenderBundle {
        id: R::RenderBundle,
        desc: crate::command::RenderBundleEncoderDescriptor<'a>,
        base: BasePass<RenderCommand<R>, Infallible>,
    },
    DestroyRenderBundle(id::RenderBundleId),
    CreateQuerySet {
        id: id::QuerySetId,
        desc: crate::resource::QuerySetDescriptor<'a>,
    },
    DestroyQuerySet(id::QuerySetId),
    WriteBuffer {
        id: R::Buffer,
        data: FileName,
        range: Range<wgt::BufferAddress>,
        queued: bool,
    },
    WriteTexture {
        to: wgt::TexelCopyTextureInfo<R::Texture>,
        data: FileName,
        layout: wgt::TexelCopyBufferLayout,
        size: wgt::Extent3d,
    },
    Submit(crate::SubmissionIndex, Vec<Command<R>>),
    CreateBlas {
        id: id::BlasId,
        desc: crate::resource::BlasDescriptor<'a>,
        sizes: wgt::BlasGeometrySizeDescriptors,
    },
    DestroyBlas(id::BlasId),
    CreateTlas {
        id: id::TlasId,
        desc: crate::resource::TlasDescriptor<'a>,
    },
    DestroyTlas(id::TlasId),
}
