#[cfg(feature = "trace")]
mod record;

use core::{convert::Infallible, ops::Range};

use alloc::{string::String, vec::Vec};
use macro_rules_attribute::apply;

use crate::{
    command::{serde_object_reference_struct, BasePass, Command, ReferenceType, RenderCommand},
    id::{markers, PointerId},
    pipeline::GeneralRenderPipelineDescriptor,
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
        id: R::ExternalTexture,
        desc: crate::resource::ExternalTextureDescriptor<'a>,
        planes: alloc::boxed::Box<[R::TextureView]>,
    },
    FreeExternalTexture(R::ExternalTexture),
    DestroyExternalTexture(R::ExternalTexture),
    CreateSampler(
        PointerId<markers::Sampler>,
        crate::resource::SamplerDescriptor<'a>,
    ),
    DestroySampler(PointerId<markers::Sampler>),
    GetSurfaceTexture {
        id: R::Texture,
        parent: R::Surface,
    },
    Present(R::Surface),
    DiscardSurfaceTexture(R::Surface),
    CreateBindGroupLayout(
        PointerId<markers::BindGroupLayout>,
        crate::binding_model::BindGroupLayoutDescriptor<'a>,
    ),
    DestroyBindGroupLayout(PointerId<markers::BindGroupLayout>),
    CreatePipelineLayout(
        PointerId<markers::PipelineLayout>,
        crate::binding_model::ResolvedPipelineLayoutDescriptor<
            'a,
            PointerId<markers::BindGroupLayout>,
        >,
    ),
    DestroyPipelineLayout(PointerId<markers::PipelineLayout>),
    CreateBindGroup(PointerId<markers::BindGroup>, TraceBindGroupDescriptor<'a>),
    DestroyBindGroup(PointerId<markers::BindGroup>),
    CreateShaderModule {
        id: PointerId<markers::ShaderModule>,
        desc: crate::pipeline::ShaderModuleDescriptor<'a>,
        data: FileName,
    },
    CreateShaderModulePassthrough {
        id: PointerId<markers::ShaderModule>,
        data: Vec<FileName>,

        entry_point: String,
        label: crate::Label<'a>,
        num_workgroups: (u32, u32, u32),
        runtime_checks: wgt::ShaderRuntimeChecks,
    },
    DestroyShaderModule(PointerId<markers::ShaderModule>),
    CreateComputePipeline {
        id: PointerId<markers::ComputePipeline>,
        desc: TraceComputePipelineDescriptor<'a>,
    },
    DestroyComputePipeline(PointerId<markers::ComputePipeline>),
    CreateGeneralRenderPipeline {
        id: PointerId<markers::RenderPipeline>,
        desc: TraceGeneralRenderPipelineDescriptor<'a>,
    },
    DestroyRenderPipeline(PointerId<markers::RenderPipeline>),
    CreatePipelineCache {
        id: PointerId<markers::PipelineCache>,
        desc: crate::pipeline::PipelineCacheDescriptor<'a>,
    },
    DestroyPipelineCache(PointerId<markers::PipelineCache>),
    CreateRenderBundle {
        id: R::RenderBundle,
        desc: crate::command::RenderBundleEncoderDescriptor<'a>,
        base: BasePass<RenderCommand<R>, Infallible>,
    },
    DestroyRenderBundle(PointerId<markers::RenderBundle>),
    CreateQuerySet {
        id: PointerId<markers::QuerySet>,
        desc: crate::resource::QuerySetDescriptor<'a>,
    },
    DestroyQuerySet(PointerId<markers::QuerySet>),
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
        id: R::Blas,
        desc: crate::resource::BlasDescriptor<'a>,
        sizes: wgt::BlasGeometrySizeDescriptors,
    },
    DestroyBlas(R::Blas),
    CreateTlas {
        id: R::Tlas,
        desc: crate::resource::TlasDescriptor<'a>,
    },
    DestroyTlas(R::Tlas),
}

/// cbindgen:ignore
pub type TraceBindGroupDescriptor<'a> = crate::binding_model::BindGroupDescriptor<
    'a,
    PointerId<markers::BindGroupLayout>,
    PointerId<markers::Buffer>,
    PointerId<markers::Sampler>,
    PointerId<markers::TextureView>,
    PointerId<markers::Tlas>,
    PointerId<markers::ExternalTexture>,
>;

/// Not a public API. For use by `player` only.
///
/// cbindgen:ignore
#[doc(hidden)]
pub type TraceGeneralRenderPipelineDescriptor<'a> = GeneralRenderPipelineDescriptor<
    'a,
    PointerId<markers::PipelineLayout>,
    PointerId<markers::ShaderModule>,
    PointerId<markers::PipelineCache>,
>;

/// Not a public API. For use by `player` only.
///
/// cbindgen:ignore
#[doc(hidden)]
pub type TraceComputePipelineDescriptor<'a> = crate::pipeline::ComputePipelineDescriptor<
    'a,
    PointerId<markers::PipelineLayout>,
    PointerId<markers::ShaderModule>,
    PointerId<markers::PipelineCache>,
>;
