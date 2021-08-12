use crate::id;
#[cfg(feature = "trace")]
use parking_lot::Mutex;
use std::ops::Range;
#[cfg(feature = "trace")]
use std::{borrow::Cow, io::Write as _, sync::atomic::{AtomicU64, Ordering}};

#[cfg(any(feature = "trace", feature="replay"))]
type FileName = String;

#[cfg(any(feature = "trace", feature="replay"))]
pub const FILE_NAME: &str = "trace.ron";

#[cfg(feature = "trace")]
pub(crate) fn new_render_bundle_encoder_descriptor<'a>(
    label: crate::Label<'a>,
    context: &'a super::RenderPassContext,
    is_ds_read_only: bool,
) -> crate::command::RenderBundleEncoderDescriptor<'a> {
    crate::command::RenderBundleEncoderDescriptor {
        label,
        color_formats: Cow::Borrowed(&context.attachments.colors),
        depth_stencil: context.attachments.depth_stencil.map(|format| {
            let aspects = hal::FormatAspects::from(format);
            wgt::RenderBundleDepthStencil {
                format,
                depth_read_only: is_ds_read_only && aspects.contains(hal::FormatAspects::DEPTH),
                stencil_read_only: is_ds_read_only && aspects.contains(hal::FormatAspects::STENCIL),
            }
        }),
        sample_count: context.sample_count,
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
#[cfg(any(feature = "trace", feature="replay"))]
pub enum Action<'a> {
    Init {
        #[serde(borrow)]
        desc: crate::device::DeviceDescriptor<'a>,
        backend: wgt::Backend,
    },
    /// Resolves a temporary TraceResourceId to its final address.
    Assign { trace_id: TraceResourceId, resource_id: usize },
    ConfigureSurface(id::SurfaceId, wgt::SurfaceConfiguration),
    CreateBuffer(id::BufferId,
                 #[serde(borrow)]
                 crate::resource::BufferDescriptor<'a>),
    FreeBuffer(id::BufferId),
    DestroyBuffer(id::BufferId),
    CreateTexture(id::TextureId,
                  #[serde(borrow)]
                  crate::resource::TextureDescriptor<'a>),
    FreeTexture(id::TextureId),
    DestroyTexture(id::TextureId),
    CreateTextureView {
        id: id::TextureViewId,
        parent_id: id::TextureId,
        #[serde(borrow)]
        desc: crate::resource::TextureViewDescriptor<'a>,
    },
    DestroyTextureView(id::TextureViewId),
    CreateSampler(/*id::SamplerId*/TraceResourceId,
                  #[serde(borrow)]
                  crate::resource::SamplerDescriptor<'a>
                  ),
    GetSurfaceTexture {
        id: id::TextureId,
        parent_id: id::SurfaceId,
    },
    Present(id::SurfaceId),
    CreateBindGroupLayout(
        /*id::BindGroupLayoutId*/TraceResourceId,
        #[serde(borrow)]
        crate::binding_model::BindGroupLayoutDescriptor<'a>,
    ),
    CreatePipelineLayout(
        /*id::PipelineLayoutId*/TraceResourceId,
        #[serde(borrow)]
        crate::binding_model::PipelineLayoutDescriptor<'a, hal::api::Empty, id::UsizeCon>,
    ),
    CreateBindGroup(
        /*id::Id<crate::binding_model::BindGroup<hal::api::Empty>>*/TraceResourceId,
        #[serde(borrow)]
        crate::binding_model::BindGroupDescriptor<'a, hal::api::Empty, id::UsizeCon,
        Vec<crate::binding_model::BindGroupEntry<'a, hal::api::Empty, id::UsizeCon>>>,
    ),
    CreateShaderModule {
        id: /*id::ShaderModuleId*/TraceResourceId,
        #[serde(borrow)]
        desc: crate::pipeline::ShaderModuleDescriptor<'a>,
        data: FileName,
    },
    CreateComputePipeline {
        id: /*id::ComputePipelineId*/TraceResourceId,
        #[serde(borrow)]
        desc: crate::pipeline::ComputePipelineDescriptor<'a, hal::api::Empty, id::UsizeCon>,
        /* #[cfg_attr(feature = "replay", serde(default))]
        implicit_context: Option<super::ImplicitPipelineContext>, */
    },
    GetComputePipelineBindGroupLayout {
        pipeline_id: usize,
        index: u32,
        resource_id: usize,
    },
    CreateRenderPipeline {
        id: /*id::RenderPipelineId*/TraceResourceId,
        #[serde(borrow)]
        desc: crate::pipeline::RenderPipelineDescriptor<'a, hal::api::Empty, id::UsizeCon>,
        /* #[cfg_attr(feature = "replay", serde(default))]
        implicit_context: Option<super::ImplicitPipelineContext>, */
    },
    GetRenderPipelineBindGroupLayout {
        pipeline_id: usize,
        index: u32,
        resource_id: usize,
    },
    CreateRenderBundle {
        id: /*id::RenderBundleId*/TraceResourceId,
        #[serde(borrow)]
        desc: crate::command::RenderBundleEncoderDescriptor<'a>,
        base: crate::command::BasePass<crate::command::RenderCommand<hal::api::Empty, /*id::IdCon*/id::UsizeCon>>,
    },
    CreateQuerySet {
        id: /*id::QuerySetId*/TraceResourceId,
        #[serde(borrow)]
        desc: crate::resource::QuerySetDescriptor<'a>,
    },
    WriteBuffer {
        id: id::BufferId,
        data: FileName,
        range: Range<wgt::BufferAddress>,
        queued: bool,
    },
    WriteTexture {
        to: crate::command::ImageCopyTexture,
        data: FileName,
        layout: wgt::ImageDataLayout,
        size: wgt::Extent3d,
    },
    Submit(crate::SubmissionIndex, Vec<Command>),
}

#[derive(Debug)]
#[cfg(any(feature = "trace", feature="replay"))]
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
        src: crate::command::ImageCopyBuffer,
        dst: crate::command::ImageCopyTexture,
        size: wgt::Extent3d,
    },
    CopyTextureToBuffer {
        src: crate::command::ImageCopyTexture,
        dst: crate::command::ImageCopyBuffer,
        size: wgt::Extent3d,
    },
    CopyTextureToTexture {
        src: crate::command::ImageCopyTexture,
        dst: crate::command::ImageCopyTexture,
        size: wgt::Extent3d,
    },
    ClearBuffer {
        dst: id::BufferId,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    },
    ClearImage {
        dst: id::TextureId,
        subresource_range: wgt::ImageSubresourceRange,
    },
    WriteTimestamp {
        query_set_id: /*id::QuerySetId*/usize,
        query_index: u32,
    },
    ResolveQuerySet {
        query_set_id: /*id::QuerySetId*/usize,
        start_query: u32,
        query_count: u32,
        destination: id::BufferId,
        destination_offset: wgt::BufferAddress,
    },
    RunComputePass {
        base: crate::command::BasePass<crate::command::ComputeCommand<hal::api::Empty, /*id::IdCon*/id::UsizeCon>>,
    },
    RunRenderPass {
        base: crate::command::BasePass<crate::command::RenderCommand<hal::api::Empty, /*id::IdCon*/id::UsizeCon>>,
        target_colors: Vec<crate::command::RenderPassColorAttachment>,
        target_depth_stencil: Option<crate::command::RenderPassDepthStencilAttachment>,
    },
}

#[cfg(feature = "replay")]
impl Command {
    #[inline]
    pub fn trace_resources<'b, E>(
        &'b self,
        mut f: impl FnMut(id::Cached<hal::api::Empty, &'b id::UsizeCon>) -> Result<(), E>,
    ) -> Result<(), E>
    {
        use id::Cached;
        use Command::*;

        match self {
            CopyBufferToBuffer { src: _, dst: _, .. } => {
                // FIXME: Uncomment when Buffer is Arc'd.
                // f(Cached::Buffer(&src.buffer))?;
                // FIXME: Uncomment when Buffer is Arc'd.
                // f(Cached::Buffer(&dst.buffer))
                Ok(())
            },
            CopyBufferToTexture { src: _, dst: _, .. } => {
                // FIXME: Uncomment when Buffer is Arc'd.
                // f(Cached::Buffer(&src.buffer))?;
                // FIXME: Uncomment when Texture is Arc'd.
                // f(Cached::Texture(&dst.texture))
                Ok(())
            },
            CopyTextureToBuffer { src: _, dst: _, .. } => {
                // FIXME: Uncomment when Texture is Arc'd.
                // f(Cached::Texture(&src.texture))?;
                // FIXME: Uncomment when Buffer is Arc'd.
                // f(Cached::Buffer(&dst.buffer))
                Ok(())
            },
            CopyTextureToTexture { src: _, dst: _, .. } => {
                // FIXME: Uncomment when Texture is Arc'd.
                // f(Cached::Texture(&src.texture))?;
                // FIXME: Uncomment when Texture is Arc'd.
                // f(Cached::Texture(&dst.texture))
                Ok(())
            },
            ClearBuffer { dst: _, .. } =>
                // FIXME: Uncomment when Buffer is Arc'd.
                // f(Cached::Buffer(dst))
                Ok(()),
            ClearImage { dst: _, .. } =>
                // FIXME: Uncomment when Texture is Arc'd.
                // f(Cached::Texture(dst)),
                Ok(()),
            WriteTimestamp { query_set_id, .. } => f(Cached::QuerySet(query_set_id)),
            ResolveQuerySet { query_set_id, destination: _, .. } => {
                // FIXME: Uncomment when Buffer is Arc'd.
                // f(Cached::Buffer(destination))?;
                f(Cached::QuerySet(query_set_id))
            },
            RunComputePass { base } =>
                base.commands.iter().try_for_each(|command| command.trace_resources(&mut f)),
            RunRenderPass { base, .. } =>
                base.commands.iter().try_for_each(|command| command.trace_resources(&mut f)),
        }
    }
}

/// Unique temporary id used for associating descriptors with ids, so we can write a message
/// with resource creation information before we construct an actual pointer for the resource.
/// This makes sure creation messages are written before execution of the creation action
/// (so if creation crashes, we can still figure out what caused it) without needing unsafe
/// code (since to figure out the real address for the resource without initializing it, we would
/// need to allocate uninitialized memory, then fill it in).
#[cfg(any(feature = "trace", feature="replay"))]
pub type TraceResourceId = u64;

#[cfg(feature = "trace")]
#[derive(Debug)]
pub struct Trace {
    path: std::path::PathBuf,
    /// TODO: Consider exposing synchronization to the user of wgpu-core, since soon
    /// there will be thread-local contexts which might be a more appropriate place to trace.
    file: Mutex<std::fs::File>,
    config: ron::ser::PrettyConfig,
    /// TODO: If we need AtomicCell for something else, use
    /// AtomicCell<NonZeroUsize> to create a niche for `Option<Trace>` to
    /// take up the same amount of space as `Trace`.
    binary_id: AtomicU64,
}

#[cfg(feature = "trace")]
impl Trace {
    pub fn new(path: &std::path::Path) -> Result<Self, std::io::Error> {
        log::info!("Tracing into '{:?}'", path);
        let mut file = std::fs::File::create(path.join(FILE_NAME))?;
        file.write_all(b"[\n")?;
        let _ = file.sync_all();
        Ok(Self {
            path: path.to_path_buf(),
            file: Mutex::new(file),
            config: ron::ser::PrettyConfig::default(),
            binary_id: AtomicU64::new(0),
        })
    }

    pub(crate) fn make_binary(&self, kind: &str, data: &[u8]) -> String {
        let binary_id = self.create_resource_id();
        let name = format!("data{}.{}", binary_id, kind);
        let _ = std::fs::File::create(&self.path.join(&name)).and_then(|mut file| {
            file.write_all(data)/*?;*/
            // Survive crashes of various kinds; see documentation in add.
            // file.sync_all()
        });
        name
    }

    pub(crate) fn create_resource_id(&self) -> TraceResourceId {
        // This id is unique for the `Trace` instance it came from (modulo overflow, which we
        // don't care to handle as tracing is not relied on for memory safety), so it can be
        // used to create a unique resource id across the trace instance without locking.
        self.binary_id.fetch_add(1, Ordering::Relaxed)
    }

    pub(crate) fn add(&self, action: Action) {
        match ron::ser::to_string_pretty(&action, self.config.clone()) {
            Ok(string) => {
                let mut file = self.file.lock();
                // We want to try to make sure data are flushed on crash, so we conservatively
                // fsync every successful write.  This is slow, but it's currently considered
                // acceptable overhead.  If in the future people want to leave tracing on for other
                // purposes and don't care about crash safety, we may add other options (like
                // buffered or thread-local writes) that avoid this overhead.
                let _ = writeln!(&mut *file, "{},", string);
                    // .and_then(|_| file.sync_all());
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
        let _ = self.file.get_mut().write_all(b"]");
    }
}
