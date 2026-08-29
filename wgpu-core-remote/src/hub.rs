/*! Allocating resource ids, and tracking the resources they refer to.

The `wgpu_core` API uses identifiers of type [`Id<R>`] to refer to
resources of type `R`. For example, [`id::DeviceId`] is an alias for
`Id<markers::Device>`, and [`id::BufferId`] is an alias for
`Id<markers::Buffer>`. `Id` implements `Copy`, `Hash`, `Eq`, `Ord`, and
of course `Debug`.

[`id::DeviceId`]: crate::id::DeviceId
[`id::BufferId`]: crate::id::BufferId

[`Id`]s also incorporate a generation number, for additional validation.

The resources to which identifiers refer are freed explicitly.
Attempting to use an identifier for a resource that has been freed
elicits an panic.

## Assigning ids to resources

Firefox and Servo allocate ids themselves in the content process using [`IdentityHub`],
then pass then via IPC to GPU process where it's passed down to
`Global::device_create_buffer` and friends the id to assign the new
resource.

Methods that create resources all expect an `id_in` argument
that the caller uses to specify the id. For example, the
declaration of `Global::device_create_buffer` looks like this:

```ignore
impl Global {
    /* ... */
    pub fn device_create_buffer(
        &self,
        device_id: id::DeviceId,
        desc: &BufferDescriptor,
        id_in: id::BufferId,
    ) {
        /* ... */
    }
    /* ... */
}
```

## Id allocation and streaming

Allowing users to assign resource ids themselves
enables major performance improvements in some applications.

This is designed for use by Firefox's and Servo's [WebGPU] implementation.
For security, web content and GPU use must be kept
segregated in separate processes, with all interaction between them
mediated by an inter-process communication protocol. As web content uses
the WebGPU API, the content process sends messages to the GPU process,
which interacts with the platform's GPU APIs on content's behalf,
occasionally sending results back.

In a classic Rust API, a resource allocation function takes parameters
describing the resource to create, and if creation succeeds, it returns
the resource id in a `Result::Ok` value. However, this design is a poor
fit for the split-process design described above: content must wait for
the reply to its buffer-creation message (say) before it can know which
id it can use in the next message that uses that buffer. On a common
usage pattern, the classic Rust design imposes the latency of a full
cross-process round trip.

We can avoid incurring these round-trip latencies simply by letting the
content process assign resource ids itself. With this approach, content
can choose an id for the new buffer, send a message to create the
buffer, and then immediately send the next message operating on that
buffer, since it already knows its id. Allowing content and GPU process
activity to be pipelined greatly improves throughput.

To help propagate errors correctly in this style of usage, when resource
creation fails, the id supplied for that resource is marked to indicate
as much, allowing subsequent operations using that id to be properly
flagged as errors as well. This is called [Contagious Invalidity].

[`Id<R>`]: crate::id::Id
[`Id`]: crate::id::Id
[WebGPU]: https://www.w3.org/TR/webgpu/
[`IdentityHub`]: wgpu_core_remote_types::identity::IdentityHub
[Contagious Invalidity]: https://www.w3.org/TR/webgpu/#invalidity

*/

use alloc::sync::Arc;

use crate::registry::Registry;
use wgpu_core::{
    binding_model::{BindGroup, BindGroupLayout, PipelineLayout},
    command::{
        CommandBuffer, CommandEncoder, ComputePass, RenderBundle, RenderBundleEncoder, RenderPass,
    },
    device::{queue::Queue, Device},
    instance::Adapter,
    pipeline::{ComputePipeline, RenderPipeline, ShaderModule},
    resource::{Buffer, ExternalTexture, QuerySet, Sampler, Texture, TextureView},
};

#[allow(rustdoc::private_intra_doc_links)]
/// All the resources tracked by a [`crate::global::Global`].
///
/// Each field in [`Hub`] is a [`Registry`] holding all the values of a
/// particular type of resource.
///
/// Most `wgpu` operations require access to several different
/// kinds of resource, so to obtain mutable references
/// to different fields of your [`Hub`] simultaneously, use
/// destructing:
/// ```ignore
/// fn f(hub: &mut Hub) {
///     let Hub {
///         command_encoders,
///         command_buffers,
///         ..
///     } = hub;
/// }
/// ```
pub struct Hub {
    pub(crate) adapters: Registry<Arc<Adapter>>,
    pub(crate) devices: Registry<Arc<Device>>,
    pub(crate) queues: Registry<Arc<Queue>>,
    pub(crate) pipeline_layouts: Registry<Arc<PipelineLayout>>,
    pub(crate) shader_modules: Registry<Arc<ShaderModule>>,
    pub(crate) bind_group_layouts: Registry<Arc<BindGroupLayout>>,
    pub(crate) bind_groups: Registry<Arc<BindGroup>>,
    pub(crate) command_encoders: Registry<Arc<CommandEncoder>>,
    pub(crate) command_buffers: Registry<Arc<CommandBuffer>>,
    pub(crate) render_bundles: Registry<Arc<RenderBundle>>,
    pub(crate) render_pipelines: Registry<Arc<RenderPipeline>>,
    pub(crate) compute_pipelines: Registry<Arc<ComputePipeline>>,
    pub(crate) query_sets: Registry<Arc<QuerySet>>,
    pub(crate) buffers: Registry<Arc<Buffer>>,
    pub(crate) textures: Registry<Arc<Texture>>,
    pub(crate) texture_views: Registry<Arc<TextureView>>,
    pub(crate) external_textures: Registry<Arc<ExternalTexture>>,
    pub(crate) samplers: Registry<Arc<Sampler>>,
    pub(crate) render_passes: Registry<RenderPass>,
    pub(crate) compute_passes: Registry<ComputePass>,
    pub(crate) render_bundle_encoders: Registry<RenderBundleEncoder>,
}

impl Hub {
    pub(crate) fn new() -> Self {
        Self {
            adapters: Registry::new(),
            devices: Registry::new(),
            queues: Registry::new(),
            pipeline_layouts: Registry::new(),
            shader_modules: Registry::new(),
            bind_group_layouts: Registry::new(),
            bind_groups: Registry::new(),
            command_encoders: Registry::new(),
            command_buffers: Registry::new(),
            render_bundles: Registry::new(),
            render_pipelines: Registry::new(),
            compute_pipelines: Registry::new(),
            query_sets: Registry::new(),
            buffers: Registry::new(),
            textures: Registry::new(),
            texture_views: Registry::new(),
            external_textures: Registry::new(),
            samplers: Registry::new(),
            render_passes: Registry::new(),
            compute_passes: Registry::new(),
            render_bundle_encoders: Registry::new(),
        }
    }
}
