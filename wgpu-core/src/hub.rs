/*! Allocating resource ids, and tracking the resources they refer to.

The `wgpu_core` API uses identifiers of type [`Id<R>`] to refer to
resources of type `R`. For example, [`id::DeviceId`] is an alias for
`Id<markers::Device>`, and [`id::BufferId`] is an alias for
`Id<markers::Buffer>`. `Id` implements `Copy`, `Hash`, `Eq`, `Ord`, and
of course `Debug`.

[`id::DeviceId`]: crate::id::DeviceId
[`id::BufferId`]: crate::id::BufferId

Each `Id` contains not only an index for the resource it denotes but
also a Backend indicating which `wgpu` backend it belongs to. You
can use the [`gfx_select`] macro to dynamically dispatch on an id's
backend to a function specialized at compile time for a specific
backend. See that macro's documentation for details.

`Id`s also incorporate a generation number, for additional validation.

The resources to which identifiers refer are freed explicitly.
Attempting to use an identifier for a resource that has been freed
elicits an error result.

## Assigning ids to resources

The users of `wgpu_core` generally want resource ids to be assigned
in one of two ways:

- Users like `wgpu` want `wgpu_core` to assign ids to resources itself.
  For example, `wgpu` expects to call `Global::device_create_buffer`
  and have the return value indicate the newly created buffer's id.

- Users like `player` and Firefox want to allocate ids themselves, and
  pass `Global::device_create_buffer` and friends the id to assign the
  new resource.

To accommodate either pattern, `wgpu_core` methods that create
resources all expect an `id_in` argument that the caller can use to
specify the id, and they all return the id used. For example, the
declaration of `Global::device_create_buffer` looks like this:

```ignore
impl Global {
    /* ... */
    pub fn device_create_buffer<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        desc: &resource::BufferDescriptor,
        id_in: Input<G>,
    ) -> (id::BufferId, Option<resource::CreateBufferError>) {
        /* ... */
    }
    /* ... */
}
```

Users that want to assign resource ids themselves pass in the id they
want as the `id_in` argument, whereas users that want `wgpu_core`
itself to choose ids always pass `()`. In either case, the id
ultimately assigned is returned as the first element of the tuple.

Producing true identifiers from `id_in` values is the job of an
[`crate::identity::IdentityManager`] or ids will be received from outside through `Option<Id>` arguments.

## Id allocation and streaming

Perhaps surprisingly, allowing users to assign resource ids themselves
enables major performance improvements in some applications.

The `wgpu_core` API is designed for use by Firefox's [WebGPU]
implementation. For security, web content and GPU use must be kept
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
flagged as errors as well.

[`gfx_select`]: crate::gfx_select
[`process`]: crate::identity::IdentityManager::process
[`Id<R>`]: crate::id::Id
[wrapped in a mutex]: trait.IdentityHandler.html#impl-IdentityHandler%3CI%3E-for-Mutex%3CIdentityManager%3E
[WebGPU]: https://www.w3.org/TR/webgpu/

*/

use crate::{
    binding_model::{BindGroup, BindGroupLayout, PipelineLayout},
    command::{CommandBuffer, RenderBundle},
    device::{queue::Queue, Device},
    hal_api::HalApi,
    instance::{Adapter, Surface},
    pipeline::{ComputePipeline, RenderPipeline, ShaderModule},
    registry::{Registry, RegistryReport},
    resource::{Buffer, QuerySet, Sampler, StagingBuffer, Texture, TextureView},
    storage::{Element, Storage},
};
use std::fmt::Debug;

#[derive(Debug, PartialEq, Eq)]
pub struct HubReport {
    pub adapters: RegistryReport,
    pub devices: RegistryReport,
    pub queues: RegistryReport,
    pub pipeline_layouts: RegistryReport,
    pub shader_modules: RegistryReport,
    pub bind_group_layouts: RegistryReport,
    pub bind_groups: RegistryReport,
    pub command_buffers: RegistryReport,
    pub render_bundles: RegistryReport,
    pub render_pipelines: RegistryReport,
    pub compute_pipelines: RegistryReport,
    pub query_sets: RegistryReport,
    pub buffers: RegistryReport,
    pub textures: RegistryReport,
    pub texture_views: RegistryReport,
    pub samplers: RegistryReport,
}

impl HubReport {
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }
}

#[allow(rustdoc::private_intra_doc_links)]
/// All the resources for a particular backend in a [`crate::global::Global`].
///
/// To obtain `global`'s `Hub` for some [`HalApi`] backend type `A`,
/// call [`A::hub(global)`].
///
/// ## Locking
///
/// Each field in `Hub` is a [`Registry`] holding all the values of a
/// particular type of resource, all protected by a single RwLock.
/// So for example, to access any [`Buffer`], you must acquire a read
/// lock on the `Hub`s entire buffers registry. The lock guard
/// gives you access to the `Registry`'s [`Storage`], which you can
/// then index with the buffer's id. (Yes, this design causes
/// contention; see [#2272].)
///
/// But most `wgpu` operations require access to several different
/// kinds of resource, so you often need to hold locks on several
/// different fields of your [`Hub`] simultaneously.
///
/// Inside the `Registry` there are `Arc<T>` where `T` is a Resource
/// Lock of `Registry` happens only when accessing to get the specific resource
///
///
/// [`A::hub(global)`]: HalApi::hub
pub struct Hub<A: HalApi> {
    pub(crate) adapters: Registry<Adapter<A>>,
    pub(crate) devices: Registry<Device<A>>,
    pub(crate) queues: Registry<Queue<A>>,
    pub(crate) pipeline_layouts: Registry<PipelineLayout<A>>,
    pub(crate) shader_modules: Registry<ShaderModule<A>>,
    pub(crate) bind_group_layouts: Registry<BindGroupLayout<A>>,
    pub(crate) bind_groups: Registry<BindGroup<A>>,
    pub(crate) command_buffers: Registry<CommandBuffer<A>>,
    pub(crate) render_bundles: Registry<RenderBundle<A>>,
    pub(crate) render_pipelines: Registry<RenderPipeline<A>>,
    pub(crate) compute_pipelines: Registry<ComputePipeline<A>>,
    pub(crate) query_sets: Registry<QuerySet<A>>,
    pub(crate) buffers: Registry<Buffer<A>>,
    pub(crate) staging_buffers: Registry<StagingBuffer<A>>,
    pub(crate) textures: Registry<Texture<A>>,
    pub(crate) texture_views: Registry<TextureView<A>>,
    pub(crate) samplers: Registry<Sampler<A>>,
}

impl<A: HalApi> Hub<A> {
    fn new() -> Self {
        Self {
            adapters: Registry::new(A::VARIANT),
            devices: Registry::new(A::VARIANT),
            queues: Registry::new(A::VARIANT),
            pipeline_layouts: Registry::new(A::VARIANT),
            shader_modules: Registry::new(A::VARIANT),
            bind_group_layouts: Registry::new(A::VARIANT),
            bind_groups: Registry::new(A::VARIANT),
            command_buffers: Registry::new(A::VARIANT),
            render_bundles: Registry::new(A::VARIANT),
            render_pipelines: Registry::new(A::VARIANT),
            compute_pipelines: Registry::new(A::VARIANT),
            query_sets: Registry::new(A::VARIANT),
            buffers: Registry::new(A::VARIANT),
            staging_buffers: Registry::new(A::VARIANT),
            textures: Registry::new(A::VARIANT),
            texture_views: Registry::new(A::VARIANT),
            samplers: Registry::new(A::VARIANT),
        }
    }

    //TODO: instead of having a hacky `with_adapters` parameter,
    // we should have `clear_device(device_id)` that specifically destroys
    // everything related to a logical device.
    pub(crate) fn clear(&self, surface_guard: &Storage<Surface>, with_adapters: bool) {
        use hal::Surface;

        let mut devices = self.devices.write();
        for element in devices.map.iter() {
            if let Element::Occupied(ref device, _) = *element {
                device.prepare_to_die();
            }
        }

        self.command_buffers.write().map.clear();
        self.samplers.write().map.clear();
        self.texture_views.write().map.clear();
        self.textures.write().map.clear();
        self.buffers.write().map.clear();
        self.bind_groups.write().map.clear();
        self.shader_modules.write().map.clear();
        self.bind_group_layouts.write().map.clear();
        self.pipeline_layouts.write().map.clear();
        self.compute_pipelines.write().map.clear();
        self.render_pipelines.write().map.clear();
        self.query_sets.write().map.clear();

        for element in surface_guard.map.iter() {
            if let Element::Occupied(ref surface, _epoch) = *element {
                if let Some(ref mut present) = surface.presentation.lock().take() {
                    if let Some(device) = present.device.downcast_ref::<A>() {
                        let suf = A::surface_as_hal(surface);
                        unsafe {
                            suf.unwrap().unconfigure(device.raw());
                            //TODO: we could destroy the surface here
                        }
                    }
                }
            }
        }

        self.queues.write().map.clear();
        devices.map.clear();

        if with_adapters {
            drop(devices);
            self.adapters.write().map.clear();
        }
    }

    pub(crate) fn surface_unconfigure(&self, device: &Device<A>, surface: &A::Surface) {
        unsafe {
            use hal::Surface;
            surface.unconfigure(device.raw());
        }
    }

    pub fn generate_report(&self) -> HubReport {
        HubReport {
            adapters: self.adapters.generate_report(),
            devices: self.devices.generate_report(),
            queues: self.queues.generate_report(),
            pipeline_layouts: self.pipeline_layouts.generate_report(),
            shader_modules: self.shader_modules.generate_report(),
            bind_group_layouts: self.bind_group_layouts.generate_report(),
            bind_groups: self.bind_groups.generate_report(),
            command_buffers: self.command_buffers.generate_report(),
            render_bundles: self.render_bundles.generate_report(),
            render_pipelines: self.render_pipelines.generate_report(),
            compute_pipelines: self.compute_pipelines.generate_report(),
            query_sets: self.query_sets.generate_report(),
            buffers: self.buffers.generate_report(),
            textures: self.textures.generate_report(),
            texture_views: self.texture_views.generate_report(),
            samplers: self.samplers.generate_report(),
        }
    }
}

pub struct Hubs {
    #[cfg(vulkan)]
    pub(crate) vulkan: Hub<hal::api::Vulkan>,
    #[cfg(metal)]
    pub(crate) metal: Hub<hal::api::Metal>,
    #[cfg(dx12)]
    pub(crate) dx12: Hub<hal::api::Dx12>,
    #[cfg(gles)]
    pub(crate) gl: Hub<hal::api::Gles>,
    #[cfg(all(not(vulkan), not(metal), not(dx12), not(gles)))]
    pub(crate) empty: Hub<hal::api::Empty>,
}

impl Hubs {
    pub(crate) fn new() -> Self {
        Self {
            #[cfg(vulkan)]
            vulkan: Hub::new(),
            #[cfg(metal)]
            metal: Hub::new(),
            #[cfg(dx12)]
            dx12: Hub::new(),
            #[cfg(gles)]
            gl: Hub::new(),
            #[cfg(all(not(vulkan), not(metal), not(dx12), not(gles)))]
            empty: Hub::new(),
        }
    }
}
