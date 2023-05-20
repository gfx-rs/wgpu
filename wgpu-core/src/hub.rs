/*! Allocating resource ids, and tracking the resources they refer to.

The `wgpu_core` API uses identifiers of type [`Id<R>`] to refer to
resources of type `R`. For example, [`id::DeviceId`] is an alias for
`Id<Device<Empty>>`, and [`id::BufferId`] is an alias for
`Id<Buffer<Empty>>`. `Id` implements `Copy`, `Hash`, `Eq`, `Ord`, and
of course `Debug`.

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
impl<G: GlobalIdentityHandlerFactory> Global<G> {
    /* ... */
    pub fn device_create_buffer<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        desc: &resource::BufferDescriptor,
        id_in: Input<G, id::BufferId>,
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
[`crate::identity::IdentityHandler`] implementation, which has an associated type
[`Input`] saying what type of `id_in` values it accepts, and a
[`process`] method that turns such values into true identifiers of
type `I`. There are two kinds of `IdentityHandler`s:

- Users that want `wgpu_core` to assign ids generally use
  [`crate::identity::IdentityManager`] ([wrapped in a mutex]). Its `Input` type is
  `()`, and it tracks assigned ids and generation numbers as
  necessary. (This is what `wgpu` does.)

- Users that want to assign ids themselves use an `IdentityHandler`
  whose `Input` type is `I` itself, and whose `process` method simply
  passes the `id_in` argument through unchanged. For example, the
  `player` crate uses an `IdentityPassThrough` type whose `process`
  method simply adjusts the id's backend (since recordings can be
  replayed on a different backend than the one they were created on)
  but passes the rest of the id's content through unchanged.

Because an `IdentityHandler<I>` can only create ids for a single
resource type `I`, constructing a [`crate::global::Global`] entails constructing a
separate `IdentityHandler<I>` for each resource type `I` that the
`Global` will manage: an `IdentityHandler<DeviceId>`, an
`IdentityHandler<TextureId>`, and so on.

The [`crate::global::Global::new`] function could simply take a large collection of
`IdentityHandler<I>` implementations as arguments, but that would be
ungainly. Instead, `Global::new` expects a `factory` argument that
implements the [`GlobalIdentityHandlerFactory`] trait, which extends
[`crate::identity::IdentityHandlerFactory<I>`] for each resource id type `I`. This
trait, in turn, has a `spawn` method that constructs an
`IdentityHandler<I>` for the `Global` to use.

What this means is that the types of resource creation functions'
`id_in` arguments depend on the `Global`'s `G` type parameter. A
`Global<G>`'s `IdentityHandler<I>` implementation is:

```ignore
<G as IdentityHandlerFactory<I>>::Filter
```

where `Filter` is an associated type of the `IdentityHandlerFactory` trait.
Thus, its `id_in` type is:

```ignore
<<G as IdentityHandlerFactory<I>>::Filter as IdentityHandler<I>>::Input
```

The [`crate::identity::Input<G, I>`] type is an alias for this construction.

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
[`Input`]: crate::identity::IdentityHandler::Input
[`process`]: crate::identity::IdentityHandler::process
[`Id<R>`]: crate::id::Id
[wrapped in a mutex]: trait.IdentityHandler.html#impl-IdentityHandler%3CI%3E-for-Mutex%3CIdentityManager%3E
[WebGPU]: https://www.w3.org/TR/webgpu/

*/

use crate::{
    binding_model::{BindGroup, BindGroupLayout, PipelineLayout},
    command::{CommandBuffer, RenderBundle},
    device::Device,
    hal_api::HalApi,
    id,
    identity::GlobalIdentityHandlerFactory,
    instance::{Adapter, HalSurface, Surface},
    pipeline::{ComputePipeline, RenderPipeline, ShaderModule},
    registry::Registry,
    resource::{Buffer, QuerySet, Sampler, StagingBuffer, Texture, TextureView},
    storage::{Element, Storage, StorageReport},
};

use std::{fmt::Debug, marker::PhantomData};

#[derive(Debug)]
pub struct HubReport {
    pub adapters: StorageReport,
    pub devices: StorageReport,
    pub pipeline_layouts: StorageReport,
    pub shader_modules: StorageReport,
    pub bind_group_layouts: StorageReport,
    pub bind_groups: StorageReport,
    pub command_buffers: StorageReport,
    pub render_bundles: StorageReport,
    pub render_pipelines: StorageReport,
    pub compute_pipelines: StorageReport,
    pub query_sets: StorageReport,
    pub buffers: StorageReport,
    pub textures: StorageReport,
    pub texture_views: StorageReport,
    pub samplers: StorageReport,
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
pub struct Hub<A: HalApi, F: GlobalIdentityHandlerFactory> {
    pub adapters: Registry<id::AdapterId, Adapter<A>, F>,
    pub devices: Registry<id::DeviceId, Device<A>, F>,
    pub pipeline_layouts: Registry<id::PipelineLayoutId, PipelineLayout<A>, F>,
    pub shader_modules: Registry<id::ShaderModuleId, ShaderModule<A>, F>,
    pub bind_group_layouts: Registry<id::BindGroupLayoutId, BindGroupLayout<A>, F>,
    pub bind_groups: Registry<id::BindGroupId, BindGroup<A>, F>,
    pub command_buffers: Registry<id::CommandBufferId, CommandBuffer<A>, F>,
    pub render_bundles: Registry<id::RenderBundleId, RenderBundle<A>, F>,
    pub render_pipelines: Registry<id::RenderPipelineId, RenderPipeline<A>, F>,
    pub compute_pipelines: Registry<id::ComputePipelineId, ComputePipeline<A>, F>,
    pub query_sets: Registry<id::QuerySetId, QuerySet<A>, F>,
    pub buffers: Registry<id::BufferId, Buffer<A>, F>,
    pub staging_buffers: Registry<id::StagingBufferId, StagingBuffer<A>, F>,
    pub textures: Registry<id::TextureId, Texture<A>, F>,
    pub texture_views: Registry<id::TextureViewId, TextureView<A>, F>,
    pub samplers: Registry<id::SamplerId, Sampler<A>, F>,
}

impl<A: HalApi, F: GlobalIdentityHandlerFactory> Hub<A, F> {
    fn new(factory: &F) -> Self {
        Self {
            adapters: Registry::new(A::VARIANT, factory),
            devices: Registry::new(A::VARIANT, factory),
            pipeline_layouts: Registry::new(A::VARIANT, factory),
            shader_modules: Registry::new(A::VARIANT, factory),
            bind_group_layouts: Registry::new(A::VARIANT, factory),
            bind_groups: Registry::new(A::VARIANT, factory),
            command_buffers: Registry::new(A::VARIANT, factory),
            render_bundles: Registry::new(A::VARIANT, factory),
            render_pipelines: Registry::new(A::VARIANT, factory),
            compute_pipelines: Registry::new(A::VARIANT, factory),
            query_sets: Registry::new(A::VARIANT, factory),
            buffers: Registry::new(A::VARIANT, factory),
            staging_buffers: Registry::new(A::VARIANT, factory),
            textures: Registry::new(A::VARIANT, factory),
            texture_views: Registry::new(A::VARIANT, factory),
            samplers: Registry::new(A::VARIANT, factory),
        }
    }

    //TODO: instead of having a hacky `with_adapters` parameter,
    // we should have `clear_device(device_id)` that specifically destroys
    // everything related to a logical device.
    pub(crate) fn clear(
        &self,
        surface_guard: &Storage<Surface, id::SurfaceId>,
        with_adapters: bool,
    ) {
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
                if surface
                    .presentation
                    .lock()
                    .as_ref()
                    .map_or(wgt::Backend::Empty, |p| p.backend())
                    != A::VARIANT
                {
                    continue;
                }
                if let Some(present) = surface.presentation.lock().take() {
                    let device = &devices[present.device_id];
                    let suf = A::get_surface(surface);
                    unsafe {
                        suf.unwrap().raw.unconfigure(device.raw());
                        //TODO: we could destroy the surface here
                    }
                }
            }
        }

        devices.map.clear();

        if with_adapters {
            drop(devices);
            self.adapters.write().map.clear();
        }
    }

    pub(crate) fn surface_unconfigure(
        &self,
        device_id: id::Valid<id::DeviceId>,
        surface: &HalSurface<A>,
    ) {
        let device = self.devices.get(device_id.0).unwrap();
        unsafe {
            use hal::Surface;
            surface.raw.unconfigure(device.raw());
        }
    }

    pub fn generate_report(&self) -> HubReport {
        HubReport {
            adapters: self.adapters.generate_report(),
            devices: self.devices.generate_report(),
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

pub struct Hubs<F: GlobalIdentityHandlerFactory> {
    #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
    pub(crate) vulkan: Hub<hal::api::Vulkan, F>,
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    pub(crate) metal: Hub<hal::api::Metal, F>,
    #[cfg(all(feature = "dx12", windows))]
    pub(crate) dx12: Hub<hal::api::Dx12, F>,
    #[cfg(all(feature = "dx11", windows))]
    pub(crate) dx11: Hub<hal::api::Dx11, F>,
    #[cfg(feature = "gles")]
    pub(crate) gl: Hub<hal::api::Gles, F>,
    _phantom_data: PhantomData<F>,
}

impl<F: GlobalIdentityHandlerFactory> Hubs<F> {
    pub(crate) fn new(factory: &F) -> Self {
        Self {
            #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
            vulkan: Hub::new(factory),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            metal: Hub::new(factory),
            #[cfg(all(feature = "dx12", windows))]
            dx12: Hub::new(factory),
            #[cfg(all(feature = "dx11", windows))]
            dx11: Hub::new(factory),
            #[cfg(feature = "gles")]
            gl: Hub::new(factory),
            _phantom_data: PhantomData,
        }
    }
}
