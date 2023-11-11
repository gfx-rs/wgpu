/*! Allocating resource ids, and tracking the resources they refer to.

The `wgpu_core` API uses identifiers of type [`Id<R>`] to refer to
resources of type `R`. For example, [`id::DeviceId`] is an alias for
`Id<Device<Empty>>`, and [`id::BufferId`] is an alias for
`Id<Buffer<Empty>>`. `Id` implements `Copy`, `Hash`, `Eq`, `Ord`, and
of course `Debug`.

Each `Id` contains not only an index for the resource it denotes but
also a [`Backend`] indicating which `wgpu` backend it belongs to. You
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
[`IdentityHandler`] implementation, which has an associated type
[`Input`] saying what type of `id_in` values it accepts, and a
[`process`] method that turns such values into true identifiers of
type `I`. There are two kinds of `IdentityHandler`s:

- Users that want `wgpu_core` to assign ids generally use
  [`IdentityManager`] ([wrapped in a mutex]). Its `Input` type is
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
resource type `I`, constructing a [`Global`] entails constructing a
separate `IdentityHandler<I>` for each resource type `I` that the
`Global` will manage: an `IdentityHandler<DeviceId>`, an
`IdentityHandler<TextureId>`, and so on.

The [`Global::new`] function could simply take a large collection of
`IdentityHandler<I>` implementations as arguments, but that would be
ungainly. Instead, `Global::new` expects a `factory` argument that
implements the [`GlobalIdentityHandlerFactory`] trait, which extends
[`IdentityHandlerFactory<I>`] for each resource id type `I`. This
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

The [`Input<G, I>`] type is an alias for this construction.

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

[`Backend`]: wgt::Backend
[`Global`]: crate::global::Global
[`Global::new`]: crate::global::Global::new
[`gfx_select`]: crate::gfx_select
[`IdentityHandler`]: crate::identity::IdentityHandler
[`Input`]: crate::identity::IdentityHandler::Input
[`process`]: crate::identity::IdentityHandler::process
[`Id<R>`]: crate::id::Id
[wrapped in a mutex]: ../identity/trait.IdentityHandler.html#impl-IdentityHandler%3CI%3E-for-Mutex%3CIdentityManager%3E
[WebGPU]: https://www.w3.org/TR/webgpu/
[`IdentityManager`]: crate::identity::IdentityManager
[`Input<G, I>`]: crate::identity::Input
[`IdentityHandlerFactory<I>`]: crate::identity::IdentityHandlerFactory
*/

use crate::{
    binding_model::{BindGroup, BindGroupLayout, PipelineLayout},
    command::{CommandBuffer, RenderBundle},
    device::Device,
    hal_api::HalApi,
    id,
    identity::GlobalIdentityHandlerFactory,
    instance::{Adapter, HalSurface, Instance, Surface},
    pipeline::{ComputePipeline, RenderPipeline, ShaderModule},
    registry::Registry,
    resource::{Buffer, QuerySet, Sampler, StagingBuffer, Texture, TextureClearMode, TextureView},
    storage::{Element, Storage, StorageReport},
};

use wgt::{strict_assert_eq, strict_assert_ne};

#[cfg(any(debug_assertions, feature = "strict_asserts"))]
use std::cell::Cell;
use std::{fmt::Debug, marker::PhantomData};

/// Type system for enforcing the lock order on [`Hub`] fields.
///
/// If type `A` implements `Access<B>`, that means we are allowed to
/// proceed with locking resource `B` after we lock `A`.
///
/// The implementations of `Access` basically describe the edges in an
/// acyclic directed graph of lock transitions. As long as it doesn't have
/// cycles, any number of threads can acquire locks along paths through
/// the graph without deadlock. That is, if you look at each thread's
/// lock acquisitions as steps along a path in the graph, then because
/// there are no cycles in the graph, there must always be some thread
/// that is able to acquire its next lock, or that is about to release
/// a lock. (Assume that no thread just sits on its locks forever.)
///
/// Locks must be acquired in the following order:
///
/// - [`Adapter`]
/// - [`Device`]
/// - [`CommandBuffer`]
/// - [`RenderBundle`]
/// - [`PipelineLayout`]
/// - [`BindGroupLayout`]
/// - [`BindGroup`]
/// - [`ComputePipeline`]
/// - [`RenderPipeline`]
/// - [`ShaderModule`]
/// - [`Buffer`]
/// - [`StagingBuffer`]
/// - [`Texture`]
/// - [`TextureView`]
/// - [`Sampler`]
/// - [`QuerySet`]
///
/// That is, you may only acquire a new lock on a `Hub` field if it
/// appears in the list after all the other fields you're already
/// holding locks for. When you are holding no locks, you can start
/// anywhere.
///
/// It's fine to add more `Access` implementations as needed, as long
/// as you do not introduce a cycle. In other words, as long as there
/// is some ordering you can put the resource types in that respects
/// the extant `Access` implementations, that's fine.
///
/// See the documentation for [`Hub`] for more details.
pub trait Access<A> {}

pub enum Root {}

// These impls are arranged so that the target types (that is, the `T`
// in `Access<T>`) appear in locking order.
//
// TODO: establish an order instead of declaring all the pairs.
impl Access<Instance> for Root {}
impl Access<Surface> for Root {}
impl Access<Surface> for Instance {}
impl<A: HalApi> Access<Adapter<A>> for Root {}
impl<A: HalApi> Access<Adapter<A>> for Surface {}
impl<A: HalApi> Access<Device<A>> for Root {}
impl<A: HalApi> Access<Device<A>> for Surface {}
impl<A: HalApi> Access<Device<A>> for Adapter<A> {}
impl<A: HalApi> Access<CommandBuffer<A>> for Root {}
impl<A: HalApi> Access<CommandBuffer<A>> for Device<A> {}
impl<A: HalApi> Access<RenderBundle<A>> for Device<A> {}
impl<A: HalApi> Access<RenderBundle<A>> for CommandBuffer<A> {}
impl<A: HalApi> Access<PipelineLayout<A>> for Root {}
impl<A: HalApi> Access<PipelineLayout<A>> for Device<A> {}
impl<A: HalApi> Access<PipelineLayout<A>> for RenderBundle<A> {}
impl<A: HalApi> Access<BindGroupLayout<A>> for Root {}
impl<A: HalApi> Access<BindGroupLayout<A>> for Device<A> {}
impl<A: HalApi> Access<BindGroupLayout<A>> for PipelineLayout<A> {}
impl<A: HalApi> Access<BindGroupLayout<A>> for QuerySet<A> {}
impl<A: HalApi> Access<BindGroup<A>> for Root {}
impl<A: HalApi> Access<BindGroup<A>> for Device<A> {}
impl<A: HalApi> Access<BindGroup<A>> for BindGroupLayout<A> {}
impl<A: HalApi> Access<BindGroup<A>> for PipelineLayout<A> {}
impl<A: HalApi> Access<BindGroup<A>> for CommandBuffer<A> {}
impl<A: HalApi> Access<ComputePipeline<A>> for Device<A> {}
impl<A: HalApi> Access<ComputePipeline<A>> for BindGroup<A> {}
impl<A: HalApi> Access<RenderPipeline<A>> for Device<A> {}
impl<A: HalApi> Access<RenderPipeline<A>> for BindGroup<A> {}
impl<A: HalApi> Access<RenderPipeline<A>> for ComputePipeline<A> {}
impl<A: HalApi> Access<ShaderModule<A>> for Device<A> {}
impl<A: HalApi> Access<ShaderModule<A>> for BindGroupLayout<A> {}
impl<A: HalApi> Access<Buffer<A>> for Root {}
impl<A: HalApi> Access<Buffer<A>> for Device<A> {}
impl<A: HalApi> Access<Buffer<A>> for BindGroupLayout<A> {}
impl<A: HalApi> Access<Buffer<A>> for BindGroup<A> {}
impl<A: HalApi> Access<Buffer<A>> for CommandBuffer<A> {}
impl<A: HalApi> Access<Buffer<A>> for ComputePipeline<A> {}
impl<A: HalApi> Access<Buffer<A>> for RenderPipeline<A> {}
impl<A: HalApi> Access<Buffer<A>> for QuerySet<A> {}
impl<A: HalApi> Access<StagingBuffer<A>> for Device<A> {}
impl<A: HalApi> Access<Texture<A>> for Root {}
impl<A: HalApi> Access<Texture<A>> for Device<A> {}
impl<A: HalApi> Access<Texture<A>> for Buffer<A> {}
impl<A: HalApi> Access<TextureView<A>> for Root {}
impl<A: HalApi> Access<TextureView<A>> for Device<A> {}
impl<A: HalApi> Access<TextureView<A>> for Texture<A> {}
impl<A: HalApi> Access<Sampler<A>> for Root {}
impl<A: HalApi> Access<Sampler<A>> for Device<A> {}
impl<A: HalApi> Access<Sampler<A>> for TextureView<A> {}
impl<A: HalApi> Access<QuerySet<A>> for Root {}
impl<A: HalApi> Access<QuerySet<A>> for Device<A> {}
impl<A: HalApi> Access<QuerySet<A>> for CommandBuffer<A> {}
impl<A: HalApi> Access<QuerySet<A>> for RenderPipeline<A> {}
impl<A: HalApi> Access<QuerySet<A>> for ComputePipeline<A> {}
impl<A: HalApi> Access<QuerySet<A>> for Sampler<A> {}

#[cfg(any(debug_assertions, feature = "strict_asserts"))]
thread_local! {
    /// Per-thread state checking `Token<Root>` creation in debug builds.
    ///
    /// This is the number of `Token` values alive on the current
    /// thread. Since `Token` creation respects the [`Access`] graph,
    /// there can never be more tokens alive than there are fields of
    /// [`Hub`], so a `u8` is plenty.
    static ACTIVE_TOKEN: Cell<u8> = Cell::new(0);
}

/// A zero-size permission token to lock some fields of [`Hub`].
///
/// Access to a `Token<T>` grants permission to lock any field of
/// [`Hub`] following the one of type [`Registry<T, ...>`], where
/// "following" is as defined by the [`Access`] implementations.
///
/// Calling [`Token::root()`] returns a `Token<Root>`, which grants
/// permission to lock any field. Dynamic checks ensure that each
/// thread has at most one `Token<Root>` live at a time, in debug
/// builds.
///
/// The locking methods on `Registry<T, ...>` take a `&'t mut
/// Token<A>`, and return a fresh `Token<'t, T>` and a lock guard with
/// lifetime `'t`, so the caller cannot access their `Token<A>` again
/// until they have dropped both the `Token<T>` and the lock guard.
///
/// Tokens are `!Send`, so one thread can't send its permissions to
/// another.
pub(crate) struct Token<'a, T: 'a> {
    // The `*const` makes us `!Send` and `!Sync`.
    level: PhantomData<&'a *const T>,
}

impl<'a, T> Token<'a, T> {
    /// Return a new token for a locked field.
    ///
    /// This should only be used by `Registry` locking methods.
    pub(crate) fn new() -> Self {
        #[cfg(any(debug_assertions, feature = "strict_asserts"))]
        ACTIVE_TOKEN.with(|active| {
            let old = active.get();
            strict_assert_ne!(old, 0, "Root token was dropped");
            active.set(old + 1);
        });
        Self { level: PhantomData }
    }
}

impl Token<'static, Root> {
    /// Return a `Token<Root>`, granting permission to lock any [`Hub`] field.
    ///
    /// Debug builds check dynamically that each thread has at most
    /// one root token at a time.
    pub fn root() -> Self {
        #[cfg(any(debug_assertions, feature = "strict_asserts"))]
        ACTIVE_TOKEN.with(|active| {
            strict_assert_eq!(0, active.replace(1), "Root token is already active");
        });

        Self { level: PhantomData }
    }
}

impl<'a, T> Drop for Token<'a, T> {
    fn drop(&mut self) {
        #[cfg(any(debug_assertions, feature = "strict_asserts"))]
        ACTIVE_TOKEN.with(|active| {
            let old = active.get();
            active.set(old - 1);
        });
    }
}

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
/// All the resources for a particular backend in a [`Global`].
///
/// To obtain `global`'s `Hub` for some [`HalApi`] backend type `A`,
/// call [`A::hub(global)`].
///
/// ## Locking
///
/// Each field in `Hub` is a [`Registry`] holding all the values of a
/// particular type of resource, all protected by a single [`RwLock`].
/// So for example, to access any [`Buffer`], you must acquire a read
/// lock on the `Hub`s entire [`buffers`] registry. The lock guard
/// gives you access to the `Registry`'s [`Storage`], which you can
/// then index with the buffer's id. (Yes, this design causes
/// contention; see [#2272].)
///
/// But most `wgpu` operations require access to several different
/// kinds of resource, so you often need to hold locks on several
/// different fields of your [`Hub`] simultaneously. To avoid
/// deadlock, there is an ordering imposed on the fields, and you may
/// only acquire new locks on fields that come *after* all those you
/// are already holding locks on, in this ordering. (The ordering is
/// described in the documentation for the [`Access`] trait.)
///
/// We use Rust's type system to statically check that `wgpu_core` can
/// only ever acquire locks in the correct order:
///
/// - A value of type [`Token<T>`] represents proof that the owner
///   only holds locks on the `Hub` fields holding resources of type
///   `T` or earlier in the lock ordering. A special value of type
///   `Token<Root>`, obtained by calling [`Token::root`], represents
///   proof that no `Hub` field locks are held.
///
/// - To lock the `Hub` field holding resources of type `T`, you must
///   call its [`read`] or [`write`] methods. These require you to
///   pass in a `&mut Token<A>`, for some `A` that implements
///   [`Access<T>`]. This implementation exists only if `T` follows `A`
///   in the field ordering, which statically ensures that you are
///   indeed allowed to lock this new `Hub` field.
///
/// - The locking methods return both an [`RwLock`] guard that you can
///   use to access the field's resources, and a new `Token<T>` value.
///   These both borrow from the lifetime of your `Token<A>`, so since
///   you passed that by mutable reference, you cannot access it again
///   until you drop the new token and lock guard.
///
/// Because a thread only ever has access to the `Token<T>` for the
/// last resource type `T` it holds a lock for, and the `Access` trait
/// implementations only permit acquiring locks for types `U` that
/// follow `T` in the lock ordering, it is statically impossible for a
/// program to violate the locking order.
///
/// This does assume that threads cannot call `Token<Root>` when they
/// already hold locks (dynamically enforced in debug builds) and that
/// threads cannot send their `Token`s to other threads (enforced by
/// making `Token` neither `Send` nor `Sync`).
///
/// [`Global`]: crate::global::Global
/// [`A::hub(global)`]: HalApi::hub
/// [`RwLock`]: parking_lot::RwLock
/// [`buffers`]: Hub::buffers
/// [`read`]: Registry::read
/// [`write`]: Registry::write
/// [`Token<T>`]: Token
/// [`Access<T>`]: Access
/// [#2272]: https://github.com/gfx-rs/wgpu/pull/2272
pub struct Hub<A: HalApi, F: GlobalIdentityHandlerFactory> {
    pub adapters: Registry<Adapter<A>, id::AdapterId, F>,
    pub devices: Registry<Device<A>, id::DeviceId, F>,
    pub pipeline_layouts: Registry<PipelineLayout<A>, id::PipelineLayoutId, F>,
    pub shader_modules: Registry<ShaderModule<A>, id::ShaderModuleId, F>,
    pub bind_group_layouts: Registry<BindGroupLayout<A>, id::BindGroupLayoutId, F>,
    pub bind_groups: Registry<BindGroup<A>, id::BindGroupId, F>,
    pub command_buffers: Registry<CommandBuffer<A>, id::CommandBufferId, F>,
    pub render_bundles: Registry<RenderBundle<A>, id::RenderBundleId, F>,
    pub render_pipelines: Registry<RenderPipeline<A>, id::RenderPipelineId, F>,
    pub compute_pipelines: Registry<ComputePipeline<A>, id::ComputePipelineId, F>,
    pub query_sets: Registry<QuerySet<A>, id::QuerySetId, F>,
    pub buffers: Registry<Buffer<A>, id::BufferId, F>,
    pub staging_buffers: Registry<StagingBuffer<A>, id::StagingBufferId, F>,
    pub textures: Registry<Texture<A>, id::TextureId, F>,
    pub texture_views: Registry<TextureView<A>, id::TextureViewId, F>,
    pub samplers: Registry<Sampler<A>, id::SamplerId, F>,
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
        surface_guard: &mut Storage<Surface, id::SurfaceId>,
        with_adapters: bool,
    ) {
        use crate::resource::TextureInner;
        use hal::{Device as _, Surface as _};

        let mut devices = self.devices.data.write();
        for element in devices.map.iter_mut() {
            if let Element::Occupied(ref mut device, _) = *element {
                device.prepare_to_die();
            }
        }

        // destroy command buffers first, since otherwise DX12 isn't happy
        for element in self.command_buffers.data.write().map.drain(..) {
            if let Element::Occupied(command_buffer, _) = element {
                let device = &devices[command_buffer.device_id.value];
                device.destroy_command_buffer(command_buffer);
            }
        }

        for element in self.samplers.data.write().map.drain(..) {
            if let Element::Occupied(sampler, _) = element {
                unsafe {
                    devices[sampler.device_id.value]
                        .raw
                        .destroy_sampler(sampler.raw);
                }
            }
        }

        for element in self.texture_views.data.write().map.drain(..) {
            if let Element::Occupied(texture_view, _) = element {
                let device = &devices[texture_view.device_id.value];
                unsafe {
                    device.raw.destroy_texture_view(texture_view.raw);
                }
            }
        }

        for element in self.textures.data.write().map.drain(..) {
            if let Element::Occupied(texture, _) = element {
                let device = &devices[texture.device_id.value];
                if let TextureInner::Native { raw: Some(raw) } = texture.inner {
                    unsafe {
                        device.raw.destroy_texture(raw);
                    }
                }
                if let TextureClearMode::RenderPass { clear_views, .. } = texture.clear_mode {
                    for view in clear_views {
                        unsafe {
                            device.raw.destroy_texture_view(view);
                        }
                    }
                }
            }
        }
        for element in self.buffers.data.write().map.drain(..) {
            if let Element::Occupied(buffer, _) = element {
                //TODO: unmap if needed
                devices[buffer.device_id.value].destroy_buffer(buffer);
            }
        }
        for element in self.bind_groups.data.write().map.drain(..) {
            if let Element::Occupied(bind_group, _) = element {
                let device = &devices[bind_group.device_id.value];
                unsafe {
                    device.raw.destroy_bind_group(bind_group.raw);
                }
            }
        }

        for element in self.shader_modules.data.write().map.drain(..) {
            if let Element::Occupied(module, _) = element {
                let device = &devices[module.device_id.value];
                unsafe {
                    device.raw.destroy_shader_module(module.raw);
                }
            }
        }
        for element in self.bind_group_layouts.data.write().map.drain(..) {
            if let Element::Occupied(bgl, _) = element {
                let device = &devices[bgl.device_id.value];
                if let Some(inner) = bgl.into_inner() {
                    unsafe {
                        device.raw.destroy_bind_group_layout(inner.raw);
                    }
                }
            }
        }
        for element in self.pipeline_layouts.data.write().map.drain(..) {
            if let Element::Occupied(pipeline_layout, _) = element {
                let device = &devices[pipeline_layout.device_id.value];
                unsafe {
                    device.raw.destroy_pipeline_layout(pipeline_layout.raw);
                }
            }
        }
        for element in self.compute_pipelines.data.write().map.drain(..) {
            if let Element::Occupied(pipeline, _) = element {
                let device = &devices[pipeline.device_id.value];
                unsafe {
                    device.raw.destroy_compute_pipeline(pipeline.raw);
                }
            }
        }
        for element in self.render_pipelines.data.write().map.drain(..) {
            if let Element::Occupied(pipeline, _) = element {
                let device = &devices[pipeline.device_id.value];
                unsafe {
                    device.raw.destroy_render_pipeline(pipeline.raw);
                }
            }
        }

        for element in surface_guard.map.iter_mut() {
            if let Element::Occupied(ref mut surface, _epoch) = *element {
                if surface
                    .presentation
                    .as_ref()
                    .map_or(wgt::Backend::Empty, |p| p.backend())
                    != A::VARIANT
                {
                    continue;
                }
                if let Some(present) = surface.presentation.take() {
                    let device = &devices[present.device_id.value];
                    let suf = A::get_surface_mut(surface);
                    unsafe {
                        suf.unwrap().raw.unconfigure(&device.raw);
                        //TODO: we could destroy the surface here
                    }
                }
            }
        }

        for element in self.query_sets.data.write().map.drain(..) {
            if let Element::Occupied(query_set, _) = element {
                let device = &devices[query_set.device_id.value];
                unsafe {
                    device.raw.destroy_query_set(query_set.raw);
                }
            }
        }

        for element in devices.map.drain(..) {
            if let Element::Occupied(device, _) = element {
                device.dispose();
            }
        }

        if with_adapters {
            drop(devices);
            self.adapters.data.write().map.clear();
        }
    }

    pub(crate) fn surface_unconfigure(
        &self,
        device_id: id::Valid<id::DeviceId>,
        surface: &mut HalSurface<A>,
    ) {
        use hal::Surface as _;

        let devices = self.devices.data.read();
        let device = &devices[device_id];
        unsafe {
            surface.raw.unconfigure(&device.raw);
        }
    }

    pub fn generate_report(&self) -> HubReport {
        HubReport {
            adapters: self.adapters.data.read().generate_report(),
            devices: self.devices.data.read().generate_report(),
            pipeline_layouts: self.pipeline_layouts.data.read().generate_report(),
            shader_modules: self.shader_modules.data.read().generate_report(),
            bind_group_layouts: self.bind_group_layouts.data.read().generate_report(),
            bind_groups: self.bind_groups.data.read().generate_report(),
            command_buffers: self.command_buffers.data.read().generate_report(),
            render_bundles: self.render_bundles.data.read().generate_report(),
            render_pipelines: self.render_pipelines.data.read().generate_report(),
            compute_pipelines: self.compute_pipelines.data.read().generate_report(),
            query_sets: self.query_sets.data.read().generate_report(),
            buffers: self.buffers.data.read().generate_report(),
            textures: self.textures.data.read().generate_report(),
            texture_views: self.texture_views.data.read().generate_report(),
            samplers: self.samplers.data.read().generate_report(),
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
    #[cfg(all(
        not(all(feature = "vulkan", not(target_arch = "wasm32"))),
        not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))),
        not(all(feature = "dx12", windows)),
        not(all(feature = "dx11", windows)),
        not(feature = "gles"),
    ))]
    pub(crate) empty: Hub<hal::api::Empty, F>,
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
            #[cfg(all(
                not(all(feature = "vulkan", not(target_arch = "wasm32"))),
                not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))),
                not(all(feature = "dx12", windows)),
                not(all(feature = "dx11", windows)),
                not(feature = "gles"),
            ))]
            empty: Hub::new(factory),
        }
    }
}
