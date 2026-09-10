#![doc = crate::macros::doc_image!("architecture.webp")]
#![doc = crate::macros::doc_image!("big-picture.webp")]
/*!
# Architecture: wgpu, wgpu-core, and wgpu-hal

Here's an overview of `wgpu`'s architecture.

![wgpu architecture][architecture.webp]

Working through this diagram from the bottom up:

- Each operating system provides its own API (or APIs) for getting at the
  GPU:

  - On Windows, [Direct3D] is the primary GPU programming API, but [Vulkan]
    and [OpenGL] are often also available.

  - On macOS, [Metal] is the official GPU programming API, although OpenGL is
    still around for legacy programs, and the [MoltenVK] open-source project
    implements the Vulkan API on top of Metal.

  - On Linux, the [Mesa] library supports Vulkan and OpenGL.

  If you want to write an application that runs on all three of these
  platforms, you have a few options:

  - You could target Vulkan, and for macOS support embed MoltenVK, or
    require your users to install it.

  - You could target OpenGL, which is no longer being developed and is
    missing many modern features, and worry that macOS will drop support for
    it altogether.

  - You could port it to all three native platform APIs: Direct3D, Vulkan,
    and Metal. This would be a lot of code.

  - Or, you could just use `wgpu`!

- The [`wgpu_hal`] crate implements a portable Rust API that can use any of
  the platform-specific APIs mentioned above as a backend: a program that
  uses `wgpu_hal` correctly should behave consistently regardless of what
  platform you're running on.

  However, `wgpu_hal`'s interface is completely unsafe; you must follow all
  of its safety requirements to the letter to avoid provoking undefined
  behavior from the underlying platforms. These requirements are complex,
  and not well documented. And `wgpu_hal` performs almost no validation
  beyond the minimum necessary to ensure portability.

- The [`wgpu_core`] crate builds on `wgpu_hal` to provide an API with a
  similar flavor, preserving its portability, and adding full bullet-proof
  validation. By "bullet-proof", we mean that `wgpu_core` is intended to be
  driven by untrusted code, like web content using the [WebGPU] API.
  `wgpu_core` only assumes that Rust's safety rules are respected. Safe Rust
  code should not be able to cause a crash using `wgpu_core`.

  This means that `wgpu_core` is fully responsible for things like tracking
  resource lifetimes, generating barriers for usage transitions, checking
  parameters, and so on. We cover its duties in more detail below.

  The `wgpu_core` crate's API is designed to be easy to use from other
  languages via foreign function interfaces (FFI). This restricts the Rust
  features we can use in the API somewhat. But this makes `wgpu_core` the
  best dependency for crates like `wgpu_native` and applications like Deno
  and Firefox that have their own binding systems.

- The [`wgpu`](crate) crate provides an idiomatic Rust API on top of
  `wgpu_core`, inheriting its validation and portability. This is what most
  users in the Rust ecosystem will want.

  You can also compile `wgpu` to WebAssembly and run it in a web browser,
  having it drive the browser's [WebGPU] implementation. In this mode of use,
  `wgpu_core` and `wgpu_hal` are not present. This can help you share code
  between in-browser and native versions of your app, by providing a single
  API for both versions to use.

[WebGPU]: https://www.w3.org/TR/webgpu/
[Direct3D]: https://learn.microsoft.com/en-us/windows/win32/direct3d
[Vulkan]: https://www.vulkan.org/
[Metal]: https://developer.apple.com/metal/
[OpenGL]: https://www.opengl.org/
[MoltenVK]: https://github.com/KhronosGroup/MoltenVK
[Mesa]: https://mesa3d.org/
[`wgpu_hal`]: https://docs.rs/wgpu-hal/latest/wgpu_hal/
[`wgpu_core`]: https://docs.rs/wgpu-core/latest/wgpu_core/

## wgpu

The `wgpu` crate provides an idiomatic Rust API for cross-platform GPU-based
graphics and computation, modeled after the [WebGPU] JavaScript API.

The `wgpu` crate itself doesn't contain much interesting graphics-related
code. It is mostly concerned with providing a consistent, idiomatic API that
applications can use to drive any one of several implementations:

- The [`wgpu::backend::wgpu_core`][w:cwc] module is available in native or
  Emscripten environments. It forwards all operations to `wgpu_core`. Most
  [`Backend`] values, like `Vulkan` or `Metal`, go through this module.

- The [`wgpu::backend::webgpu`][w:cwg] module, available when `wgpu` has been
  compiled to WebAssembly, forwards calls to a browser's WebGPU
  implementation. The [`Backend::BrowserWebGpu`] backend selects this module.

- The [`wgpu::backend::custom`][w:cc] module uses dynamic dispatch to let the
  user supply their own implementation. These are created via a separate
  mechanism, so they don't use any [`Backend`] value.

[w:cwc]: https://github.com/gfx-rs/wgpu/blob/trunk/wgpu/src/backend/wgpu_core.rs
[w:cwg]: https://github.com/gfx-rs/wgpu/blob/trunk/wgpu/src/backend/webgpu.rs
[w:cc]: https://github.com/gfx-rs/wgpu/blob/trunk/wgpu/src/backend/custom.rs

## wgpu_core

The `wgpu_core` crate implements a safe, cross-platform API on top of
`wgpu_hal`'s unsafe, cross-platform API. Almost every method in `wgpu_hal`'s
API is marked `unsafe`; it is `wgpu_core`'s job to satisfy or enforce every
requirement in `wgpu_hal`'s safety contracts. Then, `wgpu_hal`'s only
responsibility is to get consistent behavior across all its backends. This
relationship is explained more in
[`wgpu_hal`'s documentation][hal safety].

Beyond safety, `wgpu_core` also implements a few convenience APIs (for
example, mapped-at-creation buffers) on top of `wgpu_hal`'s more primitive
features.

[hal safety]: https://docs.rs/wgpu-hal/latest/wgpu_hal/#validation-is-the-calling-codes-responsibility-not-wgpu-hals

### Lifetime tracking

The `wgpu_hal` API requires some objects to outlive others. For example, a
[`wgpu_hal::Device`] must outlive all resources (`wgpu_hal::Buffer`,
`wgpu_hal::CommandEncoder`, etc.) created from it. The `wgpu_core` types
ensure that these requirements are upheld, mostly just by using `Arc`.

TODO: Command-buffer resource usage through submission probably counts as
"lifetime tracking" too.

This `wgpu_hal` requirement arises mostly from the Vulkan backend. Direct3D
and Metal both use reference counting internally for most operations, but
Vulkan foists the entire problem off on its users.

[`wgpu_hal::Device`]: https://docs.rs/wgpu-hal/latest/wgpu_hal/trait.Device.html

### Synchronization

Some types in the `wgpu_hal` API cannot be accessed simultaneously by
multiple threads. For example, most [`wgpu_hal::CommandEncoder`] methods take
`&mut self`. `wgpu_core` protects these hal types with `Mutex` as
appropriate.

There are a few other interesting requirements; for example, calls to
[`wgpu_hal::Queue::submit`] on a given `Queue` that could occur
simultaneously must use different `Fence`s. `wgpu_core` addresses this by
simply wrapping `wgpu_hal::Queue` in a `Mutex`.

[`wgpu_hal::CommandEncoder`]: https://docs.rs/wgpu-hal/latest/wgpu_hal/trait.CommandEncoder.html
[`wgpu_hal::Queue::submit`]: https://docs.rs/wgpu-hal/latest/wgpu_hal/trait.Queue.html#tymethod.submit

### Barrier generation

Vulkan and Direct3D require the user to include barriers in the command
stream to notify the driver when a particular buffer or texture is
transitioning from one kind of use to another. For example, if a texture is
used as a render target and then sampled by a shader, Direct3D 12 requires a
[ResourceBarrier] command between the two uses. Similarly, Vulkan requires
[VkCmdPipelineBarrier] commands in some cases to ensure that values written
to a resource by one operation will be seen by reads in a subsequent
operation.

The `wgpu_hal` API passes these backends' requirements through to its own
users, requiring calls to [`transition_buffers`] and [`transition_textures`]
between different kinds of access to such resources. But, following the lead
of [WebGPU], the `wgpu` API does not require its users to record barriers.
This means that `wgpu_core` must track how each resource was last used, and
generate the barriers itself.

Since `wgpu` encourages users to record `CommandBuffer`s in advance (ideally
in multiple threads), and then submit them in whatever order they please,
`wgpu_core` cannot know what states the resources that a `CommandBuffer` uses
will be in until it is actually submitted. Thus, each
`wgpu_core::CommandBuffer` merely records what states it expects the
resources it uses to be in initially, and which states it will leave those
resources in when it is done. Submission is then responsible for comparing
resources' actual states with those expected by each
`wgpu_core::CommandBuffer`, and recording the necessary barriers in a fresh
`wgpu_hal::CommandBuffer` submitted ahead of it.

When recording a command buffer, since the commands are only being recorded
for later execution, and not executed immediately, a `CommandEncoder` tracks
the state each resource *will be in* when the given commands are executed,
and adds barriers to the recorded stream as appropriate. Naturally, the
`CommandEncoder` must treat each resource's initial state as unknown, as it
has no way to know what other commands will be submitted ahead of it.

Note that, within a single `wgpu_hal::Texture`, individual array elements and
mip levels can be in different states. `wgpu_hal::Buffer`s, however, have only
a single state.

Within `wgpu_core`, the `Tracker` type records a set of resources, their
current states, and (for command buffers being recorded) the initial state
they should be in. `Tracker` also notes which stateless resources (samplers;
render pipelines) a `CommandBuffer` uses, in order to keep them alive until
the commands have been submitted and finished running.

[ResourceBarrier]: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-resourcebarrier
[VkCmdPipelineBarrier]: https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdPipelineBarrier.html
[`transition_buffers`]: https://docs.rs/wgpu-hal/latest/wgpu_hal/trait.DynCommandEncoder.html#tymethod.transition_buffers
[`transition_textures`]: https://docs.rs/wgpu-hal/latest/wgpu_hal/trait.DynCommandEncoder.html#tymethod.transition_textures

### MemoryInitTracker

Tracks the memory-initialization state of buffer resources, i.e. whether
they have been written to either by the user or by `wgpu` previously
zero-initializing them (WebGPU requires all buffers to behave *as if* they
are zero-initialized). A `MemoryInitTracker` tracks a single buffer and
allows inserting zero-initialization lazily on use. Zero-init is inserted if
necessary at:

- memory mapping, and
- `queue_submit` (for all bindings).

### Device maintain

TODO (what/why).

### Cross-device use

The `wgpu_hal` API requires that a resource created with one
[`wgpu_hal::Device`] may not be used with a different `Device`. `wgpu_core`
tracks the `Device` to which each resource belongs, and checks for
cross-device use where necessary.

### General parameter validation

The `wgpu_core` crate is responsible for all the usual parameter validation
for graphics operations: a texture's format must be supported by the device;
copies must not exceed the bounds of the buffers or textures involved; bind
groups need to have the right layout for the pipeline; and so on.

## API tracing

Enabled via a feature flag. Allows recording all usage (WebGPU API functions,
essentially) to a `.ron` file. Any additional data (data uploaded to a
buffer, etc.) is stored in additional files. A trace can be replayed with the
`player` compiled from the same `wgpu` revision. Used for testing and bug
reporting. See
[Debugging wgpu Applications](crate::documentation::debugging::debugging_applications#tracing-infrastructure).

## Other WebGPU implementations

Note that `wgpu` represents only one possible way to implement the
[WebGPU][WebGPU API] API; Google's [Dawn] and
[WebKit's WebGPU module][webkit] are other WebGPU implementations used in
production.

[WebGPU API]: https://www.w3.org/TR/webgpu/
[Dawn]: https://dawn.googlesource.com/dawn
[webkit]: https://github.com/WebKit/WebKit/tree/main/Source/WebCore/Modules/WebGPU

## The wider gfx-rs ecosystem

`wgpu` is one piece of the larger gfx-rs ecosystem. For a map of all the
components and how they fit together, see the big picture:

![The gfx-rs big picture][big-picture.webp]

*/

use crate::Backend;
