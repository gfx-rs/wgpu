# Change Log

<!--
Please add your PR to the changelog! Choose from a top level and bottom
level category, then write your changes like follows:

- Describe your change in a user friendly format by @yourslug in [#9999](https://github.com/gfx-rs/wgpu/pull/2488)

You can add additional user facing information if it's a major breaking change. You can use the following to help:

```diff
- Old code
+ New code
```

Top level categories:
- Major changes
- Added/New Features
- Changes
- Bug Fixes (that don't change API)
- Performance
- Documentation
- Dependency Updates
- deno-webgpu
- Examples
- Testing/Internal

Bottom level categories:

- General
- DX12
- Vulkan
- Metal
- DX11
- GLES
- WebGPU
- Emscripten
- Hal
-->

## Unreleased

### Major Changes

#### Backend selection by features

Whereas `wgpu-core` used to automatically select backends to enable
based on the target OS and architecture, it now has separate features
to enable each backend:

- "metal", for the Metal API on macOS and iOS
- "vulkan", for the Vulkan API (Linux, some Android, and occasionally Windows)
- "dx12", for Microsoft's Direct3D 12 API
- "gles", OpenGL ES, available on many systems
- "dx11", for Microsoft's Direct3D 11 API

None are enabled by default, but the `wgpu` crate automatically
selects these features based on the target operating system and
architecture, using the same rules that `wgpu-core` used to, so users
of `wgpu` should be unaffected by this change. However, other crates
using `wgpu-core` directly will need to copy `wgpu`'s logic or write
their own. See the `[target]` section of `wgpu/Cargo.toml` for
details.

Similarly, `wgpu-core` now has `emscripten` and `renderdoc` features
that `wgpu` enables on appropriate platforms.

In previous releases, the `wgpu-core` crate decided which backends to
support. However, this left `wgpu-core`'s users with no way to
override those choices. (Firefox doesn't want the GLES back end, for
example.) There doesn't seem to be any way to have a crate select
backends based on target OS and architecture that users of that crate
can still override. Default features can't be selected based on the
target, for example. That implies that we should do the selection as
late in the dependency DAG as feasible. Having `wgpu` (and
`wgpu-core`'s other dependents) choose backends seems like the best
option.

By @jimblandy in [#3254](https://github.com/gfx-rs/wgpu/pull/3254).

#### Surface Capabilities API

The various surface capability functions were combined into a single call that gives you all the capabilities.

```diff
- let formats = surface.get_supported_formats(&adapter);
- let present_modes = surface.get_supported_present_modes(&adapter);
- let alpha_modes = surface.get_supported_alpha_modes(&adapter);
+ let caps = surface.get_capabilities(&adapter);
+ let formats = caps.formats;
+ let present_modes = caps.present_modes;
+ let alpha_modes = caps.alpha_modes;
```

Additionally `Surface::get_default_config` now returns an Option and returns None if the surface isn't supported by the adapter.

```diff
- let config = surface.get_default_config(&adapter);
+ let config = surface.get_default_config(&adapter).expect("Surface unsupported by adapter");
```

#### Fallible surface creation

`Instance::create_surface()` now returns `Result<Surface, CreateSurfaceError>` instead of `Surface`. This allows an error to be returned instead of panicking if the given window is a HTML canvas and obtaining a WebGPU or WebGL 2 context fails. (No other platforms currently report any errors through this path.) By @kpreid in [#3052](https://github.com/gfx-rs/wgpu/pull/3052/)

#### Suballocate DX12 buffers and textures

`wgpu`'s DX12 backend can now suballocate buffers and textures when the `windows_rs` feature is enabled, which can give a significant increase in performance (in testing I've seen a 10000%+ improvement in a simple scene with 200 `write_buffer` calls per frame, and a 40%+ improvement in [Bistro using Bevy](https://github.com/vleue/bevy_bistro_playground)). Previously `wgpu-hal`'s DX12 backend created a new heap on the GPU every time you called write_buffer (by calling `CreateCommittedResource`), whereas now with the `windows_rs` feature enabled it uses [`gpu_allocator`](https://crates.io/crates/gpu-allocator) to manage GPU memory (and calls `CreatePlacedResource` with a suballocated heap). By @Elabajaba in [#3163](https://github.com/gfx-rs/wgpu/pull/3163)

### Changes

#### General

- Convert all `Default` Implementations on Enums to `derive(Default)`
- Implement `Default` for `CompositeAlphaMode`
- Improve compute shader validation error message. By @haraldreingruber in [#3139](https://github.com/gfx-rs/wgpu/pull/3139)
- New downlevel feature `UNRESTRICTED_INDEX_BUFFER` to indicate support for using `INDEX` together with other non-copy/map usages (unsupported on WebGL). By @Wumpf in [#3157](https://github.com/gfx-rs/wgpu/pull/3157)
- Combine `Surface::get_supported_formats`, `Surface::get_supported_present_modes`, and `Surface::get_supported_alpha_modes` into `Surface::get_capabilities` and `SurfaceCapabilities`. By @cwfitzgerald in [#3157](https://github.com/gfx-rs/wgpu/pull/3157)
- Make `Surface::get_default_config` return an Option to prevent panics. By @cwfitzgerald in [#3157](https://github.com/gfx-rs/wgpu/pull/3157)
- Lower the `max_buffer_size` limit value for compatibility with Apple2 and WebGPU compliance. By @jinleili in [#3255](https://github.com/gfx-rs/wgpu/pull/3255)
- Limits `min_uniform_buffer_offset_alignment` and `min_storage_buffer_offset_alignment` is now always at least 32. By @wumpf [#3262](https://github.com/gfx-rs/wgpu/pull/3262)
- Dereferencing a buffer view is now marked inline. By @Wumpf in [#3307](https://github.com/gfx-rs/wgpu/pull/3307)
- The `strict_assert` family of macros was moved to `wgpu-types`. By @i509VCB in [#3051](https://github.com/gfx-rs/wgpu/pull/3051)
- Add missing `DEPTH_BIAS_CLAMP` and `FULL_DRAW_INDEX_UINT32` downlevel flags. By @teoxoy in [#3316](https://github.com/gfx-rs/wgpu/pull/3316)
- Make `ObjectId` structure and invariants idiomatic. By @teoxoy in [#3347](https://github.com/gfx-rs/wgpu/pull/3347)
- Add validation in accordance with WebGPU `GPUSamplerDescriptor` valid usage for `lodMinClamp` and `lodMaxClamp`. By @James2022-rgb in [#3353](https://github.com/gfx-rs/wgpu/pull/3353)

#### WebGPU

- Implement `queue_validate_write_buffer` by @jinleili in [#3098](https://github.com/gfx-rs/wgpu/pull/3098)
- Sync depth/stencil copy restrictions with the spec by @teoxoy in [#3314](https://github.com/gfx-rs/wgpu/pull/3314)

#### GLES

- Browsers that support `OVR_multiview2` now report the `MULTIVIEW` feature by @expenses in [#3121](https://github.com/gfx-rs/wgpu/pull/3121).
- `Limits::max_push_constant_size` on GLES is now 256 by @Dinnerbone in [#3374](https://github.com/gfx-rs/wgpu/pull/3374).

#### Vulkan

- Set `WEBGPU_TEXTURE_FORMAT_SUPPORT` downlevel flag depending on the proper format support by @teoxoy in [#3367](https://github.com/gfx-rs/wgpu/pull/3367).
- Set `COPY_SRC`/`COPY_DST` only based on Vulkan's `TRANSFER_SRC`/`TRANSFER_DST` by @teoxoy in [#3366](https://github.com/gfx-rs/wgpu/pull/3366).

### Added/New Features

#### General

- Implement `Hash` for `DepthStencilState` and `DepthBiasState`
- Add the `"wgsl"` feature, to enable WGSL shaders in `wgpu-core` and `wgpu`. Enabled by default in `wgpu`. By @daxpedda in [#2890](https://github.com/gfx-rs/wgpu/pull/2890).
- Implement `Clone` for `ShaderSource` and `ShaderModuleDescriptor` in `wgpu`. By @daxpedda in [#3086](https://github.com/gfx-rs/wgpu/pull/3086).
- Add `get_default_config` for `Surface` to simplify user creation of `SurfaceConfiguration`. By @jinleili in [#3034](https://github.com/gfx-rs/wgpu/pull/3034)
- Native adapters can now use MSAA x2 and x8 if it's supported , previously only x1 and x4 were supported . By @39ali in [3140](https://github.com/gfx-rs/wgpu/pull/3140)
- Implemented correleation between user timestamps and platform specific presentation timestamps via [`Adapter::get_presentation_timestamp`]. By @cwfitzgerald in [#3240](https://github.com/gfx-rs/wgpu/pull/3240)
- Added support for `Features::SHADER_PRIMITIVE_INDEX` on all backends. By @cwfitzgerald in [#3272](https://github.com/gfx-rs/wgpu/pull/3272)
- Implemented `TextureFormat::Stencil8`, allowing for stencil testing without depth components. By @Dinnerbone in [#3343](https://github.com/gfx-rs/wgpu/pull/3343)

#### GLES

- Surfaces support now `TextureFormat::Rgba8Unorm` and (non-web only) `TextureFormat::Bgra8Unorm`. By @Wumpf in [#3070](https://github.com/gfx-rs/wgpu/pull/3070)
- Support alpha to coverage. By @Wumpf in [#3156](https://github.com/gfx-rs/wgpu/pull/3156)
- Support filtering f32 textures. By @expenses in [#3261](https://github.com/gfx-rs/wgpu/pull/3261)

#### WebGPU

- Add `MULTISAMPLE_X2`, `MULTISAMPLE_X4` and `MULTISAMPLE_X8` to `TextureFormatFeatureFlags`. By @39ali in [3140](https://github.com/gfx-rs/wgpu/pull/3140)
- Sync `TextureFormat.describe` with the spec. By @teoxoy in [3312](https://github.com/gfx-rs/wgpu/pull/3312)

### Bug Fixes

#### General

- Update ndk-sys to v0.4.1+23.1.7779620, to fix checksum failures. By @jimblandy in [#3232](https://github.com/gfx-rs/wgpu/pull/3232).
- Bother to free the `hal::Api::CommandBuffer` when a `wgpu_core::command::CommandEncoder` is dropped. By @jimblandy in [#3069](https://github.com/gfx-rs/wgpu/pull/3069).
- Fixed the mipmap example by adding the missing WRITE_TIMESTAMP_INSIDE_PASSES feature. By @Olaroll in [#3081](https://github.com/gfx-rs/wgpu/pull/3081).
- Avoid panicking in some interactions with invalid resources by @nical in (#3094)[https://github.com/gfx-rs/wgpu/pull/3094]
- Fixed an integer overflow in `copy_texture_to_texture` by @nical [#3090](https://github.com/gfx-rs/wgpu/pull/3090)
- Remove `wgpu_types::Features::DEPTH24PLUS_STENCIL8`, making `wgpu::TextureFormat::Depth24PlusStencil8` available on all backends. By @Healthire in (#3151)[https://github.com/gfx-rs/wgpu/pull/3151]
- Fix an integer overflow in `queue_write_texture` by @nical in (#3146)[https://github.com/gfx-rs/wgpu/pull/3146]
- Make `RenderPassCompatibilityError` and `CreateShaderModuleError` not so huge. By @jimblandy in (#3226)[https://github.com/gfx-rs/wgpu/pull/3226]
- Check for invalid bitflag bits in wgpu-core and allow them to be captured/replayed by @nical in (#3229)[https://github.com/gfx-rs/wgpu/pull/3229]
- Evaluate `gfx_select!`'s `#[cfg]` conditions at the right time. By @jimblandy in [#3253](https://github.com/gfx-rs/wgpu/pull/3253)
- Improve error messages when binding bind group with dynamic offsets. By @cwfitzgerald in [#3294](https://github.com/gfx-rs/wgpu/pull/3294)
- Allow non-filtering sampling of integer textures. By @JMS55 in [#3362](https://github.com/gfx-rs/wgpu/pull/3362).

#### Metal
- Fix texture view creation with full-resource views when using an explicit `mip_level_count` or `array_layer_count`. By @cwfitzgerald in [#3323](https://github.com/gfx-rs/wgpu/pull/3323)

#### WebGPU

- Use `log` instead of `println` in hello example by @JolifantoBambla in [#2858](https://github.com/gfx-rs/wgpu/pull/2858)

#### GLES

- Fixed WebGL not displaying srgb targets correctly if a non-screen filling viewport was previously set. By @Wumpf in [#3093](https://github.com/gfx-rs/wgpu/pull/3093)
- Fix disallowing multisampling for float textures if otherwise supported. By @Wumpf in [#3183](https://github.com/gfx-rs/wgpu/pull/3183)
- Fix a panic when creating a pipeline with opaque types other than samplers (images and atomic counters). By @James2022-rgb in [#3361](https://github.com/gfx-rs/wgpu/pull/3361)

#### Vulkan

- Document and improve extension detection. By @teoxoy in [#3327](https://github.com/gfx-rs/wgpu/pull/3327)

#### deno-webgpu

- Let `setVertexBuffer` and `setIndexBuffer` calls on
  `GPURenderBundleEncoder` throw an error if the `size` argument is
  zero, rather than treating that as "until the end of the buffer".
  By @jimblandy in [#3171](https://github.com/gfx-rs/wgpu/pull/3171)

#### Emscripten

- Let the wgpu examples `framework.rs` compile again under Emscripten. By @jimblandy in [#3246](https://github.com/gfx-rs/wgpu/pull/3246)

#### Vulkan

- Update ash to 0.37.1+1.3.235 to fix CI breaking by changing a call to the deprecated `debug_utils_set_object_name()` function to `set_debug_utils_object_name()` by @elabajaba in [#3273](https://github.com/gfx-rs/wgpu/pull/3273)

#### DX12

- Fix `depth16Unorm` formats by @teoxoy in [#3313](https://github.com/gfx-rs/wgpu/pull/3313)
- Don't re-use `GraphicsCommandList` when `close` or `reset` fails. By @xiaopengli89 in [#3204](https://github.com/gfx-rs/wgpu/pull/3204)

### Examples

- Log adapter info in hello example on wasm target by @JolifantoBambla in [#2858](https://github.com/gfx-rs/wgpu/pull/2858)
- Added new example `stencil-triangles` to show basic use of stencil testing. By @Dinnerbone in [#3343](https://github.com/gfx-rs/wgpu/pull/3343)

### Testing/Internal

- Update the `minimum supported rust version` to 1.64
- Move `ResourceMetadata` into its own module. By @jimblandy in [#3213](https://github.com/gfx-rs/wgpu/pull/3213)
- Add WebAssembly testing infrastructure. By @haraldreingruber in [#3238](https://github.com/gfx-rs/wgpu/pull/3238)
- Error message when you forget to use cargo-nextest. By @cwfitzgerald in [#3293](https://github.com/gfx-rs/wgpu/pull/3293)

#### Vulkan

- Don't use a pointer to a local copy of a `PhysicalDeviceDriverProperties` struct after it has gone out of scope. In fact, don't make a local copy at all. Introduce a helper function for building `CStr`s from C character arrays, and remove some `unsafe` blocks. By @jimblandy in [#3076](https://github.com/gfx-rs/wgpu/pull/3076).


## wgpu-0.14.2 (2022-11-28)

### Bug Fixes

- Fix incorrect offset in `get_mapped_range` by @nical in [#3233](https://github.com/gfx-rs/wgpu/pull/3233)


## wgpu-0.14.1 (2022-11-02)

### Bug Fixes

- Make `wgpu::TextureFormat::Depth24PlusStencil8` available on all backends by making the feature unconditionally available and the feature unneeded to use the format. By @Healthire and @cwfitzgerald in [#3165](https://github.com/gfx-rs/wgpu/pull/3165)

## wgpu-0.14.0 (2022-10-05)

### Major Changes

#### @invariant Warning

When using CompareFunction::Equal or CompareFunction::NotEqual on a pipeline, there is now a warning logged if the vertex
shader does not have a @invariant tag on it. On some machines, rendering the same triangles multiple times without an
@invariant tag will result in slightly different depths for every pixel. Because the \*Equal functions rely on depth being
the same every time it is rendered, we now warn if it is missing.

```diff
-@vertex
-fn vert_main(v_in: VertexInput) -> @builtin(position) vec4<f32> {...}
+@vertex
+fn vert_main(v_in: VertexInput) -> @builtin(position) @invariant vec4<f32> {...}
```

#### Surface Alpha and PresentModes

Surface supports `alpha_mode` now. When alpha_mode is equal to `PreMultiplied` or `PostMultiplied`,
the alpha channel of framebuffer is respected in the compositing process, but which mode is available depends on
the different API and `Device`. If don't care about alpha_mode, you can set it to `Auto`.

```diff
SurfaceConfiguration {
// ...
+ alpha_mode: surface.get_supported_alpha_modes(&adapter)[0],
}
```

The function to enumerate supported presentation modes changed:

```diff
- pub fn wgpu::Surface::get_supported_modes(&self, adapter: &wgpu::Adapter) -> Vec<PresentMode>
+ pub fn wgpu::Surface::get_supported_present_modes(&self, adapter: &wgpu::Adapter) -> Vec<PresentMode>
```

#### Updated raw-window-handle to 0.5

This will allow use of the latest version of winit. As such the bound on create_surface is now RWH 0.5 and requires
both `raw_window_handle::HasRawWindowHandle` and `raw_window_handle::HasRawDisplayHandle`.

### Added/New Features

- Add `Buffer::size()` and `Buffer::usage()`; by @kpreid in [#2923](https://github.com/gfx-rs/wgpu/pull/2923)
- Split Blendability and Filterability into Two Different TextureFormatFeatureFlags; by @stakka in [#3012](https://github.com/gfx-rs/wgpu/pull/3012)
- Expose `alpha_mode` on SurfaceConfiguration, by @jinleili in [#2836](https://github.com/gfx-rs/wgpu/pull/2836)
- Introduce fields for driver name and info in `AdapterInfo`, by @i509VCB in [#3037](https://github.com/gfx-rs/wgpu/pull/3037)
- Add way to create gles hal textures from raw gl names to allow externally managed textures. By @i509VCB [#3046](https://github.com/gfx-rs/wgpu/pull/3046)
- Implemented `copy_external_image_to_texture` on WebGPU, by @ybiletskyi in [#2781](https://github.com/gfx-rs/wgpu/pull/2781)

### Bug Fixes

#### General

- Free `StagingBuffers` even when an error occurs in the operation that consumes them. By @jimblandy in [#2961](https://github.com/gfx-rs/wgpu/pull/2961)
- Avoid overflow when checking that texture copies fall within bounds. By @jimblandy in [#2963](https://github.com/gfx-rs/wgpu/pull/2963)
- Improve the validation and error reporting of buffer mappings by @nical in [#2848](https://github.com/gfx-rs/wgpu/pull/2848)
- Fix compilation errors when using wgpu-core in isolation while targetting `wasm32-unknown-unknown` by @Seamooo in [#2922](https://github.com/gfx-rs/wgpu/pull/2922)
- Fixed opening of RenderDoc library by @abuffseagull in [#2930](https://github.com/gfx-rs/wgpu/pull/2930)
- Added missing validation for `BufferUsages` mismatches when `Features::MAPPABLE_PRIMARY_BUFFERS` is not
  enabled. By @imberflur in [#3023](https://github.com/gfx-rs/wgpu/pull/3023)
- Fixed `CommandEncoder` not being `Send` and `Sync` on web by @i509VCB in [#3025](https://github.com/gfx-rs/wgpu/pull/3025)
- Document meaning of `vendor` in `AdapterInfo` if the vendor has no PCI id.
- Fix missing resource labels from some Errors by @scoopr in [#3066](https://github.com/gfx-rs/wgpu/pull/3066)

#### Metal

- Add the missing `msg_send![view, retain]` call within `from_view` by @jinleili in [#2976](https://github.com/gfx-rs/wgpu/pull/2976)
- Fix `max_buffer` `max_texture` and `max_vertex_buffers` limits by @jinleili in [#2978](https://github.com/gfx-rs/wgpu/pull/2978)
- Remove PrivateCapabilities's `format_rgb10a2_unorm_surface` field by @jinleili in [#2981](https://github.com/gfx-rs/wgpu/pull/2981)
- Fix validation error when copying into a subset of a single-layer texture by @nical in [#3063](https://github.com/gfx-rs/wgpu/pull/3063)
- Fix `_buffer_sizes` encoding by @dtiselice in [#3047](https://github.com/gfx-rs/wgpu/pull/3047)

#### Vulkan

- Fix `astc_hdr` formats support by @jinleili in [#2971]](https://github.com/gfx-rs/wgpu/pull/2971)
- Update to Naga b209d911 (2022-9-1) to avoid generating SPIR-V that
  violates Vulkan valid usage rules `VUID-StandaloneSpirv-Flat-06202`
  and `VUID-StandaloneSpirv-Flat-04744`. By @jimblandy in
  [#3008](https://github.com/gfx-rs/wgpu/pull/3008)
- Fix bug where the Vulkan backend would panic when using a supported window and display handle but the
  dependent extensions are not available by @i509VCB in [#3054](https://github.com/gfx-rs/wgpu/pull/3054).

#### GLES

- Report vendor id for Mesa and Apple GPUs. By @i509VCB [#3036](https://github.com/gfx-rs/wgpu/pull/3036)
- Report Apple M2 gpu as integrated. By @i509VCB [#3036](https://github.com/gfx-rs/wgpu/pull/3036)

#### WebGPU

- When called in a web worker, `Context::init()` now uses `web_sys::WorkerGlobalContext` to create a `wgpu::Instance` instead of trying to access the unavailable `web_sys::Window` by @JolifantoBambla in [#2858](https://github.com/gfx-rs/wgpu/pull/2858)

### Changes

#### General

- Changed wgpu-hal and wgpu-core implementation to pass RawDisplayHandle and RawWindowHandle as separate
  parameters instead of passing an impl trait over both HasRawDisplayHandle and HasRawWindowHandle. By @i509VCB in [#3022](https://github.com/gfx-rs/wgpu/pull/3022)
- Changed `Instance::as_hal<A>` to just return an `Option<&A::Instance>` rather than taking a callback. By @jimb in [#2991](https://github.com/gfx-rs/wgpu/pull/2991)
- Added downlevel restriction error message for `InvalidFormatUsages` error by @Seamooo in [#2886](https://github.com/gfx-rs/wgpu/pull/2886)
- Add warning when using CompareFunction::\*Equal with vertex shader that is missing @invariant tag by @cwfitzgerald in [#2887](https://github.com/gfx-rs/wgpu/pull/2887)
- Update Winit to version 0.27 and raw-window-handle to 0.5 by @wyatt-herkamp in [#2918](https://github.com/gfx-rs/wgpu/pull/2918)
- Address Clippy 0.1.63 complaints. By @jimblandy in [#2977](https://github.com/gfx-rs/wgpu/pull/2977)
- Don't use `PhantomData` for `IdentityManager`'s `Input` type. By @jimblandy in [#2972](https://github.com/gfx-rs/wgpu/pull/2972)
- Changed Naga variant in ShaderSource to `Cow<'static, Module>`, to allow loading global variables by @daxpedda in [#2903](https://github.com/gfx-rs/wgpu/pull/2903)
- Updated the maximum binding index to match the WebGPU specification by @nical in [#2957](https://github.com/gfx-rs/wgpu/pull/2957)
- Add `unsafe_op_in_unsafe_fn` to Clippy lints in the entire workspace. By @ErichDonGubler in [#3044](https://github.com/gfx-rs/wgpu/pull/3044).

#### Metal

- Extract the generic code into `get_metal_layer` by @jinleili in [#2826](https://github.com/gfx-rs/wgpu/pull/2826)

#### Vulkan

- Remove use of Vulkan12Features/Properties types. By @i509VCB in [#2936](https://github.com/gfx-rs/wgpu/pull/2936)
- Provide a means for `wgpu` users to access `vk::Queue` and the queue index. By @anlumo in [#2950](https://github.com/gfx-rs/wgpu/pull/2950)
- Use the use effective api version for determining device features instead of wrongly assuming `VkPhysicalDeviceProperties.apiVersion`
  is the actual version of the device. By @i509VCB in [#3011](https://github.com/gfx-rs/wgpu/pull/3011)
- `DropGuard` has been moved to the root of the wgpu-hal crate. By @i509VCB [#3046](https://github.com/gfx-rs/wgpu/pull/3046)

#### GLES

- Add `Rgba16Float` format support for color attachments. By @jinleili in [#3045](https://github.com/gfx-rs/wgpu/pull/3045)
- `TEXTURE_COMPRESSION_ASTC_HDR` feature detection by @jinleili in [#3042](https://github.com/gfx-rs/wgpu/pull/3042)

### Performance

- Made `StagingBelt::write_buffer()` check more thoroughly for reusable memory; by @kpreid in [#2906](https://github.com/gfx-rs/wgpu/pull/2906)

### Documentation

- Add WGSL examples to complement existing examples written in GLSL by @norepimorphism in [#2888](https://github.com/gfx-rs/wgpu/pull/2888)
- Document `wgpu_core` resource allocation. @jimblandy in [#2973](https://github.com/gfx-rs/wgpu/pull/2973)
- Expanded `StagingBelt` documentation by @kpreid in [#2905](https://github.com/gfx-rs/wgpu/pull/2905)
- Fixed documentation for `Instance::create_surface_from_canvas` and
  `Instance::create_surface_from_offscreen_canvas` regarding their
  safety contract. These functions are not unsafe. By @jimblandy [#2990](https://github.com/gfx-rs/wgpu/pull/2990)
- Document that `write_buffer_with()` is sound but unwise to read from by @kpreid in [#3006](https://github.com/gfx-rs/wgpu/pull/3006)
- Explain why `Adapter::as_hal` and `Device::as_hal` have to take callback functions. By @jimblandy in [#2992](https://github.com/gfx-rs/wgpu/pull/2992)

### Dependency Updates

#### WebGPU

- Update wasm32 dependencies, set `alpha_mode` on web target by @jinleili in [#3040](https://github.com/gfx-rs/wgpu/pull/3040)

### Build Configuration

- Add the `"strict_asserts"` feature, to enable additional internal
  run-time validation in `wgpu-core`. By @jimblandy in
  [#2872](https://github.com/gfx-rs/wgpu/pull/2872).

### Full API Diff

Manual concatination of `cargo public-api --diff-git-checkouts v0.13.2 v0.14.0 -p wgpu` and `cargo public-api --diff-git-checkouts v0.13.2 v0.14.0 -p wgpu-types`

```diff
Removed items from the public API
=================================
-pub fn wgpu::Surface::get_supported_modes(&self, adapter: &wgpu::Adapter) -> Vec<PresentMode>
-pub const wgpu::Features::DEPTH24UNORM_STENCIL8: Self
-pub enum variant wgpu::TextureFormat::Depth24UnormStencil8

Changed items in the public API
===============================
-pub unsafe fn wgpu::Instance::as_hal<A: wgc::hub::HalApi, F: FnOnce(Option<&<A as >::Instance>) -> R, R>(&self, hal_instance_callback: F) -> R
+pub unsafe fn wgpu::Instance::as_hal<A: wgc::hub::HalApi>(&self) -> Option<&<A as >::Instance>
-pub unsafe fn wgpu::Instance::create_surface<W: raw_window_handle::HasRawWindowHandle>(&self, window: &W) -> wgpu::Surface
+pub unsafe fn wgpu::Instance::create_surface<W: raw_window_handle::HasRawWindowHandle + raw_window_handle::HasRawDisplayHandle>(&self, window: &W) -> wgpu::Surface

Added items to the public API
=============================
+pub fn wgpu::Buffer::size(&self) -> wgt::BufferAddress
+pub fn wgpu::Buffer::usage(&self) -> BufferUsages
+pub fn wgpu::Surface::get_supported_alpha_modes(&self, adapter: &wgpu::Adapter) -> Vec<CompositeAlphaMode>
+pub fn wgpu::Surface::get_supported_present_modes(&self, adapter: &wgpu::Adapter) -> Vec<PresentMode>
+#[repr(C)] pub enum wgpu::CompositeAlphaMode
+impl RefUnwindSafe for wgpu::CompositeAlphaMode
+impl Send for wgpu::CompositeAlphaMode
+impl Sync for wgpu::CompositeAlphaMode
+impl Unpin for wgpu::CompositeAlphaMode
+impl UnwindSafe for wgpu::CompositeAlphaMode
+pub const wgpu::Features::DEPTH24PLUS_STENCIL8: Self
+pub const wgpu::TextureFormatFeatureFlags::BLENDABLE: Self
+pub enum variant wgpu::CompositeAlphaMode::Auto = 0
+pub enum variant wgpu::CompositeAlphaMode::Inherit = 4
+pub enum variant wgpu::CompositeAlphaMode::Opaque = 1
+pub enum variant wgpu::CompositeAlphaMode::PostMultiplied = 3
+pub enum variant wgpu::CompositeAlphaMode::PreMultiplied = 2
+pub enum variant wgpu::TextureFormat::Depth16Unorm
+pub fn wgpu::CompositeAlphaMode::clone(&self) -> wgpu::CompositeAlphaMode
+pub fn wgpu::CompositeAlphaMode::eq(&self, other: &wgpu::CompositeAlphaMode) -> bool
+pub fn wgpu::CompositeAlphaMode::fmt(&self, f: &mut $crate::fmt::Formatter<'_>) -> $crate::fmt::Result
+pub fn wgpu::CompositeAlphaMode::hash<__H: $crate::hash::Hasher>(&self, state: &mut __H) -> ()
+pub struct field wgpu::AdapterInfo::driver: String
+pub struct field wgpu::AdapterInfo::driver_info: String
+pub struct field wgpu::SurfaceConfiguration::alpha_mode: wgpu_types::CompositeAlphaMode
```

## wgpu-0.13.2 (2022-07-13)

### Bug Fixes

#### General

- Prefer `DeviceType::DiscreteGpu` over `DeviceType::Other` for `PowerPreference::LowPower` so Vulkan is preferred over OpenGL again by @Craig-Macomber in [#2853](https://github.com/gfx-rs/wgpu/pull/2853)
- Allow running `get_texture_format_features` on unsupported texture formats (returning no flags) by @cwfitzgerald in [#2856](https://github.com/gfx-rs/wgpu/pull/2856)
- Allow multi-sampled textures that are supported by the device but not WebGPU if `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` is enabled by @cwfitzgerald in [#2856](https://github.com/gfx-rs/wgpu/pull/2856)
- `get_texture_format_features` only lists the COPY\_\* usages if the adapter actually supports that usage by @cwfitzgerald in [#2856](https://github.com/gfx-rs/wgpu/pull/2856)
- Fix bind group / pipeline deduplication not taking into account RenderBundle execution resetting these values by @shoebe [#2867](https://github.com/gfx-rs/wgpu/pull/2867)
- Fix panics that occur when using `as_hal` functions when the hal generic type does not match the hub being looked up in by @i509VCB [#2871](https://github.com/gfx-rs/wgpu/pull/2871)
- Add some validation in map_async by @nical in [#2876](https://github.com/gfx-rs/wgpu/pull/2876)
- Fix bugs when mapping/unmapping zero-sized buffers and ranges by @nical in [#2877](https://github.com/gfx-rs/wgpu/pull/2877)
- Fix out-of-bound write in `map_buffer` with non-zero offset by @nical in [#2916](https://github.com/gfx-rs/wgpu/pull/2916)
- Validate the number of color attachments in `create_render_pipeline` by @nical in [#2913](https://github.com/gfx-rs/wgpu/pull/2913)
- Validate against the maximum binding index in `create_bind_group_layout` by @nical in [#2892](https://github.com/gfx-rs/wgpu/pull/2892)
- Validate that map_async's range is not negative by @nical in [#2938](https://github.com/gfx-rs/wgpu/pull/2938)
- Fix calculation/validation of layer/mip ranges in create_texture_view by @nical in [#2955](https://github.com/gfx-rs/wgpu/pull/2955)
- Validate the sample count and mip level in `copy_texture_to_buffer` by @nical in [#2958](https://github.com/gfx-rs/wgpu/pull/2958)
- Expose the cause of the error in the `map_async` callback in [#2939](https://github.com/gfx-rs/wgpu/pull/2939)

#### DX12

- `DownlevelCapabilities::default()` now returns the `ANISOTROPIC_FILTERING` flag set to true so DX12 lists `ANISOTROPIC_FILTERING` as true again by @cwfitzgerald in [#2851](https://github.com/gfx-rs/wgpu/pull/2851)
- Properly query format features for UAV/SRV usages of depth formats by @cwfitzgerald in [#2856](https://github.com/gfx-rs/wgpu/pull/2856)

#### Vulkan

- Vulkan 1.0 drivers that support `VK_KHR_multiview` now properly report the `MULTIVIEW` feature as supported by @i509VCB in [#2934](https://github.com/gfx-rs/wgpu/pull/2934).
- Stop using `VkPhysicalDevice11Features` in Vulkan 1.1 which is confusingly provided in Vulkan 1.2 by @i509VCB in [#2934](https://github.com/gfx-rs/wgpu/pull/2934).

#### GLES

- Fix depth stencil texture format capability by @jinleili in [#2854](https://github.com/gfx-rs/wgpu/pull/2854)
- `get_texture_format_features` now only returns usages for formats it actually supports by @cwfitzgerald in [#2856](https://github.com/gfx-rs/wgpu/pull/2856)

#### Hal

- Allow access to queue family index in Vulkan hal by @i509VCB in [#2859](https://github.com/gfx-rs/wgpu/pull/2859)
- Allow access to the EGLDisplay and EGLContext pointer in Gles hal Adapter and Device by @i509VCB in [#2860](https://github.com/gfx-rs/wgpu/pull/2860)

### Documentation

- Update present_mode docs as most of them don't automatically fall back to Fifo anymore. by @Elabajaba in [#2855](https://github.com/gfx-rs/wgpu/pull/2855)

#### Hal

- Document safety requirements for `Adapter::from_external` in gles hal by @i509VCB in [#2863](https://github.com/gfx-rs/wgpu/pull/2863)
- Make `AdapterContext` a publicly accessible type in the gles hal by @i509VCB in [#2870](https://github.com/gfx-rs/wgpu/pull/2870)

## wgpu-0.13.1 (2022-07-02)

### Bug Fixes

#### General

- Fix out of bounds access when surface texture is written to by multiple command buffers by @cwfitzgerald in [#2843](https://github.com/gfx-rs/wgpu/pull/2843)

#### GLES

- AutoNoVSync now correctly falls back to Fifo by @simbleau in [#2842](https://github.com/gfx-rs/wgpu/pull/2842)
- Fix GL_EXT_color_buffer_float detection on native by @cwfitzgerald in [#2843](https://github.com/gfx-rs/wgpu/pull/2843)

## wgpu-0.13 (2022-06-30)

### Major Changes

#### WGSL Syntax

WGSL syntax has changed in a couple ways. The new syntax is easier to read and work with.

Attribute declarations are written differently:

```diff
- [[group(1), binding(0)]]
+ @group(1) @binding(0)
```

Stage declarations are now separate attributes rather than part of the `stage` attribute:

```diff
- [[stage(vertex)]]
+ @vertex
```

Structs now use `,` as field separator and no longer need semicolons after the declaration:

```diff
- struct MyStruct {
-     my_member: u32;
- };
+ struct MyStruct {
+     my_member: u32,
+ }
```

#### Surface API

The method of getting the preferred swapchain format has changed to allow viewing all formats supported by the surface.

```diff
- let format = surface.get_preferred_format(&adapter).unwrap();
+ let format = surface.get_supported_formats(&adapter)[0];
```

Presentation modes now need to match exactly what the surface supports. `FIFO` is _always_ supported,
but all other modes vary from API to API and `Device` to `Device`. To get a list of all supported modes,
call the following. The order does not indicate preference.

```rust
let modes = surface.get_supported_present_modes(&adapter);
```

#### Timestamp Queries

Timestamp queries are now restricted behind multiple features to allow implementation on TBDR (Tile-Based Deferred Rendering)
based GPUs, such as mobile devices and Apple's M chips.

`Features::TIMESTAMP_QUERIES` now allows for calling `write_timestamp` only on `CommandEncoder`s.

`Features::WRITE_TIMESTAMP_INSIDE_PASSES` is needed to call `write_timestamp` on `RenderPassEncoder`s or `ComputePassEncoder`s.

#### map_async

The function for mapping buffers no longer returns a future, and instead calls a callback when the buffer is mapped.

This aligns with the use of the API more clearly - you aren't supposed to block and wait on the future to resolve,
you are supposed to keep rendering and wait until the buffer maps on its own. Mapping and the flow of mapping
is an under-documented area that we hope to improve in the future.

```diff
- let future = buffer.slice(..).map_async(MapMode::Read);
+ buffer.slice(..).map_async(MapMode::Read, || {
+     // Called when buffer is mapped.
+ })
```

#### Submission Indexes

Calling `queue.submit` now returns an opaque submission index that can be used as an argument to
`device.poll` to say which submission to wait to complete.

### Other Breaking Changes

`Device::create_shader_module` now takes the shader descriptor by value:

```diff
- device.create_shader_module(&shader_module_descriptor)
+ device.create_shader_module(shader_module_descriptor)
```

Color attachments can be sparse, so they are now optional:

```diff
FragmentState {
-  targets: &[color_target_state]
+  targets: &[Some(color_target_state)]
  // ..
}
```

```diff
RenderPassDescriptor {
-  color_attachments: &[render_pass_color_attachment]
+  color_attachments: &[Some(render_pass_color_attachment)]
  // ..
}
```

```diff
RenderBundleEncoderDescriptor {
-  color_formats: &[texture_format]
+  color_formats: &[Some(texture_format)]
  // ..
}
```

`Extent3d::max_mips` now requires you to pass a TextureDimension to specify whether or not depth_or_array_layers should be ignored:

```diff
Extent3d {
  width: 1920,
  height: 1080,
  depth_or_array_layers: 6,
- }.max_mips()
+ }.max_mips(wgpu::TextureDimension::D3)
```

`Limits` has a new field, [`max_buffer_size`](https://docs.rs/wgpu/0.13.0/wgpu/struct.Limits.html#structfield.max_buffer_size) (not an issue if you don't define limits manually):

```diff
Limits {
  // ...
+ max_buffer_size: 256 * 1024 * 1024, // adjust as you see fit
}
```

`Features::CLEAR_COMMANDS` is now unnecessary and no longer exists. The feature to clear buffers and textures is now part of upstream WebGPU.

```diff
DeviceDescriptor {
  // ...
  features: wgpu::Features::VERTEX_WRITABLE_STORAGE
    | wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
    | wgpu::Features::TEXTURE_BINDING_ARRAY
    | wgpu::Features::BUFFER_BINDING_ARRAY
    | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
-    | wgpu::Features::CLEAR_COMMANDS
  ,
}
```

`ComputePass::dispatch` has been renamed to `ComputePass::dispatch_workgroups`

```diff
- cpass.dispatch(self.work_group_count, 1, 1)
+ cpass.dispatch_workgroups(self.work_group_count, 1, 1)
```

### Added/New Features

#### General

- Add `util::indirect::*` helper structs by @IcanDivideBy0 in [#2365](https://github.com/gfx-rs/wgpu/pull/2365)
- Add `AddressMode::ClampToZero` by @laptou in [#2364](https://github.com/gfx-rs/wgpu/pull/2364)
- Add MULTISAMPLED_SHADING downlevel flag by @jinleili in [#2425](https://github.com/gfx-rs/wgpu/pull/2425)
- Allow non struct buffers in wgsl by @IcanDivideBy0 in [#2451](https://github.com/gfx-rs/wgpu/pull/2451)
- Prefix every wgpu-generated label with `(wgpu)`. by @kpreid in [#2590](https://github.com/gfx-rs/wgpu/pull/2590)
- Permit non-struct, non-array types as buffers. by @jimblandy in [#2584](https://github.com/gfx-rs/wgpu/pull/2584)
- Return `queue_empty` for Device::poll by @xiaopengli89 in [#2643](https://github.com/gfx-rs/wgpu/pull/2643)
- Add `SHADER_FLOAT16` feature by @jinleili in [#2646](https://github.com/gfx-rs/wgpu/pull/2646)
- Add DEPTH32FLOAT_STENCIL8 featue by @jinleili in [#2664](https://github.com/gfx-rs/wgpu/pull/2664)
- Add DEPTH24UNORM_STENCIL8 feature by @jinleili in [#2689](https://github.com/gfx-rs/wgpu/pull/2689)
- Implement submission indexes by @cwfitzgerald in [#2700](https://github.com/gfx-rs/wgpu/pull/2700)
- [WebGL] Add a downlevel capability for rendering to floating point textures by @expenses in [#2729](https://github.com/gfx-rs/wgpu/pull/2729)
- allow creating wgpu::Instance from wgpu_core::Instance by @i509VCB in [#2763](https://github.com/gfx-rs/wgpu/pull/2763)
- Force binding sizes to be multiples of 16 on webgl by @cwfitzgerald in [#2808](https://github.com/gfx-rs/wgpu/pull/2808)
- Add Naga variant to ShaderSource by @rttad in [#2801](https://github.com/gfx-rs/wgpu/pull/2801)
- Implement Queue::write_buffer_with by @teoxoy in [#2777](https://github.com/gfx-rs/wgpu/pull/2777)

#### Vulkan

- Re-allow vk backend on Apple platforms via `vulkan-portability` feature by @jinleili in [#2488](https://github.com/gfx-rs/wgpu/pull/2488)
- vulkan: HDR ASTC formats support by @jinleili in [#2496](https://github.com/gfx-rs/wgpu/pull/2496)

#### Metal

- Implement push constants for metal backend by @TheOnlyMrCat in [#2314](https://github.com/gfx-rs/wgpu/pull/2314)
- Metal backend ASTC HDR formats support by @jinleili in [#2477](https://github.com/gfx-rs/wgpu/pull/2477)
- Add COPY_DST to Metal's surface usage bits by @vl4dimir in [#2491](https://github.com/gfx-rs/wgpu/pull/2491)
- Add `Features::MULTI_DRAW_INDIRECT` to Metal by @expenses in [#2737](https://github.com/gfx-rs/wgpu/pull/2737)

#### GLES

- Support externally initialized contexts by @kvark in [#2350](https://github.com/gfx-rs/wgpu/pull/2350)
- Angle support on macOS by @jinleili in [#2461](https://github.com/gfx-rs/wgpu/pull/2461)
- Use EGL surfaceless platform when windowing system is not found by @sh7dm in [#2339](https://github.com/gfx-rs/wgpu/pull/2339)
- Do a downlevel check for anisotrophy and enable it in the webgl backend by @expenses in [#2616](https://github.com/gfx-rs/wgpu/pull/2616)
- OffscreenCanvas Support for WebGL Backend by @haraldreingruber-dedalus in [#2603](https://github.com/gfx-rs/wgpu/pull/2603)

#### DX12

- Support to create surface from visual on Windows by @xiaopengli89 in [#2434](https://github.com/gfx-rs/wgpu/pull/2434)
- Add raw_queue for d3d12 device by @xiaopengli89 in [#2600](https://github.com/gfx-rs/wgpu/pull/2600)

#### DX11

- Skeleton of a DX11 backend - not working yet by @cwfitzgerald in [#2443](https://github.com/gfx-rs/wgpu/pull/2443)

#### Hal

- Adapter and Instance as_hal functions by @i509VCB in [#2663](https://github.com/gfx-rs/wgpu/pull/2663)
- expose some underlying types in Vulkan hal by @i509VCB in [#2667](https://github.com/gfx-rs/wgpu/pull/2667)
- Add raw_device method for dx12, vulkan hal by @xiaopengli89 in [#2360](https://github.com/gfx-rs/wgpu/pull/2360)
- expose egl display in gles Instance hal by @i509VCB in [#2670](https://github.com/gfx-rs/wgpu/pull/2670)
- Add raw_adapter method for dx12 hal adapter by @xiaopengli89 in [#2714](https://github.com/gfx-rs/wgpu/pull/2714)
- Acquire texture: `Option<std::time::Duration>` timeouts by @rib in [#2724](https://github.com/gfx-rs/wgpu/pull/2724)
- expose vulkan physical device capabilities, enabled device extensions by @i509VCB in [#2688](https://github.com/gfx-rs/wgpu/pull/2688)

#### Emscripten

- feature: emscripten by @caiiiycuk in [#2422](https://github.com/gfx-rs/wgpu/pull/2422)
- feature = emscripten, compability fixes for wgpu-native by @caiiiycuk in [#2450](https://github.com/gfx-rs/wgpu/pull/2450)

### Changes

#### General

- Make ShaderSource #[non_exhaustive] by @fintelia in [#2312](https://github.com/gfx-rs/wgpu/pull/2312)
- Make `execute_bundles()` receive IntoIterator by @maku693 in [#2410](https://github.com/gfx-rs/wgpu/pull/2410)
- Raise `wgpu_hal::MAX_COLOR_TARGETS` to 8. by @jimblandy in [#2640](https://github.com/gfx-rs/wgpu/pull/2640)
- Rename dispatch -> dispatch_workgroups by @jinleili in [#2619](https://github.com/gfx-rs/wgpu/pull/2619)
- Update texture_create_view logic to match spec by @jinleili in [#2621](https://github.com/gfx-rs/wgpu/pull/2621)
- Move TEXTURE_COMPRESSION_ETC2 | ASTC_LDR to web section to match spec by @jinleili in [#2671](https://github.com/gfx-rs/wgpu/pull/2671)
- Check that all vertex outputs are consumed by the fragment shader by @cwfitzgerald in [#2704](https://github.com/gfx-rs/wgpu/pull/2704)
- Convert map_async from being async to being callback based by @cwfitzgerald in [#2698](https://github.com/gfx-rs/wgpu/pull/2698)
- Align the validation of Device::create_texture with the WebGPU spec by @nical in [#2759](https://github.com/gfx-rs/wgpu/pull/2759)
- Add InvalidGroupIndex validation at create_shader_module by @jinleili in [#2775](https://github.com/gfx-rs/wgpu/pull/2775)
- Rename MAX_COLOR_TARGETS to MAX_COLOR_ATTACHMENTS to match spec by @jinleili in [#2780](https://github.com/gfx-rs/wgpu/pull/2780)
- Change get_preferred_format to get_supported_formats by @stevenhuyn in [#2783](https://github.com/gfx-rs/wgpu/pull/2783)
- Restrict WriteTimestamp Inside Passes by @cwfitzgerald in [#2802](https://github.com/gfx-rs/wgpu/pull/2802)
- Flip span labels to work better with tools by @cwfitzgerald in [#2820](https://github.com/gfx-rs/wgpu/pull/2820)

#### Gles

- Make GLES DeviceType unknown by default by @PolyMeilex in [#2647](https://github.com/gfx-rs/wgpu/pull/2647)

#### Metal

- metal: check if in the main thread when calling `create_surface` by @jinleili in [#2736](https://github.com/gfx-rs/wgpu/pull/2736)

#### Hal

- limit binding sizes to i32 by @kvark in [#2363](https://github.com/gfx-rs/wgpu/pull/2363)

### Bug Fixes

#### General

- Fix trac(y/ing) compile issue by @cwfitzgerald in [#2333](https://github.com/gfx-rs/wgpu/pull/2333)
- Improve detection and validation of cubemap views by @kvark in [#2331](https://github.com/gfx-rs/wgpu/pull/2331)
- Don't create array layer trackers for 3D textures. by @ElectronicRU in [#2348](https://github.com/gfx-rs/wgpu/pull/2348)
- Limit 1D texture mips to 1 by @kvark in [#2374](https://github.com/gfx-rs/wgpu/pull/2374)
- Texture format MSAA capabilities by @kvark in [#2377](https://github.com/gfx-rs/wgpu/pull/2377)
- Fix write_buffer to surface texture @kvark in [#2385](https://github.com/gfx-rs/wgpu/pull/2385)
- Improve some error messages by @cwfitzgerald in [#2446](https://github.com/gfx-rs/wgpu/pull/2446)
- Don't recycle indices that reach EOL by @kvark in [#2462](https://github.com/gfx-rs/wgpu/pull/2462)
- Validated render usages for 3D textures by @kvark in [#2482](https://github.com/gfx-rs/wgpu/pull/2482)
- Wrap all validation logs with catch_unwinds by @cwfitzgerald in [#2511](https://github.com/gfx-rs/wgpu/pull/2511)
- Fix clippy lints by @a1phyr in [#2560](https://github.com/gfx-rs/wgpu/pull/2560)
- Free the raw device when `wgpu::Device` is dropped. by @jimblandy in [#2567](https://github.com/gfx-rs/wgpu/pull/2567)
- wgpu-core: Register new pipelines with device's tracker. by @jimblandy in [#2565](https://github.com/gfx-rs/wgpu/pull/2565)
- impl Debug for StagingBelt by @kpreid in [#2572](https://github.com/gfx-rs/wgpu/pull/2572)
- Use fully qualified syntax for some calls. by @jimblandy in [#2655](https://github.com/gfx-rs/wgpu/pull/2655)
- fix: panic in `Storage::get` by @SparkyPotato in [#2657](https://github.com/gfx-rs/wgpu/pull/2657)
- Report invalid pipelines in render bundles as errors, not panics. by @jimblandy in [#2666](https://github.com/gfx-rs/wgpu/pull/2666)
- Perform "valid to use with" checks when recording render bundles. by @jimblandy in [#2690](https://github.com/gfx-rs/wgpu/pull/2690)
- Stop using storage usage for sampling by @cwfitzgerald in [#2703](https://github.com/gfx-rs/wgpu/pull/2703)
- Track depth and stencil writability separately. by @jimblandy in [#2693](https://github.com/gfx-rs/wgpu/pull/2693)
- Improve InvalidScissorRect error message by @jinleili in [#2713](https://github.com/gfx-rs/wgpu/pull/2713)
- Improve InvalidViewport error message by @jinleili in [#2723](https://github.com/gfx-rs/wgpu/pull/2723)
- Don't dirty the vertex buffer for stride/rate changes on bundles. by @jimblandy in [#2744](https://github.com/gfx-rs/wgpu/pull/2744)
- Clean up render bundle index buffer tracking. by @jimblandy in [#2743](https://github.com/gfx-rs/wgpu/pull/2743)
- Improve read-write and read-only texture storage error message by @jinleili in [#2745](https://github.com/gfx-rs/wgpu/pull/2745)
- Change `WEBGPU_TEXTURE_FORMAT_SUPPORT` to `1 << 14` instead of `1 << 15` by @expenses in [#2772](https://github.com/gfx-rs/wgpu/pull/2772)
- fix BufferMapCallbackC & SubmittedWorkDoneClosureC by @rajveermalviya in [#2787](https://github.com/gfx-rs/wgpu/pull/2787)
- Fix formatting of `TextureDimensionError::LimitExceeded`. by @kpreid in [#2799](https://github.com/gfx-rs/wgpu/pull/2799)
- Remove redundant `#[cfg]` conditions from `backend/direct.rs`. by @jimblandy in [#2811](https://github.com/gfx-rs/wgpu/pull/2811)
- Replace android-properties with android_system_properties. by @nical in [#2815](https://github.com/gfx-rs/wgpu/pull/2815)
- Relax render pass color_attachments validation by @jinleili in [#2778](https://github.com/gfx-rs/wgpu/pull/2778)
- Properly Barrier Compute Indirect Buffers by @cwfitzgerald in [#2810](https://github.com/gfx-rs/wgpu/pull/2810)
- Use numeric constants to define `wgpu_types::Features` values. by @jimblandy in [#2817](https://github.com/gfx-rs/wgpu/pull/2817)

#### Metal

- Fix surface texture clear view by @kvark in [#2341](https://github.com/gfx-rs/wgpu/pull/2341)
- Set preserveInvariance for shader options by @scoopr in [#2372](https://github.com/gfx-rs/wgpu/pull/2372)
- Properly set msl version to 2.3 if supported by @cwfitzgerald in [#2418](https://github.com/gfx-rs/wgpu/pull/2418)
- Identify Apple M1 GPU as integrated by @superdump in [#2429](https://github.com/gfx-rs/wgpu/pull/2429)
- Fix M1 in macOS incorrectly reports supported compressed texture formats by @superdump in [#2453](https://github.com/gfx-rs/wgpu/pull/2453)
- Msl: support unsized array not in structures by @kvark in [#2459](https://github.com/gfx-rs/wgpu/pull/2459)
- Fix `Surface::from_uiview` can not guarantee set correct `contentScaleFactor` by @jinleili in [#2470](https://github.com/gfx-rs/wgpu/pull/2470)
- Set `max_buffer_size` by the correct physical device restriction by @jinleili in [#2502](https://github.com/gfx-rs/wgpu/pull/2502)
- Refactor `PrivateCapabilities` creation by @jinleili in [#2509](https://github.com/gfx-rs/wgpu/pull/2509)
- Refactor texture_format_capabilities function by @jinleili in [#2522](https://github.com/gfx-rs/wgpu/pull/2522)
- Improve `push | pop_debug_marker` by @jinleili in [#2537](https://github.com/gfx-rs/wgpu/pull/2537)
- Fix some supported limits by @jinleili in [#2608](https://github.com/gfx-rs/wgpu/pull/2608)
- Don't skip incomplete binding resources. by @dragostis in [#2622](https://github.com/gfx-rs/wgpu/pull/2622)
- Fix `Rgb9e5Ufloat` capabilities and `sampler_lod_average` support by @jinleili in [#2656](https://github.com/gfx-rs/wgpu/pull/2656)
- Fix Depth24Plus | Depth24PlusStencil8 capabilities by @jinleili in [#2686](https://github.com/gfx-rs/wgpu/pull/2686)
- Get_supported_formats: sort like the old get_preferred_format and simplify return type by @victorvde in [#2786](https://github.com/gfx-rs/wgpu/pull/2786)
- Restrict hal::TextureUses::COLOR_TARGET condition within create_texture by @jinleili in [#2818](https://github.com/gfx-rs/wgpu/pull/2818)

#### DX12

- Fix UMA check by @kvark in [#2305](https://github.com/gfx-rs/wgpu/pull/2305)
- Fix partial texture barrier not affecting stencil aspect by @Wumpf in [#2308](https://github.com/gfx-rs/wgpu/pull/2308)
- Improve RowPitch computation by @kvark in [#2409](https://github.com/gfx-rs/wgpu/pull/2409)

#### Vulkan

- Explicitly set Vulkan debug message types instead of !empty() by @victorvde in [#2321](https://github.com/gfx-rs/wgpu/pull/2321)
- Use stencil read/write masks by @kvark in [#2382](https://github.com/gfx-rs/wgpu/pull/2382)
- Vulkan: correctly set INDEPENDENT_BLEND，make runable on Android 8.x by @jinleili in [#2498](https://github.com/gfx-rs/wgpu/pull/2498)
- Fix ASTC format mapping by @kvark in [#2476](https://github.com/gfx-rs/wgpu/pull/2476)
- Support flipped Y on VK 1.1 devices by @cwfitzgerald in [#2512](https://github.com/gfx-rs/wgpu/pull/2512)
- Fixed builtin(primitive_index) for vulkan backend by @kwillemsen in [#2716](https://github.com/gfx-rs/wgpu/pull/2716)
- Fix PIPELINE_STATISTICS_QUERY feature support by @jinleili in [#2750](https://github.com/gfx-rs/wgpu/pull/2750)
- Add a vulkan workaround for large buffers. by @nical in [#2796](https://github.com/gfx-rs/wgpu/pull/2796)

#### GLES

- Fix index buffer state not being reset in reset_state by @rparrett in [#2391](https://github.com/gfx-rs/wgpu/pull/2391)
- Allow push constants trough emulation by @JCapucho in [#2400](https://github.com/gfx-rs/wgpu/pull/2400)
- Hal/gles: fix dirty vertex buffers that are unused by @kvark in [#2427](https://github.com/gfx-rs/wgpu/pull/2427)
- Fix texture description for bgra formats by @JCapucho in [#2520](https://github.com/gfx-rs/wgpu/pull/2520)
- Remove a `log::error!` debugging statement from the gles queue by @expenses in [#2630](https://github.com/gfx-rs/wgpu/pull/2630)
- Fix clearing depth and stencil at the same time by @expenses in [#2675](https://github.com/gfx-rs/wgpu/pull/2675)
- Handle cubemap copies by @expenses in [#2725](https://github.com/gfx-rs/wgpu/pull/2725)
- Allow clearing index buffers by @grovesNL in [#2740](https://github.com/gfx-rs/wgpu/pull/2740)
- Fix buffer-texture copy for 2d arrays by @tuchs in [#2809](https://github.com/gfx-rs/wgpu/pull/2809)

#### Wayland

- Search for different versions of libwayland by @sh7dm in [#2336](https://github.com/gfx-rs/wgpu/pull/2336)

#### WebGPU

- Fix compilation on wasm32-unknown-unknown without `webgl` feature by @jakobhellermann in [#2355](https://github.com/gfx-rs/wgpu/pull/2355)
- Solve crash on WebGPU by @cwfitzgerald in [#2807](https://github.com/gfx-rs/wgpu/pull/2807)

#### Emscripten

- Fix emscripten by @cwfitzgerald in [#2494](https://github.com/gfx-rs/wgpu/pull/2494)

### Performance

- Do texture init via clear passes when possible by @Wumpf in [#2307](https://github.com/gfx-rs/wgpu/pull/2307)
- Bind group deduplication by @cwfitzgerald in [#2623](https://github.com/gfx-rs/wgpu/pull/2623)
- Tracking Optimization and Rewrite by @cwfitzgerald in [#2662](https://github.com/gfx-rs/wgpu/pull/2662)

### Documentation

- Add defaults to new limits and correct older ones by @MultisampledNight in [#/2303](https://github.com/gfx-rs/wgpu/pull/2303)
- Improve shader source documentation by @grovesNL in [#2315](https://github.com/gfx-rs/wgpu/pull/2315)
- Fix typo by @rustui in [#2393](https://github.com/gfx-rs/wgpu/pull/2393)
- Add a :star: to the feature matrix of examples README by @yutannihilation in [#2457](https://github.com/gfx-rs/wgpu/pull/2457)
- Fix get_timestamp_period type in docs by @superdump in [#2478](https://github.com/gfx-rs/wgpu/pull/2478)
- Fix mistake in Access doc comment by @nical in [#2479](https://github.com/gfx-rs/wgpu/pull/2479)
- Improve shader support documentation by @cwfitzgerald in [#2501](https://github.com/gfx-rs/wgpu/pull/2501)
- Document the gfx_select! macro. by @jimblandy in [#2555](https://github.com/gfx-rs/wgpu/pull/2555)
- Add Windows 11 to section about DX12 by @HeavyRain266 in [#2552](https://github.com/gfx-rs/wgpu/pull/2552)
- Document some aspects of resource tracking. by @jimblandy in [#2558](https://github.com/gfx-rs/wgpu/pull/2558)
- Documentation for various things. by @jimblandy in [#2566](https://github.com/gfx-rs/wgpu/pull/2566)
- Fix doc links. by @jimblandy in [#2579](https://github.com/gfx-rs/wgpu/pull/2579)
- Fixed misspelling in documentation by @zenitopires in [#2634](https://github.com/gfx-rs/wgpu/pull/2634)
- Update push constant docs to reflect the API by @Noxime in [#2637](https://github.com/gfx-rs/wgpu/pull/2637)
- Exclude dependencies from documentation by @yutannihilation in [#2642](https://github.com/gfx-rs/wgpu/pull/2642)
- Document `GpuFuture`. by @jimblandy in [#2644](https://github.com/gfx-rs/wgpu/pull/2644)
- Document random bits and pieces. by @jimblandy in [#2651](https://github.com/gfx-rs/wgpu/pull/2651)
- Add cross-references to each wgpu type's documentation. by @kpreid in [#2653](https://github.com/gfx-rs/wgpu/pull/2653)
- RenderPassDescriptor: make label lifetime match doc, and make names descriptive. by @kpreid in [#2654](https://github.com/gfx-rs/wgpu/pull/2654)
- Document `VertexStepMode`. by @jimblandy in [#2685](https://github.com/gfx-rs/wgpu/pull/2685)
- Add links for SpirV documents. by @huandzh in [#2697](https://github.com/gfx-rs/wgpu/pull/2697)
- Add symlink LICENSE files into crates. by @dskkato in [#2604](https://github.com/gfx-rs/wgpu/pull/2604)
- Fix documentation links. by @jimblandy in [#2756](https://github.com/gfx-rs/wgpu/pull/2756)
- Improve push constant documentation, including internal docs. by @jimblandy in [#2764](https://github.com/gfx-rs/wgpu/pull/2764)
- Clarify docs for `wgpu_core`'s `Id` and `gfx_select!`. by @jimblandy in [#2766](https://github.com/gfx-rs/wgpu/pull/2766)
- Update the Supported Platforms table in README by @jinleili in [#2770](https://github.com/gfx-rs/wgpu/pull/2770)
- Remove depth image from readme - we don't dictate direction of depth by @cwfitzgerald in [#2812](https://github.com/gfx-rs/wgpu/pull/2812)

### Dependency Updates

- Update `ash` to `0.37` by @a1phyr in [#2557](https://github.com/gfx-rs/wgpu/pull/2557)
- Update parking_lot to 0.12. by @emilio in [#2639](https://github.com/gfx-rs/wgpu/pull/2639)
- Accept both parking-lot 0.11 and 0.12, to avoid windows-rs. by @jimblandy in [#2660](https://github.com/gfx-rs/wgpu/pull/2660)
- Update web-sys to 0.3.58, sparse attachments support by @jinleili in [#2813](https://github.com/gfx-rs/wgpu/pull/2813)
- Remove use of inplace_it by @mockersf in [#2889](https://github.com/gfx-rs/wgpu/pull/2889)

### deno-webgpu

- Clean up features in deno by @crowlKats in [#2445](https://github.com/gfx-rs/wgpu/pull/2445)
- Dont panic when submitting same commandbuffer multiple times by @crowlKats in [#2449](https://github.com/gfx-rs/wgpu/pull/2449)
- Handle error sources to display full errors by @crowlKats in [#2454](https://github.com/gfx-rs/wgpu/pull/2454)
- Pull changes from deno repo by @crowlKats in [#2455](https://github.com/gfx-rs/wgpu/pull/2455)
- Fix cts_runner by @crowlKats in [#2456](https://github.com/gfx-rs/wgpu/pull/2456)
- Update deno_webgpu by @crowlKats in [#2539](https://github.com/gfx-rs/wgpu/pull/2539)
- Custom op arity by @crowlKats in [#2542](https://github.com/gfx-rs/wgpu/pull/2542)

### Examples

- Fix conserative-raster low res target getting zero sized on resize by @Wumpf in [#2318](https://github.com/gfx-rs/wgpu/pull/2318)
- Replace run-wasm-example.sh with aliased rust crate (xtask) by @rukai in [#2346](https://github.com/gfx-rs/wgpu/pull/2346)
- Get cargo-run-wasm from crates.io by @rukai in [#2415](https://github.com/gfx-rs/wgpu/pull/2415)
- Fix msaa-line example's unnecessary MSAA data store by @jinleili in [#2421](https://github.com/gfx-rs/wgpu/pull/2421)
- Make shadow example runnable on iOS Android devices by @jinleili in [#2433](https://github.com/gfx-rs/wgpu/pull/2433)
- Blit should only draw one triangle by @CurryPseudo in [#2474](https://github.com/gfx-rs/wgpu/pull/2474)
- Fix wasm examples failing to compile by @Liamolucko in [#2524](https://github.com/gfx-rs/wgpu/pull/2524)
- Fix incorrect filtering used in mipmap generation by @LaylBongers in [#2525](https://github.com/gfx-rs/wgpu/pull/2525)
- Correct program output ("Steps", not "Times") by @skierpage in [#2535](https://github.com/gfx-rs/wgpu/pull/2535)
- Fix resizing behaviour of hello-triangle example by @FrankenApps in [#2543](https://github.com/gfx-rs/wgpu/pull/2543)
- Switch from `cgmath` to `glam` in examples by @a1phyr in [#2544](https://github.com/gfx-rs/wgpu/pull/2544)
- Generate 1x1 mip level by @davidar in [#2551](https://github.com/gfx-rs/wgpu/pull/2551)
- Wgpu/examples/shadow: Don't run on llvmpipe. by @jimblandy in [#2595](https://github.com/gfx-rs/wgpu/pull/2595)
- Avoid new WGSL reserved words in wgpu examples. by @jimblandy in [#2606](https://github.com/gfx-rs/wgpu/pull/2606)
- Move texture-array example over to wgsl by @cwfitzgerald in [#2618](https://github.com/gfx-rs/wgpu/pull/2618)
- Remove the default features from wgpu-info by @jinleili in [#2753](https://github.com/gfx-rs/wgpu/pull/2753)
- Fix bunnymark test screenshot and replace rand with nanorand by @stevenhuyn in [#2746](https://github.com/gfx-rs/wgpu/pull/2746)
- Use FIFO swapchain in examples by @cwfitzgerald in [#2790](https://github.com/gfx-rs/wgpu/pull/2790)

### Testing/Internal

- Test WebGPU backend with extra features by @kvark in [#2362](https://github.com/gfx-rs/wgpu/pull/2362)
- Lint deno_webgpu & wgpu-core by @AaronO in [#2403](https://github.com/gfx-rs/wgpu/pull/2403)
- IdentityManager: `from_index` method is unneeded. by @jimblandy in [#2424](https://github.com/gfx-rs/wgpu/pull/2424)
- Added id32 feature by @caiiiycuk in [#2464](https://github.com/gfx-rs/wgpu/pull/2464)
- Update dev deps by @rukai in [#2493](https://github.com/gfx-rs/wgpu/pull/2493)
- Use cargo nextest for running our tests by @cwfitzgerald in [#2495](https://github.com/gfx-rs/wgpu/pull/2495)
- Many Steps Towards GL Testing Working by @cwfitzgerald in [#2504](https://github.com/gfx-rs/wgpu/pull/2504)
- Rename ci.txt to ci.yml by @simon446 in [#2510](https://github.com/gfx-rs/wgpu/pull/2510)
- Re-enable GL testing in CI by @cwfitzgerald in [#2508](https://github.com/gfx-rs/wgpu/pull/2508)
- Expect shadow example to pass on GL by @kvark in [#2541](https://github.com/gfx-rs/wgpu/pull/2541)
- Simplify implementation of RefCount and MultiRefCount. by @jimblandy in [#2548](https://github.com/gfx-rs/wgpu/pull/2548)
- Provide a proper `new` method for `RefCount`. by @jimblandy in [#2570](https://github.com/gfx-rs/wgpu/pull/2570)
- Add logging to LifetimeTracker::triage_suspected. by @jimblandy in [#2569](https://github.com/gfx-rs/wgpu/pull/2569)
- wgpu-hal: Work around cbindgen bug: ignore `gles::egl` module. by @jimblandy in [#2576](https://github.com/gfx-rs/wgpu/pull/2576)
- Specify an exact wasm-bindgen-cli version in publish.yml. by @jimblandy in [#2624](https://github.com/gfx-rs/wgpu/pull/2624)
- Rename `timeout_us` to `timeout_ns`, to match actual units. by @jimblandy in [#2645](https://github.com/gfx-rs/wgpu/pull/2645)
- Move set_index_buffer FFI functions back into wgpu. by @jimblandy in [#2661](https://github.com/gfx-rs/wgpu/pull/2661)
- New function: `Global::create_buffer_error`. by @jimblandy in [#2673](https://github.com/gfx-rs/wgpu/pull/2673)
- Actually use RenderBundleEncoder::set_bind_group in tests. by @jimblandy in [#2678](https://github.com/gfx-rs/wgpu/pull/2678)
- Eliminate wgpu_core::commands::bundle::State::raw_dynamic_offsets. by @jimblandy in [#2684](https://github.com/gfx-rs/wgpu/pull/2684)
- Move RenderBundleEncoder::finish's pipeline layout id into the state. by @jimblandy in [#2755](https://github.com/gfx-rs/wgpu/pull/2755)
- Expect shader_primitive_index tests to fail on AMD RADV POLARIS12. by @jimblandy in [#2754](https://github.com/gfx-rs/wgpu/pull/2754)
- Introduce `VertexStep`: a stride and a step mode. by @jimblandy in [#2768](https://github.com/gfx-rs/wgpu/pull/2768)
- Increase max_outliers on wgpu water example reftest. by @jimblandy in [#2767](https://github.com/gfx-rs/wgpu/pull/2767)
- wgpu_core::command::bundle: Consolidate pipeline and vertex state. by @jimblandy in [#2769](https://github.com/gfx-rs/wgpu/pull/2769)
- Add type annotation to render pass code, for rust-analyzer. by @jimblandy in [#2773](https://github.com/gfx-rs/wgpu/pull/2773)
- Expose naga span location helpers by @nical in [#2752](https://github.com/gfx-rs/wgpu/pull/2752)
- Add create_texture_error by @nical in [#2800](https://github.com/gfx-rs/wgpu/pull/2800)

## wgpu-hal 0.12.5 (2022-04-19)

- fix crashes when logging in debug message callbacks
- fix program termination when dx12 or gles error messages happen.
- implement validation canary
- DX12:
  - Ignore erroneous validation error from DXGI debug layer.

## wgpu-hal-0.12.4 (2022-01-24)

- Metal:
  - check for MSL-2.3

## wgpu-hal-0.12.3, deno-webgpu-? (2022-01-20)

- Metal:
  - preserve vertex invariance
- Vulkan
  - fix stencil read/write masks
- Gles:
  - reset index binding properly
- DX12:
  - fix copies into 1D textures

## wgpu-core-0.12.2, wgpu-hal-0.12.2 (2022-01-10)

- fix tracy compile error
- fix buffer binding limits beyond 2Gb
- fix zero initialization of 3D textures
- Metal:
  - fix surface texture views
- Gles:
  - extend `libwayland` search paths

## wgpu-core-0.12.1, wgpu-hal-0.12.1 (2021-12-29)

- zero initialization uses now render target clears when possible (faster and doesn't enforce COPY_DST internally if not necessary)
  - fix use of MSAA targets in WebGL
  - fix not providing `COPY_DST` flag for textures causing assertions in some cases
  - fix surface textures not getting zero initialized
  - clear_texture supports now depth/stencil targets
- error message on creating depth/stencil volume texture
- Vulkan:
  - fix validation error on debug message types
- DX12:
  - fix check for integrated GPUs
  - fix stencil subresource transitions
- Metal:
  - implement push constants

## wgpu-0.12 (2021-12-18)

- API:
  - `MULTIVIEW` feature
  - `DEPTH_CLIP_CONTROL` feature to replace the old `DEPTH_CLAMP`
  - `TEXTURE_FORMAT_16BIT_NORM` feature
  - push/pop error scopes on the device
  - more limits for compute shaders
  - `SamplerBindingType` instead of booleans
  - sampler arrays are supported by `TEXTURE_BINDING_ARRAY` feature
  - "glsl" cargo feature for accepting GLSL shader code
  - enforced MSRV-1.53
- correctness:
  - textures are zero-initialized
  - lots and lots of fixes
- validation:
  - match texture-sampler pairs
  - check `min_binding_size` late at draw
  - check formats to match in `copy_texture_to_texture`
  - allow `strip_index_format` to be none if unused
  - check workgroup sizes and counts
- shaders:
  - please refer to [naga-0.8 changelog](https://github.com/gfx-rs/naga/pull/1610/files)
  - nice error messages

### wgpu-core-0.11.3, wgpu-hal-0.11.5, wgpu-0.11.1 (2021-12-01)

- Core:
  - validate device descriptor before actually creating it
  - fix validation of texture-sampler pairs
- Vulkan:
  - fix running on Vulkan-1.1 instance
  - improve detection of workaround for Intel+Nvidia on Linux
  - fix resource limits on Vulkan-1.2
  - fix the check for storage buffer requirement
  - change internal semaphore logic to work around Linux+Intel bugs
  - fix enabling extension-provided features
- GLES:
  - fix running on old and bogus drivers
  - fix stale samplers on bindings change
  - fix integer textures
  - fix querying work group parameters
  - fix stale PBO bindings caused by resource copies
  - fix rendering to cubemap faces
  - fix `Rgba16Float` format
  - fix stale vertex attributes when changing the pipeline
- Metal:
  - fix window resizing for running in multiple processes
- Web:
  - fix `set_index_buffer` and `set_vertex_buffer` to have optional sizes

### wgpu-core-0.11.2, wgpu-hal-0.11.4 (2021-10-22)

- fix buffer transition barriers
- Metal:
  - disable RW buffers on macOS 10.11
  - fix memory leaks in render pass descriptor
- WebGL:
  - fix surface reconfiguration
- GLES:
  - fix mapping when persistent mapping isn't supported
  - allow presentation in Android emulator
  - fix sRGB attributes on EGL-1.4 contexts

### wgpu-hal-0.11.3 (2021-10-16)

- GL:
  - fix mapping flags and buffer initialization
  - fix context creation when sRGB is available

### wgpu-core-0.11.1 (2021-10-15)

- fix bind group layout lifetime with regard to bind groups

### wgpu-hal-0.11.2 (2021-10-12)

- GL/WebGL: fix vertex buffer bindings with non-zero first instance
- DX12: fix cube array view construction

### wgpu-hal-0.11.1 (2021-10-09)

- Vulkan: fix NV optimus detection on Linux
- GL:
  - fix indirect dispatch buffers
- WebGL:
  - fix querying storage-related limits
  - work around a browser bug in the clear shader

## wgpu-0.11 (2021-10-07)

- Infrastructure:
  - Deno WebGPU plugin is a part of the repository
  - WebGPU CTS is ran on CI via Deno
- API:
  - initial WebGL support
  - `SwapchainFrame` is removed. `SurfaceTexture::present()` needs to be called instead of dropping.
  - better SPIR-V control flow processing
  - ability to request a software (fallback) adapter
  - new limits for `min_uniform_buffer_offset_alignment` and `min_storage_buffer_offset_alignment`
  - features:
    - new `PARTIALLY_BOUND_BINDING_ARRAY`
    - `NON_FILL_POLYGON_MODE` is split into `POLYGON_MODE_LINE` and `POLYGON_MODE_POINT`
- fixes:
  - many shader-related fixes in Naga-0.7
  - fix a panic in resource cleanup happening when they are dropped on another thread
  - Vulkan:
    - create SPIR-V per entry point to work around driver bugs
    - expose higher descriptor limits based on descriptor indexing capabilities
  - GL and Vulkan:
    - Fix renderdoc device pointers
- optimization:
  - on Vulkan, bounds checks are omitted if the platform can do them natively

### wgpu-core-0.10.4, wgpu-0.10.2 (2021-09-23)

- fix `write_texture` for array textures
- fix closing an encoder on validation error
- expose Metal surface creation
- panic with an actual error message in the default handler

### wgpu-hal-0.10.7 (2021-09-14)

- Metal:
  - fix stencil back-face state
  - fix the limit on command buffer count

### wgpu-hal-0.10.6 (2021-09-12)

- Metal:
  - fix stencil operations
  - fix memory leak on M1 when out of focus
  - fix depth clamping checks
  - fix unsized storage buffers beyond the first

### wgpu-core-0.10.3, wgpu-hal-0.10.4 (2021-09-08)

- Vulkan:
  - fix read access barriers for writable storage buffers
  - fix shaders using cube array textures
  - work around Linux Intel+Nvidia driver conflicts
  - work around Adreno bug with `OpName`
- DX12:
  - fix storage binding offsets
- Metal:
  - fix compressed texture copies

### wgpu-core-0.10.2, wgpu-hal-0.10.3 (2021-09-01)

- All:
  - fix querying the size of storage textures
- Vulkan:
  - use render pass labels
- Metal:
  - fix moving the surface between displays
- DX12:
  - enable BC compressed textures
- GL:
  - fix vertex-buffer and storage related limits

### wgpu-core-0.10.1, wgpu-hal-0.10.2 (2021-08-24)

- All:
  - expose more formats via adapter-specific feature
  - fix creation of depth+stencil views
  - validate cube textures to not be used as storage
  - fix mip level count check for storage textures
- Metal:
  - fix usage of work group memory
- DX12:
  - critical fix of pipeline layout

## v0.10 (2021-08-18)

- Infrastructure:
  - `gfx-hal` is replaced by the in-house graphics abstraction `wgpu-hal`. Backends: Vulkan, Metal, D3D-12, and OpenGL ES-3.
  - examples are tested automatically for image snapshots.
- API:
  - `cross` feature is removed entirely. Only Rust code from now on.
  - processing SPIR-V inputs for later translation now requires `spirv` compile feature enabled
  - new `Features::SPIRV_SHADER_PASSTHROUGH` run-time feature allows providing pass-through SPIR-V (orthogonal to the compile feature)
  - several bitflag names are renamed to plural: `TextureUsage`, `BufferUsage`, `ColorWrite`.
  - the `SwapChain` is merged into `Surface`. Returned frames are `Texture` instead of `TextureView`.
  - renamed `TextureUsage` bits: `SAMPLED` -> `TEXTURE_BINDING`, `STORAGE` -> `STORAGE_BINDING`.
  - renamed `InputStepMode` to `VertexStepMode`.
  - readable storage textures are no longer a part of the base API. Only exposed via format-specific features, non-portably.
  - implemented `Rgb9e5Ufloat` format.
  - added limits for binding sizes, vertex data, per-stage bindings, and others.
  - reworked downlevel flags, added downlevel limits.
  - `resolver = "2"` is now required in top-level cargo manifests
- Fixed:
  - `Device::create_query_set` would return an error when creating exactly `QUERY_SET_MAX_QUERIES` (8192) queries. Now it only returns an error when trying to create _more_ than `QUERY_SET_MAX_QUERIES` queries.

### wgpu-core-0.9.2

- fix `Features::TEXTURE_SPECIFIC_FORMAT_FEATURES` not being supported for rendertargets

### wgpu-core-0.9.1 (2021-07-13)

- fix buffer inits delayed by a frame
- fix query resolves to initialize buffers
- fix pipeline statistics stride
- fix the check for maximum query count

## v0.9 (2021-06-18)

- Updated:
  - naga to `v0.5`.
- Added:
  - `Features::VERTEX_WRITABLE_STORAGE`.
  - `Features::CLEAR_COMMANDS` which allows you to use `cmd_buf.clear_texture` and `cmd_buf.clear_buffer`.
- Changed:
  - Updated default storage buffer/image limit to `8` from `4`.
- Fixed:
  - `Buffer::get_mapped_range` can now have a range of zero.
  - Fixed output spirv requiring the "kernal" capability.
  - Fixed segfault due to improper drop order.
  - Fixed incorrect dynamic stencil reference for Replace ops.
  - Fixed tracking of temporary resources.
  - Stopped unconditionally adding cubemap flags when the backend doesn't support cubemaps.
- Validation:
  - Ensure that if resources are viewed from the vertex stage, they are read only unless `Features::VERTEX_WRITABLE_STORAGE` is true.
  - Ensure storage class (i.e. storage vs uniform) is consistent between the shader and the pipeline layout.
  - Error when a color texture is used as a depth/stencil texture.
  - Check that pipeline output formats are logical
  - Added shader label to log messages if validation fails.
- Tracing:
  - Make renderpasses show up in the trace before they are run.
- Docs:
  - Fix typo in `PowerPreference::LowPower` description.
- Player:
  - Automatically start and stop RenderDoc captures.
- Examples:
  - Handle winit's unconditional exception.
- Internal:
  - Merged wgpu-rs and wgpu back into a single repository.
  - The tracker was split into two different stateful/stateless trackers to reduce overhead.
  - Added code coverage testing
  - CI can now test on lavapipe
  - Add missing extern "C" in wgpu-core on `wgpu_render_pass_execute_bundles`
  - Fix incorrect function name `wgpu_render_pass_bundle_indexed_indirect` to `wgpu_render_bundle_draw_indexed_indirect`.

### wgpu-types-0.8.1 (2021-06-08)

- fix dynamic stencil reference for Replace ops

### v0.8.1 (2021-05-06)

- fix SPIR-V generation from WGSL, which was broken due to "Kernel" capability
- validate buffer storage classes
- Added support for storage texture arrays for Vulkan and Metal.

## v0.8 (2021-04-29)

- Naga is used by default to translate shaders, SPIRV-Cross is optional behind `cross` feature
- Features:
  - buffers are zero-initialized
  - downlevel limits for DX11/OpenGL support
  - conservative rasterization (native-only)
  - buffer resource indexing (native-only)
- API adjustments to the spec:
  - Renamed `RenderPassColorAttachmentDescriptor` to `RenderPassColorAttachment`:
    - Renamed the `attachment` member to `view`
  - Renamed `RenderPassDepthStencilAttachmentDescriptor` to `RenderPassDepthStencilAttachment`:
    - Renamed the `attachment` member to `view`
  - Renamed `VertexFormat` values
    - Examples: `Float3` -> `Float32x3`, `Ushort2` -> `Uint16x2`
  - Renamed the `depth` value of `Extent3d` to `depth_or_array_layers`
  - Updated blending options in `ColorTargetState`:
    - Renamed `BlendState` to `BlendComponent`
    - Added `BlendState` struct to hold color and alpha blend state
    - Moved `color_blend` and `alpha_blend` members into `blend` member
  - Moved `clamp_depth` from `RastizerState` to `PrimitiveState`
  - Updated `PrimitiveState`:
    - Added `conservative` member for enabling conservative rasterization
  - Updated copy view structs:
    - Renamed `TextureCopyView` to `ImageCopyTexture`
    - Renamed `TextureDataLayout` to `ImageDataLayout`
    - Changed `bytes_per_row` and `rows_per_image` members of `ImageDataLayout` from `u32` to `Option<NonZeroU32>` <!-- wgpu-rs only -->
  - Changed `BindingResource::Binding` from containing fields directly to containing a `BufferBinding`
  - Added `BindingResource::BufferArray`
- Infrastructure:
  - switch from `tracing` to `profiling`
  - more concrete and detailed errors
  - API traces include the command that crashed/panicked
  - Vulkan Portability support is removed from Apple platforms
- Validation:
  - texture bindings
  - filtering of textures by samplers
  - interpolation qualifiers
  - allow vertex components to be underspecified

### wgpu-core-0.7.1 (2021-02-25)

- expose `wgc::device::queue` sub-module in public
- fix the indexed buffer check
- fix command allocator race condition

## v0.7 (2021-01-31)

- Major API changes:
  - `RenderPipelineDescriptor`
  - `BindingType`
  - new `ShaderModuleDescriptor`
  - new `RenderEncoder`
- Features:
  - (beta) WGSL support, including the ability to bypass SPIR-V entirely
  - (beta) implicit bind group layout support
  - better error messages
  - timestamp and pipeline statistics queries
  - ETC2 and ASTC compressed textures
  - (beta) targeting WASM with WebGL backend
  - reduced dependencies
  - Native-only:
    - clamp-to-border addressing
    - polygon fill modes
    - query a format for extra capabilities
    - `f64` support in shaders
- Validation:
  - shader interface
  - render pipeline descriptor
  - vertex buffers

### wgpu-0.6.2 (2020-11-24)

- don't panic in the staging belt if the channel is dropped

## v0.6 (2020-08-17)

- Crates:
  - C API is moved to [another repository](https://github.com/gfx-rs/wgpu-native)
  - `player`: standalone API replayer and tester
- Features:
  - Proper error handling with all functions returning `Result`
  - Graceful handling of "error" objects
  - API tracing [infrastructure](http://kvark.github.io/wgpu/debug/test/ron/2020/07/18/wgpu-api-tracing.html)
  - uploading data with `write_buffer`/`write_texture` queue operations
  - reusable render bundles
  - read-only depth/stencil attachments
  - bind group layout deduplication
  - Cows, cows everywhere
  - Web+Native features:
    - Depth clamping (feature)
    - BC texture compression
  - Native-only features:
    - mappable primary buffers
    - texture array bindings
    - push constants
    - multi-draw indirect
- Validation:
  - all transfer operations
  - all resource creation
  - bind group matching to the layout
  - experimental shader interface matching with Naga

### wgpu-core-0.5.6 (2020-07-09)

- add debug markers support

### wgpu-core-0.5.5 (2020-05-20)

- fix destruction of adapters, swap chains, and bind group layouts
- fix command pool leak with temporary threads
- improve assertion messages
- implement `From<TextureFormat>` for `TextureComponentType`

### wgpu-core-0.5.4 (2020-04-24)

- fix memory management of staging buffers

### wgpu-core-0.5.3 (2020-04-18)

- fix reading access to storage textures
- another fix to layout transitions for swapchain images

### wgpu-core-0.5.2 (2020-04-15)

- fix read-only storage flags
- fix pipeline layout life time
- improve various assert messages

### wgpu-core-0.5.1 (2020-04-10)

- fix tracking of swapchain images that are used multiple times in a command buffer
- fix tracking of initial usage of a resource across a command buffer

## v0.5 (2020-04-06)

- Crates:
  - `wgpu-types`: common types between native and web targets
  - `wgpu-core`: internal API for the native and remote wrappers
- Features:
  - based on gfx-hal-0.5
  - moved from Rendy to the new `gfx-memory` and `gfx-descriptor` crates
  - passes are now recorded on the client side. The user is also responsible to keep all resources referenced in the pass up until it ends recording.
  - coordinate system is changed to have Y up in the rendering space
  - revised GPU lifetime tracking of all resources
  - revised usage tracking logic
  - all IDs are now non-zero
  - Mailbox present mode
- Validation:
  - active pipeline
- Fixes:
  - lots of small API changes to closely match upstream WebGPU
  - true read-only storage bindings
  - unmapping dropped buffers
  - better error messages on misused swapchain frames

### wgpu-core-0.4.3 (2020-01-20)

- improved swap chain error handling

### wgpu-core-0.4.2 (2019-12-15)

- fixed render pass transitions

### wgpu-core-0.4.1 (2019-11-28)

- fixed depth/stencil transitions
- fixed dynamic offset iteration

## v0.4 (2019-11-03)

- Platforms: removed OpenGL/WebGL support temporarily
- Features:
  - based on gfx-hal-0.4 with the new swapchain model
  - exposing adapters from all available backends on a system
  - tracking of samplers
  - cube map support with an example
- Validation:
  - buffer and texture usage

### wgpu-core-0.3.3 (2019-08-22)

- fixed instance creation on Windows

### wgpu-core-0.3.1 (2019-08-21)

- fixed pipeline barriers that aren't transitions

## v0.3 (2019-08-21)

- Platforms: experimental OpenGL/WebGL
- Crates:
  - Rust API is moved out to [another repository](https://github.com/gfx-rs/wgpu-rs)
- Features:
  - based on gfx-hal-0.3 with help of `rendy-memory` and `rendy-descriptor`
  - type-system-assisted deadlock prevention (for locking internal structures)
  - texture sub-resource tracking
  - `raw-window-handle` integration instead of `winit`
  - multisampling with an example
  - indirect draws and dispatches
  - stencil masks and reference values
  - native "compute" example
  - everything implements `Debug`
- Validation
  - vertex/index/instance ranges at draw calls
  - bing groups vs their expected layouts
  - bind group buffer ranges
  - required stencil reference, blend color

### wgpu-core-0.2.6 (2019-04-04)

- fixed frame acquisition GPU waits

### wgpu-core-0.2.5 (2019-03-31)

- fixed submission tracking
- added support for blend colors
- fixed bind group compatibility at the gfx-hal level
- validating the bind groups and blend colors

### wgpu-core-0.2.3 (2019-03-20)

- fixed vertex format mapping
- fixed building with "empty" backend on Windows
- bumped the default descriptor pool size
- fixed host mapping alignments
- validating the uniform buffer offset

## v0.2 (2019-03-06)

- Platforms: iOS/Metal, D3D11
- Crates:
  - `wgpu-remote`: remoting layer for the cross-process boundary
  - `gfx-examples`: selected gfx pre-ll examples ported over
- Features:
  - native example for compute
  - "gfx-cube" and "gfx-shadow" examples
  - copies between buffers and textures
  - separate object identity for the remote client
  - texture view tracking
  - native swapchain resize support
  - buffer mapping
  - object index epochs
  - comprehensive list of vertex and texture formats
  - validation of pipeline compatibility with the pass
- Fixes
  - fixed resource destruction

## v0.1 (2019-01-24)

- Platforms: Linux/Vulkan, Windows/Vulkan, D3D12, macOS/Metal
- Crates:
  - `wgpu-native`: C API implementation of WebGPU, based on gfx-hal
  - `wgpu-bindings`: auto-generated C headers
  - `wgpu`: idiomatic Rust wrapper
  - `examples`: native C examples
- Features:
- native examples for triangle rendering
- basic native swapchain integration
- concept of the storage hub
- basic recording of passes and command buffers
- submission-based lifetime tracking and command buffer recycling
- automatic resource transitions
