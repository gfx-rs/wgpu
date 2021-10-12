# Change Log

## wgpu-hal-0.11.2 (2021-10-12)
  - GL/WebGL: fix vertex buffer bindings with non-zero first instance
  - DX12: fix cube array view construction

## wgpu-hal-0.11.1 (2021-10-09)
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
    - `Device::create_query_set` would return an error when creating exactly `QUERY_SET_MAX_QUERIES` (8192) queries. Now it only returns an error when trying to create *more* than `QUERY_SET_MAX_QUERIES` queries.

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
  -  Features:
    - native examples for triangle rendering
    - basic native swapchain integration
    - concept of the storage hub
    - basic recording of passes and command buffers
    - submission-based lifetime tracking and command buffer recycling
    - automatic resource transitions
