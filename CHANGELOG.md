# Change Log

## v0.8 (2021-04-29)
  - Naga is used by default to translate shaders, SPIRV-Cross is optional behind `cross` feature
  - Features:
    - buffers are zero-initialized
    - downlevel limits for DX11/OpenGL support
    - conservative rasterization (native-only)
    - buffer resource indexing (native-only)
  - API adjustments to the spec:
    - Renamed `RenderPassDepthStencilAttachmentDescriptor` to `RenderPassDepthStencilAttachment`:
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

## v0.7.1 (2021-02-25)
  - expose `wgc::device::queue` sub-module in public
  - fix the indexed buffer check
  - fix command allocator race condition

## v0.7 (2021-01-31)
  - Major API changes:
    - `RenderPipelineDescriptor`
    - `BindingType`
  - Features:
    - (beta) WGSL support, including the ability to bypass SPIR-V entirely
    - (beta) implicit bind group layout support
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

## v0.5.6 (2020-07-09)
  - add debug markers support

## v0.5.5 (2020-05-20)
  - fix destruction of adapters, swap chains, and bind group layouts
  - fix command pool leak with temporary threads
  - improve assertion messages
  - implement `From<TextureFormat>` for `TextureComponentType`

## v0.5.4 (2020-04-24)
  - fix memory management of staging buffers

## v0.5.3 (2020-04-18)
  - fix reading access to storage textures
  - another fix to layout transitions for swapchain images

## v0.5.2 (2020-04-15)
  - fix read-only storage flags
  - fix pipeline layout life time
  - improve various assert messages

## v0.5.1 (2020-04-10)
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

## v0.4.3 (2020-01-20)
  - improved swap chain error handling

## v0.4.2 (2019-12-15)
  - fixed render pass transitions

## v0.4.1 (2019-11-28)
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

## v0.3.3 (2019-08-22)
  - fixed instance creation on Windows

## v0.3.1 (2019-08-21)
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

## v0.2.6 (2019-04-04)
  - fixed frame acquisition GPU waits

## v0.2.5 (2019-03-31)
  - fixed submission tracking
  - added support for blend colors
  - fixed bind group compatibility at the gfx-hal level
  - validating the bind groups and blend colors

## v0.2.3 (2019-03-20)
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
