# Change Log

## v0.5 (2020-04-06)
  - Crates:
    - `wgpu-types`: common types between native and web targets
    - `wgpu-core`: internal API for the native and remote wrappers
  - Features:
    - based on gfx-hal-0.5
    - moved from Rendy to the new `gfx-memory` and `gfx-descriptor` crates
    - passes are now recorded on the client side. The user is also responsible to keep all resources referenced in the pass up until it ends recording.
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
