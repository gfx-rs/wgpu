# Change Log

## v0.4 (03-11-2019)
  - Platforms: removed OpenGL/WebGL support temporarily
  - Features:
    - based on gfx-hal-0.4 with the new swapchain model
    - exposing adapters from all available backends on a system
    - tracking of samplers
    - cube map support with an example
  - Validation:
    - buffer and texture usage

## v0.3.3 (22-08-2019)
  - fixed instance creation on Windows

## v0.3.1 (21-08-2019)
  - fixed pipeline barriers that aren't transitions

## v0.3 (21-08-2019)
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

## v0.2.6 (04-04-2019)
  - fixed frame acquisition GPU waits

## v0.2.5 (31-03-2019)
  - fixed submission tracking
  - added support for blend colors
  - fixed bind group compatibility at the gfx-hal level
  - validating the bind groups and blend colors

## v0.2.3 (20-03-2019)
  - fixed vertex format mapping
  - fixed building with "empty" backend on Windows
  - bumped the default descriptor pool size
  - fixed host mapping alignments
  - validating the uniform buffer offset

## v0.2 (06-03-2019)
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

## v0.1 (24-01-2019)
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
