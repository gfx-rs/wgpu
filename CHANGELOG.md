# Change Log

<!--
Please add your PR to the changelog! Choose from a top level and bottom
level category, then write your changes like follows:

- Describe your change in a user friendly format. By @yourslug in [#99999](https://github.com/gfx-rs/wgpu/pull/99999)

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

- Naga
- General
- DX12
- Vulkan
- Metal
- GLES / OpenGL
- WebGPU
- Emscripten
- Hal
-->

## Unreleased

### Major changes

#### Refactored Dispatch Between `wgpu-core` and `webgpu`

The crate `wgpu` has two different "backends", one which targets webgpu in the browser, one which targets `wgpu_core` on native platforms and webgl. This was previously very difficult to traverse and add new features to. The entire system was refactored to make it simpler. Additionally the new system has zero overhead if there is only one "backend" in use. You can see the new system in action by using go-to-definition on any wgpu functions in your IDE.

By @cwfitzgerald in [#6619](https://github.com/gfx-rs/wgpu/pull/6619).

#### Render and Compute Passes Now Properly Enforce Their Lifetime

A regression introduced in 23.0.0 caused lifetimes of render and compute passes to be incorrectly enforced. While this is not
a soundness issue, the intent is to move an error from runtime to compile time. This issue has been fixed and restored to the 22.0.0 behavior.

#### Bindless (`binding_array`) Grew More Capabilities

- DX12 now supports `PARTIALLY_BOUND_BINDING_ARRAY` on Resource Binding Tier 3 Hardware. This is most D3D12 hardware [D3D12 Feature Table] for more information on what hardware supports this feature. By @cwfitzgerald in [#6734](https://github.com/gfx-rs/wgpu/pull/6734).

[D3D12 Feature Table]: https://d3d12infodb.boolka.dev/FeatureTable.html

#### `Device::create_shader_module_unchecked` Renamed and Now Has Configuration Options

`create_shader_module_unchecked` became `create_shader_module_trusted`.

This allows you to customize which exact checks are omitted so that you can get the correct balance of performance and safety for your use case. Calling the function is still unsafe, but now can be used to skip certain checks only on certain builds.

This also allows users to disable the workarounds in the `msl-out` backend to prevent the compiler from optimizing infinite loops. This can have a big impact on performance, but is not recommended for untrusted shaders.

```diff
let desc: ShaderModuleDescriptor = include_wgsl!(...)
- let module = unsafe { device.create_shader_module_unchecked(desc) };
+ let module = unsafe { device.create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked()) };
```

By @cwfitzgerald and @rudderbucky in [#6662](https://github.com/gfx-rs/wgpu/pull/6662).

#### The `diagnostic(…);` directive is now supported in WGSL

Naga now parses `diagnostic(…);` directives according to the WGSL spec. This allows users to control certain lints, similar to Rust's `allow`, `warn`, and `deny` attributes. For example, in standard WGSL (but, notably, not Naga yet—see <https://github.com/gfx-rs/wgpu/issues/4369>) this snippet would emit a uniformity error:

```wgsl
@group(0) @binding(0) var s : sampler;
@group(0) @binding(2) var tex : texture_2d<f32>;
@group(1) @binding(0) var<storage, read> ro_buffer : array<f32, 4>;

@fragment
fn main(@builtin(position) p : vec4f) -> @location(0) vec4f {
  if ro_buffer[0] == 0 {
    // Emits a derivative uniformity error during validation.
    return textureSample(tex, s, vec2(0.,0.));
  }

  return vec4f(0.);
}
```

…but we can now silence it with the `off` severity level, like so:

```wgsl
// Disable the diagnosic with this…
diagnostic(off, derivative_uniformity);

@group(0) @binding(0) var s : sampler;
@group(0) @binding(2) var tex : texture_2d<f32>;
@group(1) @binding(0) var<storage, read> ro_buffer : array<f32, 4>;

@fragment
fn main(@builtin(position) p : vec4f) -> @location(0) vec4f {
  if ro_buffer[0] == 0 {
    // Look ma, no error!
    return textureSample(tex, s, vec2(0.,0.));
  }

  return vec4f(0.);
}
```

There are some limitations to keep in mind with this new functionality:

- We support `@diagnostic(…)` rules as `fn` attributes, but prioritization for rules in statement positions (i.e., `if (…) @diagnostic(…) { … }` is unclear. If you are blocked by not being able to parse `diagnostic(…)` rules in statement positions, please let us know in <https://github.com/gfx-rs/wgpu/issues/5320>, so we can determine how to prioritize it!
- Standard WGSL specifies `error`, `warning`, `info`, and `off` severity levels. These are all technically usable now! A caveat, though: warning- and info-level are only emitted to `stderr` via the `log` façade, rather than being reported through a `Result::Err` in Naga or the `CompilationInfo` interface in `wgpu{,-core}`. This will require breaking changes in Naga to fix, and is being tracked by <https://github.com/gfx-rs/wgpu/issues/6458>.
- Not all lints can be controlled with `diagnostic(…)` rules. In fact, only the `derivative_uniformity` triggering rule exists in the WGSL standard. That said, Naga contributors are excited to see how this level of control unlocks a new ecosystem of configurable diagnostics.
- Finally, `diagnostic(…)` rules are not yet emitted in WGSL output. This means that `wgsl-in` → `wgsl-out` is currently a lossy process. We felt that it was important to unblock users who needed `diagnostic(…)` rules (i.e., <https://github.com/gfx-rs/wgpu/issues/3135>) before we took significant effort to fix this (tracked in <https://github.com/gfx-rs/wgpu/issues/6496>).

By @ErichDonGubler in [#6456](https://github.com/gfx-rs/wgpu/pull/6456), [#6148](https://github.com/gfx-rs/wgpu/pull/6148), [#6533](https://github.com/gfx-rs/wgpu/pull/6533), [#6353](https://github.com/gfx-rs/wgpu/pull/6353), [#6537](https://github.com/gfx-rs/wgpu/pull/6537).

#### `wgpu::Instance::new` now takes `InstanceDescriptor` by reference

Previously `wgpu::Instance::new` took `InstanceDescriptor` by value (which is overall fairly uncommon in wgpu).
Furthermore, `InstanceDescriptor` is now cloneable.

```diff
- let instance = wgpu::Instance::new(instance_desc);
+ let instance = wgpu::Instance::new(&instance_desc);
```

By @wumpf in [#6849](https://github.com/gfx-rs/wgpu/pull/6849).

#### New Features

##### Naga

- Support atomic operations on fields of global structs in the SPIR-V frontend. By @schell in [#6693](https://github.com/gfx-rs/wgpu/pull/6693).
- Clean up tests for atomic operations support in SPIR-V frontend. By @schell in [#6692](https://github.com/gfx-rs/wgpu/pull/6692)
- Fix an issue where `naga` CLI would incorrectly skip the first positional argument when `--stdin-file-path` was specified. By @ErichDonGubler in [#6480](https://github.com/gfx-rs/wgpu/pull/6480).
- Fix textureNumLevels in the GLSL backend. By @magcius in [#6483](https://github.com/gfx-rs/wgpu/pull/6483).
- Support 64-bit hex literals and unary operations in constants [#6616](https://github.com/gfx-rs/wgpu/pull/6616).
- Implement `quantizeToF16()` for WGSL frontend, and WGSL, SPIR-V, HLSL, MSL, and GLSL backends. By @jamienicol in [#6519](https://github.com/gfx-rs/wgpu/pull/6519).
- Add support for GLSL `usampler*` and `isampler*`. By @DavidPeicho in [#6513](https://github.com/gfx-rs/wgpu/pull/6513).
- Expose Ray Query flags as constants in WGSL. Implement candidate intersections. By @kvark in [#5429](https://github.com/gfx-rs/wgpu/pull/5429)
- Add new vertex formats (`{U,S}{int,norm}{8,16}`, `Float16` and `Unorm8x4Bgra`). By @nolanderc in [#6632](https://github.com/gfx-rs/wgpu/pull/6632)
- Allow for override-expressions in `workgroup_size`. By @KentSlaney in [#6635](https://github.com/gfx-rs/wgpu/pull/6635).
- Add support for OpAtomicCompareExchange in SPIR-V frontend. By @schell in [#6590](https://github.com/gfx-rs/wgpu/pull/6590).
- Implement type inference for abstract arguments to user-defined functions. By @jamienicol in [#6577](https://github.com/gfx-rs/wgpu/pull/6577).
- Allow for override-expressions in array sizes. By @KentSlaney in [#6654](https://github.com/gfx-rs/wgpu/pull/6654).

##### General

- Add unified documentation for ray-tracing. By @Vecvec in [#6747](https://github.com/gfx-rs/wgpu/pull/6747)
- Return submission index in `map_async` and `on_submitted_work_done` to track down completion of async callbacks. By @eliemichel in [#6360](https://github.com/gfx-rs/wgpu/pull/6360).
- Move raytracing alignments into HAL instead of in core. By @Vecvec in [#6563](https://github.com/gfx-rs/wgpu/pull/6563).
- Allow for statically linking DXC rather than including separate `.dll` files. By @DouglasDwyer in [#6574](https://github.com/gfx-rs/wgpu/pull/6574).
- `DeviceType` and `AdapterInfo` now impl `Hash` by @cwfitzgerald in [#6868](https://github.com/gfx-rs/wgpu/pull/6868)

#### Changes

##### Naga

- Show types of LHS and RHS in binary operation type mismatch errors. By @ErichDonGubler in [#6450](https://github.com/gfx-rs/wgpu/pull/6450).
- The GLSL parser now uses less expressions for function calls. By @magcius in [#6604](https://github.com/gfx-rs/wgpu/pull/6604).
- Add a note to help with a common syntax error case for global diagnostic filter directives. By @e-hat in [#6718](https://github.com/gfx-rs/wgpu/pull/6718)
- Change arithmetic operations between two i32 variables to wrap on overflow to match WGSL spec. By @matthew-wong1 in [#6835](https://github.com/gfx-rs/wgpu/pull/6835).
- Add directives to suggestions in error message for parsing global items. By @e-hat in [#6723](https://github.com/gfx-rs/wgpu/pull/6723).

##### General

- Align Storage Access enums to the webgpu spec. By @atlv24 in [#6642](https://github.com/gfx-rs/wgpu/pull/6642)
- Make `Surface::as_hal` take an immutable reference to the surface. By @jerzywilczek in [#9999](https://github.com/gfx-rs/wgpu/pull/9999)
- Add actual sample type to `CreateBindGroupError::InvalidTextureSampleType` error message. By @ErichDonGubler in [#6530](https://github.com/gfx-rs/wgpu/pull/6530).
- Improve binding error to give a clearer message when there is a mismatch between resource binding as it is in the shader and as it is in the binding layout. By @eliemichel in [#6553](https://github.com/gfx-rs/wgpu/pull/6553).
- `Surface::configure` and `Surface::get_current_texture` are no longer fatal. By @alokedesai in [#6253](https://github.com/gfx-rs/wgpu/pull/6253)

##### D3D12

- Avoid using FXC as fallback when the DXC container was passed at instance creation. Paths to `dxcompiler.dll` & `dxil.dll` are also now required. By @teoxoy in [#6643](https://github.com/gfx-rs/wgpu/pull/6643).

##### Vulkan

- Add a cache for samplers, deduplicating any samplers, allowing more programs to stay within the global sampler limit. By @cwfitzgerald in [#6847](https://github.com/gfx-rs/wgpu/pull/6847)

##### HAL

- Replace `usage: Range<T>`, for `BufferUses`, `TextureUses`, and `AccelerationStructureBarrier` with a new `StateTransition<T>`. By @atlv24 in [#6703](https://github.com/gfx-rs/wgpu/pull/6703)
- Change the `DropCallback` API to use `FnOnce` instead of `FnMut`. By @jerzywilczek in [#6482](https://github.com/gfx-rs/wgpu/pull/6482)

### Bug Fixes

#### General

- Handle query set creation failure as an internal error that loses the `Device`, rather than panicking. By @ErichDonGubler in [#6505](https://github.com/gfx-rs/wgpu/pull/6505).
- Ensure that `Features::TIMESTAMP_QUERY` is set when using timestamp writes in render and compute passes. By @ErichDonGubler in [#6497](https://github.com/gfx-rs/wgpu/pull/6497).
- Check for device mismatches when beginning render and compute passes. By @ErichDonGubler in [#6497](https://github.com/gfx-rs/wgpu/pull/6497).
- Lower `QUERY_SET_MAX_QUERIES` (and enforced limits) from 8192 to 4096 to match WebGPU spec. By @ErichDonGubler in [#6525](https://github.com/gfx-rs/wgpu/pull/6525).
- Allow non-filterable float on texture bindings never used with samplers when using a derived bind group layout. By @ErichDonGubler in [#6531](https://github.com/gfx-rs/wgpu/pull/6531/).
- Replace potentially unsound usage of `PreHashedMap` with `FastHashMap`. By @jamienicol in [#6541](https://github.com/gfx-rs/wgpu/pull/6541).
- Add missing validation for timestamp writes in compute and render passes. By @ErichDonGubler in [#6578](https://github.com/gfx-rs/wgpu/pull/6578), [#6583](https://github.com/gfx-rs/wgpu/pull/6583).
  - Check the status of the `TIMESTAMP_QUERY` feature before other validation.
  - Check that indices are in-bounds for the query set.
  - Check that begin and end indices are not equal.
  - Check that at least one index is specified.
- Reject destroyed buffers in query set resolution. By @ErichDonGubler in [#6579](https://github.com/gfx-rs/wgpu/pull/6579).
- Fix panic when dropping `Device` on some environments. By @Dinnerbone in [#6681](https://github.com/gfx-rs/wgpu/pull/6681).
- Reduced the overhead of command buffer validation. By @nical in [#6721](https://github.com/gfx-rs/wgpu/pull/6721).
- Set index type to NONE in `get_acceleration_structure_build_sizes`. By @Vecvec in [#6802](https://github.com/gfx-rs/wgpu/pull/6802).
- Fix `wgpu-info` not showing dx12 adapters. By @wumpf in [#6844](https://github.com/gfx-rs/wgpu/pull/6844).
- Use `transform_buffer_offset` when initialising `transform_buffer`. By @Vecvec in [#6864](https://github.com/gfx-rs/wgpu/pull/6864).

#### Naga

- Fix crash when a texture argument is missing. By @aedm in [#6486](https://github.com/gfx-rs/wgpu/pull/6486)
- Emit an error in constant evaluation, rather than crash, in certain cases where `vecN` constructors have less than N arguments. By @ErichDonGubler in [#6508](https://github.com/gfx-rs/wgpu/pull/6508).

#### Vulkan

- Allocate descriptors for acceleration structures. By @Vecvec in [#6861](https://github.com/gfx-rs/wgpu/pull/6861).
- `max_color_attachment_bytes_per_sample` is now correctly set to 128. By @cwfitzgerald in [#6866](https://github.com/gfx-rs/wgpu/pull/6866)

#### D3D12

- Fix no longer showing software rasterizer adapters. By @wumpf in [#6843](https://github.com/gfx-rs/wgpu/pull/6843).
- `max_color_attachment_bytes_per_sample` is now correctly set to 128. By @cwfitzgerald in [#6866](https://github.com/gfx-rs/wgpu/pull/6866)

### Examples

- Add multiple render targets example. By @kaphula in [#5297](https://github.com/gfx-rs/wgpu/pull/5313)


### Testing

- Tests the early returns in the acceleration structure build calls with empty calls. By @Vecvec in [#6651](https://github.com/gfx-rs/wgpu/pull/6651).

## 23.0.1 (2024-11-25)

This release includes patches for `wgpu`, `wgpu-core` and `wgpu-hal`. All other crates remain at [23.0.0](https://github.com/gfx-rs/wgpu/releases/tag/v23.0.0).
Below changes were cherry-picked from 24.0.0 development line.

### Bug fixes

#### General

- Fix Texture view leaks regression. By @xiaopengli89 in [#6576](https://github.com/gfx-rs/wgpu/pull/6576)

#### Metal

- Fix surface creation crashing on iOS. By @mockersf in [#6535](https://github.com/gfx-rs/wgpu/pull/6535)

#### Vulkan

- Fix surface capabilities being advertised when its query failed. By @wumpf in [#6510](https://github.com/gfx-rs/wgpu/pull/6510)

## 23.0.0 (2024-10-25)

### Themes of this release

This release's theme is one that is likely to repeat for a few releases: convergence with the WebGPU specification! WGPU's design and base functionality are actually determined by two specifications: one for WebGPU, and one for the WebGPU Shading Language.

This may not sound exciting, but let us convince you otherwise! All major web browsers have committed to offering WebGPU in their environment. Even JS runtimes like [Node][nodejs-webgpu-interest] and [Deno][deno_webgpu-crate-manifest] have communities that are very interested in providing WebGPU! WebGPU is slowly [eating the world][eat-the-world-meaning], as it were. 😀 It's really important, then, that WebGPU implementations behave in ways that one would expect across all platforms. For example, if Firefox's WebGPU implementation were to break when running scripts and shaders that worked just fine in Chrome, that would mean sad users for both application authors _and_ browser authors.

[nodejs-webgpu-interest]: https://github.com/orgs/nodejs/discussions/41994
[deno_webgpu-crate-manifest]: https://github.com/gfx-rs/wgpu/tree/64a61ee5c69569bbb3db03563997e88a229eba17/deno_webgpu#deno_webgpu
[eat-the-world-meaning]: https://www.quora.com/What-did-Marc-Andreessen-mean-when-he-said-that-software-is-eating-the-world

WGPU also benefits from standard, portable behavior in the same way as web browsers. Because of this behavior, it's generally fairly easy to port over usage of WebGPU in JavaScript to WGPU. It is also what lets WGPU go full circle: WGPU can be an implementation of WebGPU on native targets, but _also_ it can use _other implementations of WebGPU_ as a backend in JavaScript when compiled to WASM. Therefore, the same dynamic applies: if WGPU's own behavior were significantly different, then WGPU and end users would be _sad, sad humans_ as soon as they discover places where their nice apps are breaking, right?

The answer is: yes, we _do_ have sad, sad humans that really want their WGPU code to work _everywhere_. As Firefox and others use WGPU to implement WebGPU, the above example of Firefox diverging from standard is, unfortunately, today's reality. It _mostly_ behaves the same as a standards-compliant WebGPU, but it still doesn't in many important ways. Of particular note is Naga, its implementation of the WebGPU Shader Language. Shaders are pretty much a black-and-white point of failure in GPU programming; if they don't compile, then you can't use the rest of the API! And yet, it's extremely easy to run into a case like that from <https://github.com/gfx-rs/wgpu/issues/4400>:

```wgsl
fn gimme_a_float() -> f32 {
  return 42; // fails in Naga, but standard WGSL happily converts to `f32`
}
```

We intend to continue making visible strides in converging with specifications for WebGPU and WGSL, as this release has. This is, unfortunately, one of the major reasons that WGPU has no plans to work hard at keeping a SemVer-stable interface for the foreseeable future; we have an entire platform of GPU programming functionality we have to catch up with, and SemVer stability is unfortunately in tension with that. So, for now, you're going to keep seeing major releases and breaking changes. Where possible, we'll try to make that painless, but compromises to do so don't always make sense with our limited resources.

This is also the last planned major version release of 2024; the next milestone is set for January 1st, 2025, according to our regular 12-week cadence (offset from the originally planned date of 2024-10-09 for _this_ release 😅). We'll see you next year!

### Contributor spotlight: @sagudev

This release, we'd like to spotlight the work of @sagudev, who has made significant contributions to the WGPU ecosystem this release. Among other things, they contributed a particularly notable feature where runtime-known indices are finally allowed for use with `const` array values. For example, this WGSL shader previously wasn't allowed:

```wgsl
const arr: array<u32, 4> = array(1, 2, 3, 4);

fn what_number_should_i_use(idx: u32) -> u32 {
  return arr[idx];
}
```

…but now it works! This is significant because this sort of shader rejection was one of the most impactful issues we are aware of for converging with the WGSL specification. There are more still to go—some of which we expect to even more drastically change how folks author shaders—but we suspect that many more will come in the next few releases, including with @sagudev's help.

We're excited for more of @sagudev's contributions via the Servo community. Oh, did we forget to mention that these contributions were motivated by their work on Servo? That's right, a _third_ well-known JavaScript runtime is now using WGPU to implement its WebGPU implementation. We're excited to support Servo to becoming another fully fledged browsing environment this way.

### Major Changes

In addition to the above spotlight, we have the following particularly interesting items to call out for this release:

#### `wgpu-core` is no longer generic over `wgpu-hal` backends

Dynamic dispatch between different backends has been moved from the user facing `wgpu` crate, to a new dynamic dispatch mechanism inside the backend abstraction layer `wgpu-hal`.

Whenever targeting more than a single backend (default on Windows & Linux) this leads to faster compile times and smaller binaries! This also solves a long standing issue with `cargo doc` failing to run for `wgpu-core`.

Benchmarking indicated that compute pass recording is slower as a consequence, whereas on render passes speed improvements have been observed. However, this effort simplifies many of the internals of the wgpu family of crates which we're hoping to build performance improvements upon in the future.

By @wumpf in [#6069](https://github.com/gfx-rs/wgpu/pull/6069), [#6099](https://github.com/gfx-rs/wgpu/pull/6099), [#6100](https://github.com/gfx-rs/wgpu/pull/6100).

#### `wgpu`'s resources no longer have `.global_id()` getters

`wgpu-core`'s internals no longer use nor need IDs and we are moving towards removing IDs completely. This is a step in that direction.

Current users of `.global_id()` are encouraged to make use of the `PartialEq`, `Eq`, `Hash`, `PartialOrd` and `Ord` traits that have now been implemented for `wgpu` resources.

By @teoxoy in [#6134](https://github.com/gfx-rs/wgpu/pull/6134).

#### `set_bind_group` now takes an `Option` for the bind group argument.

https://gpuweb.github.io/gpuweb/#programmable-passes-bind-groups specifies that bindGroup is nullable. This change is the start of implementing this part of the spec. Callers that specify a `Some()` value should have unchanged behavior. Handling of `None` values still needs to be implemented by backends.

For convenience, the `set_bind_group` on compute/render passes & encoders takes `impl Into<Option<&BindGroup>>`, so most code should still work the same.

By @bradwerth in [#6216](https://github.com/gfx-rs/wgpu/pull/6216).

#### `entry_point`s are now `Option`al

One of the changes in the WebGPU spec. (from [about this time last year][optional-entrypoint-in-spec] 😅) was to allow optional entry points in `GPUProgrammableStage`. In `wgpu`, this corresponds to a subset of fields in `FragmentState`, `VertexState`, and `ComputeState` as the `entry_point` member:

```wgsl
let render_pipeline = device.createRenderPipeline(wgpu::RenderPipelineDescriptor {
    module,
    entry_point: Some("cs_main"), // This is now `Option`al.
    // …
});

let compute_pipeline = device.createComputePipeline(wgpu::ComputePipelineDescriptor {
    module,
    entry_point: None, // This is now `Option`al.
    // …
});
```

When set to `None`, it's assumed that the shader only has a single entry point associated with the pipeline stage (i.e., `@compute`, `@fragment`, or `@vertex`). If there is not one and only one candidate entry point, then a validation error is returned. To continue the example, we might have written the above API usage with the following shader module:

```wgsl
// We can't use `entry_point: None` for compute pipelines with this module,
// because there are two `@compute` entry points.

@compute
fn cs_main() { /* … */ }

@compute
fn other_cs_main() { /* … */ }

// The following entry points _can_ be inferred from `entry_point: None` in a
// render pipeline, because they're the only `@vertex` and `@fragment` entry
// points:

@vertex
fn vs_main() { /* … */ }

@fragment
fn fs_main() { /* … */ }
```

[optional-entrypoint-in-spec]: https://github.com/gpuweb/gpuweb/issues/4342

#### WGPU's DX12 backend is now based on the `windows` crate ecosystem, instead of the `d3d12` crate

WGPU has retired the `d3d12` crate (based on `winapi`), and now uses the `windows` crate for interfacing with Windows. For many, this may not be a change that affects day-to-day work. However, for users who need to vet their dependencies, or who may vendor in dependencies, this may be a nontrivial migration.

By @MarijnS95 in [#6006](https://github.com/gfx-rs/wgpu/pull/6006).

### New Features

#### Wgpu

- Added initial acceleration structure and ray query support into wgpu. By @expenses @daniel-keitel @Vecvec @JMS55 @atlv24 in [#6291](https://github.com/gfx-rs/wgpu/pull/6291)

#### Naga

- Support constant evaluation for `firstLeadingBit` and `firstTrailingBit` numeric built-ins in WGSL. Front-ends that translate to these built-ins also benefit from constant evaluation. By @ErichDonGubler in [#5101](https://github.com/gfx-rs/wgpu/pull/5101).
- Add `first` and `either` sampling types for `@interpolate(flat, …)` in WGSL. By @ErichDonGubler in [#6181](https://github.com/gfx-rs/wgpu/pull/6181).
- Support for more atomic ops in the SPIR-V frontend. By @schell in [#5824](https://github.com/gfx-rs/wgpu/pull/5824).
- Support local `const` declarations in WGSL. By @sagudev in [#6156](https://github.com/gfx-rs/wgpu/pull/6156).
- Implemented `const_assert` in WGSL. By @sagudev in [#6198](https://github.com/gfx-rs/wgpu/pull/6198).
- Support polyfilling `inverse` in WGSL. By @chyyran in [#6385](https://github.com/gfx-rs/wgpu/pull/6385).
- Add base support for parsing `requires`, `enable`, and `diagnostic` directives. No extensions or diagnostic filters are yet supported, but diagnostics have improved dramatically. By @ErichDonGubler in [#6352](https://github.com/gfx-rs/wgpu/pull/6352), [#6424](https://github.com/gfx-rs/wgpu/pull/6424), [#6437](https://github.com/gfx-rs/wgpu/pull/6437).
- Include error chain information as a message and notes in shader compilation messages. By @ErichDonGubler in [#6436](https://github.com/gfx-rs/wgpu/pull/6436).
- Unify Naga CLI error output with the format of shader compilation messages. By @ErichDonGubler in [#6436](https://github.com/gfx-rs/wgpu/pull/6436).

#### General

- Add `VideoFrame` to `ExternalImageSource` enum. By @jprochazk in [#6170](https://github.com/gfx-rs/wgpu/pull/6170).
- Add `wgpu::util::new_instance_with_webgpu_detection` & `wgpu::util::is_browser_webgpu_supported` to make it easier to support WebGPU & WebGL in the same binary. By @wumpf in [#6371](https://github.com/gfx-rs/wgpu/pull/6371).

#### Vulkan

- Allow using [VK_GOOGLE_display_timing](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_GOOGLE_display_timing.html) unsafely with the `VULKAN_GOOGLE_DISPLAY_TIMING` feature. By @DJMcNab in [#6149](https://github.com/gfx-rs/wgpu/pull/6149).

#### Metal

- Implement `atomicCompareExchangeWeak`. By @AsherJingkongChen in [#6265](https://github.com/gfx-rs/wgpu/pull/6265).
- Unless an explicit `CAMetalLayer` is provided, surfaces now render to a sublayer. This improves resizing behavior, fixing glitches during on window resize. By @madsmtm in [#6107](https://github.com/gfx-rs/wgpu/pull/6107).

### Bug Fixes

- Fix incorrect hlsl image output type conversion. By @atlv24 in [#6123](https://github.com/gfx-rs/wgpu/pull/6123).

#### Naga

- SPIR-V frontend splats depth texture sample and load results. Fixes [issue #4551](https://github.com/gfx-rs/wgpu/issues/4551). By @schell in [#6384](https://github.com/gfx-rs/wgpu/pull/6384).
- Accept only `vec3` (not `vecN`) for the `cross` built-in. By @ErichDonGubler in [#6171](https://github.com/gfx-rs/wgpu/pull/6171).
- Configure `SourceLanguage` when enabling debug info in SPV-out. By @kvark in [#6256](https://github.com/gfx-rs/wgpu/pull/6256).
- Do not consider per-polygon and flat inputs subgroup uniform. By @magcius in [#6276](https://github.com/gfx-rs/wgpu/pull/6276).
- Validate all swizzle components are either color (rgba) or dimension (xyzw) in WGSL. By @sagudev in [#6187](https://github.com/gfx-rs/wgpu/pull/6187).
- Fix detection of shl overflows to detect arithmetic overflows. By @sagudev in [#6186](https://github.com/gfx-rs/wgpu/pull/6186).
- Fix type parameters to vec/mat type constructors to also support aliases. By @sagudev in [#6189](https://github.com/gfx-rs/wgpu/pull/6189).
- Accept global `var`s without explicit type. By @sagudev in [#6199](https://github.com/gfx-rs/wgpu/pull/6199).
- Fix handling of phony statements, so they are actually emitted. By @sagudev in [#6328](https://github.com/gfx-rs/wgpu/pull/6328).
- Added `gl_DrawID` to glsl and `DrawIndex` to spv. By @ChosenName in [#6325](https://github.com/gfx-rs/wgpu/pull/6325).
- Matrices can now be indexed by value (#4337), and indexing arrays by value no longer causes excessive spilling (#6358). By @jimblandy in [#6390](https://github.com/gfx-rs/wgpu/pull/6390).
- Add support for `textureQueryLevels` to the GLSL parser. By @magcius in [#6325](https://github.com/gfx-rs/wgpu/pull/6415).
- Fix unescaped identifiers in the Metal backend shader I/O structures causing shader miscompilation. By @ErichDonGubler in [#6438](https://github.com/gfx-rs/wgpu/pull/6438).

#### General

- If GL context creation fails retry with GLES. By @Rapdorian in [#5996](https://github.com/gfx-rs/wgpu/pull/5996).
- Bump MSRV for `d3d12`/`naga`/`wgpu-core`/`wgpu-hal`/`wgpu-types`' to 1.76. By @wumpf in [#6003](https://github.com/gfx-rs/wgpu/pull/6003).
- Print requested and supported usages on `UnsupportedUsage` error. By @VladasZ in [#6007](https://github.com/gfx-rs/wgpu/pull/6007).
- Deduplicate bind group layouts that are created from pipelines with "auto" layouts. By @teoxoy [#6049](https://github.com/gfx-rs/wgpu/pull/6049).
- Document `wgpu_hal` bounds-checking promises, and adapt `wgpu_core`'s lazy initialization logic to the slightly weaker-than-expected guarantees. By @jimblandy in [#6201](https://github.com/gfx-rs/wgpu/pull/6201).
- Raise validation error instead of panicking in `{Render,Compute}Pipeline::get_bind_group_layout` on native / WebGL. By @bgr360 in [#6280](https://github.com/gfx-rs/wgpu/pull/6280).
- **BREAKING**: Remove the last exposed C symbols in project, located in `wgpu_core::render::bundle::bundle_ffi`, to allow multiple versions of WGPU to compile together. By @ErichDonGubler in [#6272](https://github.com/gfx-rs/wgpu/pull/6272).
- Call `flush_mapped_ranges` when unmapping write-mapped buffers. By @teoxoy in [#6089](https://github.com/gfx-rs/wgpu/pull/6089).
- When mapping buffers for reading, mark buffers as initialized only when they have `MAP_WRITE` usage. By @teoxoy in [#6178](https://github.com/gfx-rs/wgpu/pull/6178).
- Add a separate pipeline constants error. By @teoxoy in [#6094](https://github.com/gfx-rs/wgpu/pull/6094).
- Ensure safety of indirect dispatch by injecting a compute shader that validates the content of the indirect buffer. By @teoxoy in [#5714](https://github.com/gfx-rs/wgpu/pull/5714).

#### GLES / OpenGL

- Fix GL debug message callbacks not being properly cleaned up (causing UB). By @Imberflur in [#6114](https://github.com/gfx-rs/wgpu/pull/6114).
- Fix calling `slice::from_raw_parts` with unaligned pointers in push constant handling. By @Imberflur in [#6341](https://github.com/gfx-rs/wgpu/pull/6341).
- Optimise fence checking when `Queue::submit` is called many times per frame. By @dinnerbone in [#6427](https://github.com/gfx-rs/wgpu/pull/6427).

#### WebGPU

- Fix JS `TypeError` exception in `Instance::request_adapter` when browser doesn't support WebGPU but `wgpu` not compiled with `webgl` support. By @bgr360 in [#6197](https://github.com/gfx-rs/wgpu/pull/6197).

#### Vulkan

- Avoid undefined behaviour with adversarial debug label. By @DJMcNab in [#6257](https://github.com/gfx-rs/wgpu/pull/6257).
- Add `.index_type(vk::IndexType::NONE_KHR)` when creating `AccelerationStructureGeometryTrianglesDataKHR` in the raytraced triangle example to prevent a validation error. By @Vecvec in [#6282](https://github.com/gfx-rs/wgpu/pull/6282).

### Changes

- `wgpu_hal::gles::Adapter::new_external` now requires the context to be current when dropping the adapter and related objects. By @Imberflur in [#6114](https://github.com/gfx-rs/wgpu/pull/6114).
- Reduce the amount of debug and trace logs emitted by wgpu-core and wgpu-hal. By @nical in [#6065](https://github.com/gfx-rs/wgpu/issues/6065).
- Rename `Rg11b10Float` to `Rg11b10Ufloat`. By @sagudev in [#6108](https://github.com/gfx-rs/wgpu/pull/6108).
- Invalidate the device when we encounter driver-induced device loss or on unexpected errors. By @teoxoy in [#6229](https://github.com/gfx-rs/wgpu/pull/6229).
- Make Vulkan error handling more robust. By @teoxoy in [#6119](https://github.com/gfx-rs/wgpu/pull/6119).
- Add bounds checking to Buffer slice method. By @beholdnec in [#6432](https://github.com/gfx-rs/wgpu/pull/6432).
- Replace `impl From<StorageFormat> for ScalarKind` with `impl From<StorageFormat> for Scalar` so that byte width is included. By @atlv24 in [#6451](https://github.com/gfx-rs/wgpu/pull/6451).

#### Internal

- Tracker simplifications. By @teoxoy in [#6073](https://github.com/gfx-rs/wgpu/pull/6073) & [#6088](https://github.com/gfx-rs/wgpu/pull/6088).
- D3D12 cleanup. By @teoxoy in [#6200](https://github.com/gfx-rs/wgpu/pull/6200).
- Use `ManuallyDrop` in remaining places. By @teoxoy in [#6092](https://github.com/gfx-rs/wgpu/pull/6092).
- Move out invalidity from the `Registry`. By @teoxoy in [#6243](https://github.com/gfx-rs/wgpu/pull/6243).
- Remove `backend` from ID. By @teoxoy in [#6263](https://github.com/gfx-rs/wgpu/pull/6263).

#### HAL

- Change the inconsistent `DropGuard` based API on Vulkan and GLES to a consistent, callback-based one. By @jerzywilczek in [#6164](https://github.com/gfx-rs/wgpu/pull/6164).

### Documentation

- Removed some OpenGL and Vulkan references from `wgpu-types` documentation. Fixed Storage texel types in examples. By @Nelarius in [#6271](https://github.com/gfx-rs/wgpu/pull/6271).
- Used `wgpu::include_wgsl!(…)` more in examples and tests. By @ErichDonGubler in [#6326](https://github.com/gfx-rs/wgpu/pull/6326).

### Dependency Updates

#### GLES

- Replace `winapi` code in WGL wrapper to use the `windows` crate. By @MarijnS95 in [#6006](https://github.com/gfx-rs/wgpu/pull/6006).
- Update `glutin` to `0.31` with `glutin-winit` crate. By @MarijnS95 in [#6150](https://github.com/gfx-rs/wgpu/pull/6150) and [#6176](https://github.com/gfx-rs/wgpu/pull/6176).
- Implement `Adapter::new_external()` for WGL (just like EGL) to import an external OpenGL ES context. By @MarijnS95 in [#6152](https://github.com/gfx-rs/wgpu/pull/6152).

#### DX12

- Replace `winapi` code to use the `windows` crate. By @MarijnS95 in [#5956](https://github.com/gfx-rs/wgpu/pull/5956) and [#6173](https://github.com/gfx-rs/wgpu/pull/6173).
- Get `num_workgroups` builtin working for indirect dispatches. By @teoxoy in [#5730](https://github.com/gfx-rs/wgpu/pull/5730).

#### HAL

- Update `parking_lot` to `0.12`. By @mahkoh in [#6287](https://github.com/gfx-rs/wgpu/pull/6287).

## v22.1.0 (2024-07-17)

This release includes `wgpu`, `wgpu-core` and `naga`. All other crates remain at 22.0.0.

### Added

#### Naga

- Added back implementations of PartialEq for more IR types. By @teoxoy in [#6045](https://github.com/gfx-rs/wgpu/pull/6045)

### Bug Fixes

#### General

- Fix profiling with `tracy`. By @waywardmonkeys in [#5988](https://github.com/gfx-rs/wgpu/pull/5988)
- Fix function for checking bind compatibility to error instead of panic. By @sagudev [#6012](https://github.com/gfx-rs/wgpu/pull/6012)
- Fix crash when dropping the surface after the device. By @wumpf in [#6052](https://github.com/gfx-rs/wgpu/pull/6052)
- Fix length of copy in `queue_write_texture`. By @teoxoy in [#6009](https://github.com/gfx-rs/wgpu/pull/6009)
- Fix error message that is thrown in create_render_pass to no longer say `compute_pass`. By @matthew-wong1 [#6041](https://github.com/gfx-rs/wgpu/pull/6041)
- As a workaround for [issue #4905](https://github.com/gfx-rs/wgpu/issues/4905), `wgpu-core` is undocumented unless `--cfg wgpu_core_doc` feature is enabled. By @kpreid in [#5987](https://github.com/gfx-rs/wgpu/pull/5987)

## 22.0.0 (2024-07-17)

### Overview

### Our first major version release!

For the first time ever, WGPU is being released with a major version (i.e., 22.* instead of 0.22.*)! Maintainership has decided to fully adhere to [Semantic Versioning](https://semver.org/)'s recommendations for versioning production software. According to [SemVer 2.0.0's Q&A about when to use 1.0.0 versions (and beyond)](https://semver.org/spec/v2.0.0.html#how-do-i-know-when-to-release-100):

> ### How do I know when to release 1.0.0?
>
> If your software is being used in production, it should probably already be 1.0.0. If you have a stable API on which users have come to depend, you should be 1.0.0. If you’re worrying a lot about backward compatibility, you should probably already be 1.0.0.

It is a well-known fact that WGPU has been used for applications and platforms already in production for years, at this point. We are often concerned with tracking breaking changes, and affecting these consumers' ability to ship. By releasing our first major version, we publicly acknowledge that this is the case. We encourage other projects in the Rust ecosystem to follow suit.

Note that while we start to use the major version number, WGPU is _not_ "going stable", as many Rust projects do. We anticipate many breaking changes before we fully comply with the WebGPU spec., which we expect to take a small number of years.

### Overview

A major ([pun intended](#our-first-major-version-release)) theme of this release is incremental improvement. Among the typically large set of bug fixes, new features, and other adjustments to WGPU by the many contributors listed below, @wumpf and @teoxoy have merged a series of many simplifications to WGPU's internals and, in one case, to the render and compute pass recording APIs. Many of these change WGPU to use atomically reference-counted resource tracking (i.e., `Arc<…>`), rather than using IDs to manage the lifetimes of platform-specific graphics resources in a registry of separate reference counts. This has led us to diagnose and fix many long-standing bugs, and net some neat performance improvements on the order of 40% or more of some workloads.

While the above is exciting, we acknowledge already finding and fixing some (easy-to-fix) regressions from the above work. If you migrate to WGPU 22 and encounter such bugs, please engage us in the issue tracker right away!

### Major Changes

#### Lifetime bounds on `wgpu::RenderPass` & `wgpu::ComputePass`

`wgpu::RenderPass` & `wgpu::ComputePass` recording methods (e.g. `wgpu::RenderPass:set_render_pipeline`) no longer impose a lifetime constraint to objects passed to a pass (like pipelines/buffers/bindgroups/query-sets etc.).

This means the following pattern works now as expected:

```rust
let mut pipelines: Vec<wgpu::RenderPipeline> = ...;
// ...
let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
cpass.set_pipeline(&pipelines[123]);
// Change pipeline container - this requires mutable access to `pipelines` while one of the pipelines is in use.
pipelines.push(/* ... */);
// Continue pass recording.
cpass.set_bindgroup(...);
```
Previously, a set pipeline (or other resource) had to outlive pass recording which often affected wider systems,
meaning that users needed to prove to the borrow checker that `Vec<wgpu::RenderPipeline>` (or similar constructs)
aren't accessed mutably for the duration of pass recording.


Furthermore, you can now opt out of `wgpu::RenderPass`/`wgpu::ComputePass`'s lifetime dependency on its parent `wgpu::CommandEncoder` using `wgpu::RenderPass::forget_lifetime`/`wgpu::ComputePass::forget_lifetime`:
```rust
fn independent_cpass<'enc>(encoder: &'enc mut wgpu::CommandEncoder) -> wgpu::ComputePass<'static> {
    let cpass: wgpu::ComputePass<'enc> = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
    cpass.forget_lifetime()
}
```
⚠️ As long as a `wgpu::RenderPass`/`wgpu::ComputePass` is pending for a given `wgpu::CommandEncoder`, creation of a compute or render pass is an error and invalidates the `wgpu::CommandEncoder`.
`forget_lifetime` can be very useful for library authors, but opens up an easy way for incorrect use, so use with care.
This method doesn't add any additional overhead and has no side effects on pass recording.

By @wumpf in [#5569](https://github.com/gfx-rs/wgpu/pull/5569), [#5575](https://github.com/gfx-rs/wgpu/pull/5575), [#5620](https://github.com/gfx-rs/wgpu/pull/5620), [#5768](https://github.com/gfx-rs/wgpu/pull/5768) (together with @kpreid), [#5671](https://github.com/gfx-rs/wgpu/pull/5671), [#5794](https://github.com/gfx-rs/wgpu/pull/5794), [#5884](https://github.com/gfx-rs/wgpu/pull/5884).

#### Querying shader compilation errors

Wgpu now supports querying [shader compilation info](https://www.w3.org/TR/webgpu/#dom-gpushadermodule-getcompilationinfo).

This allows you to get more structured information about compilation errors, warnings and info:

```rust
...
let lighting_shader = ctx.device.create_shader_module(include_wgsl!("lighting.wgsl"));
let compilation_info = lighting_shader.get_compilation_info().await;
for message in compilation_info
    .messages
    .iter()
    .filter(|m| m.message_type == wgpu::CompilationMessageType::Error)
{
    let line = message.location.map(|l| l.line_number).unwrap_or(1);
    println!("Compile error at line {line}");
}
```

By @stefnotch in [#5410](https://github.com/gfx-rs/wgpu/pull/5410)

#### 64 bit integer atomic support in shaders.

Add support for 64 bit integer atomic operations in shaders.

Add the following flags to `wgpu_types::Features`:

- `SHADER_INT64_ATOMIC_ALL_OPS` enables all atomic operations on `atomic<i64>` and
  `atomic<u64>` values.

- `SHADER_INT64_ATOMIC_MIN_MAX` is a subset of the above, enabling only
  `AtomicFunction::Min` and `AtomicFunction::Max` operations on `atomic<i64>` and
  `atomic<u64>` values in the `Storage` address space. These are the only 64-bit
  atomic operations available on Metal as of 3.1.

Add corresponding flags to `naga::valid::Capabilities`. These are supported by the
WGSL front end, and all Naga backends.

Platform support:

- On Direct3d 12, in `D3D12_FEATURE_DATA_D3D12_OPTIONS9`, if
  `AtomicInt64OnTypedResourceSupported` and `AtomicInt64OnGroupSharedSupported` are
  both available, then both wgpu features described above are available.

- On Metal, `SHADER_INT64_ATOMIC_MIN_MAX` is available on Apple9 hardware, and on
  hardware that advertises both Apple8 and Mac2 support. This also requires Metal
  Shading Language 2.4 or later. Metal does not yet support the more general
  `SHADER_INT64_ATOMIC_ALL_OPS`.

- On Vulkan, if the `VK_KHR_shader_atomic_int64` extension is available with both the
  `shader_buffer_int64_atomics` and `shader_shared_int64_atomics` features, then both
  wgpu features described above are available.

By @atlv24 in [#5383](https://github.com/gfx-rs/wgpu/pull/5383)

#### A compatible surface is now required for `request_adapter()` on WebGL2 + `enumerate_adapters()` is now native only.

When targeting WebGL2, it has always been the case that a surface had to be created before calling `request_adapter()`.
We now make this requirement explicit.

Validation was also added to prevent configuring the surface with a device that doesn't share the same underlying
WebGL2 context since this has never worked.

Calling `enumerate_adapters()` when targeting WebGPU used to return an empty `Vec` and since we now require users
to pass a compatible surface when targeting WebGL2, having `enumerate_adapters()` doesn't make sense.

By @teoxoy in [#5901](https://github.com/gfx-rs/wgpu/pull/5901)

### New features

#### General

- Added `as_hal` for `Buffer` to access wgpu created buffers form wgpu-hal. By @JasondeWolff in [#5724](https://github.com/gfx-rs/wgpu/pull/5724)
- `include_wgsl!` is now callable in const contexts by @9SMTM6 in [#5872](https://github.com/gfx-rs/wgpu/pull/5872)
- Added memory allocation hints to `DeviceDescriptor` by @nical in [#5875](https://github.com/gfx-rs/wgpu/pull/5875)
    - `MemoryHints::Performance`, the default, favors performance over memory usage and will likely cause large amounts of VRAM to be allocated up-front. This hint is typically good for games.
    - `MemoryHints::MemoryUsage` favors memory usage over performance. This hint is typically useful for smaller applications or UI libraries.
    - `MemoryHints::Manual` allows the user to specify parameters for the underlying GPU memory allocator. These parameters are subject to change.
    - These hints may be ignored by some backends. Currently only the Vulkan and D3D12 backends take them into account.
- Add `HTMLImageElement` and `ImageData` as external source for copying images. By @Valaphee in [#5668](https://github.com/gfx-rs/wgpu/pull/5668)

#### Naga

- Added -D, --defines option to naga CLI to define preprocessor macros by @theomonnom in [#5859](https://github.com/gfx-rs/wgpu/pull/5859)
- Added type upgrades to SPIR-V atomic support. Added related infrastructure. Tracking issue is [here](https://github.com/gfx-rs/wgpu/issues/4489). By @schell in [#5775](https://github.com/gfx-rs/wgpu/pull/5775).
- Implement `WGSL`'s `unpack4xI8`,`unpack4xU8`,`pack4xI8` and `pack4xU8`. By @VlaDexa in [#5424](https://github.com/gfx-rs/wgpu/pull/5424)
- Began work adding support for atomics to the SPIR-V frontend. Tracking issue is [here](https://github.com/gfx-rs/wgpu/issues/4489). By @schell in [#5702](https://github.com/gfx-rs/wgpu/pull/5702).
- In hlsl-out, allow passing information about the fragment entry point to omit vertex outputs that are not in the fragment inputs. By @Imberflur in [#5531](https://github.com/gfx-rs/wgpu/pull/5531)
- In spv-out, allow passing `acceleration_structure` as a function argument. By @kvark in [#5961](https://github.com/gfx-rs/wgpu/pull/5961)

  ```diff
  let writer: naga::back::hlsl::Writer = /* ... */;
  -writer.write(&module, &module_info);
  +writer.write(&module, &module_info, None);
  ```
- HLSL & MSL output can now be added conditionally on the target via the `msl-out-if-target-apple` and `hlsl-out-if-target-windows` features. This is used in wgpu-hal to no longer compile with MSL output when `metal` is enabled & MacOS isn't targeted and no longer compile with HLSL output when `dx12` is enabled & Windows isn't targeted. By @wumpf in [#5919](https://github.com/gfx-rs/wgpu/pull/5919)

#### Vulkan

- Added a `PipelineCache` resource to allow using Vulkan pipeline caches. By @DJMcNab in [#5319](https://github.com/gfx-rs/wgpu/pull/5319)

#### WebGPU

- Added support for pipeline-overridable constants to the WebGPU backend by @DouglasDwyer in [#5688](https://github.com/gfx-rs/wgpu/pull/5688)

### Changes

#### General

- Unconsumed vertex outputs are now always allowed. Removed `StageError::InputNotConsumed`, `Features::SHADER_UNUSED_VERTEX_OUTPUT`, and associated validation. By @Imberflur in [#5531](https://github.com/gfx-rs/wgpu/pull/5531)
- Avoid introducing spurious features for optional dependencies. By @bjorn3 in [#5691](https://github.com/gfx-rs/wgpu/pull/5691)
- `wgpu::Error` is now `Sync`, making it possible to be wrapped in `anyhow::Error` or `eyre::Report`. By @nolanderc in [#5820](https://github.com/gfx-rs/wgpu/pull/5820)
- Added benchmark suite. By @cwfitzgerald in [#5694](https://github.com/gfx-rs/wgpu/pull/5694), compute passes by @wumpf in [#5767](https://github.com/gfx-rs/wgpu/pull/5767)
- Improve performance of `.submit()` by 39-64% (`.submit()` + `.poll()` by 22-32%). By @teoxoy in [#5910](https://github.com/gfx-rs/wgpu/pull/5910)
- The `trace` wgpu feature has been temporarily removed. By @teoxoy in [#5975](https://github.com/gfx-rs/wgpu/pull/5975)

#### Metal
- Removed the `link` Cargo feature.

  This was used to allow weakly linking frameworks. This can be achieved with putting something like the following in your `.cargo/config.toml` instead:
  ```toml
  [target.'cfg(target_vendor = "apple")']
  rustflags = ["-C", "link-args=-weak_framework Metal -weak_framework QuartzCore -weak_framework CoreGraphics"]
  ```
  By @madsmtm in [#5752](https://github.com/gfx-rs/wgpu/pull/5752)

### Bug Fixes

#### General

- Ensure render pipelines have at least 1 target. By @ErichDonGubler in [#5715](https://github.com/gfx-rs/wgpu/pull/5715)
- `wgpu::ComputePass` now internally takes ownership of `QuerySet` for both `wgpu::ComputePassTimestampWrites` as well as timestamp writes and statistics query, fixing crashes when destroying `QuerySet` before ending the pass. By @wumpf in [#5671](https://github.com/gfx-rs/wgpu/pull/5671)
- Validate resources passed during compute pass recording for mismatching device. By @wumpf in [#5779](https://github.com/gfx-rs/wgpu/pull/5779)
- Fix staging buffers being destroyed too early. By @teoxoy in [#5910](https://github.com/gfx-rs/wgpu/pull/5910)
- Fix attachment byte cost validation panicking with native only formats. By @teoxoy in [#5934](https://github.com/gfx-rs/wgpu/pull/5934)
- [wgpu] Fix leaks from auto layout pipelines. By @teoxoy in [#5971](https://github.com/gfx-rs/wgpu/pull/5971)
- [wgpu-core] Fix length of copy in `queue_write_texture` (causing UB). By @teoxoy in [#5973](https://github.com/gfx-rs/wgpu/pull/5973)
- Add missing same device checks. By @teoxoy in [#5980](https://github.com/gfx-rs/wgpu/pull/5980)

#### GLES / OpenGL

- Fix `ClearColorF`, `ClearColorU` and `ClearColorI` commands being issued before `SetDrawColorBuffers` [#5666](https://github.com/gfx-rs/wgpu/pull/5666)
- Replace `glClear` with `glClearBufferF` because `glDrawBuffers` requires that the ith buffer must be `COLOR_ATTACHMENTi` or `NONE` [#5666](https://github.com/gfx-rs/wgpu/pull/5666)
- Return the unmodified version in driver_info. By @Valaphee in [#5753](https://github.com/gfx-rs/wgpu/pull/5753)

#### Naga

- In spv-out don't decorate a `BindingArray`'s type with `Block` if the type is a struct with a runtime array by @Vecvec in [#5776](https://github.com/gfx-rs/wgpu/pull/5776)
- Add `packed` as a keyword for GLSL by @kjarosh in [#5855](https://github.com/gfx-rs/wgpu/pull/5855)

## v0.20.2 (2024-06-12)

This release force-bumps transitive dependencies of `wgpu` on `wgpu-core` and `wgpu-hal` to 0.21.1, to resolve some undefined behavior observable in the DX12 backend after upgrading to Rust 1.79 or later.

### Bug Fixes

#### General

* Fix a `CommandBuffer` leak. By @cwfitzgerald and @nical in [#5141](https://github.com/gfx-rs/wgpu/pull/5141)

#### DX12

* Do not feed `&""` to `D3DCompile`, by @workingjubilee in [#5812](https://github.com/gfx-rs/wgpu/issues/5812).

## v0.20.1 (2024-06-12)

This release included v0.21.0 of `wgpu-core` and `wgpu-hal`, due to breaking changes needed to solve vulkan validation issues.

### Bug Fixes

This release fixes the validation errors whenever a surface is used with the vulkan backend. By @cwfitzgerald in [#5681](https://github.com/gfx-rs/wgpu/pull/5681).

#### General

- Clean up weak references to texture views and bind groups to prevent memory leaks. By @xiaopengli89 in [#5595](https://github.com/gfx-rs/wgpu/pull/5595).
- Fix segfault on exit is queue & device are dropped before surface. By @sagudev in [#5640](https://github.com/gfx-rs/wgpu/pull/5640).

#### Metal

- Fix unrecognized selector crash on iOS 12. By @vladasz in [#5744](https://github.com/gfx-rs/wgpu/pull/5744).

#### Vulkan

- Fix enablement of subgroup ops extension on Vulkan devices that don't support Vulkan 1.3. By @cwfitzgerald in [#5624](https://github.com/gfx-rs/wgpu/pull/5624).


#### GLES / OpenGL

-  Fix regression on OpenGL (EGL) where non-sRGB still used sRGB [#5642](https://github.com/gfx-rs/wgpu/pull/5642)

#### Naga

- Work around shader consumers that have bugs handling `switch` statements with a single body for all cases. These are now written as `do {} while(false);` loops in hlsl-out and glsl-out. By @Imberflur in [#5654](https://github.com/gfx-rs/wgpu/pull/5654)
- In hlsl-out, defer `continue` statements in switches by setting a flag and breaking from the switch. This allows such constructs to work with FXC which does not support `continue` within a switch. By @Imberflur in [#5654](https://github.com/gfx-rs/wgpu/pull/5654)

## v0.20.0 (2024-04-28)

### Major Changes

#### Pipeline overridable constants

Wgpu supports now [pipeline-overridable constants](https://www.w3.org/TR/webgpu/#dom-gpuprogrammablestage-constants)

This allows you to define constants in wgsl like this:
```rust
override some_factor: f32 = 42.1337; // Specifies a default of 42.1337 if it's not set.
```
And then set them at runtime like so on your pipeline consuming this shader:
```rust
// ...
fragment: Some(wgpu::FragmentState {
    compilation_options: wgpu::PipelineCompilationOptions {
        constants: &[("some_factor".to_owned(), 0.1234)].into(), // Sets `some_factor` to 0.1234.
        ..Default::default()
    },
    // ...
}),
// ...
```

By @teoxoy & @jimblandy in [#5500](https://github.com/gfx-rs/wgpu/pull/5500)

#### Changed feature requirements for timestamps

Due to a specification change `write_timestamp` is no longer supported on WebGPU.
`wgpu::CommandEncoder::write_timestamp` requires now the new `wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS` feature which is available on all native backends but not on WebGPU.

By @wumpf in [#5188](https://github.com/gfx-rs/wgpu/pull/5188)


#### Wgsl const evaluation for many more built-ins

Many numeric built-ins have had a constant evaluation implementation added for them, which allows them to be used in a `const` context:

`abs`, `acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`, `cos`, `cosh`, `round`, `saturate`, `sin`, `sinh`, `sqrt`, `step`, `tan`, `tanh`, `ceil`, `countLeadingZeros`, `countOneBits`, `countTrailingZeros`, `degrees`, `exp`, `exp2`, `floor`, `fract`, `fma`, `inverseSqrt`, `log`, `log2`, `max`, `min`, `radians`, `reverseBits`, `sign`, `trunc`

By @ErichDonGubler in [#4879](https://github.com/gfx-rs/wgpu/pull/4879), [#5098](https://github.com/gfx-rs/wgpu/pull/5098)

#### New **native-only** wgsl features

##### Subgroup operations

The following subgroup operations are available in wgsl now:

`subgroupBallot`, `subgroupAll`, `subgroupAny`, `subgroupAdd`, `subgroupMul`, `subgroupMin`, `subgroupMax`, `subgroupAnd`, `subgroupOr`, `subgroupXor`, `subgroupExclusiveAdd`, `subgroupExclusiveMul`, `subgroupInclusiveAdd`, `subgroupInclusiveMul`, `subgroupBroadcastFirst`, `subgroupBroadcast`, `subgroupShuffle`, `subgroupShuffleDown`, `subgroupShuffleUp`, `subgroupShuffleXor`


Availability is governed by the following feature flags:
* `wgpu::Features::SUBGROUP` for all operations except `subgroupBarrier` in fragment & compute, supported on Vulkan, DX12 and Metal.
* `wgpu::Features::SUBGROUP_VERTEX`, for all operations except `subgroupBarrier` general operations in  vertex shaders, supported on Vulkan
* `wgpu::Features::SUBGROUP_BARRIER`, for support of the `subgroupBarrier` operation, supported on Vulkan & Metal

Note that there currently [some differences](https://github.com/gfx-rs/wgpu/issues/5555) between wgpu's native-only implementation and the [open WebGPU proposal](https://github.com/gpuweb/gpuweb/blob/main/proposals/subgroups.md).

By @exrook and @lichtso in [#5301](https://github.com/gfx-rs/wgpu/pull/5301)

##### Signed and unsigned 64 bit integer support in shaders.

`wgpu::Features::SHADER_INT64` enables 64 bit integer signed and unsigned integer variables in wgsl (`i64` and `u64` respectively).
Supported on Vulkan, DX12 (requires DXC) and Metal (with MSL 2.3+ support).

By @atlv24 and @cwfitzgerald in [#5154](https://github.com/gfx-rs/wgpu/pull/5154)

### New features

#### General

- Implemented the `Unorm10_10_10_2` VertexFormat by @McMackety in [#5477](https://github.com/gfx-rs/wgpu/pull/5477)
- `wgpu-types`'s `trace` and `replay` features have been replaced by the `serde` feature. By @KirmesBude in [#5149](https://github.com/gfx-rs/wgpu/pull/5149)
- `wgpu-core`'s `serial-pass` feature has been removed. Use `serde` instead. By @KirmesBude in [#5149](https://github.com/gfx-rs/wgpu/pull/5149)
- Added `InstanceFlags::GPU_BASED_VALIDATION`, which enables GPU-based validation for shaders. This is currently only supported on the DX12 and Vulkan backends; other platforms ignore this flag, for now. By @ErichDonGubler in [#5146](https://github.com/gfx-rs/wgpu/pull/5146), [#5046](https://github.com/gfx-rs/wgpu/pull/5046).
  - When set, this flag implies `InstanceFlags::VALIDATION`.
  - This has been added to the set of flags set by `InstanceFlags::advanced_debugging`. Since the overhead is potentially very large, the flag is not enabled by default in debug builds when using `InstanceFlags::from_build_config`.
  - As with other instance flags, this flag can be changed in calls to `InstanceFlags::with_env` with the new `WGPU_GPU_BASED_VALIDATION` environment variable.
- `wgpu::Instance` can now report which `wgpu::Backends` are available based on the build configuration. By @wumpf [#5167](https://github.com/gfx-rs/wgpu/pull/5167)
  ```diff
  -wgpu::Instance::any_backend_feature_enabled()
  +!wgpu::Instance::enabled_backend_features().is_empty()
  ```
- Breaking change: [`wgpu_core::pipeline::ProgrammableStageDescriptor`](https://docs.rs/wgpu-core/latest/wgpu_core/pipeline/struct.ProgrammableStageDescriptor.html#structfield.entry_point) is now optional. By @ErichDonGubler in [#5305](https://github.com/gfx-rs/wgpu/pull/5305).
- `Features::downlevel{_webgl2,}_features` was made const by @MultisampledNight in [#5343](https://github.com/gfx-rs/wgpu/pull/5343)
- Breaking change: [`wgpu_core::pipeline::ShaderError`](https://docs.rs/wgpu-core/latest/wgpu_core/pipeline/struct.ShaderError.html) has been moved to `naga`. By @stefnotch in [#5410](https://github.com/gfx-rs/wgpu/pull/5410)
- More as_hal methods and improvements by @JMS55 in [#5452](https://github.com/gfx-rs/wgpu/pull/5452)
  - Added `wgpu::CommandEncoder::as_hal_mut`
  - Added `wgpu::TextureView::as_hal`
  - `wgpu::Texture::as_hal` now returns a user-defined type to match the other as_hal functions

#### Naga

- Allow user to select which MSL version to use via `--metal-version` with Naga CLI. By @pcleavelin in [#5392](https://github.com/gfx-rs/wgpu/pull/5392)
- Support `arrayLength` for runtime-sized arrays inside binding arrays (for WGSL input and SPIR-V output). By @kvark in [#5428](https://github.com/gfx-rs/wgpu/pull/5428)
- Added `--shader-stage` and `--input-kind` options to naga-cli for specifying vertex/fragment/compute shaders, and frontend. by @ratmice in [#5411](https://github.com/gfx-rs/wgpu/pull/5411)
- Added a `create_validator` function to wgpu_core `Device` to create naga `Validator`s. By @atlv24 [#5606](https://github.com/gfx-rs/wgpu/pull/5606)

#### WebGPU

- Implement the `device_set_device_lost_callback` method for `ContextWebGpu`. By @suti in [#5438](https://github.com/gfx-rs/wgpu/pull/5438)
- Add support for storage texture access modes `ReadOnly` and `ReadWrite`. By @JolifantoBambla in [#5434](https://github.com/gfx-rs/wgpu/pull/5434)

#### GLES / OpenGL

- Log an error when GLES texture format heuristics fail. By @PolyMeilex in [#5266](https://github.com/gfx-rs/wgpu/issues/5266)
- Cache the sample count to keep `get_texture_format_features` cheap. By @Dinnerbone in [#5346](https://github.com/gfx-rs/wgpu/pull/5346)
- Mark `DEPTH32FLOAT_STENCIL8` as supported in GLES. By @Dinnerbone in [#5370](https://github.com/gfx-rs/wgpu/pull/5370)
- Desktop GL now also supports `TEXTURE_COMPRESSION_ETC2`. By @Valaphee in [#5568](https://github.com/gfx-rs/wgpu/pull/5568)
- Don't create a program for shader-clearing if that workaround isn't required. By @Dinnerbone in [#5348](https://github.com/gfx-rs/wgpu/pull/5348).
- OpenGL will now be preferred over OpenGL ES on EGL, making it consistent with WGL. By @valaphee in [#5482](https://github.com/gfx-rs/wgpu/pull/5482)
- Fill out `driver` and `driver_info`, with the OpenGL flavor and version, similar to Vulkan. By @valaphee in [#5482](https://github.com/gfx-rs/wgpu/pull/5482)

#### Metal

- Metal 3.0 and 3.1 detection. By @atlv24 in [#5497](https://github.com/gfx-rs/wgpu/pull/5497)

#### DX12

- Shader Model 6.1-6.7 detection. By @atlv24 in [#5498](https://github.com/gfx-rs/wgpu/pull/5498)

### Other performance improvements

- Simplify and speed up the allocation of internal IDs. By @nical in [#5229](https://github.com/gfx-rs/wgpu/pull/5229)
- Use memory pooling for UsageScopes to avoid frequent large allocations. by @robtfm in [#5414](https://github.com/gfx-rs/wgpu/pull/5414)
- Eager release of GPU resources comes from device.trackers. By @bradwerth in [#5075](https://github.com/gfx-rs/wgpu/pull/5075)
- Support disabling zero-initialization of workgroup local memory in compute shaders. By @DJMcNab in [#5508](https://github.com/gfx-rs/wgpu/pull/5508)

### Documentation

- Improved `wgpu_hal` documentation. By @jimblandy in [#5516](https://github.com/gfx-rs/wgpu/pull/5516), [#5524](https://github.com/gfx-rs/wgpu/pull/5524), [#5562](https://github.com/gfx-rs/wgpu/pull/5562), [#5563](https://github.com/gfx-rs/wgpu/pull/5563), [#5566](https://github.com/gfx-rs/wgpu/pull/5566), [#5617](https://github.com/gfx-rs/wgpu/pull/5617), [#5618](https://github.com/gfx-rs/wgpu/pull/5618)
- Add mention of primitive restart in the description of `PrimitiveState::strip_index_format`. By @cpsdqs in [#5350](https://github.com/gfx-rs/wgpu/pull/5350)
- Document and tweak precise behaviour of `SourceLocation`. By @stefnotch in [#5386](https://github.com/gfx-rs/wgpu/pull/5386) and [#5410](https://github.com/gfx-rs/wgpu/pull/5410)
- Give short example of WGSL `push_constant` syntax. By @waywardmonkeys in [#5393](https://github.com/gfx-rs/wgpu/pull/5393)
- Fix incorrect documentation of `Limits::max_compute_workgroup_storage_size` default value. By @atlv24 in [#5601](https://github.com/gfx-rs/wgpu/pull/5601)

### Bug Fixes

#### General
- Fix `serde` feature not compiling for `wgpu-types`. By @KirmesBude in [#5149](https://github.com/gfx-rs/wgpu/pull/5149)
- Fix the validation of vertex and index ranges. By @nical in [#5144](https://github.com/gfx-rs/wgpu/pull/5144) and [#5156](https://github.com/gfx-rs/wgpu/pull/5156)
- Fix panic when creating a surface while no backend is available. By @wumpf [#5166](https://github.com/gfx-rs/wgpu/pull/5166)
- Correctly compute minimum buffer size for array-typed `storage` and `uniform` vars. By @jimblandy [#5222](https://github.com/gfx-rs/wgpu/pull/5222)
- Fix timeout when presenting a surface where no work has been done. By @waywardmonkeys in [#5200](https://github.com/gfx-rs/wgpu/pull/5200)
- Fix registry leaks with de-duplicated resources. By @nical in [#5244](https://github.com/gfx-rs/wgpu/pull/5244)
- Fix linking when targeting android. By @ashdnazg in [#5326](https://github.com/gfx-rs/wgpu/pull/5326).
- Failing to set the device lost closure will call the closure before returning. By @bradwerth in [#5358](https://github.com/gfx-rs/wgpu/pull/5358).
- Fix deadlocks caused by recursive read-write lock acquisitions [#5426](https://github.com/gfx-rs/wgpu/pull/5426).
- Remove exposed C symbols (`extern "C"` + [no_mangle]) from RenderPass & ComputePass recording. By @wumpf in [#5409](https://github.com/gfx-rs/wgpu/pull/5409).
- Fix surfaces being only compatible with first backend enabled on an instance, causing failures when manually specifying an adapter. By @Wumpf in [#5535](https://github.com/gfx-rs/wgpu/pull/5535).

#### Naga

- In spv-in, remove unnecessary "gl_PerVertex" name check so unused builtins will always be skipped. Prevents validation errors caused by capability requirements of these builtins [#4915](https://github.com/gfx-rs/wgpu/issues/4915). By @Imberflur in [#5227](https://github.com/gfx-rs/wgpu/pull/5227).
- In spv-out, check for acceleration and ray-query types when enabling ray-query extension to prevent validation error. By @Vecvec in [#5463](https://github.com/gfx-rs/wgpu/pull/5463)
- Add a limit for curly brace nesting in WGSL parsing, plus a note about stack size requirements. By @ErichDonGubler in [#5447](https://github.com/gfx-rs/wgpu/pull/5447).
- In hlsl-out, fix accesses on zero value expressions by generating helper functions for `Expression::ZeroValue`. By @Imberflur in [#5587](https://github.com/gfx-rs/wgpu/pull/5587).
- Fix behavior of `extractBits` and `insertBits` when `offset + count` overflows the bit width. By @cwfitzgerald in [#5305](https://github.com/gfx-rs/wgpu/pull/5305)
- Fix behavior of integer `clamp` when `min` argument > `max` argument. By @cwfitzgerald in [#5300](https://github.com/gfx-rs/wgpu/pull/5300).
- Fix `TypeInner::scalar_width` to be consistent with the rest of the codebase and return values in bytes not bits. By @atlv24 in [#5532](https://github.com/gfx-rs/wgpu/pull/5532).

#### GLES / OpenGL

- GLSL 410 does not support layout(binding = ...), enable only for GLSL 420. By @bes in [#5357](https://github.com/gfx-rs/wgpu/pull/5357)
- Fixes for being able to use an OpenGL 4.1 core context provided by macOS with wgpu. By @bes in [#5331](https://github.com/gfx-rs/wgpu/pull/5331).
- Fix crash when holding multiple devices on wayland/surfaceless. By @ashdnazg in [#5351](https://github.com/gfx-rs/wgpu/pull/5351).
- Fix `first_instance` getting ignored in draw indexed when `ARB_shader_draw_parameters` feature is present and `base_vertex` is 0. By @valaphee in [#5482](https://github.com/gfx-rs/wgpu/pull/5482)

#### Vulkan

- Set object labels when the DEBUG flag is set, even if the VALIDATION flag is disabled. By @DJMcNab in [#5345](https://github.com/gfx-rs/wgpu/pull/5345).
- Add safety check to `wgpu_hal::vulkan::CommandEncoder` to make sure `discard_encoding` is not called in the closed state. By @villuna in [#5557](https://github.com/gfx-rs/wgpu/pull/5557)
- Fix SPIR-V type capability requests to not depend on `LocalType` caching. By @atlv24 in [#5590](https://github.com/gfx-rs/wgpu/pull/5590)
- Upgrade `ash` to `0.38`. By @MarijnS95 in [#5504](https://github.com/gfx-rs/wgpu/pull/5504).

#### Tests

- Fix intermittent crashes on Linux in the `multithreaded_compute` test. By @jimblandy in [#5129](https://github.com/gfx-rs/wgpu/pull/5129).
- Refactor tests to read feature flags by name instead of a hardcoded hexadecimal u64. By @atlv24 in [#5155](https://github.com/gfx-rs/wgpu/pull/5155).
- Add test that verifies that we can drop the queue before using the device to create a command encoder. By @Davidster in [#5211](https://github.com/gfx-rs/wgpu/pull/5211)

## 0.19.5 (2024-07-16)

This release only releases `wgpu-hal` 0.19.5, which contains an important fix
for DX12.

### Bug Fixes

#### DX12

- Do not feed `&""` to `D3DCompile`, by @workingjubilee in [#5812](https://github.com/gfx-rs/wgpu/issues/5812), backported by @Elabajaba in [#5833](https://github.com/gfx-rs/wgpu/pull/5833).

## v0.19.4 (2024-04-17)

### Bug Fixes

#### General

- Don't depend on bind group and bind group layout entry order in backends. This caused incorrect severely incorrect command execution and, in some cases, crashes. By @ErichDonGubler in [#5421](https://github.com/gfx-rs/wgpu/pull/5421).
- Properly clean up all write_buffer/texture temporary resources. By @robtfm in [#5413](https://github.com/gfx-rs/wgpu/pull/5413).
- Fix deadlock in certain situations when mapping buffers using `wgpu-profiler`. By @cwfitzgerald in [#5517](https://github.com/gfx-rs/wgpu/pull/5517)

#### WebGPU
- Correctly pass through timestamp queries to WebGPU. By @cwfitzgerald in [#5527](https://github.com/gfx-rs/wgpu/pull/5527).

## v0.19.3 (2024-03-01)

This release includes `wgpu`, `wgpu-core`, and `wgpu-hal`. All other crates are unchanged.

### Major Changes

#### Vendored WebGPU Bindings from `web_sys`

**`--cfg=web_sys_unstable_apis` is no longer needed in your `RUSTFLAGS` to compile for WebGPU!!!**

While WebGPU's javascript api is stable in the browsers, the `web_sys` bindings for WebGPU are still improving. As such they are hidden behind the special cfg `--cfg=web_sys_unstable_apis` and are not available by default. Everyone who wanted to use our WebGPU backend needed to enable this cfg in their `RUSTFLAGS`. This was very inconvenient and made it hard to use WebGPU, especially when WebGPU is enabled by default. Additionally, the unstable APIs don't adhere to semver, so there were repeated breakages.

To combat this problem we have decided to vendor the `web_sys` bindings for WebGPU within the crate. Notably we are not forking the bindings, merely vendoring, so any improvements we make to the bindings will be contributed directly to upstream `web_sys`.

By @cwfitzgerald in [#5325](https://github.com/gfx-rs/wgpu/pull/5325).

### Bug Fixes

#### General

- Fix an issue where command encoders weren't properly freed if an error occurred during command encoding. By @ErichDonGubler in [#5251](https://github.com/gfx-rs/wgpu/pull/5251).
- Fix incorrect validation causing all indexed draws on render bundles to fail. By @wumpf in [#5430](https://github.com/gfx-rs/wgpu/pull/5340).

#### Android
- Fix linking error when targeting android without `winit`. By @ashdnazg in [#5326](https://github.com/gfx-rs/wgpu/pull/5326).


## v0.19.2 (2024-02-29)

This release includes `wgpu`, `wgpu-core`, `wgpu-hal`, `wgpu-types`, and `naga`. All other crates are unchanged.

### Added/New Features

#### General
- `wgpu::Id` now implements `PartialOrd`/`Ord` allowing it to be put in `BTreeMap`s. By @cwfitzgerald and @9291Sam in [#5176](https://github.com/gfx-rs/wgpu/pull/5176)

#### OpenGL
- Log an error when OpenGL texture format heuristics fail. By @PolyMeilex in [#5266](https://github.com/gfx-rs/wgpu/issues/5266)

#### `wgsl-out`
- Learned to generate acceleration structure types. By @JMS55 in [#5261](https://github.com/gfx-rs/wgpu/pull/5261)

### Documentation
- Fix link in `wgpu::Instance::create_surface` documentation. By @HexoKnight in [#5280](https://github.com/gfx-rs/wgpu/pull/5280).
- Fix typo in `wgpu::CommandEncoder::clear_buffer` documentation. By @PWhiddy in [#5281](https://github.com/gfx-rs/wgpu/pull/5281).
- `Surface` configuration incorrectly claimed that `wgpu::Instance::create_surface` was unsafe. By @hackaugusto in [#5265](https://github.com/gfx-rs/wgpu/pull/5265).

### Bug Fixes

#### General
- Device lost callbacks are invoked when replaced and when global is dropped. By @bradwerth in [#5168](https://github.com/gfx-rs/wgpu/pull/5168)
- Fix performance regression when allocating a large amount of resources of the same type. By @nical in [#5229](https://github.com/gfx-rs/wgpu/pull/5229)
- Fix docs.rs wasm32 builds. By @cwfitzgerald in [#5310](https://github.com/gfx-rs/wgpu/pull/5310)
- Improve error message when binding count limit hit. By @hackaugusto in [#5298](https://github.com/gfx-rs/wgpu/pull/5298)
- Remove an unnecessary `clone` during GLSL shader ingestion. By @a1phyr in [#5118](https://github.com/gfx-rs/wgpu/pull/5118).
- Fix missing validation for `Device::clear_buffer` where `offset + size > buffer.size` was not checked when `size` was omitted. By @ErichDonGubler in [#5282](https://github.com/gfx-rs/wgpu/pull/5282).

#### DX12
- Fix `panic!` when dropping `Instance` without `InstanceFlags::VALIDATION`. By @hakolao in [#5134](https://github.com/gfx-rs/wgpu/pull/5134)

#### OpenGL
- Fix internal format for the `Etc2Rgba8Unorm` format. By @andristarr in [#5178](https://github.com/gfx-rs/wgpu/pull/5178)
- Try to load `libX11.so.6` in addition to `libX11.so` on linux. [#5307](https://github.com/gfx-rs/wgpu/pull/5307)
- Make use of `GL_EXT_texture_shadow_lod` to support sampling a cube depth texture with an explicit LOD. By @cmrschwarz in #[5171](https://github.com/gfx-rs/wgpu/pull/5171).

#### `glsl-in`

- Fix code generation from nested loops. By @cwfitzgerald and @teoxoy in [#5311](https://github.com/gfx-rs/wgpu/pull/5311)


## v0.19.1 (2024-01-22)

This release includes `wgpu` and `wgpu-hal`. The rest of the crates are unchanged since 0.19.0.

### Bug Fixes

#### DX12

- Properly register all swapchain buffers to prevent error on surface present. By @dtzxporter in [#5091](https://github.com/gfx-rs/wgpu/pull/5091)
- Check for extra null states when creating resources. By @nical in [#5096](https://github.com/gfx-rs/wgpu/pull/5096)
- Fix depth-only and stencil-only views causing crashes. By @teoxoy in [#5100](https://github.com/gfx-rs/wgpu/pull/5100)

#### OpenGL

- In Surface::configure and Surface::present on Windows, fix the current GL context not being unset when releasing the lock that guards access to making the context current. This was causing other threads to panic when trying to make the context current. By @Imberflur in [#5087](https://github.com/gfx-rs/wgpu/pull/5087).

#### WebGPU

- Improve error message when compiling WebGPU backend on wasm without the `web_sys_unstable_apis` set. By @rukai in [#5104](https://github.com/gfx-rs/wgpu/pull/5104)

### Documentation

- Document Wayland specific behavior related to `SurfaceTexture::present`. By @i509VCB in [#5093](https://github.com/gfx-rs/wgpu/pull/5093).


## v0.19.0 (2024-01-17)

This release includes:
- `wgpu`
- `wgpu-core`
- `wgpu-hal`
- `wgpu-types`
- `wgpu-info`
- `naga` (skipped from 0.14 to 0.19)
- `naga-cli` (skipped from 0.14 to 0.19)
- `d3d12` (skipped from 0.7 to 0.19)

### Improved Multithreading through internal use of Reference Counting

Large refactoring of wgpu’s internals aiming at reducing lock contention, and providing better performance when using wgpu on multiple threads.

[Check the blog post!](https://gfx-rs.github.io/2023/11/24/arcanization.html)

By @gents83 in [#3626](https://github.com/gfx-rs/wgpu/pull/3626) and thanks also to @jimblandy, @nical, @Wumpf, @Elabajaba & @cwfitzgerald

### All Public Dependencies are Re-Exported

All of wgpu's public dependencies are now re-exported at the top level so that users don't need to take their own dependencies.
This includes:
- wgpu-core
- wgpu-hal
- naga
- raw_window_handle
- web_sys

### Feature Flag Changes

#### WebGPU & WebGL in the same Binary

Enabling `webgl` no longer removes the `webgpu` backend.

Instead, there's a new (default enabled) `webgpu` feature that allows to explicitly opt-out of `webgpu` if so desired.
If both `webgl` & `webgpu` are enabled, `wgpu::Instance` decides upon creation whether to target wgpu-core/WebGL or WebGPU.
This means that adapter selection is not handled as with regular adapters, but still allows to decide at runtime whether
`webgpu` or the `webgl` backend should be used using a single wasm binary.
By @wumpf in [#5044](https://github.com/gfx-rs/wgpu/pull/5044)

#### `naga-ir` Dedicated Feature

The `naga-ir` feature has been added to allow you to add naga module shaders without guessing about what other features needed to be enabled to get access to it.
By @cwfitzgerald in [#5063](https://github.com/gfx-rs/wgpu/pull/5063).

#### `expose-ids` Feature available unconditionally

This feature allowed you to call `global_id` on any wgpu opaque handle to get a unique hashable identity for the given resource. This is now available without the feature flag.
By @cwfitzgerald in [#4841](https://github.com/gfx-rs/wgpu/pull/4841).

#### `dx12` and `metal` Backend Crate Features

wgpu now exposes backend feature for the Direct3D 12 (`dx12`) and Metal (`metal`) backend. These are enabled by default, but don't do anything when not targeting the corresponding OS.
By @daxpedda in [#4815](https://github.com/gfx-rs/wgpu/pull/4815).

### Direct3D 11 Backend Removal

This backend had no functionality, and with the recent support for GL on Desktop, which allows wgpu to run on older devices, there was no need to keep this backend.
By @valaphee in [#4828](https://github.com/gfx-rs/wgpu/pull/4828).

### `WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER` Environment Variable

This adds a way to allow a Vulkan driver which is non-compliant per `VK_KHR_driver_properties` to be enumerated. This is intended for testing new Vulkan drivers which are not Vulkan compliant yet.
By @i509VCB in [#4754](https://github.com/gfx-rs/wgpu/pull/4754).

### `DeviceExt::create_texture_with_data` allows Mip-Major Data

Previously, `DeviceExt::create_texture_with_data` only allowed data to be provided in layer major order. There is now a `order` parameter which allows you to specify if the data is in layer major or mip major order.
```diff
    let tex = ctx.device.create_texture_with_data(
        &queue,
        &descriptor,
+       wgpu::util::TextureDataOrder::LayerMajor,
        src_data,
    );
```

By @cwfitzgerald in [#4780](https://github.com/gfx-rs/wgpu/pull/4780).

### Safe & unified Surface Creation

It is now possible to safely create a `wgpu::Surface` with `wgpu::Instance::create_surface()` by letting `wgpu::Surface` hold a lifetime to `window`.
Passing an owned value `window` to `Surface` will return a `wgpu::Surface<'static>`.

All possible safe variants (owned windows and web canvases) are grouped using `wgpu::SurfaceTarget`.
Conversion to `wgpu::SurfaceTarget` is automatic for any type implementing `raw-window-handle`'s `HasWindowHandle` & `HasDisplayHandle` traits, i.e. most window types.
For web canvas types this has to be done explicitly:
```rust
let surface: wgpu::Surface<'static> = instance.create_surface(wgpu::SurfaceTarget::Canvas(my_canvas))?;
```

All unsafe variants are now grouped under `wgpu::Instance::create_surface_unsafe` which takes the
`wgpu::SurfaceTargetUnsafe` enum and always returns `wgpu::Surface<'static>`.

In order to create a `wgpu::Surface<'static>` without passing ownership of the window use
`wgpu::SurfaceTargetUnsafe::from_window`:
```rust
let surface = unsafe {
  instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::from_window(&my_window))?
};
```
The easiest way to make this code safe is to use shared ownership:
```rust
let window: Arc<winit::Window>;
// ...
let surface = instance.create_surface(window.clone())?;
```

All platform specific surface creation using points have moved into `SurfaceTargetUnsafe` as well.
For example:

Safety by @daxpedda in [#4597](https://github.com/gfx-rs/wgpu/pull/4597)
Unification by @wumpf in [#4984](https://github.com/gfx-rs/wgpu/pull/4984)

### Add partial Support for WGSL Abstract Types

Abstract types make numeric literals easier to use, by
automatically converting literals and other constant expressions
from abstract numeric types to concrete types when safe and
necessary. For example, to build a vector of floating-point
numbers, Naga previously made you write:
```rust
vec3<f32>(1.0, 2.0, 3.0)
```
With this change, you can now simply write:
```rust
vec3<f32>(1, 2, 3)
```
Even though the literals are abstract integers, Naga recognizes
that it is safe and necessary to convert them to `f32` values in
order to build the vector. You can also use abstract values as
initializers for global constants and global and local variables,
like this:
```rust
var unit_x: vec2<f32> = vec2(1, 0);
```
The literals `1` and `0` are abstract integers, and the expression
`vec2(1, 0)` is an abstract vector. However, Naga recognizes that
it can convert that to the concrete type `vec2<f32>` to satisfy
the given type of `unit_x`.
The WGSL specification permits abstract integers and
floating-point values in almost all contexts, but Naga's support
for this is still incomplete. Many WGSL operators and builtin
functions are specified to produce abstract results when applied
to abstract inputs, but for now Naga simply concretizes them all
before applying the operation. We will expand Naga's abstract type
support in subsequent pull requests.
As part of this work, the public types `naga::ScalarKind` and
`naga::Literal` now have new variants, `AbstractInt` and `AbstractFloat`.

By @jimblandy in [#4743](https://github.com/gfx-rs/wgpu/pull/4743), [#4755](https://github.com/gfx-rs/wgpu/pull/4755).

### `Instance::enumerate_adapters` now returns `Vec<Adapter>` instead of an `ExactSizeIterator`

This allows us to support WebGPU and WebGL in the same binary.

```diff
- let adapters: Vec<Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).collect();
+ let adapters: Vec<Adapter> = instance.enumerate_adapters(wgpu::Backends::all());
```

By @wumpf in [#5044](https://github.com/gfx-rs/wgpu/pull/5044)

### `device.poll()` now returns a `MaintainResult` instead of a `bool`

This is a forward looking change, as we plan to add more information to the `MaintainResult` in the future.
This enum has the same data as the boolean, but with some useful helper functions.

```diff
- let queue_finished: bool = device.poll(wgpu::Maintain::Wait);
+ let queue_finished: bool = device.poll(wgpu::Maintain::Wait).is_queue_empty();
```

By @cwfitzgerald in [#5053](https://github.com/gfx-rs/wgpu/pull/5053)

### New Features

#### General
- Added `DownlevelFlags::VERTEX_AND_INSTANCE_INDEX_RESPECTS_RESPECTIVE_FIRST_VALUE_IN_INDIRECT_DRAW` to know if `@builtin(vertex_index)` and `@builtin(instance_index)` will respect the `first_vertex` / `first_instance` in indirect calls. If this is not present, both will always start counting from 0. Currently enabled on all backends except DX12. By @cwfitzgerald in [#4722](https://github.com/gfx-rs/wgpu/pull/4722).
- Added support for the `FLOAT32_FILTERABLE` feature (web and native, corresponds to WebGPU's `float32-filterable`). By @almarklein in [#4759](https://github.com/gfx-rs/wgpu/pull/4759).
- GPU buffer memory is released during "lose the device". By @bradwerth in [#4851](https://github.com/gfx-rs/wgpu/pull/4851).
- wgpu and wgpu-core cargo feature flags are now documented on docs.rs. By @wumpf in [#4886](https://github.com/gfx-rs/wgpu/pull/4886).
- DeviceLostClosure is guaranteed to be invoked exactly once. By @bradwerth in [#4862](https://github.com/gfx-rs/wgpu/pull/4862).
- Log vulkan validation layer messages during instance creation and destruction: By @exrook in [#4586](https://github.com/gfx-rs/wgpu/pull/4586).
- `TextureFormat::block_size` is deprecated, use `TextureFormat::block_copy_size` instead: By @wumpf in [#4647](https://github.com/gfx-rs/wgpu/pull/4647).
- Rename of `DispatchIndirect`, `DrawIndexedIndirect`, and `DrawIndirect` types in the `wgpu::util` module to `DispatchIndirectArgs`, `DrawIndexedIndirectArgs`, and `DrawIndirectArgs`. By @cwfitzgerald in [#4723](https://github.com/gfx-rs/wgpu/pull/4723).
- Make the size parameter of `encoder.clear_buffer` an `Option<u64>` instead of `Option<NonZero<u64>>`. By @nical in [#4737](https://github.com/gfx-rs/wgpu/pull/4737).
- Reduce the `info` log level noise. By @nical in [#4769](https://github.com/gfx-rs/wgpu/pull/4769), [#4711](https://github.com/gfx-rs/wgpu/pull/4711) and [#4772](https://github.com/gfx-rs/wgpu/pull/4772)
- Rename `features` & `limits` fields of `DeviceDescriptor` to `required_features` & `required_limits`. By @teoxoy in [#4803](https://github.com/gfx-rs/wgpu/pull/4803).
- `SurfaceConfiguration` now exposes `desired_maximum_frame_latency` which was previously hard-coded to 2. By setting it to 1 you can reduce latency under the risk of making GPU & CPU work sequential. Currently, on DX12 this affects the `MaximumFrameLatency`, on all other backends except OpenGL the size of the swapchain (on OpenGL this has no effect). By @emilk & @wumpf in [#4899](https://github.com/gfx-rs/wgpu/pull/4899)

#### OpenGL
- `@builtin(instance_index)` now properly reflects the range provided in the draw call instead of always counting from 0. By @cwfitzgerald in [#4722](https://github.com/gfx-rs/wgpu/pull/4722).
- Desktop GL now supports `POLYGON_MODE_LINE` and `POLYGON_MODE_POINT`. By @valaphee in [#4836](https://github.com/gfx-rs/wgpu/pull/4836).

#### Naga

- Naga's WGSL front end now allows operators to produce values with abstract types, rather than concretizing their operands. By @jimblandy in [#4850](https://github.com/gfx-rs/wgpu/pull/4850) and [#4870](https://github.com/gfx-rs/wgpu/pull/4870).
- Naga's WGSL front and back ends now have experimental support for 64-bit floating-point literals: `1.0lf` denotes an `f64` value. There has been experimental support for an `f64` type for a while, but until now there was no syntax for writing literals with that type. As before, Naga module validation rejects `f64` values unless `naga::valid::Capabilities::FLOAT64` is requested. By @jimblandy in [#4747](https://github.com/gfx-rs/wgpu/pull/4747).
- Naga constant evaluation can now process binary operators whose operands are both vectors. By @jimblandy in [#4861](https://github.com/gfx-rs/wgpu/pull/4861).
- Add `--bulk-validate` option to Naga CLI. By @jimblandy in [#4871](https://github.com/gfx-rs/wgpu/pull/4871).
- Naga's `cargo xtask validate` now runs validation jobs in parallel, using the [jobserver](https://crates.io/crates/jobserver) protocol to limit concurrency, and offers a `validate all` subcommand, which runs all available validation types. By @jimblandy in [#4902](https://github.com/gfx-rs/wgpu/pull/4902).
- Remove `span` and `validate` features. Always fully validate shader modules, and always track source positions for use in error messages. By @teoxoy in [#4706](https://github.com/gfx-rs/wgpu/pull/4706).
- Introduce a new `Scalar` struct type for use in Naga's IR, and update all frontend, middle, and backend code appropriately. By @jimblandy in [#4673](https://github.com/gfx-rs/wgpu/pull/4673).
- Add more metal keywords. By @fornwall in [#4707](https://github.com/gfx-rs/wgpu/pull/4707).
- Add a new `naga::Literal` variant, `I64`, for signed 64-bit literals. [#4711](https://github.com/gfx-rs/wgpu/pull/4711).
- Emit and init `struct` member padding always. By @ErichDonGubler in [#4701](https://github.com/gfx-rs/wgpu/pull/4701).
- In WGSL output, always include the `i` suffix on `i32` literals. By @jimblandy in [#4863](https://github.com/gfx-rs/wgpu/pull/4863).
- In WGSL output, always include the `f` suffix on `f32` literals. By @jimblandy in [#4869](https://github.com/gfx-rs/wgpu/pull/4869).

### Bug Fixes

#### General

- `BufferMappedRange` trait is now `WasmNotSendSync`, i.e. it is `Send`/`Sync` if not on wasm or `fragile-send-sync-non-atomic-wasm` is enabled. By @wumpf in [#4818](https://github.com/gfx-rs/wgpu/pull/4818).
- Align `wgpu_types::CompositeAlphaMode` serde serialization to spec. By @littledivy in [#4940](https://github.com/gfx-rs/wgpu/pull/4940).
- Fix error message of `ConfigureSurfaceError::TooLarge`. By @Dinnerbone in [#4960](https://github.com/gfx-rs/wgpu/pull/4960).
- Fix dropping of `DeviceLostCallbackC` params. By @bradwerth in [#5032](https://github.com/gfx-rs/wgpu/pull/5032).
- Fixed a number of panics. By @nical in [#4999](https://github.com/gfx-rs/wgpu/pull/4999), [#5014](https://github.com/gfx-rs/wgpu/pull/5014), [#5024](https://github.com/gfx-rs/wgpu/pull/5024), [#5025](https://github.com/gfx-rs/wgpu/pull/5025), [#5026](https://github.com/gfx-rs/wgpu/pull/5026), [#5027](https://github.com/gfx-rs/wgpu/pull/5027), [#5028](https://github.com/gfx-rs/wgpu/pull/5028) and [#5042](https://github.com/gfx-rs/wgpu/pull/5042).
- No longer validate surfaces against their allowed extent range on configure. This caused warnings that were almost impossible to avoid. As before, the resulting behavior depends on the compositor. By @wumpf in [#4796](https://github.com/gfx-rs/wgpu/pull/4796).

#### DX12

- Fixed D3D12_SUBRESOURCE_FOOTPRINT calculation for block compressed textures which caused a crash with `Queue::write_texture` on DX12. By @DTZxPorter in [#4990](https://github.com/gfx-rs/wgpu/pull/4990).

#### Vulkan

- Use `VK_EXT_robustness2` only when not using an outdated intel iGPU driver. By @TheoDulka in [#4602](https://github.com/gfx-rs/wgpu/pull/4602).

#### WebGPU

- Allow calling `BufferSlice::get_mapped_range` multiple times on the same buffer slice (instead of throwing a Javascript exception). By @DouglasDwyer in [#4726](https://github.com/gfx-rs/wgpu/pull/4726).

#### WGL

- Create a hidden window per `wgpu::Instance` instead of sharing a global one. By @Zoxc in [#4603](https://github.com/gfx-rs/wgpu/issues/4603)

#### Naga

- Make module compaction preserve the module's named types, even if they are unused. By @jimblandy in [#4734](https://github.com/gfx-rs/wgpu/pull/4734).
- Improve algorithm used by module compaction. By @jimblandy in [#4662](https://github.com/gfx-rs/wgpu/pull/4662).
- When reading GLSL, fix the argument types of the double-precision floating-point overloads of the `dot`, `reflect`, `distance`, and `ldexp` builtin functions. Correct the WGSL generated for constructing 64-bit floating-point matrices. Add tests for all the above. By @jimblandy in [#4684](https://github.com/gfx-rs/wgpu/pull/4684).
- Allow Naga's IR types to represent matrices with elements elements of any scalar kind. This makes it possible for Naga IR types to represent WGSL abstract matrices. By @jimblandy in [#4735](https://github.com/gfx-rs/wgpu/pull/4735).
- Preserve the source spans for constants and expressions correctly across module compaction. By @jimblandy in [#4696](https://github.com/gfx-rs/wgpu/pull/4696).
- Record the names of WGSL `alias` declarations in Naga IR `Type`s. By @jimblandy in [#4733](https://github.com/gfx-rs/wgpu/pull/4733).

#### Metal

- Allow the `COPY_SRC` usage flag in surface configuration. By @Toqozz in [#4852](https://github.com/gfx-rs/wgpu/pull/4852).

### Examples

- remove winit dependency from hello-compute example. By @psvri in [#4699](https://github.com/gfx-rs/wgpu/pull/4699)
- hello-compute example fix failure with `wgpu error: Validation Error` if arguments are missing. By @vilcans in [#4939](https://github.com/gfx-rs/wgpu/pull/4939).
- Made the examples page not crash on Chrome on Android, and responsive to screen sizes. By @Dinnerbone in [#4958](https://github.com/gfx-rs/wgpu/pull/4958).

## v0.18.2 (2023-12-06)

This release includes `naga` version 0.14.2. The crates `wgpu-core`, `wgpu-hal` are still at `0.18.1` and the crates `wgpu` and `wgpu-types` are still at `0.18.0`.

### Bug Fixes

#### Naga
- When evaluating const-expressions and generating SPIR-V, properly handle `Compose` expressions whose operands are `Splat` expressions. Such expressions are created and marked as constant by the constant evaluator. By @jimblandy in [#4695](https://github.com/gfx-rs/wgpu/pull/4695).

## v0.18.1 (2023-11-15)

(naga version 0.14.1)

### Bug Fixes

#### General
- Fix panic in `Surface::configure` in debug builds. By @cwfitzgerald in [#4635](https://github.com/gfx-rs/wgpu/pull/4635)
- Fix crash when all the following are true: By @teoxoy in #[#4642](https://github.com/gfx-rs/wgpu/pull/4642)
  - Passing a naga module directly to `Device::create_shader_module`.
  - `InstanceFlags::DEBUG` is enabled.

#### DX12
- Always use HLSL 2018 when using DXC to compile HLSL shaders. By @daxpedda in [#4629](https://github.com/gfx-rs/wgpu/pull/4629)

#### Metal
- In Metal Shading Language output, fix issue where local variables were sometimes using variable names from previous functions. By @DJMcNab in [#4594](https://github.com/gfx-rs/wgpu/pull/4594)

## v0.18.0 (2023-10-25)

For naga changelogs at or before v0.14.0. See [naga's changelog](naga/CHANGELOG.md).

### Desktop OpenGL 3.3+ Support on Windows

We now support OpenGL on Windows! This brings support for a vast majority of the hardware that used to be covered by our DX11 backend. As of this writing we support OpenGL 3.3+, though there are efforts to reduce that further.

This allows us to cover the last 12 years of Intel GPUs (starting with Ivy Bridge; aka 3xxx), and the last 16 years of AMD (starting with Terascale; aka HD 2000) / NVidia GPUs (starting with Tesla; aka GeForce 8xxx).

By @Zoxc in [#4248](https://github.com/gfx-rs/wgpu/pull/4248)

### Timestamp Queries Supported on Metal and OpenGL

Timestamp queries are now supported on both Metal and Desktop OpenGL. On Apple chips on Metal, they only support timestamp queries in command buffers or in the renderpass descriptor,
they do not support them inside a pass.

Metal: By @Wumpf in [#4008](https://github.com/gfx-rs/wgpu/pull/4008)
OpenGL: By @Zoxc in [#4267](https://github.com/gfx-rs/wgpu/pull/4267)

### Render/Compute Pass Query Writes

Addition of the `TimestampWrites` type to compute and render pass descriptors to allow profiling on tilers which do not support timestamps inside passes.

Added [an example](https://github.com/gfx-rs/wgpu/tree/trunk/examples/timestamp-queries) to demonstrate the various kinds of timestamps.

Additionally, metal now supports timestamp queries!

By @FL33TW00D & @wumpf in [#3636](https://github.com/gfx-rs/wgpu/pull/3636).

### Occlusion Queries

We now support binary occlusion queries! This allows you to determine if any of the draw calls within the query drew any pixels.

Use the new `occlusion_query_set` field on `RenderPassDescriptor` to give a query set that occlusion queries will write to.

```diff
let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
    // ...
+   occlusion_query_set: Some(&my_occlusion_query_set),
});
```

Within the renderpass do the following to write the occlusion query results to the query set at the given index:

```rust
rpass.begin_occlusion_query(index);
rpass.draw(...);
rpass.draw(...);
rpass.end_occlusion_query();
```

These are binary occlusion queries, so the result will be either 0 or an unspecified non-zero value.

By @Valaphee in [#3402](https://github.com/gfx-rs/wgpu/pull/3402)

### Shader Improvements

```rust
// WGSL constant expressions are now supported!
const BLAH: u32 = 1u + 1u;

// `rgb10a2uint` and `bgra8unorm` can now be used as a storage image format.
var image: texture_storage_2d<rgb10a2uint, write>;
var image: texture_storage_2d<bgra8unorm, write>;

// You can now use dual source blending!
struct FragmentOutput{
    @location(0) source1: vec4<f32>,
    @location(0) @second_blend_source source2: vec4<f32>,
}

// `modf`/`frexp` now return structures
let result = modf(1.5);
result.fract == 0.5;
result.whole == 1.0;

let result = frexp(1.5);
result.fract == 0.75;
result.exponent == 2i;

// `modf`/`frexp` are currently disabled on GLSL and SPIR-V input.
```

### Shader Validation Improvements

```rust
// Cannot get pointer to a workgroup variable
fn func(p: ptr<workgroup, u32>); // ERROR

// Cannot create Inf/NaN through constant expressions
const INF: f32 = 3.40282347e+38 + 1.0; // ERROR
const NAN: f32 = 0.0 / 0.0; // ERROR

// `outerProduct` function removed

// Error on repeated or missing `@workgroup_size()`
@workgroup_size(1) @workgroup_size(2) // ERROR
fn compute_main() {}

// Error on repeated attributes.
fn fragment_main(@location(0) @location(0) location_0: f32) // ERROR
```

### RenderPass `StoreOp` is now Enumeration

`wgpu::Operations::store` used to be an underdocumented boolean value,
causing misunderstandings of the effect of setting it to `false`.

The API now more closely resembles WebGPU which distinguishes between `store` and `discard`,
see [WebGPU spec on GPUStoreOp](https://gpuweb.github.io/gpuweb/#enumdef-gpustoreop).

```diff
// ...
depth_ops: Some(wgpu::Operations {
    load: wgpu::LoadOp::Clear(1.0),
-   store: false,
+   store: wgpu::StoreOp::Discard,
}),
// ...
```

By @wumpf in [#4147](https://github.com/gfx-rs/wgpu/pull/4147)

### Instance Descriptor Settings

The instance descriptor grew two more fields: `flags` and `gles_minor_version`.

`flags` allow you to toggle the underlying api validation layers, debug information about shaders and objects in capture programs, and the ability to discard labels

`gles_minor_version` is a rather niche feature that allows you to force the GLES backend to use a specific minor version, this is useful to get ANGLE to enable more than GLES 3.0.

```diff
let instance = wgpu::Instance::new(InstanceDescriptor {
    ...
+   flags: wgpu::InstanceFlags::default()
+   gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
});
```

`gles_minor_version`: By @PJB3005 in [#3998](https://github.com/gfx-rs/wgpu/pull/3998)
`flags`: By @nical in [#4230](https://github.com/gfx-rs/wgpu/pull/4230)

### Many New Examples!

- Added the following examples: By @JustAnotherCodemonkey in [#3885](https://github.com/gfx-rs/wgpu/pull/3885).
  - [repeated-compute](https://github.com/gfx-rs/wgpu/tree/trunk/examples/repeated-compute)
  - [storage-texture](https://github.com/gfx-rs/wgpu/tree/trunk/examples/storage-texture)
  - [render-to-texture](https://github.com/gfx-rs/wgpu/tree/trunk/examples/render-to-texture)
  - [uniform-values](https://github.com/gfx-rs/wgpu/tree/trunk/examples/uniform-values)
  - [hello-workgroups](https://github.com/gfx-rs/wgpu/tree/trunk/examples/hello-workgroups)
  - [hello-synchronization](https://github.com/gfx-rs/wgpu/tree/trunk/examples/hello-synchronization)

### Revamped Testing Suite

Our testing harness was completely revamped and now automatically runs against all gpus in the system, shows the expected status of every test, and is tolerant to flakes.

Additionally, we have filled out our CI to now run the latest versions of WARP and Mesa. This means we can test even more features on CI than before.

By @cwfitzgerald in [#3873](https://github.com/gfx-rs/wgpu/pull/3873)

### The GLES backend is now optional on macOS

The `angle` feature flag has to be set for the GLES backend to be enabled on Windows & macOS.

By @teoxoy in [#4185](https://github.com/gfx-rs/wgpu/pull/4185)

### Added/New Features

- Re-export Naga. By @exrook in [#4172](https://github.com/gfx-rs/wgpu/pull/4172)
- Add WinUI 3 SwapChainPanel support. By @ddrboxman in [#4191](https://github.com/gfx-rs/wgpu/pull/4191)

### Changes

#### General

- Omit texture store bound checks since they are no-ops if out of bounds on all APIs. By @teoxoy in [#3975](https://github.com/gfx-rs/wgpu/pull/3975)
- Validate `DownlevelFlags::READ_ONLY_DEPTH_STENCIL`. By @teoxoy in [#4031](https://github.com/gfx-rs/wgpu/pull/4031)
- Add validation in accordance with WebGPU `setViewport` valid usage for `x`, `y` and `this.[[attachment_size]]`. By @James2022-rgb in [#4058](https://github.com/gfx-rs/wgpu/pull/4058)
- `wgpu::CreateSurfaceError` and `wgpu::RequestDeviceError` now give details of the failure, but no longer implement `PartialEq` and cannot be constructed. By @kpreid in [#4066](https://github.com/gfx-rs/wgpu/pull/4066) and [#4145](https://github.com/gfx-rs/wgpu/pull/4145)
- Make `WGPU_POWER_PREF=none` a valid value. By @fornwall in [4076](https://github.com/gfx-rs/wgpu/pull/4076)
- Support dual source blending in OpenGL ES, Metal, Vulkan & DX12. By @freqmod in [4022](https://github.com/gfx-rs/wgpu/pull/4022)
- Add stub support for device destroy and device validity. By @bradwerth in [4163](https://github.com/gfx-rs/wgpu/pull/4163) and in [4212](https://github.com/gfx-rs/wgpu/pull/4212)
- Add trace-level logging for most entry points in wgpu-core By @nical in [4183](https://github.com/gfx-rs/wgpu/pull/4183)
- Add `Rgb10a2Uint` format. By @teoxoy in [4199](https://github.com/gfx-rs/wgpu/pull/4199)
- Validate that resources are used on the right device. By @nical in [4207](https://github.com/gfx-rs/wgpu/pull/4207)
- Expose instance flags.
- Add support for the bgra8unorm-storage feature. By @jinleili and @nical in [#4228](https://github.com/gfx-rs/wgpu/pull/4228)
- Calls to lost devices now return `DeviceError::Lost` instead of `DeviceError::Invalid`. By @bradwerth in [#4238]([https://github.com/gfx-rs/wgpu/pull/4238])
- Let the `"strict_asserts"` feature enable check that wgpu-core's lock-ordering tokens are unique per thread. By @jimblandy in [#4258]([https://github.com/gfx-rs/wgpu/pull/4258])
- Allow filtering labels out before they are passed to GPU drivers by @nical in [https://github.com/gfx-rs/wgpu/pull/4246](4246)
- `DeviceLostClosure` callback mechanism provided so user agents can resolve `GPUDevice.lost` Promises at the appropriate time by @bradwerth in [#4645](https://github.com/gfx-rs/wgpu/pull/4645)


#### Vulkan

- Rename `wgpu_hal::vulkan::Instance::required_extensions` to `desired_extensions`. By @jimblandy in [#4115](https://github.com/gfx-rs/wgpu/pull/4115)
- Don't bother calling `vkFreeCommandBuffers` when `vkDestroyCommandPool` will take care of that for us. By @jimblandy in [#4059](https://github.com/gfx-rs/wgpu/pull/4059)

#### DX12

- Bump `gpu-allocator` to 0.23. By @Elabajaba in [#4198](https://github.com/gfx-rs/wgpu/pull/4198)

### Documentation

- Use WGSL for VertexFormat example types. By @ScanMountGoat in [#4035](https://github.com/gfx-rs/wgpu/pull/4035)
- Fix description of `Features::TEXTURE_COMPRESSION_ASTC_HDR` in [#4157](https://github.com/gfx-rs/wgpu/pull/4157)

### Bug Fixes

#### General

- Derive storage bindings via `naga::StorageAccess` instead of `naga::GlobalUse`. By @teoxoy in [#3985](https://github.com/gfx-rs/wgpu/pull/3985).
- `Queue::on_submitted_work_done` callbacks will now always be called after all previous `BufferSlice::map_async` callbacks, even when there are no active submissions. By @cwfitzgerald in [#4036](https://github.com/gfx-rs/wgpu/pull/4036).
- Fix `clear` texture views being leaked when `wgpu::SurfaceTexture` is dropped before it is presented. By @rajveermalviya in [#4057](https://github.com/gfx-rs/wgpu/pull/4057).
- Add `Feature::SHADER_UNUSED_VERTEX_OUTPUT` to allow unused vertex shader outputs. By @Aaron1011 in [#4116](https://github.com/gfx-rs/wgpu/pull/4116).
- Fix a panic in `surface_configure`. By @nical in [#4220](https://github.com/gfx-rs/wgpu/pull/4220) and [#4227](https://github.com/gfx-rs/wgpu/pull/4227)
- Pipelines register their implicit layouts in error cases. By @bradwerth in [#4624](https://github.com/gfx-rs/wgpu/pull/4624)
- Better handle explicit destruction of textures and buffers. By @nical in [#4657](https://github.com/gfx-rs/wgpu/pull/4657)

#### Vulkan

- Fix enabling `wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY` not being actually enabled in vulkan backend. By @39ali in[#3772](https://github.com/gfx-rs/wgpu/pull/3772).
- Don't pass `vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR` unless the `VK_KHR_portability_enumeration` extension is available. By @jimblandy in[#4038](https://github.com/gfx-rs/wgpu/pull/4038).
- Enhancement of [#4038], using ash's definition instead of hard-coded c_str. By @hybcloud in[#4044](https://github.com/gfx-rs/wgpu/pull/4044).
- Enable vulkan presentation on (Linux) Intel Mesa >= v21.2. By @flukejones in[#4110](https://github.com/gfx-rs/wgpu/pull/4110)

#### DX12

- DX12 doesn't support `Features::POLYGON_MODE_POINT``. By @teoxoy in [#4032](https://github.com/gfx-rs/wgpu/pull/4032).
- Set `Features::VERTEX_WRITABLE_STORAGE` based on the right feature level. By @teoxoy in [#4033](https://github.com/gfx-rs/wgpu/pull/4033).

#### Metal

- Ensure that MTLCommandEncoder calls endEncoding before it is deallocated. By @bradwerth in [#4023](https://github.com/gfx-rs/wgpu/pull/4023)

#### WebGPU

- Ensure that limit requests and reporting is done correctly. By @OptimisticPeach in [#4107](https://github.com/gfx-rs/wgpu/pull/4107)
- Validate usage of polygon mode. By @teoxoy in [#4196](https://github.com/gfx-rs/wgpu/pull/4196)

#### GLES

- enable/disable blending per attachment only when available (on ES 3.2 or higher). By @teoxoy in [#4234](https://github.com/gfx-rs/wgpu/pull/4234)

### Documentation

- Add an overview of `RenderPass` and how render state works. By @kpreid in [#4055](https://github.com/gfx-rs/wgpu/pull/4055)

### Examples

- Created `wgpu-example::utils` module to contain misc functions and such that are common code but aren't part of the example framework. Add to it the functions `output_image_wasm` and `output_image_native`, both for outputting `Vec<u8>` RGBA images either to the disc or the web page. By @JustAnotherCodemonkey in [#3885](https://github.com/gfx-rs/wgpu/pull/3885).
- Removed `capture` example as it had issues (did not run on wasm) and has been replaced by `render-to-texture` (see above). By @JustAnotherCodemonkey in [#3885](https://github.com/gfx-rs/wgpu/pull/3885).

## v0.17.2 (2023-10-03)

### Bug Fixes

#### Vulkan

- Fix x11 hang while resizing on vulkan. @Azorlogh in [#4184](https://github.com/gfx-rs/wgpu/pull/4184).

## v0.17.1 (2023-09-27)

### Added/New Features

- Add `get_mapped_range_as_array_buffer` for faster buffer read-backs in wasm builds. By @ryankaplan in [#4042] (https://github.com/gfx-rs/wgpu/pull/4042).

### Bug Fixes

#### DX12

- Fix panic on resize when using DX12. By @cwfitzgerald in [#4106](https://github.com/gfx-rs/wgpu/pull/4106)

#### Vulkan

- Suppress validation error caused by OBS layer. This was also fixed upstream. By @cwfitzgerald in [#4002](https://github.com/gfx-rs/wgpu/pull/4002)
- Work around bug in nvidia's vkCmdFillBuffer implementation. By @cwfitzgerald in [#4132](https://github.com/gfx-rs/wgpu/pull/4132).

## v0.17.0 (2023-07-20)

This is the first release that featured `wgpu-info` as a binary crate for getting information about what devices wgpu sees in your system. It can dump the information in both human readable format and json.

### Major Changes

This release was fairly minor as breaking changes go.

#### `wgpu` types now `!Send` `!Sync` on wasm

Up until this point, wgpu has made the assumption that threads do not exist on wasm. With the rise of libraries like [`wasm_thread`](https://crates.io/crates/wasm_thread) making it easier and easier to do wasm multithreading this assumption is no longer sound. As all wgpu objects contain references into the JS heap, they cannot leave the thread they started on.

As we understand that this change might be very inconvenient for users who don't care about wasm threading, there is a crate feature which re-enables the old behavior: `fragile-send-sync-non-atomic-wasm`. So long as you don't compile your code with `-Ctarget-feature=+atomics`, `Send` and `Sync` will be implemented again on wgpu types on wasm. As the name implies, especially for libraries, this is very fragile, as you don't know if a user will want to compile with atomics (and therefore threads) or not.

By @daxpedda in [#3691](https://github.com/gfx-rs/wgpu/pull/3691)

#### Power Preference is now optional

The `power_preference` field of `RequestAdapterOptions` is now optional. If it is `PowerPreference::None`, we will choose the first available adapter, preferring GPU adapters over CPU adapters.

By @Aaron1011 in [#3903](https://github.com/gfx-rs/wgpu/pull/3903)

#### `initialize_adapter_from_env` argument changes

Removed the backend_bits parameter from `initialize_adapter_from_env` and `initialize_adapter_from_env_or_default`. If you want to limit the backends used by this function, only enable the wanted backends in the instance.

Added a compatible surface parameter, to ensure the given device is able to be presented onto the given surface.

```diff
- wgpu::util::initialize_adapter_from_env(instance, backend_bits);
+ wgpu::util::initialize_adapter_from_env(instance, Some(&compatible_surface));
```

By @fornwall in [#3904](https://github.com/gfx-rs/wgpu/pull/3904) and [#3905](https://github.com/gfx-rs/wgpu/pull/3905)

#### Misc Breaking Changes

- Change `AdapterInfo::{device,vendor}` to be `u32` instead of `usize`. By @ameknite in [#3760](https://github.com/gfx-rs/wgpu/pull/3760)

### Changes

- Added support for importing external buffers using `buffer_from_raw` (Dx12, Metal, Vulkan) and `create_buffer_from_hal`. By @AdrianEddy in [#3355](https://github.com/gfx-rs/wgpu/pull/3355)


#### Vulkan

- Work around [Vulkan-ValidationLayers#5671](https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/5671) by ignoring reports of violations of [VUID-vkCmdEndDebugUtilsLabelEXT-commandBuffer-01912](https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdEndDebugUtilsLabelEXT-commandBuffer-01912). By @jimblandy in [#3809](https://github.com/gfx-rs/wgpu/pull/3809).

### Added/New Features

#### General

- Empty scissor rects are allowed now, matching the specification. by @PJB3005 in [#3863](https://github.com/gfx-rs/wgpu/pull/3863).
- Add back components info to `TextureFormat`s. By @teoxoy in [#3843](https://github.com/gfx-rs/wgpu/pull/3843).
- Add `get_mapped_range_as_array_buffer` for faster buffer read-backs in wasm builds. By @ryankaplan in [#4042] (https://github.com/gfx-rs/wgpu/pull/4042).

### Documentation

- Better documentation for draw, draw_indexed, set_viewport and set_scissor_rect. By @genusistimelord in [#3860](https://github.com/gfx-rs/wgpu/pull/3860)
- Fix link to `GPUVertexBufferLayout`. By @fornwall in [#3906](https://github.com/gfx-rs/wgpu/pull/3906)
- Document feature requirements for `DEPTH32FLOAT_STENCIL8` by @ErichDonGubler in [#3734](https://github.com/gfx-rs/wgpu/pull/3734).
- Flesh out docs. for `AdapterInfo::{device,vendor}` by @ErichDonGubler in [#3763](https://github.com/gfx-rs/wgpu/pull/3763).
- Spell out which sizes are in bytes. By @jimblandy in [#3773](https://github.com/gfx-rs/wgpu/pull/3773).
- Validate that `descriptor.usage` is not empty in `create_buffer` by @nical in [#3928](https://github.com/gfx-rs/wgpu/pull/3928)
- Update `max_bindings_per_bind_group` limit to reflect spec changes by @ErichDonGubler and @nical in [#3943](https://github.com/gfx-rs/wgpu/pull/3943) [#3942](https://github.com/gfx-rs/wgpu/pull/3942)
- Add better docs for `Limits`, listing the actual limits returned by `downlevel_defaults` and `downlevel_webgl2_defaults` by @JustAnotherCodemonkey in [#3988](https://github.com/gfx-rs/wgpu/pull/3988)

### Bug Fixes

#### General

- Fix order of arguments to glPolygonOffset by @komadori in [#3783](https://github.com/gfx-rs/wgpu/pull/3783).
- Fix OpenGL/EGL backend not respecting non-sRGB texture formats in `SurfaceConfiguration`. by @liquidev in [#3817](https://github.com/gfx-rs/wgpu/pull/3817)
- Make write- and read-only marked buffers match non-readonly layouts. by @fornwall in [#3893](https://github.com/gfx-rs/wgpu/pull/3893)
- Fix leaking X11 connections. by @wez in [#3924](https://github.com/gfx-rs/wgpu/pull/3924)
- Fix ASTC feature selection in the webgl backend. by @expenses in [#3934](https://github.com/gfx-rs/wgpu/pull/3934)
- Fix Multiview to disable validation of TextureViewDimension and ArrayLayerCount. By @MalekiRe in [#3779](https://github.com/gfx-rs/wgpu/pull/3779#issue-1713269437).

#### Vulkan

- Fix incorrect aspect in barriers when using emulated Stencil8 textures. By @cwfitzgerald in [#3833](https://github.com/gfx-rs/wgpu/pull/3833).
- Implement depth-clip-control using depthClamp instead of VK_EXT_depth_clip_enable. By @AlbinBernhardssonARM [#3892](https://github.com/gfx-rs/wgpu/pull/3892).
- Fix enabling `wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY` not being actually enabled in vulkan backend. By @39ali in[#3772](https://github.com/gfx-rs/wgpu/pull/3772).

#### Metal

- Fix renderpasses being used inside of renderpasses. By @cwfitzgerald in [#3828](https://github.com/gfx-rs/wgpu/pull/3828)
- Support (simulated) visionOS. By @jinleili in [#3883](https://github.com/gfx-rs/wgpu/pull/3883)

#### DX12

- Disable suballocation on Intel Iris(R) Xe. By @xiaopengli89 in [#3668](https://github.com/gfx-rs/wgpu/pull/3668)
- Change the `max_buffer_size` limit from `u64::MAX` to `i32::MAX`. By @nical in [#4020](https://github.com/gfx-rs/wgpu/pull/4020)

#### WebGPU

- Use `get_preferred_canvas_format()` to fill `formats` of `SurfaceCapabilities`. By @jinleili in [#3744](https://github.com/gfx-rs/wgpu/pull/3744)

### Examples

- Publish examples to wgpu.rs on updates to trunk branch instead of gecko. By @paul-hansen in [#3750](https://github.com/gfx-rs/wgpu/pull/3750)
- Ignore the exception values generated by the winit resize event. By @jinleili in [#3916](https://github.com/gfx-rs/wgpu/pull/3916)

## v0.16.3 (2023-07-19)

### Changes

#### General

- Make the `Id` type that is exposed when using the `expose-ids` feature implement `Send` and `Sync` again. This was unintentionally changed by the v0.16.0 release and is now fixed.

## v0.16.2 (2023-07-09)

### Changes

#### DX12

- Increase the `max_storage_buffers_per_shader_stage` and `max_storage_textures_per_shader_stage` limits based on what the hardware supports. by @Elabajaba in [#3798]https://github.com/gfx-rs/wgpu/pull/3798

## v0.16.1 (2023-05-24)

### Bug Fixes

- Fix missing 4X MSAA support on some OpenGL backends. By @emilk in [#3780](https://github.com/gfx-rs/wgpu/pull/3780)

#### General

- Fix crash on dropping `wgpu::CommandBuffer`. By @wumpf in [#3726](https://github.com/gfx-rs/wgpu/pull/3726).
- Use `u32`s internally for bind group indices, rather than `u8`. By @ErichDonGubler in [#3743](https://github.com/gfx-rs/wgpu/pull/3743).

#### WebGPU

* Fix crash when calling `create_surface_from_canvas`. By @grovesNL in [#3718](https://github.com/gfx-rs/wgpu/pull/3718)

## v0.16.0 (2023-04-19)

### Major changes

#### Shader Changes

`type` has been replaced with `alias` to match with upstream WebGPU.

```diff
- type MyType = vec4<u32>;
+ alias MyType = vec4<u32>;
```

#### TextureFormat info API

The `TextureFormat::describe` function was removed in favor of separate functions: `block_dimensions`, `is_compressed`, `is_srgb`, `required_features`, `guaranteed_format_features`, `sample_type` and `block_size`.


```diff
- let block_dimensions = format.describe().block_dimensions;
+ let block_dimensions = format.block_dimensions();
- let is_compressed = format.describe().is_compressed();
+ let is_compressed = format.is_compressed();
- let is_srgb = format.describe().srgb;
+ let is_srgb = format.is_srgb();
- let required_features = format.describe().required_features;
+ let required_features = format.required_features();
```

Additionally `guaranteed_format_features` now takes a set of features to assume are enabled.

```diff
- let guaranteed_format_features = format.describe().guaranteed_format_features;
+ let guaranteed_format_features = format.guaranteed_format_features(device.features());
```

Additionally `sample_type` and `block_size` now take an optional `TextureAspect` and return `Option`s.

```diff
- let sample_type = format.describe().sample_type;
+ let sample_type = format.sample_type(None).expect("combined depth-stencil format requires specifying a TextureAspect");
- let block_size = format.describe().block_size;
+ let block_size = format.block_size(None).expect("combined depth-stencil format requires specifying a TextureAspect");
```

By @teoxoy in [#3436](https://github.com/gfx-rs/wgpu/pull/3436)

#### BufferUsages::QUERY_RESOLVE

Buffers used as the `destination` argument of `CommandEncoder::resolve_query_set` now have to contain the `QUERY_RESOLVE` usage instead of the `COPY_DST` usage.

```diff
  let destination = device.create_buffer(&wgpu::BufferDescriptor {
      // ...
-     usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
+     usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::MAP_READ,
      mapped_at_creation: false,
  });
  command_encoder.resolve_query_set(&query_set, query_range, &destination, destination_offset);
```

By @JolifantoBambla in [#3489](https://github.com/gfx-rs/wgpu/pull/3489)

#### Renamed features

The following `Features` have been renamed.

- `SHADER_FLOAT16` -> `SHADER_F16`
- `SHADER_FLOAT64` -> `SHADER_F64`
- `SHADER_INT16` -> `SHADER_I16`
- `TEXTURE_COMPRESSION_ASTC_LDR` -> `TEXTURE_COMPRESSION_ASTC`
- `WRITE_TIMESTAMP_INSIDE_PASSES` -> `TIMESTAMP_QUERY_INSIDE_PASSES`

By @teoxoy in [#3534](https://github.com/gfx-rs/wgpu/pull/3534)

#### Anisotropic Filtering

Anisotropic filtering has been brought in line with the spec. The anisotropic clamp is now a `u16` (was a `Option<u8>`) which must be at least 1.

If the anisotropy clamp is not 1, all the filters in a sampler must be `Linear`.

```diff
SamplerDescriptor {
-    anisotropic_clamp: None,
+    anisotropic_clamp: 1,
}
```

By @cwfitzgerald in [#3610](https://github.com/gfx-rs/wgpu/pull/3610).

#### TextureFormat Names

Some texture format names have changed to get back in line with the spec.

```diff
- TextureFormat::Bc6hRgbSfloat
+ TextureFormat::Bc6hRgbFloat
```

By @cwfitzgerald in [#3671](https://github.com/gfx-rs/wgpu/pull/3671).

#### Misc Breaking Changes

- Change type of `mip_level_count` and `array_layer_count` (members of `TextureViewDescriptor` and `ImageSubresourceRange`) from `Option<NonZeroU32>` to `Option<u32>`. By @teoxoy in [#3445](https://github.com/gfx-rs/wgpu/pull/3445)
- Change type of `bytes_per_row` and `rows_per_image` (members of `ImageDataLayout`) from `Option<NonZeroU32>` to `Option<u32>`. By @teoxoy in [#3529](https://github.com/gfx-rs/wgpu/pull/3529)
- On Web, `Instance::create_surface_from_canvas()` and `create_surface_from_offscreen_canvas()` now take the canvas by value. By @daxpedda in [#3690](https://github.com/gfx-rs/wgpu/pull/3690)

### Added/New Features

#### General
- Added feature flags for ray-tracing (currently only hal): `RAY_QUERY` and `RAY_TRACING` @daniel-keitel (started by @expenses) in [#3507](https://github.com/gfx-rs/wgpu/pull/3507)

#### Vulkan

- Implemented basic ray-tracing api for acceleration structures, and ray-queries @daniel-keitel (started by @expenses) in [#3507](https://github.com/gfx-rs/wgpu/pull/3507)

#### Hal

- Added basic ray-tracing api for acceleration structures, and ray-queries @daniel-keitel (started by @expenses) in [#3507](https://github.com/gfx-rs/wgpu/pull/3507)


### Changes

#### General

- Added `TextureFormatFeatureFlags::MULTISAMPLE_X16`. By @Dinnerbone in [#3454](https://github.com/gfx-rs/wgpu/pull/3454)
- Added `BufferUsages::QUERY_RESOLVE`. By @JolifantoBambla in [#3489](https://github.com/gfx-rs/wgpu/pull/3489)
- Support stencil-only views and copying to/from combined depth-stencil textures. By @teoxoy in [#3436](https://github.com/gfx-rs/wgpu/pull/3436)
- Added `Features::SHADER_EARLY_DEPTH_TEST`. By @teoxoy in [#3494](https://github.com/gfx-rs/wgpu/pull/3494)
- All `fxhash` dependencies have been replaced with `rustc-hash`. By @james7132 in [#3502](https://github.com/gfx-rs/wgpu/pull/3502)
- Allow copying of textures with copy-compatible formats. By @teoxoy in [#3528](https://github.com/gfx-rs/wgpu/pull/3528)
- Improve attachment related errors. By @cwfitzgerald in [#3549](https://github.com/gfx-rs/wgpu/pull/3549)
- Make error descriptions all upper case. By @cwfitzgerald in [#3549](https://github.com/gfx-rs/wgpu/pull/3549)
- Don't include ANSI terminal color escape sequences in shader module validation error messages. By @jimblandy in [#3591](https://github.com/gfx-rs/wgpu/pull/3591)
- Report error messages from DXC compile. By @Davidster in [#3632](https://github.com/gfx-rs/wgpu/pull/3632)
- Error in native when using a filterable `TextureSampleType::Float` on a multisample `BindingType::Texture`. By @mockersf in [#3686](https://github.com/gfx-rs/wgpu/pull/3686)
- On Web, the size of the canvas is adjusted when using `Surface::configure()`. If the canvas was given an explicit size (via CSS), this will not affect the visual size of the canvas. By @daxpedda in [#3690](https://github.com/gfx-rs/wgpu/pull/3690)
- Added `Global::create_render_bundle_error`. By @jimblandy in [#3746](https://github.com/gfx-rs/wgpu/pull/3746)

#### WebGPU

- Implement the new checks for readonly stencils. By @JCapucho in [#3443](https://github.com/gfx-rs/wgpu/pull/3443)
- Reimplement `adapter|device_features`. By @jinleili in [#3428](https://github.com/gfx-rs/wgpu/pull/3428)
- Implement `command_encoder_resolve_query_set`. By @JolifantoBambla in [#3489](https://github.com/gfx-rs/wgpu/pull/3489)
- Add support for `Features::RG11B10UFLOAT_RENDERABLE`. By @mockersf in [#3689](https://github.com/gfx-rs/wgpu/pull/3689)

#### Vulkan
- Set `max_memory_allocation_size` via `PhysicalDeviceMaintenance3Properties`. By @jinleili in [#3567](https://github.com/gfx-rs/wgpu/pull/3567)
- Silence false-positive validation error about surface resizing. By @seabassjh in [#3627](https://github.com/gfx-rs/wgpu/pull/3627)

### Bug Fixes

#### General
- `copyTextureToTexture` src/dst aspects must both refer to all aspects of src/dst format. By @teoxoy in [#3431](https://github.com/gfx-rs/wgpu/pull/3431)
- Validate before extracting texture selectors. By @teoxoy in [#3487](https://github.com/gfx-rs/wgpu/pull/3487)
- Fix fatal errors (those which panic even if an error handler is set) not including all of the details. By @kpreid in [#3563](https://github.com/gfx-rs/wgpu/pull/3563)
- Validate shader location clashes. By @emilk in [#3613](https://github.com/gfx-rs/wgpu/pull/3613)
- Fix surfaces not being dropped until exit. By @benjaminschaaf in [#3647](https://github.com/gfx-rs/wgpu/pull/3647)

#### WebGPU
- Fix handling of `None` values for `depth_ops` and `stencil_ops` in `RenderPassDescriptor::depth_stencil_attachment`. By @niklaskorz in [#3660](https://github.com/gfx-rs/wgpu/pull/3660)
- Avoid using `WasmAbi` functions for WebGPU backend. By @grovesNL in [#3657](https://github.com/gfx-rs/wgpu/pull/3657)

#### DX12
- Use typeless formats for textures that might be viewed as srgb or non-srgb. By @teoxoy in [#3555](https://github.com/gfx-rs/wgpu/pull/3555)

#### GLES
- Set FORCE_POINT_SIZE if it is vertex shader with mesh consist of point list. By @REASY in [3440](https://github.com/gfx-rs/wgpu/pull/3440)
- Remove unwraps inside `surface.configure`. By @cwfitzgerald in [#3585](https://github.com/gfx-rs/wgpu/pull/3585)
- Fix `copy_external_image_to_texture`, `copy_texture_to_texture` and `copy_buffer_to_texture` not taking the specified index into account if the target texture is a cube map, 2D texture array or cube map array. By @daxpedda [#3641](https://github.com/gfx-rs/wgpu/pull/3641)
- Fix disabling of vertex attributes with non-consecutive locations. By @Azorlogh in [#3706](https://github.com/gfx-rs/wgpu/pull/3706)

#### Metal
- Fix metal erroring on an `array_stride` of 0. By @teoxoy in [#3538](https://github.com/gfx-rs/wgpu/pull/3538)
- `create_texture` returns an error if `new_texture` returns NULL. By @jinleili in [#3554](https://github.com/gfx-rs/wgpu/pull/3554)
- Fix shader bounds checking being ignored. By @FL33TW00D in [#3603](https://github.com/gfx-rs/wgpu/pull/3603)

#### Vulkan
- Treat `VK_SUBOPTIMAL_KHR` as `VK_SUCCESS` on Android due to rotation issues. By @James2022-rgb in [#3525](https://github.com/gfx-rs/wgpu/pull/3525)

### Examples
- Use `BufferUsages::QUERY_RESOLVE` instead of `BufferUsages::COPY_DST` for buffers used in `CommandEncoder::resolve_query_set` calls in `mipmap` example. By @JolifantoBambla in [#3489](https://github.com/gfx-rs/wgpu/pull/3489)

## v0.15.3 (2023-03-22)

### Bug Fixes

#### Metal
- Fix incorrect mipmap being sampled when using `MinLod <= 0.0` and `MaxLod >= 32.0` or when the fragment shader samples different Lods in the same quad. By @cwfitzgerald in [#3610](https://github.com/gfx-rs/wgpu/pull/3610).

#### GLES
- Fix `Vertex buffer is not big enough for the draw call.` for ANGLE/Web when rendering with instance attributes on a single instance. By @wumpf in [#3596](https://github.com/gfx-rs/wgpu/pull/3596)
- Reset all queue state between command buffers in a submit. By @jleibs [#3589](https://github.com/gfx-rs/wgpu/pull/3589)
- Reset the state of `SAMPLE_ALPHA_TO_COVERAGE` on queue reset. By @jleibs [#3589](https://github.com/gfx-rs/wgpu/pull/3589)


## wgpu-0.15.2 (2023-03-08)

### Bug Fixes

#### Metal
- Fix definition of `NSOperatingSystemVersion` to avoid potential crashes. By @grovesNL in [#3557](https://github.com/gfx-rs/wgpu/pull/3557)

#### GLES
- Enable `WEBGL_debug_renderer_info` before querying unmasked vendor/renderer to avoid crashing on emscripten in [#3519](https://github.com/gfx-rs/wgpu/pull/3519)

## wgpu-0.15.1 (2023-02-09)

### Changes

#### General
- Fix for some minor issues in comments on some features. By @Wumpf in [#3455](https://github.com/gfx-rs/wgpu/pull/3455)

#### Vulkan
- Improve format MSAA capabilities detection. By @jinleili in [#3429](https://github.com/gfx-rs/wgpu/pull/3429)

#### DX12
- Update gpu allocator to 0.22. By @Elabajaba in [#3447](https://github.com/gfx-rs/wgpu/pull/3447)

#### WebGPU
- Implement `CommandEncoder::clear_buffer`. By @raphlinus in [#3426](https://github.com/gfx-rs/wgpu/pull/3426)

### Bug Fixes

#### General
- Re-sort supported surface formats based on srgb-ness. By @cwfitzgerald in [#3444](https://github.com/gfx-rs/wgpu/pull/3444)

#### Vulkan
- Fix surface view formats validation error. By @jinleili in [#3432](https://github.com/gfx-rs/wgpu/pull/3432)

#### DX12
- Fix DXC validation issues when using a custom `dxil_path`. By @Elabajaba in [#3434](https://github.com/gfx-rs/wgpu/pull/3434)

#### GLES
- Unbind vertex buffers at end of renderpass. By @cwfitzgerald in [#3459](https://github.com/gfx-rs/wgpu/pull/3459)

#### WebGPU

- Reimplement `{adapter|device}_features`. By @jinleili in [#3428](https://github.com/gfx-rs/wgpu/pull/3428)

### Documentation

#### General
- Build for Wasm on docs.rs. By @daxpedda in [#3462](https://github.com/gfx-rs/wgpu/pull/3428)


## wgpu-0.15.0 (2023-01-25)

### Major Changes


#### WGSL Top-Level `let` is now `const`

All top level constants are now declared with `const`, catching up with the wgsl spec.

`let` is no longer allowed at the global scope, only within functions.

```diff
-let SOME_CONSTANT = 12.0;
+const SOME_CONSTANT = 12.0;
```

See https://github.com/gfx-rs/naga/blob/master/CHANGELOG.md#v011-2023-01-25 for smaller shader improvements.

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

`Instance::create_surface()` now returns `Result<Surface, CreateSurfaceError>` instead of `Surface`. This allows an error to be returned if the given window is a HTML canvas and obtaining a WebGPU or WebGL 2 context fails. (No other platforms currently report any errors through this path.) By @kpreid in [#3052](https://github.com/gfx-rs/wgpu/pull/3052/)

#### `Queue::copy_external_image_to_texture` on WebAssembly

A new api, `Queue::copy_external_image_to_texture`, allows you to create wgpu textures from various web image primitives. Specifically from `HtmlVideoElement`, `HtmlCanvasElement`, `OffscreenCanvas`, and `ImageBitmap`. This provides multiple low-copy ways of interacting with the browser. WebGL is also supported, though WebGL has some additional restrictions, represented by the `UNRESTRICTED_EXTERNAL_IMAGE_COPIES` downlevel flag. By @cwfitzgerald in [#3288](https://github.com/gfx-rs/wgpu/pull/3288)

#### Instance creation now takes `InstanceDescriptor` instead of `Backends`

`Instance::new()` and `hub::Global::new()` now take an `InstanceDescriptor` struct which contains both the existing `Backends` selection as well as a new `Dx12Compiler` field for selecting which Dx12 shader compiler to use.

```diff
- let instance = Instance::new(wgpu::Backends::all());
+ let instance = Instance::new(wgpu::InstanceDescriptor {
+     backends: wgpu::Backends::all(),
+     dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
+ });
```

`Instance` now also also implements `Default`, which uses `wgpu::Backends::all()` and `wgpu::Dx12Compiler::Fxc` for `InstanceDescriptor`

```diff
- let instance = Instance::new(wgpu::InstanceDescriptor {
-     backends: wgpu::Backends::all(),
-     dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
- });
+ let instance = Instance::default();
```

By @Elabajaba in [#3356](https://github.com/gfx-rs/wgpu/pull/3356)

#### Texture Format Reinterpretation

The new `view_formats` field in the `TextureDescriptor` is used to specify a list of formats the texture can be re-interpreted to in a texture view. Currently only changing srgb-ness is allowed (ex. `Rgba8Unorm` <=> `Rgba8UnormSrgb`).

```diff
let texture = device.create_texture(&wgpu::TextureDescriptor {
  // ...
  format: TextureFormat::Rgba8UnormSrgb,
+ view_formats: &[TextureFormat::Rgba8Unorm],
});
```

```diff
let config = wgpu::SurfaceConfiguration {
  // ...
  format: TextureFormat::Rgba8Unorm,
+ view_formats: vec![wgpu::TextureFormat::Rgba8UnormSrgb],
};
surface.configure(&device, &config);
```

#### MSAA x2 and x8 Support

Via the `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES` feature, MSAA x2 and x8 are now supported on textures. To query for x2 or x8 support, enable the feature and look at the texture format flags for the texture format of your choice.

By @39ali in [3140](https://github.com/gfx-rs/wgpu/pull/3140)

#### DXC Shader Compiler Support for DX12

You can now choose to use the DXC compiler for DX12 instead of FXC. The DXC compiler is faster, less buggy, and allows for new features compared to the old, unmaintained FXC compiler.

You can choose which compiler to use at `Instance` creation using the `dx12_shader_compiler` field in the `InstanceDescriptor` struct. Note that DXC requires both `dxcompiler.dll` and `dxil.dll`, which can be downloaded from https://github.com/microsoft/DirectXShaderCompiler/releases. Both .dlls need to be shipped with your application when targeting DX12 and using the `DXC` compiler. If the .dlls can't be loaded, then it will fall back to the FXC compiler. By @39ali and @Elabajaba in [#3356](https://github.com/gfx-rs/wgpu/pull/3356)

#### Suballocate DX12 buffers and textures

The DX12 backend can now suballocate buffers and textures from larger chunks of memory, which can give a significant increase in performance (in testing a 100x improvement has been seen in a simple scene with 200 `write_buffer` calls per frame, and a 1.4x improvement in [Bistro using Bevy](https://github.com/vleue/bevy_bistro_playground)).

Previously `wgpu-hal`'s DX12 backend created a new heap on the GPU every time you called `write_buffer` (by calling `CreateCommittedResource`), whereas now it uses [`gpu_allocator`](https://crates.io/crates/gpu-allocator) to manage GPU memory (and calls `CreatePlacedResource` with a suballocated heap). By @Elabajaba in [#3163](https://github.com/gfx-rs/wgpu/pull/3163)

#### Backend selection by features in wgpu-core

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

### Changes

#### General

- Convert all `Default` Implementations on Enums to `derive(Default)`
- Implement `Default` for `CompositeAlphaMode`
- New downlevel feature `UNRESTRICTED_INDEX_BUFFER` to indicate support for using `INDEX` together with other non-copy/map usages (unsupported on WebGL). By @Wumpf in [#3157](https://github.com/gfx-rs/wgpu/pull/3157)
- Add missing `DEPTH_BIAS_CLAMP` and `FULL_DRAW_INDEX_UINT32` downlevel flags. By @teoxoy in [#3316](https://github.com/gfx-rs/wgpu/pull/3316)
- Combine `Surface::get_supported_formats`, `Surface::get_supported_present_modes`, and `Surface::get_supported_alpha_modes` into `Surface::get_capabilities` and `SurfaceCapabilities`. By @cwfitzgerald in [#3157](https://github.com/gfx-rs/wgpu/pull/3157)
- Make `Surface::get_default_config` return an Option to prevent panics. By @cwfitzgerald in [#3157](https://github.com/gfx-rs/wgpu/pull/3157)
- Lower the `max_buffer_size` limit value for compatibility with Apple2 and WebGPU compliance. By @jinleili in [#3255](https://github.com/gfx-rs/wgpu/pull/3255)
- Limits `min_uniform_buffer_offset_alignment` and `min_storage_buffer_offset_alignment` is now always at least 32. By @wumpf [#3262](https://github.com/gfx-rs/wgpu/pull/3262)
- Dereferencing a buffer view is now marked inline. By @Wumpf in [#3307](https://github.com/gfx-rs/wgpu/pull/3307)
- The `strict_assert` family of macros was moved to `wgpu-types`. By @i509VCB in [#3051](https://github.com/gfx-rs/wgpu/pull/3051)
- Make `ObjectId` structure and invariants idiomatic. By @teoxoy in [#3347](https://github.com/gfx-rs/wgpu/pull/3347)
- Add validation in accordance with WebGPU `GPUSamplerDescriptor` valid usage for `lodMinClamp` and `lodMaxClamp`. By @James2022-rgb in [#3353](https://github.com/gfx-rs/wgpu/pull/3353)
- Remove panics in `Deref` implementations for `QueueWriteBufferView` and `BufferViewMut`. Instead, warnings are logged, since reading from these types is not recommended. By @botahamec in [#3336]
- Implement `view_formats` in the TextureDescriptor to match the WebGPU spec. By @jinleili in [#3237](https://github.com/gfx-rs/wgpu/pull/3237)
- Show more information in error message for non-aligned buffer bindings in WebGL [#3414](https://github.com/gfx-rs/wgpu/pull/3414)
- Update `TextureView` validation according to the WebGPU spec. By @teoxoy in [#3410](https://github.com/gfx-rs/wgpu/pull/3410)
- Implement `view_formats` in the SurfaceConfiguration to match the WebGPU spec. By @jinleili in [#3409](https://github.com/gfx-rs/wgpu/pull/3409)

#### Vulkan

- Set `WEBGPU_TEXTURE_FORMAT_SUPPORT` downlevel flag depending on the proper format support by @teoxoy in [#3367](https://github.com/gfx-rs/wgpu/pull/3367).
- Set `COPY_SRC`/`COPY_DST` only based on Vulkan's `TRANSFER_SRC`/`TRANSFER_DST` by @teoxoy in [#3366](https://github.com/gfx-rs/wgpu/pull/3366).

#### GLES

- Browsers that support `OVR_multiview2` now report the `MULTIVIEW` feature by @expenses in [#3121](https://github.com/gfx-rs/wgpu/pull/3121).
- `Limits::max_push_constant_size` on GLES is now 256 by @Dinnerbone in [#3374](https://github.com/gfx-rs/wgpu/pull/3374).
- Creating multiple pipelines with the same shaders will now be faster, by @Dinnerbone in [#3380](https://github.com/gfx-rs/wgpu/pull/3380).

#### WebGPU

- Implement `queue_validate_write_buffer` by @jinleili in [#3098](https://github.com/gfx-rs/wgpu/pull/3098)
- Sync depth/stencil copy restrictions with the spec by @teoxoy in [#3314](https://github.com/gfx-rs/wgpu/pull/3314)


### Added/New Features

#### General

- Implement `Hash` for `DepthStencilState` and `DepthBiasState`
- Add the `"wgsl"` feature, to enable WGSL shaders in `wgpu-core` and `wgpu`. Enabled by default in `wgpu`. By @daxpedda in [#2890](https://github.com/gfx-rs/wgpu/pull/2890).
- Implement `Clone` for `ShaderSource` and `ShaderModuleDescriptor` in `wgpu`. By @daxpedda in [#3086](https://github.com/gfx-rs/wgpu/pull/3086).
- Add `get_default_config` for `Surface` to simplify user creation of `SurfaceConfiguration`. By @jinleili in [#3034](https://github.com/gfx-rs/wgpu/pull/3034)
- Improve compute shader validation error message. By @haraldreingruber in [#3139](https://github.com/gfx-rs/wgpu/pull/3139)
- Native adapters can now use MSAA x2 and x8 if it's supported , previously only x1 and x4 were supported . By @39ali in [3140](https://github.com/gfx-rs/wgpu/pull/3140)
- Implemented correleation between user timestamps and platform specific presentation timestamps via [`Adapter::get_presentation_timestamp`]. By @cwfitzgerald in [#3240](https://github.com/gfx-rs/wgpu/pull/3240)
- Added support for `Features::SHADER_PRIMITIVE_INDEX` on all backends. By @cwfitzgerald in [#3272](https://github.com/gfx-rs/wgpu/pull/3272)
- Implemented `TextureFormat::Stencil8`, allowing for stencil testing without depth components. By @Dinnerbone in [#3343](https://github.com/gfx-rs/wgpu/pull/3343)
- Implemented `add_srgb_suffix()` for `TextureFormat` for converting linear formats to sRGB. By @Elabajaba in [#3419](https://github.com/gfx-rs/wgpu/pull/3419)
- Zero-initialize workgroup memory. By @teoxoy in [#3174](https://github.com/gfx-rs/wgpu/pull/3174)

#### GLES

- Surfaces support now `TextureFormat::Rgba8Unorm` and (non-web only) `TextureFormat::Bgra8Unorm`. By @Wumpf in [#3070](https://github.com/gfx-rs/wgpu/pull/3070)
- Support alpha to coverage. By @Wumpf in [#3156](https://github.com/gfx-rs/wgpu/pull/3156)
- Support filtering f32 textures. By @expenses in [#3261](https://github.com/gfx-rs/wgpu/pull/3261)

#### Vulkan

- Add `SHADER_INT16` feature to enable the `shaderInt16` VkPhysicalDeviceFeature. By @Elabajaba in [#3401](https://github.com/gfx-rs/wgpu/pull/3401)

#### WebGPU

- Add `MULTISAMPLE_X2`, `MULTISAMPLE_X4` and `MULTISAMPLE_X8` to `TextureFormatFeatureFlags`. By @39ali in [3140](https://github.com/gfx-rs/wgpu/pull/3140)
- Sync `TextureFormat.describe` with the spec. By @teoxoy in [3312](https://github.com/gfx-rs/wgpu/pull/3312)

#### Metal
- Add a way to create `Device` and `Queue` from raw Metal resources in wgpu-hal. By @AdrianEddy in [#3338](https://github.com/gfx-rs/wgpu/pull/3338)

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
- Validate texture ids in `Global::queue_texture_write`. By @jimblandy in [#3378](https://github.com/gfx-rs/wgpu/pull/3378).
- Don't panic on mapped buffer in queue_submit. By @crowlKats in [#3364](https://github.com/gfx-rs/wgpu/pull/3364).
- Fix being able to sample a depth texture with a filtering sampler. By @teoxoy in [#3394](https://github.com/gfx-rs/wgpu/pull/3394).
- Make `make_spirv_raw` and `make_spirv` handle big-endian binaries. By @1e1001 in [#3411](https://github.com/gfx-rs/wgpu/pull/3411).

#### Vulkan
- Update ash to 0.37.1+1.3.235 to fix CI breaking by changing a call to the deprecated `debug_utils_set_object_name()` function to `set_debug_utils_object_name()` by @elabajaba in [#3273](https://github.com/gfx-rs/wgpu/pull/3273)
- Document and improve extension detection. By @teoxoy in [#3327](https://github.com/gfx-rs/wgpu/pull/3327)
- Don't use a pointer to a local copy of a `PhysicalDeviceDriverProperties` struct after it has gone out of scope. In fact, don't make a local copy at all. Introduce a helper function for building `CStr`s from C character arrays, and remove some `unsafe` blocks. By @jimblandy in [#3076](https://github.com/gfx-rs/wgpu/pull/3076).

#### DX12

- Fix `depth16Unorm` formats by @teoxoy in [#3313](https://github.com/gfx-rs/wgpu/pull/3313)
- Don't re-use `GraphicsCommandList` when `close` or `reset` fails. By @xiaopengli89 in [#3204](https://github.com/gfx-rs/wgpu/pull/3204)

#### Metal
- Fix texture view creation with full-resource views when using an explicit `mip_level_count` or `array_layer_count`. By @cwfitzgerald in [#3323](https://github.com/gfx-rs/wgpu/pull/3323)

#### GLES

- Fixed WebGL not displaying srgb targets correctly if a non-screen filling viewport was previously set. By @Wumpf in [#3093](https://github.com/gfx-rs/wgpu/pull/3093)
- Fix disallowing multisampling for float textures if otherwise supported. By @Wumpf in [#3183](https://github.com/gfx-rs/wgpu/pull/3183)
- Fix a panic when creating a pipeline with opaque types other than samplers (images and atomic counters). By @James2022-rgb in [#3361](https://github.com/gfx-rs/wgpu/pull/3361)
- Fix uniform buffers being empty on some vendors. By @Dinnerbone in [#3391](https://github.com/gfx-rs/wgpu/pull/3391)
- Fix a panic allocating a new buffer on webgl. By @Dinnerbone in [#3396](https://github.com/gfx-rs/wgpu/pull/3396)

#### WebGPU

- Use `log` instead of `println` in hello example by @JolifantoBambla in [#2858](https://github.com/gfx-rs/wgpu/pull/2858)

#### deno-webgpu

- Let `setVertexBuffer` and `setIndexBuffer` calls on
  `GPURenderBundleEncoder` throw an error if the `size` argument is
  zero, rather than treating that as "until the end of the buffer".
  By @jimblandy in [#3171](https://github.com/gfx-rs/wgpu/pull/3171)

#### Emscripten

- Let the wgpu examples `framework.rs` compile again under Emscripten. By @jimblandy in [#3246](https://github.com/gfx-rs/wgpu/pull/3246)

### Examples

- Log adapter info in hello example on wasm target by @JolifantoBambla in [#2858](https://github.com/gfx-rs/wgpu/pull/2858)
- Added new example `stencil-triangles` to show basic use of stencil testing. By @Dinnerbone in [#3343](https://github.com/gfx-rs/wgpu/pull/3343)

### Testing/Internal

- Update the `minimum supported rust version` to 1.64
- Move `ResourceMetadata` into its own module. By @jimblandy in [#3213](https://github.com/gfx-rs/wgpu/pull/3213)
- Add WebAssembly testing infrastructure. By @haraldreingruber in [#3238](https://github.com/gfx-rs/wgpu/pull/3238)
- Error message when you forget to use cargo-nextest. By @cwfitzgerald in [#3293](https://github.com/gfx-rs/wgpu/pull/3293)
- Fix all suggestions from `cargo clippy`

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
- Fix compilation errors when using wgpu-core in isolation while targeting `wasm32-unknown-unknown` by @Seamooo in [#2922](https://github.com/gfx-rs/wgpu/pull/2922)
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

- Fix `astc_hdr` formats support by @jinleili in [#2971]](https://github.com/gfx-rs/wgpu/pull/2971)
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

Manual concatenation of `cargo public-api --diff-git-checkouts v0.13.2 v0.14.0 -p wgpu` and `cargo public-api --diff-git-checkouts v0.13.2 v0.14.0 -p wgpu-types`

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
- Add DEPTH32FLOAT_STENCIL8 feature by @jinleili in [#2664](https://github.com/gfx-rs/wgpu/pull/2664)
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
- feature = emscripten, compatibility fixes for wgpu-native by @caiiiycuk in [#2450](https://github.com/gfx-rs/wgpu/pull/2450)

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
  - Fixed output spirv requiring the "kernel" capability.
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
  - (beta) targeting Wasm with WebGL backend
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
