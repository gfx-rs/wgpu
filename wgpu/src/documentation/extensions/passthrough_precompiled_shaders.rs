/*!
# Passthrough and Precompiled Shaders

Sometimes, you may need access to a feature that wgpu doesn't support, or you may wish to compile pipelines
ahead of time to avoid shipping naga and/or dxc. wgpu's passthrough shaders support this goal.

They allow creating shaders directly from backend code without involving any non-driver shader compilers.

## Use cases

### Unsupported features

If wgpu/naga doesn't yet support some shader feature, you may be able to use passthrough shaders to work around this.
Details will differ per-backend, for example on Vulkan you may need to manually enable certain extensions at device
creation.

This won't work for all features. For example, extra shader stages or other features which require knowledge of the
feature at pipeline creation time won't work.

It is perfect for features that stay entirely on the GPU. For example, new atomic operations or new scalar types.

### Skipping runtime compilation of shaders

Compiling shaders at runtime adds to startup time significantly. By compiling shaders ahead of time,

Using precompiled shaders also means you don't have to ship any shader compiler. The removal of naga can reduce
your binary size slightly, though the bigger benefit comes from avoiding shipping DXC. DX12 drivers only accept
DXIL, which naga cannot yet produce, so it has to invoke the DXC shader compiler (or sometimes FXC, which has
its own drawbacks).

DXC can either be linked statically or dynamically. Shipping a dynamically linked DXC
can be a pain and add to build complexity, and if it stops working, and without proper configuration your app
may silently switch to FXC. Statically linking DXC with wgpu's `static-dxc` feature depends on a relatively
small GitHub repo that completely hijacks DXC's build process, and you are trusting their CI artifacts. Both
methods of linking typically require pinning and using a prebuilt library from GitHub, which you must trust,
and require shipping another 50MB+ of binaries.

Note that in order to effectively reduce startup costs, you should pair this with pipeline caches. Driver shader
compilers typically take far longer than naga or DXC/FXC. Luckily, this is not absolutely critical, since many
drivers effectively keep pipeline caches even for apps that don't explicit manage them.

The net effect of using entirely precompiled shaders in conjunction with pipeline caches is a smaller and
self-contained binary, with a much faster startup time, especially after the first launch.

## wgpu API

**TODO: REWORK WGPU API TO SUPPORT MORE REFLECTION INFO**

## Generating reflection information from SPIR-V shaders

**TODO: SECTION ABOUT USING RSPIRV TO GENERATE SPIRV REFLECTION FOR VALIDATION**

## Precompiling shaders

**TODO: STANDALONE EXAMPLE WITH PRECOMPILED SHADERS**

**TODO: LIBRARY FOR PRECOMPILING SHADERS**

## General validation reflection information

The following is optional reflection information that wgpu can use to validate shader usage at runtime.

**EVERYTHING THAT FOLLOWS IS AI GENERATED AND UNREVIEWED**

For WGSL and other naga-parsed shaders, wgpu-core gathers all of this from the naga module and checks it
against the pipeline descriptor and pipeline layout at pipeline creation. For passthrough shaders, wgpu
currently only knows the entry point names, so none of these checks happen and a mismatch is undefined
behavior. Every item below that is provided lets wgpu turn some of that undefined behavior back into a
validation error.

### Entry points

- **Name and shader stage** of each entry point. Without the stage, wgpu cannot pick a default entry point
  for a stage, and cannot reject a module whose entry point is used for the wrong stage.
- **Workgroup size** of each compute, task and mesh entry point. This is checked against
  [`Limits::max_compute_workgroup_size_x`](crate::Limits::max_compute_workgroup_size_x) (and `_y`, `_z`),
  [`Limits::max_compute_invocations_per_workgroup`](crate::Limits::max_compute_invocations_per_workgroup)
  and the equivalent task and mesh limits.
- **Total workgroup memory** used by each entry point, to check against
  [`Limits::max_compute_workgroup_storage_size`](crate::Limits::max_compute_workgroup_storage_size).
- **Required features**: the [`Features`](crate::Features) the shader depends on, such as `SHADER_F16`,
  `SUBGROUP`, `SHADER_INT64`, `EXPERIMENTAL_RAY_QUERY` or `EXPERIMENTAL_MESH_SHADER`. This lets wgpu reject a
  shader that needs a feature the device was not created with, instead of passing it to the driver.

### Resource bindings

Each global resource, identified by its `(group, binding)` pair, and the list of entry points that
statically use it. For each resource:

- **Buffers**: whether it is a uniform or storage buffer, whether a storage buffer is read-only or
  read-write, and the minimum binding size. For a buffer that ends in a runtime-sized array, the minimum
  size is the size of the fixed part plus one array element. The minimum size is checked against
  [`BindingType::Buffer::min_binding_size`](crate::BindingType::Buffer), or checked at draw or dispatch time
  when the layout has no minimum size.
- **Sampled textures**: view dimension, whether it is multisampled, and sample type (float, depth, sint or
  uint).
- **Storage textures**: view dimension, texel format and access mode.
- **External textures**.
- **Samplers**: whether it is a comparison sampler.
- **Acceleration structures**: whether vertex return is used.
- **Binding arrays**: the element count, or that the array is unsized.

This information lets wgpu check that the pipeline layout provides every binding the shader uses, with a
compatible type and visibility. It is also what wgpu needs to derive a pipeline layout when `layout` is
`None`. Currently, creating a pipeline with a derived layout from passthrough shaders is an error.

### Texture and sampler pairs

The `(texture, sampler)` pairs that each entry point samples together. wgpu uses these to check that
filtering samplers are only used with filterable textures, and that comparison samplers are only used with
depth textures.

### Immediate data

The size of the immediate data block used by each entry point. This is checked against the
`immediate_size` of the pipeline layout.

### Stage inputs and outputs

- **Vertex inputs**: the location and numeric type of each input. These are checked against the vertex
  buffer attributes of the pipeline.
- **Inter-stage variables**: the location, numeric type, interpolation type, sampling mode and whether it is
  per-primitive. Vertex (or mesh) outputs are matched with fragment inputs, and the number of variables is
  checked against
  [`Limits::max_inter_stage_shader_variables`](crate::Limits::max_inter_stage_shader_variables).
- **Fragment outputs**: the location, numeric type and blend source index of each output. These are checked
  against the color target formats and
  [`Limits::max_color_attachments`](crate::Limits::max_color_attachments), and are used to validate dual
  source blending.
- **Built-ins** that affect pipeline validation:
  - `frag_depth` requires a depth attachment.
  - `primitive_index` must be produced by the previous stage when it is a mesh shader.
  - `view_index` relates to the pipeline's multiview mask.
  - `clip_distances` and its array size.
  - `draw_index` cannot be used by a mesh shader in a pipeline that has a task shader.
  - `barycentric` and per-vertex fragment inputs require a triangle topology.

### Mesh and task shaders

- The maximum number of output vertices and primitives of a mesh shader, checked against
  [`Limits::max_mesh_output_vertices`](crate::Limits::max_mesh_output_vertices) and
  [`Limits::max_mesh_output_primitives`](crate::Limits::max_mesh_output_primitives).
- The output primitive topology of a mesh shader, which must match the pipeline's primitive topology.
- The task payload size of task and mesh shaders. This is checked against
  [`Limits::max_task_payload_size`](crate::Limits::max_task_payload_size), and the payload sizes of the task
  and mesh shader must match.

### Pipeline-overridable constants

The name or ID, type and default value of each overridable constant. This would let wgpu check the
`constants` passed in a pipeline descriptor. Currently, no backend applies `constants` to passthrough
shaders.

### Runtime checks

These properties are not reflection information, and wgpu cannot check them. For naga shaders, wgpu
inserts code to make the shader safe to run. For passthrough shaders, the shader author is responsible for
the following, controlled by [`ShaderRuntimeChecks`](crate::ShaderRuntimeChecks) for naga shaders:

- Bounds checks on buffer, array and texture accesses.
- Forcing every loop to terminate.
- Integer division and remainder checks.
- Ray query initialization tracking.
- Task shader dispatch size limits.
- Mesh shader primitive index clamping.

The `zero_initialize_workgroup_memory` field of a pipeline stage is also ignored for passthrough shaders.

## Backend specific reflection information

The following is reflection information which is needed for specific backends. They may or may not be
optional. For example, GLSL may require reflection information in order to use immediate data, while
SPIR-V may have optional validation that checks whether the current device supports each capability
used by a shader.

Several backends do not use the `(group, binding)` numbers of a bind group layout directly. Instead, they
assign their own binding slots when the pipeline layout is created. A passthrough shader must either use
the slots that wgpu assigns, or provide reflection that lets wgpu map between the two.

### SPIR-V

- **Binding remapping**: the descriptor set is the bind group index. The binding is *not* the wgpu binding
  number. It is the index of the entry in the bind group layout, with entries sorted by binding number. An
  external texture uses 4 consecutive bindings (3 sampled images and 1 uniform buffer). If the bindings
  of a group are contiguous from 0 and there are no external textures, the two numbers are the same.
  Reflection of the `DescriptorSet` and `Binding` decorations would let wgpu validate this, or patch the
  decorations to match the layout.
- **Immediate data**: wgpu creates one push constant range at offset 0 that covers every stage, with the
  size of the pipeline layout's immediate data.
- **Specialization constants**: a map from each overridable constant to its `SpecId` and type, so that
  `constants` can be turned into `VkSpecializationInfo`.
- **Capabilities and extensions**: the `OpCapability` and `OpExtension` instructions used. These can be
  checked against the features and extensions that wgpu enabled on the device.
- **SPIR-V version**: must not be newer than the device supports. This is 1.0 for Vulkan 1.0, 1.3 for Vulkan
  1.1, 1.5 for Vulkan 1.2 and 1.6 for Vulkan 1.3. With `VK_KHR_spirv_1_4`, it is at least 1.4.
- **Subgroup size**: wgpu sets `ALLOW_VARYING_SUBGROUP_SIZE` when `SUBGROUP` is enabled. A shader that
  requires full subgroups or a specific subgroup size would need to report it.
- **Point size**: a vertex shader used with a point list topology must write `PointSize`.
- **Interface locations**: vertex input locations are the `shader_location` of each vertex attribute.
  Fragment output locations are color target indices, with `Index 1` used for the second dual source
  blending output.

### HLSL/DXIL

- **Register assignment**: wgpu builds a root signature from the pipeline layout, and assigns `b`, `t` and
  `u` registers in order of bind group, then binding number. Read-only storage buffers, sampled textures and
  acceleration structures use `t` registers. Read-write storage buffers and storage textures use `u`
  registers. Uniform buffers use `b` registers. Each external texture uses 3 `t` registers and 1 `b`
  register. Immediate data is a root constant buffer at `b0, space0`, and when it exists, all other `b`
  registers move to `space1`. A shader must use the registers wgpu assigns, or report its registers so
  that wgpu can check them or build the root signature to match. DXIL cannot easily be patched.
- **Samplers**: samplers are not bound directly. All samplers live in one global sampler heap, bound as a
  `SamplerState` array at `s0` to `s2047` and a `SamplerComparisonState` array at `s2048` to `s4095`. Each
  bind group that contains samplers has a `StructuredBuffer<uint>` in a `t` register that maps the group's
  samplers, in binding order, to indices into the heap.
- **Dynamic offsets**: dynamic uniform buffers are root CBVs. Dynamic storage buffers stay in the descriptor
  table, and their offsets are passed as root constants in a `b` register, which the shader must add itself.
- **Special constants**: HLSL has no equivalent of `first_vertex`, `first_instance` or `num_workgroups`,
  and `SV_VertexID` and `SV_InstanceID` do not include the first vertex and first instance. wgpu passes
  these values in a root constant buffer of three `uint`s. The shader needs to report whether it uses
  them, and where it expects the buffer.
- **Shader model**: the shader model must be supported by the device. HLSL passthrough shaders are compiled
  with the shader model wgpu picked for the device, so compiling also needs the entry point and stage.
- **DXIL signing**: DXIL must be validated and signed, as DXC does by default.
- **Interface semantics**: vertex inputs use the semantic `LOC<n>`, where `n` is the `shader_location` of
  the attribute. D3D12 requires the signatures of adjacent stages to match. When both stages are naga
  shaders, naga removes vertex outputs that the fragment shader does not read. This does not happen when
  either stage is a passthrough shader.

### MSL/Metallib

- **Workgroup size** (currently required): MSL does not declare the threads per threadgroup in the shader.
  It is given when dispatching, so wgpu must know it for every compute, task and mesh entry point.
- **Slot assignment**: wgpu assigns buffer, texture and sampler slots separately for each stage, counting
  only the bind group layout entries visible to that stage, in order of bind group, then binding number.
  When the pipeline layout has immediate data, it occupies buffer slot 0. Each external texture uses 3
  texture slots and 1 buffer slot. Each binding array uses 1 buffer slot for an argument buffer.
- **Buffer sizes**: shaders need the size of runtime-sized buffers to implement bounds checks and
  `arrayLength`. wgpu passes these sizes in a buffer after the last buffer slot, for each stage that can
  see a storage buffer, and always for the vertex stage. The shader needs to report which bindings it
  needs sizes for, so wgpu can fill the buffer in a matching order.
- **Threadgroup memory**: threadgroup memory declared as an entry point argument, at
  `[[threadgroup(n)]]`, needs its size reported so wgpu can call `setThreadgroupMemoryLength`. The same is
  needed for the object shader payload.
- **Immutable buffers** (optional): the buffers that are never written. wgpu marks these as immutable,
  which can improve performance.
- **Vertex buffers**: vertex buffer `i` is bound at buffer slot `30 - i`, and attributes use
  `[[attribute(n)]]` where `n` is the `shader_location`.
- **Function constants**: a map from each overridable constant to its `[[function_constant(n)]]` index and
  type.
- **MSL version**: MSL passthrough shaders are compiled with the MSL version wgpu picked for the device, and
  `preserveInvariance` is not set. A metallib must be built for an OS version and GPU family the device
  supports.
- **Point size**: a vertex shader used with a point list topology must write `[[point_size]]`.

### GLSL

- **Version**: the `#version` directive must match the shading language version of the context, such as
  `300 es` for WebGL 2.
- **Entry points**: each module has exactly one entry point, `main`, which is used for one stage. If a
  pipeline has no fragment shader, wgpu supplies an empty one.
- **Slot assignment**: wgpu assigns slots separately for samplers, textures, storage textures, uniform
  buffers and storage buffers, in order of bind group, then binding number. A binding array uses one slot
  per element.
- **Combined texture samplers**: GLSL has no separate sampler objects. wgpu binds the sampler for each
  texture slot, so the shader needs to report which sampler is used with each texture.
- **Binding names**: if the context does not support `layout(binding = n)`, wgpu sets bindings by name after
  linking. This needs the name of each uniform block and each texture or image uniform, mapped to its
  `(group, binding)`. Storage buffers cannot be remapped this way.
- **Immediate data**: immediate data is emulated with plain uniforms. wgpu needs the name, offset, size and
  type of every uniform that makes up the immediate data block.
- **First instance**: if the shader reads `gl_InstanceID` and needs it to include the first instance, it
  must read a uniform named `naga_vs_first_instance`, which wgpu sets when drawing.
- **Clip distances**: the number of clip distances a vertex shader writes, so wgpu can enable the matching
  `GL_CLIP_DISTANCEi` states.
- **Interface locations**: if the context does not support explicit locations, wgpu needs the names of the
  vertex inputs and fragment outputs, mapped to their locations.
- **Coordinate space**: naga flips the Y coordinate of the output position and remaps depth from `[0, 1]`
  to `[-1, 1]`. A passthrough vertex shader must do the same to match other backends.
- **Multiview**: a shader used with a multiview pipeline must declare the number of views.

 */
