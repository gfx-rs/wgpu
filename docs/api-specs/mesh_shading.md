# Mesh Shader Extensions

🧪Experimental🧪

`wgpu` supports an experimental version of mesh shading. The extensions allow for acceleration structures to be created and built (with
`Features::EXPERIMENTAL_MESH_SHADER` enabled) and interacted with in shaders. Currently `naga` has no support for mesh shaders beyond recognizing the additional shader stages.
For this reason, all shaders must be created with `Device::create_shader_module_passthrough`.

**Note**: The features documented here may have major bugs in them and are expected to be subject
to breaking changes, suggestions for the API exposed by this should be posted on [the mesh-shading issue](https://github.com/gfx-rs/wgpu/issues/7197).

***This is not*** a thorough explanation of mesh shading and how it works. Those wishing to understand mesh shading more broadly should look elsewhere first.

## `wgpu` API

### New `wgpu` functions

`Device::create_mesh_pipeline` - Creates a mesh shader pipeline. This is very similar to creating a standard render pipeline, except that it takes a mesh shader state and optional task shader state instead of a vertex state. If the task state is omitted, during rendering the number of workgroups is passed directly from the draw call to the mesh shader state, with an empty payload.

`RenderPass::draw_mesh_tasks` - Dispatches the mesh shader pipeline. This ignores render pipeline specific information, such as vertex buffer bindings and index buffer bindings. The dispatch size must adhere to the limits described below.

`RenderPass::draw_mesh_tasks_indirect`, `RenderPass::multi_draw_mesh_tasks_indirect` and `RenderPass::multi_draw_mesh_tasks_indirect_count` - Dispatches the mesh shader pipeline with dispatch size taken from a buffer. This ignores render pipeline specific information, such as vertex buffer bindings and index buffer bindings. The dispatch size must adhere to the limits described below. Analogous to `draw_indirect`, `multi_draw_indirect` and `multi_draw_indirect_count`. Requires the corresponding indirect feature to be enabled.

An example of using mesh shaders to render a single triangle can be seen [here](../../examples/features/src/mesh_shader).

### Features
* Using mesh shaders requires enabling `Features::EXPERIMENTAL_MESH_SHADER`.
* Using mesh shaders with multiview requires enabling `Features::EXPERIMENTAL_MESH_SHADER_MULTIVIEW`.
* Currently, only triangle rendering is tested
* Line rendering is supported but untested
* Point rendering is supported on vulkan. It is impossible on DirectX. Metal support hasn't been checked.
* Queries are unsupported

### Limits

> **NOTE**: More limits will be added when support is added to `naga`.

* `Limits::max_task_workgroup_total_count` - the maximum total number of workgroups from a `draw_mesh_tasks` command or similar. The dimensions passed must be less than or equal to this limit when multiplied together.
* `Limits::max_task_workgroups_per_dimension` - the maximum for each of the 3 workgroup dimensions in a `draw_mesh_tasks` command. Each dimension passed must be less than or equal to this limit.
* `max_mesh_multiview_count` - The maximum number of views used when multiview rendering with a mesh shader pipeline.
* `max_mesh_output_layers` - the maximum number of output layers for a mesh shader pipeline.

### Backend specific information
* Unlike Vulkan, DirectX doesn't support point rendering
* Only Vulkan is currently supported
* DirectX 12 support is planned
* Metal support is desired but not planned


## Naga implementation

### Supported frontends
* 🛠️ WGSL
* 🛠️ SPIR-V
* ❌ GLSL

### Supported backends
* 🛠️ SPIR-V
* 🛠️ HLSL
* ❌ WGSL
* ❌ GLSL
* ❌ MSL

## `WGSL` extension specification
