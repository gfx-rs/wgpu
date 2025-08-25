# Mesh Shader Extensions

🧪Experimental🧪

`wgpu` supports an experimental version of mesh shading when `Features::EXPERIMENTAL_MESH_SHADER` is enabled.
Currently `naga` has no support for parsing or writing mesh shaders.
For this reason, all shaders must be created with `Device::create_shader_module_passthrough`.

**Note**: The features documented here may have major bugs in them and are expected to be subject
to breaking changes, suggestions for the API exposed by this should be posted on [the mesh-shading issue](https://github.com/gfx-rs/wgpu/issues/7197).

## Mesh shaders overview

### What are mesh shaders
Mesh shaders are a new kind of rasterization pipeline intended to address some of the shortfalls with the vertex shader pipeline. The core idea of mesh shaders is that the GPU decides how to render the many small parts of a scene instead of the CPU issuing a draw call for every small part or issuing an inefficient monolithic draw call for a large part of the scene.

Mesh shaders are specifically designed to be used with **meshlet rendering**, a technique where every object is split into many subobjects called meshlets that are each rendered with their own parameters. With the standard vertex pipeline, each draw call specifies an exact number of primitives to render and the same parameters for all vertex shaders on an entire object (or even multiple objects). This doesn't leave room for different LODs for different parts of an object, for example a closer part having more detail, nor does it allow culling smaller sections (or primitives) of objects. With mesh shaders, each task workgroup might get assigned to a single object. It can then analyze the different meshlets(sections) of that object, determine which are visible and should actually be rendered, and for those meshlets determine what LOD to use based on the distance from the camera. It can then dispatch a mesh workgroup for each meshlet, with each mesh workgroup then reading the data for that specific LOD of its meshlet, determining which and how many vertices and primitives to output, determining which remaining primitives need to be culled, and passing the resulting primitives to the rasterizer.

Mesh shaders are most effective in scenes with many polygons. They can allow skipping processing of entire groups of primitives that are facing away from the camera or otherwise occluded, which reduces the number of primitives that need to be processed by more than half in most cases, and they can reduce the number of primitives that need to be processed for more distant objects. Scenes that are not bottlenecked by geometry (perhaps instead by fragment processing or post processing) will not see much benefit from using them.

Mesh shaders were first shown off in [NVIDIA's asteroids demo](https://www.youtube.com/watch?v=CRfZYJ_sk5E). Now, they form the basis for [Unreal Engine's Nanite](https://www.unrealengine.com/en-US/blog/unreal-engine-5-is-now-available-in-preview#Nanite).

### Mesh shader pipeline
A mesh shader pipeline is just like a standard render pipeline, except that the vertex shader stage is replaced by a mesh shader stage (and optionally a task shader stage). This functions as follows:

* If there is a task shader stage, task shader workgroups are invoked first, with the number of workgroups determined by the draw call. Each task shader workgroup outputs a workgroup size and a task payload. A dispatch group of mesh shaders with the given workgroup size is then invoked with the task payload as a parameter.
* Otherwise, a single dispatch group of mesh shaders with workgroup size given by the draw call is invoked.
* Each mesh shader dispatch group functions exactly as a compute dispatch group, except that it has special outputs and may take a task payload as input. Mesh dispatch groups invoked by different task shader workgroups cannot interact.
* Each workgroup within the mesh shader dispatch group can output vertices and primitives
  * It determines how many vertices and primitives to write and then sets those vertices and primitives.
  * Primitives have an indices field which determines the indices of the vertices of that primitive. The indices are based on the output of that mesh shader workgroup only; there is no sharing of vertices across workgroups (no vertex or index buffer equivalents).
  * Primitives can then be culled by setting the appropriate builtin
  * Each vertex output functions exactly as the output from a vertex shader would
  * There can also be per-primitive outputs passed to fragment shaders; these are not interpolated or based on the vertices of the primitive in any way.
* Once all of the primitives are written, those that weren't culled are are rasterized. From this point forward, the only difference from a standard render pipeline is that there may be some per-primitive inputs passed to fragment shaders.

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
* Only Vulkan is currently supported.
* DirectX 12 doesn't support point rendering.
* DirectX 12 support is planned.
* Metal support is desired but not currently planned.


## Naga implementation


### Supported frontends
* 🛠️ WGSL
* ❌ SPIR-V
* 🚫 GLSL

### Supported backends
* 🛠️ SPIR-V
* ❌ HLSL
* ❌ MSL
* 🚫 GLSL
* 🚫 WGSL

✔️ = Complete
🛠️ = In progress
❌ = Planned
🚫 = Unplanned/impossible

## `WGSL` extension specification

The majority of changes relating to mesh shaders will be in WGSL and `naga`.

Using any of these features in a `wgsl` program will require adding the `enable mesh_shading` directive to the top of a program.

Two new shader stages will be added to `WGSL`. Fragment shaders are also modified slightly. Both task shaders and mesh shaders are allowed to use any compute-specific functionality, such as subgroup operations.

### Task shader
This shader stage can be selected by marking a function with `@task`. Task shaders must return a `vec3<u32>` as their output type. Similar to compute shaders, task shaders run in a workgroup. The output must be uniform across all threads in a workgroup.

The output of this determines how many workgroups of mesh shaders will be dispatched. Once dispatched, global id variables will be local to the task shader workgroup dispatch, and mesh shaders won't know the position of their dispatch among all mesh shader dispatches unless this is passed through the payload. The output may be zero to skip dispatching any mesh shader workgroups for the task shader workgroup.

Task shaders must be marked with `@payload(someVar)`, where `someVar` is global variable declared like `var<task_payload> someVar: <type>`. Task shaders may use `someVar` as if it is a read-write workgroup storage variable. This payload is passed to the mesh shader workgroup that is invoked.

### Mesh shader
This shader stage can be selected by marking a function with `@mesh`. Mesh shaders must not return anything.

Mesh shaders can be marked with `@payload(someVar)` similar to task shaders. Unlike task shaders, this is optional, and mesh shaders cannot write to this memory. Declaring `@payload` in a pipeline with no task shader or in a task shader with an `@payload` that is statically sized and differently than the mesh shader payload is illegal. The `@payload` attribute can only be ignored in pipelines that don't have a task shader.

Mesh shaders must be marked with `@vertex_output(OutputType, numOutputs)`, where `numOutputs` is the maximum number of vertices to be output by a mesh shader, and `OutputType` is the data associated with vertices, similar to a standard vertex shader output, and must be a struct.

Mesh shaders must also be marked with `@primitive_output(OutputType, numOutputs)`, which is similar to `@vertex_output` except it describes the primitive outputs.

### Mesh shader outputs

Vertex outputs from mesh shaders function identically to outputs of vertex shaders, and as such must have a field with `@builtin(position)`.

Primitive outputs from mesh shaders have some additional builtins they can set. These include `@builtin(cull_primitive)`, which must be a boolean value. If this is set to true, then the primitive is skipped during rendering. All non-builtin primitive outputs must be decorated with `@per_primitive`.

Mesh shader primitive outputs must also specify exactly one of `@builtin(triangle_indices)`, `@builtin(line_indices)`, or `@builtin(point_index)`. This determines the output topology of the mesh shader, and must match the output topology of the pipeline descriptor the mesh shader is used with. These must be of type `vec3<u32>`, `vec2<u32>`, and `u32` respectively. When setting this, each of the indices must be less than the number of vertices declared in `setMeshOutputs`.

Additionally, the `@location` attributes from the vertex and primitive outputs can't overlap.

Before exiting, the mesh shader must call `setMeshOutputs(numVertices: u32, numIndices: u32)`, which declares the number of vertices and indices that will be written to. These must be less than the corresponding maximums set in `@vertex_output` and `@primitive_output`. The mesh shader must then write to exactly this range of vertices and primitives. A varying member with `@per_primitive` cannot be used in function interfaces except as a primitive output for mesh shaders or as input for fragment shaders.

The mesh shader can write to vertices using the `setVertex(idx: u32, vertex: VertexOutput)` where `VertexOutput` is replaced with the vertex type declared in `@vertex_output`, and `idx` is the index of the vertex to write. Similarly, the mesh shader can write to vertices using `setPrimitive(idx: u32, primitive: PrimitiveOutput)`. These can be written to multiple times, however unsynchronized writes are undefined behavior. The primitives and indices are shared across the entire mesh shader workgroup.

### Fragment shader

Fragment shaders can access vertex output data as if it is from a vertex shader. They can also access primitive output data, provided the input is decorated with `@per_primitive`. The `@per_primitive` attribute can be applied to a value directly, such as `@per_primitive @location(1) value: vec4<f32>`, to a struct such as `@per_primitive primitive_input: PrimitiveInput` where `PrimitiveInput` is a struct containing fields decorated with `@location` and `@builtin`, or to members of a struct that are themselves decorated with `@location` or `@builtin`.

The primitive state is part of the fragment input and must match the output of the mesh shader in the pipeline. Using `@per_primitive` also requires enabling the mesh shader extension. Additionally, the locations of vertex and primitive input cannot overlap.

### Full example

The following is a full example of WGSL shaders that could be used to create a mesh shader pipeline, showing off many of the features.

```wgsl
enable mesh_shading;

const positions = array(
	vec4(0.,1.,0.,1.),
	vec4(-1.,-1.,0.,1.),
	vec4(1.,-1.,0.,1.)
);
const colors = array(
	vec4(0.,1.,0.,1.),
	vec4(0.,0.,1.,1.),
	vec4(1.,0.,0.,1.)
);
struct TaskPayload {
	colorMask: vec4<f32>,
	visible: bool,
}
var<task_payload> taskPayload: TaskPayload;
var<workgroup> workgroupData: f32;
struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) color: vec4<f32>,
}
struct PrimitiveOutput {
	@builtin(triangle_indices) index: vec3<u32>,
	@builtin(cull_primitive) cull: bool,
	@per_primitive @location(1) colorMask: vec4<f32>,
}
struct PrimitiveInput {
	@per_primitive @location(1) colorMask: vec4<f32>,
}

@task
@payload(taskPayload)
@workgroup_size(1)
fn ts_main() -> @builtin(mesh_task_size) vec3<u32> {
	workgroupData = 1.0;
	taskPayload.colorMask = vec4(1.0, 1.0, 0.0, 1.0);
	taskPayload.visible = true;
	return vec3(3, 1, 1);
}
@mesh
@payload(taskPayload)
@vertex_output(VertexOutput, 3) @primitive_output(PrimitiveOutput, 1)
@workgroup_size(1)
fn ms_main(@builtin(local_invocation_index) index: u32, @builtin(global_invocation_id) id: vec3<u32>) {
	setMeshOutputs(3, 1);
	workgroupData = 2.0;
	var v: VertexOutput;

	v.position = positions[0];
	v.color = colors[0] * taskPayload.colorMask;
	setVertex(0, v);

	v.position = positions[1];
	v.color = colors[1] * taskPayload.colorMask;
	setVertex(1, v);

	v.position = positions[2];
	v.color = colors[2] * taskPayload.colorMask;
	setVertex(2, v);

	var p: PrimitiveOutput;
	p.index = vec3<u32>(0, 1, 2);
	p.cull = !taskPayload.visible;
	p.colorMask = vec4<f32>(1.0, 0.0, 1.0, 1.0);
	setPrimitive(0, p);
}
@fragment
fn fs_main(vertex: VertexOutput, primitive: PrimitiveInput) -> @location(0) vec4<f32> {
	return vertex.color * primitive.colorMask;
}
```