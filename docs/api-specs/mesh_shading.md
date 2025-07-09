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

If task shaders are marked with `@payload(someVar)`, where `someVar` is global variable declared like `var<workgroup> someVar: <type>`, task shaders may write to `someVar`. This payload is passed to the mesh shader workgroup that is invoked. The mesh shader can skip declaring `@payload` to ignore this input.

### Mesh shader
This shader stage can be selected by marking a function with `@mesh`. Mesh shaders must not return anything.

Mesh shaders can be marked with `@payload(someVar)` similar to task shaders. Unlike task shaders, mesh shaders cannot write to this workgroup memory. Declaring `@payload` in a pipeline with no task shader, in a pipeline with a task shader that doesn't declare `@payload`, or in a task shader with an `@payload` that is statically sized and smaller than the mesh shader payload is illegal.

Mesh shaders must be marked with `@vertex_output(OutputType, numOutputs)`, where `numOutputs` is the maximum number of vertices to be output by a mesh shader, and `OutputType` is the data associated with vertices, similar to a standard vertex shader output.

Mesh shaders must also be marked with `@primitive_output(OutputType, numOutputs)`, which is similar to `@vertex_output` except it describes the primitive outputs.

### Mesh shader outputs

Primitive outputs from mesh shaders have some additional builtins they can set. These include `@builtin(cull_primitive)`, which must be a boolean value. If this is set to true, then the primitive is skipped during rendering.

Mesh shader primitive outputs must also specify exactly one of `@builtin(triangle_indices)`, `@builtin(line_indices)`, or `@builtin(point_index)`. This determines the output topology of the mesh shader, and must match the output topology of the pipeline descriptor the mesh shader is used with. These must be of type `vec3<u32>`, `vec2<u32>`, and `u32` respectively. When setting this, each of the indices must be less than the number of vertices declared in `setMeshOutputs`.

Additionally, the `@location` attributes from the vertex and primitive outputs can't overlap.

Before setting any vertices or indices, or exiting, the mesh shader must call `setMeshOutputs(numVertices: u32, numIndices: u32)`, which declares the number of vertices and indices that will be written to. These must be less than the corresponding maximums set in `@vertex_output` and `@primitive_output`. The mesh shader must then write to exactly these numbers of vertices and primitives.

The mesh shader can write to vertices using the `setVertex(idx: u32, vertex: VertexOutput)` where `VertexOutput` is replaced with the vertex type declared in `@vertex_output`, and `idx` is the index of the vertex to write. Similarly, the mesh shader can write to vertices using `setPrimitive(idx: u32, primitive: PrimitiveOutput)`. These can be written to multiple times, however unsynchronized writes are undefined behavior. The primitives and indices are shared across the entire mesh shader workgroup.

### Fragment shader

Fragment shaders may now be passed the primitive info from a mesh shader the same was as they are passed vertex inputs, for example `fn fs_main(vertex: VertexOutput, primitive: PrimitiveOutput)`. The primitive state is part of the fragment input and must match the output of the mesh shader in the pipeline.

### Full example

The following is a full example of WGSL shaders that could be used to create a mesh shader pipeline, showing off many of the features.

```wgsl
enable mesh_shading;

const positions = array(
	vec4(0.,-1.,0.,1.),
	vec4(-1.,1.,0.,1.),
	vec4(1.,1.,0.,1.)
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
var<workgroup> taskPayload: TaskPayload;
var<workgroup> workgroupData: f32;
struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) color: vec4<f32>,
}
struct PrimitiveOutput {
	@builtin(triangle_indices) index: vec3<u32>,
	@builtin(cull_primitive) cull: bool,
	@location(1) colorMask: vec4<f32>,
}
struct PrimitiveInput {
	@location(1) colorMask: vec4<f32>,
}
fn test_function(input: u32) {

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

	test_function(1);

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