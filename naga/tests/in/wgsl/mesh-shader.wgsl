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
	@builtin(triangle_indices) index: vec3<f32>,
	@builtin(cull_primitive) cull: bool,
	@location(1) colorMask: vec4<f32>,
}
@task
@payload(taskPayload)
@workgroup_size(1)
fn ts_main() -> vec3<u32> {
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
	setMeshOutputs(3u, 1u);
	workgroupData = 2.0;
	setVertex(0, VertexOutput {
		position: positions[0],
		color: colors[0] * taskPayload.colorMask,
	});
	setVertex(1, VertexOutput {
		position: positions[1],
		color: colors[1] * taskPayload.colorMask,
	});
	setVertex(2, VertexOutput {
		position: positions[2],
		color: colors[2] * taskPayload.colorMask,
	});
	setPrimitive(0, PrimitiveOutput {
		index: vec3<u32>(0, 1, 2),
		cull: !taskPayload.visible,
		colorMask: vec4<f32>(1.0, 0.0, 1.0, 1.0),
	});
}
@fragment
fn fs_main(vertex: VertexOutput, primitive: @builtin(primitive) PrimitiveOutput) -> @location(0) vec4<f32> {
	return vertex.color * primitive.colorMask;
}