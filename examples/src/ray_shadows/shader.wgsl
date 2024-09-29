struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world_position: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, @location(0) position: vec3<f32>, @location(1) normal: vec3<f32>,) -> VertexOutput {
    var result: VertexOutput;
    let x = i32(vertex_index) / 2;
    let y = i32(vertex_index) & 1;
    let tc = vec2<f32>(
        f32(x) * 2.0,
        f32(y) * 2.0
    );
    result.tex_coords = tc;
    result.position = uniforms.vertex * vec4<f32>(position, 1.0);
    result.normal = normal;
    result.world_position = position;
    return result;
}

struct Uniforms {
    view_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    vertex: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var acc_struct: acceleration_structure;

var<push_constant> light: vec3<f32>;

const SURFACE_BRIGHTNESS = 0.5;

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let camera = (uniforms.view_inv * vec4<f32>(0.0,0.0,0.0,1.0)).xyz;
    var color = vec4<f32>(vertex.tex_coords, 0.0, 1.0);

	let d = vertex.tex_coords * 2.0 - 1.0;

	let origin = vertex.world_position;
	let direction = normalize(light - vertex.world_position);

	var normal: vec3<f32>;
	let dir_cam = normalize(camera - vertex.world_position);
	if (dot(dir_cam, vertex.normal) < 0.0) {
	    normal = -vertex.normal;
	} else {
	    normal = vertex.normal;
	}

    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.0001, 200.0, origin, direction));
    rayQueryProceed(&rq);

    let intersection = rayQueryGetCommittedIntersection(&rq);
    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
        color = vec4<f32>(vec3<f32>(0.1) * SURFACE_BRIGHTNESS, 1.0);
    } else {
        color = vec4<f32>(vec3<f32>(max(dot(direction, normal), 0.1)) * SURFACE_BRIGHTNESS, 1.0);
    }

    return color;
}
