struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var result: VertexOutput;
    let x = i32(vertex_index) / 2;
    let y = i32(vertex_index) & 1;
    let tc = vec2<f32>(
        f32(x) * 2.0,
        f32(y) * 2.0
    );
    result.position = vec4<f32>(
        tc.x * 2.0 - 1.0,
        1.0 - tc.y * 2.0,
        0.0, 1.0
    );
    result.tex_coords = tc;
    return result;
}

/*
The contents of the RayQuery struct are roughly as follows
let RAY_FLAG_NONE = 0x00u;
let RAY_FLAG_OPAQUE = 0x01u;
let RAY_FLAG_NO_OPAQUE = 0x02u;
let RAY_FLAG_TERMINATE_ON_FIRST_HIT = 0x04u;
let RAY_FLAG_SKIP_CLOSEST_HIT_SHADER = 0x08u;
let RAY_FLAG_CULL_BACK_FACING = 0x10u;
let RAY_FLAG_CULL_FRONT_FACING = 0x20u;
let RAY_FLAG_CULL_OPAQUE = 0x40u;
let RAY_FLAG_CULL_NO_OPAQUE = 0x80u;
let RAY_FLAG_SKIP_TRIANGLES = 0x100u;
let RAY_FLAG_SKIP_AABBS = 0x200u;

let RAY_QUERY_INTERSECTION_NONE = 0u;
let RAY_QUERY_INTERSECTION_TRIANGLE = 1u;
let RAY_QUERY_INTERSECTION_GENERATED = 2u;
let RAY_QUERY_INTERSECTION_AABB = 3u;

struct RayDesc {
    flags: u32,
    cull_mask: u32,
    t_min: f32,
    t_max: f32,
    origin: vec3<f32>,
    dir: vec3<f32>,
}

struct RayIntersection {
    kind: u32,
    t: f32,
    instance_custom_index: u32,
    instance_id: u32,
    sbt_record_offset: u32,
    geometry_index: u32,
    primitive_index: u32,
    barycentrics: vec2<f32>,
    front_face: bool,
    object_to_world: mat4x3<f32>,
    world_to_object: mat4x3<f32>,
}
*/

struct Uniforms {
    view_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
};

struct Vertex {
    pos: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
};


struct Instance {
    first_vertex: u32,
    first_geometry: u32,
    last_geometry: u32,
    _pad: u32
};

struct Material{
    roughness_exponent: f32,
    metalness: f32,
    specularity: f32,
    albedo: vec3<f32>
}

struct Geometry {
    first_index: u32,
    material: Material,
};


@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read> vertices: array<Vertex>;

@group(0) @binding(2)
var<storage, read> indices: array<u32>;

@group(0) @binding(3)
var<storage, read> geometries: array<Geometry>;

@group(0) @binding(4)
var<storage, read> instances: array<Instance>;

@group(0) @binding(5)
var acc_struct: acceleration_structure;

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {

    var color =  vec4<f32>(vertex.tex_coords, 0.0, 1.0);

	let d = vertex.tex_coords * 2.0 - 1.0;

	let origin = (uniforms.view_inv * vec4<f32>(0.0,0.0,0.0,1.0)).xyz;
	let temp = uniforms.proj_inv * vec4<f32>(d.x, d.y, 1.0, 1.0);
	let direction = (uniforms.view_inv * vec4<f32>(normalize(temp.xyz), 0.0)).xyz;

    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.1, 200.0, origin, direction));
    rayQueryProceed(&rq);

    let intersection = rayQueryGetCommittedIntersection(&rq);
    if (intersection.kind != RAY_QUERY_INTERSECTION_NONE) {
        let instance = instances[intersection.instance_custom_index];
        let geometry = geometries[intersection.geometry_index + instance.first_geometry];

        let index_offset = geometry.first_index;
        let vertex_offset = instance.first_vertex;

        let first_index_index = intersection.primitive_index * 3u + index_offset;

        let v_0 = vertices[vertex_offset+indices[first_index_index+0u]];
        let v_1 = vertices[vertex_offset+indices[first_index_index+1u]];
        let v_2 = vertices[vertex_offset+indices[first_index_index+2u]];

        let bary = vec3<f32>(1.0 - intersection.barycentrics.x - intersection.barycentrics.y, intersection.barycentrics);

        let pos = v_0.pos * bary.x + v_1.pos * bary.y + v_2.pos * bary.z;
        let normal_raw = v_0.normal * bary.x + v_1.normal * bary.y + v_2.normal * bary.z;
        let uv = v_0.uv * bary.x + v_1.uv * bary.y + v_2.uv * bary.z;

        let normal = normalize(normal_raw);

        let material = geometry.material;

        color = vec4<f32>(material.albedo, 1.0);

        if(intersection.instance_custom_index == 1u){
            color = vec4<f32>(normal, 1.0);
        }
    }

    return color;
}
