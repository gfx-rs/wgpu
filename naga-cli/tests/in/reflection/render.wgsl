// Vertex attributes, inter-stage variables, fragment outputs, and texture/sampler pairs.

struct Camera {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    exposure: f32,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) material: u32,
}

struct VertexOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) material: u32,
    @location(2) @interpolate(linear, centroid) normal: vec3<f32>,
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var albedo: texture_2d<f32>;
@group(1) @binding(1) var albedo_sampler: sampler;
@group(1) @binding(2) var shadow_map: texture_depth_2d_array;
@group(1) @binding(3) var shadow_sampler: sampler_comparison;
@group(1) @binding(4) var output_image: texture_storage_2d<rgba8unorm, write>;

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance: u32) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4(in.position, f32(instance));
    out.uv = in.uv;
    out.material = in.material;
    out.normal = in.normal;
    return out;
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front: bool) -> FragmentOutput {
    var out: FragmentOutput;
    let shadow = textureSampleCompare(shadow_map, shadow_sampler, in.uv, 0, 0.5);
    out.color = textureSample(albedo, albedo_sampler, in.uv) * shadow * camera.exposure;
    out.emissive = select(vec4(0.0), vec4(1.0), front);
    out.depth = in.clip_position.z;
    return out;
}

@fragment
fn fs_write() {
    textureStore(output_image, vec2(0), vec4(1.0));
}
