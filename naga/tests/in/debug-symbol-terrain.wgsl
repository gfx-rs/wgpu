// Taken from https://github.com/sotrh/learn-wgpu/blob/11820796f5e1dbce42fb1119f04ddeb4b167d2a0/code/intermediate/tutorial13-terrain/src/terrain.wgsl
// ============================
// Terrain Generation
// ============================

// https://gist.github.com/munrocket/236ed5ba7e409b8bdf1ff6eca5dcdc39
//  MIT License. © Ian McEwan, Stefan Gustavson, Munrocket
// - Less condensed glsl implementation with comments can be found at https://weber.itn.liu.se/~stegu/jgt2012/article.pdf

fn permute3(x: vec3<f32>) -> vec3<f32> { return (((x * 34.) + 1.) * x) % vec3<f32>(289.); }

fn snoise2(v: vec2<f32>) -> f32 {
    let C = vec4<f32>(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
    var i: vec2<f32> = floor(v + dot(v, C.yy));
    let x0 = v - i + dot(i, C.xx);
    // I flipped the condition here from > to < as it fixed some artifacting I was observing
    var i1: vec2<f32> = select(vec2<f32>(1., 0.), vec2<f32>(0., 1.), (x0.x < x0.y));
    var x12: vec4<f32> = x0.xyxy + C.xxzz - vec4<f32>(i1, 0., 0.);
    i = i % vec2<f32>(289.);
    let p = permute3(permute3(i.y + vec3<f32>(0., i1.y, 1.)) + i.x + vec3<f32>(0., i1.x, 1.));
    var m: vec3<f32> = max(0.5 - vec3<f32>(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), vec3<f32>(0.));
    m = m * m;
    m = m * m;
    let x = 2. * fract(p * C.www) - 1.;
    let h = abs(x) - 0.5;
    let ox = floor(x + 0.5);
    let a0 = x - ox;
    m = m * (1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h));
    let g = vec3<f32>(a0.x * x0.x + h.x * x0.y, a0.yz * x12.xz + h.yz * x12.yw);
    return 130. * dot(m, g);
}


fn fbm(p: vec2<f32>) -> f32 {
    let NUM_OCTAVES: u32 = 5u;
    var x = p * 0.01;
    var v = 0.0;
    var a = 0.5;
    let shift = vec2<f32>(100.0);
    let cs = vec2<f32>(cos(0.5), sin(0.5));
    let rot = mat2x2<f32>(cs.x, cs.y, -cs.y, cs.x);

    for (var i = 0u; i < NUM_OCTAVES; i = i + 1u) {
        v = v + a * snoise2(x);
        x = rot * x * 2.0 + shift;
        a = a * 0.5;
    }

    return v;
}

struct ChunkData {
    chunk_size: vec2<u32>,
    chunk_corner: vec2<i32>,
    min_max_height: vec2<f32>,
}

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct VertexBuffer {
    data: array<Vertex>, // stride: 32
}

struct IndexBuffer {
    data: array<u32>,
}

@group(0) @binding(0) var<uniform> chunk_data: ChunkData;
@group(0) @binding(1) var<storage, read_write> vertices: VertexBuffer;
@group(0) @binding(2) var<storage, read_write> indices: IndexBuffer;

fn terrain_point(p: vec2<f32>, min_max_height: vec2<f32>) -> vec3<f32> {
    return vec3<f32>(
        p.x,
        mix(min_max_height.x, min_max_height.y, fbm(p)),
        p.y,
    );
}

fn terrain_vertex(p: vec2<f32>, min_max_height: vec2<f32>) -> Vertex {
    let v = terrain_point(p, min_max_height);

    let tpx = terrain_point(p + vec2<f32>(0.1, 0.0), min_max_height) - v;
    let tpz = terrain_point(p + vec2<f32>(0.0, 0.1), min_max_height) - v;
    let tnx = terrain_point(p + vec2<f32>(-0.1, 0.0), min_max_height) - v;
    let tnz = terrain_point(p + vec2<f32>(0.0, -0.1), min_max_height) - v;

    let pn = normalize(cross(tpz, tpx));
    let nn = normalize(cross(tnz, tnx));

    let n = (pn + nn) * 0.5;

    return Vertex(v, n);
}

fn index_to_p(vert_index: u32, chunk_size: vec2<u32>, chunk_corner: vec2<i32>) -> vec2<f32> {
    return vec2(
        f32(vert_index) % f32(chunk_size.x + 1u),
        f32(vert_index / (chunk_size.x + 1u)),
    ) + vec2<f32>(chunk_corner);
}

@compute @workgroup_size(64)
fn gen_terrain_compute(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    // Create vert_component
    let vert_index = gid.x;

    let p = index_to_p(vert_index, chunk_data.chunk_size, chunk_data.chunk_corner);

    vertices.data[vert_index] = terrain_vertex(p, chunk_data.min_max_height);

    // Create indices
    let start_index = gid.x * 6u; // using TriangleList

    if (start_index >= (chunk_data.chunk_size.x * chunk_data.chunk_size.y * 6u)) { return; }

    let v00 = vert_index + gid.x / chunk_data.chunk_size.x;
    let v10 = v00 + 1u;
    let v01 = v00 + chunk_data.chunk_size.x + 1u;
    let v11 = v01 + 1u;

    indices.data[start_index] = v00;
    indices.data[start_index + 1u] = v01;
    indices.data[start_index + 2u] = v11;
    indices.data[start_index + 3u] = v00;
    indices.data[start_index + 4u] = v11;
    indices.data[start_index + 5u] = v10;
}

// ============================
// Terrain Gen (Fragment Shader)
// ============================

struct GenData {
    chunk_size: vec2<u32>,
    chunk_corner: vec2<i32>,
    min_max_height: vec2<f32>,
    texture_size: u32,
    start_index: u32,
}
@group(0)
@binding(0)
var<uniform> gen_data: GenData;

struct GenVertexOutput {
    @location(0)
    index: u32,
    @builtin(position)
    position: vec4<f32>,
    @location(1)
    uv: vec2<f32>,
};

@vertex
fn gen_terrain_vertex(@builtin(vertex_index) vindex: u32) -> GenVertexOutput {
    let u = f32(((vindex + 2u) / 3u) % 2u);
    let v = f32(((vindex + 1u) / 3u) % 2u);
    let uv = vec2<f32>(u, v);

    let position = vec4<f32>(-1.0 + uv * 2.0, 0.0, 1.0);

    // TODO: maybe replace this with u32(dot(uv, vec2(f32(gen_data.texture_dim.x))))
    let index = u32(uv.x * f32(gen_data.texture_size) + uv.y * f32(gen_data.texture_size)) + gen_data.start_index;

    return GenVertexOutput(index, position, uv);
}


struct GenFragmentOutput {
    @location(0) vert_component: u32,
    @location(1) index: u32,
}

@fragment
fn gen_terrain_fragment(in: GenVertexOutput) -> GenFragmentOutput {
    let i = u32(in.uv.x * f32(gen_data.texture_size) + in.uv.y * f32(gen_data.texture_size * gen_data.texture_size)) + gen_data.start_index;
    let vert_index = u32(floor(f32(i) / 6.));
    let comp_index = i % 6u;

    let p = index_to_p(vert_index, gen_data.chunk_size, gen_data.chunk_corner);
    let v = terrain_vertex(p, gen_data.min_max_height);

    var vert_component: f32 = 0.;
    
    switch comp_index {
        case 0u: { vert_component = v.position.x; }
        case 1u: { vert_component = v.position.y; }
        case 2u: { vert_component = v.position.z; }
        case 3u: { vert_component = v.normal.x; }
        case 4u: { vert_component = v.normal.y; }
        case 5u: { vert_component = v.normal.z; }
        default: {}
    }

    let v00 = vert_index + vert_index / gen_data.chunk_size.x;
    let v10 = v00 + 1u;
    let v01 = v00 + gen_data.chunk_size.x + 1u;
    let v11 = v01 + 1u;

    var index = 0u;
    switch comp_index {
        case 0u, 3u: { index = v00; }
        case 2u, 4u: { index = v11; }
        case 1u: { index = v01; }
        case 5u: { index = v10; }
        default: {}
    }
    index = in.index;
    // index = gen_data.start_index;
    // indices.data[start_index] = v00;
    // indices.data[start_index + 1u] = v01;
    // indices.data[start_index + 2u] = v11;
    // indices.data[start_index + 3u] = v00;
    // indices.data[start_index + 4u] = v11;
    // indices.data[start_index + 5u] = v10;

    let ivert_component = bitcast<u32>(vert_component);
    return GenFragmentOutput(ivert_component, index);
}

// ============================
// Terrain Rendering
// ============================

struct Camera {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}
@group(0) @binding(0)
var<uniform> camera: Camera;

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}
@group(1) @binding(0)
var<uniform> light: Light;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
}

@vertex
fn vs_main(
    vertex: Vertex,
) -> VertexOutput {
    let clip_position = camera.view_proj * vec4<f32>(vertex.position, 1.);
    let normal = vertex.normal;
    return VertexOutput(clip_position, normal, vertex.position);
}

@group(2) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(2) @binding(1)
var s_diffuse: sampler;
@group(2) @binding(2)
var t_normal: texture_2d<f32>;
@group(2) @binding(3)
var s_normal: sampler;

fn color23(p: vec2<f32>) -> vec3<f32> {
    return vec3<f32>(
        snoise2(p) * 0.5 + 0.5,
        snoise2(p + vec2<f32>(23., 32.)) * 0.5 + 0.5,
        snoise2(p + vec2<f32>(-43., 3.)) * 0.5 + 0.5,
    );
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = smoothstep(vec3<f32>(0.0), vec3<f32>(0.1), fract(in.world_pos));
    color = mix(vec3<f32>(0.5, 0.1, 0.7), vec3<f32>(0.2, 0.2, 0.2), vec3<f32>(color.x * color.y * color.z));

    let ambient_strength = 0.1;
    let ambient_color = light.color * ambient_strength;

    let light_dir = normalize(light.position - in.world_pos);
    let view_dir = normalize(camera.view_pos.xyz - in.world_pos);
    let half_dir = normalize(view_dir + light_dir);

    let diffuse_strength = max(dot(in.normal, light_dir), 0.0);
    let diffuse_color = diffuse_strength * light.color;

    let specular_strength = pow(max(dot(in.normal, half_dir), 0.0), 32.0);
    let specular_color = specular_strength * light.color;

    let result = (ambient_color + diffuse_color + specular_color) * color;

    return vec4<f32>(result, 1.0);
}