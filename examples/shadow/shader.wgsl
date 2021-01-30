[[location(0)]]
var<in> in_position: vec4<i32>;
[[builtin(position)]]
var<out> out_position: vec4<f32>;

[[block]]
struct Globals {
    view_proj: mat4x4<f32>;
    num_lights: vec4<u32>;
};

[[group(0), binding(0)]]
var<uniform> u_globals: Globals;

[[block]]
struct Entity {
    world: mat4x4<f32>;
    color: vec4<f32>;
};

[[group(1), binding(0)]]
var<uniform> u_entity: Entity;

[[stage(vertex)]]
fn vs_bake() {
    out_position = u_globals.view_proj * u_entity.world * vec4<f32>(in_position);
}


[[location(1)]]
var<in> in_normal: vec4<i32>;

[[location(0)]]
var<out> out_normal_vs: vec3<f32>;
[[location(1)]]
var<out> out_position_vs: vec4<f32>;

[[stage(vertex)]]
fn vs_main() {
    const w: mat4x4<f32> = u_entity.world;
    out_normal_vs = mat3x3<f32>(w.x.xyz, w.y.xyz, w.z.xyz) * vec3<f32>(in_normal.xyz);
    out_position_vs = u_entity.world * vec4<f32>(in_position);
    out_position = u_globals.view_proj * out_position_vs;
}

// fragment shader

[[block]]
struct Light {
    proj: mat4x4<f32>;
    pos: vec4<f32>;
    color: vec4<f32>;
};

[[block]]
struct Lights {
    data: [[stride(96)]] array<Light>;
};

[[group(0), binding(1)]]
var<storage> s_lights: [[access(read)]] Lights;
[[group(0), binding(2)]]
var t_shadow: texture_depth_2d_array;
[[group(0), binding(3)]]
var sampler_shadow: sampler;

fn fetch_shadow(light_id: u32, homogeneous_coords: vec4<f32>) -> f32 {
    if (homogeneous_coords.w <= 0.0) {
        return 1.0;
    }
    // compensate for the Y-flip difference between the NDC and texture coordinates
    const flip_correction: vec2<f32> = vec2<f32>(0.5, -0.5);
    // compute texture coordinates for shadow lookup
    const proj_correction: f32 = 1.0 / homogeneous_coords.w;
    const light_local: vec2<f32> = homogeneous_coords.xy * flip_correction * proj_correction + vec2<f32>(0.5, 0.5);
    // do the lookup, using HW PCF and comparison
    return textureSampleCompare(t_shadow, sampler_shadow, light_local, i32(light_id), homogeneous_coords.z * proj_correction);
}

[[location(0)]]
var<in> in_normal_fs: vec3<f32>;
[[location(1)]]
var<in> in_position_fs: vec4<f32>;
[[location(0)]]
var<out> out_color_fs: vec4<f32>;

const c_ambient: vec3<f32> = vec3<f32>(0.05, 0.05, 0.05);
const c_max_lights: u32 = 10u;

[[stage(fragment)]]
fn fs_main() {
    const normal: vec3<f32> = normalize(in_normal_fs);
    // accumulate color
    var color: vec3<f32> = c_ambient;
    var i: u32 = 0u;
    loop {
        if (i >= min(u_globals.num_lights.x, c_max_lights)) {
            break;
        }
        const light: Light = s_lights.data[i];
        // project into the light space
        const shadow: f32 = fetch_shadow(i, light.proj * in_position_fs);
        // compute Lambertian diffuse term
        const light_dir: vec3<f32> = normalize(light.pos.xyz - in_position_fs.xyz);
        const diffuse: f32 = max(0.0, dot(normal, light_dir));
        // add light contribution
        color = color + shadow * diffuse * light.color.xyz;
        continuing {
            i = i + 1u;
        }
    }
    // multiply the light by material color
    out_color_fs = vec4<f32>(color, 1.0) * u_entity.color;
}
