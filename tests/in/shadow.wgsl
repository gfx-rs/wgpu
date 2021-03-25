[[block]]
struct Globals {
    num_lights: vec4<u32>;
};

[[group(0), binding(0)]]
var<uniform> u_globals: Globals;

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
var sampler_shadow: sampler_comparison;

fn fetch_shadow(light_id: u32, homogeneous_coords: vec4<f32>) -> f32 {
    if (homogeneous_coords.w <= 0.0) {
        return 1.0;
    }
    const flip_correction = vec2<f32>(0.5, -0.5);
    const proj_correction = 1.0 / homogeneous_coords.w;
    const light_local = homogeneous_coords.xy * flip_correction * proj_correction + vec2<f32>(0.5, 0.5);
    return textureSampleCompare(t_shadow, sampler_shadow, light_local, i32(light_id), homogeneous_coords.z * proj_correction);
}

const c_ambient: vec3<f32> = vec3<f32>(0.05, 0.05, 0.05);
const c_max_lights: u32 = 10u;

[[stage(fragment)]]
fn fs_main(
    [[location(0)]] raw_normal: vec3<f32>,
    [[location(1)]] position: vec4<f32>
) -> [[location(0)]] vec4<f32> {
    const normal: vec3<f32> = normalize(raw_normal);
    // accumulate color
    var color: vec3<f32> = c_ambient;
    var i: u32 = 0u;
    loop {
        if (i >= min(u_globals.num_lights.x, c_max_lights)) {
            break;
        }
        const light = s_lights.data[i];
        const shadow = fetch_shadow(i, light.proj * position);
        const light_dir = normalize(light.pos.xyz - position.xyz);
        const diffuse = max(0.0, dot(normal, light_dir));
        color = color + shadow * diffuse * light.color.xyz;
        continuing {
            i = i + 1u;
        }
    }
    // multiply the light by material color
    return vec4<f32>(color, 1.0);
}
