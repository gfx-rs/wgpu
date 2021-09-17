[[block]]
struct Globals {
    num_lights: vec4<u32>;
};

struct Light {
    proj: mat4x4<f32>;
    pos: vec4<f32>;
    color: vec4<f32>;
};

[[block]]
struct Lights {
    data: [[stride(96)]] array<Light>;
};

let c_ambient: vec3<f32> = vec3<f32>(0.05000000074505806, 0.05000000074505806, 0.05000000074505806);
let c_max_lights: u32 = 10u;

[[group(0), binding(0)]]
var<uniform> u_globals: Globals;
[[group(0), binding(1)]]
var<storage> s_lights: Lights;
[[group(0), binding(2)]]
var t_shadow: texture_depth_2d_array;
[[group(0), binding(3)]]
var sampler_shadow: sampler_comparison;

fn fetch_shadow(light_id: u32, homogeneous_coords: vec4<f32>) -> f32 {
    if ((homogeneous_coords.w <= 0.0)) {
        return 1.0;
    }
    let flip_correction: vec2<f32> = vec2<f32>(0.5, -0.5);
    let light_local: vec2<f32> = (((homogeneous_coords.xy * flip_correction) / vec2<f32>(homogeneous_coords.w)) + vec2<f32>(0.5, 0.5));
    let e26: f32 = textureSampleCompareLevel(t_shadow, sampler_shadow, light_local, i32(light_id), (homogeneous_coords.z / homogeneous_coords.w));
    return e26;
}

[[stage(fragment)]]
fn fs_main([[location(0)]] raw_normal: vec3<f32>, [[location(1)]] position: vec4<f32>) -> [[location(0)]] vec4<f32> {
    var color: vec3<f32> = vec3<f32>(0.05000000074505806, 0.05000000074505806, 0.05000000074505806);
    var i: u32 = 0u;

    let normal: vec3<f32> = normalize(raw_normal);
    loop {
        let e12: u32 = i;
        let e15: u32 = u_globals.num_lights.x;
        if ((e12 >= min(e15, c_max_lights))) {
            break;
        }
        let e19: u32 = i;
        let light: Light = s_lights.data[e19];
        let e22: u32 = i;
        let e25: f32 = fetch_shadow(e22, (light.proj * position));
        let light_dir: vec3<f32> = normalize((light.pos.xyz - position.xyz));
        let diffuse: f32 = max(0.0, dot(normal, light_dir));
        let e34: vec3<f32> = color;
        color = (e34 + ((e25 * diffuse) * light.color.xyz));
        continuing {
            let e40: u32 = i;
            i = (e40 + 1u);
        }
    }
    let e43: vec3<f32> = color;
    return vec4<f32>(e43, 1.0);
}
