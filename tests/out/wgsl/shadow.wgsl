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

let c_ambient: vec3<f32> = vec3<f32>(0.05, 0.05, 0.05);
let c_max_lights: u32 = 10u;

[[group(0), binding(0)]]
var<uniform> u_globals: Globals;
[[group(0), binding(1)]]
var<storage> s_lights: [[access(read)]] Lights;
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
    let _e26: f32 = textureSampleCompareLevel(t_shadow, sampler_shadow, light_local, i32(light_id), (homogeneous_coords.z / homogeneous_coords.w));
    return _e26;
}

[[stage(fragment)]]
fn fs_main([[location(0)]] raw_normal: vec3<f32>, [[location(1)]] position: vec4<f32>) -> [[location(0)]] vec4<f32> {
    var color: vec3<f32> = vec3<f32>(0.05, 0.05, 0.05);
    var i: u32 = 0u;

    let normal: vec3<f32> = normalize(raw_normal);
    loop {
        let _e12: u32 = i;
        let _e14: vec4<u32> = u_globals.num_lights;
        if ((_e12 >= min(_e14.x, c_max_lights))) {
            break;
        }
        let _e19: u32 = i;
        let light: Light = s_lights.data[_e19];
        let _e22: u32 = i;
        let _e25: f32 = fetch_shadow(_e22, (light.proj * position));
        let light_dir: vec3<f32> = normalize((light.pos.xyz - position.xyz));
        let diffuse: f32 = max(0.0, dot(normal, light_dir));
        let _e34: vec3<f32> = color;
        color = (_e34 + ((_e25 * diffuse) * light.color.xyz));
        continuing {
            let _e40: u32 = i;
            i = (_e40 + 1u);
        }
    }
    let _e43: vec3<f32> = color;
    return vec4<f32>(_e43, 1.0);
}
