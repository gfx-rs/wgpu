struct Globals {
    view_proj: mat4x4<f32>,
    num_lights: vec4<u32>,
};

struct Entity {
    world: mat4x4<f32>,
    color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) proj_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec4<f32>,
};

struct Light {
    proj: mat4x4<f32>,
    pos: vec4<f32>,
    color: vec4<f32>,
};

let c_ambient: vec3<f32> = vec3<f32>(0.05000000074505806, 0.05000000074505806, 0.05000000074505806);
let c_max_lights: u32 = 10u;

@group(0) @binding(0) 
var<uniform> u_globals: Globals;
@group(1) @binding(0) 
var<uniform> u_entity: Entity;
@group(0) @binding(1) 
var<storage> s_lights: array<Light>;
@group(0) @binding(1) 
var<uniform> u_lights: array<Light,10>;
@group(0) @binding(2) 
var t_shadow: texture_depth_2d_array;
@group(0) @binding(3) 
var sampler_shadow: sampler_comparison;

fn fetch_shadow(light_id: u32, homogeneous_coords: vec4<f32>) -> f32 {
    if (homogeneous_coords.w <= 0.0) {
        return 1.0;
    }
    let flip_correction = vec2<f32>(0.5, -0.5);
    let proj_correction = (1.0 / homogeneous_coords.w);
    let light_local = (((homogeneous_coords.xy * flip_correction) * proj_correction) + vec2<f32>(0.5, 0.5));
    let _e28 = textureSampleCompareLevel(t_shadow, sampler_shadow, light_local, i32(light_id), (homogeneous_coords.z * proj_correction));
    return _e28;
}

@stage(vertex) 
fn vs_main(@location(0) position: vec4<i32>, @location(1) normal: vec4<i32>) -> VertexOutput {
    var out: VertexOutput;

    let w = u_entity.world;
    let _e7 = u_entity.world;
    let world_pos = (_e7 * vec4<f32>(position));
    out.world_normal = (mat3x3<f32>(w[0].xyz, w[1].xyz, w[2].xyz) * vec3<f32>(normal.xyz));
    out.world_position = world_pos;
    let _e25 = u_globals.view_proj;
    out.proj_position = (_e25 * world_pos);
    let _e27 = out;
    return _e27;
}

@stage(fragment) 
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color: vec3<f32> = vec3<f32>(0.05000000074505806, 0.05000000074505806, 0.05000000074505806);
    var i: u32 = 0u;

    let normal_1 = normalize(in.world_normal);
    loop {
        let _e14 = i;
        let _e17 = u_globals.num_lights.x;
        if (_e14 < min(_e17, c_max_lights)) {
        } else {
            break;
        }
        let _e23 = i;
        let light = s_lights[_e23];
        let _e26 = i;
        let _e30 = fetch_shadow(_e26, (light.proj * in.world_position));
        let light_dir = normalize((light.pos.xyz - in.world_position.xyz));
        let diffuse = max(0.0, dot(normal_1, light_dir));
        let _e40 = color;
        color = (_e40 + ((_e30 * diffuse) * light.color.xyz));
        continuing {
            let _e20 = i;
            i = (_e20 + 1u);
        }
    }
    let _e46 = color;
    let _e50 = u_entity.color;
    return (vec4<f32>(_e46, 1.0) * _e50);
}

@stage(fragment) 
fn fs_main_without_storage(in_1: VertexOutput) -> @location(0) vec4<f32> {
    var color_1: vec3<f32> = vec3<f32>(0.05000000074505806, 0.05000000074505806, 0.05000000074505806);
    var i_1: u32 = 0u;

    let normal_2 = normalize(in_1.world_normal);
    loop {
        let _e14 = i_1;
        let _e17 = u_globals.num_lights.x;
        if (_e14 < min(_e17, c_max_lights)) {
        } else {
            break;
        }
        let _e23 = i_1;
        let light_1 = u_lights[_e23];
        let _e26 = i_1;
        let _e30 = fetch_shadow(_e26, (light_1.proj * in_1.world_position));
        let light_dir_1 = normalize((light_1.pos.xyz - in_1.world_position.xyz));
        let diffuse_1 = max(0.0, dot(normal_2, light_dir_1));
        let _e40 = color_1;
        color_1 = (_e40 + ((_e30 * diffuse_1) * light_1.color.xyz));
        continuing {
            let _e20 = i_1;
            i_1 = (_e20 + 1u);
        }
    }
    let _e46 = color_1;
    let _e50 = u_entity.color;
    return (vec4<f32>(_e46, 1.0) * _e50);
}
