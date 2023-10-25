struct Globals {
    view_proj: mat4x4<f32>,
    num_lights: vec4<u32>,
}

struct Entity {
    world: mat4x4<f32>,
    color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) proj_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec4<f32>,
}

struct Light {
    proj: mat4x4<f32>,
    pos: vec4<f32>,
    color: vec4<f32>,
}

const c_ambient: vec3<f32> = vec3<f32>(0.05, 0.05, 0.05);
const c_max_lights: u32 = 10u;

@group(0) @binding(0) 
var<uniform> u_globals: Globals;
@group(1) @binding(0) 
var<uniform> u_entity: Entity;
@group(0) @binding(1) 
var<storage> s_lights: array<Light>;
@group(0) @binding(1) 
var<uniform> u_lights: array<Light, 10>;
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
    let _e24 = textureSampleCompareLevel(t_shadow, sampler_shadow, light_local, i32(light_id), (homogeneous_coords.z * proj_correction));
    return _e24;
}

@vertex 
fn vs_main(@location(0) @interpolate(flat) position: vec4<i32>, @location(1) @interpolate(flat) normal: vec4<i32>) -> VertexOutput {
    var out: VertexOutput;

    let w = u_entity.world;
    let _e7 = u_entity.world;
    let world_pos = (_e7 * vec4<f32>(position));
    out.world_normal = (mat3x3<f32>(w[0].xyz, w[1].xyz, w[2].xyz) * vec3<f32>(normal.xyz));
    out.world_position = world_pos;
    let _e26 = u_globals.view_proj;
    out.proj_position = (_e26 * world_pos);
    let _e28 = out;
    return _e28;
}

@fragment 
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color: vec3<f32> = c_ambient;
    var i: u32 = 0u;

    let normal_1 = normalize(in.world_normal);
    loop {
        let _e7 = i;
        let _e11 = u_globals.num_lights.x;
        if (_e7 < min(_e11, c_max_lights)) {
        } else {
            break;
        }
        {
            let _e16 = i;
            let light = s_lights[_e16];
            let _e19 = i;
            let _e23 = fetch_shadow(_e19, (light.proj * in.world_position));
            let light_dir = normalize((light.pos.xyz - in.world_position.xyz));
            let diffuse = max(0.0, dot(normal_1, light_dir));
            let _e37 = color;
            color = (_e37 + ((_e23 * diffuse) * light.color.xyz));
        }
        continuing {
            let _e40 = i;
            i = (_e40 + 1u);
        }
    }
    let _e42 = color;
    let _e47 = u_entity.color;
    return (vec4<f32>(_e42, 1.0) * _e47);
}

@fragment 
fn fs_main_without_storage(in_1: VertexOutput) -> @location(0) vec4<f32> {
    var color_1: vec3<f32> = c_ambient;
    var i_1: u32 = 0u;

    let normal_2 = normalize(in_1.world_normal);
    loop {
        let _e7 = i_1;
        let _e11 = u_globals.num_lights.x;
        if (_e7 < min(_e11, c_max_lights)) {
        } else {
            break;
        }
        {
            let _e16 = i_1;
            let light_1 = u_lights[_e16];
            let _e19 = i_1;
            let _e23 = fetch_shadow(_e19, (light_1.proj * in_1.world_position));
            let light_dir_1 = normalize((light_1.pos.xyz - in_1.world_position.xyz));
            let diffuse_1 = max(0.0, dot(normal_2, light_dir_1));
            let _e37 = color_1;
            color_1 = (_e37 + ((_e23 * diffuse_1) * light_1.color.xyz));
        }
        continuing {
            let _e40 = i_1;
            i_1 = (_e40 + 1u);
        }
    }
    let _e42 = color_1;
    let _e47 = u_entity.color;
    return (vec4<f32>(_e42, 1.0) * _e47);
}
