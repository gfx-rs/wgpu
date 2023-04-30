struct Globals {
    view_proj: mat4x4<f32>,
    num_lights: vec4<u32>,
};

@group(0)
@binding(0)
var<uniform> u_globals: Globals;

struct Entity {
    world: mat4x4<f32>,
    color: vec4<f32>,
};

@group(1)
@binding(0)
var<uniform> u_entity: Entity;

@vertex
fn vs_bake(@location(0) position: vec4<i32>) -> @builtin(position) vec4<f32> {
    return u_globals.view_proj * u_entity.world * vec4<f32>(position);
}

struct VertexOutput {
    @builtin(position) proj_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec4<f32>
};

@vertex
fn vs_main(
    @location(0) position: vec4<i32>,
    @location(1) normal: vec4<i32>,
) -> VertexOutput {
    let w = u_entity.world;
    let world_pos = u_entity.world * vec4<f32>(position);
    var result: VertexOutput;
    result.world_normal = mat3x3<f32>(w[0].xyz, w[1].xyz, w[2].xyz) * vec3<f32>(normal.xyz);
    result.world_position = world_pos;
    result.proj_position = u_globals.view_proj * world_pos;
    return result;
}

// fragment shader

struct Light {
    proj: mat4x4<f32>,
    pos: vec4<f32>,
    color: vec4<f32>,
};

@group(0)
@binding(1)
var<storage, read> s_lights: array<Light>;
@group(0)
@binding(1)
var<uniform> u_lights: array<Light, 10>; // Used when storage types are not supported
@group(0)
@binding(2)
var t_shadow: texture_depth_2d_array;
@group(0)
@binding(3)
var sampler_shadow: sampler_comparison;

fn fetch_shadow(light_id: u32, homogeneous_coords: vec4<f32>) -> f32 {
    if (homogeneous_coords.w <= 0.0) {
        return 1.0;
    }
    // compensate for the Y-flip difference between the NDC and texture coordinates
    let flip_correction = vec2<f32>(0.5, -0.5);
    // compute texture coordinates for shadow lookup
    let proj_correction = 1.0 / homogeneous_coords.w;
    let light_local = homogeneous_coords.xy * flip_correction * proj_correction + vec2<f32>(0.5, 0.5);
    // do the lookup, using HW PCF and comparison
    return textureSampleCompareLevel(t_shadow, sampler_shadow, light_local, i32(light_id), homogeneous_coords.z * proj_correction);
}

const c_ambient: vec3<f32> = vec3<f32>(0.05, 0.05, 0.05);
const c_max_lights: u32 = 10u;

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(vertex.world_normal);
    // accumulate color
    var color: vec3<f32> = c_ambient;
    for(var i = 0u; i < min(u_globals.num_lights.x, c_max_lights); i += 1u) {
        let light = s_lights[i];
        // project into the light space
        let shadow = fetch_shadow(i, light.proj * vertex.world_position);
        // compute Lambertian diffuse term
        let light_dir = normalize(light.pos.xyz - vertex.world_position.xyz);
        let diffuse = max(0.0, dot(normal, light_dir));
        // add light contribution
        color += shadow * diffuse * light.color.xyz;
    }
    // multiply the light by material color
    return vec4<f32>(color, 1.0) * u_entity.color;
}

// The fragment entrypoint used when storage buffers are not available for the lights
@fragment
fn fs_main_without_storage(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(vertex.world_normal);
    var color: vec3<f32> = c_ambient;
    for(var i = 0u; i < min(u_globals.num_lights.x, c_max_lights); i += 1u) {
        // This line is the only difference from the entrypoint above. It uses the lights
        // uniform instead of the lights storage buffer
        let light = u_lights[i];
        let shadow = fetch_shadow(i, light.proj * vertex.world_position);
        let light_dir = normalize(light.pos.xyz - vertex.world_position.xyz);
        let diffuse = max(0.0, dot(normal, light_dir));
        color += shadow * diffuse * light.color.xyz;
    }
    return vec4<f32>(color, 1.0) * u_entity.color;
}
