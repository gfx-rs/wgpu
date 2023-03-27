// const MAX_BOUNCES: i32 = 3;

// struct Parameters {
//     cam_position: vec3<f32>,
//     depth: f32,
//     cam_orientation: vec4<f32>,
//     fov: vec2<f32>,
//     torus_radius: f32,
//     rotation_angle: f32,
// };

// var<uniform> parameters: Parameters;
// var acc_struct: acceleration_structure;
@group(0) @binding(0)
var output: texture_storage_2d<rgba8unorm, write>;


@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let target_size = textureDimensions(output);
    let color =  vec4<f32>(vec2<f32>(global_id.xy) / vec2<f32>(target_size), 0.0, 0.0);
    textureStore(output, global_id.xy, color);
}
