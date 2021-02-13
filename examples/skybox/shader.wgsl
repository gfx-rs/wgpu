[[builtin(position)]]
var<out> out_position: vec4<f32>;
[[location(0)]] var<out> out_uv: vec3<f32>;
[[builtin(vertex_index)]] var<in> in_vertex_index: u32;

[[block]]
struct Data {
    // from camera to screen
    proj: mat4x4<f32>;
    // from screen to camera
    proj_inv: mat4x4<f32>;
    // from world to camera
    view: mat4x4<f32>;
    // camera position
    cam_pos: vec4<f32>;
};
[[group(0), binding(0)]]
var r_data: Data;

[[stage(vertex)]]
fn vs_sky() {
    // hacky way to draw a large triangle
    var tmp1: i32 = i32(in_vertex_index) / 2;
    var tmp2: i32 = i32(in_vertex_index) & 1;
    const pos: vec4<f32> = vec4<f32>(
        f32(tmp1) * 4.0 - 1.0,
        f32(tmp2) * 4.0 - 1.0,
        1.0,
        1.0
    );

    // transposition = inversion for this orthonormal matrix
    const inv_model_view: mat3x3<f32> = transpose(mat3x3<f32>(r_data.view.x.xyz, r_data.view.y.xyz, r_data.view.z.xyz));
    var unprojected: vec4<f32> = r_data.proj_inv * pos; //TODO: const
    out_uv = inv_model_view * unprojected.xyz;
    out_position = pos;
}

[[location(0)]] var<in> in_pos: vec3<f32>;
[[location(1)]] var<in> in_normal: vec3<f32>;
[[location(1)]] var<out> out_normal: vec3<f32>;
[[location(3)]] var<out> out_view: vec3<f32>;

[[stage(vertex)]]
fn vs_entity() {
    out_normal = in_normal;
    out_view = in_pos - r_data.cam_pos.xyz;
    out_position = r_data.proj * r_data.view * vec4<f32>(in_pos, 1.0);
}

[[group(0), binding(1)]]
var r_texture: texture_cube<f32>;
[[group(0), binding(2)]]
var r_sampler: sampler;

[[location(0)]] var<in> in_uv: vec3<f32>;
[[location(0)]] var<out> out_color: vec4<f32>;
[[location(3)]] var<in> in_view: vec3<f32>;

[[stage(fragment)]]
fn fs_sky() {
    out_color = textureSample(r_texture, r_sampler, in_uv);
}

[[stage(fragment)]]
fn fs_entity() {
    const incident: vec3<f32> = normalize(in_view);
    const normal: vec3<f32> = normalize(in_normal);
    const reflected: vec3<f32> = incident - 2.0 * dot(normal, incident) * normal;

    var reflected_color: vec4<f32> = textureSample(r_texture, r_sampler, reflected);
    out_color = vec4<f32>(0.1, 0.1, 0.1, 0.1) + 0.5 * reflected_color;
}
