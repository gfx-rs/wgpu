struct SkyOutput {
    [[builtin(position)]] position: vec4<f32>;
    [[location(0)]] uv: vec3<f32>;
};

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
fn vs_sky([[builtin(vertex_index)]] vertex_index: u32) -> SkyOutput {
    // hacky way to draw a large triangle
    let tmp1 = i32(vertex_index) / 2;
    let tmp2 = i32(vertex_index) & 1;
    let pos = vec4<f32>(
        f32(tmp1) * 4.0 - 1.0,
        f32(tmp2) * 4.0 - 1.0,
        1.0,
        1.0
    );

    // transposition = inversion for this orthonormal matrix
    let inv_model_view = transpose(mat3x3<f32>(r_data.view.x.xyz, r_data.view.y.xyz, r_data.view.z.xyz));
    let unprojected = r_data.proj_inv * pos;

    var out: SkyOutput;
    out.uv = inv_model_view * unprojected.xyz;
    out.position = pos;
    return out;
}

struct EntityOutput {
    [[builtin(position)]] position: vec4<f32>;
    [[location(1)]] normal: vec3<f32>;
    [[location(3)]] view: vec3<f32>;
};

[[stage(vertex)]]
fn vs_entity(
    [[location(0)]] pos: vec3<f32>,
    [[location(1)]] normal: vec3<f32>,
) -> EntityOutput {
    var out: EntityOutput;
    out.normal = normal;
    out.view = pos - r_data.cam_pos.xyz;
    out.position = r_data.proj * r_data.view * vec4<f32>(pos, 1.0);
    return out;
}

[[group(0), binding(1)]]
var r_texture: texture_cube<f32>;
[[group(0), binding(2)]]
var r_sampler: sampler;

[[stage(fragment)]]
fn fs_sky(in: SkyOutput) -> [[location(0)]] vec4<f32> {
    return textureSample(r_texture, r_sampler, in.uv);
}

[[stage(fragment)]]
fn fs_entity(in: EntityOutput) -> [[location(0)]] vec4<f32> {
    let incident = normalize(in.view);
    let normal = normalize(in.normal);
    let reflected = incident - 2.0 * dot(normal, incident) * normal;

    let reflected_color = textureSample(r_texture, r_sampler, reflected);
    return vec4<f32>(0.1, 0.1, 0.1, 0.1) + 0.5 * reflected_color;
}
