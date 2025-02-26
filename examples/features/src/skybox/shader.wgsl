struct SkyOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec3<f32>,
};

struct Data {
    // from camera to screen
    proj: mat4x4<f32>,
    // from screen to camera
    proj_inv: mat4x4<f32>,
    // from world to camera
    view: mat4x4<f32>,
    // camera position
    cam_pos: vec4<f32>,
};
@group(0)
@binding(0)
var<uniform> r_data: Data;

@vertex
fn vs_sky(@builtin(vertex_index) vertex_index: u32) -> SkyOutput {
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
    let inv_model_view = transpose(mat3x3<f32>(r_data.view[0].xyz, r_data.view[1].xyz, r_data.view[2].xyz));
    let unprojected = r_data.proj_inv * pos;

    var result: SkyOutput;
    result.uv = inv_model_view * unprojected.xyz;
    result.position = pos;
    return result;
}

struct EntityOutput {
    @builtin(position) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(3) view: vec3<f32>,
};

@vertex
fn vs_entity(
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
) -> EntityOutput {
    var result: EntityOutput;
    result.normal = normal;
    result.view = pos - r_data.cam_pos.xyz;
    result.position = r_data.proj * r_data.view * vec4<f32>(pos, 1.0);
    return result;
}

@group(0)
@binding(1)
var r_texture: texture_cube<f32>;
@group(0)
@binding(2)
var r_sampler: sampler;

@fragment
fn fs_sky(vertex: SkyOutput) -> @location(0) vec4<f32> {
    return textureSample(r_texture, r_sampler, vertex.uv);
}

@fragment
fn fs_entity(vertex: EntityOutput) -> @location(0) vec4<f32> {
    let incident = normalize(vertex.view);
    let normal = normalize(vertex.normal);
    let reflected = incident - 2.0 * dot(normal, incident) * normal;

    let reflected_color = textureSample(r_texture, r_sampler, reflected).rgb;
    return vec4<f32>(vec3<f32>(0.1) + 0.5 * reflected_color, 1.0);
}
