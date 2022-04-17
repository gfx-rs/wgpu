struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec3<f32>,
}

struct Data {
    proj_inv: mat4x4<f32>,
    view: mat4x4<f32>,
}

@group(0) @binding(0) 
var<uniform> r_data: Data;
@group(0) @binding(1) 
var r_texture: texture_cube<f32>;
@group(0) @binding(2) 
var r_sampler: sampler;

@vertex 
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var tmp1_: i32;
    var tmp2_: i32;

    tmp1_ = (i32(vertex_index) / 2);
    tmp2_ = (i32(vertex_index) & 1);
    let _e10 = tmp1_;
    let _e16 = tmp2_;
    let pos = vec4<f32>(((f32(_e10) * 4.0) - 1.0), ((f32(_e16) * 4.0) - 1.0), 0.0, 1.0);
    let _e27 = r_data.view[0];
    let _e31 = r_data.view[1];
    let _e35 = r_data.view[2];
    let inv_model_view = transpose(mat3x3<f32>(_e27.xyz, _e31.xyz, _e35.xyz));
    let _e40 = r_data.proj_inv;
    let unprojected = (_e40 * pos);
    return VertexOutput(pos, (inv_model_view * unprojected.xyz));
}

@fragment 
fn fs_main(_in: VertexOutput) -> @location(0) vec4<f32> {
    let _e5 = textureSample(r_texture, r_sampler, _in.uv);
    return _e5;
}
