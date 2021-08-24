struct VertexOutput {
    [[location(0)]] uv: vec2<f32>;
    [[builtin(position)]] position: vec4<f32>;
};

let c_scale: f32 = 1.2000000476837158;

[[group(0), binding(0)]]
var u_texture: texture_2d<f32>;
[[group(0), binding(1)]]
var u_sampler: sampler;

[[stage(vertex)]]
fn main([[location(0)]] pos: vec2<f32>, [[location(1)]] uv: vec2<f32>) -> VertexOutput {
    return VertexOutput(uv, vec4<f32>((c_scale * pos), 0.0, 1.0));
}

[[stage(fragment)]]
fn main1([[location(0)]] uv1: vec2<f32>) -> [[location(0)]] vec4<f32> {
    let color: vec4<f32> = textureSample(u_texture, u_sampler, uv1);
    if ((color.w == 0.0)) {
        discard;
    }
    let premultiplied: vec4<f32> = (color.w * color);
    return premultiplied;
}

[[stage(fragment)]]
fn fs_extra() -> [[location(0)]] vec4<f32> {
    return vec4<f32>(0.0, 0.5, 0.0, 0.5);
}
