@group(0) @binding(0) var t1: texture_2d<f32>;
@group(0) @binding(1) var t2: texture_2d<f32>;
@group(0) @binding(2) var s1: sampler;
@group(0) @binding(3) var s2: sampler;

@group(0) @binding(4) var<uniform> uniformOne: vec2<f32>;
@group(1) @binding(0) var<uniform> uniformTwo: vec2<f32>;

@fragment
fn entry_point_one(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return textureSample(t1, s1, pos.xy);
}

@fragment
fn entry_point_two() -> @location(0) vec4<f32> {
    return textureSample(t1, s1, uniformOne);
}

@fragment
fn entry_point_three() -> @location(0) vec4<f32> {
    return textureSample(t1, s1, uniformTwo + uniformOne) +
           textureSample(t2, s2, uniformOne);
}
