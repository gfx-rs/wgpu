@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

fn color() {
  _ = textureSample(t, s, vec2(1,2));
  _ = textureSample(t, s, vec2(1,2), vec2(3,4));
  _ = textureSampleLevel(t, s, vec2(1,2), 0);
  _ = textureSampleLevel(t, s, vec2(1,2), 0.0);
  _ = textureSampleGrad(t, s, vec2(1,2), vec2(3,4), vec2(5,6));
  _ = textureSampleBias(t, s, vec2(1,2), 1);
}

@group(0) @binding(2) var d: texture_depth_2d;
@group(0) @binding(3) var c: sampler_comparison;

fn depth() {
  _ = textureSampleLevel(d, s, vec2(1,2), 1i);
  _ = textureSampleCompare(d, c, vec2(1,2), 0);
  _ = textureGatherCompare(d, c, vec2(1,2), 0);
}

@group(0) @binding(4) var st: texture_storage_2d<rgba8unorm, read_write>;

fn storage() {
  textureStore(st, vec2(0,1), vec4(2,3,4,5));
}

@fragment
fn main() {
    color();
    depth();
    storage();
}
