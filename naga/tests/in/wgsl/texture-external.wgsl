@group(0) @binding(0)
var tex: texture_external;
@group(0) @binding(1)
var samp: sampler;

fn test(t: texture_external) -> vec4<f32> {
  var a = textureSampleBaseClampToEdge(t, samp, vec2(0.0f));
  var b = textureLoad(t, vec2(0u));
  var c = textureDimensions(t);

  return a + b + vec2f(c).xyxy;
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
  return test(tex);
}

@vertex
fn vertex_main() -> @builtin(position) vec4<f32> {
  return test(tex);
}

@compute @workgroup_size(1)
fn compute_main() {
  test(tex);
}
