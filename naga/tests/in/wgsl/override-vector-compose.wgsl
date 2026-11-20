// Vector constructors with vector arguments in override expressions contain
// `AccessIndex` expressions until overrides are processed.
override gain: f32 = 1.0;

var<private> override_compose: vec4<f32> = vec4<f32>(vec2<f32>(gain, 1.0), 2.0, 3.0);

@compute @workgroup_size(1)
fn main() {
    _ = override_compose;
}
