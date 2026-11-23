override gain: f32 = 1f;

var<private> override_compose: vec4<f32> = vec4<f32>(vec2<f32>(gain, 1f)[0], vec2<f32>(gain, 1f)[1], 2f, 3f);

@compute @workgroup_size(1, 1, 1)
fn main() {
    let phony = override_compose;
    return;
}
