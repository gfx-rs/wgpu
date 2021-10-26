[[block]]
struct TextureData {
    material: vec4<f32>;
};

[[group(1), binding(1)]]
var<uniform> global: TextureData;

fn main1() {
    var coords: vec2<f32>;

    let e2: vec4<f32> = global.material;
    coords = vec2<f32>(e2.xy);
    return;
}

[[stage(fragment)]]
fn main() {
    main1();
    return;
}
