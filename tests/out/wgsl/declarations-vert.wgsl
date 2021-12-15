struct VertexData {
    position: vec2<f32>;
    a: vec2<f32>;
};

struct FragmentData {
    position: vec2<f32>;
    a: vec2<f32>;
};

struct VertexOutput {
    [[location(0)]] position: vec2<f32>;
    [[location(1)]] a: vec2<f32>;
};

var<private> vert: VertexData;
var<private> frag: FragmentData;

fn main_1() {
}

[[stage(vertex)]]
fn main([[location(0)]] position: vec2<f32>, [[location(1)]] a: vec2<f32>) -> VertexOutput {
    vert.position = position;
    vert.a = a;
    main_1();
    let _e17 = frag.position;
    let _e19 = frag.a;
    return VertexOutput(_e17, _e19);
}
