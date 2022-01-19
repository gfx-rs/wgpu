struct VertexData {
    position: vec2<f32>;
    a: vec2<f32>;
};

struct FragmentData {
    position: vec2<f32>;
    a: vec2<f32>;
};

struct TestStruct {
    a: f32;
    b: f32;
};

struct VertexOutput {
    @location(0) position: vec2<f32>;
    @location(1) a: vec2<f32>;
};

var<private> vert: VertexData;
var<private> frag: FragmentData;

fn main_1() {
    var positions: array<vec3<f32>,2u> = array<vec3<f32>,2u>(vec3<f32>(-1.0, 1.0, 0.0), vec3<f32>(-1.0, -1.0, 0.0));
    var strct: TestStruct = TestStruct(1.0, 2.0);

}

@stage(vertex) 
fn main(@location(0) position: vec2<f32>, @location(1) a: vec2<f32>) -> VertexOutput {
    vert.position = position;
    vert.a = a;
    main_1();
    let _e17 = frag.position;
    let _e19 = frag.a;
    return VertexOutput(_e17, _e19);
}
