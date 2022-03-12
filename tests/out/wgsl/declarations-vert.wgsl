struct VertexData {
    position: vec2<f32>,
    a: vec2<f32>,
};

struct FragmentData {
    position: vec2<f32>,
    a: vec2<f32>,
};

struct TestStruct {
    a: f32,
    b: f32,
};

struct VertexOutput {
    @location(0) position: vec2<f32>,
    @location(1) a: vec2<f32>,
    @location(2) out_array: vec4<f32>,
    @location(3) out_array_1: vec4<f32>,
};

var<private> vert: VertexData;
var<private> frag: FragmentData;
var<private> in_array_2: array<vec4<f32>,2u>;
var<private> out_array: array<vec4<f32>,2u>;

fn main_1() {
    var positions: array<vec3<f32>,2u> = array<vec3<f32>,2u>(vec3<f32>(-1.0, 1.0, 0.0), vec3<f32>(-1.0, -1.0, 0.0));
    var strct: TestStruct = TestStruct(1.0, 2.0);
    var from_input_array: vec4<f32>;

    let _e32 = in_array_2;
    from_input_array = _e32[1];
    out_array[0] = vec4<f32>(2.0);
    return;
}

@stage(vertex) 
fn main(@location(0) position: vec2<f32>, @location(1) a: vec2<f32>, @location(2) in_array: vec4<f32>, @location(3) in_array_1: vec4<f32>) -> VertexOutput {
    vert.position = position;
    vert.a = a;
    in_array_2[0] = in_array;
    in_array_2[1] = in_array_1;
    main_1();
    let _e26 = frag.position;
    let _e28 = frag.a;
    let _e31 = out_array[0];
    let _e33 = out_array[1];
    return VertexOutput(_e26, _e28, _e31, _e33);
}
