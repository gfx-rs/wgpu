struct VertexData {
    position: vec2<f32>,
    a: vec2<f32>,
}

struct FragmentData {
    position: vec2<f32>,
    a: vec2<f32>,
}

struct TestStruct {
    a: f32,
    b: f32,
}

struct LightScatteringParams {
    BetaRay: f32,
    BetaMie: array<f32, 3>,
    HGg: f32,
    DistanceMul: array<f32, 4>,
    BlendCoeff: f32,
    SunDirection: vec3<f32>,
    SunColor: vec3<f32>,
}

struct FragmentOutput {
    @location(0) position: vec2<f32>,
    @location(1) a: vec2<f32>,
    @location(2) out_array: vec4<f32>,
    @location(3) out_array_1: vec4<f32>,
}

var<private> vert: VertexData;
var<private> frag: FragmentData;
var<private> in_array_2: array<vec4<f32>, 2>;
var<private> out_array: array<vec4<f32>, 2>;
var<private> array_2d: array<array<f32, 2>, 2>;
var<private> array_toomanyd: array<array<array<array<array<array<array<f32, 2>, 2>, 2>, 2>, 2>, 2>, 2>;

fn main_1() {
    var positions: array<vec3<f32>, 2> = array<vec3<f32>, 2>(vec3<f32>(-1f, 1f, 0f), vec3<f32>(-1f, -1f, 0f));
    var strct: TestStruct = TestStruct(1f, 2f);
    var from_input_array: vec4<f32>;
    var a_1: f32;
    var b: f32;
    var light_scattering_params: LightScatteringParams;

    let _e17 = in_array_2[1];
    from_input_array = _e17;
    let _e21 = array_2d[0][0];
    a_1 = _e21;
    let _e30 = array_toomanyd[0][0][0][0][0][0][0];
    b = _e30;
    out_array[0i] = vec4(2f);
    return;
}

@fragment 
fn main(@location(0) position: vec2<f32>, @location(1) a: vec2<f32>, @location(2) in_array: vec4<f32>, @location(3) in_array_1: vec4<f32>) -> FragmentOutput {
    vert.position = position;
    vert.a = a;
    in_array_2[0] = in_array;
    in_array_2[1] = in_array_1;
    main_1();
    let _e12 = frag.position;
    let _e14 = frag.a;
    let _e17 = out_array[0];
    let _e19 = out_array[1];
    return FragmentOutput(_e12, _e14, _e17, _e19);
}
