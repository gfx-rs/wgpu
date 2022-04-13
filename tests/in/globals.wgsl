// Global variable & constant declarations

let Foo: bool = true;

var<workgroup> wg : array<f32, 10u>;
var<workgroup> at: atomic<u32>;

struct Foo {
    v3: vec3<f32>,
    // test packed vec3
    v1: f32,
}
@group(0) @binding(1)
var<storage, read_write> alignment: Foo;

@group(0) @binding(2)
var<storage> dummy: array<vec2<f32>>;

@group(0) @binding(3)
var<uniform> float_vecs: array<vec4<f32>, 20>;

@group(0) @binding(4)
var<uniform> global_vec: vec4<f32>;

fn test_msl_packed_vec3_as_arg(arg: vec3<f32>) {}

fn test_msl_packed_vec3() {
    // stores
    alignment.v3 = vec3<f32>(1.0);
    var idx = 1;
    alignment.v3.x = 1.0;
    alignment.v3[0] = 2.0;
    alignment.v3[idx] = 3.0;

    // force load to happen here
    let data = alignment;

    // loads
    let _ = data.v3;
    let _ = data.v3.zx;
    test_msl_packed_vec3_as_arg(data.v3);

    // matrix vector multiplication
    let _ = data.v3 * mat3x3<f32>();
    let _ = mat3x3<f32>() * data.v3;

    // scalar vector multiplication
    let _ = data.v3 * 2.0;
    let _ = 2.0 * data.v3;
}

@stage(compute) @workgroup_size(1)
fn main() {
    test_msl_packed_vec3();

    wg[6] = global_vec.x;
    wg[5] = dummy[1].y;
    wg[4] = float_vecs[0].w;
    wg[3] = alignment.v1;
    wg[2] = alignment.v3.x;
    alignment.v1 = 4.0;
    wg[1] = f32(arrayLength(&dummy));
    atomicStore(&at, 2u);

    // Valid, Foo and at is in function scope
    var Foo: f32 = 1.0;
    var at: bool = true;
}
