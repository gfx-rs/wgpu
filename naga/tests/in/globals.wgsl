// Global variable & constant declarations

const Foo: bool = true;

var<workgroup> wg : array<f32, 10u>;
var<workgroup> at: atomic<u32>;

struct FooStruct {
    v3: vec3<f32>,
    // test packed vec3
    v1: f32,
}
@group(0) @binding(1)
var<storage, read_write> alignment: FooStruct;

@group(0) @binding(2)
var<storage> dummy: array<vec2<f32>>;

@group(0) @binding(3)
var<uniform> float_vecs: array<vec4<f32>, 20>;

@group(0) @binding(4)
var<uniform> global_vec: vec3<f32>;

@group(0) @binding(5)
var<uniform> global_mat: mat3x2<f32>;

@group(0) @binding(6)
var<uniform> global_nested_arrays_of_matrices_2x4: array<array<mat2x4<f32>, 2>, 2>;

@group(0) @binding(7)
var<uniform> global_nested_arrays_of_matrices_4x2: array<array<mat4x2<f32>, 2>, 2>;

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
    let l0 = data.v3;
    let l1 = data.v3.zx;
    test_msl_packed_vec3_as_arg(data.v3);

    // matrix vector multiplication
    let mvm0 = data.v3 * mat3x3<f32>();
    let mvm1 = mat3x3<f32>() * data.v3;

    // scalar vector multiplication
    let svm0 = data.v3 * 2.0;
    let svm1 = 2.0 * data.v3;
}

@compute @workgroup_size(1)
fn main() {
    test_msl_packed_vec3();

    wg[7] = (global_nested_arrays_of_matrices_4x2[0][0] * global_nested_arrays_of_matrices_2x4[0][0][0]).x;
    wg[6] = (global_mat * global_vec).x;
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
