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

@stage(compute) @workgroup_size(1)
fn main() {
    wg[3] = alignment.v1;
    wg[2] = alignment.v3.x;
    var _ = alignment.v3;
    var _ = alignment.v3.zx;
    alignment.v1 = 4.0;
    wg[1] = f32(arrayLength(&dummy));
    atomicStore(&at, 2u);

    alignment.v3 = vec3<f32>(1.0);
    var idx = 1;
    alignment.v3.x = 1.0;
    alignment.v3[0] = 2.0;
    alignment.v3[idx] = 3.0;

    let m = mat3x3<f32>();
    let _ = alignment.v3 * m;
    let _ = m * alignment.v3;

    // Valid, Foo and at is in function scope
    var Foo: f32 = 1.0;
    var at: bool = true;
}
