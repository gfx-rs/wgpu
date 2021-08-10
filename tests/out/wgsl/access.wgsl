[[block]]
struct Bar {
    matrix: mat4x4<f32>;
    atom: atomic<i32>;
    arr: [[stride(8)]] array<vec2<u32>,2>;
    data: [[stride(4)]] array<i32>;
};

[[group(0), binding(0)]]
var<storage,read_write> bar: Bar;

[[stage(vertex)]]
fn foo([[builtin(vertex_index)]] vi: u32) -> [[builtin(position)]] vec4<f32> {
    var foo1: f32 = 0.0;
    var c: array<i32,5>;

    let baz: f32 = foo1;
    foo1 = 1.0;
    let matrix: mat4x4<f32> = bar.matrix;
    let arr: array<vec2<u32>,2> = bar.arr;
    let _e13: vec4<f32> = bar.matrix[3];
    let b: f32 = _e13.x;
    let a: i32 = bar.data[(arrayLength(&bar.data) - 2u)];
    bar.matrix[1][2] = 1.0;
    bar.matrix = mat4x4<f32>(vec4<f32>(0.0), vec4<f32>(1.0), vec4<f32>(2.0), vec4<f32>(3.0));
    bar.arr = array<vec2<u32>,2>(vec2<u32>(0u), vec2<u32>(1u));
    c = array<i32,5>(a, i32(b), 3, 4, 5);
    c[(vi + 1u)] = 42;
    let value: i32 = c[vi];
    return (matrix * vec4<f32>(vec4<i32>(value)));
}

[[stage(compute), workgroup_size(1, 1, 1)]]
fn atomics() {
    var tmp: i32;

    let value: i32 = atomicLoad(&bar.atom);
    let _e6: i32 = atomicAdd(&bar.atom, 5);
    tmp = _e6;
    let _e9: i32 = atomicAnd(&bar.atom, 5);
    tmp = _e9;
    let _e12: i32 = atomicOr(&bar.atom, 5);
    tmp = _e12;
    let _e15: i32 = atomicXor(&bar.atom, 5);
    tmp = _e15;
    let _e18: i32 = atomicMin(&bar.atom, 5);
    tmp = _e18;
    let _e21: i32 = atomicMax(&bar.atom, 5);
    tmp = _e21;
    let _e24: i32 = atomicExchange(&bar.atom, 5);
    tmp = _e24;
    atomicStore(&bar.atom, value);
    return;
}
