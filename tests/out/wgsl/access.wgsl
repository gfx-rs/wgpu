[[block]]
struct Bar {
    matrix: mat4x4<f32>;
    atom: atomic<i32>;
    arr: [[stride(8)]] array<vec2<u32>,2>;
    data: [[stride(8)]] array<i32>;
};

[[group(0), binding(0)]]
var<storage, read_write> bar: Bar;

fn read_from_private(foo_2: ptr<function, f32>) -> f32 {
    let e2: f32 = (*foo_2);
    return e2;
}

[[stage(vertex)]]
fn foo([[builtin(vertex_index)]] vi: u32) -> [[builtin(position)]] vec4<f32> {
    var foo_1: f32 = 0.0;
    var c: array<i32,5>;

    let baz: f32 = foo_1;
    foo_1 = 1.0;
    let matrix: mat4x4<f32> = bar.matrix;
    let arr: array<vec2<u32>,2> = bar.arr;
    let b: f32 = bar.matrix[3][0];
    let a: i32 = bar.data[(arrayLength((&bar.data)) - 2u)];
    let data_pointer: ptr<storage, i32, read_write> = (&bar.data[0]);
    let e25: f32 = read_from_private((&foo_1));
    bar.matrix[1][2] = 1.0;
    bar.matrix = mat4x4<f32>(vec4<f32>(0.0), vec4<f32>(1.0), vec4<f32>(2.0), vec4<f32>(3.0));
    bar.arr = array<vec2<u32>,2>(vec2<u32>(0u), vec2<u32>(1u));
    bar.data[1] = 1;
    c = array<i32,5>(a, i32(b), 3, 4, 5);
    c[(vi + 1u)] = 42;
    let value: i32 = c[vi];
    return (matrix * vec4<f32>(vec4<i32>(value)));
}

[[stage(compute), workgroup_size(1, 1, 1)]]
fn atomics() {
    var tmp: i32;

    let value_1: i32 = atomicLoad((&bar.atom));
    let e6: i32 = atomicAdd((&bar.atom), 5);
    tmp = e6;
    let e9: i32 = atomicSub((&bar.atom), 5);
    tmp = e9;
    let e12: i32 = atomicAnd((&bar.atom), 5);
    tmp = e12;
    let e15: i32 = atomicOr((&bar.atom), 5);
    tmp = e15;
    let e18: i32 = atomicXor((&bar.atom), 5);
    tmp = e18;
    let e21: i32 = atomicMin((&bar.atom), 5);
    tmp = e21;
    let e24: i32 = atomicMax((&bar.atom), 5);
    tmp = e24;
    let e27: i32 = atomicExchange((&bar.atom), 5);
    tmp = e27;
    atomicStore((&bar.atom), value_1);
    return;
}
