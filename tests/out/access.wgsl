[[block]]
struct Bar {
    matrix: mat4x4<f32>;
    data: [[stride(4)]] array<i32>;
};

[[group(0), binding(0)]]
var<storage> bar: [[access(read_write)]] Bar;

[[stage(vertex)]]
fn foo([[builtin(vertex_index)]] vi: u32) -> [[builtin(position)]] vec4<f32> {
    var foo1: f32 = 0.0;
    var c: array<i32,5>;

    let baz: f32 = foo1;
    foo1 = 1.0;
    let _e9: vec4<f32> = bar.matrix[3u];
    let b: f32 = _e9.x;
    let a: i32 = bar.data[(arrayLength(&bar.data) - 1u)];
    c = array<i32,5>(a, i32(b), 3, 4, 5);
    let value: i32 = c[vi];
    return vec4<f32>(vec4<i32>(value));
}
