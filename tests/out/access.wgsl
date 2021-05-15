[[block]]
struct Bar {
    matrix: mat4x4<f32>;
    data: [[stride(4)]] array<i32>;
};

[[group(0), binding(0)]]
var<storage> bar: [[access(read_write)]] Bar;

[[stage(vertex)]]
fn foo([[builtin(vertex_index)]] vi: u32) -> [[builtin(position)]] vec4<f32> {
    return vec4<f32>(vec4<i32>(array<i32,5>(bar.data[(arrayLength(&bar.data) - 1u)], i32(bar.matrix[3u].x), 3, 4, 5)[vi]));
}
