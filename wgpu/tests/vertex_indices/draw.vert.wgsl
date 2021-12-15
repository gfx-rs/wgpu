struct Indices {
    arr: array<u32>;
}; // this is used as both input and output for convenience

[[group(0), binding(0)]]
var<storage, read_write> indices: Indices;

[[stage(vertex)]]
fn vs_main([[builtin(instance_index)]] instance: u32, [[builtin(vertex_index)]] index: u32) -> [[builtin(position)]] vec4<f32> {
    let idx = instance * 3u + index;
    indices.arr[idx] = idx;
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

[[stage(fragment)]]
fn fs_main() -> [[location(0)]] vec4<f32> {
    return vec4<f32>(0.0);
}
