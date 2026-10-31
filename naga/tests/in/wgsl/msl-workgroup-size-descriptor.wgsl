@group(0) @binding(0) var<storage, read_write> values: array<u32>;

fn store(index: u32) {
    values[index] = index + 1u;
}

@compute @workgroup_size(4, 2, 1)
fn main(@builtin(local_invocation_index) index: u32) {
    store(index);
}
