@compute @workgroup_size(1, 1)
fn main() {
    let idx = 2;
    _ = unpack4xI8(12u)[idx];
    _ = unpack4xU8(12u)[1];
}
