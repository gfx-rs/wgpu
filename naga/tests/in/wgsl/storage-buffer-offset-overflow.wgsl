@group(0) @binding(0) var<storage, read_write> buf: array<u32>;

@compute @workgroup_size(1)
fn main() {
  buf[1073741824] = buf[0];
  buf[0] = buf[1073741824];
}
