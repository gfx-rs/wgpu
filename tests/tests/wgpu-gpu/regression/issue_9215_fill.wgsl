@group(0) @binding(0)
var<storage, read_write> scratch: u32;

@compute
@workgroup_size(1)
fn main() {
    scratch = 1u;
}
