@group(0)
@binding(0)
var<storage, read_write> buf: array<u32>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    buf[global_id.x] = buf[global_id.x] + global_id.x;
}
