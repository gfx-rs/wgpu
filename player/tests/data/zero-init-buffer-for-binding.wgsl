@group(0)
@binding(0)
var<storage, read_write> buffer: array<u32>;

@stage(compute)
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    buffer[global_id.x] = buffer[global_id.x] + global_id.x;
}
