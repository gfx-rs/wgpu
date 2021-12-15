struct InOutBuffer {
    data: [[stride(4)]] array<u32>;
};

[[group(0), binding(0)]]
var<storage, read_write> buffer: InOutBuffer;

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    buffer.data[global_id.x] = buffer.data[global_id.x] + global_id.x;
}
