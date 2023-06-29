@group(0)
@binding(0)
var<storage, read_write> a: array<i32>;

@group(0)
@binding(1)
var<storage, read_write> b: array<i32>;

@compute
@workgroup_size(2, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    a[0] += 1;
    if lid.x == 0u {
        a[wid.x] += 1;
    } else if lid.x == 1u {
        b[wid.x] += 1;
    }
}