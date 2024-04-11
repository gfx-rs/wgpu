// Not a multiple of the workgroup size
const array_size = 544u; // 8.5 * 64

var<workgroup> w_mem_array: array<array<u32,8>, array_size>;

@group(0) @binding(0)
var<storage, read_write> output: array<u32>;

@compute @workgroup_size(8, 4, 2)
fn read_(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    if all(local_id == vec3(0u, 0u, 0u)) {
        var is_zero = true;
        for (var i = 0u; i < array_size; i++) {
            for (var j = 0u; j < 8; j++) {
                is_zero &= w_mem_array[i][j] == 0u;
            }
        }

        let idx = wgid.x + (wgid.y * num_workgroups.x) + (wgid.z * num_workgroups.x * num_workgroups.y);
        output[idx] = u32(!is_zero);
    }
}

@compute @workgroup_size(1)
fn write_() {
    for (var i = 0u; i < array_size; i++) {
        for (var j = 0u; j < 8; j++) {
            w_mem_array[i][j] = i;
        }
    }
}
