struct Params {
    max_count: u32,
    stride: u32,
    src_offset: u32,
    count_offset: u32,
}

var<immediate> params: Params;

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read> count_buf: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let count = min(count_buf[params.count_offset], params.max_count);
    let src_base = params.src_offset + gid.x * params.stride;
    let dst_base = gid.x * params.stride;

    if (gid.x < count) {
        for (var i = 0u; i < params.stride; i = i + 1u) {
            dst[dst_base + i] = src[src_base + i];
        }
    } else if (gid.x < params.max_count) {
        for (var i = 0u; i < params.stride; i = i + 1u) {
            dst[dst_base + i] = 0u;
        }
    }
}
