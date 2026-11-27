override supports_indirect_first_instance: bool;
override write_d3d12_special_constants: bool;

struct MetadataEntry {
    // offset into `src`
    src_offset: u32,
    // offset into `dst`
    dst_offset: u32,
    vertex_or_index_limit: u32,
    instance_limit: u32,
    // bit 0 is set for an indexed draw
    // bit 1 is the most significant bit of `vertex_or_index_limit`
    // bit 2 is the most significant bit of `instance_limit`
    //
    // The flags are not packed into the offsets: the Metal compiler for some
    // Apple GPUs miscompiles address arithmetic on a value that also has
    // tested flag bits. See <https://github.com/gfx-rs/wgpu/issues/9187>.
    flags: u32,
}

struct MetadataRange {
    start: u32,
    count: u32,
}
var<immediate> metadata_range: MetadataRange;

@group(0) @binding(0)
var<storage, read> metadata: array<MetadataEntry>;
@group(1) @binding(0)
var<storage, read> src: array<u32>;
@group(2) @binding(0)
var<storage, read_write> dst: array<u32>;

fn is_bit_set(data: u32, index: u32) -> bool {
    return ((data >> index) & 1u) == 1u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3u) {
    if global_invocation_id.x >= metadata_range.count { return; }

    let metadata = metadata[metadata_range.start + global_invocation_id.x];
    var failed = false;

    let is_indexed = is_bit_set(metadata.flags, 0);
    let src_base_offset = metadata.src_offset;
    let dst_base_offset = metadata.dst_offset;

    let first_vertex_or_index = src[src_base_offset + 2];
    let vertex_or_index_count = src[src_base_offset + 0];

    {
        let overflow_permitted = is_bit_set(metadata.flags, 1);
        let sub_overflows = metadata.vertex_or_index_limit < first_vertex_or_index;
        failed |= sub_overflows && !overflow_permitted;
        let vertex_or_index_limit = metadata.vertex_or_index_limit - first_vertex_or_index;
        failed |= vertex_or_index_limit < vertex_or_index_count;
    }

    let first_instance = src[src_base_offset + 3 + u32(is_indexed)];
    let instance_count = src[src_base_offset + 1];

    {
        let overflow_permitted = is_bit_set(metadata.flags, 2);
        let sub_overflows = metadata.instance_limit < first_instance;
        failed |= sub_overflows && !overflow_permitted;
        let instance_limit = metadata.instance_limit - first_instance;
        failed |= instance_limit < instance_count;
    }

    if !supports_indirect_first_instance {
        failed |= first_instance != 0u;
    }

    let dst_offset = select(0u, 3u, write_d3d12_special_constants);
    if failed {
        if write_d3d12_special_constants {
            dst[dst_base_offset + 0] = 0u;
            dst[dst_base_offset + 1] = 0u;
            dst[dst_base_offset + 2] = 0u;
        }
        dst[dst_base_offset + dst_offset + 0] = 0u;
        dst[dst_base_offset + dst_offset + 1] = 0u;
        dst[dst_base_offset + dst_offset + 2] = 0u;
        dst[dst_base_offset + dst_offset + 3] = 0u;
        if (is_indexed) {
            dst[dst_base_offset + dst_offset + 4] = 0u;
        }
    } else {
        if write_d3d12_special_constants {
            dst[dst_base_offset + 0] = src[src_base_offset + 2 + u32(is_indexed)];
            dst[dst_base_offset + 1] = src[src_base_offset + 3 + u32(is_indexed)];
            dst[dst_base_offset + 2] = 0u;
        }
        dst[dst_base_offset + dst_offset + 0] = src[src_base_offset + 0];
        dst[dst_base_offset + dst_offset + 1] = src[src_base_offset + 1];
        dst[dst_base_offset + dst_offset + 2] = src[src_base_offset + 2];
        dst[dst_base_offset + dst_offset + 3] = src[src_base_offset + 3];
        if (is_indexed) {
            dst[dst_base_offset + dst_offset + 4] = src[src_base_offset + 4];
        }
    }
}
