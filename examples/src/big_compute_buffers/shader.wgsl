const OFFSET: u32 = 1u << 8u;
const BUFF_LENGTH: u32 = 1u << 25u;
const NUM_BUFFERS: u32 = 8u;
const TOTAL_SIZE: u32 = BUFF_LENGTH * NUM_BUFFERS;


// `binding_array` requires a custom struct
struct ContigiousArray{
    inner: array<f32>
}

@group(0) @binding(0)
var<storage, read_write> storage_array: binding_array<ContigiousArray, NUM_BUFFERS>;


@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_index = global_id.x * OFFSET;

    for (var i = 0u; i < OFFSET; i++) {
        let index = base_index + i;

        if (index < TOTAL_SIZE) {
            let buffer_index = index / BUFF_LENGTH;
            let inner_index = index % BUFF_LENGTH;

            storage_array[buffer_index].inner[inner_index] = add_one(storage_array[buffer_index].inner[inner_index]);
        }
    }
}

fn add_one(n: f32) -> f32 {
    return n + 1.0;
}
