enable wgpu_binding_array;
struct UniformIndex {
    index: u32
}

struct Inner {
    y: u32,
}
struct Foo { x: u32, nested: Inner, far: array<i32> }

struct PlainData {
    values: array<u32>,
}

@group(0) @binding(0)
var<storage, read> storage_array: binding_array<Foo>;
@group(0) @binding(1)
var<storage, read> plain_storage: PlainData;
@group(0) @binding(10)
var<uniform> uni: UniformIndex;

struct FragmentIn {
    @location(0) @interpolate(flat) index: u32,
}

@fragment
fn main(fragment_in: FragmentIn) -> @location(0) u32 {
    let uniform_index = uni.index;
    let non_uniform_index = fragment_in.index;

    var u1 = 0u;

    u1 += storage_array[0].x;
    u1 += storage_array[uniform_index].x;
    u1 += storage_array[non_uniform_index].x;
    u1 += storage_array[7].x;

    u1 += storage_array[0].nested.y;
    u1 += storage_array[uniform_index].nested.y;
    u1 += storage_array[non_uniform_index].nested.y;
    u1 += storage_array[7].nested.y;

    u1 += arrayLength(&storage_array[0].far);
    u1 += arrayLength(&storage_array[uniform_index].far);
    u1 += arrayLength(&storage_array[non_uniform_index].far);
    u1 += arrayLength(&storage_array[7].far);

    u1 += bitcast<u32>(storage_array[0].far[0]);
    u1 += bitcast<u32>(storage_array[uniform_index].far[0]);
    u1 += bitcast<u32>(storage_array[non_uniform_index].far[0]);
    u1 += bitcast<u32>(storage_array[7].far[0]);

    u1 += plain_storage.values[0];
    u1 += arrayLength(&plain_storage.values);

    return u1;
}
