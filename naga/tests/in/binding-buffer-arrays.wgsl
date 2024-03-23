struct UniformIndex {
    index: u32
}

struct Foo { x: u32, far: array<i32> }
@group(0) @binding(0)
var<storage, read> storage_array: binding_array<Foo, 1>;
@group(0) @binding(10)
var<uniform> uni: UniformIndex;

struct FragmentIn {
    @location(0) index: u32,
}

@fragment
fn main(fragment_in: FragmentIn) -> @location(0) u32 {
    let uniform_index = uni.index;
    let non_uniform_index = fragment_in.index;

    var u1 = 0u;

    u1 += storage_array[0].x;
    u1 += storage_array[uniform_index].x;
    u1 += storage_array[non_uniform_index].x;

    u1 += arrayLength(&storage_array[0].far);

    return u1;
}
