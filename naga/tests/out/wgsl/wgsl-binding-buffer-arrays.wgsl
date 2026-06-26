enable wgpu_binding_array;

struct UniformIndex {
    index: u32,
}

struct Inner {
    y: u32,
}

struct Foo {
    x: u32,
    nested: Inner,
    far: array<i32>,
}

struct PlainData {
    values: array<u32>,
}

struct FragmentIn {
    @location(0) @interpolate(flat) index: u32,
}

@group(0) @binding(0) 
var<storage> storage_array: binding_array<Foo>;
@group(0) @binding(1) 
var<storage> plain_storage: PlainData;
@group(0) @binding(10) 
var<uniform> uni: UniformIndex;

@fragment 
fn main(fragment_in: FragmentIn) -> @location(0) u32 {
    var u1_: u32 = 0u;

    let uniform_index = uni.index;
    let non_uniform_index = fragment_in.index;
    let _e7 = u1_;
    let _e11 = storage_array[0].x;
    u1_ = (_e7 + _e11);
    let _e13 = u1_;
    let _e17 = storage_array[uniform_index].x;
    u1_ = (_e13 + _e17);
    let _e19 = u1_;
    let _e23 = storage_array[non_uniform_index].x;
    u1_ = (_e19 + _e23);
    let _e25 = u1_;
    let _e30 = storage_array[0].nested.y;
    u1_ = (_e25 + _e30);
    let _e32 = u1_;
    let _e37 = storage_array[uniform_index].nested.y;
    u1_ = (_e32 + _e37);
    let _e39 = u1_;
    let _e44 = storage_array[non_uniform_index].nested.y;
    u1_ = (_e39 + _e44);
    let _e46 = u1_;
    u1_ = (_e46 + arrayLength((&storage_array[0].far)));
    let _e52 = u1_;
    u1_ = (_e52 + arrayLength((&storage_array[uniform_index].far)));
    let _e58 = u1_;
    u1_ = (_e58 + arrayLength((&storage_array[non_uniform_index].far)));
    let _e64 = u1_;
    let _e68 = plain_storage.values[0];
    u1_ = (_e64 + _e68);
    let _e70 = u1_;
    u1_ = (_e70 + arrayLength((&plain_storage.values)));
    let _e75 = u1_;
    return _e75;
}
