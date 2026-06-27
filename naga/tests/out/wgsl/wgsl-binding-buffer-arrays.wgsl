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
    let _e29 = storage_array[7].x;
    u1_ = (_e25 + _e29);
    let _e31 = u1_;
    let _e36 = storage_array[0].nested.y;
    u1_ = (_e31 + _e36);
    let _e38 = u1_;
    let _e43 = storage_array[uniform_index].nested.y;
    u1_ = (_e38 + _e43);
    let _e45 = u1_;
    let _e50 = storage_array[non_uniform_index].nested.y;
    u1_ = (_e45 + _e50);
    let _e52 = u1_;
    let _e57 = storage_array[7].nested.y;
    u1_ = (_e52 + _e57);
    let _e59 = u1_;
    u1_ = (_e59 + arrayLength((&storage_array[0].far)));
    let _e65 = u1_;
    u1_ = (_e65 + arrayLength((&storage_array[uniform_index].far)));
    let _e71 = u1_;
    u1_ = (_e71 + arrayLength((&storage_array[non_uniform_index].far)));
    let _e77 = u1_;
    u1_ = (_e77 + arrayLength((&storage_array[7].far)));
    let _e83 = u1_;
    let _e88 = storage_array[0].far[0];
    u1_ = (_e83 + bitcast<u32>(_e88));
    let _e91 = u1_;
    let _e96 = storage_array[uniform_index].far[0];
    u1_ = (_e91 + bitcast<u32>(_e96));
    let _e99 = u1_;
    let _e104 = storage_array[non_uniform_index].far[0];
    u1_ = (_e99 + bitcast<u32>(_e104));
    let _e107 = u1_;
    let _e112 = storage_array[7].far[0];
    u1_ = (_e107 + bitcast<u32>(_e112));
    let _e115 = u1_;
    let _e119 = plain_storage.values[0];
    u1_ = (_e115 + _e119);
    let _e121 = u1_;
    u1_ = (_e121 + arrayLength((&plain_storage.values)));
    let _e126 = u1_;
    return _e126;
}
