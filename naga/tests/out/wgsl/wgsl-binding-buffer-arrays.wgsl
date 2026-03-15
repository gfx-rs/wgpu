struct UniformIndex {
    index: u32,
}

struct Foo {
    x: u32,
    far: array<i32>,
}

struct FragmentIn {
    @location(0) @interpolate(flat) index: u32,
}

@group(0) @binding(0) 
var<storage> storage_array: binding_array<Foo, 1>;
@group(0) @binding(10) 
var<uniform> uni: UniformIndex;

@fragment 
fn main(fragment_in: FragmentIn) -> @location(0) @interpolate(flat) u32 {
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
    u1_ = (_e25 + arrayLength((&storage_array[0].far)));
    let _e31 = u1_;
    u1_ = (_e31 + arrayLength((&storage_array[uniform_index].far)));
    let _e37 = u1_;
    u1_ = (_e37 + arrayLength((&storage_array[non_uniform_index].far)));
    let _e43 = u1_;
    return _e43;
}
