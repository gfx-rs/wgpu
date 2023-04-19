struct UniformIndex {
    index: u32,
}

struct Foo {
    x: u32,
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
    let _e10 = storage_array[0].x;
    let _e11 = u1_;
    u1_ = (_e11 + _e10);
    let _e16 = storage_array[uniform_index].x;
    let _e17 = u1_;
    u1_ = (_e17 + _e16);
    let _e22 = storage_array[non_uniform_index].x;
    let _e23 = u1_;
    u1_ = (_e23 + _e22);
    let _e25 = u1_;
    return _e25;
}
