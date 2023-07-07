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
var<storage> storage_array: binding_array<Foo,1>;
@group(0) @binding(10) 
var<uniform> uni: UniformIndex;

@fragment 
fn main(fragment_in: FragmentIn) -> @location(0) @interpolate(flat) u32 {
    var u1_: u32;

    let uniform_index = uni.index;
    let non_uniform_index = fragment_in.index;
    u1_ = 0u;
    let _e11 = storage_array[0].x;
    let _e12 = u1_;
    u1_ = (_e12 + _e11);
    let _e17 = storage_array[uniform_index].x;
    let _e18 = u1_;
    u1_ = (_e18 + _e17);
    let _e23 = storage_array[non_uniform_index].x;
    let _e24 = u1_;
    u1_ = (_e24 + _e23);
    let _e26 = u1_;
    return _e26;
}
