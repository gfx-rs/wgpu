@group(0) @binding(0) 
var<storage, read_write> out: vec4<i32>;
@group(0) @binding(1) 
var<storage, read_write> out2_: i32;

fn swizzle_of_compose() {
    let a = vec2<i32>(1, 2);
    let b = vec2<i32>(3, 4);
    out = vec4<i32>(4, 3, 2, 1);
    return;
}

fn index_of_compose() {
    let a_1 = vec2<i32>(1, 2);
    let b_1 = vec2<i32>(3, 4);
    let _e7 = out2_;
    out2_ = (_e7 + 2);
    return;
}

fn compose_three_deep() {
    let _e2 = out2_;
    out2_ = (_e2 + 6);
    return;
}

fn non_constant_initializers() {
    var w: i32 = 30;
    var x: i32;
    var y: i32;
    var z: i32 = 70;

    let _e2 = w;
    x = _e2;
    let _e4 = x;
    y = _e4;
    let _e9 = w;
    let _e10 = x;
    let _e11 = y;
    let _e12 = z;
    let _e14 = out;
    out = (_e14 + vec4<i32>(_e9, _e10, _e11, _e12));
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    swizzle_of_compose();
    index_of_compose();
    compose_three_deep();
    non_constant_initializers();
    return;
}
