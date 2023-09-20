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

@compute @workgroup_size(1, 1, 1) 
fn main() {
    swizzle_of_compose();
    index_of_compose();
    compose_three_deep();
    return;
}
