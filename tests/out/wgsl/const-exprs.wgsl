@group(0) @binding(0) 
var<storage, read_write> out: vec4<i32>;
@group(0) @binding(1) 
var<storage, read_write> out2_: i32;
@group(0) @binding(2) 
var<storage, read_write> out3_: i32;

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let a = vec2<i32>(1, 2);
    let b = vec2<i32>(3, 4);
    out = vec4<i32>(4, 3, 2, 1);
    out2_ = 2;
    out3_ = 6;
    return;
}
