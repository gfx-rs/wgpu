@group(0) @binding(0) var<storage, read_write> out: vec4<i32>;
@group(0) @binding(1) var<storage, read_write> out2: i32;
@group(0) @binding(2) var<storage, read_write> out3: i32;

@compute @workgroup_size(1)
fn main() {
   let a = vec2(1, 2);
   let b = vec2(3, 4);
   out = vec4(a, b).wzyx;

   out2 = vec4(a, b)[1];

   out3 = vec4(vec3(vec2(6, 7), 8), 9)[0];
}
