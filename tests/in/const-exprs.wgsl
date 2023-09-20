@group(0) @binding(0) var<storage, read_write> out: vec4<i32>;
@group(0) @binding(1) var<storage, read_write> out2: i32;

@compute @workgroup_size(1)
fn main() {
   swizzle_of_compose();
   index_of_compose();
   compose_three_deep();
}

// Swizzle the value of nested Compose expressions.
fn swizzle_of_compose() {
   let a = vec2(1, 2);
   let b = vec2(3, 4);
   out = vec4(a, b).wzyx; // should assign vec4(4, 3, 2, 1);
}

// Index the value of nested Compose expressions.
fn index_of_compose() {
   let a = vec2(1, 2);
   let b = vec2(3, 4);
   out2 += vec4(a, b)[1]; // should assign 2
}

// Index the value of Compose expressions nested three deep
fn compose_three_deep() {
   out2 += vec4(vec3(vec2(6, 7), 8), 9)[0]; // should assign 6
}
