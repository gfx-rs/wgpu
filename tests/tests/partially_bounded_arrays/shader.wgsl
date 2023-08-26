@group(0)
@binding(0)
var texture_array_storage: binding_array<texture_storage_2d<rgba32float,write>,1>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
 
 textureStore(texture_array_storage[0],vec2<i32>(0,0), vec4<f32>(4.0,3.0,2.0,1.0));

}
