@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_storage: texture_storage_2d<rgba8uint, write>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
}
