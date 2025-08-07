@group(0) @binding(0)
var tex: texture_external;

@group(0) @binding(1)
var<storage, read_write> output: vec2<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    output = textureDimensions(tex);
}
