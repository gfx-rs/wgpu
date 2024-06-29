@group(0) @binding(0)
var tex: binding_array<texture_2d<f32>>;

@group(0) @binding(1)
// TODO(https://github.com/gfx-rs/wgpu/issues/5765): The extra whitespace between the angle brackets is needed to workaround a parsing bug.
var images: binding_array<texture_storage_2d<r32float, read_write> >;
struct BufferElement {
    element: vec4f,
}

@group(0) @binding(2)
var<storage, read_write> buffers: binding_array<BufferElement>;

@compute
@workgroup_size(16)
fn cs_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let offset = global_invocation_id.x; // Would be nice to offset this dynamically (it's just 0 always in the current setup)
    
    let idx0 = offset * 2 + 0;
    let idx1 = offset * 2 + 1;
    
    let tex = textureLoad(tex[idx0], vec2u(0), 0) + textureLoad(tex[idx0], vec2u(0), 0);
    let image = textureLoad(images[idx0], vec2u(0)) + textureLoad(images[idx1], vec2u(0));
    buffers[idx0].element = tex.rrrr;
    buffers[idx1].element = image.rrrr;
}