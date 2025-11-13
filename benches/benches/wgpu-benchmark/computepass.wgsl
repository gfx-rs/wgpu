@group(0) @binding(0)
var tex_0: texture_2d<f32>;

@group(0) @binding(1)
var tex_1: texture_2d<f32>;

@group(0) @binding(2)
var image_0: texture_storage_2d<r32float, read_write>;

@group(0) @binding(3)
var image_1: texture_storage_2d<r32float, read_write>;

@group(0) @binding(4)
var<storage, read_write> buffer0 : array<vec4f>;

@group(0) @binding(5)
var<storage, read_write> buffer1 : array<vec4f>;

@compute
@workgroup_size(16)
fn cs_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let tex = textureLoad(tex_0, vec2u(0), 0) + textureLoad(tex_1, vec2u(0), 0);
    let image = textureLoad(image_0, vec2u(0)) + textureLoad(image_1, vec2u(0));
    buffer0[0] = tex.rrrr;
    buffer1[0] = image.rrrr;
}
