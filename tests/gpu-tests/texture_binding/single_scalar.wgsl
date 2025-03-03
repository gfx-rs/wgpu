@group(0) @binding(0)
var tex_w: texture_storage_2d<rgba32float, write>;
@group(0) @binding(1)
var tex_r: texture_storage_2d<r32float, read>;

@compute @workgroup_size(1) fn csStore() {
    textureStore(tex_w, vec2u(0), textureLoad(tex_r, vec2u(0)));
}