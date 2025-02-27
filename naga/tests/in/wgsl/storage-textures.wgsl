@group(0) @binding(0) var s_r_r: texture_storage_2d<r32float, read>;
@group(0) @binding(1) var s_rg_r: texture_storage_2d<rg32float, read>;
@group(0) @binding(2) var s_rgba_r: texture_storage_2d<rgba32float, read>;
@compute @workgroup_size(1) fn csLoad() {
    _ = textureLoad(s_r_r, vec2u(0));
    _ = textureLoad(s_rg_r, vec2u(0));
    _ = textureLoad(s_rgba_r, vec2u(0));
}

@group(1) @binding(0) var s_r_w: texture_storage_2d<r32float, write>;
@group(1) @binding(1) var s_rg_w: texture_storage_2d<rg32float, write>;
@group(1) @binding(2) var s_rgba_w: texture_storage_2d<rgba32float, write>;
@compute @workgroup_size(1) fn csStore() {
    textureStore(s_r_w, vec2u(0), vec4f(0.0));
    textureStore(s_rg_w, vec2u(0), vec4f(0.0));
    textureStore(s_rgba_w, vec2u(0), vec4f(0.0));
}