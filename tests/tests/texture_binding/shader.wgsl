@group(0) @binding(0)
var tex: texture_storage_2d<rg32float, read>;

@compute @workgroup_size(1) fn csStore() {
    _ = textureLoad(tex, vec2u(0));
}