@group(0) @binding(0) 
var s_r: texture_storage_2d<r32float,read>;
@group(0) @binding(1) 
var s_rg: texture_storage_2d<rg32float,read>;
@group(0) @binding(2) 
var s_rgba: texture_storage_2d<rgba32float,read>;

@compute @workgroup_size(1, 1, 1) 
fn csWithStorageUsage() {
    let phony = textureLoad(s_r, vec2(0u));
    let phony_1 = textureLoad(s_rg, vec2(0u));
    let phony_2 = textureLoad(s_rgba, vec2(0u));
}
