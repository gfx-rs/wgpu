@group(0) @binding(0)
var tex: texture_storage_2d<r32uint, atomic>;

@compute @workgroup_size(8, 8, 1)
fn clear(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(tex);
    if id.x >= dims.x || id.y >= dims.y { return; }
    textureStore(tex, id.xy, vec4<u32>(0u));
}

@vertex
fn fullscreen(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) % 2) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn write_atomic(@builtin(position) pos: vec4<f32>) {
    textureAtomicMax(tex, vec2<u32>(pos.xy), 1u);
}

@group(0) @binding(0)
var tex_read: texture_storage_2d<r32uint, read>;

@fragment
fn visualize(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let val = textureLoad(tex_read, vec2<u32>(pos.xy)).r;
    if val == 1u { return vec4<f32>(0.0, 0.8, 0.0, 1.0); }
    if val == 0u { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
