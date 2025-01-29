struct NoPadding {
    @location(0) v3_: vec3<f32>,
    @location(1) f3_: f32,
}

struct NeedsPadding {
    @location(0) f3_forces_padding: f32,
    @location(1) v3_needs_padding: vec3<f32>,
    @location(2) f3_: f32,
}

@group(0) @binding(0) 
var<uniform> no_padding_uniform: NoPadding;
@group(0) @binding(1) 
var<storage, read_write> no_padding_storage: NoPadding;
@group(0) @binding(2) 
var<uniform> needs_padding_uniform: NeedsPadding;
@group(0) @binding(3) 
var<storage, read_write> needs_padding_storage: NeedsPadding;

@fragment 
fn no_padding_frag(input: NoPadding) -> @location(0) vec4<f32> {
    return vec4(0f);
}

@vertex 
fn no_padding_vert(input_1: NoPadding) -> @builtin(position) vec4<f32> {
    return vec4(0f);
}

@compute @workgroup_size(16, 1, 1) 
fn no_padding_comp() {
    var x: NoPadding;

    let _e2 = no_padding_uniform;
    x = _e2;
    let _e4 = no_padding_storage;
    x = _e4;
    return;
}

@fragment 
fn needs_padding_frag(input_2: NeedsPadding) -> @location(0) vec4<f32> {
    return vec4(0f);
}

@vertex 
fn needs_padding_vert(input_3: NeedsPadding) -> @builtin(position) vec4<f32> {
    return vec4(0f);
}

@compute @workgroup_size(16, 1, 1) 
fn needs_padding_comp() {
    var x_1: NeedsPadding;

    let _e2 = needs_padding_uniform;
    x_1 = _e2;
    let _e4 = needs_padding_storage;
    x_1 = _e4;
    return;
}
