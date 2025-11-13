@group(0) @binding(0) 
var tex: texture_external;
@group(0) @binding(1) 
var samp: sampler;

fn test(t: texture_external) -> vec4<f32> {
    var a: vec4<f32>;
    var b: vec4<f32>;
    var c: vec4<f32>;
    var d: vec2<u32>;

    let _e4 = textureSampleBaseClampToEdge(t, samp, vec2(0f));
    a = _e4;
    let _e8 = textureLoad(t, vec2(0i));
    b = _e8;
    let _e12 = textureLoad(t, vec2(0u));
    c = _e12;
    let _e14 = textureDimensions(t);
    d = _e14;
    let _e16 = a;
    let _e17 = b;
    let _e19 = c;
    let _e21 = d;
    return (((_e16 + _e17) + _e19) + vec2<f32>(_e21).xyxy);
}

@fragment 
fn fragment_main() -> @location(0) vec4<f32> {
    let _e1 = test(tex);
    return _e1;
}

@vertex 
fn vertex_main() -> @builtin(position) vec4<f32> {
    let _e1 = test(tex);
    return _e1;
}

@compute @workgroup_size(1, 1, 1) 
fn compute_main() {
    let _e1 = test(tex);
    return;
}
