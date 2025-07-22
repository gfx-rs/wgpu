@group(0) @binding(0) 
var tex: texture_external;
@group(0) @binding(1) 
var samp: sampler;

fn test(t: texture_external) -> vec4<f32> {
    var a: vec4<f32>;
    var b: vec4<f32>;
    var c: vec2<u32>;

    let _e4 = textureSampleBaseClampToEdge(t, samp, vec2(0f));
    a = _e4;
    let _e8 = textureLoad(t, vec2(0u));
    b = _e8;
    let _e10 = textureDimensions(t);
    c = _e10;
    let _e12 = a;
    let _e13 = b;
    let _e15 = c;
    return ((_e12 + _e13) + vec2<f32>(_e15).xyxy);
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
