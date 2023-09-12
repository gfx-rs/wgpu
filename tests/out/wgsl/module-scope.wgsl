struct S {
    x: i32,
}

const Value: i32 = 1;

@group(0) @binding(0) 
var Texture: texture_2d<f32>;
@group(0) @binding(1) 
var Sampler: sampler;

fn returns() -> S {
    return S(Value);
}

fn statement() {
    return;
}

fn call() {
    statement();
    let _e0 = returns();
    let vf = f32(Value);
    let s = textureSample(Texture, Sampler, vec2(vf));
}

