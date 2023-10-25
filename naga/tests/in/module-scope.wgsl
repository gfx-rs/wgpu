fn call() {
    statement();
    let x: S = returns();
    let vf = f32(Value);
    let s = textureSample(Texture, Sampler, Vec2(vf));
}

fn statement() {}

fn returns() -> S {
    return S(Value);
}

struct S {
    x: i32,
}

const Value: i32 = 1;

@group(0) @binding(0)
var Texture: texture_2d<f32>;

@group(0) @binding(1)
var Sampler: sampler;

alias Vec2 = vec2<f32>;
