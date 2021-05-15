fn splat() -> vec4<f32> {
    return ((((vec2<f32>(1.0) + vec2<f32>(2.0)) - vec2<f32>(3.0)) / vec2<f32>(4.0)).xyxy + vec4<f32>((vec4<i32>(5) % vec4<i32>(2))));
}

fn unary() -> i32 {
    if (!(true)) {
        return 1;
    } else {
        return ~(1);
    }
}

