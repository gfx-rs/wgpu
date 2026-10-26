// Keep in sync with select_vector_downlevel_webgl.wgsl
//
// `select` with a vector condition. GLSL's `mix` with a boolean vector selector
// only has integer and boolean overloads from GLSL ES 3.10 / GLSL 4.50 on, so
// for non-float vectors the GLSL backend writes the selection per component.

fn side_effect(x: i32) -> vec3<i32> {
    return vec3<i32>(x, x + 1, x + 2);
}

@fragment
fn main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let b2 = p.xy > vec2<f32>(0.5);
    let b3 = p.xyz > vec3<f32>(0.5);
    let b4 = p > vec4<f32>(0.5);

    let i2 = vec2<i32>(p.xy);
    let i3 = vec3<i32>(i32(p.x), -2, 7);
    let i4 = vec4<i32>(p);
    let u2 = vec2<u32>(p.xy);
    let u3 = vec3<u32>(p.xyz);
    let u4 = vec4<u32>(p);

    // f32: native `mix` is available everywhere.
    let f2 = select(p.xy, p.yx, b2);
    let f3 = select(p.xyz, p.zyx, b3);
    let f4 = select(p, p.wzyx, b4);

    // i32
    let si2 = select(i2, -i2, b2);
    let si3 = select(i3, -i3, i3 < vec3<i32>(0));
    let si4 = select(vec4<i32>(), i4, b4);

    // u32
    let su2 = select(u2, u2 + vec2<u32>(1u), b2);
    let su3 = select(u3, u3 * 2u, b3);
    let su4 = select(vec4<u32>(1u), u4, b4);

    // bool
    let sb2 = select(b2, !b2, vec2<bool>(true, false));
    let sb3 = select(b3, vec3<bool>(true), b3 == vec3<bool>(false));
    let sb4 = select(!b4, b4, b4);

    // Operands with side effects must be evaluated exactly once.
    let once = select(side_effect(1), side_effect(2), b3);

    // A scalar condition still uses the ternary operator.
    let scalar = select(i3, u3_as_i3(u3), p.x > 0.5);

    var acc = f2.x + f3.y + f4.z;
    acc += f32(si2.x + si3.y + si4.z + once.x + scalar.y);
    acc += f32(su2.x + su3.y + su4.z);
    acc += select(0.0, 1.0, sb2.x && sb3.y && sb4.z);
    return vec4<f32>(acc);
}

fn u3_as_i3(u: vec3<u32>) -> vec3<i32> {
    return vec3<i32>(u);
}
