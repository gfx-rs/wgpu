@fragment 
fn main() {
    const v = vec4(0f);
    let a = degrees(1f);
    let b = radians(1f);
    let c = degrees(v);
    let d = radians(v);
    let e = saturate(v);
    let g = refract(v, v, 1f);
    const sign_b = vec4<i32>(-1i, -1i, -1i, -1i);
    const sign_d = vec4<f32>(-1f, -1f, -1f, -1f);
    let const_dot = dot(vec2<i32>(), vec2<i32>());
    const flb_b = vec2<i32>(-1i, -1i);
    const flb_c = vec2<u32>(0u, 0u);
    const ftb_c = vec2<i32>(0i, 0i);
    const ftb_d = vec2<u32>(0u, 0u);
    const ctz_e = vec2<u32>(32u, 32u);
    const ctz_f = vec2<i32>(32i, 32i);
    const ctz_g = vec2<u32>(0u, 0u);
    const ctz_h = vec2<i32>(0i, 0i);
    const clz_c = vec2<i32>(0i, 0i);
    const clz_d = vec2<u32>(31u, 31u);
    let lde_a = ldexp(1f, 2i);
    let lde_b = ldexp(vec2<f32>(1f, 2f), vec2<i32>(3i, 4i));
    let modf_a = modf(1.5f);
    let modf_b = modf(1.5f).fract;
    let modf_c = modf(1.5f).whole;
    let modf_d = modf(vec2<f32>(1.5f, 1.5f));
    let modf_e = modf(vec4<f32>(1.5f, 1.5f, 1.5f, 1.5f)).whole.x;
    let modf_f = modf(vec2<f32>(1.5f, 1.5f)).fract.y;
    let frexp_a = frexp(1.5f);
    let frexp_b = frexp(1.5f).fract;
    let frexp_c = frexp(1.5f).exp;
    let frexp_d = frexp(vec4<f32>(1.5f, 1.5f, 1.5f, 1.5f)).exp.x;
}
