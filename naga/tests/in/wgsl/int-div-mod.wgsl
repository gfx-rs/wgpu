// Integer `/` and `%` must follow WGSL semantics on targets where the
// degenerate cases are undefined (e.g. GLSL):
//
// - `x / 0 == x`, `x % 0 == 0`
// - `i32::MIN / -1 == i32::MIN`, `i32::MIN % -1 == 0`
// - `%` truncates, also for negative operands

struct Inputs {
    a: vec4<i32>,
    b: vec4<i32>,
    ua: vec4<u32>,
    ub: vec4<u32>,
}

@group(0) @binding(0)
var<uniform> inputs: Inputs;

fn signed_scalar(a: i32, b: i32) -> i32 {
    var x = a / b;
    x += a % b;
    x /= b;
    x %= b;
    return x;
}

fn unsigned_scalar(a: u32, b: u32) -> u32 {
    var x = a / b;
    x += a % b;
    x /= b;
    x %= b;
    return x;
}

fn signed_vectors(a: vec4<i32>, b: vec4<i32>) -> vec4<i32> {
    var x2 = a.xy / b.xy;
    x2 %= b.zw;
    var x3 = a.xyz % b.xyz;
    x3 /= b.yzw;
    var x4 = a / b + a % b;
    // vector-scalar and scalar-vector operands
    x4 += (a / b.x) % b.y;
    x4 += a.x / b + a.y % b;
    return x4 + vec4(x2, x3.xy);
}

fn unsigned_vectors(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    var x2 = a.xy / b.xy;
    x2 %= b.zw;
    var x3 = a.xyz % b.xyz;
    x3 /= b.yzw;
    var x4 = a / b + a % b;
    // vector-scalar and scalar-vector operands
    x4 += (a / b.x) % b.y;
    x4 += a.x / b + a.y % b;
    return x4 + vec4(x2, x3.xy);
}

@fragment
fn main() -> @location(0) vec4<i32> {
    let s = signed_scalar(inputs.a.x, inputs.b.x);
    let u = unsigned_scalar(inputs.ua.x, inputs.ub.x);
    let sv = signed_vectors(inputs.a, inputs.b);
    let uv = unsigned_vectors(inputs.ua, inputs.ub);
    return sv + vec4<i32>(uv) + vec4(s, i32(u), 0, 0);
}
