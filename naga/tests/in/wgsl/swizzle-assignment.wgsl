// Tests for the `swizzle_assignment` language extension.
// <https://www.w3.org/TR/WGSL/#language_extension-swizzle_assignment>

var<private> v: array<vec4u, 1>;

fn update_and_index() -> u32 {
    v[0] += vec4u(2, 4, 8, 10);
    return 0;
}

fn update_and_rhs() -> vec2u {
    v[0].x = 900;
    v[0].z = 100;
    return vec2u(55, 66);
}

// The examples of evaluation order from the specification.
fn eval_order() -> vec4u {
    v[0] = vec4u(0, 0, 0, 0);
    v[update_and_index()].yx = update_and_rhs();
    let assigned = v[0]; // vec4u(66, 55, 100, 10)

    v[0] = vec4u(0, 0, 0, 0);
    v[update_and_index()].yx += update_and_rhs();
    return assigned + v[0]; // vec4u(68, 59, 100, 10)
}

fn swizzle_views(i: i32) -> vec4u {
    var v: vec4u;
    v.yz = vec2u(1, 2);
    let u = v.agb;
    // Assigning every component only reorders the right-hand side.
    v.xyzw = vec4(1, 2, 3, 4);
    v.wzyx = vec4(1, 2, 3, 4);
    // Chained swizzles.
    v.yz.yx = vec2(99, 100);
    v.yz.y = 99;
    v.zy.x++;
    // Indexing a swizzle view.
    v.zy[1] = 50;
    v.wx[i] = 7u;
    v.xy[i] = 8u;
    // Compound assignments.
    v.xz *= 3;
    v.yw <<= vec2(1u, 2u);
    return v + u.xyzx;
}

fn via_pointer(p: ptr<function, vec4u>) {
    p.yz = vec2u(1, 2);
    let u = p.agb;
    p.yz.yx = vec2(99, 100);
    p.zy[1] = 50;
    (*p).xw = u.xy;
}

@group(0) @binding(0) var<storage, read_write> buf: array<vec3f>;
var<workgroup> w: mat2x4f;

@compute @workgroup_size(1)
fn main(@builtin(local_invocation_index) index: u32) {
    var a = vec4u();
    via_pointer(&a);
    buf[index].zx = vec2(1.0, 2.0);
    buf[index].yz -= vec2f(1.0);
    w[1].wy = vec2(3.0, 4.0);
    let r = eval_order() + swizzle_views(1) + a;
    buf[0] = vec3f(r.xyz) + w[1].xyz;
}
