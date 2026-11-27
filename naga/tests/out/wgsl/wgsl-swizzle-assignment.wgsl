var<private> v_1: array<vec4<u32>, 1>;
@group(0) @binding(0)
var<storage, read_write> buf: array<vec3<f32>>;
var<workgroup> w: mat2x4<f32>;

fn update_and_index() -> u32 {
    let _e2 = v_1[0];
    v_1[0] = (_e2 + vec4<u32>(2u, 4u, 8u, 10u));
    return 0u;
}

fn update_and_rhs() -> vec2<u32> {
    v_1[0].x = 900u;
    v_1[0].z = 100u;
    return vec2<u32>(55u, 66u);
}

fn eval_order() -> vec4<u32> {
    v_1[0] = vec4<u32>(0u, 0u, 0u, 0u);
    let _e8 = update_and_index();
    let _e10 = update_and_rhs();
    let _e11 = v_1[_e8];
    v_1[_e8] = vec4<u32>(_e10.y, _e10.x, _e11.z, _e11.w);
    let assigned = v_1[0];
    v_1[0] = vec4<u32>(0u, 0u, 0u, 0u);
    let _e28 = update_and_index();
    let _e30 = v_1[_e28];
    let _e32 = update_and_rhs();
    let _e33 = (_e30.yx + _e32);
    let _e34 = v_1[_e28];
    v_1[_e28] = vec4<u32>(_e33.y, _e33.x, _e34.z, _e34.w);
    let _e42 = v_1[0];
    return (assigned + _e42);
}

fn swizzle_views(i: i32) -> vec4<u32> {
    var v: vec4<u32>;

    let _e4 = v;
    v = vec4<u32>(_e4.x, 1u, 2u, _e4.w);
    let _e8 = v;
    let u = _e8.wyz;
    v = vec4<u32>(1u, 2u, 3u, 4u);
    v = vec4<u32>(4u, 3u, 2u, 1u);
    let _e22 = v;
    v = vec4<u32>(_e22.x, 100u, 99u, _e22.w);
    v.z = 99u;
    let _e30 = v.z;
    v.z = (_e30 + 1u);
    v.y = 50u;
    v[select(3i, 0i, (i == 1i))] = 7u;
    v[i] = 8u;
    let _e43 = v;
    let _e46 = (_e43.xz * 3u);
    let _e47 = v;
    v = vec4<u32>(_e46.x, _e47.y, _e46.y, _e47.w);
    let _e53 = v;
    let _e58 = (_e53.yw << vec2<u32>(1u, 2u));
    let _e59 = v;
    v = vec4<u32>(_e59.x, _e58.x, _e59.z, _e58.y);
    let _e65 = v;
    return (_e65 + u.xyzx);
}

fn via_pointer(p: ptr<function, vec4<u32>>) {
    let _e3 = (*p);
    (*p) = vec4<u32>(_e3.x, 1u, 2u, _e3.w);
    let _e7 = (*p);
    let u_1 = _e7.wyz;
    let _e11 = (*p);
    (*p) = vec4<u32>(_e11.x, 100u, 99u, _e11.w);
    (*p).y = 50u;
    let _e17 = u_1.xy;
    let _e18 = (*p);
    (*p) = vec4<u32>(_e17.x, _e18.y, _e18.z, _e17.y);
    return;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(local_invocation_index) index: u32) {
    var a: vec4<u32> = vec4<u32>();

    via_pointer((&a));
    let _e7 = buf[index];
    buf[index] = vec3<f32>(2f, _e7.y, 1f);
    let _e12 = buf[index];
    let _e16 = (_e12.yz - vec2(1f));
    let _e17 = buf[index];
    buf[index] = vec3<f32>(_e17.x, _e16.x, _e16.y);
    let _e26 = w[1];
    w[1] = vec4<f32>(_e26.x, 4f, _e26.z, 3f);
    let _e30 = eval_order();
    let _e32 = swizzle_views(1i);
    let _e34 = a;
    let r = ((_e30 + _e32) + _e34);
    let _e42 = w[1];
    buf[0] = (vec3<f32>(r.xyz) + _e42.xyz);
    return;
}
