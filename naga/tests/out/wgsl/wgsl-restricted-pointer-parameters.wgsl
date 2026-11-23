struct S {
    a: u32,
    b: array<u32, 4>,
}

@group(0) @binding(0)
var<storage, read_write> buf: S;
@group(0) @binding(1)
var<storage, read_write> arr: array<u32>;
@group(0) @binding(2)
var<storage> lut: array<f32>;
@group(0) @binding(3)
var t: texture_2d<f32>;
@group(0) @binding(4)
var s: sampler;
var<workgroup> wg: array<u32, 64>;
var<private> private_arr: array<u32, 4>;

fn bump(p: ptr<function, u32>) {
    var count_2: u32 = 1u;

    let _e3 = (*p);
    let _e4 = count_2;
    (*p) = (_e3 + _e4);
    let _e6 = count_2;
    count_2 = (_e6 + 1u);
    return;
}

fn bump_twice(p_1: ptr<function, u32>) {
    bump(p_1);
    bump(p_1);
    return;
}

fn set_private(p_2: ptr<private, u32>) {
    (*p_2) = 7u;
    return;
}

fn not_inlined(p_3: ptr<function, u32>) -> u32 {
    let _e1 = (*p_3);
    return _e1;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(local_invocation_index) index: u32) {
    var local: array<u32, 2> = array<u32, 2>();
    var i: u32 = 0u;
    var whole: u32 = 3u;
    var v: u32;
    var v_1: u32;
    var v_2: u32;
    var local_1: u32;
    var v_3: u32;
    var local_2: u32;
    var needle: u32;
    var i_1: i32;
    var local_3: i32;
    var local_4: bool;
    var local_5: u32;
    var local_6: bool;
    var count: u32;
    var count_1: u32;
    var c: bool;

    v = index;
    let _e3 = v;
    buf.a = _e3;
    v_1 = 1u;
    let _e8 = v_1;
    arr[index] = _e8;
    v_2 = 2u;
    let _e13 = v_2;
    let _e14 = buf.b[1];
    buf.b[1] = (_e14 + _e13);
    local_1 = _e14;
    let _e17 = local_1;
    v_3 = 3u;
    let _e22 = v_3;
    let _e23 = buf.b[2];
    buf.b[2] = (_e23 + _e22);
    local_2 = _e23;
    let _e26 = local_2;
    let x = (_e17 + _e26);
    arr[0] = x;
    needle = index;
    let _e33 = needle;
    switch 0i {
        default: {
            i_1 = 0i;
            local_4 = false;
            loop {
                let _e35 = i_1;
                if (_e35 < 64i) {
                } else {
                    break;
                }
                {
                    let _e39 = i_1;
                    let _e41 = wg[_e39];
                    if (_e41 == _e33) {
                        let _e43 = i_1;
                        local_3 = _e43;
                        local_4 = true;
                        break;
                    }
                }
                continuing {
                    let _e45 = i_1;
                    i_1 = (_e45 + 1i);
                }
            }
            let _e48 = local_4;
            if _e48 {
                break;
            }
            local_3 = -1i;
        }
    }
    let _e50 = local_3;
    arr[1] = u32(_e50);
    switch 0i {
        default: {
            local_6 = false;
            let _e56 = arr[3];
            switch _e56 {
                case 0u: {
                    local_5 = 10u;
                    local_6 = true;
                    break;
                }
                case 1u: {
                    arr[3] = 2u;
                }
                default: {
                    local_5 = 30u;
                    local_6 = true;
                    break;
                }
            }
            let _e58 = local_6;
            if _e58 {
                break;
            }
            local_5 = 20u;
        }
    }
    let _e60 = local_5;
    arr[2] = _e60;
    loop {
        let _e62 = i;
        if (_e62 < 2u) {
        } else {
            break;
        }
        {
            let _e66 = i;
            count = 1u;
            let _e69 = local[_e66];
            let _e70 = count;
            local[_e66] = (_e69 + _e70);
            let _e72 = count;
            count = (_e72 + 1u);
            count_1 = 1u;
            let _e76 = local[_e66];
            let _e77 = count_1;
            local[_e66] = (_e76 + _e77);
            let _e79 = count_1;
            count_1 = (_e79 + 1u);
        }
        continuing {
            let _e83 = i;
            i = (_e83 + 1u);
        }
    }
    private_arr[1] = 7u;
    c = (index == 0u);
    let _e92 = c;
    switch 0i {
        default: {
            if _e92 {
                break;
            }
            arr[4] = 5u;
        }
    }
    let _e96 = local[0];
    let _e99 = private_arr[1];
    let _e101 = not_inlined((&whole));
    arr[5] = ((_e96 + _e99) + _e101);
    return;
}

@fragment
fn frag(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var i_2: u32;
    var local_7: f32;
    var local_8: bool;

    switch 0i {
        default: {
            i_2 = 0u;
            local_8 = false;
            loop {
                let _e2 = i_2;
                if (_e2 < arrayLength((&lut))) {
                } else {
                    break;
                }
                {
                    let _e5 = i_2;
                    let _e7 = lut[_e5];
                    if (_e7 > 0f) {
                        let _e10 = i_2;
                        let _e12 = lut[_e10];
                        local_7 = _e12;
                        local_8 = true;
                        break;
                    }
                }
                continuing {
                    let _e14 = i_2;
                    i_2 = (_e14 + 1u);
                }
            }
            let _e17 = local_8;
            if _e17 {
                break;
            }
            local_7 = 0f;
        }
    }
    let v_4 = local_7;
    let _e23 = textureSample(t, s, uv);
    return (_e23 * v_4);
}
