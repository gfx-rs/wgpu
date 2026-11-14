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

fn store(p: ptr<storage, u32, read_write>, v: u32) {
    (*p) = v;
    return;
}

fn load_add(p_1: ptr<storage, u32, read_write>, v_1: u32) -> u32 {
    let old = (*p_1);
    (*p_1) = (old + v_1);
    return old;
}

fn find(p_2: ptr<workgroup, array<u32, 64>>, needle: u32) -> i32 {
    var i_1: i32 = 0i;

    loop {
        let _e4 = i_1;
        if (_e4 < 64i) {
        } else {
            break;
        }
        {
            let _e7 = i_1;
            let _e9 = (*p_2)[_e7];
            if (_e9 == needle) {
                let _e11 = i_1;
                return _e11;
            }
        }
        continuing {
            let _e13 = i_1;
            i_1 = (_e13 + 1i);
        }
    }
    return -1i;
}

fn classify(p_3: ptr<storage, u32, read_write>) -> u32 {
    let _e1 = (*p_3);
    switch _e1 {
        case 0u: {
            return 10u;
        }
        case 1u: {
            (*p_3) = 2u;
        }
        default: {
            return 30u;
        }
    }
    return 20u;
}

fn bump(p_4: ptr<function, u32>) {
    var count: u32 = 1u;

    let _e3 = (*p_4);
    let _e4 = count;
    (*p_4) = (_e3 + _e4);
    let _e6 = count;
    count = (_e6 + 1u);
    return;
}

fn bump_twice(p_5: ptr<function, u32>) {
    bump(p_5);
    bump(p_5);
    return;
}

fn set_private(p_6: ptr<private, u32>) {
    (*p_6) = 7u;
    return;
}

fn early(p_7: ptr<storage, u32, read_write>, c: bool) {
    if c {
        return;
    }
    (*p_7) = 5u;
    return;
}

fn not_inlined(p_8: ptr<function, u32>) -> u32 {
    let _e1 = (*p_8);
    return _e1;
}

fn first_positive(p_9: ptr<storage, array<f32>>) -> f32 {
    var i_2: u32 = 0u;

    loop {
        let _e3 = i_2;
        if (_e3 < arrayLength(p_9)) {
        } else {
            break;
        }
        {
            let _e6 = i_2;
            let _e8 = (*p_9)[_e6];
            if (_e8 > 0f) {
                let _e11 = i_2;
                let _e13 = (*p_9)[_e11];
                return _e13;
            }
        }
        continuing {
            let _e15 = i_2;
            i_2 = (_e15 + 1u);
        }
    }
    return 0f;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_index) index: u32) {
    var local: array<u32, 2> = array<u32, 2>();
    var i: u32 = 0u;
    var whole: u32 = 3u;

    store((&buf.a), index);
    store((&arr[index]), 1u);
    let _e10 = load_add((&buf.b[1]), 2u);
    let _e15 = load_add((&buf.b[2]), 3u);
    let x = (_e10 + _e15);
    arr[0] = x;
    let _e22 = find((&wg), index);
    arr[1] = u32(_e22);
    let _e28 = classify((&arr[3]));
    arr[2] = _e28;
    loop {
        let _e33 = i;
        if (_e33 < 2u) {
        } else {
            break;
        }
        {
            let _e36 = i;
            bump_twice((&local[_e36]));
        }
        continuing {
            let _e39 = i;
            i = (_e39 + 1u);
        }
    }
    set_private((&private_arr[1]));
    early((&arr[4]), (index == 0u));
    let _e52 = local[0];
    let _e55 = private_arr[1];
    let _e57 = not_inlined((&whole));
    arr[5] = ((_e52 + _e55) + _e57);
    return;
}

@fragment
fn frag(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let _e2 = first_positive((&lut));
    let _e5 = textureSample(t, s, uv);
    return (_e5 * _e2);
}
