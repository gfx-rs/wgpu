requires unrestricted_pointer_parameters;

struct S {
    a: u32,
    b: array<u32, 4>,
}

@group(0) @binding(0) var<storage, read_write> buf: S;
@group(0) @binding(1) var<storage, read_write> arr: array<u32>;
@group(0) @binding(2) var<storage, read> lut: array<f32>;
@group(0) @binding(3) var t: texture_2d<f32>;
@group(0) @binding(4) var s: sampler;
var<workgroup> wg: array<u32, 64>;
var<private> private_arr: array<u32, 4>;

fn store(p: ptr<storage, u32, read_write>, v: u32) {
    *p = v;
}

fn load_add(p: ptr<storage, u32, read_write>, v: u32) -> u32 {
    let old = *p;
    *p = old + v;
    return old;
}

fn find(p: ptr<workgroup, array<u32, 64>>, needle: u32) -> i32 {
    for (var i = 0; i < 64; i++) {
        if p[i] == needle {
            return i;
        }
    }
    return -1;
}

fn classify(p: ptr<storage, u32, read_write>) -> u32 {
    switch *p {
        case 0u: {
            return 10u;
        }
        case 1u: {
            *p = 2u;
        }
        default: {
            return 30u;
        }
    }
    return 20u;
}

fn bump(p: ptr<function, u32>) {
    var count = 1u;
    *p += count;
    count += 1u;
}

fn bump_twice(p: ptr<function, u32>) {
    bump(p);
    bump(p);
}

fn set_private(p: ptr<private, u32>) {
    *p = 7u;
}

fn early(p: ptr<storage, u32, read_write>, c: bool) {
    if c {
        return;
    }
    *p = 5u;
}

fn not_inlined(p: ptr<function, u32>) -> u32 {
    return *p;
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) index: u32) {
    store(&buf.a, index);
    store(&arr[index], 1u);
    let x = load_add(&buf.b[1], 2u) + load_add(&buf.b[2], 3u);
    arr[0] = x;
    arr[1] = u32(find(&wg, index));
    arr[2] = classify(&arr[3]);
    var local = array<u32, 2>();
    for (var i = 0u; i < 2u; i++) {
        bump_twice(&local[i]);
    }
    set_private(&private_arr[1]);
    early(&arr[4], index == 0u);
    var whole = 3u;
    arr[5] = local[0] + private_arr[1] + not_inlined(&whole);
}

fn first_positive(p: ptr<storage, array<f32>, read>) -> f32 {
    for (var i = 0u; i < arrayLength(p); i++) {
        if p[i] > 0.0 {
            return p[i];
        }
    }
    return 0.0;
}

@fragment
fn frag(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let v = first_positive(&lut);
    return textureSample(t, s, uv) * v;
}
