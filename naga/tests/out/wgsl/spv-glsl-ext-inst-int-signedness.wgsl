struct type_8 {
    member: array<u32>,
}

@group(0) @binding(0)
var<storage, read_write> global: type_8;
@group(0) @binding(1)
var global_1: texture_2d<f32>;
var<private> global_2: vec3<u32>;

fn function_() {
    var local: array<u32, 4> = array<u32, 4>(1u, 2u, 3u, 4u);

    let _e12 = global_2;
    let _e16 = (bitcast<i32>(_e12.x) - 2i);
    let _e18 = textureNumLevels(global_1);
    let _e24 = i32(min(u32(_e16), u32((i32(_e18) - 1i))));
    let _e25 = textureDimensions(global_1, _e24);
    let _e32 = textureLoad(global_1, vec2<i32>(min(vec2<u32>(vec2(_e16)), vec2<u32>((vec2<i32>(_e25) - vec2<i32>(1i, 1i))))), _e24);
    let _e36 = local[min(u32(_e16), 3u)];
    global.member[min(u32(_e16), (arrayLength((&global.member)) - 1u))] = (_e36 + u32(clamp(_e32.x, 0f, 4294967000f)));
    return;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) param: vec3<u32>) {
    global_2 = param;
    function_();
}
