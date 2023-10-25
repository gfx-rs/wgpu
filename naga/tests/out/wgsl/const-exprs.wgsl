const TWO: u32 = 2u;
const THREE: i32 = 3;
const FOUR: i32 = 4;
const FOUR_ALIAS: i32 = 4;
const TEST_CONSTANT_ADDITION: i32 = 8;
const TEST_CONSTANT_ALIAS_ADDITION: i32 = 8;
const PI: f32 = 3.141;
const phi_sun: f32 = 6.282;
const DIV: vec4<f32> = vec4<f32>(0.44444445, 0.0, 0.0, 0.0);
const TEXTURE_KIND_REGULAR: i32 = 0;
const TEXTURE_KIND_WARP: i32 = 1;
const TEXTURE_KIND_SKY: i32 = 2;

fn swizzle_of_compose() {
    var out: vec4<i32> = vec4<i32>(4, 3, 2, 1);

}

fn index_of_compose() {
    var out_1: i32 = 2;

}

fn compose_three_deep() {
    var out_2: i32 = 6;

}

fn non_constant_initializers() {
    var w: i32 = 30;
    var x: i32;
    var y: i32;
    var z: i32 = 70;
    var out_3: vec4<i32>;

    let _e2 = w;
    x = _e2;
    let _e4 = x;
    y = _e4;
    let _e8 = w;
    let _e9 = x;
    let _e10 = y;
    let _e11 = z;
    out_3 = vec4<i32>(_e8, _e9, _e10, _e11);
    return;
}

fn splat_of_constant() {
    var out_4: vec4<i32> = vec4<i32>(-4, -4, -4, -4);

}

fn compose_of_constant() {
    var out_5: vec4<i32> = vec4<i32>(-4, -4, -4, -4);

}

fn map_texture_kind(texture_kind: i32) -> u32 {
    switch texture_kind {
        case 0: {
            return 10u;
        }
        case 1: {
            return 20u;
        }
        case 2: {
            return 30u;
        }
        default: {
            return 0u;
        }
    }
}

@compute @workgroup_size(2, 3, 1) 
fn main() {
    swizzle_of_compose();
    index_of_compose();
    compose_three_deep();
    non_constant_initializers();
    splat_of_constant();
    compose_of_constant();
    return;
}
