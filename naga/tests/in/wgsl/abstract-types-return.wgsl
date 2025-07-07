fn return_i32_ai() -> i32 {
    return 1;
}

fn return_u32_ai() -> u32 {
    return 1;
}

fn return_f32_ai() -> f32 {
    return 1;
}

fn return_f32_af() -> f32 {
    return 1.0;
}

fn return_vec2f32_ai() -> vec2<f32> {
    return vec2(1);
}

fn return_arrf32_ai() -> array<f32, 4> {
    return array(1, 1, 1, 1);
}

const one = 1;
fn return_const_f32_const_ai() -> f32 {
    return one;
}

fn return_vec2f32_const_ai() -> vec2<f32> {
    const vec_one = vec2(1);
    return vec_one;
}

@compute @workgroup_size(1)
fn main() {
    return_i32_ai();
    return_u32_ai();
    return_f32_ai();
    return_f32_af();
    return_vec2f32_ai();
    return_arrf32_ai();
    return_const_f32_const_ai();
    return_vec2f32_const_ai();
}
