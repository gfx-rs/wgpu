fn return_i32_ai() -> i32 {
    return 1i;
}

fn return_u32_ai() -> u32 {
    return 1u;
}

fn return_f32_ai() -> f32 {
    return 1f;
}

fn return_f32_af() -> f32 {
    return 1f;
}

fn return_vec2f32_ai() -> vec2<f32> {
    return vec2(1f);
}

fn return_arrf32_ai() -> array<f32, 4> {
    return array<f32, 4>(1f, 1f, 1f, 1f);
}

fn return_const_f32_const_ai() -> f32 {
    return 1f;
}

fn return_vec2f32_const_ai() -> vec2<f32> {
    return vec2(1f);
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let _e0 = return_i32_ai();
    let _e1 = return_u32_ai();
    let _e2 = return_f32_ai();
    let _e3 = return_f32_af();
    let _e4 = return_vec2f32_ai();
    let _e5 = return_arrf32_ai();
    let _e6 = return_const_f32_const_ai();
    let _e7 = return_vec2f32_const_ai();
    return;
}
