@compute @workgroup_size(1, 1, 1) 
fn main() {
    var x0_: vec2<i32> = vec2<i32>(1i, 2i);
    var i1_: vec2<f32>;

    let _e12 = x0_.x;
    let _e14 = x0_.y;
    i1_ = select(vec2<f32>(1f, 0f), vec2<f32>(0f, 1f), (_e12 < _e14));
    return;
}
