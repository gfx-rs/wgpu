type FVec3 = vec3<f32>;
type IVec3 = vec3<i32>;
type Mat2 = mat2x2<f32>;

fn main() {
    let a = FVec3(0.0, 0.0, 0.0);
    let c = FVec3(0.0);
    let b = FVec3(vec2<f32>(0.0), 0.0);
    let d = FVec3(vec2<f32>(0.0), 0.0);
    let e = IVec3(d);

    let f = Mat2(1.0, 2.0, 3.0, 4.0);
}
