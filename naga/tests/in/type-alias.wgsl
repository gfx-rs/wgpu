alias FVec3 = vec3<f32>;
alias IVec3 = vec3i;
alias Mat2 = mat2x2<f32>;
alias Mat3 = mat3x3f;
alias I32 = i32;
alias F32 = f32;

fn main() {
    let a = FVec3(0.0, 0.0, 0.0);
    let c = FVec3(0.0);
    let b = FVec3(vec2<f32>(0.0), 0.0);
    let d = FVec3(vec2<f32>(0.0), 0.0);
    let e = IVec3(d);

    let f = Mat2(1.0, 2.0, 3.0, 4.0);
    let g = Mat3(a, a, a);

    let h = vec2<I32>();
    let i = mat2x2<F32>();
}
