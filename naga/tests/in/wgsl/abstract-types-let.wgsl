// i/x: type inferred / explicit
// vX/mX/aX: vector / matrix / array of X
//     where X: u/i/f: u32 / i32 / f32
// s: vector splat
// r: vector spread (vector arg to vector constructor)
// p: "partial" constructor (type parameter inferred)
// u/i/f/ai/af: u32 / i32 / f32 / abstract float / abstract integer as parameter
// _: just for alignment

// Ensure that:
// - the inferred type is correct.
// - all parameters' types are considered.
// - all parameters are converted to the consensus type.

fn all_constant_arguments() {
    let xvipaiai: vec2<i32> = vec2(42, 43);
    let xvupaiai: vec2<u32> = vec2(44, 45);
    let xvfpaiai: vec2<f32> = vec2(46, 47);
    let xvfpafaf: vec2<f32> = vec2(48.0, 49.0);
    let xvfpaiaf: vec2<f32> = vec2(48, 49.0);

    let xvupuai: vec2<u32> = vec2(42u, 43);
    let xvupaiu: vec2<u32> = vec2(42, 43u);

    let xvuuai: vec2<u32> = vec2<u32>(42u, 43);
    let xvuaiu: vec2<u32> = vec2<u32>(42, 43u);

    let xvip____: vec2<i32> = vec2();
    let xvup____: vec2<u32> = vec2();
    let xvfp____: vec2<f32> = vec2();
    let xmfp____: mat2x2f = mat2x2(vec2(), vec2());

    let xmfpaiaiaiai: mat2x2<f32> = mat2x2(1, 2, 3, 4);
    let xmfpafaiaiai: mat2x2<f32> = mat2x2(1.0, 2, 3, 4);
    let xmfpaiafaiai: mat2x2<f32> = mat2x2(1, 2.0, 3, 4);
    let xmfpaiaiafai: mat2x2<f32> = mat2x2(1, 2, 3.0, 4);
    let xmfpaiaiaiaf: mat2x2<f32> = mat2x2(1, 2, 3, 4.0);

    let xmfp_faiaiai: mat2x2<f32> = mat2x2(1.0f, 2, 3, 4);
    let xmfpai_faiai: mat2x2<f32> = mat2x2(1, 2.0f, 3, 4);
    let xmfpaiai_fai: mat2x2<f32> = mat2x2(1, 2, 3.0f, 4);
    let xmfpaiaiai_f: mat2x2<f32> = mat2x2(1, 2, 3, 4.0f);

    let xvispai: vec2<i32> = vec2(1);
    let xvfspaf: vec2<f32> = vec2(1.0);
    let xvis_ai: vec2<i32> = vec2<i32>(1);
    let xvus_ai: vec2<u32> = vec2<u32>(1);
    let xvfs_ai: vec2<f32> = vec2<f32>(1);
    let xvfs_af: vec2<f32> = vec2<f32>(1.0);

    let xafafaf: array<f32, 2> = array<f32, 2>(1.0, 2.0);
    let xaf_faf: array<f32, 2> = array<f32, 2>(1.0f, 2.0);
    let xafaf_f: array<f32, 2> = array<f32, 2>(1.0, 2.0f);
    let xafaiai: array<f32, 2> = array<f32, 2>(1, 2);
    let xai_iai: array<i32, 2> = array<i32, 2>(1i, 2);
    let xaiai_i: array<i32, 2> = array<i32, 2>(1, 2i);

    let xaipaiai: array<i32, 2> = array(1,   2);
    let xafpaiai: array<f32, 2> = array(1,   2);
    let xafpaiaf: array<f32, 2> = array(1,   2.0);
    let xafpafai: array<f32, 2> = array(1.0, 2);
    let xafpafaf: array<f32, 2> = array(1.0, 2.0);

    let xavipai: array<vec3<i32>, 1> = array(vec3(1));
    let xavfpai: array<vec3<f32>, 1> = array(vec3(1));
    let xavfpaf: array<vec3<f32>, 1> = array(vec3(1.0));

    // Construction with splats
    let xvisai: vec2<i32> = vec2(1);
    let xvusai: vec2<u32> = vec2(1);
    let xvfsai: vec2<f32> = vec2(1);
    let xvfsaf: vec2<f32> = vec2(1.0);

    let iaipaiai = array(1,   2);
    let iafpaiaf = array(1,   2.0);
    let iafpafai = array(1.0, 2);
    let iafpafaf = array(1.0, 2.0);
}

fn mixed_constant_and_runtime_arguments() {
    var u: u32;
    var i: i32;
    var f: f32;

    let xvupuai: vec2<u32> = vec2(u,  43);
    let xvupaiu: vec2<u32> = vec2(42, u);
    let xvfpfai: vec2<f32> = vec2(f, 47); // differs slightly from const version
    let xvfpfaf: vec2<f32> = vec2(f, 49.0);

    let xvuuai: vec2<u32> = vec2<u32>(u, 43);
    let xvuaiu: vec2<u32> = vec2<u32>(42, u);

    let xmfp_faiaiai: mat2x2<f32> = mat2x2(f, 2, 3, 4);
    let xmfpai_faiai: mat2x2<f32> = mat2x2(1, f, 3, 4);
    let xmfpaiai_fai: mat2x2<f32> = mat2x2(1, 2, f, 4);
    let xmfpaiaiai_f: mat2x2<f32> = mat2x2(1, 2, 3, f);

    let xaf_faf: array<f32, 2> = array<f32, 2>(f, 2.0);
    let xafaf_f: array<f32, 2> = array<f32, 2>(1.0, f);
    let xaf_fai: array<f32, 2> = array<f32, 2>(f, 2);
    let xafai_f: array<f32, 2> = array<f32, 2>(1, f);
    let xai_iai: array<i32, 2> = array<i32, 2>(i, 2);
    let xaiai_i: array<i32, 2> = array<i32, 2>(1, i);

    let xafp_faf: array<f32, 2> = array(f, 2.0);
    let xafpaf_f: array<f32, 2> = array(1.0, f);
    let xafp_fai: array<f32, 2> = array(f, 2);
    let xafpai_f: array<f32, 2> = array(1, f);
    let xaip_iai: array<i32, 2> = array(i, 2);
    let xaipai_i: array<i32, 2> = array(1, i);

    let xvisi: vec2<i32> = vec2(i);
    let xvusu: vec2<u32> = vec2(u);
    let xvfsf: vec2<f32> = vec2(f);
}

@compute @workgroup_size(1)
fn main() {
    all_constant_arguments();
    mixed_constant_and_runtime_arguments();
}
