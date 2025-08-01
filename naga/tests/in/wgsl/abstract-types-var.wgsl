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

var<private> xvipaiai: vec2<i32> = vec2(42, 43);
var<private> xvupaiai: vec2<u32> = vec2(44, 45);
var<private> xvfpaiai: vec2<f32> = vec2(46, 47);
var<private> xvfpafaf: vec2<f32> = vec2(48.0, 49.0);
var<private> xvfpaiaf: vec2<f32> = vec2(48, 49.0);

var<private> xvupuai: vec2<u32> = vec2(42u, 43);
var<private> xvupaiu: vec2<u32> = vec2(42, 43u); 

var<private> xvuuai: vec2<u32> = vec2<u32>(42u, 43);
var<private> xvuaiu: vec2<u32> = vec2<u32>(42, 43u);

var<private> xvip____: vec2<i32> = vec2();
var<private> xvup____: vec2<u32> = vec2();
var<private> xvfp____: vec2<f32> = vec2();
var<private> xmfp____: mat2x2f = mat2x2(vec2(), vec2());

var<private> xmfpaiaiaiai: mat2x2<f32> = mat2x2(1, 2, 3, 4);
var<private> xmfpafaiaiai: mat2x2<f32> = mat2x2(1.0, 2, 3, 4);
var<private> xmfpaiafaiai: mat2x2<f32> = mat2x2(1, 2.0, 3, 4);
var<private> xmfpaiaiafai: mat2x2<f32> = mat2x2(1, 2, 3.0, 4);
var<private> xmfpaiaiaiaf: mat2x2<f32> = mat2x2(1, 2, 3, 4.0);

var<private> xvispai: vec2<i32> = vec2(1);
var<private> xvfspaf: vec2<f32> = vec2(1.0);
var<private> xvis_ai: vec2<i32> = vec2<i32>(1);
var<private> xvus_ai: vec2<u32> = vec2<u32>(1);
var<private> xvfs_ai: vec2<f32> = vec2<f32>(1);
var<private> xvfs_af: vec2<f32> = vec2<f32>(1.0);

var<private> xafafaf: array<f32, 2> = array<f32, 2>(1.0, 2.0);
var<private> xafaiai: array<f32, 2> = array<f32, 2>(1, 2);

var<private> xaipaiai: array<i32, 2> = array(1,   2);
var<private> xaupaiai: array<u32, 2> = array(1,   2);
var<private> xafpaiaf: array<f32, 2> = array(1,   2.0);
var<private> xafpafai: array<f32, 2> = array(1.0, 2);
var<private> xafpafaf: array<f32, 2> = array(1.0, 2.0);

var<private> xavipai: array<vec3<i32>, 1> = array(vec3(1));
var<private> xavfpai: array<vec3<f32>, 1> = array(vec3(1));
var<private> xavfpaf: array<vec3<f32>, 1> = array(vec3(1.0));

// Construction with splats
var<private> xvisai: vec2<i32> = vec2(1);
var<private> xvusai: vec2<u32> = vec2(1);
var<private> xvfsai: vec2<f32> = vec2(1);
var<private> xvfsaf: vec2<f32> = vec2(1.0);

var<private> ivispai = vec2(1);
var<private> ivfspaf = vec2(1.0);
var<private> ivis_ai = vec2<i32>(1);
var<private> ivus_ai = vec2<u32>(1);
var<private> ivfs_ai = vec2<f32>(1);
var<private> ivfs_af = vec2<f32>(1.0);

var<private> iafafaf = array<f32, 2>(1.0, 2.0);
var<private> iafaiai = array<f32, 2>(1, 2);

var<private> iaipaiai = array(1,   2);
var<private> iafpafaf = array(1.0, 2.0);
var<private> iafpaiaf = array(1, 2.0);
var<private> iafpafai = array(1.0, 2);

var<private> iavipai = array(vec3(1));
var<private> iavfpai = array(vec3(1));
var<private> iavfpaf = array(vec3(1.0));

fn globals() {
    _ = xvipaiai;
    _ = xvupaiai;
    _ = xvfpaiai;
    _ = xvfpafaf;
    _ = xvfpaiaf;

    _ = xvupuai;
    _ = xvupaiu;

    _ = xvuuai;
    _ = xvuaiu;

    _ = xvip____;
    _ = xvup____;
    _ = xvfp____;
    _ = xmfp____;

    _ = xmfpaiaiaiai;
    _ = xmfpafaiaiai;
    _ = xmfpaiafaiai;
    _ = xmfpaiaiafai;
    _ = xmfpaiaiaiaf;

    _ = xvispai;
    _ = xvfspaf;
    _ = xvis_ai;
    _ = xvus_ai;
    _ = xvfs_ai;
    _ = xvfs_af;

    _ = xafafaf;
    _ = xafaiai;

    _ = xaipaiai;
    _ = xaupaiai;
    _ = xafpaiaf;
    _ = xafpafai;
    _ = xafpafaf;

    _ = xavipai;
    _ = xavfpai;
    _ = xavfpaf;

    // Construction with splats
    _ = xvisai;
    _ = xvusai;
    _ = xvfsai;
    _ = xvfsaf;

    _ = ivispai;
    _ = ivfspaf;
    _ = ivis_ai;
    _ = ivus_ai;
    _ = ivfs_ai;
    _ = ivfs_af;

    _ = iafafaf;
    _ = iafaiai;

    _ = iaipaiai;
    _ = iafpafaf;
    _ = iafpaiaf;
    _ = iafpafai;

    _ = iavipai;
    _ = iavfpai;
    _ = iavfpaf;
}

fn all_constant_arguments() {
    var xvipaiai: vec2<i32> = vec2(42, 43);
    var xvupaiai: vec2<u32> = vec2(44, 45);
    var xvfpaiai: vec2<f32> = vec2(46, 47);
    var xvfpafaf: vec2<f32> = vec2(48.0, 49.0);
    var xvfpaiaf: vec2<f32> = vec2(48, 49.0);

    var xvupuai: vec2<u32> = vec2(42u, 43);
    var xvupaiu: vec2<u32> = vec2(42, 43u);

    var xvuuai: vec2<u32> = vec2<u32>(42u, 43);
    var xvuaiu: vec2<u32> = vec2<u32>(42, 43u);

    var xvip____: vec2<i32> = vec2();
    var xvup____: vec2<u32> = vec2();
    var xvfp____: vec2<f32> = vec2();
    var xmfp____: mat2x2f = mat2x2(vec2(), vec2());

    var xmfpaiaiaiai: mat2x2<f32> = mat2x2(1, 2, 3, 4);
    var xmfpafaiaiai: mat2x2<f32> = mat2x2(1.0, 2, 3, 4);
    var xmfpaiafaiai: mat2x2<f32> = mat2x2(1, 2.0, 3, 4);
    var xmfpaiaiafai: mat2x2<f32> = mat2x2(1, 2, 3.0, 4);
    var xmfpaiaiaiaf: mat2x2<f32> = mat2x2(1, 2, 3, 4.0);

    var xmfp_faiaiai: mat2x2<f32> = mat2x2(1.0f, 2, 3, 4);
    var xmfpai_faiai: mat2x2<f32> = mat2x2(1, 2.0f, 3, 4);
    var xmfpaiai_fai: mat2x2<f32> = mat2x2(1, 2, 3.0f, 4);
    var xmfpaiaiai_f: mat2x2<f32> = mat2x2(1, 2, 3, 4.0f);

    var xvispai: vec2<i32> = vec2(1);
    var xvfspaf: vec2<f32> = vec2(1.0);
    var xvis_ai: vec2<i32> = vec2<i32>(1);
    var xvus_ai: vec2<u32> = vec2<u32>(1);
    var xvfs_ai: vec2<f32> = vec2<f32>(1);
    var xvfs_af: vec2<f32> = vec2<f32>(1.0);

    var xafafaf: array<f32, 2> = array<f32, 2>(1.0, 2.0);
    var xaf_faf: array<f32, 2> = array<f32, 2>(1.0f, 2.0);
    var xafaf_f: array<f32, 2> = array<f32, 2>(1.0, 2.0f);
    var xafaiai: array<f32, 2> = array<f32, 2>(1, 2);
    var xai_iai: array<i32, 2> = array<i32, 2>(1i, 2);
    var xaiai_i: array<i32, 2> = array<i32, 2>(1, 2i);

    var xaipaiai: array<i32, 2> = array(1,   2);
    var xafpaiai: array<f32, 2> = array(1,   2);
    var xafpaiaf: array<f32, 2> = array(1,   2.0);
    var xafpafai: array<f32, 2> = array(1.0, 2);
    var xafpafaf: array<f32, 2> = array(1.0, 2.0);

    var xavipai: array<vec3<i32>, 1> = array(vec3(1));
    var xavfpai: array<vec3<f32>, 1> = array(vec3(1));
    var xavfpaf: array<vec3<f32>, 1> = array(vec3(1.0));

    // Construction with splats
    var xvisai: vec2<i32> = vec2(1);
    var xvusai: vec2<u32> = vec2(1);
    var xvfsai: vec2<f32> = vec2(1);
    var xvfsaf: vec2<f32> = vec2(1.0);

    var iaipaiai = array(1,   2);
    var iafpaiaf = array(1,   2.0);
    var iafpafai = array(1.0, 2);
    var iafpafaf = array(1.0, 2.0);

    // Assignments to all of the above.
    xvipaiai = vec2(42, 43);
    xvupaiai = vec2(44, 45);
    xvfpaiai = vec2(46, 47);
    xvfpafaf = vec2(48.0, 49.0);
    xvfpaiaf = vec2(48, 49.0);

    xvupuai = vec2(42u, 43);
    xvupaiu = vec2(42, 43u);

    xvuuai = vec2<u32>(42u, 43);
    xvuaiu = vec2<u32>(42, 43u);

    xvip____ = vec2();
    xvup____ = vec2();
    xvfp____ = vec2();
    xmfp____ = mat2x2(vec2(), vec2());

    xmfpaiaiaiai = mat2x2(1, 2, 3, 4);
    xmfpafaiaiai = mat2x2(1.0, 2, 3, 4);
    xmfpaiafaiai = mat2x2(1, 2.0, 3, 4);
    xmfpaiaiafai = mat2x2(1, 2, 3.0, 4);
    xmfpaiaiaiaf = mat2x2(1, 2, 3, 4.0);

    xmfp_faiaiai = mat2x2(1.0f, 2, 3, 4);
    xmfpai_faiai = mat2x2(1, 2.0f, 3, 4);
    xmfpaiai_fai = mat2x2(1, 2, 3.0f, 4);
    xmfpaiaiai_f = mat2x2(1, 2, 3, 4.0f);

    xvispai = vec2(1);
    xvfspaf = vec2(1.0);
    xvis_ai = vec2<i32>(1);
    xvus_ai = vec2<u32>(1);
    xvfs_ai = vec2<f32>(1);
    xvfs_af = vec2<f32>(1.0);

    xafafaf = array<f32, 2>(1.0, 2.0);
    xaf_faf = array<f32, 2>(1.0f, 2.0);
    xafaf_f = array<f32, 2>(1.0, 2.0f);
    xafaiai = array<f32, 2>(1, 2);
    xai_iai = array<i32, 2>(1i, 2);
    xaiai_i = array<i32, 2>(1, 2i);

    xaipaiai = array(1,   2);
    xafpaiai = array(1,   2);
    xafpaiaf = array(1,   2.0);
    xafpafai = array(1.0, 2);
    xafpafaf = array(1.0, 2.0);

    xavipai = array(vec3(1));
    xavfpai = array(vec3(1));
    xavfpaf = array(vec3(1.0));

    // Construction with splats
    xvisai = vec2(1);
    xvusai = vec2(1);
    xvfsai = vec2(1);
    xvfsaf = vec2(1.0);

    iaipaiai = array(1,   2);
    iafpaiaf = array(1,   2.0);
    iafpafai = array(1.0, 2);
    iafpafaf = array(1.0, 2.0);
}

fn mixed_constant_and_runtime_arguments() {
    var u: u32;
    var i: i32;
    var f: f32;

    var xvupuai: vec2<u32> = vec2(u,  43);
    var xvupaiu: vec2<u32> = vec2(42, u);
    var xvfpfai: vec2<f32> = vec2(f, 47); // differs slightly from const version
    var xvfpfaf: vec2<f32> = vec2(f, 49.0);

    var xvuuai: vec2<u32> = vec2<u32>(u, 43);
    var xvuaiu: vec2<u32> = vec2<u32>(42, u);

    var xmfp_faiaiai: mat2x2<f32> = mat2x2(f, 2, 3, 4);
    var xmfpai_faiai: mat2x2<f32> = mat2x2(1, f, 3, 4);
    var xmfpaiai_fai: mat2x2<f32> = mat2x2(1, 2, f, 4);
    var xmfpaiaiai_f: mat2x2<f32> = mat2x2(1, 2, 3, f);

    var xaf_faf: array<f32, 2> = array<f32, 2>(f, 2.0);
    var xafaf_f: array<f32, 2> = array<f32, 2>(1.0, f);
    var xaf_fai: array<f32, 2> = array<f32, 2>(f, 2);
    var xafai_f: array<f32, 2> = array<f32, 2>(1, f);
    var xai_iai: array<i32, 2> = array<i32, 2>(i, 2);
    var xaiai_i: array<i32, 2> = array<i32, 2>(1, i);

    var xafp_faf: array<f32, 2> = array(f, 2.0);
    var xafpaf_f: array<f32, 2> = array(1.0, f);
    var xafp_fai: array<f32, 2> = array(f, 2);
    var xafpai_f: array<f32, 2> = array(1, f);
    var xaip_iai: array<i32, 2> = array(i, 2);
    var xaipai_i: array<i32, 2> = array(1, i);

    var xvisi: vec2<i32> = vec2(i);
    var xvusu: vec2<u32> = vec2(u);
    var xvfsf: vec2<f32> = vec2(f);

    // Assignments to all of the above.
    xvupuai = vec2(u,  43);
    xvupaiu = vec2(42, u);

    xvuuai = vec2<u32>(u, 43);
    xvuaiu = vec2<u32>(42, u);

    xmfp_faiaiai = mat2x2(f, 2, 3, 4);
    xmfpai_faiai = mat2x2(1, f, 3, 4);
    xmfpaiai_fai = mat2x2(1, 2, f, 4);
    xmfpaiaiai_f = mat2x2(1, 2, 3, f);

    xaf_faf = array<f32, 2>(f, 2.0);
    xafaf_f = array<f32, 2>(1.0, f);
    xaf_fai = array<f32, 2>(f, 2);
    xafai_f = array<f32, 2>(1, f);
    xai_iai = array<i32, 2>(i, 2);
    xaiai_i = array<i32, 2>(1, i);

    xafp_faf = array(f, 2.0);
    xafpaf_f = array(1.0, f);
    xafp_fai = array(f, 2);
    xafpai_f = array(1, f);
    xaip_iai = array(i, 2);
    xaipai_i = array(1, i);

    xvisi = vec2(i);
    xvusu = vec2(u);
    xvfsf = vec2(f);
}

@compute @workgroup_size(1)
fn main() {
    globals();
    all_constant_arguments();
    mixed_constant_and_runtime_arguments();
}
