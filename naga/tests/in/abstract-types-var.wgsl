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

var<private> xvupuai: vec2<u32> = vec2(42u, 43);
var<private> xvupaiu: vec2<u32> = vec2(42, 43u); 

var<private> xvuuai: vec2<u32> = vec2<u32>(42u, 43);
var<private> xvuaiu: vec2<u32> = vec2<u32>(42, 43u);

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

var<private> xafpaiai: array<i32, 2> = array(1,   2);
var<private> xafpaiaf: array<f32, 2> = array(1,   2.0);
var<private> xafpafai: array<f32, 2> = array(1.0, 2);
var<private> xafpafaf: array<f32, 2> = array(1.0, 2.0);
