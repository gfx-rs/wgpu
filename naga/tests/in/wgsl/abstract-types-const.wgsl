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

const xvupaiai: vec2<u32> = vec2(42, 43);
const xvfpaiai: vec2<f32> = vec2(44, 45);

const xvupuai: vec2<u32> = vec2(42u, 43);
const xvupaiu: vec2<u32> = vec2(42, 43u); 

const xvuuai: vec2<u32> = vec2<u32>(42u, 43);
const xvuaiu: vec2<u32> = vec2<u32>(42, 43u);

const xmfpaiaiaiai: mat2x2<f32> = mat2x2(1, 2, 3, 4);
const xmfpafaiaiai: mat2x2<f32> = mat2x2(1.0, 2, 3, 4);
const xmfpaiafaiai: mat2x2<f32> = mat2x2(1, 2.0, 3, 4);
const xmfpaiaiafai: mat2x2<f32> = mat2x2(1, 2, 3.0, 4);
const xmfpaiaiaiaf: mat2x2<f32> = mat2x2(1, 2, 3, 4.0);

const imfpaiaiaiai = mat2x2(1, 2, 3, 4);
const imfpafaiaiai = mat2x2(1.0, 2, 3, 4);
const imfpafafafaf = mat2x2(1.0, 2.0, 3.0, 4.0);

const ivispai = vec2(1);
const ivfspaf = vec2(1.0);
const ivis_ai = vec2<i32>(1);
const ivus_ai = vec2<u32>(1);
const ivfs_ai = vec2<f32>(1);
const ivfs_af = vec2<f32>(1.0);

const iafafaf = array<f32, 2>(1.0, 2.0);
const iafaiai = array<f32, 2>(1, 2);

const iafpafaf = array(1.0, 2.0);
const iafpaiaf = array(1, 2.0);
const iafpafai = array(1.0, 2);
const xafpafaf: array<f32, 2> = array(1.0, 2.0);

struct S {
    f: f32,
    i: i32,
    u: u32,
}

const s_f_i_u: S = S(1.0f, 1i, 1u);
const s_f_iai: S = S(1.0f, 1i, 1);
const s_fai_u: S = S(1.0f, 1,  1u);
const s_faiai: S = S(1.0f, 1,  1);
const saf_i_u: S = S(1.0,  1i, 1u);
const saf_iai: S = S(1.0,  1i, 1);
const safai_u: S = S(1.0,  1,  1u);
const safaiai: S = S(1.0,  1,  1);

// Vector construction with spreads
const ivfr_f__f = vec3<f32>(vec2<f32>(1.0f, 2.0f), 3.0f);
const ivfr_f_af = vec3<f32>(vec2<f32>(1.0f, 2.0f), 3.0 );
const ivfraf__f = vec3<f32>(vec2     (1.0 , 2.0 ), 3.0f);
const ivfraf_af = vec3<f32>(vec2     (1.0 , 2.0 ), 3.0 );

const ivf__fr_f = vec3<f32>(1.0f, vec2<f32>(2.0f, 3.0f));
const ivf__fraf = vec3<f32>(1.0f, vec2     (2.0 , 3.0 ));
const ivf_afr_f = vec3<f32>(1.0 , vec2<f32>(2.0f, 3.0f));
const ivf_afraf = vec3<f32>(1.0 , vec2     (2.0 , 3.0 ));

const ivfr_f_ai = vec3<f32>(vec2<f32>(1.0f, 2.0f), 3   );
const ivfrai__f = vec3<f32>(vec2     (1   , 2   ), 3.0f);
const ivfrai_ai = vec3<f32>(vec2     (1   , 2   ), 3   );

const ivf__frai = vec3<f32>(1.0f, vec2     (2   , 3   ));
const ivf_air_f = vec3<f32>(1   , vec2<f32>(2.0f, 3.0f));
const ivf_airai = vec3<f32>(1   , vec2     (2   , 3   ));
