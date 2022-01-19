@group(1) @binding(0) 
var tex1D: texture_1d<f32>;
@group(1) @binding(1) 
var tex1DArray: texture_1d_array<f32>;
@group(1) @binding(2) 
var tex2D: texture_2d<f32>;
@group(1) @binding(3) 
var tex2DArray: texture_2d_array<f32>;
@group(1) @binding(4) 
var texCube: texture_cube<f32>;
@group(1) @binding(5) 
var texCubeArray: texture_cube_array<f32>;
@group(1) @binding(6) 
var tex3D: texture_3d<f32>;
@group(1) @binding(7) 
var samp: sampler;
@group(1) @binding(12) 
var tex2DShadow: texture_depth_2d;
@group(1) @binding(13) 
var tex2DArrayShadow: texture_depth_2d_array;
@group(1) @binding(14) 
var texCubeShadow: texture_depth_cube;
@group(1) @binding(15) 
var texCubeArrayShadow: texture_depth_cube_array;
@group(1) @binding(16) 
var tex3DShadow: texture_3d<f32>;
@group(1) @binding(17) 
var sampShadow: sampler_comparison;
var<private> texcoord_1: vec4<f32>;

fn testTex1D(coord: f32) {
    var coord_1: f32;
    var c: vec4<f32>;

    coord_1 = coord;
    let _e18 = coord_1;
    let _e19 = textureSample(tex1D, samp, _e18);
    c = _e19;
    let _e22 = coord_1;
    let _e24 = textureSampleBias(tex1D, samp, _e22, 2.0);
    c = _e24;
    let _e28 = coord_1;
    let _e31 = textureSampleGrad(tex1D, samp, _e28, 4.0, 4.0);
    c = _e31;
    let _e36 = coord_1;
    let _e40 = textureSampleGrad(tex1D, samp, _e36, 4.0, 4.0, 5);
    c = _e40;
    let _e43 = coord_1;
    let _e45 = textureSampleLevel(tex1D, samp, _e43, 3.0);
    c = _e45;
    let _e49 = coord_1;
    let _e52 = textureSampleLevel(tex1D, samp, _e49, 3.0, 5);
    c = _e52;
    let _e55 = coord_1;
    let _e57 = textureSample(tex1D, samp, _e55, 5);
    c = _e57;
    let _e61 = coord_1;
    let _e64 = textureSampleBias(tex1D, samp, _e61, 2.0, 5);
    c = _e64;
    let _e65 = coord_1;
    let _e68 = coord_1;
    let _e70 = vec2<f32>(_e68, 6.0);
    let _e74 = textureSample(tex1D, samp, (_e70.x / _e70.y));
    c = _e74;
    let _e75 = coord_1;
    let _e80 = coord_1;
    let _e84 = vec4<f32>(_e80, 0.0, 0.0, 6.0);
    let _e90 = textureSample(tex1D, samp, (_e84.xyz / vec3<f32>(_e84.w)).x);
    c = _e90;
    let _e91 = coord_1;
    let _e95 = coord_1;
    let _e97 = vec2<f32>(_e95, 6.0);
    let _e102 = textureSampleBias(tex1D, samp, (_e97.x / _e97.y), 2.0);
    c = _e102;
    let _e103 = coord_1;
    let _e109 = coord_1;
    let _e113 = vec4<f32>(_e109, 0.0, 0.0, 6.0);
    let _e120 = textureSampleBias(tex1D, samp, (_e113.xyz / vec3<f32>(_e113.w)).x, 2.0);
    c = _e120;
    let _e121 = coord_1;
    let _e126 = coord_1;
    let _e128 = vec2<f32>(_e126, 6.0);
    let _e134 = textureSampleGrad(tex1D, samp, (_e128.x / _e128.y), 4.0, 4.0);
    c = _e134;
    let _e135 = coord_1;
    let _e142 = coord_1;
    let _e146 = vec4<f32>(_e142, 0.0, 0.0, 6.0);
    let _e154 = textureSampleGrad(tex1D, samp, (_e146.xyz / vec3<f32>(_e146.w)).x, 4.0, 4.0);
    c = _e154;
    let _e155 = coord_1;
    let _e161 = coord_1;
    let _e163 = vec2<f32>(_e161, 6.0);
    let _e170 = textureSampleGrad(tex1D, samp, (_e163.x / _e163.y), 4.0, 4.0, 5);
    c = _e170;
    let _e171 = coord_1;
    let _e179 = coord_1;
    let _e183 = vec4<f32>(_e179, 0.0, 0.0, 6.0);
    let _e192 = textureSampleGrad(tex1D, samp, (_e183.xyz / vec3<f32>(_e183.w)).x, 4.0, 4.0, 5);
    c = _e192;
    let _e193 = coord_1;
    let _e197 = coord_1;
    let _e199 = vec2<f32>(_e197, 6.0);
    let _e204 = textureSampleLevel(tex1D, samp, (_e199.x / _e199.y), 3.0);
    c = _e204;
    let _e205 = coord_1;
    let _e211 = coord_1;
    let _e215 = vec4<f32>(_e211, 0.0, 0.0, 6.0);
    let _e222 = textureSampleLevel(tex1D, samp, (_e215.xyz / vec3<f32>(_e215.w)).x, 3.0);
    c = _e222;
    let _e223 = coord_1;
    let _e228 = coord_1;
    let _e230 = vec2<f32>(_e228, 6.0);
    let _e236 = textureSampleLevel(tex1D, samp, (_e230.x / _e230.y), 3.0, 5);
    c = _e236;
    let _e237 = coord_1;
    let _e244 = coord_1;
    let _e248 = vec4<f32>(_e244, 0.0, 0.0, 6.0);
    let _e256 = textureSampleLevel(tex1D, samp, (_e248.xyz / vec3<f32>(_e248.w)).x, 3.0, 5);
    c = _e256;
    let _e257 = coord_1;
    let _e261 = coord_1;
    let _e263 = vec2<f32>(_e261, 6.0);
    let _e268 = textureSample(tex1D, samp, (_e263.x / _e263.y), 5);
    c = _e268;
    let _e269 = coord_1;
    let _e275 = coord_1;
    let _e279 = vec4<f32>(_e275, 0.0, 0.0, 6.0);
    let _e286 = textureSample(tex1D, samp, (_e279.xyz / vec3<f32>(_e279.w)).x, 5);
    c = _e286;
    let _e287 = coord_1;
    let _e292 = coord_1;
    let _e294 = vec2<f32>(_e292, 6.0);
    let _e300 = textureSampleBias(tex1D, samp, (_e294.x / _e294.y), 2.0, 5);
    c = _e300;
    let _e301 = coord_1;
    let _e308 = coord_1;
    let _e312 = vec4<f32>(_e308, 0.0, 0.0, 6.0);
    let _e320 = textureSampleBias(tex1D, samp, (_e312.xyz / vec3<f32>(_e312.w)).x, 2.0, 5);
    c = _e320;
    return;
}

fn testTex1DArray(coord_2: vec2<f32>) {
    var coord_3: vec2<f32>;
    var c_1: vec4<f32>;

    coord_3 = coord_2;
    let _e18 = coord_3;
    let _e22 = textureSample(tex1DArray, samp, _e18.x, i32(_e18.y));
    c_1 = _e22;
    let _e25 = coord_3;
    let _e30 = textureSampleBias(tex1DArray, samp, _e25.x, i32(_e25.y), 2.0);
    c_1 = _e30;
    let _e34 = coord_3;
    let _e40 = textureSampleGrad(tex1DArray, samp, _e34.x, i32(_e34.y), 4.0, 4.0);
    c_1 = _e40;
    let _e45 = coord_3;
    let _e52 = textureSampleGrad(tex1DArray, samp, _e45.x, i32(_e45.y), 4.0, 4.0, 5);
    c_1 = _e52;
    let _e55 = coord_3;
    let _e60 = textureSampleLevel(tex1DArray, samp, _e55.x, i32(_e55.y), 3.0);
    c_1 = _e60;
    let _e64 = coord_3;
    let _e70 = textureSampleLevel(tex1DArray, samp, _e64.x, i32(_e64.y), 3.0, 5);
    c_1 = _e70;
    let _e73 = coord_3;
    let _e78 = textureSample(tex1DArray, samp, _e73.x, i32(_e73.y), 5);
    c_1 = _e78;
    let _e82 = coord_3;
    let _e88 = textureSampleBias(tex1DArray, samp, _e82.x, i32(_e82.y), 2.0, 5);
    c_1 = _e88;
    return;
}

fn testTex2D(coord_4: vec2<f32>) {
    var coord_5: vec2<f32>;
    var c_2: vec4<f32>;

    coord_5 = coord_4;
    let _e18 = coord_5;
    let _e19 = textureSample(tex2D, samp, _e18);
    c_2 = _e19;
    let _e22 = coord_5;
    let _e24 = textureSampleBias(tex2D, samp, _e22, 2.0);
    c_2 = _e24;
    let _e30 = coord_5;
    let _e35 = textureSampleGrad(tex2D, samp, _e30, vec2<f32>(4.0), vec2<f32>(4.0));
    c_2 = _e35;
    let _e43 = coord_5;
    let _e50 = textureSampleGrad(tex2D, samp, _e43, vec2<f32>(4.0), vec2<f32>(4.0), vec2<i32>(5, 5));
    c_2 = _e50;
    let _e53 = coord_5;
    let _e55 = textureSampleLevel(tex2D, samp, _e53, 3.0);
    c_2 = _e55;
    let _e60 = coord_5;
    let _e64 = textureSampleLevel(tex2D, samp, _e60, 3.0, vec2<i32>(5, 5));
    c_2 = _e64;
    let _e68 = coord_5;
    let _e71 = textureSample(tex2D, samp, _e68, vec2<i32>(5, 5));
    c_2 = _e71;
    let _e76 = coord_5;
    let _e80 = textureSampleBias(tex2D, samp, _e76, 2.0, vec2<i32>(5, 5));
    c_2 = _e80;
    let _e81 = coord_5;
    let _e86 = coord_5;
    let _e90 = vec3<f32>(_e86.x, _e86.y, 6.0);
    let _e95 = textureSample(tex2D, samp, (_e90.xy / vec2<f32>(_e90.z)));
    c_2 = _e95;
    let _e96 = coord_5;
    let _e102 = coord_5;
    let _e107 = vec4<f32>(_e102.x, _e102.y, 0.0, 6.0);
    let _e113 = textureSample(tex2D, samp, (_e107.xyz / vec3<f32>(_e107.w)).xy);
    c_2 = _e113;
    let _e114 = coord_5;
    let _e120 = coord_5;
    let _e124 = vec3<f32>(_e120.x, _e120.y, 6.0);
    let _e130 = textureSampleBias(tex2D, samp, (_e124.xy / vec2<f32>(_e124.z)), 2.0);
    c_2 = _e130;
    let _e131 = coord_5;
    let _e138 = coord_5;
    let _e143 = vec4<f32>(_e138.x, _e138.y, 0.0, 6.0);
    let _e150 = textureSampleBias(tex2D, samp, (_e143.xyz / vec3<f32>(_e143.w)).xy, 2.0);
    c_2 = _e150;
    let _e151 = coord_5;
    let _e160 = coord_5;
    let _e164 = vec3<f32>(_e160.x, _e160.y, 6.0);
    let _e173 = textureSampleGrad(tex2D, samp, (_e164.xy / vec2<f32>(_e164.z)), vec2<f32>(4.0), vec2<f32>(4.0));
    c_2 = _e173;
    let _e174 = coord_5;
    let _e184 = coord_5;
    let _e189 = vec4<f32>(_e184.x, _e184.y, 0.0, 6.0);
    let _e199 = textureSampleGrad(tex2D, samp, (_e189.xyz / vec3<f32>(_e189.w)).xy, vec2<f32>(4.0), vec2<f32>(4.0));
    c_2 = _e199;
    let _e200 = coord_5;
    let _e211 = coord_5;
    let _e215 = vec3<f32>(_e211.x, _e211.y, 6.0);
    let _e226 = textureSampleGrad(tex2D, samp, (_e215.xy / vec2<f32>(_e215.z)), vec2<f32>(4.0), vec2<f32>(4.0), vec2<i32>(5, 5));
    c_2 = _e226;
    let _e227 = coord_5;
    let _e239 = coord_5;
    let _e244 = vec4<f32>(_e239.x, _e239.y, 0.0, 6.0);
    let _e256 = textureSampleGrad(tex2D, samp, (_e244.xyz / vec3<f32>(_e244.w)).xy, vec2<f32>(4.0), vec2<f32>(4.0), vec2<i32>(5, 5));
    c_2 = _e256;
    let _e257 = coord_5;
    let _e263 = coord_5;
    let _e267 = vec3<f32>(_e263.x, _e263.y, 6.0);
    let _e273 = textureSampleLevel(tex2D, samp, (_e267.xy / vec2<f32>(_e267.z)), 3.0);
    c_2 = _e273;
    let _e274 = coord_5;
    let _e281 = coord_5;
    let _e286 = vec4<f32>(_e281.x, _e281.y, 0.0, 6.0);
    let _e293 = textureSampleLevel(tex2D, samp, (_e286.xyz / vec3<f32>(_e286.w)).xy, 3.0);
    c_2 = _e293;
    let _e294 = coord_5;
    let _e302 = coord_5;
    let _e306 = vec3<f32>(_e302.x, _e302.y, 6.0);
    let _e314 = textureSampleLevel(tex2D, samp, (_e306.xy / vec2<f32>(_e306.z)), 3.0, vec2<i32>(5, 5));
    c_2 = _e314;
    let _e315 = coord_5;
    let _e324 = coord_5;
    let _e329 = vec4<f32>(_e324.x, _e324.y, 0.0, 6.0);
    let _e338 = textureSampleLevel(tex2D, samp, (_e329.xyz / vec3<f32>(_e329.w)).xy, 3.0, vec2<i32>(5, 5));
    c_2 = _e338;
    let _e339 = coord_5;
    let _e346 = coord_5;
    let _e350 = vec3<f32>(_e346.x, _e346.y, 6.0);
    let _e357 = textureSample(tex2D, samp, (_e350.xy / vec2<f32>(_e350.z)), vec2<i32>(5, 5));
    c_2 = _e357;
    let _e358 = coord_5;
    let _e366 = coord_5;
    let _e371 = vec4<f32>(_e366.x, _e366.y, 0.0, 6.0);
    let _e379 = textureSample(tex2D, samp, (_e371.xyz / vec3<f32>(_e371.w)).xy, vec2<i32>(5, 5));
    c_2 = _e379;
    let _e380 = coord_5;
    let _e388 = coord_5;
    let _e392 = vec3<f32>(_e388.x, _e388.y, 6.0);
    let _e400 = textureSampleBias(tex2D, samp, (_e392.xy / vec2<f32>(_e392.z)), 2.0, vec2<i32>(5, 5));
    c_2 = _e400;
    let _e401 = coord_5;
    let _e410 = coord_5;
    let _e415 = vec4<f32>(_e410.x, _e410.y, 0.0, 6.0);
    let _e424 = textureSampleBias(tex2D, samp, (_e415.xyz / vec3<f32>(_e415.w)).xy, 2.0, vec2<i32>(5, 5));
    c_2 = _e424;
    return;
}

fn testTex2DShadow(coord_6: vec2<f32>) {
    var coord_7: vec2<f32>;
    var d: f32;

    coord_7 = coord_6;
    let _e17 = coord_7;
    let _e22 = coord_7;
    let _e26 = vec3<f32>(_e22.x, _e22.y, 1.0);
    let _e29 = textureSampleCompare(tex2DShadow, sampShadow, _e26.xy, _e26.z);
    d = _e29;
    let _e30 = coord_7;
    let _e39 = coord_7;
    let _e43 = vec3<f32>(_e39.x, _e39.y, 1.0);
    let _e50 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e43.xy, _e43.z);
    d = _e50;
    let _e51 = coord_7;
    let _e62 = coord_7;
    let _e66 = vec3<f32>(_e62.x, _e62.y, 1.0);
    let _e75 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e66.xy, _e66.z, vec2<i32>(5, 5));
    d = _e75;
    let _e76 = coord_7;
    let _e82 = coord_7;
    let _e86 = vec3<f32>(_e82.x, _e82.y, 1.0);
    let _e90 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e86.xy, _e86.z);
    d = _e90;
    let _e91 = coord_7;
    let _e99 = coord_7;
    let _e103 = vec3<f32>(_e99.x, _e99.y, 1.0);
    let _e109 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e103.xy, _e103.z, vec2<i32>(5, 5));
    d = _e109;
    let _e110 = coord_7;
    let _e117 = coord_7;
    let _e121 = vec3<f32>(_e117.x, _e117.y, 1.0);
    let _e126 = textureSampleCompare(tex2DShadow, sampShadow, _e121.xy, _e121.z, vec2<i32>(5, 5));
    d = _e126;
    let _e127 = coord_7;
    let _e133 = coord_7;
    let _e138 = vec4<f32>(_e133.x, _e133.y, 1.0, 6.0);
    let _e142 = (_e138.xyz / vec3<f32>(_e138.w));
    let _e145 = textureSampleCompare(tex2DShadow, sampShadow, _e142.xy, _e142.z);
    d = _e145;
    let _e146 = coord_7;
    let _e156 = coord_7;
    let _e161 = vec4<f32>(_e156.x, _e156.y, 1.0, 6.0);
    let _e169 = (_e161.xyz / vec3<f32>(_e161.w));
    let _e172 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e169.xy, _e169.z);
    d = _e172;
    let _e173 = coord_7;
    let _e185 = coord_7;
    let _e190 = vec4<f32>(_e185.x, _e185.y, 1.0, 6.0);
    let _e200 = (_e190.xyz / vec3<f32>(_e190.w));
    let _e203 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e200.xy, _e200.z, vec2<i32>(5, 5));
    d = _e203;
    let _e204 = coord_7;
    let _e211 = coord_7;
    let _e216 = vec4<f32>(_e211.x, _e211.y, 1.0, 6.0);
    let _e221 = (_e216.xyz / vec3<f32>(_e216.w));
    let _e224 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e221.xy, _e221.z);
    d = _e224;
    let _e225 = coord_7;
    let _e234 = coord_7;
    let _e239 = vec4<f32>(_e234.x, _e234.y, 1.0, 6.0);
    let _e246 = (_e239.xyz / vec3<f32>(_e239.w));
    let _e249 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e246.xy, _e246.z, vec2<i32>(5, 5));
    d = _e249;
    let _e250 = coord_7;
    let _e258 = coord_7;
    let _e263 = vec4<f32>(_e258.x, _e258.y, 1.0, 6.0);
    let _e269 = (_e263.xyz / vec3<f32>(_e263.w));
    let _e272 = textureSampleCompare(tex2DShadow, sampShadow, _e269.xy, _e269.z, vec2<i32>(5, 5));
    d = _e272;
    return;
}

fn testTex2DArray(coord_8: vec3<f32>) {
    var coord_9: vec3<f32>;
    var c_3: vec4<f32>;

    coord_9 = coord_8;
    let _e18 = coord_9;
    let _e22 = textureSample(tex2DArray, samp, _e18.xy, i32(_e18.z));
    c_3 = _e22;
    let _e25 = coord_9;
    let _e30 = textureSampleBias(tex2DArray, samp, _e25.xy, i32(_e25.z), 2.0);
    c_3 = _e30;
    let _e36 = coord_9;
    let _e44 = textureSampleGrad(tex2DArray, samp, _e36.xy, i32(_e36.z), vec2<f32>(4.0), vec2<f32>(4.0));
    c_3 = _e44;
    let _e52 = coord_9;
    let _e62 = textureSampleGrad(tex2DArray, samp, _e52.xy, i32(_e52.z), vec2<f32>(4.0), vec2<f32>(4.0), vec2<i32>(5, 5));
    c_3 = _e62;
    let _e65 = coord_9;
    let _e70 = textureSampleLevel(tex2DArray, samp, _e65.xy, i32(_e65.z), 3.0);
    c_3 = _e70;
    let _e75 = coord_9;
    let _e82 = textureSampleLevel(tex2DArray, samp, _e75.xy, i32(_e75.z), 3.0, vec2<i32>(5, 5));
    c_3 = _e82;
    let _e86 = coord_9;
    let _e92 = textureSample(tex2DArray, samp, _e86.xy, i32(_e86.z), vec2<i32>(5, 5));
    c_3 = _e92;
    let _e97 = coord_9;
    let _e104 = textureSampleBias(tex2DArray, samp, _e97.xy, i32(_e97.z), 2.0, vec2<i32>(5, 5));
    c_3 = _e104;
    return;
}

fn testTex2DArrayShadow(coord_10: vec3<f32>) {
    var coord_11: vec3<f32>;
    var d_1: f32;

    coord_11 = coord_10;
    let _e17 = coord_11;
    let _e23 = coord_11;
    let _e28 = vec4<f32>(_e23.x, _e23.y, _e23.z, 1.0);
    let _e33 = textureSampleCompare(tex2DArrayShadow, sampShadow, _e28.xy, i32(_e28.z), _e28.w);
    d_1 = _e33;
    let _e34 = coord_11;
    let _e44 = coord_11;
    let _e49 = vec4<f32>(_e44.x, _e44.y, _e44.z, 1.0);
    let _e58 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e49.xy, i32(_e49.z), _e49.w);
    d_1 = _e58;
    let _e59 = coord_11;
    let _e71 = coord_11;
    let _e76 = vec4<f32>(_e71.x, _e71.y, _e71.z, 1.0);
    let _e87 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e76.xy, i32(_e76.z), _e76.w, vec2<i32>(5, 5));
    d_1 = _e87;
    let _e88 = coord_11;
    let _e95 = coord_11;
    let _e100 = vec4<f32>(_e95.x, _e95.y, _e95.z, 1.0);
    let _e106 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e100.xy, i32(_e100.z), _e100.w);
    d_1 = _e106;
    let _e107 = coord_11;
    let _e116 = coord_11;
    let _e121 = vec4<f32>(_e116.x, _e116.y, _e116.z, 1.0);
    let _e129 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e121.xy, i32(_e121.z), _e121.w, vec2<i32>(5, 5));
    d_1 = _e129;
    let _e130 = coord_11;
    let _e138 = coord_11;
    let _e143 = vec4<f32>(_e138.x, _e138.y, _e138.z, 1.0);
    let _e150 = textureSampleCompare(tex2DArrayShadow, sampShadow, _e143.xy, i32(_e143.z), _e143.w, vec2<i32>(5, 5));
    d_1 = _e150;
    return;
}

fn testTexCube(coord_12: vec3<f32>) {
    var coord_13: vec3<f32>;
    var c_4: vec4<f32>;

    coord_13 = coord_12;
    let _e18 = coord_13;
    let _e19 = textureSample(texCube, samp, _e18);
    c_4 = _e19;
    let _e22 = coord_13;
    let _e24 = textureSampleBias(texCube, samp, _e22, 2.0);
    c_4 = _e24;
    let _e30 = coord_13;
    let _e35 = textureSampleGrad(texCube, samp, _e30, vec3<f32>(4.0), vec3<f32>(4.0));
    c_4 = _e35;
    let _e38 = coord_13;
    let _e40 = textureSampleLevel(texCube, samp, _e38, 3.0);
    c_4 = _e40;
    let _e45 = coord_13;
    let _e49 = textureSampleLevel(texCube, samp, _e45, 3.0, vec3<i32>(5, 5, 5));
    c_4 = _e49;
    let _e53 = coord_13;
    let _e56 = textureSample(texCube, samp, _e53, vec3<i32>(5, 5, 5));
    c_4 = _e56;
    let _e61 = coord_13;
    let _e65 = textureSampleBias(texCube, samp, _e61, 2.0, vec3<i32>(5, 5, 5));
    c_4 = _e65;
    return;
}

fn testTexCubeShadow(coord_14: vec3<f32>) {
    var coord_15: vec3<f32>;
    var d_2: f32;

    coord_15 = coord_14;
    let _e17 = coord_15;
    let _e23 = coord_15;
    let _e28 = vec4<f32>(_e23.x, _e23.y, _e23.z, 1.0);
    let _e31 = textureSampleCompare(texCubeShadow, sampShadow, _e28.xyz, _e28.w);
    d_2 = _e31;
    let _e32 = coord_15;
    let _e42 = coord_15;
    let _e47 = vec4<f32>(_e42.x, _e42.y, _e42.z, 1.0);
    let _e54 = textureSampleCompareLevel(texCubeShadow, sampShadow, _e47.xyz, _e47.w);
    d_2 = _e54;
    let _e55 = coord_15;
    let _e62 = coord_15;
    let _e67 = vec4<f32>(_e62.x, _e62.y, _e62.z, 1.0);
    let _e71 = textureSampleCompareLevel(texCubeShadow, sampShadow, _e67.xyz, _e67.w);
    d_2 = _e71;
    let _e72 = coord_15;
    let _e81 = coord_15;
    let _e86 = vec4<f32>(_e81.x, _e81.y, _e81.z, 1.0);
    let _e92 = textureSampleCompareLevel(texCubeShadow, sampShadow, _e86.xyz, _e86.w, vec3<i32>(5, 5, 5));
    d_2 = _e92;
    let _e93 = coord_15;
    let _e101 = coord_15;
    let _e106 = vec4<f32>(_e101.x, _e101.y, _e101.z, 1.0);
    let _e111 = textureSampleCompare(texCubeShadow, sampShadow, _e106.xyz, _e106.w, vec3<i32>(5, 5, 5));
    d_2 = _e111;
    return;
}

fn testTexCubeArray(coord_16: vec4<f32>) {
    var coord_17: vec4<f32>;
    var c_5: vec4<f32>;

    coord_17 = coord_16;
    let _e18 = coord_17;
    let _e22 = textureSample(texCubeArray, samp, _e18.xyz, i32(_e18.w));
    c_5 = _e22;
    let _e25 = coord_17;
    let _e30 = textureSampleBias(texCubeArray, samp, _e25.xyz, i32(_e25.w), 2.0);
    c_5 = _e30;
    let _e36 = coord_17;
    let _e44 = textureSampleGrad(texCubeArray, samp, _e36.xyz, i32(_e36.w), vec3<f32>(4.0), vec3<f32>(4.0));
    c_5 = _e44;
    let _e47 = coord_17;
    let _e52 = textureSampleLevel(texCubeArray, samp, _e47.xyz, i32(_e47.w), 3.0);
    c_5 = _e52;
    let _e57 = coord_17;
    let _e64 = textureSampleLevel(texCubeArray, samp, _e57.xyz, i32(_e57.w), 3.0, vec3<i32>(5, 5, 5));
    c_5 = _e64;
    let _e68 = coord_17;
    let _e74 = textureSample(texCubeArray, samp, _e68.xyz, i32(_e68.w), vec3<i32>(5, 5, 5));
    c_5 = _e74;
    let _e79 = coord_17;
    let _e86 = textureSampleBias(texCubeArray, samp, _e79.xyz, i32(_e79.w), 2.0, vec3<i32>(5, 5, 5));
    c_5 = _e86;
    return;
}

fn testTexCubeArrayShadow(coord_18: vec4<f32>) {
    var coord_19: vec4<f32>;
    var d_3: f32;

    coord_19 = coord_18;
    let _e19 = coord_19;
    let _e24 = textureSampleCompare(texCubeArrayShadow, sampShadow, _e19.xyz, i32(_e19.w), 1.0);
    d_3 = _e24;
    return;
}

fn testTex3D(coord_20: vec3<f32>) {
    var coord_21: vec3<f32>;
    var c_6: vec4<f32>;

    coord_21 = coord_20;
    let _e18 = coord_21;
    let _e19 = textureSample(tex3D, samp, _e18);
    c_6 = _e19;
    let _e22 = coord_21;
    let _e24 = textureSampleBias(tex3D, samp, _e22, 2.0);
    c_6 = _e24;
    let _e30 = coord_21;
    let _e35 = textureSampleGrad(tex3D, samp, _e30, vec3<f32>(4.0), vec3<f32>(4.0));
    c_6 = _e35;
    let _e43 = coord_21;
    let _e50 = textureSampleGrad(tex3D, samp, _e43, vec3<f32>(4.0), vec3<f32>(4.0), vec3<i32>(5, 5, 5));
    c_6 = _e50;
    let _e53 = coord_21;
    let _e55 = textureSampleLevel(tex3D, samp, _e53, 3.0);
    c_6 = _e55;
    let _e60 = coord_21;
    let _e64 = textureSampleLevel(tex3D, samp, _e60, 3.0, vec3<i32>(5, 5, 5));
    c_6 = _e64;
    let _e68 = coord_21;
    let _e71 = textureSample(tex3D, samp, _e68, vec3<i32>(5, 5, 5));
    c_6 = _e71;
    let _e76 = coord_21;
    let _e80 = textureSampleBias(tex3D, samp, _e76, 2.0, vec3<i32>(5, 5, 5));
    c_6 = _e80;
    return;
}

fn main_1() {
    return;
}

@stage(fragment) 
fn main(@location(0) texcoord: vec4<f32>) {
    texcoord_1 = texcoord;
    main_1();
    return;
}
