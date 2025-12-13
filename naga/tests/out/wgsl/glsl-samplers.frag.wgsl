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
var utex2D: texture_2d<u32>;
@group(1) @binding(8) 
var itex2D: texture_2d<i32>;
@group(2) @binding(0) 
var samp: sampler;
@group(1) @binding(12) 
var tex2DShadow: texture_depth_2d;
@group(1) @binding(13) 
var tex2DArrayShadow: texture_depth_2d_array;
@group(1) @binding(14) 
var texCubeShadow: texture_depth_cube;
@group(1) @binding(15) 
var texCubeArrayShadow: texture_depth_cube_array;
@group(1) @binding(17) 
var sampShadow: sampler_comparison;
@group(0) @binding(18) 
var tex2DMS: texture_multisampled_2d<f32>;
@group(0) @binding(19) 
var tex2DMSArray: texture_multisampled_2d_array<f32>;

fn testTex1D(coord: f32) {
    var coord_1: f32;
    var size1D: i32;
    var levels: i32;
    var c: vec4<f32>;

    coord_1 = coord;
    let _e5 = textureDimensions(tex1D, 0i);
    size1D = i32(_e5);
    let _e8 = textureNumLevels(tex1D);
    levels = i32(_e8);
    let _e12 = coord_1;
    let _e13 = textureSample(tex1D, samp, _e12);
    c = _e13;
    let _e14 = coord_1;
    let _e17 = textureSampleGrad(tex1D, samp, _e14, 4f, 4f);
    c = _e17;
    let _e18 = coord_1;
    let _e22 = textureSampleGrad(tex1D, samp, _e18, 4f, 4f, 5i);
    c = _e22;
    let _e23 = coord_1;
    let _e25 = textureSampleLevel(tex1D, samp, _e23, 3f);
    c = _e25;
    let _e26 = coord_1;
    let _e29 = textureSampleLevel(tex1D, samp, _e26, 3f, 5i);
    c = _e29;
    let _e30 = coord_1;
    let _e32 = textureSample(tex1D, samp, _e30, 5i);
    c = _e32;
    let _e33 = coord_1;
    let _e35 = vec2<f32>(_e33, 6f);
    let _e39 = textureSample(tex1D, samp, (_e35.x / _e35.y));
    c = _e39;
    let _e40 = coord_1;
    let _e44 = vec4<f32>(_e40, 0f, 0f, 6f);
    let _e50 = textureSample(tex1D, samp, (_e44.xyz / vec3(_e44.w)).x);
    c = _e50;
    let _e51 = coord_1;
    let _e53 = vec2<f32>(_e51, 6f);
    let _e59 = textureSampleGrad(tex1D, samp, (_e53.x / _e53.y), 4f, 4f);
    c = _e59;
    let _e60 = coord_1;
    let _e64 = vec4<f32>(_e60, 0f, 0f, 6f);
    let _e72 = textureSampleGrad(tex1D, samp, (_e64.xyz / vec3(_e64.w)).x, 4f, 4f);
    c = _e72;
    let _e73 = coord_1;
    let _e75 = vec2<f32>(_e73, 6f);
    let _e82 = textureSampleGrad(tex1D, samp, (_e75.x / _e75.y), 4f, 4f, 5i);
    c = _e82;
    let _e83 = coord_1;
    let _e87 = vec4<f32>(_e83, 0f, 0f, 6f);
    let _e96 = textureSampleGrad(tex1D, samp, (_e87.xyz / vec3(_e87.w)).x, 4f, 4f, 5i);
    c = _e96;
    let _e97 = coord_1;
    let _e99 = vec2<f32>(_e97, 6f);
    let _e104 = textureSampleLevel(tex1D, samp, (_e99.x / _e99.y), 3f);
    c = _e104;
    let _e105 = coord_1;
    let _e109 = vec4<f32>(_e105, 0f, 0f, 6f);
    let _e116 = textureSampleLevel(tex1D, samp, (_e109.xyz / vec3(_e109.w)).x, 3f);
    c = _e116;
    let _e117 = coord_1;
    let _e119 = vec2<f32>(_e117, 6f);
    let _e125 = textureSampleLevel(tex1D, samp, (_e119.x / _e119.y), 3f, 5i);
    c = _e125;
    let _e126 = coord_1;
    let _e130 = vec4<f32>(_e126, 0f, 0f, 6f);
    let _e138 = textureSampleLevel(tex1D, samp, (_e130.xyz / vec3(_e130.w)).x, 3f, 5i);
    c = _e138;
    let _e139 = coord_1;
    let _e141 = vec2<f32>(_e139, 6f);
    let _e146 = textureSample(tex1D, samp, (_e141.x / _e141.y), 5i);
    c = _e146;
    let _e147 = coord_1;
    let _e151 = vec4<f32>(_e147, 0f, 0f, 6f);
    let _e158 = textureSample(tex1D, samp, (_e151.xyz / vec3(_e151.w)).x, 5i);
    c = _e158;
    let _e159 = coord_1;
    let _e162 = textureLoad(tex1D, i32(_e159), 3i);
    c = _e162;
    let _e163 = coord_1;
    let _e166 = textureLoad(tex1D, i32(_e163), 3i);
    c = _e166;
    return;
}

fn testTex1DArray(coord_2: vec2<f32>) {
    var coord_3: vec2<f32>;
    var size1DArray: vec2<i32>;
    var levels_1: i32;
    var c_1: vec4<f32>;

    coord_3 = coord_2;
    let _e5 = textureDimensions(tex1DArray, 0i);
    let _e6 = textureNumLayers(tex1DArray);
    size1DArray = vec2<i32>(vec2<u32>(_e5, _e6));
    let _e10 = textureNumLevels(tex1DArray);
    levels_1 = i32(_e10);
    let _e14 = coord_3;
    let _e18 = textureSample(tex1DArray, samp, _e14.x, i32(_e14.y));
    c_1 = _e18;
    let _e19 = coord_3;
    let _e25 = textureSampleGrad(tex1DArray, samp, _e19.x, i32(_e19.y), 4f, 4f);
    c_1 = _e25;
    let _e26 = coord_3;
    let _e33 = textureSampleGrad(tex1DArray, samp, _e26.x, i32(_e26.y), 4f, 4f, 5i);
    c_1 = _e33;
    let _e34 = coord_3;
    let _e39 = textureSampleLevel(tex1DArray, samp, _e34.x, i32(_e34.y), 3f);
    c_1 = _e39;
    let _e40 = coord_3;
    let _e46 = textureSampleLevel(tex1DArray, samp, _e40.x, i32(_e40.y), 3f, 5i);
    c_1 = _e46;
    let _e47 = coord_3;
    let _e52 = textureSample(tex1DArray, samp, _e47.x, i32(_e47.y), 5i);
    c_1 = _e52;
    let _e53 = coord_3;
    let _e54 = vec2<i32>(_e53);
    let _e58 = textureLoad(tex1DArray, _e54.x, _e54.y, 3i);
    c_1 = _e58;
    let _e59 = coord_3;
    let _e60 = vec2<i32>(_e59);
    let _e64 = textureLoad(tex1DArray, _e60.x, _e60.y, 3i);
    c_1 = _e64;
    return;
}

fn testTex2D(coord_4: vec2<f32>) {
    var coord_5: vec2<f32>;
    var size2D: vec2<i32>;
    var levels_2: i32;
    var c_2: vec4<f32>;

    coord_5 = coord_4;
    let _e7 = textureDimensions(tex2D, 0i);
    size2D = vec2<i32>(_e7);
    let _e10 = textureNumLevels(tex2D);
    levels_2 = i32(_e10);
    let _e14 = coord_5;
    let _e15 = textureSample(tex2D, samp, _e14);
    c_2 = _e15;
    let _e16 = coord_5;
    let _e18 = textureSampleBias(tex2D, samp, _e16, 2f);
    c_2 = _e18;
    let _e19 = coord_5;
    let _e24 = textureSampleGrad(tex2D, samp, _e19, vec2(4f), vec2(4f));
    c_2 = _e24;
    let _e25 = coord_5;
    let _e32 = textureSampleGrad(tex2D, samp, _e25, vec2(4f), vec2(4f), vec2(5i));
    c_2 = _e32;
    let _e33 = coord_5;
    let _e35 = textureSampleLevel(tex2D, samp, _e33, 3f);
    c_2 = _e35;
    let _e36 = coord_5;
    let _e40 = textureSampleLevel(tex2D, samp, _e36, 3f, vec2(5i));
    c_2 = _e40;
    let _e41 = coord_5;
    let _e44 = textureSample(tex2D, samp, _e41, vec2(5i));
    c_2 = _e44;
    let _e45 = coord_5;
    let _e49 = textureSampleBias(tex2D, samp, _e45, 2f, vec2(5i));
    c_2 = _e49;
    let _e50 = coord_5;
    let _e54 = vec3<f32>(_e50.x, _e50.y, 6f);
    let _e59 = textureSample(tex2D, samp, (_e54.xy / vec2(_e54.z)));
    c_2 = _e59;
    let _e60 = coord_5;
    let _e65 = vec4<f32>(_e60.x, _e60.y, 0f, 6f);
    let _e71 = textureSample(tex2D, samp, (_e65.xyz / vec3(_e65.w)).xy);
    c_2 = _e71;
    let _e72 = coord_5;
    let _e76 = vec3<f32>(_e72.x, _e72.y, 6f);
    let _e82 = textureSampleBias(tex2D, samp, (_e76.xy / vec2(_e76.z)), 2f);
    c_2 = _e82;
    let _e83 = coord_5;
    let _e88 = vec4<f32>(_e83.x, _e83.y, 0f, 6f);
    let _e95 = textureSampleBias(tex2D, samp, (_e88.xyz / vec3(_e88.w)).xy, 2f);
    c_2 = _e95;
    let _e96 = coord_5;
    let _e100 = vec3<f32>(_e96.x, _e96.y, 6f);
    let _e109 = textureSampleGrad(tex2D, samp, (_e100.xy / vec2(_e100.z)), vec2(4f), vec2(4f));
    c_2 = _e109;
    let _e110 = coord_5;
    let _e115 = vec4<f32>(_e110.x, _e110.y, 0f, 6f);
    let _e125 = textureSampleGrad(tex2D, samp, (_e115.xyz / vec3(_e115.w)).xy, vec2(4f), vec2(4f));
    c_2 = _e125;
    let _e126 = coord_5;
    let _e130 = vec3<f32>(_e126.x, _e126.y, 6f);
    let _e141 = textureSampleGrad(tex2D, samp, (_e130.xy / vec2(_e130.z)), vec2(4f), vec2(4f), vec2(5i));
    c_2 = _e141;
    let _e142 = coord_5;
    let _e147 = vec4<f32>(_e142.x, _e142.y, 0f, 6f);
    let _e159 = textureSampleGrad(tex2D, samp, (_e147.xyz / vec3(_e147.w)).xy, vec2(4f), vec2(4f), vec2(5i));
    c_2 = _e159;
    let _e160 = coord_5;
    let _e164 = vec3<f32>(_e160.x, _e160.y, 6f);
    let _e170 = textureSampleLevel(tex2D, samp, (_e164.xy / vec2(_e164.z)), 3f);
    c_2 = _e170;
    let _e171 = coord_5;
    let _e176 = vec4<f32>(_e171.x, _e171.y, 0f, 6f);
    let _e183 = textureSampleLevel(tex2D, samp, (_e176.xyz / vec3(_e176.w)).xy, 3f);
    c_2 = _e183;
    let _e184 = coord_5;
    let _e188 = vec3<f32>(_e184.x, _e184.y, 6f);
    let _e196 = textureSampleLevel(tex2D, samp, (_e188.xy / vec2(_e188.z)), 3f, vec2(5i));
    c_2 = _e196;
    let _e197 = coord_5;
    let _e202 = vec4<f32>(_e197.x, _e197.y, 0f, 6f);
    let _e211 = textureSampleLevel(tex2D, samp, (_e202.xyz / vec3(_e202.w)).xy, 3f, vec2(5i));
    c_2 = _e211;
    let _e212 = coord_5;
    let _e216 = vec3<f32>(_e212.x, _e212.y, 6f);
    let _e223 = textureSample(tex2D, samp, (_e216.xy / vec2(_e216.z)), vec2(5i));
    c_2 = _e223;
    let _e224 = coord_5;
    let _e229 = vec4<f32>(_e224.x, _e224.y, 0f, 6f);
    let _e237 = textureSample(tex2D, samp, (_e229.xyz / vec3(_e229.w)).xy, vec2(5i));
    c_2 = _e237;
    let _e238 = coord_5;
    let _e242 = vec3<f32>(_e238.x, _e238.y, 6f);
    let _e250 = textureSampleBias(tex2D, samp, (_e242.xy / vec2(_e242.z)), 2f, vec2(5i));
    c_2 = _e250;
    let _e251 = coord_5;
    let _e256 = vec4<f32>(_e251.x, _e251.y, 0f, 6f);
    let _e265 = textureSampleBias(tex2D, samp, (_e256.xyz / vec3(_e256.w)).xy, 2f, vec2(5i));
    c_2 = _e265;
    let _e266 = coord_5;
    let _e269 = textureLoad(tex2D, vec2<i32>(_e266), 3i);
    c_2 = _e269;
    let _e270 = coord_5;
    let _e273 = textureLoad(utex2D, vec2<i32>(_e270), 3i);
    c_2 = vec4<f32>(_e273);
    let _e275 = coord_5;
    let _e278 = textureLoad(itex2D, vec2<i32>(_e275), 3i);
    c_2 = vec4<f32>(_e278);
    let _e280 = coord_5;
    let _e283 = textureLoad(tex2D, vec2<i32>(_e280), 3i);
    c_2 = _e283;
    let _e284 = coord_5;
    let _e287 = textureLoad(utex2D, vec2<i32>(_e284), 3i);
    c_2 = vec4<f32>(_e287);
    let _e289 = coord_5;
    let _e292 = textureLoad(itex2D, vec2<i32>(_e289), 3i);
    c_2 = vec4<f32>(_e292);
    return;
}

fn testTex2DShadow(coord_6: vec2<f32>) {
    var coord_7: vec2<f32>;
    var size2DShadow: vec2<i32>;
    var levels_3: i32;
    var d: f32;

    coord_7 = coord_6;
    let _e5 = textureDimensions(tex2DShadow, 0i);
    size2DShadow = vec2<i32>(_e5);
    let _e8 = textureNumLevels(tex2DShadow);
    levels_3 = i32(_e8);
    let _e12 = coord_7;
    let _e16 = vec3<f32>(_e12.x, _e12.y, 1f);
    let _e19 = textureSampleCompare(tex2DShadow, sampShadow, _e16.xy, _e16.z);
    d = _e19;
    let _e20 = coord_7;
    let _e24 = vec3<f32>(_e20.x, _e20.y, 1f);
    let _e27 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e24.xy, _e24.z);
    d = _e27;
    let _e28 = coord_7;
    let _e32 = vec3<f32>(_e28.x, _e28.y, 1f);
    let _e37 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e32.xy, _e32.z, vec2(5i));
    d = _e37;
    let _e38 = coord_7;
    let _e42 = vec3<f32>(_e38.x, _e38.y, 1f);
    let _e45 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e42.xy, _e42.z);
    d = _e45;
    let _e46 = coord_7;
    let _e50 = vec3<f32>(_e46.x, _e46.y, 1f);
    let _e55 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e50.xy, _e50.z, vec2(5i));
    d = _e55;
    let _e56 = coord_7;
    let _e60 = vec3<f32>(_e56.x, _e56.y, 1f);
    let _e65 = textureSampleCompare(tex2DShadow, sampShadow, _e60.xy, _e60.z, vec2(5i));
    d = _e65;
    let _e66 = coord_7;
    let _e71 = vec4<f32>(_e66.x, _e66.y, 1f, 6f);
    let _e75 = (_e71.xyz / vec3(_e71.w));
    let _e78 = textureSampleCompare(tex2DShadow, sampShadow, _e75.xy, _e75.z);
    d = _e78;
    let _e79 = coord_7;
    let _e84 = vec4<f32>(_e79.x, _e79.y, 1f, 6f);
    let _e88 = (_e84.xyz / vec3(_e84.w));
    let _e91 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e88.xy, _e88.z);
    d = _e91;
    let _e92 = coord_7;
    let _e97 = vec4<f32>(_e92.x, _e92.y, 1f, 6f);
    let _e103 = (_e97.xyz / vec3(_e97.w));
    let _e106 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e103.xy, _e103.z, vec2(5i));
    d = _e106;
    let _e107 = coord_7;
    let _e112 = vec4<f32>(_e107.x, _e107.y, 1f, 6f);
    let _e116 = (_e112.xyz / vec3(_e112.w));
    let _e119 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e116.xy, _e116.z);
    d = _e119;
    let _e120 = coord_7;
    let _e125 = vec4<f32>(_e120.x, _e120.y, 1f, 6f);
    let _e131 = (_e125.xyz / vec3(_e125.w));
    let _e134 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e131.xy, _e131.z, vec2(5i));
    d = _e134;
    let _e135 = coord_7;
    let _e140 = vec4<f32>(_e135.x, _e135.y, 1f, 6f);
    let _e146 = (_e140.xyz / vec3(_e140.w));
    let _e149 = textureSampleCompare(tex2DShadow, sampShadow, _e146.xy, _e146.z, vec2(5i));
    d = _e149;
    return;
}

fn testTex2DArray(coord_8: vec3<f32>) {
    var coord_9: vec3<f32>;
    var size2DArray: vec3<i32>;
    var levels_4: i32;
    var c_3: vec4<f32>;

    coord_9 = coord_8;
    let _e5 = textureDimensions(tex2DArray, 0i);
    let _e8 = textureNumLayers(tex2DArray);
    size2DArray = vec3<i32>(vec3<u32>(_e5.x, _e5.y, _e8));
    let _e12 = textureNumLevels(tex2DArray);
    levels_4 = i32(_e12);
    let _e16 = coord_9;
    let _e20 = textureSample(tex2DArray, samp, _e16.xy, i32(_e16.z));
    c_3 = _e20;
    let _e21 = coord_9;
    let _e26 = textureSampleBias(tex2DArray, samp, _e21.xy, i32(_e21.z), 2f);
    c_3 = _e26;
    let _e27 = coord_9;
    let _e35 = textureSampleGrad(tex2DArray, samp, _e27.xy, i32(_e27.z), vec2(4f), vec2(4f));
    c_3 = _e35;
    let _e36 = coord_9;
    let _e46 = textureSampleGrad(tex2DArray, samp, _e36.xy, i32(_e36.z), vec2(4f), vec2(4f), vec2(5i));
    c_3 = _e46;
    let _e47 = coord_9;
    let _e52 = textureSampleLevel(tex2DArray, samp, _e47.xy, i32(_e47.z), 3f);
    c_3 = _e52;
    let _e53 = coord_9;
    let _e60 = textureSampleLevel(tex2DArray, samp, _e53.xy, i32(_e53.z), 3f, vec2(5i));
    c_3 = _e60;
    let _e61 = coord_9;
    let _e67 = textureSample(tex2DArray, samp, _e61.xy, i32(_e61.z), vec2(5i));
    c_3 = _e67;
    let _e68 = coord_9;
    let _e75 = textureSampleBias(tex2DArray, samp, _e68.xy, i32(_e68.z), 2f, vec2(5i));
    c_3 = _e75;
    let _e76 = coord_9;
    let _e77 = vec3<i32>(_e76);
    let _e81 = textureLoad(tex2DArray, _e77.xy, _e77.z, 3i);
    c_3 = _e81;
    let _e82 = coord_9;
    let _e83 = vec3<i32>(_e82);
    let _e87 = textureLoad(tex2DArray, _e83.xy, _e83.z, 3i);
    c_3 = _e87;
    return;
}

fn testTex2DArrayShadow(coord_10: vec3<f32>) {
    var coord_11: vec3<f32>;
    var size2DArrayShadow: vec3<i32>;
    var levels_5: i32;
    var d_1: f32;

    coord_11 = coord_10;
    let _e5 = textureDimensions(tex2DArrayShadow, 0i);
    let _e8 = textureNumLayers(tex2DArrayShadow);
    size2DArrayShadow = vec3<i32>(vec3<u32>(_e5.x, _e5.y, _e8));
    let _e12 = textureNumLevels(tex2DArrayShadow);
    levels_5 = i32(_e12);
    let _e16 = coord_11;
    let _e21 = vec4<f32>(_e16.x, _e16.y, _e16.z, 1f);
    let _e26 = textureSampleCompare(tex2DArrayShadow, sampShadow, _e21.xy, i32(_e21.z), _e21.w);
    d_1 = _e26;
    let _e27 = coord_11;
    let _e32 = vec4<f32>(_e27.x, _e27.y, _e27.z, 1f);
    let _e37 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e32.xy, i32(_e32.z), _e32.w);
    d_1 = _e37;
    let _e38 = coord_11;
    let _e43 = vec4<f32>(_e38.x, _e38.y, _e38.z, 1f);
    let _e50 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e43.xy, i32(_e43.z), _e43.w, vec2(5i));
    d_1 = _e50;
    let _e51 = coord_11;
    let _e56 = vec4<f32>(_e51.x, _e51.y, _e51.z, 1f);
    let _e63 = textureSampleCompare(tex2DArrayShadow, sampShadow, _e56.xy, i32(_e56.z), _e56.w, vec2(5i));
    d_1 = _e63;
    return;
}

fn testTexCube(coord_12: vec3<f32>) {
    var coord_13: vec3<f32>;
    var sizeCube: vec2<i32>;
    var levels_6: i32;
    var c_4: vec4<f32>;

    coord_13 = coord_12;
    let _e5 = textureDimensions(texCube, 0i);
    sizeCube = vec2<i32>(_e5);
    let _e8 = textureNumLevels(texCube);
    levels_6 = i32(_e8);
    let _e12 = coord_13;
    let _e13 = textureSample(texCube, samp, _e12);
    c_4 = _e13;
    let _e14 = coord_13;
    let _e16 = textureSampleBias(texCube, samp, _e14, 2f);
    c_4 = _e16;
    let _e17 = coord_13;
    let _e22 = textureSampleGrad(texCube, samp, _e17, vec3(4f), vec3(4f));
    c_4 = _e22;
    let _e23 = coord_13;
    let _e25 = textureSampleLevel(texCube, samp, _e23, 3f);
    c_4 = _e25;
    return;
}

fn testTexCubeShadow(coord_14: vec3<f32>) {
    var coord_15: vec3<f32>;
    var sizeCubeShadow: vec2<i32>;
    var levels_7: i32;
    var d_2: f32;

    coord_15 = coord_14;
    let _e5 = textureDimensions(texCubeShadow, 0i);
    sizeCubeShadow = vec2<i32>(_e5);
    let _e8 = textureNumLevels(texCubeShadow);
    levels_7 = i32(_e8);
    let _e12 = coord_15;
    let _e17 = vec4<f32>(_e12.x, _e12.y, _e12.z, 1f);
    let _e20 = textureSampleCompare(texCubeShadow, sampShadow, _e17.xyz, _e17.w);
    d_2 = _e20;
    let _e21 = coord_15;
    let _e26 = vec4<f32>(_e21.x, _e21.y, _e21.z, 1f);
    let _e29 = textureSampleCompareLevel(texCubeShadow, sampShadow, _e26.xyz, _e26.w);
    d_2 = _e29;
    return;
}

fn testTexCubeArray(coord_16: vec4<f32>) {
    var coord_17: vec4<f32>;
    var sizeCubeArray: vec3<i32>;
    var levels_8: i32;
    var c_5: vec4<f32>;

    coord_17 = coord_16;
    let _e5 = textureDimensions(texCubeArray, 0i);
    let _e8 = textureNumLayers(texCubeArray);
    sizeCubeArray = vec3<i32>(vec3<u32>(_e5.x, _e5.y, _e8));
    let _e12 = textureNumLevels(texCubeArray);
    levels_8 = i32(_e12);
    let _e16 = coord_17;
    let _e20 = textureSample(texCubeArray, samp, _e16.xyz, i32(_e16.w));
    c_5 = _e20;
    let _e21 = coord_17;
    let _e26 = textureSampleBias(texCubeArray, samp, _e21.xyz, i32(_e21.w), 2f);
    c_5 = _e26;
    let _e27 = coord_17;
    let _e35 = textureSampleGrad(texCubeArray, samp, _e27.xyz, i32(_e27.w), vec3(4f), vec3(4f));
    c_5 = _e35;
    let _e36 = coord_17;
    let _e41 = textureSampleLevel(texCubeArray, samp, _e36.xyz, i32(_e36.w), 3f);
    c_5 = _e41;
    return;
}

fn testTexCubeArrayShadow(coord_18: vec4<f32>) {
    var coord_19: vec4<f32>;
    var sizeCubeArrayShadow: vec3<i32>;
    var levels_9: i32;
    var d_3: f32;

    coord_19 = coord_18;
    let _e5 = textureDimensions(texCubeArrayShadow, 0i);
    let _e8 = textureNumLayers(texCubeArrayShadow);
    sizeCubeArrayShadow = vec3<i32>(vec3<u32>(_e5.x, _e5.y, _e8));
    let _e12 = textureNumLevels(texCubeArrayShadow);
    levels_9 = i32(_e12);
    let _e16 = coord_19;
    let _e21 = textureSampleCompare(texCubeArrayShadow, sampShadow, _e16.xyz, i32(_e16.w), 1f);
    d_3 = _e21;
    return;
}

fn testTex3D(coord_20: vec3<f32>) {
    var coord_21: vec3<f32>;
    var size3D: vec3<i32>;
    var levels_10: i32;
    var c_6: vec4<f32>;

    coord_21 = coord_20;
    let _e5 = textureDimensions(tex3D, 0i);
    size3D = vec3<i32>(_e5);
    let _e8 = textureNumLevels(tex3D);
    levels_10 = i32(_e8);
    let _e12 = coord_21;
    let _e13 = textureSample(tex3D, samp, _e12);
    c_6 = _e13;
    let _e14 = coord_21;
    let _e16 = textureSampleBias(tex3D, samp, _e14, 2f);
    c_6 = _e16;
    let _e17 = coord_21;
    let _e22 = vec4<f32>(_e17.x, _e17.y, _e17.z, 6f);
    let _e27 = textureSample(tex3D, samp, (_e22.xyz / vec3(_e22.w)));
    c_6 = _e27;
    let _e28 = coord_21;
    let _e33 = vec4<f32>(_e28.x, _e28.y, _e28.z, 6f);
    let _e39 = textureSampleBias(tex3D, samp, (_e33.xyz / vec3(_e33.w)), 2f);
    c_6 = _e39;
    let _e40 = coord_21;
    let _e45 = vec4<f32>(_e40.x, _e40.y, _e40.z, 6f);
    let _e52 = textureSample(tex3D, samp, (_e45.xyz / vec3(_e45.w)), vec3(5i));
    c_6 = _e52;
    let _e53 = coord_21;
    let _e58 = vec4<f32>(_e53.x, _e53.y, _e53.z, 6f);
    let _e66 = textureSampleBias(tex3D, samp, (_e58.xyz / vec3(_e58.w)), 2f, vec3(5i));
    c_6 = _e66;
    let _e67 = coord_21;
    let _e72 = vec4<f32>(_e67.x, _e67.y, _e67.z, 6f);
    let _e78 = textureSampleLevel(tex3D, samp, (_e72.xyz / vec3(_e72.w)), 3f);
    c_6 = _e78;
    let _e79 = coord_21;
    let _e84 = vec4<f32>(_e79.x, _e79.y, _e79.z, 6f);
    let _e92 = textureSampleLevel(tex3D, samp, (_e84.xyz / vec3(_e84.w)), 3f, vec3(5i));
    c_6 = _e92;
    let _e93 = coord_21;
    let _e98 = vec4<f32>(_e93.x, _e93.y, _e93.z, 6f);
    let _e107 = textureSampleGrad(tex3D, samp, (_e98.xyz / vec3(_e98.w)), vec3(4f), vec3(4f));
    c_6 = _e107;
    let _e108 = coord_21;
    let _e113 = vec4<f32>(_e108.x, _e108.y, _e108.z, 6f);
    let _e124 = textureSampleGrad(tex3D, samp, (_e113.xyz / vec3(_e113.w)), vec3(4f), vec3(4f), vec3(5i));
    c_6 = _e124;
    let _e125 = coord_21;
    let _e130 = textureSampleGrad(tex3D, samp, _e125, vec3(4f), vec3(4f));
    c_6 = _e130;
    let _e131 = coord_21;
    let _e138 = textureSampleGrad(tex3D, samp, _e131, vec3(4f), vec3(4f), vec3(5i));
    c_6 = _e138;
    let _e139 = coord_21;
    let _e141 = textureSampleLevel(tex3D, samp, _e139, 3f);
    c_6 = _e141;
    let _e142 = coord_21;
    let _e146 = textureSampleLevel(tex3D, samp, _e142, 3f, vec3(5i));
    c_6 = _e146;
    let _e147 = coord_21;
    let _e150 = textureSample(tex3D, samp, _e147, vec3(5i));
    c_6 = _e150;
    let _e151 = coord_21;
    let _e155 = textureSampleBias(tex3D, samp, _e151, 2f, vec3(5i));
    c_6 = _e155;
    let _e156 = coord_21;
    let _e159 = textureLoad(tex3D, vec3<i32>(_e156), 3i);
    c_6 = _e159;
    let _e160 = coord_21;
    let _e163 = textureLoad(tex3D, vec3<i32>(_e160), 3i);
    c_6 = _e163;
    return;
}

fn testTex2DMS(coord_22: vec2<f32>) {
    var coord_23: vec2<f32>;
    var size2DMS: vec2<i32>;
    var c_7: vec4<f32>;

    coord_23 = coord_22;
    let _e3 = textureDimensions(tex2DMS);
    size2DMS = vec2<i32>(_e3);
    let _e7 = coord_23;
    let _e10 = textureLoad(tex2DMS, vec2<i32>(_e7), 3i);
    c_7 = _e10;
    return;
}

fn testTex2DMSArray(coord_24: vec3<f32>) {
    var coord_25: vec3<f32>;
    var size2DMSArray: vec3<i32>;
    var c_8: vec4<f32>;

    coord_25 = coord_24;
    let _e3 = textureDimensions(tex2DMSArray);
    let _e6 = textureNumLayers(tex2DMSArray);
    size2DMSArray = vec3<i32>(vec3<u32>(_e3.x, _e3.y, _e6));
    let _e11 = coord_25;
    let _e12 = vec3<i32>(_e11);
    let _e16 = textureLoad(tex2DMSArray, _e12.xy, _e12.z, 3i);
    c_8 = _e16;
    return;
}

fn main_1() {
    testTex1D(1f);
    testTex1DArray(vec2(3f));
    testTex2D(vec2(1f));
    testTex2DShadow(vec2(1f));
    testTex2DArray(vec3(1f));
    testTex2DArrayShadow(vec3(1f));
    testTexCube(vec3(1f));
    testTexCubeShadow(vec3(1f));
    testTexCubeArray(vec4(1f));
    testTexCubeArrayShadow(vec4(1f));
    testTex3D(vec3(1f));
    testTex2DMS(vec2(1f));
    testTex2DMSArray(vec3(1f));
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}
