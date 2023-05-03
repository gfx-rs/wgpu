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
@group(0) @binding(18) 
var tex2DMS: texture_multisampled_2d<f32>;
@group(0) @binding(19) 
var tex2DMSArray: texture_multisampled_2d_array<f32>;

fn testTex1D(coord: f32) {
    var coord_1: f32;
    var size1D: i32;
    var c: vec4<f32>;

    coord_1 = coord;
    let _e20 = textureDimensions(tex1D, 0);
    size1D = i32(_e20);
    let _e25 = coord_1;
    let _e26 = textureSample(tex1D, samp, _e25);
    c = _e26;
    let _e29 = coord_1;
    let _e31 = textureSampleBias(tex1D, samp, _e29, 2.0);
    c = _e31;
    let _e35 = coord_1;
    let _e38 = textureSampleGrad(tex1D, samp, _e35, 4.0, 4.0);
    c = _e38;
    let _e43 = coord_1;
    let _e47 = textureSampleGrad(tex1D, samp, _e43, 4.0, 4.0, 5);
    c = _e47;
    let _e50 = coord_1;
    let _e52 = textureSampleLevel(tex1D, samp, _e50, 3.0);
    c = _e52;
    let _e56 = coord_1;
    let _e59 = textureSampleLevel(tex1D, samp, _e56, 3.0, 5);
    c = _e59;
    let _e62 = coord_1;
    let _e64 = textureSample(tex1D, samp, _e62, 5);
    c = _e64;
    let _e68 = coord_1;
    let _e71 = textureSampleBias(tex1D, samp, _e68, 2.0, 5);
    c = _e71;
    let _e72 = coord_1;
    let _e75 = coord_1;
    let _e79 = textureSample(tex1D, samp, (_e75 / 6.0));
    c = _e79;
    let _e80 = coord_1;
    let _e85 = coord_1;
    let _e95 = textureSample(tex1D, samp, (vec3<f32>(_e85, 0.0, 0.0) / vec3(6.0)).x);
    c = _e95;
    let _e96 = coord_1;
    let _e100 = coord_1;
    let _e105 = textureSampleBias(tex1D, samp, (_e100 / 6.0), 2.0);
    c = _e105;
    let _e106 = coord_1;
    let _e112 = coord_1;
    let _e123 = textureSampleBias(tex1D, samp, (vec3<f32>(_e112, 0.0, 0.0) / vec3(6.0)).x, 2.0);
    c = _e123;
    let _e124 = coord_1;
    let _e129 = coord_1;
    let _e135 = textureSampleGrad(tex1D, samp, (_e129 / 6.0), 4.0, 4.0);
    c = _e135;
    let _e136 = coord_1;
    let _e143 = coord_1;
    let _e155 = textureSampleGrad(tex1D, samp, (vec3<f32>(_e143, 0.0, 0.0) / vec3(6.0)).x, 4.0, 4.0);
    c = _e155;
    let _e156 = coord_1;
    let _e162 = coord_1;
    let _e169 = textureSampleGrad(tex1D, samp, (_e162 / 6.0), 4.0, 4.0, 5);
    c = _e169;
    let _e170 = coord_1;
    let _e178 = coord_1;
    let _e191 = textureSampleGrad(tex1D, samp, (vec3<f32>(_e178, 0.0, 0.0) / vec3(6.0)).x, 4.0, 4.0, 5);
    c = _e191;
    let _e192 = coord_1;
    let _e196 = coord_1;
    let _e201 = textureSampleLevel(tex1D, samp, (_e196 / 6.0), 3.0);
    c = _e201;
    let _e202 = coord_1;
    let _e208 = coord_1;
    let _e219 = textureSampleLevel(tex1D, samp, (vec3<f32>(_e208, 0.0, 0.0) / vec3(6.0)).x, 3.0);
    c = _e219;
    let _e220 = coord_1;
    let _e225 = coord_1;
    let _e231 = textureSampleLevel(tex1D, samp, (_e225 / 6.0), 3.0, 5);
    c = _e231;
    let _e232 = coord_1;
    let _e239 = coord_1;
    let _e251 = textureSampleLevel(tex1D, samp, (vec3<f32>(_e239, 0.0, 0.0) / vec3(6.0)).x, 3.0, 5);
    c = _e251;
    let _e252 = coord_1;
    let _e256 = coord_1;
    let _e261 = textureSample(tex1D, samp, (_e256 / 6.0), 5);
    c = _e261;
    let _e262 = coord_1;
    let _e268 = coord_1;
    let _e279 = textureSample(tex1D, samp, (vec3<f32>(_e268, 0.0, 0.0) / vec3(6.0)).x, 5);
    c = _e279;
    let _e280 = coord_1;
    let _e285 = coord_1;
    let _e291 = textureSampleBias(tex1D, samp, (_e285 / 6.0), 2.0, 5);
    c = _e291;
    let _e292 = coord_1;
    let _e299 = coord_1;
    let _e311 = textureSampleBias(tex1D, samp, (vec3<f32>(_e299, 0.0, 0.0) / vec3(6.0)).x, 2.0, 5);
    c = _e311;
    let _e312 = coord_1;
    let _e315 = coord_1;
    let _e318 = textureLoad(tex1D, i32(_e315), 3);
    c = _e318;
    let _e319 = coord_1;
    let _e323 = coord_1;
    let _e327 = textureLoad(tex1D, i32(_e323), 3);
    c = _e327;
    return;
}

fn testTex1DArray(coord_2: vec2<f32>) {
    var coord_3: vec2<f32>;
    var size1DArray: vec2<i32>;
    var c_1: vec4<f32>;

    coord_3 = coord_2;
    let _e20 = textureDimensions(tex1DArray, 0);
    let _e21 = textureNumLayers(tex1DArray);
    size1DArray = vec2<i32>(vec2<u32>(_e20, _e21));
    let _e27 = coord_3;
    let _e31 = textureSample(tex1DArray, samp, _e27.x, i32(_e27.y));
    c_1 = _e31;
    let _e34 = coord_3;
    let _e39 = textureSampleBias(tex1DArray, samp, _e34.x, i32(_e34.y), 2.0);
    c_1 = _e39;
    let _e43 = coord_3;
    let _e49 = textureSampleGrad(tex1DArray, samp, _e43.x, i32(_e43.y), 4.0, 4.0);
    c_1 = _e49;
    let _e54 = coord_3;
    let _e61 = textureSampleGrad(tex1DArray, samp, _e54.x, i32(_e54.y), 4.0, 4.0, 5);
    c_1 = _e61;
    let _e64 = coord_3;
    let _e69 = textureSampleLevel(tex1DArray, samp, _e64.x, i32(_e64.y), 3.0);
    c_1 = _e69;
    let _e73 = coord_3;
    let _e79 = textureSampleLevel(tex1DArray, samp, _e73.x, i32(_e73.y), 3.0, 5);
    c_1 = _e79;
    let _e82 = coord_3;
    let _e87 = textureSample(tex1DArray, samp, _e82.x, i32(_e82.y), 5);
    c_1 = _e87;
    let _e91 = coord_3;
    let _e97 = textureSampleBias(tex1DArray, samp, _e91.x, i32(_e91.y), 2.0, 5);
    c_1 = _e97;
    let _e98 = coord_3;
    let _e101 = coord_3;
    let _e102 = vec2<i32>(_e101);
    let _e106 = textureLoad(tex1DArray, _e102.x, _e102.y, 3);
    c_1 = _e106;
    let _e107 = coord_3;
    let _e111 = coord_3;
    let _e112 = vec2<i32>(_e111);
    let _e117 = textureLoad(tex1DArray, _e112.x, _e112.y, 3);
    c_1 = _e117;
    return;
}

fn testTex2D(coord_4: vec2<f32>) {
    var coord_5: vec2<f32>;
    var size2D: vec2<i32>;
    var c_2: vec4<f32>;

    coord_5 = coord_4;
    let _e20 = textureDimensions(tex2D, 0);
    size2D = vec2<i32>(_e20);
    let _e25 = coord_5;
    let _e26 = textureSample(tex2D, samp, _e25);
    c_2 = _e26;
    let _e29 = coord_5;
    let _e31 = textureSampleBias(tex2D, samp, _e29, 2.0);
    c_2 = _e31;
    let _e37 = coord_5;
    let _e42 = textureSampleGrad(tex2D, samp, _e37, vec2(4.0), vec2(4.0));
    c_2 = _e42;
    let _e50 = coord_5;
    let _e57 = textureSampleGrad(tex2D, samp, _e50, vec2(4.0), vec2(4.0), vec2(5));
    c_2 = _e57;
    let _e60 = coord_5;
    let _e62 = textureSampleLevel(tex2D, samp, _e60, 3.0);
    c_2 = _e62;
    let _e67 = coord_5;
    let _e71 = textureSampleLevel(tex2D, samp, _e67, 3.0, vec2(5));
    c_2 = _e71;
    let _e75 = coord_5;
    let _e78 = textureSample(tex2D, samp, _e75, vec2(5));
    c_2 = _e78;
    let _e83 = coord_5;
    let _e87 = textureSampleBias(tex2D, samp, _e83, 2.0, vec2(5));
    c_2 = _e87;
    let _e88 = coord_5;
    let _e93 = coord_5;
    let _e102 = textureSample(tex2D, samp, (vec2<f32>(_e93.x, _e93.y) / vec2(6.0)));
    c_2 = _e102;
    let _e103 = coord_5;
    let _e109 = coord_5;
    let _e120 = textureSample(tex2D, samp, (vec3<f32>(_e109.x, _e109.y, 0.0) / vec3(6.0)).xy);
    c_2 = _e120;
    let _e121 = coord_5;
    let _e127 = coord_5;
    let _e137 = textureSampleBias(tex2D, samp, (vec2<f32>(_e127.x, _e127.y) / vec2(6.0)), 2.0);
    c_2 = _e137;
    let _e138 = coord_5;
    let _e145 = coord_5;
    let _e157 = textureSampleBias(tex2D, samp, (vec3<f32>(_e145.x, _e145.y, 0.0) / vec3(6.0)).xy, 2.0);
    c_2 = _e157;
    let _e158 = coord_5;
    let _e167 = coord_5;
    let _e180 = textureSampleGrad(tex2D, samp, (vec2<f32>(_e167.x, _e167.y) / vec2(6.0)), vec2(4.0), vec2(4.0));
    c_2 = _e180;
    let _e181 = coord_5;
    let _e191 = coord_5;
    let _e206 = textureSampleGrad(tex2D, samp, (vec3<f32>(_e191.x, _e191.y, 0.0) / vec3(6.0)).xy, vec2(4.0), vec2(4.0));
    c_2 = _e206;
    let _e207 = coord_5;
    let _e218 = coord_5;
    let _e233 = textureSampleGrad(tex2D, samp, (vec2<f32>(_e218.x, _e218.y) / vec2(6.0)), vec2(4.0), vec2(4.0), vec2(5));
    c_2 = _e233;
    let _e234 = coord_5;
    let _e246 = coord_5;
    let _e263 = textureSampleGrad(tex2D, samp, (vec3<f32>(_e246.x, _e246.y, 0.0) / vec3(6.0)).xy, vec2(4.0), vec2(4.0), vec2(5));
    c_2 = _e263;
    let _e264 = coord_5;
    let _e270 = coord_5;
    let _e280 = textureSampleLevel(tex2D, samp, (vec2<f32>(_e270.x, _e270.y) / vec2(6.0)), 3.0);
    c_2 = _e280;
    let _e281 = coord_5;
    let _e288 = coord_5;
    let _e300 = textureSampleLevel(tex2D, samp, (vec3<f32>(_e288.x, _e288.y, 0.0) / vec3(6.0)).xy, 3.0);
    c_2 = _e300;
    let _e301 = coord_5;
    let _e309 = coord_5;
    let _e321 = textureSampleLevel(tex2D, samp, (vec2<f32>(_e309.x, _e309.y) / vec2(6.0)), 3.0, vec2(5));
    c_2 = _e321;
    let _e322 = coord_5;
    let _e331 = coord_5;
    let _e345 = textureSampleLevel(tex2D, samp, (vec3<f32>(_e331.x, _e331.y, 0.0) / vec3(6.0)).xy, 3.0, vec2(5));
    c_2 = _e345;
    let _e346 = coord_5;
    let _e353 = coord_5;
    let _e364 = textureSample(tex2D, samp, (vec2<f32>(_e353.x, _e353.y) / vec2(6.0)), vec2(5));
    c_2 = _e364;
    let _e365 = coord_5;
    let _e373 = coord_5;
    let _e386 = textureSample(tex2D, samp, (vec3<f32>(_e373.x, _e373.y, 0.0) / vec3(6.0)).xy, vec2(5));
    c_2 = _e386;
    let _e387 = coord_5;
    let _e395 = coord_5;
    let _e407 = textureSampleBias(tex2D, samp, (vec2<f32>(_e395.x, _e395.y) / vec2(6.0)), 2.0, vec2(5));
    c_2 = _e407;
    let _e408 = coord_5;
    let _e417 = coord_5;
    let _e431 = textureSampleBias(tex2D, samp, (vec3<f32>(_e417.x, _e417.y, 0.0) / vec3(6.0)).xy, 2.0, vec2(5));
    c_2 = _e431;
    let _e432 = coord_5;
    let _e435 = coord_5;
    let _e438 = textureLoad(tex2D, vec2<i32>(_e435), 3);
    c_2 = _e438;
    let _e439 = coord_5;
    let _e444 = coord_5;
    let _e449 = textureLoad(tex2D, vec2<i32>(_e444), 3);
    c_2 = _e449;
    return;
}

fn testTex2DShadow(coord_6: vec2<f32>) {
    var coord_7: vec2<f32>;
    var size2DShadow: vec2<i32>;
    var d: f32;

    coord_7 = coord_6;
    let _e20 = textureDimensions(tex2DShadow, 0);
    size2DShadow = vec2<i32>(_e20);
    let _e24 = coord_7;
    let _e29 = coord_7;
    let _e35 = textureSampleCompare(tex2DShadow, sampShadow, vec2<f32>(_e29.x, _e29.y), 1.0);
    d = _e35;
    let _e36 = coord_7;
    let _e45 = coord_7;
    let _e55 = textureSampleCompareLevel(tex2DShadow, sampShadow, vec2<f32>(_e45.x, _e45.y), 1.0);
    d = _e55;
    let _e56 = coord_7;
    let _e67 = coord_7;
    let _e79 = textureSampleCompareLevel(tex2DShadow, sampShadow, vec2<f32>(_e67.x, _e67.y), 1.0, vec2(5));
    d = _e79;
    let _e80 = coord_7;
    let _e86 = coord_7;
    let _e93 = textureSampleCompareLevel(tex2DShadow, sampShadow, vec2<f32>(_e86.x, _e86.y), 1.0);
    d = _e93;
    let _e94 = coord_7;
    let _e102 = coord_7;
    let _e111 = textureSampleCompareLevel(tex2DShadow, sampShadow, vec2<f32>(_e102.x, _e102.y), 1.0, vec2(5));
    d = _e111;
    let _e112 = coord_7;
    let _e119 = coord_7;
    let _e127 = textureSampleCompare(tex2DShadow, sampShadow, vec2<f32>(_e119.x, _e119.y), 1.0, vec2(5));
    d = _e127;
    let _e128 = coord_7;
    let _e134 = coord_7;
    let _e143 = (vec3<f32>(_e134.x, _e134.y, 1.0) / vec3(6.0));
    let _e146 = textureSampleCompare(tex2DShadow, sampShadow, _e143.xy, _e143.z);
    d = _e146;
    let _e147 = coord_7;
    let _e157 = coord_7;
    let _e170 = (vec3<f32>(_e157.x, _e157.y, 1.0) / vec3(6.0));
    let _e173 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e170.xy, _e170.z);
    d = _e173;
    let _e174 = coord_7;
    let _e186 = coord_7;
    let _e201 = (vec3<f32>(_e186.x, _e186.y, 1.0) / vec3(6.0));
    let _e204 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e201.xy, _e201.z, vec2(5));
    d = _e204;
    let _e205 = coord_7;
    let _e212 = coord_7;
    let _e222 = (vec3<f32>(_e212.x, _e212.y, 1.0) / vec3(6.0));
    let _e225 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e222.xy, _e222.z);
    d = _e225;
    let _e226 = coord_7;
    let _e235 = coord_7;
    let _e247 = (vec3<f32>(_e235.x, _e235.y, 1.0) / vec3(6.0));
    let _e250 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e247.xy, _e247.z, vec2(5));
    d = _e250;
    let _e251 = coord_7;
    let _e259 = coord_7;
    let _e270 = (vec3<f32>(_e259.x, _e259.y, 1.0) / vec3(6.0));
    let _e273 = textureSampleCompare(tex2DShadow, sampShadow, _e270.xy, _e270.z, vec2(5));
    d = _e273;
    return;
}

fn testTex2DArray(coord_8: vec3<f32>) {
    var coord_9: vec3<f32>;
    var size2DArray: vec3<i32>;
    var c_3: vec4<f32>;

    coord_9 = coord_8;
    let _e20 = textureDimensions(tex2DArray, 0);
    let _e23 = textureNumLayers(tex2DArray);
    size2DArray = vec3<i32>(vec3<u32>(_e20.x, _e20.y, _e23));
    let _e29 = coord_9;
    let _e33 = textureSample(tex2DArray, samp, _e29.xy, i32(_e29.z));
    c_3 = _e33;
    let _e36 = coord_9;
    let _e41 = textureSampleBias(tex2DArray, samp, _e36.xy, i32(_e36.z), 2.0);
    c_3 = _e41;
    let _e47 = coord_9;
    let _e55 = textureSampleGrad(tex2DArray, samp, _e47.xy, i32(_e47.z), vec2(4.0), vec2(4.0));
    c_3 = _e55;
    let _e63 = coord_9;
    let _e73 = textureSampleGrad(tex2DArray, samp, _e63.xy, i32(_e63.z), vec2(4.0), vec2(4.0), vec2(5));
    c_3 = _e73;
    let _e76 = coord_9;
    let _e81 = textureSampleLevel(tex2DArray, samp, _e76.xy, i32(_e76.z), 3.0);
    c_3 = _e81;
    let _e86 = coord_9;
    let _e93 = textureSampleLevel(tex2DArray, samp, _e86.xy, i32(_e86.z), 3.0, vec2(5));
    c_3 = _e93;
    let _e97 = coord_9;
    let _e103 = textureSample(tex2DArray, samp, _e97.xy, i32(_e97.z), vec2(5));
    c_3 = _e103;
    let _e108 = coord_9;
    let _e115 = textureSampleBias(tex2DArray, samp, _e108.xy, i32(_e108.z), 2.0, vec2(5));
    c_3 = _e115;
    let _e116 = coord_9;
    let _e119 = coord_9;
    let _e120 = vec3<i32>(_e119);
    let _e124 = textureLoad(tex2DArray, _e120.xy, _e120.z, 3);
    c_3 = _e124;
    let _e125 = coord_9;
    let _e130 = coord_9;
    let _e131 = vec3<i32>(_e130);
    let _e137 = textureLoad(tex2DArray, _e131.xy, _e131.z, 3);
    c_3 = _e137;
    return;
}

fn testTex2DArrayShadow(coord_10: vec3<f32>) {
    var coord_11: vec3<f32>;
    var size2DArrayShadow: vec3<i32>;
    var d_1: f32;

    coord_11 = coord_10;
    let _e20 = textureDimensions(tex2DArrayShadow, 0);
    let _e23 = textureNumLayers(tex2DArrayShadow);
    size2DArrayShadow = vec3<i32>(vec3<u32>(_e20.x, _e20.y, _e23));
    let _e28 = coord_11;
    let _e34 = coord_11;
    let _e42 = textureSampleCompare(tex2DArrayShadow, sampShadow, vec2<f32>(_e34.x, _e34.y), i32(_e34.z), 1.0);
    d_1 = _e42;
    let _e43 = coord_11;
    let _e53 = coord_11;
    let _e65 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, vec2<f32>(_e53.x, _e53.y), i32(_e53.z), 1.0);
    d_1 = _e65;
    let _e66 = coord_11;
    let _e78 = coord_11;
    let _e92 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, vec2<f32>(_e78.x, _e78.y), i32(_e78.z), 1.0, vec2(5));
    d_1 = _e92;
    let _e93 = coord_11;
    let _e101 = coord_11;
    let _e111 = textureSampleCompare(tex2DArrayShadow, sampShadow, vec2<f32>(_e101.x, _e101.y), i32(_e101.z), 1.0, vec2(5));
    d_1 = _e111;
    return;
}

fn testTexCube(coord_12: vec3<f32>) {
    var coord_13: vec3<f32>;
    var sizeCube: vec2<i32>;
    var c_4: vec4<f32>;

    coord_13 = coord_12;
    let _e20 = textureDimensions(texCube, 0);
    sizeCube = vec2<i32>(_e20);
    let _e25 = coord_13;
    let _e26 = textureSample(texCube, samp, _e25);
    c_4 = _e26;
    let _e29 = coord_13;
    let _e31 = textureSampleBias(texCube, samp, _e29, 2.0);
    c_4 = _e31;
    let _e37 = coord_13;
    let _e42 = textureSampleGrad(texCube, samp, _e37, vec3(4.0), vec3(4.0));
    c_4 = _e42;
    let _e45 = coord_13;
    let _e47 = textureSampleLevel(texCube, samp, _e45, 3.0);
    c_4 = _e47;
    return;
}

fn testTexCubeShadow(coord_14: vec3<f32>) {
    var coord_15: vec3<f32>;
    var sizeCubeShadow: vec2<i32>;
    var d_2: f32;

    coord_15 = coord_14;
    let _e20 = textureDimensions(texCubeShadow, 0);
    sizeCubeShadow = vec2<i32>(_e20);
    let _e24 = coord_15;
    let _e30 = coord_15;
    let _e37 = textureSampleCompare(texCubeShadow, sampShadow, vec3<f32>(_e30.x, _e30.y, _e30.z), 1.0);
    d_2 = _e37;
    let _e38 = coord_15;
    let _e48 = coord_15;
    let _e59 = textureSampleCompareLevel(texCubeShadow, sampShadow, vec3<f32>(_e48.x, _e48.y, _e48.z), 1.0);
    d_2 = _e59;
    return;
}

fn testTexCubeArray(coord_16: vec4<f32>) {
    var coord_17: vec4<f32>;
    var sizeCubeArray: vec3<i32>;
    var c_5: vec4<f32>;

    coord_17 = coord_16;
    let _e20 = textureDimensions(texCubeArray, 0);
    let _e23 = textureNumLayers(texCubeArray);
    sizeCubeArray = vec3<i32>(vec3<u32>(_e20.x, _e20.y, _e23));
    let _e29 = coord_17;
    let _e33 = textureSample(texCubeArray, samp, _e29.xyz, i32(_e29.w));
    c_5 = _e33;
    let _e36 = coord_17;
    let _e41 = textureSampleBias(texCubeArray, samp, _e36.xyz, i32(_e36.w), 2.0);
    c_5 = _e41;
    let _e47 = coord_17;
    let _e55 = textureSampleGrad(texCubeArray, samp, _e47.xyz, i32(_e47.w), vec3(4.0), vec3(4.0));
    c_5 = _e55;
    let _e58 = coord_17;
    let _e63 = textureSampleLevel(texCubeArray, samp, _e58.xyz, i32(_e58.w), 3.0);
    c_5 = _e63;
    return;
}

fn testTexCubeArrayShadow(coord_18: vec4<f32>) {
    var coord_19: vec4<f32>;
    var sizeCubeArrayShadow: vec3<i32>;
    var d_3: f32;

    coord_19 = coord_18;
    let _e20 = textureDimensions(texCubeArrayShadow, 0);
    let _e23 = textureNumLayers(texCubeArrayShadow);
    sizeCubeArrayShadow = vec3<i32>(vec3<u32>(_e20.x, _e20.y, _e23));
    let _e30 = coord_19;
    let _e35 = textureSampleCompare(texCubeArrayShadow, sampShadow, _e30.xyz, i32(_e30.w), 1.0);
    d_3 = _e35;
    return;
}

fn testTex3D(coord_20: vec3<f32>) {
    var coord_21: vec3<f32>;
    var size3D: vec3<i32>;
    var c_6: vec4<f32>;

    coord_21 = coord_20;
    let _e20 = textureDimensions(tex3D, 0);
    size3D = vec3<i32>(_e20);
    let _e25 = coord_21;
    let _e26 = textureSample(tex3D, samp, _e25);
    c_6 = _e26;
    let _e29 = coord_21;
    let _e31 = textureSampleBias(tex3D, samp, _e29, 2.0);
    c_6 = _e31;
    let _e32 = coord_21;
    let _e38 = coord_21;
    let _e48 = textureSample(tex3D, samp, (vec3<f32>(_e38.x, _e38.y, _e38.z) / vec3(6.0)));
    c_6 = _e48;
    let _e49 = coord_21;
    let _e56 = coord_21;
    let _e67 = textureSampleBias(tex3D, samp, (vec3<f32>(_e56.x, _e56.y, _e56.z) / vec3(6.0)), 2.0);
    c_6 = _e67;
    let _e68 = coord_21;
    let _e76 = coord_21;
    let _e88 = textureSample(tex3D, samp, (vec3<f32>(_e76.x, _e76.y, _e76.z) / vec3(6.0)), vec3(5));
    c_6 = _e88;
    let _e89 = coord_21;
    let _e98 = coord_21;
    let _e111 = textureSampleBias(tex3D, samp, (vec3<f32>(_e98.x, _e98.y, _e98.z) / vec3(6.0)), 2.0, vec3(5));
    c_6 = _e111;
    let _e112 = coord_21;
    let _e119 = coord_21;
    let _e130 = textureSampleLevel(tex3D, samp, (vec3<f32>(_e119.x, _e119.y, _e119.z) / vec3(6.0)), 3.0);
    c_6 = _e130;
    let _e131 = coord_21;
    let _e140 = coord_21;
    let _e153 = textureSampleLevel(tex3D, samp, (vec3<f32>(_e140.x, _e140.y, _e140.z) / vec3(6.0)), 3.0, vec3(5));
    c_6 = _e153;
    let _e154 = coord_21;
    let _e164 = coord_21;
    let _e178 = textureSampleGrad(tex3D, samp, (vec3<f32>(_e164.x, _e164.y, _e164.z) / vec3(6.0)), vec3(4.0), vec3(4.0));
    c_6 = _e178;
    let _e179 = coord_21;
    let _e191 = coord_21;
    let _e207 = textureSampleGrad(tex3D, samp, (vec3<f32>(_e191.x, _e191.y, _e191.z) / vec3(6.0)), vec3(4.0), vec3(4.0), vec3(5));
    c_6 = _e207;
    let _e213 = coord_21;
    let _e218 = textureSampleGrad(tex3D, samp, _e213, vec3(4.0), vec3(4.0));
    c_6 = _e218;
    let _e226 = coord_21;
    let _e233 = textureSampleGrad(tex3D, samp, _e226, vec3(4.0), vec3(4.0), vec3(5));
    c_6 = _e233;
    let _e236 = coord_21;
    let _e238 = textureSampleLevel(tex3D, samp, _e236, 3.0);
    c_6 = _e238;
    let _e243 = coord_21;
    let _e247 = textureSampleLevel(tex3D, samp, _e243, 3.0, vec3(5));
    c_6 = _e247;
    let _e251 = coord_21;
    let _e254 = textureSample(tex3D, samp, _e251, vec3(5));
    c_6 = _e254;
    let _e259 = coord_21;
    let _e263 = textureSampleBias(tex3D, samp, _e259, 2.0, vec3(5));
    c_6 = _e263;
    let _e264 = coord_21;
    let _e267 = coord_21;
    let _e270 = textureLoad(tex3D, vec3<i32>(_e267), 3);
    c_6 = _e270;
    let _e271 = coord_21;
    let _e276 = coord_21;
    let _e281 = textureLoad(tex3D, vec3<i32>(_e276), 3);
    c_6 = _e281;
    return;
}

fn testTex2DMS(coord_22: vec2<f32>) {
    var coord_23: vec2<f32>;
    var size2DMS: vec2<i32>;
    var c_7: vec4<f32>;

    coord_23 = coord_22;
    let _e18 = textureDimensions(tex2DMS);
    size2DMS = vec2<i32>(_e18);
    let _e22 = coord_23;
    let _e25 = coord_23;
    let _e28 = textureLoad(tex2DMS, vec2<i32>(_e25), 3);
    c_7 = _e28;
    return;
}

fn testTex2DMSArray(coord_24: vec3<f32>) {
    var coord_25: vec3<f32>;
    var size2DMSArray: vec3<i32>;
    var c_8: vec4<f32>;

    coord_25 = coord_24;
    let _e18 = textureDimensions(tex2DMSArray);
    let _e21 = textureNumLayers(tex2DMSArray);
    size2DMSArray = vec3<i32>(vec3<u32>(_e18.x, _e18.y, _e21));
    let _e26 = coord_25;
    let _e29 = coord_25;
    let _e30 = vec3<i32>(_e29);
    let _e34 = textureLoad(tex2DMSArray, _e30.xy, _e30.z, 3);
    c_8 = _e34;
    return;
}

fn main_1() {
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}
