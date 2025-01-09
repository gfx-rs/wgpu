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
    var levels: i32;
    var c: vec4<f32>;

    coord_1 = coord;
    let _e21 = textureDimensions(tex1D, 0i);
    size1D = i32(_e21);
    let _e24 = textureNumLevels(tex1D);
    levels = i32(_e24);
    let _e28 = coord_1;
    let _e29 = textureSample(tex1D, samp, _e28);
    c = _e29;
    let _e30 = coord_1;
    let _e33 = textureSampleGrad(tex1D, samp, _e30, 4f, 4f);
    c = _e33;
    let _e34 = coord_1;
    let _e38 = textureSampleGrad(tex1D, samp, _e34, 4f, 4f, 5i);
    c = _e38;
    let _e39 = coord_1;
    let _e41 = textureSampleLevel(tex1D, samp, _e39, 3f);
    c = _e41;
    let _e42 = coord_1;
    let _e45 = textureSampleLevel(tex1D, samp, _e42, 3f, 5i);
    c = _e45;
    let _e46 = coord_1;
    let _e48 = textureSample(tex1D, samp, _e46, 5i);
    c = _e48;
    let _e49 = coord_1;
    let _e51 = vec2<f32>(_e49, 6f);
    let _e55 = textureSample(tex1D, samp, (_e51.x / _e51.y));
    c = _e55;
    let _e56 = coord_1;
    let _e60 = vec4<f32>(_e56, 0f, 0f, 6f);
    let _e66 = textureSample(tex1D, samp, (_e60.xyz / vec3(_e60.w)).x);
    c = _e66;
    let _e67 = coord_1;
    let _e69 = vec2<f32>(_e67, 6f);
    let _e75 = textureSampleGrad(tex1D, samp, (_e69.x / _e69.y), 4f, 4f);
    c = _e75;
    let _e76 = coord_1;
    let _e80 = vec4<f32>(_e76, 0f, 0f, 6f);
    let _e88 = textureSampleGrad(tex1D, samp, (_e80.xyz / vec3(_e80.w)).x, 4f, 4f);
    c = _e88;
    let _e89 = coord_1;
    let _e91 = vec2<f32>(_e89, 6f);
    let _e98 = textureSampleGrad(tex1D, samp, (_e91.x / _e91.y), 4f, 4f, 5i);
    c = _e98;
    let _e99 = coord_1;
    let _e103 = vec4<f32>(_e99, 0f, 0f, 6f);
    let _e112 = textureSampleGrad(tex1D, samp, (_e103.xyz / vec3(_e103.w)).x, 4f, 4f, 5i);
    c = _e112;
    let _e113 = coord_1;
    let _e115 = vec2<f32>(_e113, 6f);
    let _e120 = textureSampleLevel(tex1D, samp, (_e115.x / _e115.y), 3f);
    c = _e120;
    let _e121 = coord_1;
    let _e125 = vec4<f32>(_e121, 0f, 0f, 6f);
    let _e132 = textureSampleLevel(tex1D, samp, (_e125.xyz / vec3(_e125.w)).x, 3f);
    c = _e132;
    let _e133 = coord_1;
    let _e135 = vec2<f32>(_e133, 6f);
    let _e141 = textureSampleLevel(tex1D, samp, (_e135.x / _e135.y), 3f, 5i);
    c = _e141;
    let _e142 = coord_1;
    let _e146 = vec4<f32>(_e142, 0f, 0f, 6f);
    let _e154 = textureSampleLevel(tex1D, samp, (_e146.xyz / vec3(_e146.w)).x, 3f, 5i);
    c = _e154;
    let _e155 = coord_1;
    let _e157 = vec2<f32>(_e155, 6f);
    let _e162 = textureSample(tex1D, samp, (_e157.x / _e157.y), 5i);
    c = _e162;
    let _e163 = coord_1;
    let _e167 = vec4<f32>(_e163, 0f, 0f, 6f);
    let _e174 = textureSample(tex1D, samp, (_e167.xyz / vec3(_e167.w)).x, 5i);
    c = _e174;
    let _e175 = coord_1;
    let _e178 = textureLoad(tex1D, i32(_e175), 3i);
    c = _e178;
    let _e179 = coord_1;
    let _e183 = textureLoad(tex1D, i32(_e179), 3i);
    c = _e183;
    return;
}

fn testTex1DArray(coord_2: vec2<f32>) {
    var coord_3: vec2<f32>;
    var size1DArray: vec2<i32>;
    var levels_1: i32;
    var c_1: vec4<f32>;

    coord_3 = coord_2;
    let _e21 = textureDimensions(tex1DArray, 0i);
    let _e22 = textureNumLayers(tex1DArray);
    size1DArray = vec2<i32>(vec2<u32>(_e21, _e22));
    let _e26 = textureNumLevels(tex1DArray);
    levels_1 = i32(_e26);
    let _e30 = coord_3;
    let _e34 = textureSample(tex1DArray, samp, _e30.x, i32(_e30.y));
    c_1 = _e34;
    let _e35 = coord_3;
    let _e41 = textureSampleGrad(tex1DArray, samp, _e35.x, i32(_e35.y), 4f, 4f);
    c_1 = _e41;
    let _e42 = coord_3;
    let _e49 = textureSampleGrad(tex1DArray, samp, _e42.x, i32(_e42.y), 4f, 4f, 5i);
    c_1 = _e49;
    let _e50 = coord_3;
    let _e55 = textureSampleLevel(tex1DArray, samp, _e50.x, i32(_e50.y), 3f);
    c_1 = _e55;
    let _e56 = coord_3;
    let _e62 = textureSampleLevel(tex1DArray, samp, _e56.x, i32(_e56.y), 3f, 5i);
    c_1 = _e62;
    let _e63 = coord_3;
    let _e68 = textureSample(tex1DArray, samp, _e63.x, i32(_e63.y), 5i);
    c_1 = _e68;
    let _e69 = coord_3;
    let _e70 = vec2<i32>(_e69);
    let _e74 = textureLoad(tex1DArray, _e70.x, _e70.y, 3i);
    c_1 = _e74;
    let _e75 = coord_3;
    let _e76 = vec2<i32>(_e75);
    let _e81 = textureLoad(tex1DArray, _e76.x, _e76.y, 3i);
    c_1 = _e81;
    return;
}

fn testTex2D(coord_4: vec2<f32>) {
    var coord_5: vec2<f32>;
    var size2D: vec2<i32>;
    var levels_2: i32;
    var c_2: vec4<f32>;

    coord_5 = coord_4;
    let _e21 = textureDimensions(tex2D, 0i);
    size2D = vec2<i32>(_e21);
    let _e24 = textureNumLevels(tex2D);
    levels_2 = i32(_e24);
    let _e28 = coord_5;
    let _e29 = textureSample(tex2D, samp, _e28);
    c_2 = _e29;
    let _e30 = coord_5;
    let _e32 = textureSampleBias(tex2D, samp, _e30, 2f);
    c_2 = _e32;
    let _e33 = coord_5;
    let _e38 = textureSampleGrad(tex2D, samp, _e33, vec2(4f), vec2(4f));
    c_2 = _e38;
    let _e39 = coord_5;
    let _e46 = textureSampleGrad(tex2D, samp, _e39, vec2(4f), vec2(4f), vec2(5i));
    c_2 = _e46;
    let _e47 = coord_5;
    let _e49 = textureSampleLevel(tex2D, samp, _e47, 3f);
    c_2 = _e49;
    let _e50 = coord_5;
    let _e54 = textureSampleLevel(tex2D, samp, _e50, 3f, vec2(5i));
    c_2 = _e54;
    let _e55 = coord_5;
    let _e58 = textureSample(tex2D, samp, _e55, vec2(5i));
    c_2 = _e58;
    let _e59 = coord_5;
    let _e63 = textureSampleBias(tex2D, samp, _e59, 2f, vec2(5i));
    c_2 = _e63;
    let _e64 = coord_5;
    let _e68 = vec3<f32>(_e64.x, _e64.y, 6f);
    let _e73 = textureSample(tex2D, samp, (_e68.xy / vec2(_e68.z)));
    c_2 = _e73;
    let _e74 = coord_5;
    let _e79 = vec4<f32>(_e74.x, _e74.y, 0f, 6f);
    let _e85 = textureSample(tex2D, samp, (_e79.xyz / vec3(_e79.w)).xy);
    c_2 = _e85;
    let _e86 = coord_5;
    let _e90 = vec3<f32>(_e86.x, _e86.y, 6f);
    let _e96 = textureSampleBias(tex2D, samp, (_e90.xy / vec2(_e90.z)), 2f);
    c_2 = _e96;
    let _e97 = coord_5;
    let _e102 = vec4<f32>(_e97.x, _e97.y, 0f, 6f);
    let _e109 = textureSampleBias(tex2D, samp, (_e102.xyz / vec3(_e102.w)).xy, 2f);
    c_2 = _e109;
    let _e110 = coord_5;
    let _e114 = vec3<f32>(_e110.x, _e110.y, 6f);
    let _e123 = textureSampleGrad(tex2D, samp, (_e114.xy / vec2(_e114.z)), vec2(4f), vec2(4f));
    c_2 = _e123;
    let _e124 = coord_5;
    let _e129 = vec4<f32>(_e124.x, _e124.y, 0f, 6f);
    let _e139 = textureSampleGrad(tex2D, samp, (_e129.xyz / vec3(_e129.w)).xy, vec2(4f), vec2(4f));
    c_2 = _e139;
    let _e140 = coord_5;
    let _e144 = vec3<f32>(_e140.x, _e140.y, 6f);
    let _e155 = textureSampleGrad(tex2D, samp, (_e144.xy / vec2(_e144.z)), vec2(4f), vec2(4f), vec2(5i));
    c_2 = _e155;
    let _e156 = coord_5;
    let _e161 = vec4<f32>(_e156.x, _e156.y, 0f, 6f);
    let _e173 = textureSampleGrad(tex2D, samp, (_e161.xyz / vec3(_e161.w)).xy, vec2(4f), vec2(4f), vec2(5i));
    c_2 = _e173;
    let _e174 = coord_5;
    let _e178 = vec3<f32>(_e174.x, _e174.y, 6f);
    let _e184 = textureSampleLevel(tex2D, samp, (_e178.xy / vec2(_e178.z)), 3f);
    c_2 = _e184;
    let _e185 = coord_5;
    let _e190 = vec4<f32>(_e185.x, _e185.y, 0f, 6f);
    let _e197 = textureSampleLevel(tex2D, samp, (_e190.xyz / vec3(_e190.w)).xy, 3f);
    c_2 = _e197;
    let _e198 = coord_5;
    let _e202 = vec3<f32>(_e198.x, _e198.y, 6f);
    let _e210 = textureSampleLevel(tex2D, samp, (_e202.xy / vec2(_e202.z)), 3f, vec2(5i));
    c_2 = _e210;
    let _e211 = coord_5;
    let _e216 = vec4<f32>(_e211.x, _e211.y, 0f, 6f);
    let _e225 = textureSampleLevel(tex2D, samp, (_e216.xyz / vec3(_e216.w)).xy, 3f, vec2(5i));
    c_2 = _e225;
    let _e226 = coord_5;
    let _e230 = vec3<f32>(_e226.x, _e226.y, 6f);
    let _e237 = textureSample(tex2D, samp, (_e230.xy / vec2(_e230.z)), vec2(5i));
    c_2 = _e237;
    let _e238 = coord_5;
    let _e243 = vec4<f32>(_e238.x, _e238.y, 0f, 6f);
    let _e251 = textureSample(tex2D, samp, (_e243.xyz / vec3(_e243.w)).xy, vec2(5i));
    c_2 = _e251;
    let _e252 = coord_5;
    let _e256 = vec3<f32>(_e252.x, _e252.y, 6f);
    let _e264 = textureSampleBias(tex2D, samp, (_e256.xy / vec2(_e256.z)), 2f, vec2(5i));
    c_2 = _e264;
    let _e265 = coord_5;
    let _e270 = vec4<f32>(_e265.x, _e265.y, 0f, 6f);
    let _e279 = textureSampleBias(tex2D, samp, (_e270.xyz / vec3(_e270.w)).xy, 2f, vec2(5i));
    c_2 = _e279;
    let _e280 = coord_5;
    let _e283 = textureLoad(tex2D, vec2<i32>(_e280), 3i);
    c_2 = _e283;
    let _e284 = coord_5;
    let _e287 = textureLoad(utex2D, vec2<i32>(_e284), 3i);
    c_2 = vec4<f32>(_e287);
    let _e289 = coord_5;
    let _e292 = textureLoad(itex2D, vec2<i32>(_e289), 3i);
    c_2 = vec4<f32>(_e292);
    let _e294 = coord_5;
    let _e299 = textureLoad(tex2D, vec2<i32>(_e294), 3i);
    c_2 = _e299;
    let _e300 = coord_5;
    let _e305 = textureLoad(utex2D, vec2<i32>(_e300), 3i);
    c_2 = vec4<f32>(_e305);
    let _e307 = coord_5;
    let _e312 = textureLoad(itex2D, vec2<i32>(_e307), 3i);
    c_2 = vec4<f32>(_e312);
    return;
}

fn testTex2DShadow(coord_6: vec2<f32>) {
    var coord_7: vec2<f32>;
    var size2DShadow: vec2<i32>;
    var levels_3: i32;
    var d: f32;

    coord_7 = coord_6;
    let _e21 = textureDimensions(tex2DShadow, 0i);
    size2DShadow = vec2<i32>(_e21);
    let _e24 = textureNumLevels(tex2DShadow);
    levels_3 = i32(_e24);
    let _e28 = coord_7;
    let _e32 = vec3<f32>(_e28.x, _e28.y, 1f);
    let _e35 = textureSampleCompare(tex2DShadow, sampShadow, _e32.xy, _e32.z);
    d = _e35;
    let _e36 = coord_7;
    let _e40 = vec3<f32>(_e36.x, _e36.y, 1f);
    let _e47 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e40.xy, _e40.z);
    d = _e47;
    let _e48 = coord_7;
    let _e52 = vec3<f32>(_e48.x, _e48.y, 1f);
    let _e61 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e52.xy, _e52.z, vec2(5i));
    d = _e61;
    let _e62 = coord_7;
    let _e66 = vec3<f32>(_e62.x, _e62.y, 1f);
    let _e70 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e66.xy, _e66.z);
    d = _e70;
    let _e71 = coord_7;
    let _e75 = vec3<f32>(_e71.x, _e71.y, 1f);
    let _e81 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e75.xy, _e75.z, vec2(5i));
    d = _e81;
    let _e82 = coord_7;
    let _e86 = vec3<f32>(_e82.x, _e82.y, 1f);
    let _e91 = textureSampleCompare(tex2DShadow, sampShadow, _e86.xy, _e86.z, vec2(5i));
    d = _e91;
    let _e92 = coord_7;
    let _e97 = vec4<f32>(_e92.x, _e92.y, 1f, 6f);
    let _e101 = (_e97.xyz / vec3(_e97.w));
    let _e104 = textureSampleCompare(tex2DShadow, sampShadow, _e101.xy, _e101.z);
    d = _e104;
    let _e105 = coord_7;
    let _e110 = vec4<f32>(_e105.x, _e105.y, 1f, 6f);
    let _e118 = (_e110.xyz / vec3(_e110.w));
    let _e121 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e118.xy, _e118.z);
    d = _e121;
    let _e122 = coord_7;
    let _e127 = vec4<f32>(_e122.x, _e122.y, 1f, 6f);
    let _e137 = (_e127.xyz / vec3(_e127.w));
    let _e140 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e137.xy, _e137.z, vec2(5i));
    d = _e140;
    let _e141 = coord_7;
    let _e146 = vec4<f32>(_e141.x, _e141.y, 1f, 6f);
    let _e151 = (_e146.xyz / vec3(_e146.w));
    let _e154 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e151.xy, _e151.z);
    d = _e154;
    let _e155 = coord_7;
    let _e160 = vec4<f32>(_e155.x, _e155.y, 1f, 6f);
    let _e167 = (_e160.xyz / vec3(_e160.w));
    let _e170 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e167.xy, _e167.z, vec2(5i));
    d = _e170;
    let _e171 = coord_7;
    let _e176 = vec4<f32>(_e171.x, _e171.y, 1f, 6f);
    let _e182 = (_e176.xyz / vec3(_e176.w));
    let _e185 = textureSampleCompare(tex2DShadow, sampShadow, _e182.xy, _e182.z, vec2(5i));
    d = _e185;
    return;
}

fn testTex2DArray(coord_8: vec3<f32>) {
    var coord_9: vec3<f32>;
    var size2DArray: vec3<i32>;
    var levels_4: i32;
    var c_3: vec4<f32>;

    coord_9 = coord_8;
    let _e21 = textureDimensions(tex2DArray, 0i);
    let _e24 = textureNumLayers(tex2DArray);
    size2DArray = vec3<i32>(vec3<u32>(_e21.x, _e21.y, _e24));
    let _e28 = textureNumLevels(tex2DArray);
    levels_4 = i32(_e28);
    let _e32 = coord_9;
    let _e36 = textureSample(tex2DArray, samp, _e32.xy, i32(_e32.z));
    c_3 = _e36;
    let _e37 = coord_9;
    let _e42 = textureSampleBias(tex2DArray, samp, _e37.xy, i32(_e37.z), 2f);
    c_3 = _e42;
    let _e43 = coord_9;
    let _e51 = textureSampleGrad(tex2DArray, samp, _e43.xy, i32(_e43.z), vec2(4f), vec2(4f));
    c_3 = _e51;
    let _e52 = coord_9;
    let _e62 = textureSampleGrad(tex2DArray, samp, _e52.xy, i32(_e52.z), vec2(4f), vec2(4f), vec2(5i));
    c_3 = _e62;
    let _e63 = coord_9;
    let _e68 = textureSampleLevel(tex2DArray, samp, _e63.xy, i32(_e63.z), 3f);
    c_3 = _e68;
    let _e69 = coord_9;
    let _e76 = textureSampleLevel(tex2DArray, samp, _e69.xy, i32(_e69.z), 3f, vec2(5i));
    c_3 = _e76;
    let _e77 = coord_9;
    let _e83 = textureSample(tex2DArray, samp, _e77.xy, i32(_e77.z), vec2(5i));
    c_3 = _e83;
    let _e84 = coord_9;
    let _e91 = textureSampleBias(tex2DArray, samp, _e84.xy, i32(_e84.z), 2f, vec2(5i));
    c_3 = _e91;
    let _e92 = coord_9;
    let _e93 = vec3<i32>(_e92);
    let _e97 = textureLoad(tex2DArray, _e93.xy, _e93.z, 3i);
    c_3 = _e97;
    let _e98 = coord_9;
    let _e99 = vec3<i32>(_e98);
    let _e105 = textureLoad(tex2DArray, _e99.xy, _e99.z, 3i);
    c_3 = _e105;
    return;
}

fn testTex2DArrayShadow(coord_10: vec3<f32>) {
    var coord_11: vec3<f32>;
    var size2DArrayShadow: vec3<i32>;
    var levels_5: i32;
    var d_1: f32;

    coord_11 = coord_10;
    let _e21 = textureDimensions(tex2DArrayShadow, 0i);
    let _e24 = textureNumLayers(tex2DArrayShadow);
    size2DArrayShadow = vec3<i32>(vec3<u32>(_e21.x, _e21.y, _e24));
    let _e28 = textureNumLevels(tex2DArrayShadow);
    levels_5 = i32(_e28);
    let _e32 = coord_11;
    let _e37 = vec4<f32>(_e32.x, _e32.y, _e32.z, 1f);
    let _e42 = textureSampleCompare(tex2DArrayShadow, sampShadow, _e37.xy, i32(_e37.z), _e37.w);
    d_1 = _e42;
    let _e43 = coord_11;
    let _e48 = vec4<f32>(_e43.x, _e43.y, _e43.z, 1f);
    let _e57 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e48.xy, i32(_e48.z), _e48.w);
    d_1 = _e57;
    let _e58 = coord_11;
    let _e63 = vec4<f32>(_e58.x, _e58.y, _e58.z, 1f);
    let _e74 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e63.xy, i32(_e63.z), _e63.w, vec2(5i));
    d_1 = _e74;
    let _e75 = coord_11;
    let _e80 = vec4<f32>(_e75.x, _e75.y, _e75.z, 1f);
    let _e87 = textureSampleCompare(tex2DArrayShadow, sampShadow, _e80.xy, i32(_e80.z), _e80.w, vec2(5i));
    d_1 = _e87;
    return;
}

fn testTexCube(coord_12: vec3<f32>) {
    var coord_13: vec3<f32>;
    var sizeCube: vec2<i32>;
    var levels_6: i32;
    var c_4: vec4<f32>;

    coord_13 = coord_12;
    let _e21 = textureDimensions(texCube, 0i);
    sizeCube = vec2<i32>(_e21);
    let _e24 = textureNumLevels(texCube);
    levels_6 = i32(_e24);
    let _e28 = coord_13;
    let _e29 = textureSample(texCube, samp, _e28);
    c_4 = _e29;
    let _e30 = coord_13;
    let _e32 = textureSampleBias(texCube, samp, _e30, 2f);
    c_4 = _e32;
    let _e33 = coord_13;
    let _e38 = textureSampleGrad(texCube, samp, _e33, vec3(4f), vec3(4f));
    c_4 = _e38;
    let _e39 = coord_13;
    let _e41 = textureSampleLevel(texCube, samp, _e39, 3f);
    c_4 = _e41;
    return;
}

fn testTexCubeShadow(coord_14: vec3<f32>) {
    var coord_15: vec3<f32>;
    var sizeCubeShadow: vec2<i32>;
    var levels_7: i32;
    var d_2: f32;

    coord_15 = coord_14;
    let _e21 = textureDimensions(texCubeShadow, 0i);
    sizeCubeShadow = vec2<i32>(_e21);
    let _e24 = textureNumLevels(texCubeShadow);
    levels_7 = i32(_e24);
    let _e28 = coord_15;
    let _e33 = vec4<f32>(_e28.x, _e28.y, _e28.z, 1f);
    let _e36 = textureSampleCompare(texCubeShadow, sampShadow, _e33.xyz, _e33.w);
    d_2 = _e36;
    let _e37 = coord_15;
    let _e42 = vec4<f32>(_e37.x, _e37.y, _e37.z, 1f);
    let _e49 = textureSampleCompareLevel(texCubeShadow, sampShadow, _e42.xyz, _e42.w);
    d_2 = _e49;
    return;
}

fn testTexCubeArray(coord_16: vec4<f32>) {
    var coord_17: vec4<f32>;
    var sizeCubeArray: vec3<i32>;
    var levels_8: i32;
    var c_5: vec4<f32>;

    coord_17 = coord_16;
    let _e21 = textureDimensions(texCubeArray, 0i);
    let _e24 = textureNumLayers(texCubeArray);
    sizeCubeArray = vec3<i32>(vec3<u32>(_e21.x, _e21.y, _e24));
    let _e28 = textureNumLevels(texCubeArray);
    levels_8 = i32(_e28);
    let _e32 = coord_17;
    let _e36 = textureSample(texCubeArray, samp, _e32.xyz, i32(_e32.w));
    c_5 = _e36;
    let _e37 = coord_17;
    let _e42 = textureSampleBias(texCubeArray, samp, _e37.xyz, i32(_e37.w), 2f);
    c_5 = _e42;
    let _e43 = coord_17;
    let _e51 = textureSampleGrad(texCubeArray, samp, _e43.xyz, i32(_e43.w), vec3(4f), vec3(4f));
    c_5 = _e51;
    let _e52 = coord_17;
    let _e57 = textureSampleLevel(texCubeArray, samp, _e52.xyz, i32(_e52.w), 3f);
    c_5 = _e57;
    return;
}

fn testTexCubeArrayShadow(coord_18: vec4<f32>) {
    var coord_19: vec4<f32>;
    var sizeCubeArrayShadow: vec3<i32>;
    var levels_9: i32;
    var d_3: f32;

    coord_19 = coord_18;
    let _e21 = textureDimensions(texCubeArrayShadow, 0i);
    let _e24 = textureNumLayers(texCubeArrayShadow);
    sizeCubeArrayShadow = vec3<i32>(vec3<u32>(_e21.x, _e21.y, _e24));
    let _e28 = textureNumLevels(texCubeArrayShadow);
    levels_9 = i32(_e28);
    let _e32 = coord_19;
    let _e37 = textureSampleCompare(texCubeArrayShadow, sampShadow, _e32.xyz, i32(_e32.w), 1f);
    d_3 = _e37;
    return;
}

fn testTex3D(coord_20: vec3<f32>) {
    var coord_21: vec3<f32>;
    var size3D: vec3<i32>;
    var levels_10: i32;
    var c_6: vec4<f32>;

    coord_21 = coord_20;
    let _e21 = textureDimensions(tex3D, 0i);
    size3D = vec3<i32>(_e21);
    let _e24 = textureNumLevels(tex3D);
    levels_10 = i32(_e24);
    let _e28 = coord_21;
    let _e29 = textureSample(tex3D, samp, _e28);
    c_6 = _e29;
    let _e30 = coord_21;
    let _e32 = textureSampleBias(tex3D, samp, _e30, 2f);
    c_6 = _e32;
    let _e33 = coord_21;
    let _e38 = vec4<f32>(_e33.x, _e33.y, _e33.z, 6f);
    let _e43 = textureSample(tex3D, samp, (_e38.xyz / vec3(_e38.w)));
    c_6 = _e43;
    let _e44 = coord_21;
    let _e49 = vec4<f32>(_e44.x, _e44.y, _e44.z, 6f);
    let _e55 = textureSampleBias(tex3D, samp, (_e49.xyz / vec3(_e49.w)), 2f);
    c_6 = _e55;
    let _e56 = coord_21;
    let _e61 = vec4<f32>(_e56.x, _e56.y, _e56.z, 6f);
    let _e68 = textureSample(tex3D, samp, (_e61.xyz / vec3(_e61.w)), vec3(5i));
    c_6 = _e68;
    let _e69 = coord_21;
    let _e74 = vec4<f32>(_e69.x, _e69.y, _e69.z, 6f);
    let _e82 = textureSampleBias(tex3D, samp, (_e74.xyz / vec3(_e74.w)), 2f, vec3(5i));
    c_6 = _e82;
    let _e83 = coord_21;
    let _e88 = vec4<f32>(_e83.x, _e83.y, _e83.z, 6f);
    let _e94 = textureSampleLevel(tex3D, samp, (_e88.xyz / vec3(_e88.w)), 3f);
    c_6 = _e94;
    let _e95 = coord_21;
    let _e100 = vec4<f32>(_e95.x, _e95.y, _e95.z, 6f);
    let _e108 = textureSampleLevel(tex3D, samp, (_e100.xyz / vec3(_e100.w)), 3f, vec3(5i));
    c_6 = _e108;
    let _e109 = coord_21;
    let _e114 = vec4<f32>(_e109.x, _e109.y, _e109.z, 6f);
    let _e123 = textureSampleGrad(tex3D, samp, (_e114.xyz / vec3(_e114.w)), vec3(4f), vec3(4f));
    c_6 = _e123;
    let _e124 = coord_21;
    let _e129 = vec4<f32>(_e124.x, _e124.y, _e124.z, 6f);
    let _e140 = textureSampleGrad(tex3D, samp, (_e129.xyz / vec3(_e129.w)), vec3(4f), vec3(4f), vec3(5i));
    c_6 = _e140;
    let _e141 = coord_21;
    let _e146 = textureSampleGrad(tex3D, samp, _e141, vec3(4f), vec3(4f));
    c_6 = _e146;
    let _e147 = coord_21;
    let _e154 = textureSampleGrad(tex3D, samp, _e147, vec3(4f), vec3(4f), vec3(5i));
    c_6 = _e154;
    let _e155 = coord_21;
    let _e157 = textureSampleLevel(tex3D, samp, _e155, 3f);
    c_6 = _e157;
    let _e158 = coord_21;
    let _e162 = textureSampleLevel(tex3D, samp, _e158, 3f, vec3(5i));
    c_6 = _e162;
    let _e163 = coord_21;
    let _e166 = textureSample(tex3D, samp, _e163, vec3(5i));
    c_6 = _e166;
    let _e167 = coord_21;
    let _e171 = textureSampleBias(tex3D, samp, _e167, 2f, vec3(5i));
    c_6 = _e171;
    let _e172 = coord_21;
    let _e175 = textureLoad(tex3D, vec3<i32>(_e172), 3i);
    c_6 = _e175;
    let _e176 = coord_21;
    let _e181 = textureLoad(tex3D, vec3<i32>(_e176), 3i);
    c_6 = _e181;
    return;
}

fn testTex2DMS(coord_22: vec2<f32>) {
    var coord_23: vec2<f32>;
    var size2DMS: vec2<i32>;
    var c_7: vec4<f32>;

    coord_23 = coord_22;
    let _e20 = textureDimensions(tex2DMS);
    size2DMS = vec2<i32>(_e20);
    let _e24 = coord_23;
    let _e27 = textureLoad(tex2DMS, vec2<i32>(_e24), 3i);
    c_7 = _e27;
    return;
}

fn testTex2DMSArray(coord_24: vec3<f32>) {
    var coord_25: vec3<f32>;
    var size2DMSArray: vec3<i32>;
    var c_8: vec4<f32>;

    coord_25 = coord_24;
    let _e20 = textureDimensions(tex2DMSArray);
    let _e23 = textureNumLayers(tex2DMSArray);
    size2DMSArray = vec3<i32>(vec3<u32>(_e20.x, _e20.y, _e23));
    let _e28 = coord_25;
    let _e29 = vec3<i32>(_e28);
    let _e33 = textureLoad(tex2DMSArray, _e29.xy, _e29.z, 3i);
    c_8 = _e33;
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
