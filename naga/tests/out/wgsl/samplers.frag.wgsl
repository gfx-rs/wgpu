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
    let _e21 = textureSampleGrad(tex1D, samp, _e18, 4f, 4f, 5i);
    c = _e21;
    let _e22 = coord_1;
    let _e24 = textureSampleLevel(tex1D, samp, _e22, 3f);
    c = _e24;
    let _e25 = coord_1;
    let _e27 = textureSampleLevel(tex1D, samp, _e25, 3f, 5i);
    c = _e27;
    let _e28 = coord_1;
    let _e29 = textureSample(tex1D, samp, _e28, 5i);
    c = _e29;
    let _e30 = coord_1;
    let _e32 = vec2<f32>(_e30, 6f);
    let _e36 = textureSample(tex1D, samp, (_e32.x / _e32.y));
    c = _e36;
    let _e37 = coord_1;
    let _e41 = vec4<f32>(_e37, 0f, 0f, 6f);
    let _e47 = textureSample(tex1D, samp, (_e41.xyz / vec3(_e41.w)).x);
    c = _e47;
    let _e48 = coord_1;
    let _e50 = vec2<f32>(_e48, 6f);
    let _e56 = textureSampleGrad(tex1D, samp, (_e50.x / _e50.y), 4f, 4f);
    c = _e56;
    let _e57 = coord_1;
    let _e61 = vec4<f32>(_e57, 0f, 0f, 6f);
    let _e69 = textureSampleGrad(tex1D, samp, (_e61.xyz / vec3(_e61.w)).x, 4f, 4f);
    c = _e69;
    let _e70 = coord_1;
    let _e72 = vec2<f32>(_e70, 6f);
    let _e78 = textureSampleGrad(tex1D, samp, (_e72.x / _e72.y), 4f, 4f, 5i);
    c = _e78;
    let _e79 = coord_1;
    let _e83 = vec4<f32>(_e79, 0f, 0f, 6f);
    let _e91 = textureSampleGrad(tex1D, samp, (_e83.xyz / vec3(_e83.w)).x, 4f, 4f, 5i);
    c = _e91;
    let _e92 = coord_1;
    let _e94 = vec2<f32>(_e92, 6f);
    let _e99 = textureSampleLevel(tex1D, samp, (_e94.x / _e94.y), 3f);
    c = _e99;
    let _e100 = coord_1;
    let _e104 = vec4<f32>(_e100, 0f, 0f, 6f);
    let _e111 = textureSampleLevel(tex1D, samp, (_e104.xyz / vec3(_e104.w)).x, 3f);
    c = _e111;
    let _e112 = coord_1;
    let _e114 = vec2<f32>(_e112, 6f);
    let _e119 = textureSampleLevel(tex1D, samp, (_e114.x / _e114.y), 3f, 5i);
    c = _e119;
    let _e120 = coord_1;
    let _e124 = vec4<f32>(_e120, 0f, 0f, 6f);
    let _e131 = textureSampleLevel(tex1D, samp, (_e124.xyz / vec3(_e124.w)).x, 3f, 5i);
    c = _e131;
    let _e132 = coord_1;
    let _e134 = vec2<f32>(_e132, 6f);
    let _e138 = textureSample(tex1D, samp, (_e134.x / _e134.y), 5i);
    c = _e138;
    let _e139 = coord_1;
    let _e143 = vec4<f32>(_e139, 0f, 0f, 6f);
    let _e149 = textureSample(tex1D, samp, (_e143.xyz / vec3(_e143.w)).x, 5i);
    c = _e149;
    let _e150 = coord_1;
    let _e153 = textureLoad(tex1D, i32(_e150), 3i);
    c = _e153;
    let _e154 = coord_1;
    let _e157 = textureLoad(tex1D, i32(_e154), 3i);
    c = _e157;
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
    let _e32 = textureSampleGrad(tex1DArray, samp, _e26.x, i32(_e26.y), 4f, 4f, 5i);
    c_1 = _e32;
    let _e33 = coord_3;
    let _e38 = textureSampleLevel(tex1DArray, samp, _e33.x, i32(_e33.y), 3f);
    c_1 = _e38;
    let _e39 = coord_3;
    let _e44 = textureSampleLevel(tex1DArray, samp, _e39.x, i32(_e39.y), 3f, 5i);
    c_1 = _e44;
    let _e45 = coord_3;
    let _e49 = textureSample(tex1DArray, samp, _e45.x, i32(_e45.y), 5i);
    c_1 = _e49;
    let _e50 = coord_3;
    let _e51 = vec2<i32>(_e50);
    let _e55 = textureLoad(tex1DArray, _e51.x, _e51.y, 3i);
    c_1 = _e55;
    let _e56 = coord_3;
    let _e57 = vec2<i32>(_e56);
    let _e61 = textureLoad(tex1DArray, _e57.x, _e57.y, 3i);
    c_1 = _e61;
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
    let _e30 = textureSampleGrad(tex2D, samp, _e25, vec2(4f), vec2(4f), vec2(5i));
    c_2 = _e30;
    let _e31 = coord_5;
    let _e33 = textureSampleLevel(tex2D, samp, _e31, 3f);
    c_2 = _e33;
    let _e34 = coord_5;
    let _e36 = textureSampleLevel(tex2D, samp, _e34, 3f, vec2(5i));
    c_2 = _e36;
    let _e37 = coord_5;
    let _e38 = textureSample(tex2D, samp, _e37, vec2(5i));
    c_2 = _e38;
    let _e39 = coord_5;
    let _e41 = textureSampleBias(tex2D, samp, _e39, 2f, vec2(5i));
    c_2 = _e41;
    let _e42 = coord_5;
    let _e46 = vec3<f32>(_e42.x, _e42.y, 6f);
    let _e51 = textureSample(tex2D, samp, (_e46.xy / vec2(_e46.z)));
    c_2 = _e51;
    let _e52 = coord_5;
    let _e57 = vec4<f32>(_e52.x, _e52.y, 0f, 6f);
    let _e63 = textureSample(tex2D, samp, (_e57.xyz / vec3(_e57.w)).xy);
    c_2 = _e63;
    let _e64 = coord_5;
    let _e68 = vec3<f32>(_e64.x, _e64.y, 6f);
    let _e74 = textureSampleBias(tex2D, samp, (_e68.xy / vec2(_e68.z)), 2f);
    c_2 = _e74;
    let _e75 = coord_5;
    let _e80 = vec4<f32>(_e75.x, _e75.y, 0f, 6f);
    let _e87 = textureSampleBias(tex2D, samp, (_e80.xyz / vec3(_e80.w)).xy, 2f);
    c_2 = _e87;
    let _e88 = coord_5;
    let _e92 = vec3<f32>(_e88.x, _e88.y, 6f);
    let _e101 = textureSampleGrad(tex2D, samp, (_e92.xy / vec2(_e92.z)), vec2(4f), vec2(4f));
    c_2 = _e101;
    let _e102 = coord_5;
    let _e107 = vec4<f32>(_e102.x, _e102.y, 0f, 6f);
    let _e117 = textureSampleGrad(tex2D, samp, (_e107.xyz / vec3(_e107.w)).xy, vec2(4f), vec2(4f));
    c_2 = _e117;
    let _e118 = coord_5;
    let _e122 = vec3<f32>(_e118.x, _e118.y, 6f);
    let _e131 = textureSampleGrad(tex2D, samp, (_e122.xy / vec2(_e122.z)), vec2(4f), vec2(4f), vec2(5i));
    c_2 = _e131;
    let _e132 = coord_5;
    let _e137 = vec4<f32>(_e132.x, _e132.y, 0f, 6f);
    let _e147 = textureSampleGrad(tex2D, samp, (_e137.xyz / vec3(_e137.w)).xy, vec2(4f), vec2(4f), vec2(5i));
    c_2 = _e147;
    let _e148 = coord_5;
    let _e152 = vec3<f32>(_e148.x, _e148.y, 6f);
    let _e158 = textureSampleLevel(tex2D, samp, (_e152.xy / vec2(_e152.z)), 3f);
    c_2 = _e158;
    let _e159 = coord_5;
    let _e164 = vec4<f32>(_e159.x, _e159.y, 0f, 6f);
    let _e171 = textureSampleLevel(tex2D, samp, (_e164.xyz / vec3(_e164.w)).xy, 3f);
    c_2 = _e171;
    let _e172 = coord_5;
    let _e176 = vec3<f32>(_e172.x, _e172.y, 6f);
    let _e182 = textureSampleLevel(tex2D, samp, (_e176.xy / vec2(_e176.z)), 3f, vec2(5i));
    c_2 = _e182;
    let _e183 = coord_5;
    let _e188 = vec4<f32>(_e183.x, _e183.y, 0f, 6f);
    let _e195 = textureSampleLevel(tex2D, samp, (_e188.xyz / vec3(_e188.w)).xy, 3f, vec2(5i));
    c_2 = _e195;
    let _e196 = coord_5;
    let _e200 = vec3<f32>(_e196.x, _e196.y, 6f);
    let _e205 = textureSample(tex2D, samp, (_e200.xy / vec2(_e200.z)), vec2(5i));
    c_2 = _e205;
    let _e206 = coord_5;
    let _e211 = vec4<f32>(_e206.x, _e206.y, 0f, 6f);
    let _e217 = textureSample(tex2D, samp, (_e211.xyz / vec3(_e211.w)).xy, vec2(5i));
    c_2 = _e217;
    let _e218 = coord_5;
    let _e222 = vec3<f32>(_e218.x, _e218.y, 6f);
    let _e228 = textureSampleBias(tex2D, samp, (_e222.xy / vec2(_e222.z)), 2f, vec2(5i));
    c_2 = _e228;
    let _e229 = coord_5;
    let _e234 = vec4<f32>(_e229.x, _e229.y, 0f, 6f);
    let _e241 = textureSampleBias(tex2D, samp, (_e234.xyz / vec3(_e234.w)).xy, 2f, vec2(5i));
    c_2 = _e241;
    let _e242 = coord_5;
    let _e245 = textureLoad(tex2D, vec2<i32>(_e242), 3i);
    c_2 = _e245;
    let _e246 = coord_5;
    let _e249 = textureLoad(utex2D, vec2<i32>(_e246), 3i);
    c_2 = vec4<f32>(_e249);
    let _e251 = coord_5;
    let _e254 = textureLoad(itex2D, vec2<i32>(_e251), 3i);
    c_2 = vec4<f32>(_e254);
    let _e256 = coord_5;
    let _e259 = textureLoad(tex2D, vec2<i32>(_e256), 3i);
    c_2 = _e259;
    let _e260 = coord_5;
    let _e263 = textureLoad(utex2D, vec2<i32>(_e260), 3i);
    c_2 = vec4<f32>(_e263);
    let _e265 = coord_5;
    let _e268 = textureLoad(itex2D, vec2<i32>(_e265), 3i);
    c_2 = vec4<f32>(_e268);
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
    let _e35 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e32.xy, _e32.z, vec2(5i));
    d = _e35;
    let _e36 = coord_7;
    let _e40 = vec3<f32>(_e36.x, _e36.y, 1f);
    let _e43 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e40.xy, _e40.z);
    d = _e43;
    let _e44 = coord_7;
    let _e48 = vec3<f32>(_e44.x, _e44.y, 1f);
    let _e51 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e48.xy, _e48.z, vec2(5i));
    d = _e51;
    let _e52 = coord_7;
    let _e56 = vec3<f32>(_e52.x, _e52.y, 1f);
    let _e59 = textureSampleCompare(tex2DShadow, sampShadow, _e56.xy, _e56.z, vec2(5i));
    d = _e59;
    let _e60 = coord_7;
    let _e65 = vec4<f32>(_e60.x, _e60.y, 1f, 6f);
    let _e69 = (_e65.xyz / vec3(_e65.w));
    let _e72 = textureSampleCompare(tex2DShadow, sampShadow, _e69.xy, _e69.z);
    d = _e72;
    let _e73 = coord_7;
    let _e78 = vec4<f32>(_e73.x, _e73.y, 1f, 6f);
    let _e82 = (_e78.xyz / vec3(_e78.w));
    let _e85 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e82.xy, _e82.z);
    d = _e85;
    let _e86 = coord_7;
    let _e91 = vec4<f32>(_e86.x, _e86.y, 1f, 6f);
    let _e95 = (_e91.xyz / vec3(_e91.w));
    let _e98 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e95.xy, _e95.z, vec2(5i));
    d = _e98;
    let _e99 = coord_7;
    let _e104 = vec4<f32>(_e99.x, _e99.y, 1f, 6f);
    let _e108 = (_e104.xyz / vec3(_e104.w));
    let _e111 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e108.xy, _e108.z);
    d = _e111;
    let _e112 = coord_7;
    let _e117 = vec4<f32>(_e112.x, _e112.y, 1f, 6f);
    let _e121 = (_e117.xyz / vec3(_e117.w));
    let _e124 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e121.xy, _e121.z, vec2(5i));
    d = _e124;
    let _e125 = coord_7;
    let _e130 = vec4<f32>(_e125.x, _e125.y, 1f, 6f);
    let _e134 = (_e130.xyz / vec3(_e130.w));
    let _e137 = textureSampleCompare(tex2DShadow, sampShadow, _e134.xy, _e134.z, vec2(5i));
    d = _e137;
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
    let _e44 = textureSampleGrad(tex2DArray, samp, _e36.xy, i32(_e36.z), vec2(4f), vec2(4f), vec2(5i));
    c_3 = _e44;
    let _e45 = coord_9;
    let _e50 = textureSampleLevel(tex2DArray, samp, _e45.xy, i32(_e45.z), 3f);
    c_3 = _e50;
    let _e51 = coord_9;
    let _e56 = textureSampleLevel(tex2DArray, samp, _e51.xy, i32(_e51.z), 3f, vec2(5i));
    c_3 = _e56;
    let _e57 = coord_9;
    let _e61 = textureSample(tex2DArray, samp, _e57.xy, i32(_e57.z), vec2(5i));
    c_3 = _e61;
    let _e62 = coord_9;
    let _e67 = textureSampleBias(tex2DArray, samp, _e62.xy, i32(_e62.z), 2f, vec2(5i));
    c_3 = _e67;
    let _e68 = coord_9;
    let _e69 = vec3<i32>(_e68);
    let _e73 = textureLoad(tex2DArray, _e69.xy, _e69.z, 3i);
    c_3 = _e73;
    let _e74 = coord_9;
    let _e75 = vec3<i32>(_e74);
    let _e79 = textureLoad(tex2DArray, _e75.xy, _e75.z, 3i);
    c_3 = _e79;
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
    let _e48 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e43.xy, i32(_e43.z), _e43.w, vec2(5i));
    d_1 = _e48;
    let _e49 = coord_11;
    let _e54 = vec4<f32>(_e49.x, _e49.y, _e49.z, 1f);
    let _e59 = textureSampleCompare(tex2DArrayShadow, sampShadow, _e54.xy, i32(_e54.z), _e54.w, vec2(5i));
    d_1 = _e59;
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
    let _e50 = textureSample(tex3D, samp, (_e45.xyz / vec3(_e45.w)), vec3(5i));
    c_6 = _e50;
    let _e51 = coord_21;
    let _e56 = vec4<f32>(_e51.x, _e51.y, _e51.z, 6f);
    let _e62 = textureSampleBias(tex3D, samp, (_e56.xyz / vec3(_e56.w)), 2f, vec3(5i));
    c_6 = _e62;
    let _e63 = coord_21;
    let _e68 = vec4<f32>(_e63.x, _e63.y, _e63.z, 6f);
    let _e74 = textureSampleLevel(tex3D, samp, (_e68.xyz / vec3(_e68.w)), 3f);
    c_6 = _e74;
    let _e75 = coord_21;
    let _e80 = vec4<f32>(_e75.x, _e75.y, _e75.z, 6f);
    let _e86 = textureSampleLevel(tex3D, samp, (_e80.xyz / vec3(_e80.w)), 3f, vec3(5i));
    c_6 = _e86;
    let _e87 = coord_21;
    let _e92 = vec4<f32>(_e87.x, _e87.y, _e87.z, 6f);
    let _e101 = textureSampleGrad(tex3D, samp, (_e92.xyz / vec3(_e92.w)), vec3(4f), vec3(4f));
    c_6 = _e101;
    let _e102 = coord_21;
    let _e107 = vec4<f32>(_e102.x, _e102.y, _e102.z, 6f);
    let _e116 = textureSampleGrad(tex3D, samp, (_e107.xyz / vec3(_e107.w)), vec3(4f), vec3(4f), vec3(5i));
    c_6 = _e116;
    let _e117 = coord_21;
    let _e122 = textureSampleGrad(tex3D, samp, _e117, vec3(4f), vec3(4f));
    c_6 = _e122;
    let _e123 = coord_21;
    let _e128 = textureSampleGrad(tex3D, samp, _e123, vec3(4f), vec3(4f), vec3(5i));
    c_6 = _e128;
    let _e129 = coord_21;
    let _e131 = textureSampleLevel(tex3D, samp, _e129, 3f);
    c_6 = _e131;
    let _e132 = coord_21;
    let _e134 = textureSampleLevel(tex3D, samp, _e132, 3f, vec3(5i));
    c_6 = _e134;
    let _e135 = coord_21;
    let _e136 = textureSample(tex3D, samp, _e135, vec3(5i));
    c_6 = _e136;
    let _e137 = coord_21;
    let _e139 = textureSampleBias(tex3D, samp, _e137, 2f, vec3(5i));
    c_6 = _e139;
    let _e140 = coord_21;
    let _e143 = textureLoad(tex3D, vec3<i32>(_e140), 3i);
    c_6 = _e143;
    let _e144 = coord_21;
    let _e147 = textureLoad(tex3D, vec3<i32>(_e144), 3i);
    c_6 = _e147;
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
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}
