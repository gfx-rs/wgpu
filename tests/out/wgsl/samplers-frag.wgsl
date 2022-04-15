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
    size1D = _e20;
    let _e24 = coord_1;
    let _e25 = textureSample(tex1D, samp, _e24);
    c = _e25;
    let _e28 = coord_1;
    let _e30 = textureSampleBias(tex1D, samp, _e28, 2.0);
    c = _e30;
    let _e34 = coord_1;
    let _e37 = textureSampleGrad(tex1D, samp, _e34, 4.0, 4.0);
    c = _e37;
    let _e42 = coord_1;
    let _e46 = textureSampleGrad(tex1D, samp, _e42, 4.0, 4.0, 5);
    c = _e46;
    let _e49 = coord_1;
    let _e51 = textureSampleLevel(tex1D, samp, _e49, 3.0);
    c = _e51;
    let _e55 = coord_1;
    let _e58 = textureSampleLevel(tex1D, samp, _e55, 3.0, 5);
    c = _e58;
    let _e61 = coord_1;
    let _e63 = textureSample(tex1D, samp, _e61, 5);
    c = _e63;
    let _e67 = coord_1;
    let _e70 = textureSampleBias(tex1D, samp, _e67, 2.0, 5);
    c = _e70;
    let _e71 = coord_1;
    let _e74 = coord_1;
    let _e76 = vec2<f32>(_e74, 6.0);
    let _e80 = textureSample(tex1D, samp, (_e76.x / _e76.y));
    c = _e80;
    let _e81 = coord_1;
    let _e86 = coord_1;
    let _e90 = vec4<f32>(_e86, 0.0, 0.0, 6.0);
    let _e96 = textureSample(tex1D, samp, (_e90.xyz / vec3<f32>(_e90.w)).x);
    c = _e96;
    let _e97 = coord_1;
    let _e101 = coord_1;
    let _e103 = vec2<f32>(_e101, 6.0);
    let _e108 = textureSampleBias(tex1D, samp, (_e103.x / _e103.y), 2.0);
    c = _e108;
    let _e109 = coord_1;
    let _e115 = coord_1;
    let _e119 = vec4<f32>(_e115, 0.0, 0.0, 6.0);
    let _e126 = textureSampleBias(tex1D, samp, (_e119.xyz / vec3<f32>(_e119.w)).x, 2.0);
    c = _e126;
    let _e127 = coord_1;
    let _e132 = coord_1;
    let _e134 = vec2<f32>(_e132, 6.0);
    let _e140 = textureSampleGrad(tex1D, samp, (_e134.x / _e134.y), 4.0, 4.0);
    c = _e140;
    let _e141 = coord_1;
    let _e148 = coord_1;
    let _e152 = vec4<f32>(_e148, 0.0, 0.0, 6.0);
    let _e160 = textureSampleGrad(tex1D, samp, (_e152.xyz / vec3<f32>(_e152.w)).x, 4.0, 4.0);
    c = _e160;
    let _e161 = coord_1;
    let _e167 = coord_1;
    let _e169 = vec2<f32>(_e167, 6.0);
    let _e176 = textureSampleGrad(tex1D, samp, (_e169.x / _e169.y), 4.0, 4.0, 5);
    c = _e176;
    let _e177 = coord_1;
    let _e185 = coord_1;
    let _e189 = vec4<f32>(_e185, 0.0, 0.0, 6.0);
    let _e198 = textureSampleGrad(tex1D, samp, (_e189.xyz / vec3<f32>(_e189.w)).x, 4.0, 4.0, 5);
    c = _e198;
    let _e199 = coord_1;
    let _e203 = coord_1;
    let _e205 = vec2<f32>(_e203, 6.0);
    let _e210 = textureSampleLevel(tex1D, samp, (_e205.x / _e205.y), 3.0);
    c = _e210;
    let _e211 = coord_1;
    let _e217 = coord_1;
    let _e221 = vec4<f32>(_e217, 0.0, 0.0, 6.0);
    let _e228 = textureSampleLevel(tex1D, samp, (_e221.xyz / vec3<f32>(_e221.w)).x, 3.0);
    c = _e228;
    let _e229 = coord_1;
    let _e234 = coord_1;
    let _e236 = vec2<f32>(_e234, 6.0);
    let _e242 = textureSampleLevel(tex1D, samp, (_e236.x / _e236.y), 3.0, 5);
    c = _e242;
    let _e243 = coord_1;
    let _e250 = coord_1;
    let _e254 = vec4<f32>(_e250, 0.0, 0.0, 6.0);
    let _e262 = textureSampleLevel(tex1D, samp, (_e254.xyz / vec3<f32>(_e254.w)).x, 3.0, 5);
    c = _e262;
    let _e263 = coord_1;
    let _e267 = coord_1;
    let _e269 = vec2<f32>(_e267, 6.0);
    let _e274 = textureSample(tex1D, samp, (_e269.x / _e269.y), 5);
    c = _e274;
    let _e275 = coord_1;
    let _e281 = coord_1;
    let _e285 = vec4<f32>(_e281, 0.0, 0.0, 6.0);
    let _e292 = textureSample(tex1D, samp, (_e285.xyz / vec3<f32>(_e285.w)).x, 5);
    c = _e292;
    let _e293 = coord_1;
    let _e298 = coord_1;
    let _e300 = vec2<f32>(_e298, 6.0);
    let _e306 = textureSampleBias(tex1D, samp, (_e300.x / _e300.y), 2.0, 5);
    c = _e306;
    let _e307 = coord_1;
    let _e314 = coord_1;
    let _e318 = vec4<f32>(_e314, 0.0, 0.0, 6.0);
    let _e326 = textureSampleBias(tex1D, samp, (_e318.xyz / vec3<f32>(_e318.w)).x, 2.0, 5);
    c = _e326;
    let _e327 = coord_1;
    let _e330 = coord_1;
    let _e333 = textureLoad(tex1D, i32(_e330), 3);
    c = _e333;
    let _e334 = coord_1;
    let _e338 = coord_1;
    let _e342 = textureLoad(tex1D, i32(_e338), 3);
    c = _e342;
    return;
}

fn testTex1DArray(coord_2: vec2<f32>) {
    var coord_3: vec2<f32>;
    var size1DArray: vec2<i32>;
    var c_1: vec4<f32>;

    coord_3 = coord_2;
    let _e20 = textureDimensions(tex1DArray, 0);
    let _e21 = textureNumLayers(tex1DArray);
    size1DArray = vec2<i32>(_e20, _e21);
    let _e26 = coord_3;
    let _e30 = textureSample(tex1DArray, samp, _e26.x, i32(_e26.y));
    c_1 = _e30;
    let _e33 = coord_3;
    let _e38 = textureSampleBias(tex1DArray, samp, _e33.x, i32(_e33.y), 2.0);
    c_1 = _e38;
    let _e42 = coord_3;
    let _e48 = textureSampleGrad(tex1DArray, samp, _e42.x, i32(_e42.y), 4.0, 4.0);
    c_1 = _e48;
    let _e53 = coord_3;
    let _e60 = textureSampleGrad(tex1DArray, samp, _e53.x, i32(_e53.y), 4.0, 4.0, 5);
    c_1 = _e60;
    let _e63 = coord_3;
    let _e68 = textureSampleLevel(tex1DArray, samp, _e63.x, i32(_e63.y), 3.0);
    c_1 = _e68;
    let _e72 = coord_3;
    let _e78 = textureSampleLevel(tex1DArray, samp, _e72.x, i32(_e72.y), 3.0, 5);
    c_1 = _e78;
    let _e81 = coord_3;
    let _e86 = textureSample(tex1DArray, samp, _e81.x, i32(_e81.y), 5);
    c_1 = _e86;
    let _e90 = coord_3;
    let _e96 = textureSampleBias(tex1DArray, samp, _e90.x, i32(_e90.y), 2.0, 5);
    c_1 = _e96;
    let _e97 = coord_3;
    let _e100 = coord_3;
    let _e101 = vec2<i32>(_e100);
    let _e105 = textureLoad(tex1DArray, _e101.x, _e101.y, 3);
    c_1 = _e105;
    let _e106 = coord_3;
    let _e110 = coord_3;
    let _e111 = vec2<i32>(_e110);
    let _e116 = textureLoad(tex1DArray, _e111.x, _e111.y, 3);
    c_1 = _e116;
    return;
}

fn testTex2D(coord_4: vec2<f32>) {
    var coord_5: vec2<f32>;
    var size2D: vec2<i32>;
    var c_2: vec4<f32>;

    coord_5 = coord_4;
    let _e20 = textureDimensions(tex2D, 0);
    size2D = _e20;
    let _e24 = coord_5;
    let _e25 = textureSample(tex2D, samp, _e24);
    c_2 = _e25;
    let _e28 = coord_5;
    let _e30 = textureSampleBias(tex2D, samp, _e28, 2.0);
    c_2 = _e30;
    let _e36 = coord_5;
    let _e41 = textureSampleGrad(tex2D, samp, _e36, vec2<f32>(4.0), vec2<f32>(4.0));
    c_2 = _e41;
    let _e49 = coord_5;
    let _e56 = textureSampleGrad(tex2D, samp, _e49, vec2<f32>(4.0), vec2<f32>(4.0), vec2<i32>(5, 5));
    c_2 = _e56;
    let _e59 = coord_5;
    let _e61 = textureSampleLevel(tex2D, samp, _e59, 3.0);
    c_2 = _e61;
    let _e66 = coord_5;
    let _e70 = textureSampleLevel(tex2D, samp, _e66, 3.0, vec2<i32>(5, 5));
    c_2 = _e70;
    let _e74 = coord_5;
    let _e77 = textureSample(tex2D, samp, _e74, vec2<i32>(5, 5));
    c_2 = _e77;
    let _e82 = coord_5;
    let _e86 = textureSampleBias(tex2D, samp, _e82, 2.0, vec2<i32>(5, 5));
    c_2 = _e86;
    let _e87 = coord_5;
    let _e92 = coord_5;
    let _e96 = vec3<f32>(_e92.x, _e92.y, 6.0);
    let _e101 = textureSample(tex2D, samp, (_e96.xy / vec2<f32>(_e96.z)));
    c_2 = _e101;
    let _e102 = coord_5;
    let _e108 = coord_5;
    let _e113 = vec4<f32>(_e108.x, _e108.y, 0.0, 6.0);
    let _e119 = textureSample(tex2D, samp, (_e113.xyz / vec3<f32>(_e113.w)).xy);
    c_2 = _e119;
    let _e120 = coord_5;
    let _e126 = coord_5;
    let _e130 = vec3<f32>(_e126.x, _e126.y, 6.0);
    let _e136 = textureSampleBias(tex2D, samp, (_e130.xy / vec2<f32>(_e130.z)), 2.0);
    c_2 = _e136;
    let _e137 = coord_5;
    let _e144 = coord_5;
    let _e149 = vec4<f32>(_e144.x, _e144.y, 0.0, 6.0);
    let _e156 = textureSampleBias(tex2D, samp, (_e149.xyz / vec3<f32>(_e149.w)).xy, 2.0);
    c_2 = _e156;
    let _e157 = coord_5;
    let _e166 = coord_5;
    let _e170 = vec3<f32>(_e166.x, _e166.y, 6.0);
    let _e179 = textureSampleGrad(tex2D, samp, (_e170.xy / vec2<f32>(_e170.z)), vec2<f32>(4.0), vec2<f32>(4.0));
    c_2 = _e179;
    let _e180 = coord_5;
    let _e190 = coord_5;
    let _e195 = vec4<f32>(_e190.x, _e190.y, 0.0, 6.0);
    let _e205 = textureSampleGrad(tex2D, samp, (_e195.xyz / vec3<f32>(_e195.w)).xy, vec2<f32>(4.0), vec2<f32>(4.0));
    c_2 = _e205;
    let _e206 = coord_5;
    let _e217 = coord_5;
    let _e221 = vec3<f32>(_e217.x, _e217.y, 6.0);
    let _e232 = textureSampleGrad(tex2D, samp, (_e221.xy / vec2<f32>(_e221.z)), vec2<f32>(4.0), vec2<f32>(4.0), vec2<i32>(5, 5));
    c_2 = _e232;
    let _e233 = coord_5;
    let _e245 = coord_5;
    let _e250 = vec4<f32>(_e245.x, _e245.y, 0.0, 6.0);
    let _e262 = textureSampleGrad(tex2D, samp, (_e250.xyz / vec3<f32>(_e250.w)).xy, vec2<f32>(4.0), vec2<f32>(4.0), vec2<i32>(5, 5));
    c_2 = _e262;
    let _e263 = coord_5;
    let _e269 = coord_5;
    let _e273 = vec3<f32>(_e269.x, _e269.y, 6.0);
    let _e279 = textureSampleLevel(tex2D, samp, (_e273.xy / vec2<f32>(_e273.z)), 3.0);
    c_2 = _e279;
    let _e280 = coord_5;
    let _e287 = coord_5;
    let _e292 = vec4<f32>(_e287.x, _e287.y, 0.0, 6.0);
    let _e299 = textureSampleLevel(tex2D, samp, (_e292.xyz / vec3<f32>(_e292.w)).xy, 3.0);
    c_2 = _e299;
    let _e300 = coord_5;
    let _e308 = coord_5;
    let _e312 = vec3<f32>(_e308.x, _e308.y, 6.0);
    let _e320 = textureSampleLevel(tex2D, samp, (_e312.xy / vec2<f32>(_e312.z)), 3.0, vec2<i32>(5, 5));
    c_2 = _e320;
    let _e321 = coord_5;
    let _e330 = coord_5;
    let _e335 = vec4<f32>(_e330.x, _e330.y, 0.0, 6.0);
    let _e344 = textureSampleLevel(tex2D, samp, (_e335.xyz / vec3<f32>(_e335.w)).xy, 3.0, vec2<i32>(5, 5));
    c_2 = _e344;
    let _e345 = coord_5;
    let _e352 = coord_5;
    let _e356 = vec3<f32>(_e352.x, _e352.y, 6.0);
    let _e363 = textureSample(tex2D, samp, (_e356.xy / vec2<f32>(_e356.z)), vec2<i32>(5, 5));
    c_2 = _e363;
    let _e364 = coord_5;
    let _e372 = coord_5;
    let _e377 = vec4<f32>(_e372.x, _e372.y, 0.0, 6.0);
    let _e385 = textureSample(tex2D, samp, (_e377.xyz / vec3<f32>(_e377.w)).xy, vec2<i32>(5, 5));
    c_2 = _e385;
    let _e386 = coord_5;
    let _e394 = coord_5;
    let _e398 = vec3<f32>(_e394.x, _e394.y, 6.0);
    let _e406 = textureSampleBias(tex2D, samp, (_e398.xy / vec2<f32>(_e398.z)), 2.0, vec2<i32>(5, 5));
    c_2 = _e406;
    let _e407 = coord_5;
    let _e416 = coord_5;
    let _e421 = vec4<f32>(_e416.x, _e416.y, 0.0, 6.0);
    let _e430 = textureSampleBias(tex2D, samp, (_e421.xyz / vec3<f32>(_e421.w)).xy, 2.0, vec2<i32>(5, 5));
    c_2 = _e430;
    let _e431 = coord_5;
    let _e434 = coord_5;
    let _e437 = textureLoad(tex2D, vec2<i32>(_e434), 3);
    c_2 = _e437;
    let _e438 = coord_5;
    let _e443 = coord_5;
    let _e448 = textureLoad(tex2D, vec2<i32>(_e443), 3);
    c_2 = _e448;
    return;
}

fn testTex2DShadow(coord_6: vec2<f32>) {
    var coord_7: vec2<f32>;
    var size2DShadow: vec2<i32>;
    var d: f32;

    coord_7 = coord_6;
    let _e20 = textureDimensions(tex2DShadow, 0);
    size2DShadow = _e20;
    let _e23 = coord_7;
    let _e28 = coord_7;
    let _e32 = vec3<f32>(_e28.x, _e28.y, 1.0);
    let _e35 = textureSampleCompare(tex2DShadow, sampShadow, _e32.xy, _e32.z);
    d = _e35;
    let _e36 = coord_7;
    let _e45 = coord_7;
    let _e49 = vec3<f32>(_e45.x, _e45.y, 1.0);
    let _e56 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e49.xy, _e49.z);
    d = _e56;
    let _e57 = coord_7;
    let _e68 = coord_7;
    let _e72 = vec3<f32>(_e68.x, _e68.y, 1.0);
    let _e81 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e72.xy, _e72.z, vec2<i32>(5, 5));
    d = _e81;
    let _e82 = coord_7;
    let _e88 = coord_7;
    let _e92 = vec3<f32>(_e88.x, _e88.y, 1.0);
    let _e96 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e92.xy, _e92.z);
    d = _e96;
    let _e97 = coord_7;
    let _e105 = coord_7;
    let _e109 = vec3<f32>(_e105.x, _e105.y, 1.0);
    let _e115 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e109.xy, _e109.z, vec2<i32>(5, 5));
    d = _e115;
    let _e116 = coord_7;
    let _e123 = coord_7;
    let _e127 = vec3<f32>(_e123.x, _e123.y, 1.0);
    let _e132 = textureSampleCompare(tex2DShadow, sampShadow, _e127.xy, _e127.z, vec2<i32>(5, 5));
    d = _e132;
    let _e133 = coord_7;
    let _e139 = coord_7;
    let _e144 = vec4<f32>(_e139.x, _e139.y, 1.0, 6.0);
    let _e148 = (_e144.xyz / vec3<f32>(_e144.w));
    let _e151 = textureSampleCompare(tex2DShadow, sampShadow, _e148.xy, _e148.z);
    d = _e151;
    let _e152 = coord_7;
    let _e162 = coord_7;
    let _e167 = vec4<f32>(_e162.x, _e162.y, 1.0, 6.0);
    let _e175 = (_e167.xyz / vec3<f32>(_e167.w));
    let _e178 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e175.xy, _e175.z);
    d = _e178;
    let _e179 = coord_7;
    let _e191 = coord_7;
    let _e196 = vec4<f32>(_e191.x, _e191.y, 1.0, 6.0);
    let _e206 = (_e196.xyz / vec3<f32>(_e196.w));
    let _e209 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e206.xy, _e206.z, vec2<i32>(5, 5));
    d = _e209;
    let _e210 = coord_7;
    let _e217 = coord_7;
    let _e222 = vec4<f32>(_e217.x, _e217.y, 1.0, 6.0);
    let _e227 = (_e222.xyz / vec3<f32>(_e222.w));
    let _e230 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e227.xy, _e227.z);
    d = _e230;
    let _e231 = coord_7;
    let _e240 = coord_7;
    let _e245 = vec4<f32>(_e240.x, _e240.y, 1.0, 6.0);
    let _e252 = (_e245.xyz / vec3<f32>(_e245.w));
    let _e255 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e252.xy, _e252.z, vec2<i32>(5, 5));
    d = _e255;
    let _e256 = coord_7;
    let _e264 = coord_7;
    let _e269 = vec4<f32>(_e264.x, _e264.y, 1.0, 6.0);
    let _e275 = (_e269.xyz / vec3<f32>(_e269.w));
    let _e278 = textureSampleCompare(tex2DShadow, sampShadow, _e275.xy, _e275.z, vec2<i32>(5, 5));
    d = _e278;
    return;
}

fn testTex2DArray(coord_8: vec3<f32>) {
    var coord_9: vec3<f32>;
    var size2DArray: vec3<i32>;
    var c_3: vec4<f32>;

    coord_9 = coord_8;
    let _e20 = textureDimensions(tex2DArray, 0);
    let _e23 = textureNumLayers(tex2DArray);
    size2DArray = vec3<i32>(_e20.x, _e20.y, _e23);
    let _e28 = coord_9;
    let _e32 = textureSample(tex2DArray, samp, _e28.xy, i32(_e28.z));
    c_3 = _e32;
    let _e35 = coord_9;
    let _e40 = textureSampleBias(tex2DArray, samp, _e35.xy, i32(_e35.z), 2.0);
    c_3 = _e40;
    let _e46 = coord_9;
    let _e54 = textureSampleGrad(tex2DArray, samp, _e46.xy, i32(_e46.z), vec2<f32>(4.0), vec2<f32>(4.0));
    c_3 = _e54;
    let _e62 = coord_9;
    let _e72 = textureSampleGrad(tex2DArray, samp, _e62.xy, i32(_e62.z), vec2<f32>(4.0), vec2<f32>(4.0), vec2<i32>(5, 5));
    c_3 = _e72;
    let _e75 = coord_9;
    let _e80 = textureSampleLevel(tex2DArray, samp, _e75.xy, i32(_e75.z), 3.0);
    c_3 = _e80;
    let _e85 = coord_9;
    let _e92 = textureSampleLevel(tex2DArray, samp, _e85.xy, i32(_e85.z), 3.0, vec2<i32>(5, 5));
    c_3 = _e92;
    let _e96 = coord_9;
    let _e102 = textureSample(tex2DArray, samp, _e96.xy, i32(_e96.z), vec2<i32>(5, 5));
    c_3 = _e102;
    let _e107 = coord_9;
    let _e114 = textureSampleBias(tex2DArray, samp, _e107.xy, i32(_e107.z), 2.0, vec2<i32>(5, 5));
    c_3 = _e114;
    let _e115 = coord_9;
    let _e118 = coord_9;
    let _e119 = vec3<i32>(_e118);
    let _e123 = textureLoad(tex2DArray, _e119.xy, _e119.z, 3);
    c_3 = _e123;
    let _e124 = coord_9;
    let _e129 = coord_9;
    let _e130 = vec3<i32>(_e129);
    let _e136 = textureLoad(tex2DArray, _e130.xy, _e130.z, 3);
    c_3 = _e136;
    return;
}

fn testTex2DArrayShadow(coord_10: vec3<f32>) {
    var coord_11: vec3<f32>;
    var size2DArrayShadow: vec3<i32>;
    var d_1: f32;

    coord_11 = coord_10;
    let _e20 = textureDimensions(tex2DArrayShadow, 0);
    let _e23 = textureNumLayers(tex2DArrayShadow);
    size2DArrayShadow = vec3<i32>(_e20.x, _e20.y, _e23);
    let _e27 = coord_11;
    let _e33 = coord_11;
    let _e38 = vec4<f32>(_e33.x, _e33.y, _e33.z, 1.0);
    let _e43 = textureSampleCompare(tex2DArrayShadow, sampShadow, _e38.xy, i32(_e38.z), _e38.w);
    d_1 = _e43;
    let _e44 = coord_11;
    let _e54 = coord_11;
    let _e59 = vec4<f32>(_e54.x, _e54.y, _e54.z, 1.0);
    let _e68 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e59.xy, i32(_e59.z), _e59.w);
    d_1 = _e68;
    let _e69 = coord_11;
    let _e81 = coord_11;
    let _e86 = vec4<f32>(_e81.x, _e81.y, _e81.z, 1.0);
    let _e97 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e86.xy, i32(_e86.z), _e86.w, vec2<i32>(5, 5));
    d_1 = _e97;
    let _e98 = coord_11;
    let _e106 = coord_11;
    let _e111 = vec4<f32>(_e106.x, _e106.y, _e106.z, 1.0);
    let _e118 = textureSampleCompare(tex2DArrayShadow, sampShadow, _e111.xy, i32(_e111.z), _e111.w, vec2<i32>(5, 5));
    d_1 = _e118;
    return;
}

fn testTexCube(coord_12: vec3<f32>) {
    var coord_13: vec3<f32>;
    var sizeCube: vec2<i32>;
    var c_4: vec4<f32>;

    coord_13 = coord_12;
    let _e20 = textureDimensions(texCube, 0);
    sizeCube = _e20;
    let _e24 = coord_13;
    let _e25 = textureSample(texCube, samp, _e24);
    c_4 = _e25;
    let _e28 = coord_13;
    let _e30 = textureSampleBias(texCube, samp, _e28, 2.0);
    c_4 = _e30;
    let _e36 = coord_13;
    let _e41 = textureSampleGrad(texCube, samp, _e36, vec3<f32>(4.0), vec3<f32>(4.0));
    c_4 = _e41;
    let _e44 = coord_13;
    let _e46 = textureSampleLevel(texCube, samp, _e44, 3.0);
    c_4 = _e46;
    return;
}

fn testTexCubeShadow(coord_14: vec3<f32>) {
    var coord_15: vec3<f32>;
    var sizeCubeShadow: vec2<i32>;
    var d_2: f32;

    coord_15 = coord_14;
    let _e20 = textureDimensions(texCubeShadow, 0);
    sizeCubeShadow = _e20;
    let _e23 = coord_15;
    let _e29 = coord_15;
    let _e34 = vec4<f32>(_e29.x, _e29.y, _e29.z, 1.0);
    let _e37 = textureSampleCompare(texCubeShadow, sampShadow, _e34.xyz, _e34.w);
    d_2 = _e37;
    let _e38 = coord_15;
    let _e48 = coord_15;
    let _e53 = vec4<f32>(_e48.x, _e48.y, _e48.z, 1.0);
    let _e60 = textureSampleCompareLevel(texCubeShadow, sampShadow, _e53.xyz, _e53.w);
    d_2 = _e60;
    return;
}

fn testTexCubeArray(coord_16: vec4<f32>) {
    var coord_17: vec4<f32>;
    var sizeCubeArray: vec3<i32>;
    var c_5: vec4<f32>;

    coord_17 = coord_16;
    let _e20 = textureDimensions(texCubeArray, 0);
    let _e23 = textureNumLayers(texCubeArray);
    sizeCubeArray = vec3<i32>(_e20.x, _e20.y, _e23);
    let _e28 = coord_17;
    let _e32 = textureSample(texCubeArray, samp, _e28.xyz, i32(_e28.w));
    c_5 = _e32;
    let _e35 = coord_17;
    let _e40 = textureSampleBias(texCubeArray, samp, _e35.xyz, i32(_e35.w), 2.0);
    c_5 = _e40;
    let _e46 = coord_17;
    let _e54 = textureSampleGrad(texCubeArray, samp, _e46.xyz, i32(_e46.w), vec3<f32>(4.0), vec3<f32>(4.0));
    c_5 = _e54;
    let _e57 = coord_17;
    let _e62 = textureSampleLevel(texCubeArray, samp, _e57.xyz, i32(_e57.w), 3.0);
    c_5 = _e62;
    return;
}

fn testTexCubeArrayShadow(coord_18: vec4<f32>) {
    var coord_19: vec4<f32>;
    var sizeCubeArrayShadow: vec3<i32>;
    var d_3: f32;

    coord_19 = coord_18;
    let _e20 = textureDimensions(texCubeArrayShadow, 0);
    let _e23 = textureNumLayers(texCubeArrayShadow);
    sizeCubeArrayShadow = vec3<i32>(_e20.x, _e20.y, _e23);
    let _e29 = coord_19;
    let _e34 = textureSampleCompare(texCubeArrayShadow, sampShadow, _e29.xyz, i32(_e29.w), 1.0);
    d_3 = _e34;
    return;
}

fn testTex3D(coord_20: vec3<f32>) {
    var coord_21: vec3<f32>;
    var size3D: vec3<i32>;
    var c_6: vec4<f32>;

    coord_21 = coord_20;
    let _e20 = textureDimensions(tex3D, 0);
    size3D = _e20;
    let _e24 = coord_21;
    let _e25 = textureSample(tex3D, samp, _e24);
    c_6 = _e25;
    let _e28 = coord_21;
    let _e30 = textureSampleBias(tex3D, samp, _e28, 2.0);
    c_6 = _e30;
    let _e31 = coord_21;
    let _e37 = coord_21;
    let _e42 = vec4<f32>(_e37.x, _e37.y, _e37.z, 6.0);
    let _e47 = textureSample(tex3D, samp, (_e42.xyz / vec3<f32>(_e42.w)));
    c_6 = _e47;
    let _e48 = coord_21;
    let _e55 = coord_21;
    let _e60 = vec4<f32>(_e55.x, _e55.y, _e55.z, 6.0);
    let _e66 = textureSampleBias(tex3D, samp, (_e60.xyz / vec3<f32>(_e60.w)), 2.0);
    c_6 = _e66;
    let _e67 = coord_21;
    let _e75 = coord_21;
    let _e80 = vec4<f32>(_e75.x, _e75.y, _e75.z, 6.0);
    let _e87 = textureSample(tex3D, samp, (_e80.xyz / vec3<f32>(_e80.w)), vec3<i32>(5, 5, 5));
    c_6 = _e87;
    let _e88 = coord_21;
    let _e97 = coord_21;
    let _e102 = vec4<f32>(_e97.x, _e97.y, _e97.z, 6.0);
    let _e110 = textureSampleBias(tex3D, samp, (_e102.xyz / vec3<f32>(_e102.w)), 2.0, vec3<i32>(5, 5, 5));
    c_6 = _e110;
    let _e111 = coord_21;
    let _e118 = coord_21;
    let _e123 = vec4<f32>(_e118.x, _e118.y, _e118.z, 6.0);
    let _e129 = textureSampleLevel(tex3D, samp, (_e123.xyz / vec3<f32>(_e123.w)), 3.0);
    c_6 = _e129;
    let _e130 = coord_21;
    let _e139 = coord_21;
    let _e144 = vec4<f32>(_e139.x, _e139.y, _e139.z, 6.0);
    let _e152 = textureSampleLevel(tex3D, samp, (_e144.xyz / vec3<f32>(_e144.w)), 3.0, vec3<i32>(5, 5, 5));
    c_6 = _e152;
    let _e153 = coord_21;
    let _e163 = coord_21;
    let _e168 = vec4<f32>(_e163.x, _e163.y, _e163.z, 6.0);
    let _e177 = textureSampleGrad(tex3D, samp, (_e168.xyz / vec3<f32>(_e168.w)), vec3<f32>(4.0), vec3<f32>(4.0));
    c_6 = _e177;
    let _e178 = coord_21;
    let _e190 = coord_21;
    let _e195 = vec4<f32>(_e190.x, _e190.y, _e190.z, 6.0);
    let _e206 = textureSampleGrad(tex3D, samp, (_e195.xyz / vec3<f32>(_e195.w)), vec3<f32>(4.0), vec3<f32>(4.0), vec3<i32>(5, 5, 5));
    c_6 = _e206;
    let _e212 = coord_21;
    let _e217 = textureSampleGrad(tex3D, samp, _e212, vec3<f32>(4.0), vec3<f32>(4.0));
    c_6 = _e217;
    let _e225 = coord_21;
    let _e232 = textureSampleGrad(tex3D, samp, _e225, vec3<f32>(4.0), vec3<f32>(4.0), vec3<i32>(5, 5, 5));
    c_6 = _e232;
    let _e235 = coord_21;
    let _e237 = textureSampleLevel(tex3D, samp, _e235, 3.0);
    c_6 = _e237;
    let _e242 = coord_21;
    let _e246 = textureSampleLevel(tex3D, samp, _e242, 3.0, vec3<i32>(5, 5, 5));
    c_6 = _e246;
    let _e250 = coord_21;
    let _e253 = textureSample(tex3D, samp, _e250, vec3<i32>(5, 5, 5));
    c_6 = _e253;
    let _e258 = coord_21;
    let _e262 = textureSampleBias(tex3D, samp, _e258, 2.0, vec3<i32>(5, 5, 5));
    c_6 = _e262;
    let _e263 = coord_21;
    let _e266 = coord_21;
    let _e269 = textureLoad(tex3D, vec3<i32>(_e266), 3);
    c_6 = _e269;
    let _e270 = coord_21;
    let _e275 = coord_21;
    let _e280 = textureLoad(tex3D, vec3<i32>(_e275), 3);
    c_6 = _e280;
    return;
}

fn testTex2DMS(coord_22: vec2<f32>) {
    var coord_23: vec2<f32>;
    var size2DMS: vec2<i32>;
    var c_7: vec4<f32>;

    coord_23 = coord_22;
    let _e18 = textureDimensions(tex2DMS);
    size2DMS = _e18;
    let _e21 = coord_23;
    let _e24 = coord_23;
    let _e27 = textureLoad(tex2DMS, vec2<i32>(_e24), 3);
    c_7 = _e27;
    return;
}

fn testTex2DMSArray(coord_24: vec3<f32>) {
    var coord_25: vec3<f32>;
    var size2DMSArray: vec3<i32>;
    var c_8: vec4<f32>;

    coord_25 = coord_24;
    let _e18 = textureDimensions(tex2DMSArray);
    let _e21 = textureNumLayers(tex2DMSArray);
    size2DMSArray = vec3<i32>(_e18.x, _e18.y, _e21);
    let _e25 = coord_25;
    let _e28 = coord_25;
    let _e29 = vec3<i32>(_e28);
    let _e33 = textureLoad(tex2DMSArray, _e29.xy, _e29.z, 3);
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
