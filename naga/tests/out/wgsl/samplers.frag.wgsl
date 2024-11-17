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
    let _e22 = textureDimensions(tex1D, 0i);
    size1D = i32(_e22);
    let _e25 = textureNumLevels(tex1D);
    levels = i32(_e25);
    let _e30 = coord_1;
    let _e31 = textureSample(tex1D, samp, _e30);
    c = _e31;
    let _e34 = coord_1;
    let _e36 = textureSampleBias(tex1D, samp, _e34, 2f);
    c = _e36;
    let _e40 = coord_1;
    let _e43 = textureSampleGrad(tex1D, samp, _e40, 4f, 4f);
    c = _e43;
    let _e48 = coord_1;
    let _e52 = textureSampleGrad(tex1D, samp, _e48, 4f, 4f, 5i);
    c = _e52;
    let _e55 = coord_1;
    let _e57 = textureSampleLevel(tex1D, samp, _e55, 3f);
    c = _e57;
    let _e61 = coord_1;
    let _e64 = textureSampleLevel(tex1D, samp, _e61, 3f, 5i);
    c = _e64;
    let _e67 = coord_1;
    let _e69 = textureSample(tex1D, samp, _e67, 5i);
    c = _e69;
    let _e73 = coord_1;
    let _e76 = textureSampleBias(tex1D, samp, _e73, 2f, 5i);
    c = _e76;
    let _e77 = coord_1;
    let _e80 = coord_1;
    let _e82 = vec2<f32>(_e80, 6f);
    let _e86 = textureSample(tex1D, samp, (_e82.x / _e82.y));
    c = _e86;
    let _e87 = coord_1;
    let _e92 = coord_1;
    let _e96 = vec4<f32>(_e92, 0f, 0f, 6f);
    let _e102 = textureSample(tex1D, samp, (_e96.xyz / vec3(_e96.w)).x);
    c = _e102;
    let _e103 = coord_1;
    let _e107 = coord_1;
    let _e109 = vec2<f32>(_e107, 6f);
    let _e114 = textureSampleBias(tex1D, samp, (_e109.x / _e109.y), 2f);
    c = _e114;
    let _e115 = coord_1;
    let _e121 = coord_1;
    let _e125 = vec4<f32>(_e121, 0f, 0f, 6f);
    let _e132 = textureSampleBias(tex1D, samp, (_e125.xyz / vec3(_e125.w)).x, 2f);
    c = _e132;
    let _e133 = coord_1;
    let _e138 = coord_1;
    let _e140 = vec2<f32>(_e138, 6f);
    let _e146 = textureSampleGrad(tex1D, samp, (_e140.x / _e140.y), 4f, 4f);
    c = _e146;
    let _e147 = coord_1;
    let _e154 = coord_1;
    let _e158 = vec4<f32>(_e154, 0f, 0f, 6f);
    let _e166 = textureSampleGrad(tex1D, samp, (_e158.xyz / vec3(_e158.w)).x, 4f, 4f);
    c = _e166;
    let _e167 = coord_1;
    let _e173 = coord_1;
    let _e175 = vec2<f32>(_e173, 6f);
    let _e182 = textureSampleGrad(tex1D, samp, (_e175.x / _e175.y), 4f, 4f, 5i);
    c = _e182;
    let _e183 = coord_1;
    let _e191 = coord_1;
    let _e195 = vec4<f32>(_e191, 0f, 0f, 6f);
    let _e204 = textureSampleGrad(tex1D, samp, (_e195.xyz / vec3(_e195.w)).x, 4f, 4f, 5i);
    c = _e204;
    let _e205 = coord_1;
    let _e209 = coord_1;
    let _e211 = vec2<f32>(_e209, 6f);
    let _e216 = textureSampleLevel(tex1D, samp, (_e211.x / _e211.y), 3f);
    c = _e216;
    let _e217 = coord_1;
    let _e223 = coord_1;
    let _e227 = vec4<f32>(_e223, 0f, 0f, 6f);
    let _e234 = textureSampleLevel(tex1D, samp, (_e227.xyz / vec3(_e227.w)).x, 3f);
    c = _e234;
    let _e235 = coord_1;
    let _e240 = coord_1;
    let _e242 = vec2<f32>(_e240, 6f);
    let _e248 = textureSampleLevel(tex1D, samp, (_e242.x / _e242.y), 3f, 5i);
    c = _e248;
    let _e249 = coord_1;
    let _e256 = coord_1;
    let _e260 = vec4<f32>(_e256, 0f, 0f, 6f);
    let _e268 = textureSampleLevel(tex1D, samp, (_e260.xyz / vec3(_e260.w)).x, 3f, 5i);
    c = _e268;
    let _e269 = coord_1;
    let _e273 = coord_1;
    let _e275 = vec2<f32>(_e273, 6f);
    let _e280 = textureSample(tex1D, samp, (_e275.x / _e275.y), 5i);
    c = _e280;
    let _e281 = coord_1;
    let _e287 = coord_1;
    let _e291 = vec4<f32>(_e287, 0f, 0f, 6f);
    let _e298 = textureSample(tex1D, samp, (_e291.xyz / vec3(_e291.w)).x, 5i);
    c = _e298;
    let _e299 = coord_1;
    let _e304 = coord_1;
    let _e306 = vec2<f32>(_e304, 6f);
    let _e312 = textureSampleBias(tex1D, samp, (_e306.x / _e306.y), 2f, 5i);
    c = _e312;
    let _e313 = coord_1;
    let _e320 = coord_1;
    let _e324 = vec4<f32>(_e320, 0f, 0f, 6f);
    let _e332 = textureSampleBias(tex1D, samp, (_e324.xyz / vec3(_e324.w)).x, 2f, 5i);
    c = _e332;
    let _e333 = coord_1;
    let _e336 = coord_1;
    let _e339 = textureLoad(tex1D, i32(_e336), 3i);
    c = _e339;
    let _e340 = coord_1;
    let _e344 = coord_1;
    let _e348 = textureLoad(tex1D, i32(_e344), 3i);
    c = _e348;
    return;
}

fn testTex1DArray(coord_2: vec2<f32>) {
    var coord_3: vec2<f32>;
    var size1DArray: vec2<i32>;
    var levels_1: i32;
    var c_1: vec4<f32>;

    coord_3 = coord_2;
    let _e22 = textureDimensions(tex1DArray, 0i);
    let _e23 = textureNumLayers(tex1DArray);
    size1DArray = vec2<i32>(vec2<u32>(_e22, _e23));
    let _e27 = textureNumLevels(tex1DArray);
    levels_1 = i32(_e27);
    let _e32 = coord_3;
    let _e36 = textureSample(tex1DArray, samp, _e32.x, i32(_e32.y));
    c_1 = _e36;
    let _e39 = coord_3;
    let _e44 = textureSampleBias(tex1DArray, samp, _e39.x, i32(_e39.y), 2f);
    c_1 = _e44;
    let _e48 = coord_3;
    let _e54 = textureSampleGrad(tex1DArray, samp, _e48.x, i32(_e48.y), 4f, 4f);
    c_1 = _e54;
    let _e59 = coord_3;
    let _e66 = textureSampleGrad(tex1DArray, samp, _e59.x, i32(_e59.y), 4f, 4f, 5i);
    c_1 = _e66;
    let _e69 = coord_3;
    let _e74 = textureSampleLevel(tex1DArray, samp, _e69.x, i32(_e69.y), 3f);
    c_1 = _e74;
    let _e78 = coord_3;
    let _e84 = textureSampleLevel(tex1DArray, samp, _e78.x, i32(_e78.y), 3f, 5i);
    c_1 = _e84;
    let _e87 = coord_3;
    let _e92 = textureSample(tex1DArray, samp, _e87.x, i32(_e87.y), 5i);
    c_1 = _e92;
    let _e96 = coord_3;
    let _e102 = textureSampleBias(tex1DArray, samp, _e96.x, i32(_e96.y), 2f, 5i);
    c_1 = _e102;
    let _e103 = coord_3;
    let _e106 = coord_3;
    let _e107 = vec2<i32>(_e106);
    let _e111 = textureLoad(tex1DArray, _e107.x, _e107.y, 3i);
    c_1 = _e111;
    let _e112 = coord_3;
    let _e116 = coord_3;
    let _e117 = vec2<i32>(_e116);
    let _e122 = textureLoad(tex1DArray, _e117.x, _e117.y, 3i);
    c_1 = _e122;
    return;
}

fn testTex2D(coord_4: vec2<f32>) {
    var coord_5: vec2<f32>;
    var size2D: vec2<i32>;
    var levels_2: i32;
    var c_2: vec4<f32>;

    coord_5 = coord_4;
    let _e22 = textureDimensions(tex2D, 0i);
    size2D = vec2<i32>(_e22);
    let _e25 = textureNumLevels(tex2D);
    levels_2 = i32(_e25);
    let _e30 = coord_5;
    let _e31 = textureSample(tex2D, samp, _e30);
    c_2 = _e31;
    let _e34 = coord_5;
    let _e36 = textureSampleBias(tex2D, samp, _e34, 2f);
    c_2 = _e36;
    let _e42 = coord_5;
    let _e47 = textureSampleGrad(tex2D, samp, _e42, vec2(4f), vec2(4f));
    c_2 = _e47;
    let _e55 = coord_5;
    let _e62 = textureSampleGrad(tex2D, samp, _e55, vec2(4f), vec2(4f), vec2(5i));
    c_2 = _e62;
    let _e65 = coord_5;
    let _e67 = textureSampleLevel(tex2D, samp, _e65, 3f);
    c_2 = _e67;
    let _e72 = coord_5;
    let _e76 = textureSampleLevel(tex2D, samp, _e72, 3f, vec2(5i));
    c_2 = _e76;
    let _e80 = coord_5;
    let _e83 = textureSample(tex2D, samp, _e80, vec2(5i));
    c_2 = _e83;
    let _e88 = coord_5;
    let _e92 = textureSampleBias(tex2D, samp, _e88, 2f, vec2(5i));
    c_2 = _e92;
    let _e93 = coord_5;
    let _e98 = coord_5;
    let _e102 = vec3<f32>(_e98.x, _e98.y, 6f);
    let _e107 = textureSample(tex2D, samp, (_e102.xy / vec2(_e102.z)));
    c_2 = _e107;
    let _e108 = coord_5;
    let _e114 = coord_5;
    let _e119 = vec4<f32>(_e114.x, _e114.y, 0f, 6f);
    let _e125 = textureSample(tex2D, samp, (_e119.xyz / vec3(_e119.w)).xy);
    c_2 = _e125;
    let _e126 = coord_5;
    let _e132 = coord_5;
    let _e136 = vec3<f32>(_e132.x, _e132.y, 6f);
    let _e142 = textureSampleBias(tex2D, samp, (_e136.xy / vec2(_e136.z)), 2f);
    c_2 = _e142;
    let _e143 = coord_5;
    let _e150 = coord_5;
    let _e155 = vec4<f32>(_e150.x, _e150.y, 0f, 6f);
    let _e162 = textureSampleBias(tex2D, samp, (_e155.xyz / vec3(_e155.w)).xy, 2f);
    c_2 = _e162;
    let _e163 = coord_5;
    let _e172 = coord_5;
    let _e176 = vec3<f32>(_e172.x, _e172.y, 6f);
    let _e185 = textureSampleGrad(tex2D, samp, (_e176.xy / vec2(_e176.z)), vec2(4f), vec2(4f));
    c_2 = _e185;
    let _e186 = coord_5;
    let _e196 = coord_5;
    let _e201 = vec4<f32>(_e196.x, _e196.y, 0f, 6f);
    let _e211 = textureSampleGrad(tex2D, samp, (_e201.xyz / vec3(_e201.w)).xy, vec2(4f), vec2(4f));
    c_2 = _e211;
    let _e212 = coord_5;
    let _e223 = coord_5;
    let _e227 = vec3<f32>(_e223.x, _e223.y, 6f);
    let _e238 = textureSampleGrad(tex2D, samp, (_e227.xy / vec2(_e227.z)), vec2(4f), vec2(4f), vec2(5i));
    c_2 = _e238;
    let _e239 = coord_5;
    let _e251 = coord_5;
    let _e256 = vec4<f32>(_e251.x, _e251.y, 0f, 6f);
    let _e268 = textureSampleGrad(tex2D, samp, (_e256.xyz / vec3(_e256.w)).xy, vec2(4f), vec2(4f), vec2(5i));
    c_2 = _e268;
    let _e269 = coord_5;
    let _e275 = coord_5;
    let _e279 = vec3<f32>(_e275.x, _e275.y, 6f);
    let _e285 = textureSampleLevel(tex2D, samp, (_e279.xy / vec2(_e279.z)), 3f);
    c_2 = _e285;
    let _e286 = coord_5;
    let _e293 = coord_5;
    let _e298 = vec4<f32>(_e293.x, _e293.y, 0f, 6f);
    let _e305 = textureSampleLevel(tex2D, samp, (_e298.xyz / vec3(_e298.w)).xy, 3f);
    c_2 = _e305;
    let _e306 = coord_5;
    let _e314 = coord_5;
    let _e318 = vec3<f32>(_e314.x, _e314.y, 6f);
    let _e326 = textureSampleLevel(tex2D, samp, (_e318.xy / vec2(_e318.z)), 3f, vec2(5i));
    c_2 = _e326;
    let _e327 = coord_5;
    let _e336 = coord_5;
    let _e341 = vec4<f32>(_e336.x, _e336.y, 0f, 6f);
    let _e350 = textureSampleLevel(tex2D, samp, (_e341.xyz / vec3(_e341.w)).xy, 3f, vec2(5i));
    c_2 = _e350;
    let _e351 = coord_5;
    let _e358 = coord_5;
    let _e362 = vec3<f32>(_e358.x, _e358.y, 6f);
    let _e369 = textureSample(tex2D, samp, (_e362.xy / vec2(_e362.z)), vec2(5i));
    c_2 = _e369;
    let _e370 = coord_5;
    let _e378 = coord_5;
    let _e383 = vec4<f32>(_e378.x, _e378.y, 0f, 6f);
    let _e391 = textureSample(tex2D, samp, (_e383.xyz / vec3(_e383.w)).xy, vec2(5i));
    c_2 = _e391;
    let _e392 = coord_5;
    let _e400 = coord_5;
    let _e404 = vec3<f32>(_e400.x, _e400.y, 6f);
    let _e412 = textureSampleBias(tex2D, samp, (_e404.xy / vec2(_e404.z)), 2f, vec2(5i));
    c_2 = _e412;
    let _e413 = coord_5;
    let _e422 = coord_5;
    let _e427 = vec4<f32>(_e422.x, _e422.y, 0f, 6f);
    let _e436 = textureSampleBias(tex2D, samp, (_e427.xyz / vec3(_e427.w)).xy, 2f, vec2(5i));
    c_2 = _e436;
    let _e437 = coord_5;
    let _e440 = coord_5;
    let _e443 = textureLoad(tex2D, vec2<i32>(_e440), 3i);
    c_2 = _e443;
    let _e444 = coord_5;
    let _e447 = coord_5;
    let _e450 = textureLoad(utex2D, vec2<i32>(_e447), 3i);
    c_2 = vec4<f32>(_e450);
    let _e452 = coord_5;
    let _e455 = coord_5;
    let _e458 = textureLoad(itex2D, vec2<i32>(_e455), 3i);
    c_2 = vec4<f32>(_e458);
    let _e460 = coord_5;
    let _e465 = coord_5;
    let _e470 = textureLoad(tex2D, vec2<i32>(_e465), 3i);
    c_2 = _e470;
    let _e471 = coord_5;
    let _e476 = coord_5;
    let _e481 = textureLoad(utex2D, vec2<i32>(_e476), 3i);
    c_2 = vec4<f32>(_e481);
    let _e483 = coord_5;
    let _e488 = coord_5;
    let _e493 = textureLoad(itex2D, vec2<i32>(_e488), 3i);
    c_2 = vec4<f32>(_e493);
    return;
}

fn testTex2DShadow(coord_6: vec2<f32>) {
    var coord_7: vec2<f32>;
    var size2DShadow: vec2<i32>;
    var levels_3: i32;
    var d: f32;

    coord_7 = coord_6;
    let _e22 = textureDimensions(tex2DShadow, 0i);
    size2DShadow = vec2<i32>(_e22);
    let _e25 = textureNumLevels(tex2DShadow);
    levels_3 = i32(_e25);
    let _e29 = coord_7;
    let _e34 = coord_7;
    let _e38 = vec3<f32>(_e34.x, _e34.y, 1f);
    let _e41 = textureSampleCompare(tex2DShadow, sampShadow, _e38.xy, _e38.z);
    d = _e41;
    let _e42 = coord_7;
    let _e51 = coord_7;
    let _e55 = vec3<f32>(_e51.x, _e51.y, 1f);
    let _e62 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e55.xy, _e55.z);
    d = _e62;
    let _e63 = coord_7;
    let _e74 = coord_7;
    let _e78 = vec3<f32>(_e74.x, _e74.y, 1f);
    let _e87 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e78.xy, _e78.z, vec2(5i));
    d = _e87;
    let _e88 = coord_7;
    let _e94 = coord_7;
    let _e98 = vec3<f32>(_e94.x, _e94.y, 1f);
    let _e102 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e98.xy, _e98.z);
    d = _e102;
    let _e103 = coord_7;
    let _e111 = coord_7;
    let _e115 = vec3<f32>(_e111.x, _e111.y, 1f);
    let _e121 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e115.xy, _e115.z, vec2(5i));
    d = _e121;
    let _e122 = coord_7;
    let _e129 = coord_7;
    let _e133 = vec3<f32>(_e129.x, _e129.y, 1f);
    let _e138 = textureSampleCompare(tex2DShadow, sampShadow, _e133.xy, _e133.z, vec2(5i));
    d = _e138;
    let _e139 = coord_7;
    let _e145 = coord_7;
    let _e150 = vec4<f32>(_e145.x, _e145.y, 1f, 6f);
    let _e154 = (_e150.xyz / vec3(_e150.w));
    let _e157 = textureSampleCompare(tex2DShadow, sampShadow, _e154.xy, _e154.z);
    d = _e157;
    let _e158 = coord_7;
    let _e168 = coord_7;
    let _e173 = vec4<f32>(_e168.x, _e168.y, 1f, 6f);
    let _e181 = (_e173.xyz / vec3(_e173.w));
    let _e184 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e181.xy, _e181.z);
    d = _e184;
    let _e185 = coord_7;
    let _e197 = coord_7;
    let _e202 = vec4<f32>(_e197.x, _e197.y, 1f, 6f);
    let _e212 = (_e202.xyz / vec3(_e202.w));
    let _e215 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e212.xy, _e212.z, vec2(5i));
    d = _e215;
    let _e216 = coord_7;
    let _e223 = coord_7;
    let _e228 = vec4<f32>(_e223.x, _e223.y, 1f, 6f);
    let _e233 = (_e228.xyz / vec3(_e228.w));
    let _e236 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e233.xy, _e233.z);
    d = _e236;
    let _e237 = coord_7;
    let _e246 = coord_7;
    let _e251 = vec4<f32>(_e246.x, _e246.y, 1f, 6f);
    let _e258 = (_e251.xyz / vec3(_e251.w));
    let _e261 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e258.xy, _e258.z, vec2(5i));
    d = _e261;
    let _e262 = coord_7;
    let _e270 = coord_7;
    let _e275 = vec4<f32>(_e270.x, _e270.y, 1f, 6f);
    let _e281 = (_e275.xyz / vec3(_e275.w));
    let _e284 = textureSampleCompare(tex2DShadow, sampShadow, _e281.xy, _e281.z, vec2(5i));
    d = _e284;
    return;
}

fn testTex2DArray(coord_8: vec3<f32>) {
    var coord_9: vec3<f32>;
    var size2DArray: vec3<i32>;
    var levels_4: i32;
    var c_3: vec4<f32>;

    coord_9 = coord_8;
    let _e22 = textureDimensions(tex2DArray, 0i);
    let _e25 = textureNumLayers(tex2DArray);
    size2DArray = vec3<i32>(vec3<u32>(_e22.x, _e22.y, _e25));
    let _e29 = textureNumLevels(tex2DArray);
    levels_4 = i32(_e29);
    let _e34 = coord_9;
    let _e38 = textureSample(tex2DArray, samp, _e34.xy, i32(_e34.z));
    c_3 = _e38;
    let _e41 = coord_9;
    let _e46 = textureSampleBias(tex2DArray, samp, _e41.xy, i32(_e41.z), 2f);
    c_3 = _e46;
    let _e52 = coord_9;
    let _e60 = textureSampleGrad(tex2DArray, samp, _e52.xy, i32(_e52.z), vec2(4f), vec2(4f));
    c_3 = _e60;
    let _e68 = coord_9;
    let _e78 = textureSampleGrad(tex2DArray, samp, _e68.xy, i32(_e68.z), vec2(4f), vec2(4f), vec2(5i));
    c_3 = _e78;
    let _e81 = coord_9;
    let _e86 = textureSampleLevel(tex2DArray, samp, _e81.xy, i32(_e81.z), 3f);
    c_3 = _e86;
    let _e91 = coord_9;
    let _e98 = textureSampleLevel(tex2DArray, samp, _e91.xy, i32(_e91.z), 3f, vec2(5i));
    c_3 = _e98;
    let _e102 = coord_9;
    let _e108 = textureSample(tex2DArray, samp, _e102.xy, i32(_e102.z), vec2(5i));
    c_3 = _e108;
    let _e113 = coord_9;
    let _e120 = textureSampleBias(tex2DArray, samp, _e113.xy, i32(_e113.z), 2f, vec2(5i));
    c_3 = _e120;
    let _e121 = coord_9;
    let _e124 = coord_9;
    let _e125 = vec3<i32>(_e124);
    let _e129 = textureLoad(tex2DArray, _e125.xy, _e125.z, 3i);
    c_3 = _e129;
    let _e130 = coord_9;
    let _e135 = coord_9;
    let _e136 = vec3<i32>(_e135);
    let _e142 = textureLoad(tex2DArray, _e136.xy, _e136.z, 3i);
    c_3 = _e142;
    return;
}

fn testTex2DArrayShadow(coord_10: vec3<f32>) {
    var coord_11: vec3<f32>;
    var size2DArrayShadow: vec3<i32>;
    var levels_5: i32;
    var d_1: f32;

    coord_11 = coord_10;
    let _e22 = textureDimensions(tex2DArrayShadow, 0i);
    let _e25 = textureNumLayers(tex2DArrayShadow);
    size2DArrayShadow = vec3<i32>(vec3<u32>(_e22.x, _e22.y, _e25));
    let _e29 = textureNumLevels(tex2DArrayShadow);
    levels_5 = i32(_e29);
    let _e33 = coord_11;
    let _e39 = coord_11;
    let _e44 = vec4<f32>(_e39.x, _e39.y, _e39.z, 1f);
    let _e49 = textureSampleCompare(tex2DArrayShadow, sampShadow, _e44.xy, i32(_e44.z), _e44.w);
    d_1 = _e49;
    let _e50 = coord_11;
    let _e60 = coord_11;
    let _e65 = vec4<f32>(_e60.x, _e60.y, _e60.z, 1f);
    let _e74 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e65.xy, i32(_e65.z), _e65.w);
    d_1 = _e74;
    let _e75 = coord_11;
    let _e87 = coord_11;
    let _e92 = vec4<f32>(_e87.x, _e87.y, _e87.z, 1f);
    let _e103 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e92.xy, i32(_e92.z), _e92.w, vec2(5i));
    d_1 = _e103;
    let _e104 = coord_11;
    let _e112 = coord_11;
    let _e117 = vec4<f32>(_e112.x, _e112.y, _e112.z, 1f);
    let _e124 = textureSampleCompare(tex2DArrayShadow, sampShadow, _e117.xy, i32(_e117.z), _e117.w, vec2(5i));
    d_1 = _e124;
    return;
}

fn testTexCube(coord_12: vec3<f32>) {
    var coord_13: vec3<f32>;
    var sizeCube: vec2<i32>;
    var levels_6: i32;
    var c_4: vec4<f32>;

    coord_13 = coord_12;
    let _e22 = textureDimensions(texCube, 0i);
    sizeCube = vec2<i32>(_e22);
    let _e25 = textureNumLevels(texCube);
    levels_6 = i32(_e25);
    let _e30 = coord_13;
    let _e31 = textureSample(texCube, samp, _e30);
    c_4 = _e31;
    let _e34 = coord_13;
    let _e36 = textureSampleBias(texCube, samp, _e34, 2f);
    c_4 = _e36;
    let _e42 = coord_13;
    let _e47 = textureSampleGrad(texCube, samp, _e42, vec3(4f), vec3(4f));
    c_4 = _e47;
    let _e50 = coord_13;
    let _e52 = textureSampleLevel(texCube, samp, _e50, 3f);
    c_4 = _e52;
    return;
}

fn testTexCubeShadow(coord_14: vec3<f32>) {
    var coord_15: vec3<f32>;
    var sizeCubeShadow: vec2<i32>;
    var levels_7: i32;
    var d_2: f32;

    coord_15 = coord_14;
    let _e22 = textureDimensions(texCubeShadow, 0i);
    sizeCubeShadow = vec2<i32>(_e22);
    let _e25 = textureNumLevels(texCubeShadow);
    levels_7 = i32(_e25);
    let _e29 = coord_15;
    let _e35 = coord_15;
    let _e40 = vec4<f32>(_e35.x, _e35.y, _e35.z, 1f);
    let _e43 = textureSampleCompare(texCubeShadow, sampShadow, _e40.xyz, _e40.w);
    d_2 = _e43;
    let _e44 = coord_15;
    let _e54 = coord_15;
    let _e59 = vec4<f32>(_e54.x, _e54.y, _e54.z, 1f);
    let _e66 = textureSampleCompareLevel(texCubeShadow, sampShadow, _e59.xyz, _e59.w);
    d_2 = _e66;
    return;
}

fn testTexCubeArray(coord_16: vec4<f32>) {
    var coord_17: vec4<f32>;
    var sizeCubeArray: vec3<i32>;
    var levels_8: i32;
    var c_5: vec4<f32>;

    coord_17 = coord_16;
    let _e22 = textureDimensions(texCubeArray, 0i);
    let _e25 = textureNumLayers(texCubeArray);
    sizeCubeArray = vec3<i32>(vec3<u32>(_e22.x, _e22.y, _e25));
    let _e29 = textureNumLevels(texCubeArray);
    levels_8 = i32(_e29);
    let _e34 = coord_17;
    let _e38 = textureSample(texCubeArray, samp, _e34.xyz, i32(_e34.w));
    c_5 = _e38;
    let _e41 = coord_17;
    let _e46 = textureSampleBias(texCubeArray, samp, _e41.xyz, i32(_e41.w), 2f);
    c_5 = _e46;
    let _e52 = coord_17;
    let _e60 = textureSampleGrad(texCubeArray, samp, _e52.xyz, i32(_e52.w), vec3(4f), vec3(4f));
    c_5 = _e60;
    let _e63 = coord_17;
    let _e68 = textureSampleLevel(texCubeArray, samp, _e63.xyz, i32(_e63.w), 3f);
    c_5 = _e68;
    return;
}

fn testTexCubeArrayShadow(coord_18: vec4<f32>) {
    var coord_19: vec4<f32>;
    var sizeCubeArrayShadow: vec3<i32>;
    var levels_9: i32;
    var d_3: f32;

    coord_19 = coord_18;
    let _e22 = textureDimensions(texCubeArrayShadow, 0i);
    let _e25 = textureNumLayers(texCubeArrayShadow);
    sizeCubeArrayShadow = vec3<i32>(vec3<u32>(_e22.x, _e22.y, _e25));
    let _e29 = textureNumLevels(texCubeArrayShadow);
    levels_9 = i32(_e29);
    let _e35 = coord_19;
    let _e40 = textureSampleCompare(texCubeArrayShadow, sampShadow, _e35.xyz, i32(_e35.w), 1f);
    d_3 = _e40;
    return;
}

fn testTex3D(coord_20: vec3<f32>) {
    var coord_21: vec3<f32>;
    var size3D: vec3<i32>;
    var levels_10: i32;
    var c_6: vec4<f32>;

    coord_21 = coord_20;
    let _e22 = textureDimensions(tex3D, 0i);
    size3D = vec3<i32>(_e22);
    let _e25 = textureNumLevels(tex3D);
    levels_10 = i32(_e25);
    let _e30 = coord_21;
    let _e31 = textureSample(tex3D, samp, _e30);
    c_6 = _e31;
    let _e34 = coord_21;
    let _e36 = textureSampleBias(tex3D, samp, _e34, 2f);
    c_6 = _e36;
    let _e37 = coord_21;
    let _e43 = coord_21;
    let _e48 = vec4<f32>(_e43.x, _e43.y, _e43.z, 6f);
    let _e53 = textureSample(tex3D, samp, (_e48.xyz / vec3(_e48.w)));
    c_6 = _e53;
    let _e54 = coord_21;
    let _e61 = coord_21;
    let _e66 = vec4<f32>(_e61.x, _e61.y, _e61.z, 6f);
    let _e72 = textureSampleBias(tex3D, samp, (_e66.xyz / vec3(_e66.w)), 2f);
    c_6 = _e72;
    let _e73 = coord_21;
    let _e81 = coord_21;
    let _e86 = vec4<f32>(_e81.x, _e81.y, _e81.z, 6f);
    let _e93 = textureSample(tex3D, samp, (_e86.xyz / vec3(_e86.w)), vec3(5i));
    c_6 = _e93;
    let _e94 = coord_21;
    let _e103 = coord_21;
    let _e108 = vec4<f32>(_e103.x, _e103.y, _e103.z, 6f);
    let _e116 = textureSampleBias(tex3D, samp, (_e108.xyz / vec3(_e108.w)), 2f, vec3(5i));
    c_6 = _e116;
    let _e117 = coord_21;
    let _e124 = coord_21;
    let _e129 = vec4<f32>(_e124.x, _e124.y, _e124.z, 6f);
    let _e135 = textureSampleLevel(tex3D, samp, (_e129.xyz / vec3(_e129.w)), 3f);
    c_6 = _e135;
    let _e136 = coord_21;
    let _e145 = coord_21;
    let _e150 = vec4<f32>(_e145.x, _e145.y, _e145.z, 6f);
    let _e158 = textureSampleLevel(tex3D, samp, (_e150.xyz / vec3(_e150.w)), 3f, vec3(5i));
    c_6 = _e158;
    let _e159 = coord_21;
    let _e169 = coord_21;
    let _e174 = vec4<f32>(_e169.x, _e169.y, _e169.z, 6f);
    let _e183 = textureSampleGrad(tex3D, samp, (_e174.xyz / vec3(_e174.w)), vec3(4f), vec3(4f));
    c_6 = _e183;
    let _e184 = coord_21;
    let _e196 = coord_21;
    let _e201 = vec4<f32>(_e196.x, _e196.y, _e196.z, 6f);
    let _e212 = textureSampleGrad(tex3D, samp, (_e201.xyz / vec3(_e201.w)), vec3(4f), vec3(4f), vec3(5i));
    c_6 = _e212;
    let _e218 = coord_21;
    let _e223 = textureSampleGrad(tex3D, samp, _e218, vec3(4f), vec3(4f));
    c_6 = _e223;
    let _e231 = coord_21;
    let _e238 = textureSampleGrad(tex3D, samp, _e231, vec3(4f), vec3(4f), vec3(5i));
    c_6 = _e238;
    let _e241 = coord_21;
    let _e243 = textureSampleLevel(tex3D, samp, _e241, 3f);
    c_6 = _e243;
    let _e248 = coord_21;
    let _e252 = textureSampleLevel(tex3D, samp, _e248, 3f, vec3(5i));
    c_6 = _e252;
    let _e256 = coord_21;
    let _e259 = textureSample(tex3D, samp, _e256, vec3(5i));
    c_6 = _e259;
    let _e264 = coord_21;
    let _e268 = textureSampleBias(tex3D, samp, _e264, 2f, vec3(5i));
    c_6 = _e268;
    let _e269 = coord_21;
    let _e272 = coord_21;
    let _e275 = textureLoad(tex3D, vec3<i32>(_e272), 3i);
    c_6 = _e275;
    let _e276 = coord_21;
    let _e281 = coord_21;
    let _e286 = textureLoad(tex3D, vec3<i32>(_e281), 3i);
    c_6 = _e286;
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
    let _e27 = coord_23;
    let _e30 = textureLoad(tex2DMS, vec2<i32>(_e27), 3i);
    c_7 = _e30;
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
    let _e31 = coord_25;
    let _e32 = vec3<i32>(_e31);
    let _e36 = textureLoad(tex2DMSArray, _e32.xy, _e32.z, 3i);
    c_8 = _e36;
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
