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
    var levels: i32;
    var c: vec4<f32>;

    coord_1 = coord;
    let _e20 = textureDimensions(tex1D, 0i);
    size1D = i32(_e20);
    let _e23 = textureNumLevels(tex1D);
    levels = i32(_e23);
    let _e28 = coord_1;
    let _e29 = textureSample(tex1D, samp, _e28);
    c = _e29;
    let _e32 = coord_1;
    let _e34 = textureSampleBias(tex1D, samp, _e32, 2f);
    c = _e34;
    let _e38 = coord_1;
    let _e41 = textureSampleGrad(tex1D, samp, _e38, 4f, 4f);
    c = _e41;
    let _e46 = coord_1;
    let _e50 = textureSampleGrad(tex1D, samp, _e46, 4f, 4f, 5i);
    c = _e50;
    let _e53 = coord_1;
    let _e55 = textureSampleLevel(tex1D, samp, _e53, 3f);
    c = _e55;
    let _e59 = coord_1;
    let _e62 = textureSampleLevel(tex1D, samp, _e59, 3f, 5i);
    c = _e62;
    let _e65 = coord_1;
    let _e67 = textureSample(tex1D, samp, _e65, 5i);
    c = _e67;
    let _e71 = coord_1;
    let _e74 = textureSampleBias(tex1D, samp, _e71, 2f, 5i);
    c = _e74;
    let _e75 = coord_1;
    let _e78 = coord_1;
    let _e80 = vec2<f32>(_e78, 6f);
    let _e84 = textureSample(tex1D, samp, (_e80.x / _e80.y));
    c = _e84;
    let _e85 = coord_1;
    let _e90 = coord_1;
    let _e94 = vec4<f32>(_e90, 0f, 0f, 6f);
    let _e100 = textureSample(tex1D, samp, (_e94.xyz / vec3(_e94.w)).x);
    c = _e100;
    let _e101 = coord_1;
    let _e105 = coord_1;
    let _e107 = vec2<f32>(_e105, 6f);
    let _e112 = textureSampleBias(tex1D, samp, (_e107.x / _e107.y), 2f);
    c = _e112;
    let _e113 = coord_1;
    let _e119 = coord_1;
    let _e123 = vec4<f32>(_e119, 0f, 0f, 6f);
    let _e130 = textureSampleBias(tex1D, samp, (_e123.xyz / vec3(_e123.w)).x, 2f);
    c = _e130;
    let _e131 = coord_1;
    let _e136 = coord_1;
    let _e138 = vec2<f32>(_e136, 6f);
    let _e144 = textureSampleGrad(tex1D, samp, (_e138.x / _e138.y), 4f, 4f);
    c = _e144;
    let _e145 = coord_1;
    let _e152 = coord_1;
    let _e156 = vec4<f32>(_e152, 0f, 0f, 6f);
    let _e164 = textureSampleGrad(tex1D, samp, (_e156.xyz / vec3(_e156.w)).x, 4f, 4f);
    c = _e164;
    let _e165 = coord_1;
    let _e171 = coord_1;
    let _e173 = vec2<f32>(_e171, 6f);
    let _e180 = textureSampleGrad(tex1D, samp, (_e173.x / _e173.y), 4f, 4f, 5i);
    c = _e180;
    let _e181 = coord_1;
    let _e189 = coord_1;
    let _e193 = vec4<f32>(_e189, 0f, 0f, 6f);
    let _e202 = textureSampleGrad(tex1D, samp, (_e193.xyz / vec3(_e193.w)).x, 4f, 4f, 5i);
    c = _e202;
    let _e203 = coord_1;
    let _e207 = coord_1;
    let _e209 = vec2<f32>(_e207, 6f);
    let _e214 = textureSampleLevel(tex1D, samp, (_e209.x / _e209.y), 3f);
    c = _e214;
    let _e215 = coord_1;
    let _e221 = coord_1;
    let _e225 = vec4<f32>(_e221, 0f, 0f, 6f);
    let _e232 = textureSampleLevel(tex1D, samp, (_e225.xyz / vec3(_e225.w)).x, 3f);
    c = _e232;
    let _e233 = coord_1;
    let _e238 = coord_1;
    let _e240 = vec2<f32>(_e238, 6f);
    let _e246 = textureSampleLevel(tex1D, samp, (_e240.x / _e240.y), 3f, 5i);
    c = _e246;
    let _e247 = coord_1;
    let _e254 = coord_1;
    let _e258 = vec4<f32>(_e254, 0f, 0f, 6f);
    let _e266 = textureSampleLevel(tex1D, samp, (_e258.xyz / vec3(_e258.w)).x, 3f, 5i);
    c = _e266;
    let _e267 = coord_1;
    let _e271 = coord_1;
    let _e273 = vec2<f32>(_e271, 6f);
    let _e278 = textureSample(tex1D, samp, (_e273.x / _e273.y), 5i);
    c = _e278;
    let _e279 = coord_1;
    let _e285 = coord_1;
    let _e289 = vec4<f32>(_e285, 0f, 0f, 6f);
    let _e296 = textureSample(tex1D, samp, (_e289.xyz / vec3(_e289.w)).x, 5i);
    c = _e296;
    let _e297 = coord_1;
    let _e302 = coord_1;
    let _e304 = vec2<f32>(_e302, 6f);
    let _e310 = textureSampleBias(tex1D, samp, (_e304.x / _e304.y), 2f, 5i);
    c = _e310;
    let _e311 = coord_1;
    let _e318 = coord_1;
    let _e322 = vec4<f32>(_e318, 0f, 0f, 6f);
    let _e330 = textureSampleBias(tex1D, samp, (_e322.xyz / vec3(_e322.w)).x, 2f, 5i);
    c = _e330;
    let _e331 = coord_1;
    let _e334 = coord_1;
    let _e337 = textureLoad(tex1D, i32(_e334), 3i);
    c = _e337;
    let _e338 = coord_1;
    let _e342 = coord_1;
    let _e346 = textureLoad(tex1D, i32(_e342), 3i);
    c = _e346;
    return;
}

fn testTex1DArray(coord_2: vec2<f32>) {
    var coord_3: vec2<f32>;
    var size1DArray: vec2<i32>;
    var levels_1: i32;
    var c_1: vec4<f32>;

    coord_3 = coord_2;
    let _e20 = textureDimensions(tex1DArray, 0i);
    let _e21 = textureNumLayers(tex1DArray);
    size1DArray = vec2<i32>(vec2<u32>(_e20, _e21));
    let _e25 = textureNumLevels(tex1DArray);
    levels_1 = i32(_e25);
    let _e30 = coord_3;
    let _e34 = textureSample(tex1DArray, samp, _e30.x, i32(_e30.y));
    c_1 = _e34;
    let _e37 = coord_3;
    let _e42 = textureSampleBias(tex1DArray, samp, _e37.x, i32(_e37.y), 2f);
    c_1 = _e42;
    let _e46 = coord_3;
    let _e52 = textureSampleGrad(tex1DArray, samp, _e46.x, i32(_e46.y), 4f, 4f);
    c_1 = _e52;
    let _e57 = coord_3;
    let _e64 = textureSampleGrad(tex1DArray, samp, _e57.x, i32(_e57.y), 4f, 4f, 5i);
    c_1 = _e64;
    let _e67 = coord_3;
    let _e72 = textureSampleLevel(tex1DArray, samp, _e67.x, i32(_e67.y), 3f);
    c_1 = _e72;
    let _e76 = coord_3;
    let _e82 = textureSampleLevel(tex1DArray, samp, _e76.x, i32(_e76.y), 3f, 5i);
    c_1 = _e82;
    let _e85 = coord_3;
    let _e90 = textureSample(tex1DArray, samp, _e85.x, i32(_e85.y), 5i);
    c_1 = _e90;
    let _e94 = coord_3;
    let _e100 = textureSampleBias(tex1DArray, samp, _e94.x, i32(_e94.y), 2f, 5i);
    c_1 = _e100;
    let _e101 = coord_3;
    let _e104 = coord_3;
    let _e105 = vec2<i32>(_e104);
    let _e109 = textureLoad(tex1DArray, _e105.x, _e105.y, 3i);
    c_1 = _e109;
    let _e110 = coord_3;
    let _e114 = coord_3;
    let _e115 = vec2<i32>(_e114);
    let _e120 = textureLoad(tex1DArray, _e115.x, _e115.y, 3i);
    c_1 = _e120;
    return;
}

fn testTex2D(coord_4: vec2<f32>) {
    var coord_5: vec2<f32>;
    var size2D: vec2<i32>;
    var levels_2: i32;
    var c_2: vec4<f32>;

    coord_5 = coord_4;
    let _e20 = textureDimensions(tex2D, 0i);
    size2D = vec2<i32>(_e20);
    let _e23 = textureNumLevels(tex2D);
    levels_2 = i32(_e23);
    let _e28 = coord_5;
    let _e29 = textureSample(tex2D, samp, _e28);
    c_2 = _e29;
    let _e32 = coord_5;
    let _e34 = textureSampleBias(tex2D, samp, _e32, 2f);
    c_2 = _e34;
    let _e40 = coord_5;
    let _e45 = textureSampleGrad(tex2D, samp, _e40, vec2(4f), vec2(4f));
    c_2 = _e45;
    let _e53 = coord_5;
    let _e60 = textureSampleGrad(tex2D, samp, _e53, vec2(4f), vec2(4f), vec2(5i));
    c_2 = _e60;
    let _e63 = coord_5;
    let _e65 = textureSampleLevel(tex2D, samp, _e63, 3f);
    c_2 = _e65;
    let _e70 = coord_5;
    let _e74 = textureSampleLevel(tex2D, samp, _e70, 3f, vec2(5i));
    c_2 = _e74;
    let _e78 = coord_5;
    let _e81 = textureSample(tex2D, samp, _e78, vec2(5i));
    c_2 = _e81;
    let _e86 = coord_5;
    let _e90 = textureSampleBias(tex2D, samp, _e86, 2f, vec2(5i));
    c_2 = _e90;
    let _e91 = coord_5;
    let _e96 = coord_5;
    let _e100 = vec3<f32>(_e96.x, _e96.y, 6f);
    let _e105 = textureSample(tex2D, samp, (_e100.xy / vec2(_e100.z)));
    c_2 = _e105;
    let _e106 = coord_5;
    let _e112 = coord_5;
    let _e117 = vec4<f32>(_e112.x, _e112.y, 0f, 6f);
    let _e123 = textureSample(tex2D, samp, (_e117.xyz / vec3(_e117.w)).xy);
    c_2 = _e123;
    let _e124 = coord_5;
    let _e130 = coord_5;
    let _e134 = vec3<f32>(_e130.x, _e130.y, 6f);
    let _e140 = textureSampleBias(tex2D, samp, (_e134.xy / vec2(_e134.z)), 2f);
    c_2 = _e140;
    let _e141 = coord_5;
    let _e148 = coord_5;
    let _e153 = vec4<f32>(_e148.x, _e148.y, 0f, 6f);
    let _e160 = textureSampleBias(tex2D, samp, (_e153.xyz / vec3(_e153.w)).xy, 2f);
    c_2 = _e160;
    let _e161 = coord_5;
    let _e170 = coord_5;
    let _e174 = vec3<f32>(_e170.x, _e170.y, 6f);
    let _e183 = textureSampleGrad(tex2D, samp, (_e174.xy / vec2(_e174.z)), vec2(4f), vec2(4f));
    c_2 = _e183;
    let _e184 = coord_5;
    let _e194 = coord_5;
    let _e199 = vec4<f32>(_e194.x, _e194.y, 0f, 6f);
    let _e209 = textureSampleGrad(tex2D, samp, (_e199.xyz / vec3(_e199.w)).xy, vec2(4f), vec2(4f));
    c_2 = _e209;
    let _e210 = coord_5;
    let _e221 = coord_5;
    let _e225 = vec3<f32>(_e221.x, _e221.y, 6f);
    let _e236 = textureSampleGrad(tex2D, samp, (_e225.xy / vec2(_e225.z)), vec2(4f), vec2(4f), vec2(5i));
    c_2 = _e236;
    let _e237 = coord_5;
    let _e249 = coord_5;
    let _e254 = vec4<f32>(_e249.x, _e249.y, 0f, 6f);
    let _e266 = textureSampleGrad(tex2D, samp, (_e254.xyz / vec3(_e254.w)).xy, vec2(4f), vec2(4f), vec2(5i));
    c_2 = _e266;
    let _e267 = coord_5;
    let _e273 = coord_5;
    let _e277 = vec3<f32>(_e273.x, _e273.y, 6f);
    let _e283 = textureSampleLevel(tex2D, samp, (_e277.xy / vec2(_e277.z)), 3f);
    c_2 = _e283;
    let _e284 = coord_5;
    let _e291 = coord_5;
    let _e296 = vec4<f32>(_e291.x, _e291.y, 0f, 6f);
    let _e303 = textureSampleLevel(tex2D, samp, (_e296.xyz / vec3(_e296.w)).xy, 3f);
    c_2 = _e303;
    let _e304 = coord_5;
    let _e312 = coord_5;
    let _e316 = vec3<f32>(_e312.x, _e312.y, 6f);
    let _e324 = textureSampleLevel(tex2D, samp, (_e316.xy / vec2(_e316.z)), 3f, vec2(5i));
    c_2 = _e324;
    let _e325 = coord_5;
    let _e334 = coord_5;
    let _e339 = vec4<f32>(_e334.x, _e334.y, 0f, 6f);
    let _e348 = textureSampleLevel(tex2D, samp, (_e339.xyz / vec3(_e339.w)).xy, 3f, vec2(5i));
    c_2 = _e348;
    let _e349 = coord_5;
    let _e356 = coord_5;
    let _e360 = vec3<f32>(_e356.x, _e356.y, 6f);
    let _e367 = textureSample(tex2D, samp, (_e360.xy / vec2(_e360.z)), vec2(5i));
    c_2 = _e367;
    let _e368 = coord_5;
    let _e376 = coord_5;
    let _e381 = vec4<f32>(_e376.x, _e376.y, 0f, 6f);
    let _e389 = textureSample(tex2D, samp, (_e381.xyz / vec3(_e381.w)).xy, vec2(5i));
    c_2 = _e389;
    let _e390 = coord_5;
    let _e398 = coord_5;
    let _e402 = vec3<f32>(_e398.x, _e398.y, 6f);
    let _e410 = textureSampleBias(tex2D, samp, (_e402.xy / vec2(_e402.z)), 2f, vec2(5i));
    c_2 = _e410;
    let _e411 = coord_5;
    let _e420 = coord_5;
    let _e425 = vec4<f32>(_e420.x, _e420.y, 0f, 6f);
    let _e434 = textureSampleBias(tex2D, samp, (_e425.xyz / vec3(_e425.w)).xy, 2f, vec2(5i));
    c_2 = _e434;
    let _e435 = coord_5;
    let _e438 = coord_5;
    let _e441 = textureLoad(tex2D, vec2<i32>(_e438), 3i);
    c_2 = _e441;
    let _e442 = coord_5;
    let _e447 = coord_5;
    let _e452 = textureLoad(tex2D, vec2<i32>(_e447), 3i);
    c_2 = _e452;
    return;
}

fn testTex2DShadow(coord_6: vec2<f32>) {
    var coord_7: vec2<f32>;
    var size2DShadow: vec2<i32>;
    var levels_3: i32;
    var d: f32;

    coord_7 = coord_6;
    let _e20 = textureDimensions(tex2DShadow, 0i);
    size2DShadow = vec2<i32>(_e20);
    let _e23 = textureNumLevels(tex2DShadow);
    levels_3 = i32(_e23);
    let _e27 = coord_7;
    let _e32 = coord_7;
    let _e36 = vec3<f32>(_e32.x, _e32.y, 1f);
    let _e39 = textureSampleCompare(tex2DShadow, sampShadow, _e36.xy, _e36.z);
    d = _e39;
    let _e40 = coord_7;
    let _e49 = coord_7;
    let _e53 = vec3<f32>(_e49.x, _e49.y, 1f);
    let _e60 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e53.xy, _e53.z);
    d = _e60;
    let _e61 = coord_7;
    let _e72 = coord_7;
    let _e76 = vec3<f32>(_e72.x, _e72.y, 1f);
    let _e85 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e76.xy, _e76.z, vec2(5i));
    d = _e85;
    let _e86 = coord_7;
    let _e92 = coord_7;
    let _e96 = vec3<f32>(_e92.x, _e92.y, 1f);
    let _e100 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e96.xy, _e96.z);
    d = _e100;
    let _e101 = coord_7;
    let _e109 = coord_7;
    let _e113 = vec3<f32>(_e109.x, _e109.y, 1f);
    let _e119 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e113.xy, _e113.z, vec2(5i));
    d = _e119;
    let _e120 = coord_7;
    let _e127 = coord_7;
    let _e131 = vec3<f32>(_e127.x, _e127.y, 1f);
    let _e136 = textureSampleCompare(tex2DShadow, sampShadow, _e131.xy, _e131.z, vec2(5i));
    d = _e136;
    let _e137 = coord_7;
    let _e143 = coord_7;
    let _e148 = vec4<f32>(_e143.x, _e143.y, 1f, 6f);
    let _e152 = (_e148.xyz / vec3(_e148.w));
    let _e155 = textureSampleCompare(tex2DShadow, sampShadow, _e152.xy, _e152.z);
    d = _e155;
    let _e156 = coord_7;
    let _e166 = coord_7;
    let _e171 = vec4<f32>(_e166.x, _e166.y, 1f, 6f);
    let _e179 = (_e171.xyz / vec3(_e171.w));
    let _e182 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e179.xy, _e179.z);
    d = _e182;
    let _e183 = coord_7;
    let _e195 = coord_7;
    let _e200 = vec4<f32>(_e195.x, _e195.y, 1f, 6f);
    let _e210 = (_e200.xyz / vec3(_e200.w));
    let _e213 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e210.xy, _e210.z, vec2(5i));
    d = _e213;
    let _e214 = coord_7;
    let _e221 = coord_7;
    let _e226 = vec4<f32>(_e221.x, _e221.y, 1f, 6f);
    let _e231 = (_e226.xyz / vec3(_e226.w));
    let _e234 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e231.xy, _e231.z);
    d = _e234;
    let _e235 = coord_7;
    let _e244 = coord_7;
    let _e249 = vec4<f32>(_e244.x, _e244.y, 1f, 6f);
    let _e256 = (_e249.xyz / vec3(_e249.w));
    let _e259 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e256.xy, _e256.z, vec2(5i));
    d = _e259;
    let _e260 = coord_7;
    let _e268 = coord_7;
    let _e273 = vec4<f32>(_e268.x, _e268.y, 1f, 6f);
    let _e279 = (_e273.xyz / vec3(_e273.w));
    let _e282 = textureSampleCompare(tex2DShadow, sampShadow, _e279.xy, _e279.z, vec2(5i));
    d = _e282;
    return;
}

fn testTex2DArray(coord_8: vec3<f32>) {
    var coord_9: vec3<f32>;
    var size2DArray: vec3<i32>;
    var levels_4: i32;
    var c_3: vec4<f32>;

    coord_9 = coord_8;
    let _e20 = textureDimensions(tex2DArray, 0i);
    let _e23 = textureNumLayers(tex2DArray);
    size2DArray = vec3<i32>(vec3<u32>(_e20.x, _e20.y, _e23));
    let _e27 = textureNumLevels(tex2DArray);
    levels_4 = i32(_e27);
    let _e32 = coord_9;
    let _e36 = textureSample(tex2DArray, samp, _e32.xy, i32(_e32.z));
    c_3 = _e36;
    let _e39 = coord_9;
    let _e44 = textureSampleBias(tex2DArray, samp, _e39.xy, i32(_e39.z), 2f);
    c_3 = _e44;
    let _e50 = coord_9;
    let _e58 = textureSampleGrad(tex2DArray, samp, _e50.xy, i32(_e50.z), vec2(4f), vec2(4f));
    c_3 = _e58;
    let _e66 = coord_9;
    let _e76 = textureSampleGrad(tex2DArray, samp, _e66.xy, i32(_e66.z), vec2(4f), vec2(4f), vec2(5i));
    c_3 = _e76;
    let _e79 = coord_9;
    let _e84 = textureSampleLevel(tex2DArray, samp, _e79.xy, i32(_e79.z), 3f);
    c_3 = _e84;
    let _e89 = coord_9;
    let _e96 = textureSampleLevel(tex2DArray, samp, _e89.xy, i32(_e89.z), 3f, vec2(5i));
    c_3 = _e96;
    let _e100 = coord_9;
    let _e106 = textureSample(tex2DArray, samp, _e100.xy, i32(_e100.z), vec2(5i));
    c_3 = _e106;
    let _e111 = coord_9;
    let _e118 = textureSampleBias(tex2DArray, samp, _e111.xy, i32(_e111.z), 2f, vec2(5i));
    c_3 = _e118;
    let _e119 = coord_9;
    let _e122 = coord_9;
    let _e123 = vec3<i32>(_e122);
    let _e127 = textureLoad(tex2DArray, _e123.xy, _e123.z, 3i);
    c_3 = _e127;
    let _e128 = coord_9;
    let _e133 = coord_9;
    let _e134 = vec3<i32>(_e133);
    let _e140 = textureLoad(tex2DArray, _e134.xy, _e134.z, 3i);
    c_3 = _e140;
    return;
}

fn testTex2DArrayShadow(coord_10: vec3<f32>) {
    var coord_11: vec3<f32>;
    var size2DArrayShadow: vec3<i32>;
    var levels_5: i32;
    var d_1: f32;

    coord_11 = coord_10;
    let _e20 = textureDimensions(tex2DArrayShadow, 0i);
    let _e23 = textureNumLayers(tex2DArrayShadow);
    size2DArrayShadow = vec3<i32>(vec3<u32>(_e20.x, _e20.y, _e23));
    let _e27 = textureNumLevels(tex2DArrayShadow);
    levels_5 = i32(_e27);
    let _e31 = coord_11;
    let _e37 = coord_11;
    let _e42 = vec4<f32>(_e37.x, _e37.y, _e37.z, 1f);
    let _e47 = textureSampleCompare(tex2DArrayShadow, sampShadow, _e42.xy, i32(_e42.z), _e42.w);
    d_1 = _e47;
    let _e48 = coord_11;
    let _e58 = coord_11;
    let _e63 = vec4<f32>(_e58.x, _e58.y, _e58.z, 1f);
    let _e72 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e63.xy, i32(_e63.z), _e63.w);
    d_1 = _e72;
    let _e73 = coord_11;
    let _e85 = coord_11;
    let _e90 = vec4<f32>(_e85.x, _e85.y, _e85.z, 1f);
    let _e101 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e90.xy, i32(_e90.z), _e90.w, vec2(5i));
    d_1 = _e101;
    let _e102 = coord_11;
    let _e110 = coord_11;
    let _e115 = vec4<f32>(_e110.x, _e110.y, _e110.z, 1f);
    let _e122 = textureSampleCompare(tex2DArrayShadow, sampShadow, _e115.xy, i32(_e115.z), _e115.w, vec2(5i));
    d_1 = _e122;
    return;
}

fn testTexCube(coord_12: vec3<f32>) {
    var coord_13: vec3<f32>;
    var sizeCube: vec2<i32>;
    var levels_6: i32;
    var c_4: vec4<f32>;

    coord_13 = coord_12;
    let _e20 = textureDimensions(texCube, 0i);
    sizeCube = vec2<i32>(_e20);
    let _e23 = textureNumLevels(texCube);
    levels_6 = i32(_e23);
    let _e28 = coord_13;
    let _e29 = textureSample(texCube, samp, _e28);
    c_4 = _e29;
    let _e32 = coord_13;
    let _e34 = textureSampleBias(texCube, samp, _e32, 2f);
    c_4 = _e34;
    let _e40 = coord_13;
    let _e45 = textureSampleGrad(texCube, samp, _e40, vec3(4f), vec3(4f));
    c_4 = _e45;
    let _e48 = coord_13;
    let _e50 = textureSampleLevel(texCube, samp, _e48, 3f);
    c_4 = _e50;
    return;
}

fn testTexCubeShadow(coord_14: vec3<f32>) {
    var coord_15: vec3<f32>;
    var sizeCubeShadow: vec2<i32>;
    var levels_7: i32;
    var d_2: f32;

    coord_15 = coord_14;
    let _e20 = textureDimensions(texCubeShadow, 0i);
    sizeCubeShadow = vec2<i32>(_e20);
    let _e23 = textureNumLevels(texCubeShadow);
    levels_7 = i32(_e23);
    let _e27 = coord_15;
    let _e33 = coord_15;
    let _e38 = vec4<f32>(_e33.x, _e33.y, _e33.z, 1f);
    let _e41 = textureSampleCompare(texCubeShadow, sampShadow, _e38.xyz, _e38.w);
    d_2 = _e41;
    let _e42 = coord_15;
    let _e52 = coord_15;
    let _e57 = vec4<f32>(_e52.x, _e52.y, _e52.z, 1f);
    let _e64 = textureSampleCompareLevel(texCubeShadow, sampShadow, _e57.xyz, _e57.w);
    d_2 = _e64;
    return;
}

fn testTexCubeArray(coord_16: vec4<f32>) {
    var coord_17: vec4<f32>;
    var sizeCubeArray: vec3<i32>;
    var levels_8: i32;
    var c_5: vec4<f32>;

    coord_17 = coord_16;
    let _e20 = textureDimensions(texCubeArray, 0i);
    let _e23 = textureNumLayers(texCubeArray);
    sizeCubeArray = vec3<i32>(vec3<u32>(_e20.x, _e20.y, _e23));
    let _e27 = textureNumLevels(texCubeArray);
    levels_8 = i32(_e27);
    let _e32 = coord_17;
    let _e36 = textureSample(texCubeArray, samp, _e32.xyz, i32(_e32.w));
    c_5 = _e36;
    let _e39 = coord_17;
    let _e44 = textureSampleBias(texCubeArray, samp, _e39.xyz, i32(_e39.w), 2f);
    c_5 = _e44;
    let _e50 = coord_17;
    let _e58 = textureSampleGrad(texCubeArray, samp, _e50.xyz, i32(_e50.w), vec3(4f), vec3(4f));
    c_5 = _e58;
    let _e61 = coord_17;
    let _e66 = textureSampleLevel(texCubeArray, samp, _e61.xyz, i32(_e61.w), 3f);
    c_5 = _e66;
    return;
}

fn testTexCubeArrayShadow(coord_18: vec4<f32>) {
    var coord_19: vec4<f32>;
    var sizeCubeArrayShadow: vec3<i32>;
    var levels_9: i32;
    var d_3: f32;

    coord_19 = coord_18;
    let _e20 = textureDimensions(texCubeArrayShadow, 0i);
    let _e23 = textureNumLayers(texCubeArrayShadow);
    sizeCubeArrayShadow = vec3<i32>(vec3<u32>(_e20.x, _e20.y, _e23));
    let _e27 = textureNumLevels(texCubeArrayShadow);
    levels_9 = i32(_e27);
    let _e33 = coord_19;
    let _e38 = textureSampleCompare(texCubeArrayShadow, sampShadow, _e33.xyz, i32(_e33.w), 1f);
    d_3 = _e38;
    return;
}

fn testTex3D(coord_20: vec3<f32>) {
    var coord_21: vec3<f32>;
    var size3D: vec3<i32>;
    var levels_10: i32;
    var c_6: vec4<f32>;

    coord_21 = coord_20;
    let _e20 = textureDimensions(tex3D, 0i);
    size3D = vec3<i32>(_e20);
    let _e23 = textureNumLevels(tex3D);
    levels_10 = i32(_e23);
    let _e28 = coord_21;
    let _e29 = textureSample(tex3D, samp, _e28);
    c_6 = _e29;
    let _e32 = coord_21;
    let _e34 = textureSampleBias(tex3D, samp, _e32, 2f);
    c_6 = _e34;
    let _e35 = coord_21;
    let _e41 = coord_21;
    let _e46 = vec4<f32>(_e41.x, _e41.y, _e41.z, 6f);
    let _e51 = textureSample(tex3D, samp, (_e46.xyz / vec3(_e46.w)));
    c_6 = _e51;
    let _e52 = coord_21;
    let _e59 = coord_21;
    let _e64 = vec4<f32>(_e59.x, _e59.y, _e59.z, 6f);
    let _e70 = textureSampleBias(tex3D, samp, (_e64.xyz / vec3(_e64.w)), 2f);
    c_6 = _e70;
    let _e71 = coord_21;
    let _e79 = coord_21;
    let _e84 = vec4<f32>(_e79.x, _e79.y, _e79.z, 6f);
    let _e91 = textureSample(tex3D, samp, (_e84.xyz / vec3(_e84.w)), vec3(5i));
    c_6 = _e91;
    let _e92 = coord_21;
    let _e101 = coord_21;
    let _e106 = vec4<f32>(_e101.x, _e101.y, _e101.z, 6f);
    let _e114 = textureSampleBias(tex3D, samp, (_e106.xyz / vec3(_e106.w)), 2f, vec3(5i));
    c_6 = _e114;
    let _e115 = coord_21;
    let _e122 = coord_21;
    let _e127 = vec4<f32>(_e122.x, _e122.y, _e122.z, 6f);
    let _e133 = textureSampleLevel(tex3D, samp, (_e127.xyz / vec3(_e127.w)), 3f);
    c_6 = _e133;
    let _e134 = coord_21;
    let _e143 = coord_21;
    let _e148 = vec4<f32>(_e143.x, _e143.y, _e143.z, 6f);
    let _e156 = textureSampleLevel(tex3D, samp, (_e148.xyz / vec3(_e148.w)), 3f, vec3(5i));
    c_6 = _e156;
    let _e157 = coord_21;
    let _e167 = coord_21;
    let _e172 = vec4<f32>(_e167.x, _e167.y, _e167.z, 6f);
    let _e181 = textureSampleGrad(tex3D, samp, (_e172.xyz / vec3(_e172.w)), vec3(4f), vec3(4f));
    c_6 = _e181;
    let _e182 = coord_21;
    let _e194 = coord_21;
    let _e199 = vec4<f32>(_e194.x, _e194.y, _e194.z, 6f);
    let _e210 = textureSampleGrad(tex3D, samp, (_e199.xyz / vec3(_e199.w)), vec3(4f), vec3(4f), vec3(5i));
    c_6 = _e210;
    let _e216 = coord_21;
    let _e221 = textureSampleGrad(tex3D, samp, _e216, vec3(4f), vec3(4f));
    c_6 = _e221;
    let _e229 = coord_21;
    let _e236 = textureSampleGrad(tex3D, samp, _e229, vec3(4f), vec3(4f), vec3(5i));
    c_6 = _e236;
    let _e239 = coord_21;
    let _e241 = textureSampleLevel(tex3D, samp, _e239, 3f);
    c_6 = _e241;
    let _e246 = coord_21;
    let _e250 = textureSampleLevel(tex3D, samp, _e246, 3f, vec3(5i));
    c_6 = _e250;
    let _e254 = coord_21;
    let _e257 = textureSample(tex3D, samp, _e254, vec3(5i));
    c_6 = _e257;
    let _e262 = coord_21;
    let _e266 = textureSampleBias(tex3D, samp, _e262, 2f, vec3(5i));
    c_6 = _e266;
    let _e267 = coord_21;
    let _e270 = coord_21;
    let _e273 = textureLoad(tex3D, vec3<i32>(_e270), 3i);
    c_6 = _e273;
    let _e274 = coord_21;
    let _e279 = coord_21;
    let _e284 = textureLoad(tex3D, vec3<i32>(_e279), 3i);
    c_6 = _e284;
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
    let _e28 = textureLoad(tex2DMS, vec2<i32>(_e25), 3i);
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
    let _e34 = textureLoad(tex2DMSArray, _e30.xy, _e30.z, 3i);
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
