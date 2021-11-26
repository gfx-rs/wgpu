[[group(1), binding(0)]]
var tex1D: texture_1d<f32>;
[[group(1), binding(1)]]
var tex1DArray: texture_1d_array<f32>;
[[group(1), binding(2)]]
var tex2D: texture_2d<f32>;
[[group(1), binding(3)]]
var tex2DArray: texture_2d_array<f32>;
[[group(1), binding(4)]]
var texCube: texture_cube<f32>;
[[group(1), binding(5)]]
var texCubeArray: texture_cube_array<f32>;
[[group(1), binding(6)]]
var tex3D: texture_3d<f32>;
[[group(1), binding(7)]]
var samp: sampler;
[[group(1), binding(12)]]
var tex2DShadow: texture_depth_2d;
[[group(1), binding(13)]]
var tex2DArrayShadow: texture_depth_2d_array;
[[group(1), binding(14)]]
var texCubeShadow: texture_depth_cube;
[[group(1), binding(15)]]
var texCubeArrayShadow: texture_depth_cube_array;
[[group(1), binding(16)]]
var tex3DShadow: texture_3d<f32>;
[[group(1), binding(17)]]
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
    let _e84 = coord_5;
    let _e86 = vec3<f32>(_e84, 6.0);
    let _e91 = textureSample(tex2D, samp, (_e86.xy / vec2<f32>(_e86.z)));
    c_2 = _e91;
    let _e92 = coord_5;
    let _e96 = coord_5;
    let _e99 = vec4<f32>(_e96, 0.0, 6.0);
    let _e105 = textureSample(tex2D, samp, (_e99.xyz / vec3<f32>(_e99.w)).xy);
    c_2 = _e105;
    let _e106 = coord_5;
    let _e110 = coord_5;
    let _e112 = vec3<f32>(_e110, 6.0);
    let _e118 = textureSampleBias(tex2D, samp, (_e112.xy / vec2<f32>(_e112.z)), 2.0);
    c_2 = _e118;
    let _e119 = coord_5;
    let _e124 = coord_5;
    let _e127 = vec4<f32>(_e124, 0.0, 6.0);
    let _e134 = textureSampleBias(tex2D, samp, (_e127.xyz / vec3<f32>(_e127.w)).xy, 2.0);
    c_2 = _e134;
    let _e135 = coord_5;
    let _e142 = coord_5;
    let _e144 = vec3<f32>(_e142, 6.0);
    let _e153 = textureSampleGrad(tex2D, samp, (_e144.xy / vec2<f32>(_e144.z)), vec2<f32>(4.0), vec2<f32>(4.0));
    c_2 = _e153;
    let _e154 = coord_5;
    let _e162 = coord_5;
    let _e165 = vec4<f32>(_e162, 0.0, 6.0);
    let _e175 = textureSampleGrad(tex2D, samp, (_e165.xyz / vec3<f32>(_e165.w)).xy, vec2<f32>(4.0), vec2<f32>(4.0));
    c_2 = _e175;
    let _e176 = coord_5;
    let _e185 = coord_5;
    let _e187 = vec3<f32>(_e185, 6.0);
    let _e198 = textureSampleGrad(tex2D, samp, (_e187.xy / vec2<f32>(_e187.z)), vec2<f32>(4.0), vec2<f32>(4.0), vec2<i32>(5, 5));
    c_2 = _e198;
    let _e199 = coord_5;
    let _e209 = coord_5;
    let _e212 = vec4<f32>(_e209, 0.0, 6.0);
    let _e224 = textureSampleGrad(tex2D, samp, (_e212.xyz / vec3<f32>(_e212.w)).xy, vec2<f32>(4.0), vec2<f32>(4.0), vec2<i32>(5, 5));
    c_2 = _e224;
    let _e225 = coord_5;
    let _e229 = coord_5;
    let _e231 = vec3<f32>(_e229, 6.0);
    let _e237 = textureSampleLevel(tex2D, samp, (_e231.xy / vec2<f32>(_e231.z)), 3.0);
    c_2 = _e237;
    let _e238 = coord_5;
    let _e243 = coord_5;
    let _e246 = vec4<f32>(_e243, 0.0, 6.0);
    let _e253 = textureSampleLevel(tex2D, samp, (_e246.xyz / vec3<f32>(_e246.w)).xy, 3.0);
    c_2 = _e253;
    let _e254 = coord_5;
    let _e260 = coord_5;
    let _e262 = vec3<f32>(_e260, 6.0);
    let _e270 = textureSampleLevel(tex2D, samp, (_e262.xy / vec2<f32>(_e262.z)), 3.0, vec2<i32>(5, 5));
    c_2 = _e270;
    let _e271 = coord_5;
    let _e278 = coord_5;
    let _e281 = vec4<f32>(_e278, 0.0, 6.0);
    let _e290 = textureSampleLevel(tex2D, samp, (_e281.xyz / vec3<f32>(_e281.w)).xy, 3.0, vec2<i32>(5, 5));
    c_2 = _e290;
    let _e291 = coord_5;
    let _e296 = coord_5;
    let _e298 = vec3<f32>(_e296, 6.0);
    let _e305 = textureSample(tex2D, samp, (_e298.xy / vec2<f32>(_e298.z)), vec2<i32>(5, 5));
    c_2 = _e305;
    let _e306 = coord_5;
    let _e312 = coord_5;
    let _e315 = vec4<f32>(_e312, 0.0, 6.0);
    let _e323 = textureSample(tex2D, samp, (_e315.xyz / vec3<f32>(_e315.w)).xy, vec2<i32>(5, 5));
    c_2 = _e323;
    let _e324 = coord_5;
    let _e330 = coord_5;
    let _e332 = vec3<f32>(_e330, 6.0);
    let _e340 = textureSampleBias(tex2D, samp, (_e332.xy / vec2<f32>(_e332.z)), 2.0, vec2<i32>(5, 5));
    c_2 = _e340;
    let _e341 = coord_5;
    let _e348 = coord_5;
    let _e351 = vec4<f32>(_e348, 0.0, 6.0);
    let _e360 = textureSampleBias(tex2D, samp, (_e351.xyz / vec3<f32>(_e351.w)).xy, 2.0, vec2<i32>(5, 5));
    c_2 = _e360;
    return;
}

fn testTex2DShadow(coord_6: vec2<f32>) {
    var coord_7: vec2<f32>;
    var d: f32;

    coord_7 = coord_6;
    let _e17 = coord_7;
    let _e20 = coord_7;
    let _e22 = vec3<f32>(_e20, 1.0);
    let _e25 = textureSampleCompare(tex2DShadow, sampShadow, _e22.xy, _e22.z);
    d = _e25;
    let _e26 = coord_7;
    let _e33 = coord_7;
    let _e35 = vec3<f32>(_e33, 1.0);
    let _e42 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e35.xy, _e35.z);
    d = _e42;
    let _e43 = coord_7;
    let _e52 = coord_7;
    let _e54 = vec3<f32>(_e52, 1.0);
    let _e63 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e54.xy, _e54.z, vec2<i32>(5, 5));
    d = _e63;
    let _e64 = coord_7;
    let _e68 = coord_7;
    let _e70 = vec3<f32>(_e68, 1.0);
    let _e74 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e70.xy, _e70.z);
    d = _e74;
    let _e75 = coord_7;
    let _e81 = coord_7;
    let _e83 = vec3<f32>(_e81, 1.0);
    let _e89 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e83.xy, _e83.z, vec2<i32>(5, 5));
    d = _e89;
    let _e90 = coord_7;
    let _e95 = coord_7;
    let _e97 = vec3<f32>(_e95, 1.0);
    let _e102 = textureSampleCompare(tex2DShadow, sampShadow, _e97.xy, _e97.z, vec2<i32>(5, 5));
    d = _e102;
    let _e103 = coord_7;
    let _e107 = coord_7;
    let _e110 = vec4<f32>(_e107, 1.0, 6.0);
    let _e114 = (_e110.xyz / vec3<f32>(_e110.w));
    let _e117 = textureSampleCompare(tex2DShadow, sampShadow, _e114.xy, _e114.z);
    d = _e117;
    let _e118 = coord_7;
    let _e126 = coord_7;
    let _e129 = vec4<f32>(_e126, 1.0, 6.0);
    let _e137 = (_e129.xyz / vec3<f32>(_e129.w));
    let _e140 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e137.xy, _e137.z);
    d = _e140;
    let _e141 = coord_7;
    let _e151 = coord_7;
    let _e154 = vec4<f32>(_e151, 1.0, 6.0);
    let _e164 = (_e154.xyz / vec3<f32>(_e154.w));
    let _e167 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e164.xy, _e164.z, vec2<i32>(5, 5));
    d = _e167;
    let _e168 = coord_7;
    let _e173 = coord_7;
    let _e176 = vec4<f32>(_e173, 1.0, 6.0);
    let _e181 = (_e176.xyz / vec3<f32>(_e176.w));
    let _e184 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e181.xy, _e181.z);
    d = _e184;
    let _e185 = coord_7;
    let _e192 = coord_7;
    let _e195 = vec4<f32>(_e192, 1.0, 6.0);
    let _e202 = (_e195.xyz / vec3<f32>(_e195.w));
    let _e205 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e202.xy, _e202.z, vec2<i32>(5, 5));
    d = _e205;
    let _e206 = coord_7;
    let _e212 = coord_7;
    let _e215 = vec4<f32>(_e212, 1.0, 6.0);
    let _e221 = (_e215.xyz / vec3<f32>(_e215.w));
    let _e224 = textureSampleCompare(tex2DShadow, sampShadow, _e221.xy, _e221.z, vec2<i32>(5, 5));
    d = _e224;
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
    let _e20 = coord_11;
    let _e22 = vec4<f32>(_e20, 1.0);
    let _e27 = textureSampleCompare(tex2DArrayShadow, sampShadow, _e22.xy, i32(_e22.z), _e22.w);
    d_1 = _e27;
    let _e28 = coord_11;
    let _e35 = coord_11;
    let _e37 = vec4<f32>(_e35, 1.0);
    let _e46 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e37.xy, i32(_e37.z), _e37.w);
    d_1 = _e46;
    let _e47 = coord_11;
    let _e56 = coord_11;
    let _e58 = vec4<f32>(_e56, 1.0);
    let _e69 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e58.xy, i32(_e58.z), _e58.w, vec2<i32>(5, 5));
    d_1 = _e69;
    let _e70 = coord_11;
    let _e74 = coord_11;
    let _e76 = vec4<f32>(_e74, 1.0);
    let _e82 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e76.xy, i32(_e76.z), _e76.w);
    d_1 = _e82;
    let _e83 = coord_11;
    let _e89 = coord_11;
    let _e91 = vec4<f32>(_e89, 1.0);
    let _e99 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e91.xy, i32(_e91.z), _e91.w, vec2<i32>(5, 5));
    d_1 = _e99;
    let _e100 = coord_11;
    let _e105 = coord_11;
    let _e107 = vec4<f32>(_e105, 1.0);
    let _e114 = textureSampleCompare(tex2DArrayShadow, sampShadow, _e107.xy, i32(_e107.z), _e107.w, vec2<i32>(5, 5));
    d_1 = _e114;
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
    let _e20 = coord_15;
    let _e22 = vec4<f32>(_e20, 1.0);
    let _e25 = textureSampleCompare(texCubeShadow, sampShadow, _e22.xyz, _e22.w);
    d_2 = _e25;
    let _e26 = coord_15;
    let _e33 = coord_15;
    let _e35 = vec4<f32>(_e33, 1.0);
    let _e42 = textureSampleCompareLevel(texCubeShadow, sampShadow, _e35.xyz, _e35.w);
    d_2 = _e42;
    let _e43 = coord_15;
    let _e47 = coord_15;
    let _e49 = vec4<f32>(_e47, 1.0);
    let _e53 = textureSampleCompareLevel(texCubeShadow, sampShadow, _e49.xyz, _e49.w);
    d_2 = _e53;
    let _e54 = coord_15;
    let _e60 = coord_15;
    let _e62 = vec4<f32>(_e60, 1.0);
    let _e68 = textureSampleCompareLevel(texCubeShadow, sampShadow, _e62.xyz, _e62.w, vec3<i32>(5, 5, 5));
    d_2 = _e68;
    let _e69 = coord_15;
    let _e74 = coord_15;
    let _e76 = vec4<f32>(_e74, 1.0);
    let _e81 = textureSampleCompare(texCubeShadow, sampShadow, _e76.xyz, _e76.w, vec3<i32>(5, 5, 5));
    d_2 = _e81;
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

[[stage(fragment)]]
fn main([[location(0)]] texcoord: vec4<f32>) {
    texcoord_1 = texcoord;
    main_1();
    return;
}
