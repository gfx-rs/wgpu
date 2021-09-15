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
var<private> texcoord1: vec4<f32>;

fn testTex1D(coord: f32) {
    var coord1: f32;
    var c: vec4<f32>;

    coord1 = coord;
    let _e18: f32 = coord1;
    let _e19: vec4<f32> = textureSample(tex1D, samp, _e18);
    c = _e19;
    let _e22: f32 = coord1;
    let _e24: vec4<f32> = textureSampleBias(tex1D, samp, _e22, 2.0);
    c = _e24;
    let _e28: f32 = coord1;
    let _e31: vec4<f32> = textureSampleGrad(tex1D, samp, _e28, 4.0, 4.0);
    c = _e31;
    let _e36: f32 = coord1;
    let _e40: vec4<f32> = textureSampleGrad(tex1D, samp, _e36, 4.0, 4.0, 5);
    c = _e40;
    let _e43: f32 = coord1;
    let _e45: vec4<f32> = textureSampleLevel(tex1D, samp, _e43, 3.0);
    c = _e45;
    let _e49: f32 = coord1;
    let _e52: vec4<f32> = textureSampleLevel(tex1D, samp, _e49, 3.0, 5);
    c = _e52;
    let _e55: f32 = coord1;
    let _e57: vec4<f32> = textureSample(tex1D, samp, _e55, 5);
    c = _e57;
    let _e61: f32 = coord1;
    let _e64: vec4<f32> = textureSampleBias(tex1D, samp, _e61, 2.0, 5);
    c = _e64;
    let _e65: f32 = coord1;
    let _e68: f32 = coord1;
    let _e70: vec2<f32> = vec2<f32>(_e68, 6.0);
    let _e74: vec4<f32> = textureSample(tex1D, samp, (_e70.x / _e70.y));
    c = _e74;
    let _e75: f32 = coord1;
    let _e80: f32 = coord1;
    let _e84: vec4<f32> = vec4<f32>(_e80, 0.0, 0.0, 6.0);
    let _e90: vec4<f32> = textureSample(tex1D, samp, (_e84.xyz / vec3<f32>(_e84.w)).x);
    c = _e90;
    let _e91: f32 = coord1;
    let _e95: f32 = coord1;
    let _e97: vec2<f32> = vec2<f32>(_e95, 6.0);
    let _e102: vec4<f32> = textureSampleBias(tex1D, samp, (_e97.x / _e97.y), 2.0);
    c = _e102;
    let _e103: f32 = coord1;
    let _e109: f32 = coord1;
    let _e113: vec4<f32> = vec4<f32>(_e109, 0.0, 0.0, 6.0);
    let _e120: vec4<f32> = textureSampleBias(tex1D, samp, (_e113.xyz / vec3<f32>(_e113.w)).x, 2.0);
    c = _e120;
    let _e121: f32 = coord1;
    let _e126: f32 = coord1;
    let _e128: vec2<f32> = vec2<f32>(_e126, 6.0);
    let _e134: vec4<f32> = textureSampleGrad(tex1D, samp, (_e128.x / _e128.y), 4.0, 4.0);
    c = _e134;
    let _e135: f32 = coord1;
    let _e142: f32 = coord1;
    let _e146: vec4<f32> = vec4<f32>(_e142, 0.0, 0.0, 6.0);
    let _e154: vec4<f32> = textureSampleGrad(tex1D, samp, (_e146.xyz / vec3<f32>(_e146.w)).x, 4.0, 4.0);
    c = _e154;
    let _e155: f32 = coord1;
    let _e161: f32 = coord1;
    let _e163: vec2<f32> = vec2<f32>(_e161, 6.0);
    let _e170: vec4<f32> = textureSampleGrad(tex1D, samp, (_e163.x / _e163.y), 4.0, 4.0, 5);
    c = _e170;
    let _e171: f32 = coord1;
    let _e179: f32 = coord1;
    let _e183: vec4<f32> = vec4<f32>(_e179, 0.0, 0.0, 6.0);
    let _e192: vec4<f32> = textureSampleGrad(tex1D, samp, (_e183.xyz / vec3<f32>(_e183.w)).x, 4.0, 4.0, 5);
    c = _e192;
    let _e193: f32 = coord1;
    let _e197: f32 = coord1;
    let _e199: vec2<f32> = vec2<f32>(_e197, 6.0);
    let _e204: vec4<f32> = textureSampleLevel(tex1D, samp, (_e199.x / _e199.y), 3.0);
    c = _e204;
    let _e205: f32 = coord1;
    let _e211: f32 = coord1;
    let _e215: vec4<f32> = vec4<f32>(_e211, 0.0, 0.0, 6.0);
    let _e222: vec4<f32> = textureSampleLevel(tex1D, samp, (_e215.xyz / vec3<f32>(_e215.w)).x, 3.0);
    c = _e222;
    let _e223: f32 = coord1;
    let _e228: f32 = coord1;
    let _e230: vec2<f32> = vec2<f32>(_e228, 6.0);
    let _e236: vec4<f32> = textureSampleLevel(tex1D, samp, (_e230.x / _e230.y), 3.0, 5);
    c = _e236;
    let _e237: f32 = coord1;
    let _e244: f32 = coord1;
    let _e248: vec4<f32> = vec4<f32>(_e244, 0.0, 0.0, 6.0);
    let _e256: vec4<f32> = textureSampleLevel(tex1D, samp, (_e248.xyz / vec3<f32>(_e248.w)).x, 3.0, 5);
    c = _e256;
    let _e257: f32 = coord1;
    let _e261: f32 = coord1;
    let _e263: vec2<f32> = vec2<f32>(_e261, 6.0);
    let _e268: vec4<f32> = textureSample(tex1D, samp, (_e263.x / _e263.y), 5);
    c = _e268;
    let _e269: f32 = coord1;
    let _e275: f32 = coord1;
    let _e279: vec4<f32> = vec4<f32>(_e275, 0.0, 0.0, 6.0);
    let _e286: vec4<f32> = textureSample(tex1D, samp, (_e279.xyz / vec3<f32>(_e279.w)).x, 5);
    c = _e286;
    let _e287: f32 = coord1;
    let _e292: f32 = coord1;
    let _e294: vec2<f32> = vec2<f32>(_e292, 6.0);
    let _e300: vec4<f32> = textureSampleBias(tex1D, samp, (_e294.x / _e294.y), 2.0, 5);
    c = _e300;
    let _e301: f32 = coord1;
    let _e308: f32 = coord1;
    let _e312: vec4<f32> = vec4<f32>(_e308, 0.0, 0.0, 6.0);
    let _e320: vec4<f32> = textureSampleBias(tex1D, samp, (_e312.xyz / vec3<f32>(_e312.w)).x, 2.0, 5);
    c = _e320;
    return;
}

fn testTex1DArray(coord2: vec2<f32>) {
    var coord3: vec2<f32>;
    var c1: vec4<f32>;

    coord3 = coord2;
    let _e18: vec2<f32> = coord3;
    let _e22: vec4<f32> = textureSample(tex1DArray, samp, _e18.x, i32(_e18.x));
    c1 = _e22;
    let _e25: vec2<f32> = coord3;
    let _e30: vec4<f32> = textureSampleBias(tex1DArray, samp, _e25.x, i32(_e25.x), 2.0);
    c1 = _e30;
    let _e34: vec2<f32> = coord3;
    let _e40: vec4<f32> = textureSampleGrad(tex1DArray, samp, _e34.x, i32(_e34.x), 4.0, 4.0);
    c1 = _e40;
    let _e45: vec2<f32> = coord3;
    let _e52: vec4<f32> = textureSampleGrad(tex1DArray, samp, _e45.x, i32(_e45.x), 4.0, 4.0, 5);
    c1 = _e52;
    let _e55: vec2<f32> = coord3;
    let _e60: vec4<f32> = textureSampleLevel(tex1DArray, samp, _e55.x, i32(_e55.x), 3.0);
    c1 = _e60;
    let _e64: vec2<f32> = coord3;
    let _e70: vec4<f32> = textureSampleLevel(tex1DArray, samp, _e64.x, i32(_e64.x), 3.0, 5);
    c1 = _e70;
    let _e73: vec2<f32> = coord3;
    let _e78: vec4<f32> = textureSample(tex1DArray, samp, _e73.x, i32(_e73.x), 5);
    c1 = _e78;
    let _e82: vec2<f32> = coord3;
    let _e88: vec4<f32> = textureSampleBias(tex1DArray, samp, _e82.x, i32(_e82.x), 2.0, 5);
    c1 = _e88;
    return;
}

fn testTex2D(coord4: vec2<f32>) {
    var coord5: vec2<f32>;
    var c2: vec4<f32>;

    coord5 = coord4;
    let _e18: vec2<f32> = coord5;
    let _e19: vec4<f32> = textureSample(tex2D, samp, _e18);
    c2 = _e19;
    let _e22: vec2<f32> = coord5;
    let _e24: vec4<f32> = textureSampleBias(tex2D, samp, _e22, 2.0);
    c2 = _e24;
    let _e30: vec2<f32> = coord5;
    let _e35: vec4<f32> = textureSampleGrad(tex2D, samp, _e30, vec2<f32>(4.0), vec2<f32>(4.0));
    c2 = _e35;
    let _e43: vec2<f32> = coord5;
    let _e50: vec4<f32> = textureSampleGrad(tex2D, samp, _e43, vec2<f32>(4.0), vec2<f32>(4.0), vec2<i32>(5, 5));
    c2 = _e50;
    let _e53: vec2<f32> = coord5;
    let _e55: vec4<f32> = textureSampleLevel(tex2D, samp, _e53, 3.0);
    c2 = _e55;
    let _e60: vec2<f32> = coord5;
    let _e64: vec4<f32> = textureSampleLevel(tex2D, samp, _e60, 3.0, vec2<i32>(5, 5));
    c2 = _e64;
    let _e68: vec2<f32> = coord5;
    let _e71: vec4<f32> = textureSample(tex2D, samp, _e68, vec2<i32>(5, 5));
    c2 = _e71;
    let _e76: vec2<f32> = coord5;
    let _e80: vec4<f32> = textureSampleBias(tex2D, samp, _e76, 2.0, vec2<i32>(5, 5));
    c2 = _e80;
    let _e81: vec2<f32> = coord5;
    let _e84: vec2<f32> = coord5;
    let _e86: vec3<f32> = vec3<f32>(_e84, 6.0);
    let _e91: vec4<f32> = textureSample(tex2D, samp, (_e86.xy / vec2<f32>(_e86.z)));
    c2 = _e91;
    let _e92: vec2<f32> = coord5;
    let _e96: vec2<f32> = coord5;
    let _e99: vec4<f32> = vec4<f32>(_e96, 0.0, 6.0);
    let _e105: vec4<f32> = textureSample(tex2D, samp, (_e99.xyz / vec3<f32>(_e99.w)).xy);
    c2 = _e105;
    let _e106: vec2<f32> = coord5;
    let _e110: vec2<f32> = coord5;
    let _e112: vec3<f32> = vec3<f32>(_e110, 6.0);
    let _e118: vec4<f32> = textureSampleBias(tex2D, samp, (_e112.xy / vec2<f32>(_e112.z)), 2.0);
    c2 = _e118;
    let _e119: vec2<f32> = coord5;
    let _e124: vec2<f32> = coord5;
    let _e127: vec4<f32> = vec4<f32>(_e124, 0.0, 6.0);
    let _e134: vec4<f32> = textureSampleBias(tex2D, samp, (_e127.xyz / vec3<f32>(_e127.w)).xy, 2.0);
    c2 = _e134;
    let _e135: vec2<f32> = coord5;
    let _e142: vec2<f32> = coord5;
    let _e144: vec3<f32> = vec3<f32>(_e142, 6.0);
    let _e153: vec4<f32> = textureSampleGrad(tex2D, samp, (_e144.xy / vec2<f32>(_e144.z)), vec2<f32>(4.0), vec2<f32>(4.0));
    c2 = _e153;
    let _e154: vec2<f32> = coord5;
    let _e162: vec2<f32> = coord5;
    let _e165: vec4<f32> = vec4<f32>(_e162, 0.0, 6.0);
    let _e175: vec4<f32> = textureSampleGrad(tex2D, samp, (_e165.xyz / vec3<f32>(_e165.w)).xy, vec2<f32>(4.0), vec2<f32>(4.0));
    c2 = _e175;
    let _e176: vec2<f32> = coord5;
    let _e185: vec2<f32> = coord5;
    let _e187: vec3<f32> = vec3<f32>(_e185, 6.0);
    let _e198: vec4<f32> = textureSampleGrad(tex2D, samp, (_e187.xy / vec2<f32>(_e187.z)), vec2<f32>(4.0), vec2<f32>(4.0), vec2<i32>(5, 5));
    c2 = _e198;
    let _e199: vec2<f32> = coord5;
    let _e209: vec2<f32> = coord5;
    let _e212: vec4<f32> = vec4<f32>(_e209, 0.0, 6.0);
    let _e224: vec4<f32> = textureSampleGrad(tex2D, samp, (_e212.xyz / vec3<f32>(_e212.w)).xy, vec2<f32>(4.0), vec2<f32>(4.0), vec2<i32>(5, 5));
    c2 = _e224;
    let _e225: vec2<f32> = coord5;
    let _e229: vec2<f32> = coord5;
    let _e231: vec3<f32> = vec3<f32>(_e229, 6.0);
    let _e237: vec4<f32> = textureSampleLevel(tex2D, samp, (_e231.xy / vec2<f32>(_e231.z)), 3.0);
    c2 = _e237;
    let _e238: vec2<f32> = coord5;
    let _e243: vec2<f32> = coord5;
    let _e246: vec4<f32> = vec4<f32>(_e243, 0.0, 6.0);
    let _e253: vec4<f32> = textureSampleLevel(tex2D, samp, (_e246.xyz / vec3<f32>(_e246.w)).xy, 3.0);
    c2 = _e253;
    let _e254: vec2<f32> = coord5;
    let _e260: vec2<f32> = coord5;
    let _e262: vec3<f32> = vec3<f32>(_e260, 6.0);
    let _e270: vec4<f32> = textureSampleLevel(tex2D, samp, (_e262.xy / vec2<f32>(_e262.z)), 3.0, vec2<i32>(5, 5));
    c2 = _e270;
    let _e271: vec2<f32> = coord5;
    let _e278: vec2<f32> = coord5;
    let _e281: vec4<f32> = vec4<f32>(_e278, 0.0, 6.0);
    let _e290: vec4<f32> = textureSampleLevel(tex2D, samp, (_e281.xyz / vec3<f32>(_e281.w)).xy, 3.0, vec2<i32>(5, 5));
    c2 = _e290;
    let _e291: vec2<f32> = coord5;
    let _e296: vec2<f32> = coord5;
    let _e298: vec3<f32> = vec3<f32>(_e296, 6.0);
    let _e305: vec4<f32> = textureSample(tex2D, samp, (_e298.xy / vec2<f32>(_e298.z)), vec2<i32>(5, 5));
    c2 = _e305;
    let _e306: vec2<f32> = coord5;
    let _e312: vec2<f32> = coord5;
    let _e315: vec4<f32> = vec4<f32>(_e312, 0.0, 6.0);
    let _e323: vec4<f32> = textureSample(tex2D, samp, (_e315.xyz / vec3<f32>(_e315.w)).xy, vec2<i32>(5, 5));
    c2 = _e323;
    let _e324: vec2<f32> = coord5;
    let _e330: vec2<f32> = coord5;
    let _e332: vec3<f32> = vec3<f32>(_e330, 6.0);
    let _e340: vec4<f32> = textureSampleBias(tex2D, samp, (_e332.xy / vec2<f32>(_e332.z)), 2.0, vec2<i32>(5, 5));
    c2 = _e340;
    let _e341: vec2<f32> = coord5;
    let _e348: vec2<f32> = coord5;
    let _e351: vec4<f32> = vec4<f32>(_e348, 0.0, 6.0);
    let _e360: vec4<f32> = textureSampleBias(tex2D, samp, (_e351.xyz / vec3<f32>(_e351.w)).xy, 2.0, vec2<i32>(5, 5));
    c2 = _e360;
    return;
}

fn testTex2DShadow(coord6: vec2<f32>) {
    var coord7: vec2<f32>;
    var d: f32;

    coord7 = coord6;
    let _e17: vec2<f32> = coord7;
    let _e20: vec2<f32> = coord7;
    let _e22: vec3<f32> = vec3<f32>(_e20, 1.0);
    let _e25: f32 = textureSampleCompare(tex2DShadow, sampShadow, _e22.xy, _e22.z);
    d = _e25;
    let _e26: vec2<f32> = coord7;
    let _e33: vec2<f32> = coord7;
    let _e35: vec3<f32> = vec3<f32>(_e33, 1.0);
    let _e42: f32 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e35.xy, _e35.z);
    d = _e42;
    let _e43: vec2<f32> = coord7;
    let _e52: vec2<f32> = coord7;
    let _e54: vec3<f32> = vec3<f32>(_e52, 1.0);
    let _e63: f32 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e54.xy, _e54.z, vec2<i32>(5, 5));
    d = _e63;
    let _e64: vec2<f32> = coord7;
    let _e68: vec2<f32> = coord7;
    let _e70: vec3<f32> = vec3<f32>(_e68, 1.0);
    let _e74: f32 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e70.xy, _e70.z);
    d = _e74;
    let _e75: vec2<f32> = coord7;
    let _e81: vec2<f32> = coord7;
    let _e83: vec3<f32> = vec3<f32>(_e81, 1.0);
    let _e89: f32 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e83.xy, _e83.z, vec2<i32>(5, 5));
    d = _e89;
    let _e90: vec2<f32> = coord7;
    let _e95: vec2<f32> = coord7;
    let _e97: vec3<f32> = vec3<f32>(_e95, 1.0);
    let _e102: f32 = textureSampleCompare(tex2DShadow, sampShadow, _e97.xy, _e97.z, vec2<i32>(5, 5));
    d = _e102;
    let _e103: vec2<f32> = coord7;
    let _e107: vec2<f32> = coord7;
    let _e110: vec4<f32> = vec4<f32>(_e107, 1.0, 6.0);
    let _e114: vec3<f32> = (_e110.xyz / vec3<f32>(_e110.w));
    let _e117: f32 = textureSampleCompare(tex2DShadow, sampShadow, _e114.xy, _e114.z);
    d = _e117;
    let _e118: vec2<f32> = coord7;
    let _e126: vec2<f32> = coord7;
    let _e129: vec4<f32> = vec4<f32>(_e126, 1.0, 6.0);
    let _e137: vec3<f32> = (_e129.xyz / vec3<f32>(_e129.w));
    let _e140: f32 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e137.xy, _e137.z);
    d = _e140;
    let _e141: vec2<f32> = coord7;
    let _e151: vec2<f32> = coord7;
    let _e154: vec4<f32> = vec4<f32>(_e151, 1.0, 6.0);
    let _e164: vec3<f32> = (_e154.xyz / vec3<f32>(_e154.w));
    let _e167: f32 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e164.xy, _e164.z, vec2<i32>(5, 5));
    d = _e167;
    let _e168: vec2<f32> = coord7;
    let _e173: vec2<f32> = coord7;
    let _e176: vec4<f32> = vec4<f32>(_e173, 1.0, 6.0);
    let _e181: vec3<f32> = (_e176.xyz / vec3<f32>(_e176.w));
    let _e184: f32 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e181.xy, _e181.z);
    d = _e184;
    let _e185: vec2<f32> = coord7;
    let _e192: vec2<f32> = coord7;
    let _e195: vec4<f32> = vec4<f32>(_e192, 1.0, 6.0);
    let _e202: vec3<f32> = (_e195.xyz / vec3<f32>(_e195.w));
    let _e205: f32 = textureSampleCompareLevel(tex2DShadow, sampShadow, _e202.xy, _e202.z, vec2<i32>(5, 5));
    d = _e205;
    let _e206: vec2<f32> = coord7;
    let _e212: vec2<f32> = coord7;
    let _e215: vec4<f32> = vec4<f32>(_e212, 1.0, 6.0);
    let _e221: vec3<f32> = (_e215.xyz / vec3<f32>(_e215.w));
    let _e224: f32 = textureSampleCompare(tex2DShadow, sampShadow, _e221.xy, _e221.z, vec2<i32>(5, 5));
    d = _e224;
    return;
}

fn testTex2DArray(coord8: vec3<f32>) {
    var coord9: vec3<f32>;
    var c3: vec4<f32>;

    coord9 = coord8;
    let _e18: vec3<f32> = coord9;
    let _e22: vec4<f32> = textureSample(tex2DArray, samp, _e18.xy, i32(_e18.z));
    c3 = _e22;
    let _e25: vec3<f32> = coord9;
    let _e30: vec4<f32> = textureSampleBias(tex2DArray, samp, _e25.xy, i32(_e25.z), 2.0);
    c3 = _e30;
    let _e36: vec3<f32> = coord9;
    let _e44: vec4<f32> = textureSampleGrad(tex2DArray, samp, _e36.xy, i32(_e36.z), vec2<f32>(4.0), vec2<f32>(4.0));
    c3 = _e44;
    let _e52: vec3<f32> = coord9;
    let _e62: vec4<f32> = textureSampleGrad(tex2DArray, samp, _e52.xy, i32(_e52.z), vec2<f32>(4.0), vec2<f32>(4.0), vec2<i32>(5, 5));
    c3 = _e62;
    let _e65: vec3<f32> = coord9;
    let _e70: vec4<f32> = textureSampleLevel(tex2DArray, samp, _e65.xy, i32(_e65.z), 3.0);
    c3 = _e70;
    let _e75: vec3<f32> = coord9;
    let _e82: vec4<f32> = textureSampleLevel(tex2DArray, samp, _e75.xy, i32(_e75.z), 3.0, vec2<i32>(5, 5));
    c3 = _e82;
    let _e86: vec3<f32> = coord9;
    let _e92: vec4<f32> = textureSample(tex2DArray, samp, _e86.xy, i32(_e86.z), vec2<i32>(5, 5));
    c3 = _e92;
    let _e97: vec3<f32> = coord9;
    let _e104: vec4<f32> = textureSampleBias(tex2DArray, samp, _e97.xy, i32(_e97.z), 2.0, vec2<i32>(5, 5));
    c3 = _e104;
    return;
}

fn testTex2DArrayShadow(coord10: vec3<f32>) {
    var coord11: vec3<f32>;
    var d1: f32;

    coord11 = coord10;
    let _e17: vec3<f32> = coord11;
    let _e20: vec3<f32> = coord11;
    let _e22: vec4<f32> = vec4<f32>(_e20, 1.0);
    let _e27: f32 = textureSampleCompare(tex2DArrayShadow, sampShadow, _e22.xy, i32(_e22.z), _e22.w);
    d1 = _e27;
    let _e28: vec3<f32> = coord11;
    let _e35: vec3<f32> = coord11;
    let _e37: vec4<f32> = vec4<f32>(_e35, 1.0);
    let _e46: f32 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e37.xy, i32(_e37.z), _e37.w);
    d1 = _e46;
    let _e47: vec3<f32> = coord11;
    let _e56: vec3<f32> = coord11;
    let _e58: vec4<f32> = vec4<f32>(_e56, 1.0);
    let _e69: f32 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e58.xy, i32(_e58.z), _e58.w, vec2<i32>(5, 5));
    d1 = _e69;
    let _e70: vec3<f32> = coord11;
    let _e74: vec3<f32> = coord11;
    let _e76: vec4<f32> = vec4<f32>(_e74, 1.0);
    let _e82: f32 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e76.xy, i32(_e76.z), _e76.w);
    d1 = _e82;
    let _e83: vec3<f32> = coord11;
    let _e89: vec3<f32> = coord11;
    let _e91: vec4<f32> = vec4<f32>(_e89, 1.0);
    let _e99: f32 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, _e91.xy, i32(_e91.z), _e91.w, vec2<i32>(5, 5));
    d1 = _e99;
    let _e100: vec3<f32> = coord11;
    let _e105: vec3<f32> = coord11;
    let _e107: vec4<f32> = vec4<f32>(_e105, 1.0);
    let _e114: f32 = textureSampleCompare(tex2DArrayShadow, sampShadow, _e107.xy, i32(_e107.z), _e107.w, vec2<i32>(5, 5));
    d1 = _e114;
    return;
}

fn testTexCube(coord12: vec3<f32>) {
    var coord13: vec3<f32>;
    var c4: vec4<f32>;

    coord13 = coord12;
    let _e18: vec3<f32> = coord13;
    let _e19: vec4<f32> = textureSample(texCube, samp, _e18);
    c4 = _e19;
    let _e22: vec3<f32> = coord13;
    let _e24: vec4<f32> = textureSampleBias(texCube, samp, _e22, 2.0);
    c4 = _e24;
    let _e30: vec3<f32> = coord13;
    let _e35: vec4<f32> = textureSampleGrad(texCube, samp, _e30, vec3<f32>(4.0), vec3<f32>(4.0));
    c4 = _e35;
    let _e38: vec3<f32> = coord13;
    let _e40: vec4<f32> = textureSampleLevel(texCube, samp, _e38, 3.0);
    c4 = _e40;
    let _e45: vec3<f32> = coord13;
    let _e49: vec4<f32> = textureSampleLevel(texCube, samp, _e45, 3.0, vec3<i32>(5, 5, 5));
    c4 = _e49;
    let _e53: vec3<f32> = coord13;
    let _e56: vec4<f32> = textureSample(texCube, samp, _e53, vec3<i32>(5, 5, 5));
    c4 = _e56;
    let _e61: vec3<f32> = coord13;
    let _e65: vec4<f32> = textureSampleBias(texCube, samp, _e61, 2.0, vec3<i32>(5, 5, 5));
    c4 = _e65;
    return;
}

fn testTexCubeShadow(coord14: vec3<f32>) {
    var coord15: vec3<f32>;
    var d2: f32;

    coord15 = coord14;
    let _e17: vec3<f32> = coord15;
    let _e20: vec3<f32> = coord15;
    let _e22: vec4<f32> = vec4<f32>(_e20, 1.0);
    let _e25: f32 = textureSampleCompare(texCubeShadow, sampShadow, _e22.xyz, _e22.w);
    d2 = _e25;
    let _e26: vec3<f32> = coord15;
    let _e33: vec3<f32> = coord15;
    let _e35: vec4<f32> = vec4<f32>(_e33, 1.0);
    let _e42: f32 = textureSampleCompareLevel(texCubeShadow, sampShadow, _e35.xyz, _e35.w);
    d2 = _e42;
    let _e43: vec3<f32> = coord15;
    let _e47: vec3<f32> = coord15;
    let _e49: vec4<f32> = vec4<f32>(_e47, 1.0);
    let _e53: f32 = textureSampleCompareLevel(texCubeShadow, sampShadow, _e49.xyz, _e49.w);
    d2 = _e53;
    let _e54: vec3<f32> = coord15;
    let _e60: vec3<f32> = coord15;
    let _e62: vec4<f32> = vec4<f32>(_e60, 1.0);
    let _e68: f32 = textureSampleCompareLevel(texCubeShadow, sampShadow, _e62.xyz, _e62.w, vec3<i32>(5, 5, 5));
    d2 = _e68;
    let _e69: vec3<f32> = coord15;
    let _e74: vec3<f32> = coord15;
    let _e76: vec4<f32> = vec4<f32>(_e74, 1.0);
    let _e81: f32 = textureSampleCompare(texCubeShadow, sampShadow, _e76.xyz, _e76.w, vec3<i32>(5, 5, 5));
    d2 = _e81;
    return;
}

fn testTexCubeArray(coord16: vec4<f32>) {
    var coord17: vec4<f32>;
    var c5: vec4<f32>;

    coord17 = coord16;
    let _e18: vec4<f32> = coord17;
    let _e22: vec4<f32> = textureSample(texCubeArray, samp, _e18.xyz, i32(_e18.w));
    c5 = _e22;
    let _e25: vec4<f32> = coord17;
    let _e30: vec4<f32> = textureSampleBias(texCubeArray, samp, _e25.xyz, i32(_e25.w), 2.0);
    c5 = _e30;
    let _e36: vec4<f32> = coord17;
    let _e44: vec4<f32> = textureSampleGrad(texCubeArray, samp, _e36.xyz, i32(_e36.w), vec3<f32>(4.0), vec3<f32>(4.0));
    c5 = _e44;
    let _e47: vec4<f32> = coord17;
    let _e52: vec4<f32> = textureSampleLevel(texCubeArray, samp, _e47.xyz, i32(_e47.w), 3.0);
    c5 = _e52;
    let _e57: vec4<f32> = coord17;
    let _e64: vec4<f32> = textureSampleLevel(texCubeArray, samp, _e57.xyz, i32(_e57.w), 3.0, vec3<i32>(5, 5, 5));
    c5 = _e64;
    let _e68: vec4<f32> = coord17;
    let _e74: vec4<f32> = textureSample(texCubeArray, samp, _e68.xyz, i32(_e68.w), vec3<i32>(5, 5, 5));
    c5 = _e74;
    let _e79: vec4<f32> = coord17;
    let _e86: vec4<f32> = textureSampleBias(texCubeArray, samp, _e79.xyz, i32(_e79.w), 2.0, vec3<i32>(5, 5, 5));
    c5 = _e86;
    return;
}

fn testTexCubeArrayShadow(coord18: vec4<f32>) {
    var coord19: vec4<f32>;
    var d3: f32;

    coord19 = coord18;
    let _e19: vec4<f32> = coord19;
    let _e24: f32 = textureSampleCompare(texCubeArrayShadow, sampShadow, _e19.xyz, i32(_e19.w), 1.0);
    d3 = _e24;
    return;
}

fn testTex3D(coord20: vec3<f32>) {
    var coord21: vec3<f32>;
    var c6: vec4<f32>;

    coord21 = coord20;
    let _e18: vec3<f32> = coord21;
    let _e19: vec4<f32> = textureSample(tex3D, samp, _e18);
    c6 = _e19;
    let _e22: vec3<f32> = coord21;
    let _e24: vec4<f32> = textureSampleBias(tex3D, samp, _e22, 2.0);
    c6 = _e24;
    let _e30: vec3<f32> = coord21;
    let _e35: vec4<f32> = textureSampleGrad(tex3D, samp, _e30, vec3<f32>(4.0), vec3<f32>(4.0));
    c6 = _e35;
    let _e43: vec3<f32> = coord21;
    let _e50: vec4<f32> = textureSampleGrad(tex3D, samp, _e43, vec3<f32>(4.0), vec3<f32>(4.0), vec3<i32>(5, 5, 5));
    c6 = _e50;
    let _e53: vec3<f32> = coord21;
    let _e55: vec4<f32> = textureSampleLevel(tex3D, samp, _e53, 3.0);
    c6 = _e55;
    let _e60: vec3<f32> = coord21;
    let _e64: vec4<f32> = textureSampleLevel(tex3D, samp, _e60, 3.0, vec3<i32>(5, 5, 5));
    c6 = _e64;
    let _e68: vec3<f32> = coord21;
    let _e71: vec4<f32> = textureSample(tex3D, samp, _e68, vec3<i32>(5, 5, 5));
    c6 = _e71;
    let _e76: vec3<f32> = coord21;
    let _e80: vec4<f32> = textureSampleBias(tex3D, samp, _e76, 2.0, vec3<i32>(5, 5, 5));
    c6 = _e80;
    return;
}

fn main1() {
    return;
}

[[stage(fragment)]]
fn main([[location(0)]] texcoord: vec4<f32>) {
    texcoord1 = texcoord;
    main1();
    return;
}
