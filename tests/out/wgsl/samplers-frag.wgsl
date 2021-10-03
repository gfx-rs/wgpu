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
    let e18: f32 = coord1;
    let e19: vec4<f32> = textureSample(tex1D, samp, e18);
    c = e19;
    let e22: f32 = coord1;
    let e24: vec4<f32> = textureSampleBias(tex1D, samp, e22, 2.0);
    c = e24;
    let e28: f32 = coord1;
    let e31: vec4<f32> = textureSampleGrad(tex1D, samp, e28, 4.0, 4.0);
    c = e31;
    let e36: f32 = coord1;
    let e40: vec4<f32> = textureSampleGrad(tex1D, samp, e36, 4.0, 4.0, 5);
    c = e40;
    let e43: f32 = coord1;
    let e45: vec4<f32> = textureSampleLevel(tex1D, samp, e43, 3.0);
    c = e45;
    let e49: f32 = coord1;
    let e52: vec4<f32> = textureSampleLevel(tex1D, samp, e49, 3.0, 5);
    c = e52;
    let e55: f32 = coord1;
    let e57: vec4<f32> = textureSample(tex1D, samp, e55, 5);
    c = e57;
    let e61: f32 = coord1;
    let e64: vec4<f32> = textureSampleBias(tex1D, samp, e61, 2.0, 5);
    c = e64;
    let e65: f32 = coord1;
    let e68: f32 = coord1;
    let e70: vec2<f32> = vec2<f32>(e68, 6.0);
    let e74: vec4<f32> = textureSample(tex1D, samp, (e70.x / e70.y));
    c = e74;
    let e75: f32 = coord1;
    let e80: f32 = coord1;
    let e84: vec4<f32> = vec4<f32>(e80, 0.0, 0.0, 6.0);
    let e90: vec4<f32> = textureSample(tex1D, samp, (e84.xyz / vec3<f32>(e84.w)).x);
    c = e90;
    let e91: f32 = coord1;
    let e95: f32 = coord1;
    let e97: vec2<f32> = vec2<f32>(e95, 6.0);
    let e102: vec4<f32> = textureSampleBias(tex1D, samp, (e97.x / e97.y), 2.0);
    c = e102;
    let e103: f32 = coord1;
    let e109: f32 = coord1;
    let e113: vec4<f32> = vec4<f32>(e109, 0.0, 0.0, 6.0);
    let e120: vec4<f32> = textureSampleBias(tex1D, samp, (e113.xyz / vec3<f32>(e113.w)).x, 2.0);
    c = e120;
    let e121: f32 = coord1;
    let e126: f32 = coord1;
    let e128: vec2<f32> = vec2<f32>(e126, 6.0);
    let e134: vec4<f32> = textureSampleGrad(tex1D, samp, (e128.x / e128.y), 4.0, 4.0);
    c = e134;
    let e135: f32 = coord1;
    let e142: f32 = coord1;
    let e146: vec4<f32> = vec4<f32>(e142, 0.0, 0.0, 6.0);
    let e154: vec4<f32> = textureSampleGrad(tex1D, samp, (e146.xyz / vec3<f32>(e146.w)).x, 4.0, 4.0);
    c = e154;
    let e155: f32 = coord1;
    let e161: f32 = coord1;
    let e163: vec2<f32> = vec2<f32>(e161, 6.0);
    let e170: vec4<f32> = textureSampleGrad(tex1D, samp, (e163.x / e163.y), 4.0, 4.0, 5);
    c = e170;
    let e171: f32 = coord1;
    let e179: f32 = coord1;
    let e183: vec4<f32> = vec4<f32>(e179, 0.0, 0.0, 6.0);
    let e192: vec4<f32> = textureSampleGrad(tex1D, samp, (e183.xyz / vec3<f32>(e183.w)).x, 4.0, 4.0, 5);
    c = e192;
    let e193: f32 = coord1;
    let e197: f32 = coord1;
    let e199: vec2<f32> = vec2<f32>(e197, 6.0);
    let e204: vec4<f32> = textureSampleLevel(tex1D, samp, (e199.x / e199.y), 3.0);
    c = e204;
    let e205: f32 = coord1;
    let e211: f32 = coord1;
    let e215: vec4<f32> = vec4<f32>(e211, 0.0, 0.0, 6.0);
    let e222: vec4<f32> = textureSampleLevel(tex1D, samp, (e215.xyz / vec3<f32>(e215.w)).x, 3.0);
    c = e222;
    let e223: f32 = coord1;
    let e228: f32 = coord1;
    let e230: vec2<f32> = vec2<f32>(e228, 6.0);
    let e236: vec4<f32> = textureSampleLevel(tex1D, samp, (e230.x / e230.y), 3.0, 5);
    c = e236;
    let e237: f32 = coord1;
    let e244: f32 = coord1;
    let e248: vec4<f32> = vec4<f32>(e244, 0.0, 0.0, 6.0);
    let e256: vec4<f32> = textureSampleLevel(tex1D, samp, (e248.xyz / vec3<f32>(e248.w)).x, 3.0, 5);
    c = e256;
    let e257: f32 = coord1;
    let e261: f32 = coord1;
    let e263: vec2<f32> = vec2<f32>(e261, 6.0);
    let e268: vec4<f32> = textureSample(tex1D, samp, (e263.x / e263.y), 5);
    c = e268;
    let e269: f32 = coord1;
    let e275: f32 = coord1;
    let e279: vec4<f32> = vec4<f32>(e275, 0.0, 0.0, 6.0);
    let e286: vec4<f32> = textureSample(tex1D, samp, (e279.xyz / vec3<f32>(e279.w)).x, 5);
    c = e286;
    let e287: f32 = coord1;
    let e292: f32 = coord1;
    let e294: vec2<f32> = vec2<f32>(e292, 6.0);
    let e300: vec4<f32> = textureSampleBias(tex1D, samp, (e294.x / e294.y), 2.0, 5);
    c = e300;
    let e301: f32 = coord1;
    let e308: f32 = coord1;
    let e312: vec4<f32> = vec4<f32>(e308, 0.0, 0.0, 6.0);
    let e320: vec4<f32> = textureSampleBias(tex1D, samp, (e312.xyz / vec3<f32>(e312.w)).x, 2.0, 5);
    c = e320;
    return;
}

fn testTex1DArray(coord2: vec2<f32>) {
    var coord3: vec2<f32>;
    var c1: vec4<f32>;

    coord3 = coord2;
    let e18: vec2<f32> = coord3;
    let e22: vec4<f32> = textureSample(tex1DArray, samp, e18.x, i32(e18.y));
    c1 = e22;
    let e25: vec2<f32> = coord3;
    let e30: vec4<f32> = textureSampleBias(tex1DArray, samp, e25.x, i32(e25.y), 2.0);
    c1 = e30;
    let e34: vec2<f32> = coord3;
    let e40: vec4<f32> = textureSampleGrad(tex1DArray, samp, e34.x, i32(e34.y), 4.0, 4.0);
    c1 = e40;
    let e45: vec2<f32> = coord3;
    let e52: vec4<f32> = textureSampleGrad(tex1DArray, samp, e45.x, i32(e45.y), 4.0, 4.0, 5);
    c1 = e52;
    let e55: vec2<f32> = coord3;
    let e60: vec4<f32> = textureSampleLevel(tex1DArray, samp, e55.x, i32(e55.y), 3.0);
    c1 = e60;
    let e64: vec2<f32> = coord3;
    let e70: vec4<f32> = textureSampleLevel(tex1DArray, samp, e64.x, i32(e64.y), 3.0, 5);
    c1 = e70;
    let e73: vec2<f32> = coord3;
    let e78: vec4<f32> = textureSample(tex1DArray, samp, e73.x, i32(e73.y), 5);
    c1 = e78;
    let e82: vec2<f32> = coord3;
    let e88: vec4<f32> = textureSampleBias(tex1DArray, samp, e82.x, i32(e82.y), 2.0, 5);
    c1 = e88;
    return;
}

fn testTex2D(coord4: vec2<f32>) {
    var coord5: vec2<f32>;
    var c2: vec4<f32>;

    coord5 = coord4;
    let e18: vec2<f32> = coord5;
    let e19: vec4<f32> = textureSample(tex2D, samp, e18);
    c2 = e19;
    let e22: vec2<f32> = coord5;
    let e24: vec4<f32> = textureSampleBias(tex2D, samp, e22, 2.0);
    c2 = e24;
    let e30: vec2<f32> = coord5;
    let e35: vec4<f32> = textureSampleGrad(tex2D, samp, e30, vec2<f32>(4.0), vec2<f32>(4.0));
    c2 = e35;
    let e43: vec2<f32> = coord5;
    let e50: vec4<f32> = textureSampleGrad(tex2D, samp, e43, vec2<f32>(4.0), vec2<f32>(4.0), vec2<i32>(5, 5));
    c2 = e50;
    let e53: vec2<f32> = coord5;
    let e55: vec4<f32> = textureSampleLevel(tex2D, samp, e53, 3.0);
    c2 = e55;
    let e60: vec2<f32> = coord5;
    let e64: vec4<f32> = textureSampleLevel(tex2D, samp, e60, 3.0, vec2<i32>(5, 5));
    c2 = e64;
    let e68: vec2<f32> = coord5;
    let e71: vec4<f32> = textureSample(tex2D, samp, e68, vec2<i32>(5, 5));
    c2 = e71;
    let e76: vec2<f32> = coord5;
    let e80: vec4<f32> = textureSampleBias(tex2D, samp, e76, 2.0, vec2<i32>(5, 5));
    c2 = e80;
    let e81: vec2<f32> = coord5;
    let e84: vec2<f32> = coord5;
    let e86: vec3<f32> = vec3<f32>(e84, 6.0);
    let e91: vec4<f32> = textureSample(tex2D, samp, (e86.xy / vec2<f32>(e86.z)));
    c2 = e91;
    let e92: vec2<f32> = coord5;
    let e96: vec2<f32> = coord5;
    let e99: vec4<f32> = vec4<f32>(e96, 0.0, 6.0);
    let e105: vec4<f32> = textureSample(tex2D, samp, (e99.xyz / vec3<f32>(e99.w)).xy);
    c2 = e105;
    let e106: vec2<f32> = coord5;
    let e110: vec2<f32> = coord5;
    let e112: vec3<f32> = vec3<f32>(e110, 6.0);
    let e118: vec4<f32> = textureSampleBias(tex2D, samp, (e112.xy / vec2<f32>(e112.z)), 2.0);
    c2 = e118;
    let e119: vec2<f32> = coord5;
    let e124: vec2<f32> = coord5;
    let e127: vec4<f32> = vec4<f32>(e124, 0.0, 6.0);
    let e134: vec4<f32> = textureSampleBias(tex2D, samp, (e127.xyz / vec3<f32>(e127.w)).xy, 2.0);
    c2 = e134;
    let e135: vec2<f32> = coord5;
    let e142: vec2<f32> = coord5;
    let e144: vec3<f32> = vec3<f32>(e142, 6.0);
    let e153: vec4<f32> = textureSampleGrad(tex2D, samp, (e144.xy / vec2<f32>(e144.z)), vec2<f32>(4.0), vec2<f32>(4.0));
    c2 = e153;
    let e154: vec2<f32> = coord5;
    let e162: vec2<f32> = coord5;
    let e165: vec4<f32> = vec4<f32>(e162, 0.0, 6.0);
    let e175: vec4<f32> = textureSampleGrad(tex2D, samp, (e165.xyz / vec3<f32>(e165.w)).xy, vec2<f32>(4.0), vec2<f32>(4.0));
    c2 = e175;
    let e176: vec2<f32> = coord5;
    let e185: vec2<f32> = coord5;
    let e187: vec3<f32> = vec3<f32>(e185, 6.0);
    let e198: vec4<f32> = textureSampleGrad(tex2D, samp, (e187.xy / vec2<f32>(e187.z)), vec2<f32>(4.0), vec2<f32>(4.0), vec2<i32>(5, 5));
    c2 = e198;
    let e199: vec2<f32> = coord5;
    let e209: vec2<f32> = coord5;
    let e212: vec4<f32> = vec4<f32>(e209, 0.0, 6.0);
    let e224: vec4<f32> = textureSampleGrad(tex2D, samp, (e212.xyz / vec3<f32>(e212.w)).xy, vec2<f32>(4.0), vec2<f32>(4.0), vec2<i32>(5, 5));
    c2 = e224;
    let e225: vec2<f32> = coord5;
    let e229: vec2<f32> = coord5;
    let e231: vec3<f32> = vec3<f32>(e229, 6.0);
    let e237: vec4<f32> = textureSampleLevel(tex2D, samp, (e231.xy / vec2<f32>(e231.z)), 3.0);
    c2 = e237;
    let e238: vec2<f32> = coord5;
    let e243: vec2<f32> = coord5;
    let e246: vec4<f32> = vec4<f32>(e243, 0.0, 6.0);
    let e253: vec4<f32> = textureSampleLevel(tex2D, samp, (e246.xyz / vec3<f32>(e246.w)).xy, 3.0);
    c2 = e253;
    let e254: vec2<f32> = coord5;
    let e260: vec2<f32> = coord5;
    let e262: vec3<f32> = vec3<f32>(e260, 6.0);
    let e270: vec4<f32> = textureSampleLevel(tex2D, samp, (e262.xy / vec2<f32>(e262.z)), 3.0, vec2<i32>(5, 5));
    c2 = e270;
    let e271: vec2<f32> = coord5;
    let e278: vec2<f32> = coord5;
    let e281: vec4<f32> = vec4<f32>(e278, 0.0, 6.0);
    let e290: vec4<f32> = textureSampleLevel(tex2D, samp, (e281.xyz / vec3<f32>(e281.w)).xy, 3.0, vec2<i32>(5, 5));
    c2 = e290;
    let e291: vec2<f32> = coord5;
    let e296: vec2<f32> = coord5;
    let e298: vec3<f32> = vec3<f32>(e296, 6.0);
    let e305: vec4<f32> = textureSample(tex2D, samp, (e298.xy / vec2<f32>(e298.z)), vec2<i32>(5, 5));
    c2 = e305;
    let e306: vec2<f32> = coord5;
    let e312: vec2<f32> = coord5;
    let e315: vec4<f32> = vec4<f32>(e312, 0.0, 6.0);
    let e323: vec4<f32> = textureSample(tex2D, samp, (e315.xyz / vec3<f32>(e315.w)).xy, vec2<i32>(5, 5));
    c2 = e323;
    let e324: vec2<f32> = coord5;
    let e330: vec2<f32> = coord5;
    let e332: vec3<f32> = vec3<f32>(e330, 6.0);
    let e340: vec4<f32> = textureSampleBias(tex2D, samp, (e332.xy / vec2<f32>(e332.z)), 2.0, vec2<i32>(5, 5));
    c2 = e340;
    let e341: vec2<f32> = coord5;
    let e348: vec2<f32> = coord5;
    let e351: vec4<f32> = vec4<f32>(e348, 0.0, 6.0);
    let e360: vec4<f32> = textureSampleBias(tex2D, samp, (e351.xyz / vec3<f32>(e351.w)).xy, 2.0, vec2<i32>(5, 5));
    c2 = e360;
    return;
}

fn testTex2DShadow(coord6: vec2<f32>) {
    var coord7: vec2<f32>;
    var d: f32;

    coord7 = coord6;
    let e17: vec2<f32> = coord7;
    let e20: vec2<f32> = coord7;
    let e22: vec3<f32> = vec3<f32>(e20, 1.0);
    let e25: f32 = textureSampleCompare(tex2DShadow, sampShadow, e22.xy, e22.z);
    d = e25;
    let e26: vec2<f32> = coord7;
    let e33: vec2<f32> = coord7;
    let e35: vec3<f32> = vec3<f32>(e33, 1.0);
    let e42: f32 = textureSampleCompareLevel(tex2DShadow, sampShadow, e35.xy, e35.z);
    d = e42;
    let e43: vec2<f32> = coord7;
    let e52: vec2<f32> = coord7;
    let e54: vec3<f32> = vec3<f32>(e52, 1.0);
    let e63: f32 = textureSampleCompareLevel(tex2DShadow, sampShadow, e54.xy, e54.z, vec2<i32>(5, 5));
    d = e63;
    let e64: vec2<f32> = coord7;
    let e68: vec2<f32> = coord7;
    let e70: vec3<f32> = vec3<f32>(e68, 1.0);
    let e74: f32 = textureSampleCompareLevel(tex2DShadow, sampShadow, e70.xy, e70.z);
    d = e74;
    let e75: vec2<f32> = coord7;
    let e81: vec2<f32> = coord7;
    let e83: vec3<f32> = vec3<f32>(e81, 1.0);
    let e89: f32 = textureSampleCompareLevel(tex2DShadow, sampShadow, e83.xy, e83.z, vec2<i32>(5, 5));
    d = e89;
    let e90: vec2<f32> = coord7;
    let e95: vec2<f32> = coord7;
    let e97: vec3<f32> = vec3<f32>(e95, 1.0);
    let e102: f32 = textureSampleCompare(tex2DShadow, sampShadow, e97.xy, e97.z, vec2<i32>(5, 5));
    d = e102;
    let e103: vec2<f32> = coord7;
    let e107: vec2<f32> = coord7;
    let e110: vec4<f32> = vec4<f32>(e107, 1.0, 6.0);
    let e114: vec3<f32> = (e110.xyz / vec3<f32>(e110.w));
    let e117: f32 = textureSampleCompare(tex2DShadow, sampShadow, e114.xy, e114.z);
    d = e117;
    let e118: vec2<f32> = coord7;
    let e126: vec2<f32> = coord7;
    let e129: vec4<f32> = vec4<f32>(e126, 1.0, 6.0);
    let e137: vec3<f32> = (e129.xyz / vec3<f32>(e129.w));
    let e140: f32 = textureSampleCompareLevel(tex2DShadow, sampShadow, e137.xy, e137.z);
    d = e140;
    let e141: vec2<f32> = coord7;
    let e151: vec2<f32> = coord7;
    let e154: vec4<f32> = vec4<f32>(e151, 1.0, 6.0);
    let e164: vec3<f32> = (e154.xyz / vec3<f32>(e154.w));
    let e167: f32 = textureSampleCompareLevel(tex2DShadow, sampShadow, e164.xy, e164.z, vec2<i32>(5, 5));
    d = e167;
    let e168: vec2<f32> = coord7;
    let e173: vec2<f32> = coord7;
    let e176: vec4<f32> = vec4<f32>(e173, 1.0, 6.0);
    let e181: vec3<f32> = (e176.xyz / vec3<f32>(e176.w));
    let e184: f32 = textureSampleCompareLevel(tex2DShadow, sampShadow, e181.xy, e181.z);
    d = e184;
    let e185: vec2<f32> = coord7;
    let e192: vec2<f32> = coord7;
    let e195: vec4<f32> = vec4<f32>(e192, 1.0, 6.0);
    let e202: vec3<f32> = (e195.xyz / vec3<f32>(e195.w));
    let e205: f32 = textureSampleCompareLevel(tex2DShadow, sampShadow, e202.xy, e202.z, vec2<i32>(5, 5));
    d = e205;
    let e206: vec2<f32> = coord7;
    let e212: vec2<f32> = coord7;
    let e215: vec4<f32> = vec4<f32>(e212, 1.0, 6.0);
    let e221: vec3<f32> = (e215.xyz / vec3<f32>(e215.w));
    let e224: f32 = textureSampleCompare(tex2DShadow, sampShadow, e221.xy, e221.z, vec2<i32>(5, 5));
    d = e224;
    return;
}

fn testTex2DArray(coord8: vec3<f32>) {
    var coord9: vec3<f32>;
    var c3: vec4<f32>;

    coord9 = coord8;
    let e18: vec3<f32> = coord9;
    let e22: vec4<f32> = textureSample(tex2DArray, samp, e18.xy, i32(e18.z));
    c3 = e22;
    let e25: vec3<f32> = coord9;
    let e30: vec4<f32> = textureSampleBias(tex2DArray, samp, e25.xy, i32(e25.z), 2.0);
    c3 = e30;
    let e36: vec3<f32> = coord9;
    let e44: vec4<f32> = textureSampleGrad(tex2DArray, samp, e36.xy, i32(e36.z), vec2<f32>(4.0), vec2<f32>(4.0));
    c3 = e44;
    let e52: vec3<f32> = coord9;
    let e62: vec4<f32> = textureSampleGrad(tex2DArray, samp, e52.xy, i32(e52.z), vec2<f32>(4.0), vec2<f32>(4.0), vec2<i32>(5, 5));
    c3 = e62;
    let e65: vec3<f32> = coord9;
    let e70: vec4<f32> = textureSampleLevel(tex2DArray, samp, e65.xy, i32(e65.z), 3.0);
    c3 = e70;
    let e75: vec3<f32> = coord9;
    let e82: vec4<f32> = textureSampleLevel(tex2DArray, samp, e75.xy, i32(e75.z), 3.0, vec2<i32>(5, 5));
    c3 = e82;
    let e86: vec3<f32> = coord9;
    let e92: vec4<f32> = textureSample(tex2DArray, samp, e86.xy, i32(e86.z), vec2<i32>(5, 5));
    c3 = e92;
    let e97: vec3<f32> = coord9;
    let e104: vec4<f32> = textureSampleBias(tex2DArray, samp, e97.xy, i32(e97.z), 2.0, vec2<i32>(5, 5));
    c3 = e104;
    return;
}

fn testTex2DArrayShadow(coord10: vec3<f32>) {
    var coord11: vec3<f32>;
    var d1: f32;

    coord11 = coord10;
    let e17: vec3<f32> = coord11;
    let e20: vec3<f32> = coord11;
    let e22: vec4<f32> = vec4<f32>(e20, 1.0);
    let e27: f32 = textureSampleCompare(tex2DArrayShadow, sampShadow, e22.xy, i32(e22.z), e22.w);
    d1 = e27;
    let e28: vec3<f32> = coord11;
    let e35: vec3<f32> = coord11;
    let e37: vec4<f32> = vec4<f32>(e35, 1.0);
    let e46: f32 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, e37.xy, i32(e37.z), e37.w);
    d1 = e46;
    let e47: vec3<f32> = coord11;
    let e56: vec3<f32> = coord11;
    let e58: vec4<f32> = vec4<f32>(e56, 1.0);
    let e69: f32 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, e58.xy, i32(e58.z), e58.w, vec2<i32>(5, 5));
    d1 = e69;
    let e70: vec3<f32> = coord11;
    let e74: vec3<f32> = coord11;
    let e76: vec4<f32> = vec4<f32>(e74, 1.0);
    let e82: f32 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, e76.xy, i32(e76.z), e76.w);
    d1 = e82;
    let e83: vec3<f32> = coord11;
    let e89: vec3<f32> = coord11;
    let e91: vec4<f32> = vec4<f32>(e89, 1.0);
    let e99: f32 = textureSampleCompareLevel(tex2DArrayShadow, sampShadow, e91.xy, i32(e91.z), e91.w, vec2<i32>(5, 5));
    d1 = e99;
    let e100: vec3<f32> = coord11;
    let e105: vec3<f32> = coord11;
    let e107: vec4<f32> = vec4<f32>(e105, 1.0);
    let e114: f32 = textureSampleCompare(tex2DArrayShadow, sampShadow, e107.xy, i32(e107.z), e107.w, vec2<i32>(5, 5));
    d1 = e114;
    return;
}

fn testTexCube(coord12: vec3<f32>) {
    var coord13: vec3<f32>;
    var c4: vec4<f32>;

    coord13 = coord12;
    let e18: vec3<f32> = coord13;
    let e19: vec4<f32> = textureSample(texCube, samp, e18);
    c4 = e19;
    let e22: vec3<f32> = coord13;
    let e24: vec4<f32> = textureSampleBias(texCube, samp, e22, 2.0);
    c4 = e24;
    let e30: vec3<f32> = coord13;
    let e35: vec4<f32> = textureSampleGrad(texCube, samp, e30, vec3<f32>(4.0), vec3<f32>(4.0));
    c4 = e35;
    let e38: vec3<f32> = coord13;
    let e40: vec4<f32> = textureSampleLevel(texCube, samp, e38, 3.0);
    c4 = e40;
    let e45: vec3<f32> = coord13;
    let e49: vec4<f32> = textureSampleLevel(texCube, samp, e45, 3.0, vec3<i32>(5, 5, 5));
    c4 = e49;
    let e53: vec3<f32> = coord13;
    let e56: vec4<f32> = textureSample(texCube, samp, e53, vec3<i32>(5, 5, 5));
    c4 = e56;
    let e61: vec3<f32> = coord13;
    let e65: vec4<f32> = textureSampleBias(texCube, samp, e61, 2.0, vec3<i32>(5, 5, 5));
    c4 = e65;
    return;
}

fn testTexCubeShadow(coord14: vec3<f32>) {
    var coord15: vec3<f32>;
    var d2: f32;

    coord15 = coord14;
    let e17: vec3<f32> = coord15;
    let e20: vec3<f32> = coord15;
    let e22: vec4<f32> = vec4<f32>(e20, 1.0);
    let e25: f32 = textureSampleCompare(texCubeShadow, sampShadow, e22.xyz, e22.w);
    d2 = e25;
    let e26: vec3<f32> = coord15;
    let e33: vec3<f32> = coord15;
    let e35: vec4<f32> = vec4<f32>(e33, 1.0);
    let e42: f32 = textureSampleCompareLevel(texCubeShadow, sampShadow, e35.xyz, e35.w);
    d2 = e42;
    let e43: vec3<f32> = coord15;
    let e47: vec3<f32> = coord15;
    let e49: vec4<f32> = vec4<f32>(e47, 1.0);
    let e53: f32 = textureSampleCompareLevel(texCubeShadow, sampShadow, e49.xyz, e49.w);
    d2 = e53;
    let e54: vec3<f32> = coord15;
    let e60: vec3<f32> = coord15;
    let e62: vec4<f32> = vec4<f32>(e60, 1.0);
    let e68: f32 = textureSampleCompareLevel(texCubeShadow, sampShadow, e62.xyz, e62.w, vec3<i32>(5, 5, 5));
    d2 = e68;
    let e69: vec3<f32> = coord15;
    let e74: vec3<f32> = coord15;
    let e76: vec4<f32> = vec4<f32>(e74, 1.0);
    let e81: f32 = textureSampleCompare(texCubeShadow, sampShadow, e76.xyz, e76.w, vec3<i32>(5, 5, 5));
    d2 = e81;
    return;
}

fn testTexCubeArray(coord16: vec4<f32>) {
    var coord17: vec4<f32>;
    var c5: vec4<f32>;

    coord17 = coord16;
    let e18: vec4<f32> = coord17;
    let e22: vec4<f32> = textureSample(texCubeArray, samp, e18.xyz, i32(e18.w));
    c5 = e22;
    let e25: vec4<f32> = coord17;
    let e30: vec4<f32> = textureSampleBias(texCubeArray, samp, e25.xyz, i32(e25.w), 2.0);
    c5 = e30;
    let e36: vec4<f32> = coord17;
    let e44: vec4<f32> = textureSampleGrad(texCubeArray, samp, e36.xyz, i32(e36.w), vec3<f32>(4.0), vec3<f32>(4.0));
    c5 = e44;
    let e47: vec4<f32> = coord17;
    let e52: vec4<f32> = textureSampleLevel(texCubeArray, samp, e47.xyz, i32(e47.w), 3.0);
    c5 = e52;
    let e57: vec4<f32> = coord17;
    let e64: vec4<f32> = textureSampleLevel(texCubeArray, samp, e57.xyz, i32(e57.w), 3.0, vec3<i32>(5, 5, 5));
    c5 = e64;
    let e68: vec4<f32> = coord17;
    let e74: vec4<f32> = textureSample(texCubeArray, samp, e68.xyz, i32(e68.w), vec3<i32>(5, 5, 5));
    c5 = e74;
    let e79: vec4<f32> = coord17;
    let e86: vec4<f32> = textureSampleBias(texCubeArray, samp, e79.xyz, i32(e79.w), 2.0, vec3<i32>(5, 5, 5));
    c5 = e86;
    return;
}

fn testTexCubeArrayShadow(coord18: vec4<f32>) {
    var coord19: vec4<f32>;
    var d3: f32;

    coord19 = coord18;
    let e19: vec4<f32> = coord19;
    let e24: f32 = textureSampleCompare(texCubeArrayShadow, sampShadow, e19.xyz, i32(e19.w), 1.0);
    d3 = e24;
    return;
}

fn testTex3D(coord20: vec3<f32>) {
    var coord21: vec3<f32>;
    var c6: vec4<f32>;

    coord21 = coord20;
    let e18: vec3<f32> = coord21;
    let e19: vec4<f32> = textureSample(tex3D, samp, e18);
    c6 = e19;
    let e22: vec3<f32> = coord21;
    let e24: vec4<f32> = textureSampleBias(tex3D, samp, e22, 2.0);
    c6 = e24;
    let e30: vec3<f32> = coord21;
    let e35: vec4<f32> = textureSampleGrad(tex3D, samp, e30, vec3<f32>(4.0), vec3<f32>(4.0));
    c6 = e35;
    let e43: vec3<f32> = coord21;
    let e50: vec4<f32> = textureSampleGrad(tex3D, samp, e43, vec3<f32>(4.0), vec3<f32>(4.0), vec3<i32>(5, 5, 5));
    c6 = e50;
    let e53: vec3<f32> = coord21;
    let e55: vec4<f32> = textureSampleLevel(tex3D, samp, e53, 3.0);
    c6 = e55;
    let e60: vec3<f32> = coord21;
    let e64: vec4<f32> = textureSampleLevel(tex3D, samp, e60, 3.0, vec3<i32>(5, 5, 5));
    c6 = e64;
    let e68: vec3<f32> = coord21;
    let e71: vec4<f32> = textureSample(tex3D, samp, e68, vec3<i32>(5, 5, 5));
    c6 = e71;
    let e76: vec3<f32> = coord21;
    let e80: vec4<f32> = textureSampleBias(tex3D, samp, e76, 2.0, vec3<i32>(5, 5, 5));
    c6 = e80;
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
