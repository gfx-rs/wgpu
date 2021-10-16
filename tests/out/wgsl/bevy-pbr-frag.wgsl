struct PointLight {
    pos: vec4<f32>;
    color: vec4<f32>;
    lightParams: vec4<f32>;
};

struct DirectionalLight {
    direction: vec4<f32>;
    color: vec4<f32>;
};

[[block]]
struct CameraViewProj {
    ViewProj: mat4x4<f32>;
};

[[block]]
struct CameraPosition {
    CameraPos: vec4<f32>;
};

[[block]]
struct Lights {
    AmbientColor: vec4<f32>;
    NumLights: vec4<u32>;
    PointLights: [[stride(48)]] array<PointLight,10u>;
    DirectionalLights: [[stride(32)]] array<DirectionalLight,1u>;
};

[[block]]
struct StandardMaterial_base_color {
    base_color: vec4<f32>;
};

[[block]]
struct StandardMaterial_roughness {
    perceptual_roughness: f32;
};

[[block]]
struct StandardMaterial_metallic {
    metallic: f32;
};

[[block]]
struct StandardMaterial_reflectance {
    reflectance: f32;
};

[[block]]
struct StandardMaterial_emissive {
    emissive: vec4<f32>;
};

struct FragmentOutput {
    [[location(0)]] o_Target: vec4<f32>;
};

var<private> v_WorldPosition1: vec3<f32>;
var<private> v_WorldNormal1: vec3<f32>;
var<private> v_Uv1: vec2<f32>;
var<private> v_WorldTangent1: vec4<f32>;
var<private> o_Target: vec4<f32>;
[[group(0), binding(0)]]
var<uniform> global: CameraViewProj;
[[group(0), binding(1)]]
var<uniform> global1: CameraPosition;
[[group(1), binding(0)]]
var<uniform> global2: Lights;
[[group(3), binding(0)]]
var<uniform> global3: StandardMaterial_base_color;
[[group(3), binding(1)]]
var StandardMaterial_base_color_texture: texture_2d<f32>;
[[group(3), binding(2)]]
var StandardMaterial_base_color_texture_sampler: sampler;
[[group(3), binding(3)]]
var<uniform> global4: StandardMaterial_roughness;
[[group(3), binding(4)]]
var<uniform> global5: StandardMaterial_metallic;
[[group(3), binding(5)]]
var StandardMaterial_metallic_roughness_texture: texture_2d<f32>;
[[group(3), binding(6)]]
var StandardMaterial_metallic_roughness_texture_sampler: sampler;
[[group(3), binding(7)]]
var<uniform> global6: StandardMaterial_reflectance;
[[group(3), binding(8)]]
var StandardMaterial_normal_map: texture_2d<f32>;
[[group(3), binding(9)]]
var StandardMaterial_normal_map_sampler: sampler;
[[group(3), binding(10)]]
var StandardMaterial_occlusion_texture: texture_2d<f32>;
[[group(3), binding(11)]]
var StandardMaterial_occlusion_texture_sampler: sampler;
[[group(3), binding(12)]]
var<uniform> global7: StandardMaterial_emissive;
[[group(3), binding(13)]]
var StandardMaterial_emissive_texture: texture_2d<f32>;
[[group(3), binding(14)]]
var StandardMaterial_emissive_texture_sampler: sampler;
var<private> gl_FrontFacing: bool;

fn pow5(x: f32) -> f32 {
    var x1: f32;
    var x2: f32;

    x1 = x;
    let e42: f32 = x1;
    let e43: f32 = x1;
    x2 = (e42 * e43);
    let e46: f32 = x2;
    let e47: f32 = x2;
    let e49: f32 = x1;
    return ((e46 * e47) * e49);
}

fn getDistanceAttenuation(distanceSquare: f32, inverseRangeSquared: f32) -> f32 {
    var distanceSquare1: f32;
    var inverseRangeSquared1: f32;
    var factor: f32;
    var smoothFactor: f32;
    var attenuation: f32;

    distanceSquare1 = distanceSquare;
    inverseRangeSquared1 = inverseRangeSquared;
    let e44: f32 = distanceSquare1;
    let e45: f32 = inverseRangeSquared1;
    factor = (e44 * e45);
    let e49: f32 = factor;
    let e50: f32 = factor;
    let e56: f32 = factor;
    let e57: f32 = factor;
    smoothFactor = clamp((1.0 - (e56 * e57)), 0.0, 1.0);
    let e64: f32 = smoothFactor;
    let e65: f32 = smoothFactor;
    attenuation = (e64 * e65);
    let e68: f32 = attenuation;
    let e73: f32 = distanceSquare1;
    return ((e68 * 1.0) / max(e73, 0.00009999999747378752));
}

fn D_GGX(roughness: f32, NoH: f32, h: vec3<f32>) -> f32 {
    var roughness1: f32;
    var NoH1: f32;
    var oneMinusNoHSquared: f32;
    var a: f32;
    var k: f32;
    var d: f32;

    roughness1 = roughness;
    NoH1 = NoH;
    let e46: f32 = NoH1;
    let e47: f32 = NoH1;
    oneMinusNoHSquared = (1.0 - (e46 * e47));
    let e51: f32 = NoH1;
    let e52: f32 = roughness1;
    a = (e51 * e52);
    let e55: f32 = roughness1;
    let e56: f32 = oneMinusNoHSquared;
    let e57: f32 = a;
    let e58: f32 = a;
    k = (e55 / (e56 + (e57 * e58)));
    let e63: f32 = k;
    let e64: f32 = k;
    d = ((e63 * e64) * (1.0 / 3.1415927410125732));
    let e70: f32 = d;
    return e70;
}

fn V_SmithGGXCorrelated(roughness2: f32, NoV: f32, NoL: f32) -> f32 {
    var roughness3: f32;
    var NoV1: f32;
    var NoL1: f32;
    var a2: f32;
    var lambdaV: f32;
    var lambdaL: f32;
    var v: f32;

    roughness3 = roughness2;
    NoV1 = NoV;
    NoL1 = NoL;
    let e46: f32 = roughness3;
    let e47: f32 = roughness3;
    a2 = (e46 * e47);
    let e50: f32 = NoL1;
    let e51: f32 = NoV1;
    let e52: f32 = a2;
    let e53: f32 = NoV1;
    let e56: f32 = NoV1;
    let e58: f32 = a2;
    let e60: f32 = NoV1;
    let e61: f32 = a2;
    let e62: f32 = NoV1;
    let e65: f32 = NoV1;
    let e67: f32 = a2;
    lambdaV = (e50 * sqrt((((e60 - (e61 * e62)) * e65) + e67)));
    let e72: f32 = NoV1;
    let e73: f32 = NoL1;
    let e74: f32 = a2;
    let e75: f32 = NoL1;
    let e78: f32 = NoL1;
    let e80: f32 = a2;
    let e82: f32 = NoL1;
    let e83: f32 = a2;
    let e84: f32 = NoL1;
    let e87: f32 = NoL1;
    let e89: f32 = a2;
    lambdaL = (e72 * sqrt((((e82 - (e83 * e84)) * e87) + e89)));
    let e95: f32 = lambdaV;
    let e96: f32 = lambdaL;
    v = (0.5 / (e95 + e96));
    let e100: f32 = v;
    return e100;
}

fn F_Schlick(f0: vec3<f32>, f90: f32, VoH: f32) -> vec3<f32> {
    var f90_1: f32;
    var VoH1: f32;

    f90_1 = f90;
    VoH1 = VoH;
    let e45: f32 = f90_1;
    let e49: f32 = VoH1;
    let e52: f32 = VoH1;
    let e54: f32 = pow5((1.0 - e52));
    return (f0 + ((vec3<f32>(e45) - f0) * e54));
}

fn F_Schlick1(f0_1: f32, f90_2: f32, VoH2: f32) -> f32 {
    var f0_2: f32;
    var f90_3: f32;
    var VoH3: f32;

    f0_2 = f0_1;
    f90_3 = f90_2;
    VoH3 = VoH2;
    let e46: f32 = f0_2;
    let e47: f32 = f90_3;
    let e48: f32 = f0_2;
    let e51: f32 = VoH3;
    let e54: f32 = VoH3;
    let e56: f32 = pow5((1.0 - e54));
    return (e46 + ((e47 - e48) * e56));
}

fn fresnel(f0_3: vec3<f32>, LoH: f32) -> vec3<f32> {
    var f0_4: vec3<f32>;
    var LoH1: f32;
    var f90_4: f32;

    f0_4 = f0_3;
    LoH1 = LoH;
    let e49: vec3<f32> = f0_4;
    let e62: vec3<f32> = f0_4;
    f90_4 = clamp(dot(e62, vec3<f32>((50.0 * 0.33000001311302185))), 0.0, 1.0);
    let e75: vec3<f32> = f0_4;
    let e76: f32 = f90_4;
    let e77: f32 = LoH1;
    let e78: vec3<f32> = F_Schlick(e75, e76, e77);
    return e78;
}

fn specular(f0_5: vec3<f32>, roughness4: f32, h1: vec3<f32>, NoV2: f32, NoL2: f32, NoH2: f32, LoH2: f32, specularIntensity: f32) -> vec3<f32> {
    var f0_6: vec3<f32>;
    var roughness5: f32;
    var NoV3: f32;
    var NoL3: f32;
    var NoH3: f32;
    var LoH3: f32;
    var specularIntensity1: f32;
    var D: f32;
    var V: f32;
    var F: vec3<f32>;

    f0_6 = f0_5;
    roughness5 = roughness4;
    NoV3 = NoV2;
    NoL3 = NoL2;
    NoH3 = NoH2;
    LoH3 = LoH2;
    specularIntensity1 = specularIntensity;
    let e57: f32 = roughness5;
    let e58: f32 = NoH3;
    let e59: f32 = D_GGX(e57, e58, h1);
    D = e59;
    let e64: f32 = roughness5;
    let e65: f32 = NoV3;
    let e66: f32 = NoL3;
    let e67: f32 = V_SmithGGXCorrelated(e64, e65, e66);
    V = e67;
    let e71: vec3<f32> = f0_6;
    let e72: f32 = LoH3;
    let e73: vec3<f32> = fresnel(e71, e72);
    F = e73;
    let e75: f32 = specularIntensity1;
    let e76: f32 = D;
    let e78: f32 = V;
    let e80: vec3<f32> = F;
    return (((e75 * e76) * e78) * e80);
}

fn Fd_Burley(roughness6: f32, NoV4: f32, NoL4: f32, LoH4: f32) -> f32 {
    var roughness7: f32;
    var NoV5: f32;
    var NoL5: f32;
    var LoH5: f32;
    var f90_5: f32;
    var lightScatter: f32;
    var viewScatter: f32;

    roughness7 = roughness6;
    NoV5 = NoV4;
    NoL5 = NoL4;
    LoH5 = LoH4;
    let e50: f32 = roughness7;
    let e52: f32 = LoH5;
    let e54: f32 = LoH5;
    f90_5 = (0.5 + (((2.0 * e50) * e52) * e54));
    let e62: f32 = f90_5;
    let e63: f32 = NoL5;
    let e64: f32 = F_Schlick1(1.0, e62, e63);
    lightScatter = e64;
    let e70: f32 = f90_5;
    let e71: f32 = NoV5;
    let e72: f32 = F_Schlick1(1.0, e70, e71);
    viewScatter = e72;
    let e74: f32 = lightScatter;
    let e75: f32 = viewScatter;
    return ((e74 * e75) * (1.0 / 3.1415927410125732));
}

fn EnvBRDFApprox(f0_7: vec3<f32>, perceptual_roughness: f32, NoV6: f32) -> vec3<f32> {
    var f0_8: vec3<f32>;
    var perceptual_roughness1: f32;
    var NoV7: f32;
    var c0: vec4<f32> = vec4<f32>(-1.0, -0.027499999850988388, -0.5720000267028809, 0.02199999988079071);
    var c1: vec4<f32> = vec4<f32>(1.0, 0.042500000447034836, 1.0399999618530273, -0.03999999910593033);
    var r: vec4<f32>;
    var a004: f32;
    var AB: vec2<f32>;

    f0_8 = f0_7;
    perceptual_roughness1 = perceptual_roughness;
    NoV7 = NoV6;
    let e62: f32 = perceptual_roughness1;
    let e64: vec4<f32> = c0;
    let e66: vec4<f32> = c1;
    r = ((vec4<f32>(e62) * e64) + e66);
    let e69: vec4<f32> = r;
    let e71: vec4<f32> = r;
    let e76: f32 = NoV7;
    let e80: f32 = NoV7;
    let e83: vec4<f32> = r;
    let e85: vec4<f32> = r;
    let e90: f32 = NoV7;
    let e94: f32 = NoV7;
    let e98: vec4<f32> = r;
    let e101: vec4<f32> = r;
    a004 = ((min((e83.x * e85.x), exp2((-(9.279999732971191) * e94))) * e98.x) + e101.y);
    let e109: f32 = a004;
    let e112: vec4<f32> = r;
    AB = ((vec2<f32>(-(1.0399999618530273), 1.0399999618530273) * vec2<f32>(e109)) + e112.zw);
    let e116: vec3<f32> = f0_8;
    let e117: vec2<f32> = AB;
    let e121: vec2<f32> = AB;
    return ((e116 * vec3<f32>(e117.x)) + vec3<f32>(e121.y));
}

fn perceptualRoughnessToRoughness(perceptualRoughness: f32) -> f32 {
    var perceptualRoughness1: f32;
    var clampedPerceptualRoughness: f32;

    perceptualRoughness1 = perceptualRoughness;
    let e45: f32 = perceptualRoughness1;
    clampedPerceptualRoughness = clamp(e45, 0.08900000154972076, 1.0);
    let e50: f32 = clampedPerceptualRoughness;
    let e51: f32 = clampedPerceptualRoughness;
    return (e50 * e51);
}

fn reinhard(color: vec3<f32>) -> vec3<f32> {
    var color1: vec3<f32>;

    color1 = color;
    let e42: vec3<f32> = color1;
    let e45: vec3<f32> = color1;
    return (e42 / (vec3<f32>(1.0) + e45));
}

fn reinhard_extended(color2: vec3<f32>, max_white: f32) -> vec3<f32> {
    var color3: vec3<f32>;
    var max_white1: f32;
    var numerator: vec3<f32>;

    color3 = color2;
    max_white1 = max_white;
    let e44: vec3<f32> = color3;
    let e47: vec3<f32> = color3;
    let e48: f32 = max_white1;
    let e49: f32 = max_white1;
    numerator = (e44 * (vec3<f32>(1.0) + (e47 / vec3<f32>((e48 * e49)))));
    let e56: vec3<f32> = numerator;
    let e59: vec3<f32> = color3;
    return (e56 / (vec3<f32>(1.0) + e59));
}

fn luminance(v1: vec3<f32>) -> f32 {
    var v2: vec3<f32>;

    v2 = v1;
    let e47: vec3<f32> = v2;
    return dot(e47, vec3<f32>(0.2125999927520752, 0.7152000069618225, 0.0722000002861023));
}

fn change_luminance(c_in: vec3<f32>, l_out: f32) -> vec3<f32> {
    var c_in1: vec3<f32>;
    var l_out1: f32;
    var l_in: f32;

    c_in1 = c_in;
    l_out1 = l_out;
    let e45: vec3<f32> = c_in1;
    let e46: f32 = luminance(e45);
    l_in = e46;
    let e48: vec3<f32> = c_in1;
    let e49: f32 = l_out1;
    let e50: f32 = l_in;
    return (e48 * (e49 / e50));
}

fn reinhard_luminance(color4: vec3<f32>) -> vec3<f32> {
    var color5: vec3<f32>;
    var l_old: f32;
    var l_new: f32;

    color5 = color4;
    let e43: vec3<f32> = color5;
    let e44: f32 = luminance(e43);
    l_old = e44;
    let e46: f32 = l_old;
    let e48: f32 = l_old;
    l_new = (e46 / (1.0 + e48));
    let e54: vec3<f32> = color5;
    let e55: f32 = l_new;
    let e56: vec3<f32> = change_luminance(e54, e55);
    return e56;
}

fn reinhard_extended_luminance(color6: vec3<f32>, max_white_l: f32) -> vec3<f32> {
    var color7: vec3<f32>;
    var max_white_l1: f32;
    var l_old1: f32;
    var numerator1: f32;
    var l_new1: f32;

    color7 = color6;
    max_white_l1 = max_white_l;
    let e45: vec3<f32> = color7;
    let e46: f32 = luminance(e45);
    l_old1 = e46;
    let e48: f32 = l_old1;
    let e50: f32 = l_old1;
    let e51: f32 = max_white_l1;
    let e52: f32 = max_white_l1;
    numerator1 = (e48 * (1.0 + (e50 / (e51 * e52))));
    let e58: f32 = numerator1;
    let e60: f32 = l_old1;
    l_new1 = (e58 / (1.0 + e60));
    let e66: vec3<f32> = color7;
    let e67: f32 = l_new1;
    let e68: vec3<f32> = change_luminance(e66, e67);
    return e68;
}

fn point_light(light: PointLight, roughness8: f32, NdotV: f32, N: vec3<f32>, V1: vec3<f32>, R: vec3<f32>, F0: vec3<f32>, diffuseColor: vec3<f32>) -> vec3<f32> {
    var light1: PointLight;
    var roughness9: f32;
    var NdotV1: f32;
    var N1: vec3<f32>;
    var V2: vec3<f32>;
    var R1: vec3<f32>;
    var F0_1: vec3<f32>;
    var diffuseColor1: vec3<f32>;
    var light_to_frag: vec3<f32>;
    var distance_square: f32;
    var rangeAttenuation: f32;
    var a1: f32;
    var radius: f32;
    var centerToRay: vec3<f32>;
    var closestPoint: vec3<f32>;
    var LspecLengthInverse: f32;
    var normalizationFactor: f32;
    var specularIntensity2: f32;
    var L: vec3<f32>;
    var H: vec3<f32>;
    var NoL6: f32;
    var NoH4: f32;
    var LoH6: f32;
    var specular1: vec3<f32>;
    var diffuse: vec3<f32>;

    light1 = light;
    roughness9 = roughness8;
    NdotV1 = NdotV;
    N1 = N;
    V2 = V1;
    R1 = R;
    F0_1 = F0;
    diffuseColor1 = diffuseColor;
    let e56: PointLight = light1;
    let e59: vec3<f32> = v_WorldPosition1;
    light_to_frag = (e56.pos.xyz - e59.xyz);
    let e65: vec3<f32> = light_to_frag;
    let e66: vec3<f32> = light_to_frag;
    distance_square = dot(e65, e66);
    let e70: PointLight = light1;
    let e73: f32 = distance_square;
    let e74: PointLight = light1;
    let e77: f32 = getDistanceAttenuation(e73, e74.lightParams.x);
    rangeAttenuation = e77;
    let e79: f32 = roughness9;
    a1 = e79;
    let e81: PointLight = light1;
    radius = e81.lightParams.y;
    let e87: vec3<f32> = light_to_frag;
    let e88: vec3<f32> = R1;
    let e90: vec3<f32> = R1;
    let e92: vec3<f32> = light_to_frag;
    centerToRay = ((dot(e87, e88) * e90) - e92);
    let e95: vec3<f32> = light_to_frag;
    let e96: vec3<f32> = centerToRay;
    let e97: f32 = radius;
    let e100: vec3<f32> = centerToRay;
    let e101: vec3<f32> = centerToRay;
    let e105: vec3<f32> = centerToRay;
    let e106: vec3<f32> = centerToRay;
    let e112: f32 = radius;
    let e115: vec3<f32> = centerToRay;
    let e116: vec3<f32> = centerToRay;
    let e120: vec3<f32> = centerToRay;
    let e121: vec3<f32> = centerToRay;
    closestPoint = (e95 + (e96 * clamp((e112 * inverseSqrt(dot(e120, e121))), 0.0, 1.0)));
    let e133: vec3<f32> = closestPoint;
    let e134: vec3<f32> = closestPoint;
    let e138: vec3<f32> = closestPoint;
    let e139: vec3<f32> = closestPoint;
    LspecLengthInverse = inverseSqrt(dot(e138, e139));
    let e143: f32 = a1;
    let e144: f32 = a1;
    let e145: f32 = radius;
    let e148: f32 = LspecLengthInverse;
    let e153: f32 = a1;
    let e154: f32 = radius;
    let e157: f32 = LspecLengthInverse;
    normalizationFactor = (e143 / clamp((e153 + ((e154 * 0.5) * e157)), 0.0, 1.0));
    let e165: f32 = normalizationFactor;
    let e166: f32 = normalizationFactor;
    specularIntensity2 = (e165 * e166);
    let e169: vec3<f32> = closestPoint;
    let e170: f32 = LspecLengthInverse;
    L = (e169 * e170);
    let e173: vec3<f32> = L;
    let e174: vec3<f32> = V2;
    let e176: vec3<f32> = L;
    let e177: vec3<f32> = V2;
    H = normalize((e176 + e177));
    let e183: vec3<f32> = N1;
    let e184: vec3<f32> = L;
    let e190: vec3<f32> = N1;
    let e191: vec3<f32> = L;
    NoL6 = clamp(dot(e190, e191), 0.0, 1.0);
    let e199: vec3<f32> = N1;
    let e200: vec3<f32> = H;
    let e206: vec3<f32> = N1;
    let e207: vec3<f32> = H;
    NoH4 = clamp(dot(e206, e207), 0.0, 1.0);
    let e215: vec3<f32> = L;
    let e216: vec3<f32> = H;
    let e222: vec3<f32> = L;
    let e223: vec3<f32> = H;
    LoH6 = clamp(dot(e222, e223), 0.0, 1.0);
    let e237: vec3<f32> = F0_1;
    let e238: f32 = roughness9;
    let e239: vec3<f32> = H;
    let e240: f32 = NdotV1;
    let e241: f32 = NoL6;
    let e242: f32 = NoH4;
    let e243: f32 = LoH6;
    let e244: f32 = specularIntensity2;
    let e245: vec3<f32> = specular(e237, e238, e239, e240, e241, e242, e243, e244);
    specular1 = e245;
    let e248: vec3<f32> = light_to_frag;
    L = normalize(e248);
    let e250: vec3<f32> = L;
    let e251: vec3<f32> = V2;
    let e253: vec3<f32> = L;
    let e254: vec3<f32> = V2;
    H = normalize((e253 + e254));
    let e259: vec3<f32> = N1;
    let e260: vec3<f32> = L;
    let e266: vec3<f32> = N1;
    let e267: vec3<f32> = L;
    NoL6 = clamp(dot(e266, e267), 0.0, 1.0);
    let e274: vec3<f32> = N1;
    let e275: vec3<f32> = H;
    let e281: vec3<f32> = N1;
    let e282: vec3<f32> = H;
    NoH4 = clamp(dot(e281, e282), 0.0, 1.0);
    let e289: vec3<f32> = L;
    let e290: vec3<f32> = H;
    let e296: vec3<f32> = L;
    let e297: vec3<f32> = H;
    LoH6 = clamp(dot(e296, e297), 0.0, 1.0);
    let e302: vec3<f32> = diffuseColor1;
    let e307: f32 = roughness9;
    let e308: f32 = NdotV1;
    let e309: f32 = NoL6;
    let e310: f32 = LoH6;
    let e311: f32 = Fd_Burley(e307, e308, e309, e310);
    diffuse = (e302 * e311);
    let e314: vec3<f32> = diffuse;
    let e315: vec3<f32> = specular1;
    let e317: PointLight = light1;
    let e321: f32 = rangeAttenuation;
    let e322: f32 = NoL6;
    return (((e314 + e315) * e317.color.xyz) * (e321 * e322));
}

fn dir_light(light2: DirectionalLight, roughness10: f32, NdotV2: f32, normal: vec3<f32>, view: vec3<f32>, R2: vec3<f32>, F0_2: vec3<f32>, diffuseColor2: vec3<f32>) -> vec3<f32> {
    var light3: DirectionalLight;
    var roughness11: f32;
    var NdotV3: f32;
    var normal1: vec3<f32>;
    var view1: vec3<f32>;
    var R3: vec3<f32>;
    var F0_3: vec3<f32>;
    var diffuseColor3: vec3<f32>;
    var incident_light: vec3<f32>;
    var half_vector: vec3<f32>;
    var NoL7: f32;
    var NoH5: f32;
    var LoH7: f32;
    var diffuse1: vec3<f32>;
    var specularIntensity3: f32 = 1.0;
    var specular2: vec3<f32>;

    light3 = light2;
    roughness11 = roughness10;
    NdotV3 = NdotV2;
    normal1 = normal;
    view1 = view;
    R3 = R2;
    F0_3 = F0_2;
    diffuseColor3 = diffuseColor2;
    let e56: DirectionalLight = light3;
    incident_light = e56.direction.xyz;
    let e60: vec3<f32> = incident_light;
    let e61: vec3<f32> = view1;
    let e63: vec3<f32> = incident_light;
    let e64: vec3<f32> = view1;
    half_vector = normalize((e63 + e64));
    let e70: vec3<f32> = normal1;
    let e71: vec3<f32> = incident_light;
    let e77: vec3<f32> = normal1;
    let e78: vec3<f32> = incident_light;
    NoL7 = clamp(dot(e77, e78), 0.0, 1.0);
    let e86: vec3<f32> = normal1;
    let e87: vec3<f32> = half_vector;
    let e93: vec3<f32> = normal1;
    let e94: vec3<f32> = half_vector;
    NoH5 = clamp(dot(e93, e94), 0.0, 1.0);
    let e102: vec3<f32> = incident_light;
    let e103: vec3<f32> = half_vector;
    let e109: vec3<f32> = incident_light;
    let e110: vec3<f32> = half_vector;
    LoH7 = clamp(dot(e109, e110), 0.0, 1.0);
    let e116: vec3<f32> = diffuseColor3;
    let e121: f32 = roughness11;
    let e122: f32 = NdotV3;
    let e123: f32 = NoL7;
    let e124: f32 = LoH7;
    let e125: f32 = Fd_Burley(e121, e122, e123, e124);
    diffuse1 = (e116 * e125);
    let e138: vec3<f32> = F0_3;
    let e139: f32 = roughness11;
    let e140: vec3<f32> = half_vector;
    let e141: f32 = NdotV3;
    let e142: f32 = NoL7;
    let e143: f32 = NoH5;
    let e144: f32 = LoH7;
    let e145: f32 = specularIntensity3;
    let e146: vec3<f32> = specular(e138, e139, e140, e141, e142, e143, e144, e145);
    specular2 = e146;
    let e148: vec3<f32> = specular2;
    let e149: vec3<f32> = diffuse1;
    let e151: DirectionalLight = light3;
    let e155: f32 = NoL7;
    return (((e148 + e149) * e151.color.xyz) * e155);
}

fn main1() {
    var output_color: vec4<f32>;
    var metallic_roughness: vec4<f32>;
    var metallic: f32;
    var perceptual_roughness2: f32;
    var roughness12: f32;
    var N2: vec3<f32>;
    var T: vec3<f32>;
    var B: vec3<f32>;
    var TBN: mat3x3<f32>;
    var occlusion: f32;
    var emissive: vec4<f32>;
    var V3: vec3<f32>;
    var NdotV4: f32;
    var F0_4: vec3<f32>;
    var diffuseColor4: vec3<f32>;
    var R4: vec3<f32>;
    var light_accum: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var i: i32 = 0;
    var i1: i32 = 0;
    var diffuse_ambient: vec3<f32>;
    var specular_ambient: vec3<f32>;

    let e40: vec4<f32> = global3.base_color;
    output_color = e40;
    let e42: vec4<f32> = output_color;
    let e44: vec2<f32> = v_Uv1;
    let e45: vec4<f32> = textureSample(StandardMaterial_base_color_texture, StandardMaterial_base_color_texture_sampler, e44);
    output_color = (e42 * e45);
    let e48: vec2<f32> = v_Uv1;
    let e49: vec4<f32> = textureSample(StandardMaterial_metallic_roughness_texture, StandardMaterial_metallic_roughness_texture_sampler, e48);
    metallic_roughness = e49;
    let e51: f32 = global5.metallic;
    let e52: vec4<f32> = metallic_roughness;
    metallic = (e51 * e52.z);
    let e56: f32 = global4.perceptual_roughness;
    let e57: vec4<f32> = metallic_roughness;
    perceptual_roughness2 = (e56 * e57.y);
    let e62: f32 = perceptual_roughness2;
    let e63: f32 = perceptualRoughnessToRoughness(e62);
    roughness12 = e63;
    let e66: vec3<f32> = v_WorldNormal1;
    N2 = normalize(e66);
    let e69: vec4<f32> = v_WorldTangent1;
    let e71: vec4<f32> = v_WorldTangent1;
    T = normalize(e71.xyz);
    let e77: vec3<f32> = N2;
    let e78: vec3<f32> = T;
    let e80: vec4<f32> = v_WorldTangent1;
    B = (cross(e77, e78) * e80.w);
    let e85: bool = gl_FrontFacing;
    let e86: vec3<f32> = N2;
    let e87: vec3<f32> = N2;
    N2 = select(-(e87), e86, e85);
    let e90: bool = gl_FrontFacing;
    let e91: vec3<f32> = T;
    let e92: vec3<f32> = T;
    T = select(-(e92), e91, e90);
    let e95: bool = gl_FrontFacing;
    let e96: vec3<f32> = B;
    let e97: vec3<f32> = B;
    B = select(-(e97), e96, e95);
    let e100: vec3<f32> = T;
    let e101: vec3<f32> = B;
    let e102: vec3<f32> = N2;
    TBN = mat3x3<f32>(vec3<f32>(e100.x, e100.y, e100.z), vec3<f32>(e101.x, e101.y, e101.z), vec3<f32>(e102.x, e102.y, e102.z));
    let e117: mat3x3<f32> = TBN;
    let e119: vec2<f32> = v_Uv1;
    let e120: vec4<f32> = textureSample(StandardMaterial_normal_map, StandardMaterial_normal_map_sampler, e119);
    let e128: vec2<f32> = v_Uv1;
    let e129: vec4<f32> = textureSample(StandardMaterial_normal_map, StandardMaterial_normal_map_sampler, e128);
    N2 = (e117 * normalize(((e129.xyz * 2.0) - vec3<f32>(1.0))));
    let e139: vec2<f32> = v_Uv1;
    let e140: vec4<f32> = textureSample(StandardMaterial_occlusion_texture, StandardMaterial_occlusion_texture_sampler, e139);
    occlusion = e140.x;
    let e143: vec4<f32> = global7.emissive;
    emissive = e143;
    let e145: vec4<f32> = emissive;
    let e147: vec4<f32> = emissive;
    let e150: vec2<f32> = v_Uv1;
    let e151: vec4<f32> = textureSample(StandardMaterial_emissive_texture, StandardMaterial_emissive_texture_sampler, e150);
    let e153: vec3<f32> = (e147.xyz * e151.xyz);
    emissive.x = e153.x;
    emissive.y = e153.y;
    emissive.z = e153.z;
    let e160: vec4<f32> = global1.CameraPos;
    let e162: vec3<f32> = v_WorldPosition1;
    let e165: vec4<f32> = global1.CameraPos;
    let e167: vec3<f32> = v_WorldPosition1;
    V3 = normalize((e165.xyz - e167.xyz));
    let e174: vec3<f32> = N2;
    let e175: vec3<f32> = V3;
    let e180: vec3<f32> = N2;
    let e181: vec3<f32> = V3;
    NdotV4 = max(dot(e180, e181), 0.00009999999747378752);
    let e187: f32 = global6.reflectance;
    let e189: f32 = global6.reflectance;
    let e192: f32 = metallic;
    let e196: vec4<f32> = output_color;
    let e198: f32 = metallic;
    F0_4 = (vec3<f32>((((0.1599999964237213 * e187) * e189) * (1.0 - e192))) + (e196.xyz * vec3<f32>(e198)));
    let e203: vec4<f32> = output_color;
    let e206: f32 = metallic;
    diffuseColor4 = (e203.xyz * vec3<f32>((1.0 - e206)));
    let e211: vec3<f32> = V3;
    let e214: vec3<f32> = V3;
    let e216: vec3<f32> = N2;
    R4 = reflect(-(e214), e216);
    loop {
        let e224: i32 = i;
        let e225: vec4<u32> = global2.NumLights;
        let e229: i32 = i;
        if (!(((e224 < i32(e225.x)) && (e229 < 10)))) {
            break;
        }
        {
            let e236: vec3<f32> = light_accum;
            let e237: i32 = i;
            let e247: i32 = i;
            let e249: PointLight = global2.PointLights[e247];
            let e250: f32 = roughness12;
            let e251: f32 = NdotV4;
            let e252: vec3<f32> = N2;
            let e253: vec3<f32> = V3;
            let e254: vec3<f32> = R4;
            let e255: vec3<f32> = F0_4;
            let e256: vec3<f32> = diffuseColor4;
            let e257: vec3<f32> = point_light(e249, e250, e251, e252, e253, e254, e255, e256);
            light_accum = (e236 + e257);
        }
        continuing {
            let e233: i32 = i;
            i = (e233 + 1);
        }
    }
    loop {
        let e261: i32 = i1;
        let e262: vec4<u32> = global2.NumLights;
        let e266: i32 = i1;
        if (!(((e261 < i32(e262.y)) && (e266 < 1)))) {
            break;
        }
        {
            let e273: vec3<f32> = light_accum;
            let e274: i32 = i1;
            let e284: i32 = i1;
            let e286: DirectionalLight = global2.DirectionalLights[e284];
            let e287: f32 = roughness12;
            let e288: f32 = NdotV4;
            let e289: vec3<f32> = N2;
            let e290: vec3<f32> = V3;
            let e291: vec3<f32> = R4;
            let e292: vec3<f32> = F0_4;
            let e293: vec3<f32> = diffuseColor4;
            let e294: vec3<f32> = dir_light(e286, e287, e288, e289, e290, e291, e292, e293);
            light_accum = (e273 + e294);
        }
        continuing {
            let e270: i32 = i1;
            i1 = (e270 + 1);
        }
    }
    let e299: vec3<f32> = diffuseColor4;
    let e301: f32 = NdotV4;
    let e302: vec3<f32> = EnvBRDFApprox(e299, 1.0, e301);
    diffuse_ambient = e302;
    let e307: vec3<f32> = F0_4;
    let e308: f32 = perceptual_roughness2;
    let e309: f32 = NdotV4;
    let e310: vec3<f32> = EnvBRDFApprox(e307, e308, e309);
    specular_ambient = e310;
    let e312: vec4<f32> = output_color;
    let e314: vec3<f32> = light_accum;
    output_color.x = e314.x;
    output_color.y = e314.y;
    output_color.z = e314.z;
    let e321: vec4<f32> = output_color;
    let e323: vec4<f32> = output_color;
    let e325: vec3<f32> = diffuse_ambient;
    let e326: vec3<f32> = specular_ambient;
    let e328: vec4<f32> = global2.AmbientColor;
    let e331: f32 = occlusion;
    let e333: vec3<f32> = (e323.xyz + (((e325 + e326) * e328.xyz) * e331));
    output_color.x = e333.x;
    output_color.y = e333.y;
    output_color.z = e333.z;
    let e340: vec4<f32> = output_color;
    let e342: vec4<f32> = output_color;
    let e344: vec4<f32> = emissive;
    let e346: vec4<f32> = output_color;
    let e349: vec3<f32> = (e342.xyz + (e344.xyz * e346.w));
    output_color.x = e349.x;
    output_color.y = e349.y;
    output_color.z = e349.z;
    let e356: vec4<f32> = output_color;
    let e358: vec4<f32> = output_color;
    let e360: vec4<f32> = output_color;
    let e362: vec3<f32> = reinhard_luminance(e360.xyz);
    output_color.x = e362.x;
    output_color.y = e362.y;
    output_color.z = e362.z;
    let e369: vec4<f32> = output_color;
    o_Target = e369;
    return;
}

[[stage(fragment)]]
fn main([[location(0)]] v_WorldPosition: vec3<f32>, [[location(1)]] v_WorldNormal: vec3<f32>, [[location(2)]] v_Uv: vec2<f32>, [[location(3)]] v_WorldTangent: vec4<f32>, [[builtin(front_facing)]] param: bool) -> FragmentOutput {
    v_WorldPosition1 = v_WorldPosition;
    v_WorldNormal1 = v_WorldNormal;
    v_Uv1 = v_Uv;
    v_WorldTangent1 = v_WorldTangent;
    gl_FrontFacing = param;
    main1();
    let e72: vec4<f32> = o_Target;
    return FragmentOutput(e72);
}
