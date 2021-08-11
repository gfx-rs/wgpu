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
    PointLights: [[stride(48)]] array<PointLight,10>;
    DirectionalLights: [[stride(32)]] array<DirectionalLight,1>;
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

fn pow5_(x: f32) -> f32 {
    var x1: f32;
    var x2_: f32;

    x1 = x;
    let _e42: f32 = x1;
    let _e43: f32 = x1;
    x2_ = (_e42 * _e43);
    let _e46: f32 = x2_;
    let _e47: f32 = x2_;
    let _e49: f32 = x1;
    return ((_e46 * _e47) * _e49);
}

fn getDistanceAttenuation(distanceSquare: f32, inverseRangeSquared: f32) -> f32 {
    var distanceSquare1: f32;
    var inverseRangeSquared1: f32;
    var factor: f32;
    var smoothFactor: f32;
    var attenuation: f32;

    distanceSquare1 = distanceSquare;
    inverseRangeSquared1 = inverseRangeSquared;
    let _e44: f32 = distanceSquare1;
    let _e45: f32 = inverseRangeSquared1;
    factor = (_e44 * _e45);
    let _e49: f32 = factor;
    let _e50: f32 = factor;
    let _e56: f32 = factor;
    let _e57: f32 = factor;
    smoothFactor = clamp((1.0 - (_e56 * _e57)), 0.0, 1.0);
    let _e64: f32 = smoothFactor;
    let _e65: f32 = smoothFactor;
    attenuation = (_e64 * _e65);
    let _e68: f32 = attenuation;
    let _e73: f32 = distanceSquare1;
    return ((_e68 * 1.0) / max(_e73, 0.00009999999747378752));
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
    let _e46: f32 = NoH1;
    let _e47: f32 = NoH1;
    oneMinusNoHSquared = (1.0 - (_e46 * _e47));
    let _e51: f32 = NoH1;
    let _e52: f32 = roughness1;
    a = (_e51 * _e52);
    let _e55: f32 = roughness1;
    let _e56: f32 = oneMinusNoHSquared;
    let _e57: f32 = a;
    let _e58: f32 = a;
    k = (_e55 / (_e56 + (_e57 * _e58)));
    let _e63: f32 = k;
    let _e64: f32 = k;
    d = ((_e63 * _e64) * (1.0 / 3.1415927410125732));
    let _e70: f32 = d;
    return _e70;
}

fn V_SmithGGXCorrelated(roughness2: f32, NoV: f32, NoL: f32) -> f32 {
    var roughness3: f32;
    var NoV1: f32;
    var NoL1: f32;
    var a2_: f32;
    var lambdaV: f32;
    var lambdaL: f32;
    var v: f32;

    roughness3 = roughness2;
    NoV1 = NoV;
    NoL1 = NoL;
    let _e46: f32 = roughness3;
    let _e47: f32 = roughness3;
    a2_ = (_e46 * _e47);
    let _e50: f32 = NoL1;
    let _e51: f32 = NoV1;
    let _e52: f32 = a2_;
    let _e53: f32 = NoV1;
    let _e56: f32 = NoV1;
    let _e58: f32 = a2_;
    let _e60: f32 = NoV1;
    let _e61: f32 = a2_;
    let _e62: f32 = NoV1;
    let _e65: f32 = NoV1;
    let _e67: f32 = a2_;
    lambdaV = (_e50 * sqrt((((_e60 - (_e61 * _e62)) * _e65) + _e67)));
    let _e72: f32 = NoV1;
    let _e73: f32 = NoL1;
    let _e74: f32 = a2_;
    let _e75: f32 = NoL1;
    let _e78: f32 = NoL1;
    let _e80: f32 = a2_;
    let _e82: f32 = NoL1;
    let _e83: f32 = a2_;
    let _e84: f32 = NoL1;
    let _e87: f32 = NoL1;
    let _e89: f32 = a2_;
    lambdaL = (_e72 * sqrt((((_e82 - (_e83 * _e84)) * _e87) + _e89)));
    let _e95: f32 = lambdaV;
    let _e96: f32 = lambdaL;
    v = (0.5 / (_e95 + _e96));
    let _e100: f32 = v;
    return _e100;
}

fn F_Schlick(f0_: vec3<f32>, f90_: f32, VoH: f32) -> vec3<f32> {
    var f90_1: f32;
    var VoH1: f32;

    f90_1 = f90_;
    VoH1 = VoH;
    let _e45: f32 = f90_1;
    let _e49: f32 = VoH1;
    let _e52: f32 = VoH1;
    let _e54: f32 = pow5_((1.0 - _e52));
    return (f0_ + ((vec3<f32>(_e45) - f0_) * _e54));
}

fn F_Schlick1(f0_1: f32, f90_2: f32, VoH2: f32) -> f32 {
    var f0_2: f32;
    var f90_3: f32;
    var VoH3: f32;

    f0_2 = f0_1;
    f90_3 = f90_2;
    VoH3 = VoH2;
    let _e46: f32 = f0_2;
    let _e47: f32 = f90_3;
    let _e48: f32 = f0_2;
    let _e51: f32 = VoH3;
    let _e54: f32 = VoH3;
    let _e56: f32 = pow5_((1.0 - _e54));
    return (_e46 + ((_e47 - _e48) * _e56));
}

fn fresnel(f0_3: vec3<f32>, LoH: f32) -> vec3<f32> {
    var f0_4: vec3<f32>;
    var LoH1: f32;
    var f90_4: f32;

    f0_4 = f0_3;
    LoH1 = LoH;
    let _e49: vec3<f32> = f0_4;
    let _e62: vec3<f32> = f0_4;
    f90_4 = clamp(dot(_e62, vec3<f32>((50.0 * 0.33000001311302185))), 0.0, 1.0);
    let _e75: vec3<f32> = f0_4;
    let _e76: f32 = f90_4;
    let _e77: f32 = LoH1;
    let _e78: vec3<f32> = F_Schlick(_e75, _e76, _e77);
    return _e78;
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
    let _e57: f32 = roughness5;
    let _e58: f32 = NoH3;
    let _e59: f32 = D_GGX(_e57, _e58, h1);
    D = _e59;
    let _e64: f32 = roughness5;
    let _e65: f32 = NoV3;
    let _e66: f32 = NoL3;
    let _e67: f32 = V_SmithGGXCorrelated(_e64, _e65, _e66);
    V = _e67;
    let _e71: vec3<f32> = f0_6;
    let _e72: f32 = LoH3;
    let _e73: vec3<f32> = fresnel(_e71, _e72);
    F = _e73;
    let _e75: f32 = specularIntensity1;
    let _e76: f32 = D;
    let _e78: f32 = V;
    let _e80: vec3<f32> = F;
    return (((_e75 * _e76) * _e78) * _e80);
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
    let _e50: f32 = roughness7;
    let _e52: f32 = LoH5;
    let _e54: f32 = LoH5;
    f90_5 = (0.5 + (((2.0 * _e50) * _e52) * _e54));
    let _e62: f32 = f90_5;
    let _e63: f32 = NoL5;
    let _e64: f32 = F_Schlick1(1.0, _e62, _e63);
    lightScatter = _e64;
    let _e70: f32 = f90_5;
    let _e71: f32 = NoV5;
    let _e72: f32 = F_Schlick1(1.0, _e70, _e71);
    viewScatter = _e72;
    let _e74: f32 = lightScatter;
    let _e75: f32 = viewScatter;
    return ((_e74 * _e75) * (1.0 / 3.1415927410125732));
}

fn EnvBRDFApprox(f0_7: vec3<f32>, perceptual_roughness: f32, NoV6: f32) -> vec3<f32> {
    var f0_8: vec3<f32>;
    var perceptual_roughness1: f32;
    var NoV7: f32;
    var c0_: vec4<f32> = vec4<f32>(-1.0, -0.027499999850988388, -0.5720000267028809, 0.02199999988079071);
    var c1_: vec4<f32> = vec4<f32>(1.0, 0.042500000447034836, 1.0399999618530273, -0.03999999910593033);
    var r: vec4<f32>;
    var a004_: f32;
    var AB: vec2<f32>;

    f0_8 = f0_7;
    perceptual_roughness1 = perceptual_roughness;
    NoV7 = NoV6;
    let _e62: f32 = perceptual_roughness1;
    let _e64: vec4<f32> = c0_;
    let _e66: vec4<f32> = c1_;
    r = ((vec4<f32>(_e62) * _e64) + _e66);
    let _e69: vec4<f32> = r;
    let _e71: vec4<f32> = r;
    let _e76: f32 = NoV7;
    let _e80: f32 = NoV7;
    let _e83: vec4<f32> = r;
    let _e85: vec4<f32> = r;
    let _e90: f32 = NoV7;
    let _e94: f32 = NoV7;
    let _e98: vec4<f32> = r;
    let _e101: vec4<f32> = r;
    a004_ = ((min((_e83.x * _e85.x), exp2((-(9.279999732971191) * _e94))) * _e98.x) + _e101.y);
    let _e109: f32 = a004_;
    let _e112: vec4<f32> = r;
    AB = ((vec2<f32>(-(1.0399999618530273), 1.0399999618530273) * vec2<f32>(_e109)) + _e112.zw);
    let _e116: vec3<f32> = f0_8;
    let _e117: vec2<f32> = AB;
    let _e121: vec2<f32> = AB;
    return ((_e116 * vec3<f32>(_e117.x)) + vec3<f32>(_e121.y));
}

fn perceptualRoughnessToRoughness(perceptualRoughness: f32) -> f32 {
    var perceptualRoughness1: f32;
    var clampedPerceptualRoughness: f32;

    perceptualRoughness1 = perceptualRoughness;
    let _e45: f32 = perceptualRoughness1;
    clampedPerceptualRoughness = clamp(_e45, 0.08900000154972076, 1.0);
    let _e50: f32 = clampedPerceptualRoughness;
    let _e51: f32 = clampedPerceptualRoughness;
    return (_e50 * _e51);
}

fn reinhard(color: vec3<f32>) -> vec3<f32> {
    var color1: vec3<f32>;

    color1 = color;
    let _e42: vec3<f32> = color1;
    let _e45: vec3<f32> = color1;
    return (_e42 / (vec3<f32>(1.0) + _e45));
}

fn reinhard_extended(color2: vec3<f32>, max_white: f32) -> vec3<f32> {
    var color3: vec3<f32>;
    var max_white1: f32;
    var numerator: vec3<f32>;

    color3 = color2;
    max_white1 = max_white;
    let _e44: vec3<f32> = color3;
    let _e47: vec3<f32> = color3;
    let _e48: f32 = max_white1;
    let _e49: f32 = max_white1;
    numerator = (_e44 * (vec3<f32>(1.0) + (_e47 / vec3<f32>((_e48 * _e49)))));
    let _e56: vec3<f32> = numerator;
    let _e59: vec3<f32> = color3;
    return (_e56 / (vec3<f32>(1.0) + _e59));
}

fn luminance(v1: vec3<f32>) -> f32 {
    var v2: vec3<f32>;

    v2 = v1;
    let _e47: vec3<f32> = v2;
    return dot(_e47, vec3<f32>(0.2125999927520752, 0.7152000069618225, 0.0722000002861023));
}

fn change_luminance(c_in: vec3<f32>, l_out: f32) -> vec3<f32> {
    var c_in1: vec3<f32>;
    var l_out1: f32;
    var l_in: f32;

    c_in1 = c_in;
    l_out1 = l_out;
    let _e45: vec3<f32> = c_in1;
    let _e46: f32 = luminance(_e45);
    l_in = _e46;
    let _e48: vec3<f32> = c_in1;
    let _e49: f32 = l_out1;
    let _e50: f32 = l_in;
    return (_e48 * (_e49 / _e50));
}

fn reinhard_luminance(color4: vec3<f32>) -> vec3<f32> {
    var color5: vec3<f32>;
    var l_old: f32;
    var l_new: f32;

    color5 = color4;
    let _e43: vec3<f32> = color5;
    let _e44: f32 = luminance(_e43);
    l_old = _e44;
    let _e46: f32 = l_old;
    let _e48: f32 = l_old;
    l_new = (_e46 / (1.0 + _e48));
    let _e54: vec3<f32> = color5;
    let _e55: f32 = l_new;
    let _e56: vec3<f32> = change_luminance(_e54, _e55);
    return _e56;
}

fn reinhard_extended_luminance(color6: vec3<f32>, max_white_l: f32) -> vec3<f32> {
    var color7: vec3<f32>;
    var max_white_l1: f32;
    var l_old1: f32;
    var numerator1: f32;
    var l_new1: f32;

    color7 = color6;
    max_white_l1 = max_white_l;
    let _e45: vec3<f32> = color7;
    let _e46: f32 = luminance(_e45);
    l_old1 = _e46;
    let _e48: f32 = l_old1;
    let _e50: f32 = l_old1;
    let _e51: f32 = max_white_l1;
    let _e52: f32 = max_white_l1;
    numerator1 = (_e48 * (1.0 + (_e50 / (_e51 * _e52))));
    let _e58: f32 = numerator1;
    let _e60: f32 = l_old1;
    l_new1 = (_e58 / (1.0 + _e60));
    let _e66: vec3<f32> = color7;
    let _e67: f32 = l_new1;
    let _e68: vec3<f32> = change_luminance(_e66, _e67);
    return _e68;
}

fn point_light(light: PointLight, roughness8: f32, NdotV: f32, N: vec3<f32>, V1: vec3<f32>, R: vec3<f32>, F0_: vec3<f32>, diffuseColor: vec3<f32>) -> vec3<f32> {
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
    F0_1 = F0_;
    diffuseColor1 = diffuseColor;
    let _e56: PointLight = light1;
    let _e59: vec3<f32> = v_WorldPosition1;
    light_to_frag = (_e56.pos.xyz - _e59.xyz);
    let _e65: vec3<f32> = light_to_frag;
    let _e66: vec3<f32> = light_to_frag;
    distance_square = dot(_e65, _e66);
    let _e70: PointLight = light1;
    let _e73: f32 = distance_square;
    let _e74: PointLight = light1;
    let _e77: f32 = getDistanceAttenuation(_e73, _e74.lightParams.x);
    rangeAttenuation = _e77;
    let _e79: f32 = roughness9;
    a1 = _e79;
    let _e81: PointLight = light1;
    radius = _e81.lightParams.y;
    let _e87: vec3<f32> = light_to_frag;
    let _e88: vec3<f32> = R1;
    let _e90: vec3<f32> = R1;
    let _e92: vec3<f32> = light_to_frag;
    centerToRay = ((dot(_e87, _e88) * _e90) - _e92);
    let _e95: vec3<f32> = light_to_frag;
    let _e96: vec3<f32> = centerToRay;
    let _e97: f32 = radius;
    let _e100: vec3<f32> = centerToRay;
    let _e101: vec3<f32> = centerToRay;
    let _e105: vec3<f32> = centerToRay;
    let _e106: vec3<f32> = centerToRay;
    let _e112: f32 = radius;
    let _e115: vec3<f32> = centerToRay;
    let _e116: vec3<f32> = centerToRay;
    let _e120: vec3<f32> = centerToRay;
    let _e121: vec3<f32> = centerToRay;
    closestPoint = (_e95 + (_e96 * clamp((_e112 * inverseSqrt(dot(_e120, _e121))), 0.0, 1.0)));
    let _e133: vec3<f32> = closestPoint;
    let _e134: vec3<f32> = closestPoint;
    let _e138: vec3<f32> = closestPoint;
    let _e139: vec3<f32> = closestPoint;
    LspecLengthInverse = inverseSqrt(dot(_e138, _e139));
    let _e143: f32 = a1;
    let _e144: f32 = a1;
    let _e145: f32 = radius;
    let _e148: f32 = LspecLengthInverse;
    let _e153: f32 = a1;
    let _e154: f32 = radius;
    let _e157: f32 = LspecLengthInverse;
    normalizationFactor = (_e143 / clamp((_e153 + ((_e154 * 0.5) * _e157)), 0.0, 1.0));
    let _e165: f32 = normalizationFactor;
    let _e166: f32 = normalizationFactor;
    specularIntensity2 = (_e165 * _e166);
    let _e169: vec3<f32> = closestPoint;
    let _e170: f32 = LspecLengthInverse;
    L = (_e169 * _e170);
    let _e173: vec3<f32> = L;
    let _e174: vec3<f32> = V2;
    let _e176: vec3<f32> = L;
    let _e177: vec3<f32> = V2;
    H = normalize((_e176 + _e177));
    let _e183: vec3<f32> = N1;
    let _e184: vec3<f32> = L;
    let _e190: vec3<f32> = N1;
    let _e191: vec3<f32> = L;
    NoL6 = clamp(dot(_e190, _e191), 0.0, 1.0);
    let _e199: vec3<f32> = N1;
    let _e200: vec3<f32> = H;
    let _e206: vec3<f32> = N1;
    let _e207: vec3<f32> = H;
    NoH4 = clamp(dot(_e206, _e207), 0.0, 1.0);
    let _e215: vec3<f32> = L;
    let _e216: vec3<f32> = H;
    let _e222: vec3<f32> = L;
    let _e223: vec3<f32> = H;
    LoH6 = clamp(dot(_e222, _e223), 0.0, 1.0);
    let _e237: vec3<f32> = F0_1;
    let _e238: f32 = roughness9;
    let _e239: vec3<f32> = H;
    let _e240: f32 = NdotV1;
    let _e241: f32 = NoL6;
    let _e242: f32 = NoH4;
    let _e243: f32 = LoH6;
    let _e244: f32 = specularIntensity2;
    let _e245: vec3<f32> = specular(_e237, _e238, _e239, _e240, _e241, _e242, _e243, _e244);
    specular1 = _e245;
    let _e248: vec3<f32> = light_to_frag;
    L = normalize(_e248);
    let _e250: vec3<f32> = L;
    let _e251: vec3<f32> = V2;
    let _e253: vec3<f32> = L;
    let _e254: vec3<f32> = V2;
    H = normalize((_e253 + _e254));
    let _e259: vec3<f32> = N1;
    let _e260: vec3<f32> = L;
    let _e266: vec3<f32> = N1;
    let _e267: vec3<f32> = L;
    NoL6 = clamp(dot(_e266, _e267), 0.0, 1.0);
    let _e274: vec3<f32> = N1;
    let _e275: vec3<f32> = H;
    let _e281: vec3<f32> = N1;
    let _e282: vec3<f32> = H;
    NoH4 = clamp(dot(_e281, _e282), 0.0, 1.0);
    let _e289: vec3<f32> = L;
    let _e290: vec3<f32> = H;
    let _e296: vec3<f32> = L;
    let _e297: vec3<f32> = H;
    LoH6 = clamp(dot(_e296, _e297), 0.0, 1.0);
    let _e302: vec3<f32> = diffuseColor1;
    let _e307: f32 = roughness9;
    let _e308: f32 = NdotV1;
    let _e309: f32 = NoL6;
    let _e310: f32 = LoH6;
    let _e311: f32 = Fd_Burley(_e307, _e308, _e309, _e310);
    diffuse = (_e302 * _e311);
    let _e314: vec3<f32> = diffuse;
    let _e315: vec3<f32> = specular1;
    let _e317: PointLight = light1;
    let _e321: f32 = rangeAttenuation;
    let _e322: f32 = NoL6;
    return (((_e314 + _e315) * _e317.color.xyz) * (_e321 * _e322));
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
    let _e56: DirectionalLight = light3;
    incident_light = _e56.direction.xyz;
    let _e60: vec3<f32> = incident_light;
    let _e61: vec3<f32> = view1;
    let _e63: vec3<f32> = incident_light;
    let _e64: vec3<f32> = view1;
    half_vector = normalize((_e63 + _e64));
    let _e70: vec3<f32> = normal1;
    let _e71: vec3<f32> = incident_light;
    let _e77: vec3<f32> = normal1;
    let _e78: vec3<f32> = incident_light;
    NoL7 = clamp(dot(_e77, _e78), 0.0, 1.0);
    let _e86: vec3<f32> = normal1;
    let _e87: vec3<f32> = half_vector;
    let _e93: vec3<f32> = normal1;
    let _e94: vec3<f32> = half_vector;
    NoH5 = clamp(dot(_e93, _e94), 0.0, 1.0);
    let _e102: vec3<f32> = incident_light;
    let _e103: vec3<f32> = half_vector;
    let _e109: vec3<f32> = incident_light;
    let _e110: vec3<f32> = half_vector;
    LoH7 = clamp(dot(_e109, _e110), 0.0, 1.0);
    let _e116: vec3<f32> = diffuseColor3;
    let _e121: f32 = roughness11;
    let _e122: f32 = NdotV3;
    let _e123: f32 = NoL7;
    let _e124: f32 = LoH7;
    let _e125: f32 = Fd_Burley(_e121, _e122, _e123, _e124);
    diffuse1 = (_e116 * _e125);
    let _e138: vec3<f32> = F0_3;
    let _e139: f32 = roughness11;
    let _e140: vec3<f32> = half_vector;
    let _e141: f32 = NdotV3;
    let _e142: f32 = NoL7;
    let _e143: f32 = NoH5;
    let _e144: f32 = LoH7;
    let _e145: f32 = specularIntensity3;
    let _e146: vec3<f32> = specular(_e138, _e139, _e140, _e141, _e142, _e143, _e144, _e145);
    specular2 = _e146;
    let _e148: vec3<f32> = specular2;
    let _e149: vec3<f32> = diffuse1;
    let _e151: DirectionalLight = light3;
    let _e155: f32 = NoL7;
    return (((_e148 + _e149) * _e151.color.xyz) * _e155);
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

    let _e40: vec4<f32> = global3.base_color;
    output_color = _e40;
    let _e42: vec4<f32> = output_color;
    let _e44: vec2<f32> = v_Uv1;
    let _e45: vec4<f32> = textureSample(StandardMaterial_base_color_texture, StandardMaterial_base_color_texture_sampler, _e44);
    output_color = (_e42 * _e45);
    let _e48: vec2<f32> = v_Uv1;
    let _e49: vec4<f32> = textureSample(StandardMaterial_metallic_roughness_texture, StandardMaterial_metallic_roughness_texture_sampler, _e48);
    metallic_roughness = _e49;
    let _e51: f32 = global5.metallic;
    let _e52: vec4<f32> = metallic_roughness;
    metallic = (_e51 * _e52.z);
    let _e56: f32 = global4.perceptual_roughness;
    let _e57: vec4<f32> = metallic_roughness;
    perceptual_roughness2 = (_e56 * _e57.y);
    let _e62: f32 = perceptual_roughness2;
    let _e63: f32 = perceptualRoughnessToRoughness(_e62);
    roughness12 = _e63;
    let _e66: vec3<f32> = v_WorldNormal1;
    N2 = normalize(_e66);
    let _e69: vec4<f32> = v_WorldTangent1;
    let _e71: vec4<f32> = v_WorldTangent1;
    T = normalize(_e71.xyz);
    let _e77: vec3<f32> = N2;
    let _e78: vec3<f32> = T;
    let _e80: vec4<f32> = v_WorldTangent1;
    B = (cross(_e77, _e78) * _e80.w);
    let _e85: bool = gl_FrontFacing;
    let _e86: vec3<f32> = N2;
    let _e87: vec3<f32> = N2;
    N2 = select(-(_e87), _e86, _e85);
    let _e90: bool = gl_FrontFacing;
    let _e91: vec3<f32> = T;
    let _e92: vec3<f32> = T;
    T = select(-(_e92), _e91, _e90);
    let _e95: bool = gl_FrontFacing;
    let _e96: vec3<f32> = B;
    let _e97: vec3<f32> = B;
    B = select(-(_e97), _e96, _e95);
    let _e100: vec3<f32> = T;
    let _e101: vec3<f32> = B;
    let _e102: vec3<f32> = N2;
    TBN = mat3x3<f32>(vec3<f32>(_e100.x, _e100.y, _e100.z), vec3<f32>(_e101.x, _e101.y, _e101.z), vec3<f32>(_e102.x, _e102.y, _e102.z));
    let _e117: mat3x3<f32> = TBN;
    let _e119: vec2<f32> = v_Uv1;
    let _e120: vec4<f32> = textureSample(StandardMaterial_normal_map, StandardMaterial_normal_map_sampler, _e119);
    let _e128: vec2<f32> = v_Uv1;
    let _e129: vec4<f32> = textureSample(StandardMaterial_normal_map, StandardMaterial_normal_map_sampler, _e128);
    N2 = (_e117 * normalize(((_e129.xyz * 2.0) - vec3<f32>(1.0))));
    let _e139: vec2<f32> = v_Uv1;
    let _e140: vec4<f32> = textureSample(StandardMaterial_occlusion_texture, StandardMaterial_occlusion_texture_sampler, _e139);
    occlusion = _e140.x;
    let _e143: vec4<f32> = global7.emissive;
    emissive = _e143;
    let _e145: vec4<f32> = emissive;
    let _e147: vec4<f32> = emissive;
    let _e150: vec2<f32> = v_Uv1;
    let _e151: vec4<f32> = textureSample(StandardMaterial_emissive_texture, StandardMaterial_emissive_texture_sampler, _e150);
    let _e153: vec3<f32> = (_e147.xyz * _e151.xyz);
    emissive.x = _e153.x;
    emissive.y = _e153.y;
    emissive.z = _e153.z;
    let _e160: vec4<f32> = global1.CameraPos;
    let _e162: vec3<f32> = v_WorldPosition1;
    let _e165: vec4<f32> = global1.CameraPos;
    let _e167: vec3<f32> = v_WorldPosition1;
    V3 = normalize((_e165.xyz - _e167.xyz));
    let _e174: vec3<f32> = N2;
    let _e175: vec3<f32> = V3;
    let _e180: vec3<f32> = N2;
    let _e181: vec3<f32> = V3;
    NdotV4 = max(dot(_e180, _e181), 0.00009999999747378752);
    let _e187: f32 = global6.reflectance;
    let _e189: f32 = global6.reflectance;
    let _e192: f32 = metallic;
    let _e196: vec4<f32> = output_color;
    let _e198: f32 = metallic;
    F0_4 = (vec3<f32>((((0.1599999964237213 * _e187) * _e189) * (1.0 - _e192))) + (_e196.xyz * vec3<f32>(_e198)));
    let _e203: vec4<f32> = output_color;
    let _e206: f32 = metallic;
    diffuseColor4 = (_e203.xyz * vec3<f32>((1.0 - _e206)));
    let _e211: vec3<f32> = V3;
    let _e214: vec3<f32> = V3;
    let _e216: vec3<f32> = N2;
    R4 = reflect(-(_e214), _e216);
    loop {
        let _e224: i32 = i;
        let _e225: vec4<u32> = global2.NumLights;
        let _e229: i32 = i;
        if (!(((_e224 < i32(_e225.x)) && (_e229 < 10)))) {
            break;
        }
        {
            let _e236: vec3<f32> = light_accum;
            let _e237: i32 = i;
            let _e247: i32 = i;
            let _e249: PointLight = global2.PointLights[_e247];
            let _e250: f32 = roughness12;
            let _e251: f32 = NdotV4;
            let _e252: vec3<f32> = N2;
            let _e253: vec3<f32> = V3;
            let _e254: vec3<f32> = R4;
            let _e255: vec3<f32> = F0_4;
            let _e256: vec3<f32> = diffuseColor4;
            let _e257: vec3<f32> = point_light(_e249, _e250, _e251, _e252, _e253, _e254, _e255, _e256);
            light_accum = (_e236 + _e257);
        }
        continuing {
            let _e233: i32 = i;
            i = (_e233 + 1);
        }
    }
    loop {
        let _e261: i32 = i1;
        let _e262: vec4<u32> = global2.NumLights;
        let _e266: i32 = i1;
        if (!(((_e261 < i32(_e262.y)) && (_e266 < 1)))) {
            break;
        }
        {
            let _e273: vec3<f32> = light_accum;
            let _e274: i32 = i1;
            let _e284: i32 = i1;
            let _e286: DirectionalLight = global2.DirectionalLights[_e284];
            let _e287: f32 = roughness12;
            let _e288: f32 = NdotV4;
            let _e289: vec3<f32> = N2;
            let _e290: vec3<f32> = V3;
            let _e291: vec3<f32> = R4;
            let _e292: vec3<f32> = F0_4;
            let _e293: vec3<f32> = diffuseColor4;
            let _e294: vec3<f32> = dir_light(_e286, _e287, _e288, _e289, _e290, _e291, _e292, _e293);
            light_accum = (_e273 + _e294);
        }
        continuing {
            let _e270: i32 = i1;
            i1 = (_e270 + 1);
        }
    }
    let _e299: vec3<f32> = diffuseColor4;
    let _e301: f32 = NdotV4;
    let _e302: vec3<f32> = EnvBRDFApprox(_e299, 1.0, _e301);
    diffuse_ambient = _e302;
    let _e307: vec3<f32> = F0_4;
    let _e308: f32 = perceptual_roughness2;
    let _e309: f32 = NdotV4;
    let _e310: vec3<f32> = EnvBRDFApprox(_e307, _e308, _e309);
    specular_ambient = _e310;
    let _e312: vec4<f32> = output_color;
    let _e314: vec3<f32> = light_accum;
    output_color.x = _e314.x;
    output_color.y = _e314.y;
    output_color.z = _e314.z;
    let _e321: vec4<f32> = output_color;
    let _e323: vec4<f32> = output_color;
    let _e325: vec3<f32> = diffuse_ambient;
    let _e326: vec3<f32> = specular_ambient;
    let _e328: vec4<f32> = global2.AmbientColor;
    let _e331: f32 = occlusion;
    let _e333: vec3<f32> = (_e323.xyz + (((_e325 + _e326) * _e328.xyz) * _e331));
    output_color.x = _e333.x;
    output_color.y = _e333.y;
    output_color.z = _e333.z;
    let _e340: vec4<f32> = output_color;
    let _e342: vec4<f32> = output_color;
    let _e344: vec4<f32> = emissive;
    let _e346: vec4<f32> = output_color;
    let _e349: vec3<f32> = (_e342.xyz + (_e344.xyz * _e346.w));
    output_color.x = _e349.x;
    output_color.y = _e349.y;
    output_color.z = _e349.z;
    let _e356: vec4<f32> = output_color;
    let _e358: vec4<f32> = output_color;
    let _e360: vec4<f32> = output_color;
    let _e362: vec3<f32> = reinhard_luminance(_e360.xyz);
    output_color.x = _e362.x;
    output_color.y = _e362.y;
    output_color.z = _e362.z;
    let _e369: vec4<f32> = output_color;
    o_Target = _e369;
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
    let _e72: vec4<f32> = o_Target;
    return FragmentOutput(_e72);
}
