struct PointLight {
    pos: vec4<f32>;
    color: vec4<f32>;
    lightParams: vec4<f32>;
};

struct DirectionalLight {
    direction: vec4<f32>;
    color1: vec4<f32>;
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
    [[location(0), interpolate(perspective)]] member: vec4<f32>;
};

var<private> gen_entry_v_WorldPosition: vec3<f32>;
var<private> gen_entry_v_WorldNormal: vec3<f32>;
var<private> gen_entry_v_Uv: vec2<f32>;
var<private> gen_entry_v_WorldTangent: vec4<f32>;
var<private> gen_entry_o_Target: vec4<f32>;
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
    smoothFactor = clamp((1.0 - (_e49 * _e50)), 0.0, 1.0);
    let _e57: f32 = smoothFactor;
    let _e58: f32 = smoothFactor;
    attenuation = (_e57 * _e58);
    let _e61: f32 = attenuation;
    let _e64: f32 = distanceSquare1;
    return ((_e61 * 1.0) / max(_e64, 0.00009999999747378752));
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
    lambdaV = (_e50 * sqrt((((_e51 - (_e52 * _e53)) * _e56) + _e58)));
    let _e63: f32 = NoV1;
    let _e64: f32 = NoL1;
    let _e65: f32 = a2_;
    let _e66: f32 = NoL1;
    let _e69: f32 = NoL1;
    let _e71: f32 = a2_;
    lambdaL = (_e63 * sqrt((((_e64 - (_e65 * _e66)) * _e69) + _e71)));
    let _e77: f32 = lambdaV;
    let _e78: f32 = lambdaL;
    v = (0.5 / (_e77 + _e78));
    let _e82: f32 = v;
    return _e82;
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
    let _e44: vec3<f32> = f0_4;
    f90_4 = clamp(dot(_e44, vec3<f32>((50.0 * 0.33000001311302185))), 0.0, 1.0);
    let _e57: vec3<f32> = f0_4;
    let _e58: f32 = f90_4;
    let _e59: f32 = LoH1;
    let _e60: vec3<f32> = F_Schlick(_e57, _e58, _e59);
    return _e60;
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

fn EnvBRDFApprox(f0_7: vec3<f32>, perceptual_roughness1: f32, NoV6: f32) -> vec3<f32> {
    var f0_8: vec3<f32>;
    var perceptual_roughness2: f32;
    var NoV7: f32;
    var c0_: vec4<f32> = vec4<f32>(-1.0, -0.027499999850988388, -0.5720000267028809, 0.02199999988079071);
    var c1_: vec4<f32> = vec4<f32>(1.0, 0.042500000447034836, 1.0399999618530273, -0.03999999910593033);
    var r: vec4<f32>;
    var a004_: f32;
    var AB: vec2<f32>;

    f0_8 = f0_7;
    perceptual_roughness2 = perceptual_roughness1;
    NoV7 = NoV6;
    let _e62: f32 = perceptual_roughness2;
    let _e64: vec4<f32> = c0_;
    let _e66: vec4<f32> = c1_;
    r = ((vec4<f32>(_e62) * _e64) + _e66);
    let _e69: vec4<f32> = r;
    let _e71: vec4<f32> = r;
    let _e76: f32 = NoV7;
    let _e80: vec4<f32> = r;
    let _e83: vec4<f32> = r;
    a004_ = ((min((_e69.x * _e71.x), exp2((-(9.279999732971191) * _e76))) * _e80.x) + _e83.y);
    let _e91: f32 = a004_;
    let _e94: vec4<f32> = r;
    AB = ((vec2<f32>(-(1.0399999618530273), 1.0399999618530273) * vec2<f32>(_e91)) + _e94.zw);
    let _e98: vec3<f32> = f0_8;
    let _e99: vec2<f32> = AB;
    let _e103: vec2<f32> = AB;
    return ((_e98 * vec3<f32>(_e99.x)) + vec3<f32>(_e103.y));
}

fn perceptualRoughnessToRoughness(perceptualRoughness: f32) -> f32 {
    var perceptualRoughness1: f32;
    var clampedPerceptualRoughness: f32;

    perceptualRoughness1 = perceptualRoughness;
    let _e42: f32 = perceptualRoughness1;
    clampedPerceptualRoughness = clamp(_e42, 0.08900000154972076, 1.0);
    let _e47: f32 = clampedPerceptualRoughness;
    let _e48: f32 = clampedPerceptualRoughness;
    return (_e47 * _e48);
}

fn reinhard(color2: vec3<f32>) -> vec3<f32> {
    var color3: vec3<f32>;

    color3 = color2;
    let _e42: vec3<f32> = color3;
    let _e45: vec3<f32> = color3;
    return (_e42 / (vec3<f32>(1.0) + _e45));
}

fn reinhard_extended(color4: vec3<f32>, max_white: f32) -> vec3<f32> {
    var color5: vec3<f32>;
    var max_white1: f32;
    var numerator: vec3<f32>;

    color5 = color4;
    max_white1 = max_white;
    let _e44: vec3<f32> = color5;
    let _e47: vec3<f32> = color5;
    let _e48: f32 = max_white1;
    let _e49: f32 = max_white1;
    numerator = (_e44 * (vec3<f32>(1.0) + (_e47 / vec3<f32>((_e48 * _e49)))));
    let _e56: vec3<f32> = numerator;
    let _e59: vec3<f32> = color5;
    return (_e56 / (vec3<f32>(1.0) + _e59));
}

fn luminance(v1: vec3<f32>) -> f32 {
    var v2: vec3<f32>;

    v2 = v1;
    let _e42: vec3<f32> = v2;
    return dot(_e42, vec3<f32>(0.2125999927520752, 0.7152000069618225, 0.0722000002861023));
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

fn reinhard_luminance(color6: vec3<f32>) -> vec3<f32> {
    var color7: vec3<f32>;
    var l_old: f32;
    var l_new: f32;

    color7 = color6;
    let _e43: vec3<f32> = color7;
    let _e44: f32 = luminance(_e43);
    l_old = _e44;
    let _e46: f32 = l_old;
    let _e48: f32 = l_old;
    l_new = (_e46 / (1.0 + _e48));
    let _e54: vec3<f32> = color7;
    let _e55: f32 = l_new;
    let _e56: vec3<f32> = change_luminance(_e54, _e55);
    return _e56;
}

fn reinhard_extended_luminance(color8: vec3<f32>, max_white_l: f32) -> vec3<f32> {
    var color9: vec3<f32>;
    var max_white_l1: f32;
    var l_old1: f32;
    var numerator1: f32;
    var l_new1: f32;

    color9 = color8;
    max_white_l1 = max_white_l;
    let _e45: vec3<f32> = color9;
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
    let _e66: vec3<f32> = color9;
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
    let _e59: vec3<f32> = gen_entry_v_WorldPosition;
    light_to_frag = (_e56.pos.xyz - _e59.xyz);
    let _e63: vec3<f32> = light_to_frag;
    let _e64: vec3<f32> = light_to_frag;
    distance_square = dot(_e63, _e64);
    let _e68: PointLight = light1;
    let _e71: f32 = distance_square;
    let _e72: PointLight = light1;
    let _e75: f32 = getDistanceAttenuation(_e71, _e72.lightParams.x);
    rangeAttenuation = _e75;
    let _e77: f32 = roughness9;
    a1 = _e77;
    let _e79: PointLight = light1;
    radius = _e79.lightParams.y;
    let _e83: vec3<f32> = light_to_frag;
    let _e84: vec3<f32> = R1;
    let _e86: vec3<f32> = R1;
    let _e88: vec3<f32> = light_to_frag;
    centerToRay = ((dot(_e83, _e84) * _e86) - _e88);
    let _e91: vec3<f32> = light_to_frag;
    let _e92: vec3<f32> = centerToRay;
    let _e93: f32 = radius;
    let _e94: vec3<f32> = centerToRay;
    let _e95: vec3<f32> = centerToRay;
    closestPoint = (_e91 + (_e92 * clamp((_e93 * inverseSqrt(dot(_e94, _e95))), 0.0, 1.0)));
    let _e105: vec3<f32> = closestPoint;
    let _e106: vec3<f32> = closestPoint;
    LspecLengthInverse = inverseSqrt(dot(_e105, _e106));
    let _e110: f32 = a1;
    let _e111: f32 = a1;
    let _e112: f32 = radius;
    let _e115: f32 = LspecLengthInverse;
    normalizationFactor = (_e110 / clamp((_e111 + ((_e112 * 0.5) * _e115)), 0.0, 1.0));
    let _e123: f32 = normalizationFactor;
    let _e124: f32 = normalizationFactor;
    specularIntensity2 = (_e123 * _e124);
    let _e127: vec3<f32> = closestPoint;
    let _e128: f32 = LspecLengthInverse;
    L = (_e127 * _e128);
    let _e131: vec3<f32> = L;
    let _e132: vec3<f32> = V2;
    H = normalize((_e131 + _e132));
    let _e136: vec3<f32> = N1;
    let _e137: vec3<f32> = L;
    NoL6 = clamp(dot(_e136, _e137), 0.0, 1.0);
    let _e143: vec3<f32> = N1;
    let _e144: vec3<f32> = H;
    NoH4 = clamp(dot(_e143, _e144), 0.0, 1.0);
    let _e150: vec3<f32> = L;
    let _e151: vec3<f32> = H;
    LoH6 = clamp(dot(_e150, _e151), 0.0, 1.0);
    let _e165: vec3<f32> = F0_1;
    let _e166: f32 = roughness9;
    let _e167: vec3<f32> = H;
    let _e168: f32 = NdotV1;
    let _e169: f32 = NoL6;
    let _e170: f32 = NoH4;
    let _e171: f32 = LoH6;
    let _e172: f32 = specularIntensity2;
    let _e173: vec3<f32> = specular(_e165, _e166, _e167, _e168, _e169, _e170, _e171, _e172);
    specular1 = _e173;
    let _e175: vec3<f32> = light_to_frag;
    L = normalize(_e175);
    let _e177: vec3<f32> = L;
    let _e178: vec3<f32> = V2;
    H = normalize((_e177 + _e178));
    let _e181: vec3<f32> = N1;
    let _e182: vec3<f32> = L;
    NoL6 = clamp(dot(_e181, _e182), 0.0, 1.0);
    let _e187: vec3<f32> = N1;
    let _e188: vec3<f32> = H;
    NoH4 = clamp(dot(_e187, _e188), 0.0, 1.0);
    let _e193: vec3<f32> = L;
    let _e194: vec3<f32> = H;
    LoH6 = clamp(dot(_e193, _e194), 0.0, 1.0);
    let _e199: vec3<f32> = diffuseColor1;
    let _e204: f32 = roughness9;
    let _e205: f32 = NdotV1;
    let _e206: f32 = NoL6;
    let _e207: f32 = LoH6;
    let _e208: f32 = Fd_Burley(_e204, _e205, _e206, _e207);
    diffuse = (_e199 * _e208);
    let _e211: vec3<f32> = diffuse;
    let _e212: vec3<f32> = specular1;
    let _e214: PointLight = light1;
    let _e218: f32 = rangeAttenuation;
    let _e219: f32 = NoL6;
    return (((_e211 + _e212) * _e214.color.xyz) * (_e218 * _e219));
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
    half_vector = normalize((_e60 + _e61));
    let _e65: vec3<f32> = normal1;
    let _e66: vec3<f32> = incident_light;
    NoL7 = clamp(dot(_e65, _e66), 0.0, 1.0);
    let _e72: vec3<f32> = normal1;
    let _e73: vec3<f32> = half_vector;
    NoH5 = clamp(dot(_e72, _e73), 0.0, 1.0);
    let _e79: vec3<f32> = incident_light;
    let _e80: vec3<f32> = half_vector;
    LoH7 = clamp(dot(_e79, _e80), 0.0, 1.0);
    let _e86: vec3<f32> = diffuseColor3;
    let _e91: f32 = roughness11;
    let _e92: f32 = NdotV3;
    let _e93: f32 = NoL7;
    let _e94: f32 = LoH7;
    let _e95: f32 = Fd_Burley(_e91, _e92, _e93, _e94);
    diffuse1 = (_e86 * _e95);
    let _e108: vec3<f32> = F0_3;
    let _e109: f32 = roughness11;
    let _e110: vec3<f32> = half_vector;
    let _e111: f32 = NdotV3;
    let _e112: f32 = NoL7;
    let _e113: f32 = NoH5;
    let _e114: f32 = LoH7;
    let _e115: f32 = specularIntensity3;
    let _e116: vec3<f32> = specular(_e108, _e109, _e110, _e111, _e112, _e113, _e114, _e115);
    specular2 = _e116;
    let _e118: vec3<f32> = specular2;
    let _e119: vec3<f32> = diffuse1;
    let _e121: DirectionalLight = light3;
    let _e125: f32 = NoL7;
    return (((_e118 + _e119) * _e121.color1.xyz) * _e125);
}

fn main() {
    var output_color: vec4<f32>;
    var metallic_roughness: vec4<f32>;
    var metallic1: f32;
    var perceptual_roughness3: f32;
    var roughness12: f32;
    var N2: vec3<f32>;
    var T: vec3<f32>;
    var B: vec3<f32>;
    var TBN: mat3x3<f32>;
    var occlusion: f32;
    var emissive1: vec4<f32>;
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
    let _e43: vec2<f32> = gen_entry_v_Uv;
    let _e44: vec4<f32> = textureSample(StandardMaterial_base_color_texture, StandardMaterial_base_color_texture_sampler, _e43);
    output_color = (_e42 * _e44);
    let _e46: vec2<f32> = gen_entry_v_Uv;
    let _e47: vec4<f32> = textureSample(StandardMaterial_metallic_roughness_texture, StandardMaterial_metallic_roughness_texture_sampler, _e46);
    metallic_roughness = _e47;
    let _e49: f32 = global5.metallic;
    let _e50: vec4<f32> = metallic_roughness;
    metallic1 = (_e49 * _e50.z);
    let _e54: f32 = global4.perceptual_roughness;
    let _e55: vec4<f32> = metallic_roughness;
    perceptual_roughness3 = (_e54 * _e55.y);
    let _e60: f32 = perceptual_roughness3;
    let _e61: f32 = perceptualRoughnessToRoughness(_e60);
    roughness12 = _e61;
    let _e63: vec3<f32> = gen_entry_v_WorldNormal;
    N2 = normalize(_e63);
    let _e66: vec4<f32> = gen_entry_v_WorldTangent;
    T = normalize(_e66.xyz);
    let _e70: vec3<f32> = N2;
    let _e71: vec3<f32> = T;
    let _e73: vec4<f32> = gen_entry_v_WorldTangent;
    B = (cross(_e70, _e71) * _e73.w);
    let _e78: bool = gl_FrontFacing;
    let _e79: vec3<f32> = N2;
    let _e80: vec3<f32> = N2;
    N2 = select(_e79, -(_e80), _e78);
    let _e83: bool = gl_FrontFacing;
    let _e84: vec3<f32> = T;
    let _e85: vec3<f32> = T;
    T = select(_e84, -(_e85), _e83);
    let _e88: bool = gl_FrontFacing;
    let _e89: vec3<f32> = B;
    let _e90: vec3<f32> = B;
    B = select(_e89, -(_e90), _e88);
    let _e93: vec3<f32> = T;
    let _e94: vec3<f32> = B;
    let _e95: vec3<f32> = N2;
    TBN = mat3x3<f32>(_e93, _e94, _e95);
    let _e98: mat3x3<f32> = TBN;
    let _e99: vec2<f32> = gen_entry_v_Uv;
    let _e100: vec4<f32> = textureSample(StandardMaterial_normal_map, StandardMaterial_normal_map_sampler, _e99);
    N2 = (_e98 * normalize(((_e100.xyz * 2.0) - vec3<f32>(1.0))));
    let _e109: vec2<f32> = gen_entry_v_Uv;
    let _e110: vec4<f32> = textureSample(StandardMaterial_occlusion_texture, StandardMaterial_occlusion_texture_sampler, _e109);
    occlusion = _e110.x;
    let _e113: vec4<f32> = global7.emissive;
    emissive1 = _e113;
    let _e115: vec4<f32> = emissive1;
    let _e117: vec4<f32> = emissive1;
    let _e119: vec2<f32> = gen_entry_v_Uv;
    let _e120: vec4<f32> = textureSample(StandardMaterial_emissive_texture, StandardMaterial_emissive_texture_sampler, _e119);
    let _e122: vec3<f32> = (_e117.xyz * _e120.xyz);
    emissive1.x = _e122.x;
    emissive1.y = _e122.y;
    emissive1.z = _e122.z;
    let _e129: vec4<f32> = global1.CameraPos;
    let _e131: vec3<f32> = gen_entry_v_WorldPosition;
    V3 = normalize((_e129.xyz - _e131.xyz));
    let _e136: vec3<f32> = N2;
    let _e137: vec3<f32> = V3;
    NdotV4 = max(dot(_e136, _e137), 0.00009999999747378752);
    let _e143: f32 = global6.reflectance;
    let _e145: f32 = global6.reflectance;
    let _e148: f32 = metallic1;
    let _e152: vec4<f32> = output_color;
    let _e154: f32 = metallic1;
    F0_4 = (vec3<f32>((((0.1599999964237213 * _e143) * _e145) * (1.0 - _e148))) + (_e152.xyz * vec3<f32>(_e154)));
    let _e159: vec4<f32> = output_color;
    let _e162: f32 = metallic1;
    diffuseColor4 = (_e159.xyz * vec3<f32>((1.0 - _e162)));
    let _e167: vec3<f32> = V3;
    let _e169: vec3<f32> = N2;
    R4 = reflect(-(_e167), _e169);
    loop {
        let _e177: i32 = i;
        let _e178: vec4<u32> = global2.NumLights;
        let _e182: i32 = i;
        if (!(((_e177 < i32(_e178.x)) && (_e182 < 10)))) {
            break;
        }
        {
            let _e189: vec3<f32> = light_accum;
            let _e190: i32 = i;
            let _e200: i32 = i;
            let _e202: PointLight = global2.PointLights[_e200];
            let _e203: f32 = roughness12;
            let _e204: f32 = NdotV4;
            let _e205: vec3<f32> = N2;
            let _e206: vec3<f32> = V3;
            let _e207: vec3<f32> = R4;
            let _e208: vec3<f32> = F0_4;
            let _e209: vec3<f32> = diffuseColor4;
            let _e210: vec3<f32> = point_light(_e202, _e203, _e204, _e205, _e206, _e207, _e208, _e209);
            light_accum = (_e189 + _e210);
        }
        continuing {
            let _e186: i32 = i;
            i = (_e186 + 1);
        }
    }
    loop {
        let _e214: i32 = i1;
        let _e215: vec4<u32> = global2.NumLights;
        let _e219: i32 = i1;
        if (!(((_e214 < i32(_e215.y)) && (_e219 < 1)))) {
            break;
        }
        {
            let _e226: vec3<f32> = light_accum;
            let _e227: i32 = i1;
            let _e237: i32 = i1;
            let _e239: DirectionalLight = global2.DirectionalLights[_e237];
            let _e240: f32 = roughness12;
            let _e241: f32 = NdotV4;
            let _e242: vec3<f32> = N2;
            let _e243: vec3<f32> = V3;
            let _e244: vec3<f32> = R4;
            let _e245: vec3<f32> = F0_4;
            let _e246: vec3<f32> = diffuseColor4;
            let _e247: vec3<f32> = dir_light(_e239, _e240, _e241, _e242, _e243, _e244, _e245, _e246);
            light_accum = (_e226 + _e247);
        }
        continuing {
            let _e223: i32 = i1;
            i1 = (_e223 + 1);
        }
    }
    let _e252: vec3<f32> = diffuseColor4;
    let _e254: f32 = NdotV4;
    let _e255: vec3<f32> = EnvBRDFApprox(_e252, 1.0, _e254);
    diffuse_ambient = _e255;
    let _e260: vec3<f32> = F0_4;
    let _e261: f32 = perceptual_roughness3;
    let _e262: f32 = NdotV4;
    let _e263: vec3<f32> = EnvBRDFApprox(_e260, _e261, _e262);
    specular_ambient = _e263;
    let _e265: vec4<f32> = output_color;
    let _e267: vec3<f32> = light_accum;
    output_color.x = _e267.x;
    output_color.y = _e267.y;
    output_color.z = _e267.z;
    let _e274: vec4<f32> = output_color;
    let _e276: vec4<f32> = output_color;
    let _e278: vec3<f32> = diffuse_ambient;
    let _e279: vec3<f32> = specular_ambient;
    let _e281: vec4<f32> = global2.AmbientColor;
    let _e284: f32 = occlusion;
    let _e286: vec3<f32> = (_e276.xyz + (((_e278 + _e279) * _e281.xyz) * _e284));
    output_color.x = _e286.x;
    output_color.y = _e286.y;
    output_color.z = _e286.z;
    let _e293: vec4<f32> = output_color;
    let _e295: vec4<f32> = output_color;
    let _e297: vec4<f32> = emissive1;
    let _e299: vec4<f32> = output_color;
    let _e302: vec3<f32> = (_e295.xyz + (_e297.xyz * _e299.w));
    output_color.x = _e302.x;
    output_color.y = _e302.y;
    output_color.z = _e302.z;
    let _e309: vec4<f32> = output_color;
    let _e311: vec4<f32> = output_color;
    let _e313: vec4<f32> = output_color;
    let _e315: vec3<f32> = reinhard_luminance(_e313.xyz);
    output_color.x = _e315.x;
    output_color.y = _e315.y;
    output_color.z = _e315.z;
    let _e322: vec4<f32> = output_color;
    gen_entry_o_Target = _e322;
    return;
}

[[stage(fragment)]]
fn main1([[location(0), interpolate(perspective)]] v_WorldPosition: vec3<f32>, [[location(1), interpolate(perspective)]] v_WorldNormal: vec3<f32>, [[location(2), interpolate(perspective)]] v_Uv: vec2<f32>, [[location(3), interpolate(perspective)]] v_WorldTangent: vec4<f32>, [[builtin(front_facing)]] param: bool) -> FragmentOutput {
    gen_entry_v_WorldPosition = v_WorldPosition;
    gen_entry_v_WorldNormal = v_WorldNormal;
    gen_entry_v_Uv = v_Uv;
    gen_entry_v_WorldTangent = v_WorldTangent;
    gl_FrontFacing = param;
    main();
    let _e11: vec4<f32> = gen_entry_o_Target;
    return FragmentOutput(_e11);
}
