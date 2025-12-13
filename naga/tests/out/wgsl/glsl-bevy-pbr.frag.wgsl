struct PointLight {
    pos: vec4<f32>,
    color: vec4<f32>,
    lightParams: vec4<f32>,
}

struct DirectionalLight {
    direction: vec4<f32>,
    color: vec4<f32>,
}

struct CameraPosition {
    CameraPos: vec4<f32>,
}

struct Lights {
    AmbientColor: vec4<f32>,
    NumLights: vec4<u32>,
    PointLights: array<PointLight, 10>,
    DirectionalLights: array<DirectionalLight, 1>,
}

struct StandardMaterial_base_color {
    base_color: vec4<f32>,
}

struct StandardMaterial_roughness {
    perceptual_roughness: f32,
}

struct StandardMaterial_metallic {
    metallic: f32,
}

struct StandardMaterial_reflectance {
    reflectance: f32,
}

struct StandardMaterial_emissive {
    emissive: vec4<f32>,
}

struct FragmentOutput {
    @location(0) o_Target: vec4<f32>,
}

const MAX_POINT_LIGHTS: i32 = 10i;
const MAX_DIRECTIONAL_LIGHTS: i32 = 1i;
const PI: f32 = 3.1415927f;

var<private> v_WorldPosition_1: vec3<f32>;
var<private> v_WorldNormal_1: vec3<f32>;
var<private> v_Uv_1: vec2<f32>;
var<private> v_WorldTangent_1: vec4<f32>;
var<private> o_Target: vec4<f32>;
@group(0) @binding(1) 
var<uniform> global: CameraPosition;
@group(1) @binding(0) 
var<uniform> global_1: Lights;
@group(3) @binding(0) 
var<uniform> global_2: StandardMaterial_base_color;
@group(3) @binding(1) 
var StandardMaterial_base_color_texture: texture_2d<f32>;
@group(3) @binding(2) 
var StandardMaterial_base_color_texture_sampler: sampler;
@group(3) @binding(3) 
var<uniform> global_3: StandardMaterial_roughness;
@group(3) @binding(4) 
var<uniform> global_4: StandardMaterial_metallic;
@group(3) @binding(5) 
var StandardMaterial_metallic_roughness_texture: texture_2d<f32>;
@group(3) @binding(6) 
var StandardMaterial_metallic_roughness_texture_sampler: sampler;
@group(3) @binding(7) 
var<uniform> global_5: StandardMaterial_reflectance;
@group(3) @binding(8) 
var StandardMaterial_normal_map: texture_2d<f32>;
@group(3) @binding(9) 
var StandardMaterial_normal_map_sampler: sampler;
@group(3) @binding(10) 
var StandardMaterial_occlusion_texture: texture_2d<f32>;
@group(3) @binding(11) 
var StandardMaterial_occlusion_texture_sampler: sampler;
@group(3) @binding(12) 
var<uniform> global_6: StandardMaterial_emissive;
@group(3) @binding(13) 
var StandardMaterial_emissive_texture: texture_2d<f32>;
@group(3) @binding(14) 
var StandardMaterial_emissive_texture_sampler: sampler;
var<private> gl_FrontFacing_1: bool;

fn pow5_(x: f32) -> f32 {
    var x_1: f32;
    var x2_: f32;

    x_1 = x;
    let _e2 = x_1;
    let _e3 = x_1;
    x2_ = (_e2 * _e3);
    let _e6 = x2_;
    let _e7 = x2_;
    let _e9 = x_1;
    return ((_e6 * _e7) * _e9);
}

fn getDistanceAttenuation(distanceSquare: f32, inverseRangeSquared: f32) -> f32 {
    var distanceSquare_1: f32;
    var inverseRangeSquared_1: f32;
    var factor: f32;
    var smoothFactor: f32;
    var attenuation: f32;

    distanceSquare_1 = distanceSquare;
    inverseRangeSquared_1 = inverseRangeSquared;
    let _e4 = distanceSquare_1;
    let _e5 = inverseRangeSquared_1;
    factor = (_e4 * _e5);
    let _e9 = factor;
    let _e10 = factor;
    smoothFactor = clamp((1f - (_e9 * _e10)), 0f, 1f);
    let _e17 = smoothFactor;
    let _e18 = smoothFactor;
    attenuation = (_e17 * _e18);
    let _e21 = attenuation;
    let _e24 = distanceSquare_1;
    return ((_e21 * 1f) / max(_e24, 0.001f));
}

fn D_GGX(roughness: f32, NoH: f32, h: vec3<f32>) -> f32 {
    var roughness_1: f32;
    var NoH_1: f32;
    var oneMinusNoHSquared: f32;
    var a: f32;
    var k: f32;
    var d: f32;

    roughness_1 = roughness;
    NoH_1 = NoH;
    let _e5 = NoH_1;
    let _e6 = NoH_1;
    oneMinusNoHSquared = (1f - (_e5 * _e6));
    let _e10 = NoH_1;
    let _e11 = roughness_1;
    a = (_e10 * _e11);
    let _e14 = roughness_1;
    let _e15 = oneMinusNoHSquared;
    let _e16 = a;
    let _e17 = a;
    k = (_e14 / (_e15 + (_e16 * _e17)));
    let _e22 = k;
    let _e23 = k;
    d = ((_e22 * _e23) * 0.31830987f);
    let _e28 = d;
    return _e28;
}

fn V_SmithGGXCorrelated(roughness_2: f32, NoV: f32, NoL: f32) -> f32 {
    var roughness_3: f32;
    var NoV_1: f32;
    var NoL_1: f32;
    var a2_: f32;
    var lambdaV: f32;
    var lambdaL: f32;
    var v: f32;

    roughness_3 = roughness_2;
    NoV_1 = NoV;
    NoL_1 = NoL;
    let _e6 = roughness_3;
    let _e7 = roughness_3;
    a2_ = (_e6 * _e7);
    let _e10 = NoL_1;
    let _e11 = NoV_1;
    let _e12 = a2_;
    let _e13 = NoV_1;
    let _e16 = NoV_1;
    let _e18 = a2_;
    lambdaV = (_e10 * sqrt((((_e11 - (_e12 * _e13)) * _e16) + _e18)));
    let _e23 = NoV_1;
    let _e24 = NoL_1;
    let _e25 = a2_;
    let _e26 = NoL_1;
    let _e29 = NoL_1;
    let _e31 = a2_;
    lambdaL = (_e23 * sqrt((((_e24 - (_e25 * _e26)) * _e29) + _e31)));
    let _e37 = lambdaV;
    let _e38 = lambdaL;
    v = (0.5f / (_e37 + _e38));
    let _e42 = v;
    return _e42;
}

fn F_Schlick(f0_: vec3<f32>, f90_: f32, VoH: f32) -> vec3<f32> {
    var f90_1: f32;
    var VoH_1: f32;

    f90_1 = f90_;
    VoH_1 = VoH;
    let _e5 = f90_1;
    let _e9 = VoH_1;
    let _e11 = pow5_((1f - _e9));
    return (f0_ + ((vec3(_e5) - f0_) * _e11));
}

fn F_Schlick_1(f0_1: f32, f90_2: f32, VoH_2: f32) -> f32 {
    var f0_2: f32;
    var f90_3: f32;
    var VoH_3: f32;

    f0_2 = f0_1;
    f90_3 = f90_2;
    VoH_3 = VoH_2;
    let _e6 = f0_2;
    let _e7 = f90_3;
    let _e8 = f0_2;
    let _e11 = VoH_3;
    let _e13 = pow5_((1f - _e11));
    return (_e6 + ((_e7 - _e8) * _e13));
}

fn fresnel(f0_3: vec3<f32>, LoH: f32) -> vec3<f32> {
    var f0_4: vec3<f32>;
    var LoH_1: f32;
    var f90_4: f32;

    f0_4 = f0_3;
    LoH_1 = LoH;
    let _e4 = f0_4;
    f90_4 = clamp(dot(_e4, vec3(16.5f)), 0f, 1f);
    let _e12 = f0_4;
    let _e13 = f90_4;
    let _e14 = LoH_1;
    let _e15 = F_Schlick(_e12, _e13, _e14);
    return _e15;
}

fn specular(f0_5: vec3<f32>, roughness_4: f32, h_1: vec3<f32>, NoV_2: f32, NoL_2: f32, NoH_2: f32, LoH_2: f32, specularIntensity: f32) -> vec3<f32> {
    var f0_6: vec3<f32>;
    var roughness_5: f32;
    var NoV_3: f32;
    var NoL_3: f32;
    var NoH_3: f32;
    var LoH_3: f32;
    var specularIntensity_1: f32;
    var D: f32;
    var V: f32;
    var F: vec3<f32>;

    f0_6 = f0_5;
    roughness_5 = roughness_4;
    NoV_3 = NoV_2;
    NoL_3 = NoL_2;
    NoH_3 = NoH_2;
    LoH_3 = LoH_2;
    specularIntensity_1 = specularIntensity;
    let _e15 = roughness_5;
    let _e16 = NoH_3;
    let _e17 = D_GGX(_e15, _e16, h_1);
    D = _e17;
    let _e19 = roughness_5;
    let _e20 = NoV_3;
    let _e21 = NoL_3;
    let _e22 = V_SmithGGXCorrelated(_e19, _e20, _e21);
    V = _e22;
    let _e24 = f0_6;
    let _e25 = LoH_3;
    let _e26 = fresnel(_e24, _e25);
    F = _e26;
    let _e28 = specularIntensity_1;
    let _e29 = D;
    let _e31 = V;
    let _e33 = F;
    return (((_e28 * _e29) * _e31) * _e33);
}

fn Fd_Burley(roughness_6: f32, NoV_4: f32, NoL_4: f32, LoH_4: f32) -> f32 {
    var roughness_7: f32;
    var NoV_5: f32;
    var NoL_5: f32;
    var LoH_5: f32;
    var f90_5: f32;
    var lightScatter: f32;
    var viewScatter: f32;

    roughness_7 = roughness_6;
    NoV_5 = NoV_4;
    NoL_5 = NoL_4;
    LoH_5 = LoH_4;
    let _e10 = roughness_7;
    let _e12 = LoH_5;
    let _e14 = LoH_5;
    f90_5 = (0.5f + (((2f * _e10) * _e12) * _e14));
    let _e19 = f90_5;
    let _e20 = NoL_5;
    let _e21 = F_Schlick_1(1f, _e19, _e20);
    lightScatter = _e21;
    let _e24 = f90_5;
    let _e25 = NoV_5;
    let _e26 = F_Schlick_1(1f, _e24, _e25);
    viewScatter = _e26;
    let _e28 = lightScatter;
    let _e29 = viewScatter;
    return ((_e28 * _e29) * 0.31830987f);
}

fn EnvBRDFApprox(f0_7: vec3<f32>, perceptual_roughness: f32, NoV_6: f32) -> vec3<f32> {
    var f0_8: vec3<f32>;
    var perceptual_roughness_1: f32;
    var NoV_7: f32;
    var c0_: vec4<f32> = vec4<f32>(-1f, -0.0275f, -0.572f, 0.022f);
    var c1_: vec4<f32> = vec4<f32>(1f, 0.0425f, 1.04f, -0.04f);
    var r: vec4<f32>;
    var a004_: f32;
    var AB: vec2<f32>;

    f0_8 = f0_7;
    perceptual_roughness_1 = perceptual_roughness;
    NoV_7 = NoV_6;
    let _e18 = perceptual_roughness_1;
    let _e20 = c0_;
    let _e22 = c1_;
    r = ((vec4(_e18) * _e20) + _e22);
    let _e25 = r;
    let _e27 = r;
    let _e31 = NoV_7;
    let _e35 = r;
    let _e38 = r;
    a004_ = ((min((_e25.x * _e27.x), exp2((-9.28f * _e31))) * _e35.x) + _e38.y);
    let _e45 = a004_;
    let _e48 = r;
    AB = ((vec2<f32>(-1.04f, 1.04f) * vec2(_e45)) + _e48.zw);
    let _e52 = f0_8;
    let _e53 = AB;
    let _e57 = AB;
    return ((_e52 * vec3(_e53.x)) + vec3(_e57.y));
}

fn perceptualRoughnessToRoughness(perceptualRoughness: f32) -> f32 {
    var perceptualRoughness_1: f32;
    var clampedPerceptualRoughness: f32;

    perceptualRoughness_1 = perceptualRoughness;
    let _e2 = perceptualRoughness_1;
    clampedPerceptualRoughness = clamp(_e2, 0.089f, 1f);
    let _e7 = clampedPerceptualRoughness;
    let _e8 = clampedPerceptualRoughness;
    return (_e7 * _e8);
}

fn luminance(v_1: vec3<f32>) -> f32 {
    var v_2: vec3<f32>;

    v_2 = v_1;
    let _e2 = v_2;
    return dot(_e2, vec3<f32>(0.2126f, 0.7152f, 0.0722f));
}

fn change_luminance(c_in: vec3<f32>, l_out: f32) -> vec3<f32> {
    var c_in_1: vec3<f32>;
    var l_out_1: f32;
    var l_in: f32;

    c_in_1 = c_in;
    l_out_1 = l_out;
    let _e4 = c_in_1;
    let _e5 = luminance(_e4);
    l_in = _e5;
    let _e7 = c_in_1;
    let _e8 = l_out_1;
    let _e9 = l_in;
    return (_e7 * (_e8 / _e9));
}

fn reinhard_luminance(color: vec3<f32>) -> vec3<f32> {
    var color_1: vec3<f32>;
    var l_old: f32;
    var l_new: f32;

    color_1 = color;
    let _e2 = color_1;
    let _e3 = luminance(_e2);
    l_old = _e3;
    let _e5 = l_old;
    let _e7 = l_old;
    l_new = (_e5 / (1f + _e7));
    let _e11 = color_1;
    let _e12 = l_new;
    let _e13 = change_luminance(_e11, _e12);
    return _e13;
}

fn point_light(light: PointLight, roughness_8: f32, NdotV: f32, N: vec3<f32>, V_1: vec3<f32>, R: vec3<f32>, F0_: vec3<f32>, diffuseColor: vec3<f32>) -> vec3<f32> {
    var light_1: PointLight;
    var roughness_9: f32;
    var NdotV_1: f32;
    var N_1: vec3<f32>;
    var V_2: vec3<f32>;
    var R_1: vec3<f32>;
    var F0_1: vec3<f32>;
    var diffuseColor_1: vec3<f32>;
    var light_to_frag: vec3<f32>;
    var distance_square: f32;
    var rangeAttenuation: f32;
    var a_1: f32;
    var radius: f32;
    var centerToRay: vec3<f32>;
    var closestPoint: vec3<f32>;
    var LspecLengthInverse: f32;
    var normalizationFactor: f32;
    var specularIntensity_2: f32;
    var L: vec3<f32>;
    var H: vec3<f32>;
    var NoL_6: f32;
    var NoH_4: f32;
    var LoH_6: f32;
    var specular_1: vec3<f32>;
    var diffuse: vec3<f32>;

    light_1 = light;
    roughness_9 = roughness_8;
    NdotV_1 = NdotV;
    N_1 = N;
    V_2 = V_1;
    R_1 = R;
    F0_1 = F0_;
    diffuseColor_1 = diffuseColor;
    let _e17 = light_1;
    let _e20 = v_WorldPosition_1;
    light_to_frag = (_e17.pos.xyz - _e20.xyz);
    let _e24 = light_to_frag;
    let _e25 = light_to_frag;
    distance_square = dot(_e24, _e25);
    let _e28 = distance_square;
    let _e29 = light_1;
    let _e32 = getDistanceAttenuation(_e28, _e29.lightParams.x);
    rangeAttenuation = _e32;
    let _e34 = roughness_9;
    a_1 = _e34;
    let _e36 = light_1;
    radius = _e36.lightParams.y;
    let _e40 = light_to_frag;
    let _e41 = R_1;
    let _e43 = R_1;
    let _e45 = light_to_frag;
    centerToRay = ((dot(_e40, _e41) * _e43) - _e45);
    let _e48 = light_to_frag;
    let _e49 = centerToRay;
    let _e50 = radius;
    let _e51 = centerToRay;
    let _e52 = centerToRay;
    closestPoint = (_e48 + (_e49 * clamp((_e50 * inverseSqrt(dot(_e51, _e52))), 0f, 1f)));
    let _e62 = closestPoint;
    let _e63 = closestPoint;
    LspecLengthInverse = inverseSqrt(dot(_e62, _e63));
    let _e67 = a_1;
    let _e68 = a_1;
    let _e69 = radius;
    let _e72 = LspecLengthInverse;
    normalizationFactor = (_e67 / clamp((_e68 + ((_e69 * 0.5f) * _e72)), 0f, 1f));
    let _e80 = normalizationFactor;
    let _e81 = normalizationFactor;
    specularIntensity_2 = (_e80 * _e81);
    let _e84 = closestPoint;
    let _e85 = LspecLengthInverse;
    L = (_e84 * _e85);
    let _e88 = L;
    let _e89 = V_2;
    H = normalize((_e88 + _e89));
    let _e93 = N_1;
    let _e94 = L;
    NoL_6 = clamp(dot(_e93, _e94), 0f, 1f);
    let _e100 = N_1;
    let _e101 = H;
    NoH_4 = clamp(dot(_e100, _e101), 0f, 1f);
    let _e107 = L;
    let _e108 = H;
    LoH_6 = clamp(dot(_e107, _e108), 0f, 1f);
    let _e114 = F0_1;
    let _e115 = roughness_9;
    let _e116 = H;
    let _e117 = NdotV_1;
    let _e118 = NoL_6;
    let _e119 = NoH_4;
    let _e120 = LoH_6;
    let _e121 = specularIntensity_2;
    let _e122 = specular(_e114, _e115, _e116, _e117, _e118, _e119, _e120, _e121);
    specular_1 = _e122;
    let _e124 = light_to_frag;
    L = normalize(_e124);
    let _e126 = L;
    let _e127 = V_2;
    H = normalize((_e126 + _e127));
    let _e130 = N_1;
    let _e131 = L;
    NoL_6 = clamp(dot(_e130, _e131), 0f, 1f);
    let _e136 = N_1;
    let _e137 = H;
    NoH_4 = clamp(dot(_e136, _e137), 0f, 1f);
    let _e142 = L;
    let _e143 = H;
    LoH_6 = clamp(dot(_e142, _e143), 0f, 1f);
    let _e148 = diffuseColor_1;
    let _e149 = roughness_9;
    let _e150 = NdotV_1;
    let _e151 = NoL_6;
    let _e152 = LoH_6;
    let _e153 = Fd_Burley(_e149, _e150, _e151, _e152);
    diffuse = (_e148 * _e153);
    let _e156 = diffuse;
    let _e157 = specular_1;
    let _e159 = light_1;
    let _e163 = rangeAttenuation;
    let _e164 = NoL_6;
    return (((_e156 + _e157) * _e159.color.xyz) * (_e163 * _e164));
}

fn dir_light(light_2: DirectionalLight, roughness_10: f32, NdotV_2: f32, normal: vec3<f32>, view: vec3<f32>, R_2: vec3<f32>, F0_2: vec3<f32>, diffuseColor_2: vec3<f32>) -> vec3<f32> {
    var light_3: DirectionalLight;
    var roughness_11: f32;
    var NdotV_3: f32;
    var normal_1: vec3<f32>;
    var view_1: vec3<f32>;
    var R_3: vec3<f32>;
    var F0_3: vec3<f32>;
    var diffuseColor_3: vec3<f32>;
    var incident_light: vec3<f32>;
    var half_vector: vec3<f32>;
    var NoL_7: f32;
    var NoH_5: f32;
    var LoH_7: f32;
    var diffuse_1: vec3<f32>;
    var specularIntensity_3: f32 = 1f;
    var specular_2: vec3<f32>;

    light_3 = light_2;
    roughness_11 = roughness_10;
    NdotV_3 = NdotV_2;
    normal_1 = normal;
    view_1 = view;
    R_3 = R_2;
    F0_3 = F0_2;
    diffuseColor_3 = diffuseColor_2;
    let _e16 = light_3;
    incident_light = _e16.direction.xyz;
    let _e20 = incident_light;
    let _e21 = view_1;
    half_vector = normalize((_e20 + _e21));
    let _e25 = normal_1;
    let _e26 = incident_light;
    NoL_7 = clamp(dot(_e25, _e26), 0f, 1f);
    let _e32 = normal_1;
    let _e33 = half_vector;
    NoH_5 = clamp(dot(_e32, _e33), 0f, 1f);
    let _e39 = incident_light;
    let _e40 = half_vector;
    LoH_7 = clamp(dot(_e39, _e40), 0f, 1f);
    let _e46 = diffuseColor_3;
    let _e47 = roughness_11;
    let _e48 = NdotV_3;
    let _e49 = NoL_7;
    let _e50 = LoH_7;
    let _e51 = Fd_Burley(_e47, _e48, _e49, _e50);
    diffuse_1 = (_e46 * _e51);
    let _e56 = F0_3;
    let _e57 = roughness_11;
    let _e58 = half_vector;
    let _e59 = NdotV_3;
    let _e60 = NoL_7;
    let _e61 = NoH_5;
    let _e62 = LoH_7;
    let _e63 = specularIntensity_3;
    let _e64 = specular(_e56, _e57, _e58, _e59, _e60, _e61, _e62, _e63);
    specular_2 = _e64;
    let _e66 = specular_2;
    let _e67 = diffuse_1;
    let _e69 = light_3;
    let _e73 = NoL_7;
    return (((_e66 + _e67) * _e69.color.xyz) * _e73);
}

fn main_1() {
    var output_color: vec4<f32>;
    var metallic_roughness: vec4<f32>;
    var metallic: f32;
    var perceptual_roughness_2: f32;
    var roughness_12: f32;
    var N_2: vec3<f32>;
    var T: vec3<f32>;
    var B: vec3<f32>;
    var local: vec3<f32>;
    var local_1: vec3<f32>;
    var local_2: vec3<f32>;
    var TBN: mat3x3<f32>;
    var occlusion: f32;
    var emissive: vec4<f32>;
    var V_3: vec3<f32>;
    var NdotV_4: f32;
    var F0_4: vec3<f32>;
    var diffuseColor_4: vec3<f32>;
    var R_4: vec3<f32>;
    var light_accum: vec3<f32> = vec3(0f);
    var i: i32 = 0i;
    var i_1: i32 = 0i;
    var diffuse_ambient: vec3<f32>;
    var specular_ambient: vec3<f32>;

    let _e37 = global_2.base_color;
    output_color = _e37;
    let _e39 = output_color;
    let _e40 = v_Uv_1;
    let _e41 = textureSample(StandardMaterial_base_color_texture, StandardMaterial_base_color_texture_sampler, _e40);
    output_color = (_e39 * _e41);
    let _e43 = v_Uv_1;
    let _e44 = textureSample(StandardMaterial_metallic_roughness_texture, StandardMaterial_metallic_roughness_texture_sampler, _e43);
    metallic_roughness = _e44;
    let _e46 = global_4.metallic;
    let _e47 = metallic_roughness;
    metallic = (_e46 * _e47.z);
    let _e51 = global_3.perceptual_roughness;
    let _e52 = metallic_roughness;
    perceptual_roughness_2 = (_e51 * _e52.y);
    let _e56 = perceptual_roughness_2;
    let _e57 = perceptualRoughnessToRoughness(_e56);
    roughness_12 = _e57;
    let _e59 = v_WorldNormal_1;
    N_2 = normalize(_e59);
    let _e62 = v_WorldTangent_1;
    T = normalize(_e62.xyz);
    let _e66 = N_2;
    let _e67 = T;
    let _e69 = v_WorldTangent_1;
    B = (cross(_e66, _e67) * _e69.w);
    let _e74 = gl_FrontFacing_1;
    if _e74 {
        let _e75 = N_2;
        local = _e75;
    } else {
        let _e76 = N_2;
        local = -(_e76);
    }
    let _e79 = local;
    N_2 = _e79;
    let _e80 = gl_FrontFacing_1;
    if _e80 {
        let _e81 = T;
        local_1 = _e81;
    } else {
        let _e82 = T;
        local_1 = -(_e82);
    }
    let _e85 = local_1;
    T = _e85;
    let _e86 = gl_FrontFacing_1;
    if _e86 {
        let _e87 = B;
        local_2 = _e87;
    } else {
        let _e88 = B;
        local_2 = -(_e88);
    }
    let _e91 = local_2;
    B = _e91;
    let _e92 = T;
    let _e93 = B;
    let _e94 = N_2;
    TBN = mat3x3<f32>(vec3<f32>(_e92.x, _e92.y, _e92.z), vec3<f32>(_e93.x, _e93.y, _e93.z), vec3<f32>(_e94.x, _e94.y, _e94.z));
    let _e109 = TBN;
    let _e110 = v_Uv_1;
    let _e111 = textureSample(StandardMaterial_normal_map, StandardMaterial_normal_map_sampler, _e110);
    N_2 = (_e109 * normalize(((_e111.xyz * 2f) - vec3(1f))));
    let _e120 = v_Uv_1;
    let _e121 = textureSample(StandardMaterial_occlusion_texture, StandardMaterial_occlusion_texture_sampler, _e120);
    occlusion = _e121.x;
    let _e124 = global_6.emissive;
    emissive = _e124;
    let _e126 = emissive;
    let _e128 = v_Uv_1;
    let _e129 = textureSample(StandardMaterial_emissive_texture, StandardMaterial_emissive_texture_sampler, _e128);
    let _e131 = (_e126.xyz * _e129.xyz);
    emissive.x = _e131.x;
    emissive.y = _e131.y;
    emissive.z = _e131.z;
    let _e138 = global.CameraPos;
    let _e140 = v_WorldPosition_1;
    V_3 = normalize((_e138.xyz - _e140.xyz));
    let _e145 = N_2;
    let _e146 = V_3;
    NdotV_4 = max(dot(_e145, _e146), 0.001f);
    let _e152 = global_5.reflectance;
    let _e154 = global_5.reflectance;
    let _e157 = metallic;
    let _e161 = output_color;
    let _e163 = metallic;
    F0_4 = (vec3((((0.16f * _e152) * _e154) * (1f - _e157))) + (_e161.xyz * vec3(_e163)));
    let _e168 = output_color;
    let _e171 = metallic;
    diffuseColor_4 = (_e168.xyz * vec3((1f - _e171)));
    let _e176 = V_3;
    let _e178 = N_2;
    R_4 = reflect(-(_e176), _e178);
    loop {
        let _e186 = i;
        let _e187 = global_1.NumLights;
        let _e191 = i;
        if !(((_e186 < i32(_e187.x)) && (_e191 < MAX_POINT_LIGHTS))) {
            break;
        }
        {
            let _e198 = light_accum;
            let _e199 = i;
            let _e201 = global_1.PointLights[_e199];
            let _e202 = roughness_12;
            let _e203 = NdotV_4;
            let _e204 = N_2;
            let _e205 = V_3;
            let _e206 = R_4;
            let _e207 = F0_4;
            let _e208 = diffuseColor_4;
            let _e209 = point_light(_e201, _e202, _e203, _e204, _e205, _e206, _e207, _e208);
            light_accum = (_e198 + _e209);
        }
        continuing {
            let _e195 = i;
            i = (_e195 + 1i);
        }
    }
    loop {
        let _e213 = i_1;
        let _e214 = global_1.NumLights;
        let _e218 = i_1;
        if !(((_e213 < i32(_e214.y)) && (_e218 < MAX_DIRECTIONAL_LIGHTS))) {
            break;
        }
        {
            let _e225 = light_accum;
            let _e226 = i_1;
            let _e228 = global_1.DirectionalLights[_e226];
            let _e229 = roughness_12;
            let _e230 = NdotV_4;
            let _e231 = N_2;
            let _e232 = V_3;
            let _e233 = R_4;
            let _e234 = F0_4;
            let _e235 = diffuseColor_4;
            let _e236 = dir_light(_e228, _e229, _e230, _e231, _e232, _e233, _e234, _e235);
            light_accum = (_e225 + _e236);
        }
        continuing {
            let _e222 = i_1;
            i_1 = (_e222 + 1i);
        }
    }
    let _e238 = diffuseColor_4;
    let _e240 = NdotV_4;
    let _e241 = EnvBRDFApprox(_e238, 1f, _e240);
    diffuse_ambient = _e241;
    let _e243 = F0_4;
    let _e244 = perceptual_roughness_2;
    let _e245 = NdotV_4;
    let _e246 = EnvBRDFApprox(_e243, _e244, _e245);
    specular_ambient = _e246;
    let _e248 = light_accum;
    output_color.x = _e248.x;
    output_color.y = _e248.y;
    output_color.z = _e248.z;
    let _e255 = output_color;
    let _e257 = diffuse_ambient;
    let _e258 = specular_ambient;
    let _e260 = global_1.AmbientColor;
    let _e263 = occlusion;
    let _e265 = (_e255.xyz + (((_e257 + _e258) * _e260.xyz) * _e263));
    output_color.x = _e265.x;
    output_color.y = _e265.y;
    output_color.z = _e265.z;
    let _e272 = output_color;
    let _e274 = emissive;
    let _e276 = output_color;
    let _e279 = (_e272.xyz + (_e274.xyz * _e276.w));
    output_color.x = _e279.x;
    output_color.y = _e279.y;
    output_color.z = _e279.z;
    let _e286 = output_color;
    let _e288 = reinhard_luminance(_e286.xyz);
    output_color.x = _e288.x;
    output_color.y = _e288.y;
    output_color.z = _e288.z;
    let _e295 = output_color;
    o_Target = _e295;
    return;
}

@fragment 
fn main(@location(0) v_WorldPosition: vec3<f32>, @location(1) v_WorldNormal: vec3<f32>, @location(2) v_Uv: vec2<f32>, @location(3) v_WorldTangent: vec4<f32>, @builtin(front_facing) gl_FrontFacing: bool) -> FragmentOutput {
    v_WorldPosition_1 = v_WorldPosition;
    v_WorldNormal_1 = v_WorldNormal;
    v_Uv_1 = v_Uv;
    v_WorldTangent_1 = v_WorldTangent;
    gl_FrontFacing_1 = gl_FrontFacing;
    main_1();
    let _e11 = o_Target;
    return FragmentOutput(_e11);
}
