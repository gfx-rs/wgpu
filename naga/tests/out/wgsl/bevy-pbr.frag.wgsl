struct PointLight {
    pos: vec4<f32>,
    color: vec4<f32>,
    lightParams: vec4<f32>,
}

struct DirectionalLight {
    direction: vec4<f32>,
    color: vec4<f32>,
}

struct CameraViewProj {
    ViewProj: mat4x4<f32>,
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
@group(0) @binding(0) 
var<uniform> global: CameraViewProj;
@group(0) @binding(1) 
var<uniform> global_1: CameraPosition;
@group(1) @binding(0) 
var<uniform> global_2: Lights;
@group(3) @binding(0) 
var<uniform> global_3: StandardMaterial_base_color;
@group(3) @binding(1) 
var StandardMaterial_base_color_texture: texture_2d<f32>;
@group(3) @binding(2) 
var StandardMaterial_base_color_texture_sampler: sampler;
@group(3) @binding(3) 
var<uniform> global_4: StandardMaterial_roughness;
@group(3) @binding(4) 
var<uniform> global_5: StandardMaterial_metallic;
@group(3) @binding(5) 
var StandardMaterial_metallic_roughness_texture: texture_2d<f32>;
@group(3) @binding(6) 
var StandardMaterial_metallic_roughness_texture_sampler: sampler;
@group(3) @binding(7) 
var<uniform> global_6: StandardMaterial_reflectance;
@group(3) @binding(8) 
var StandardMaterial_normal_map: texture_2d<f32>;
@group(3) @binding(9) 
var StandardMaterial_normal_map_sampler: sampler;
@group(3) @binding(10) 
var StandardMaterial_occlusion_texture: texture_2d<f32>;
@group(3) @binding(11) 
var StandardMaterial_occlusion_texture_sampler: sampler;
@group(3) @binding(12) 
var<uniform> global_7: StandardMaterial_emissive;
@group(3) @binding(13) 
var StandardMaterial_emissive_texture: texture_2d<f32>;
@group(3) @binding(14) 
var StandardMaterial_emissive_texture_sampler: sampler;
var<private> gl_FrontFacing_1: bool;

fn pow5_(x: f32) -> f32 {
    var x_1: f32;
    var x2_: f32;

    x_1 = x;
    let _e42 = x_1;
    let _e43 = x_1;
    x2_ = (_e42 * _e43);
    let _e46 = x2_;
    let _e47 = x2_;
    let _e49 = x_1;
    return ((_e46 * _e47) * _e49);
}

fn getDistanceAttenuation(distanceSquare: f32, inverseRangeSquared: f32) -> f32 {
    var distanceSquare_1: f32;
    var inverseRangeSquared_1: f32;
    var factor: f32;
    var smoothFactor: f32;
    var attenuation: f32;

    distanceSquare_1 = distanceSquare;
    inverseRangeSquared_1 = inverseRangeSquared;
    let _e44 = distanceSquare_1;
    let _e45 = inverseRangeSquared_1;
    factor = (_e44 * _e45);
    let _e49 = factor;
    let _e50 = factor;
    smoothFactor = clamp((1f - (_e49 * _e50)), 0f, 1f);
    let _e57 = smoothFactor;
    let _e58 = smoothFactor;
    attenuation = (_e57 * _e58);
    let _e61 = attenuation;
    let _e64 = distanceSquare_1;
    return ((_e61 * 1f) / max(_e64, 0.001f));
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
    let _e46 = NoH_1;
    let _e47 = NoH_1;
    oneMinusNoHSquared = (1f - (_e46 * _e47));
    let _e51 = NoH_1;
    let _e52 = roughness_1;
    a = (_e51 * _e52);
    let _e55 = roughness_1;
    let _e56 = oneMinusNoHSquared;
    let _e57 = a;
    let _e58 = a;
    k = (_e55 / (_e56 + (_e57 * _e58)));
    let _e63 = k;
    let _e64 = k;
    d = ((_e63 * _e64) * 0.31830987f);
    let _e71 = d;
    return _e71;
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
    let _e46 = roughness_3;
    let _e47 = roughness_3;
    a2_ = (_e46 * _e47);
    let _e50 = NoL_1;
    let _e51 = NoV_1;
    let _e52 = a2_;
    let _e53 = NoV_1;
    let _e56 = NoV_1;
    let _e58 = a2_;
    lambdaV = (_e50 * sqrt((((_e51 - (_e52 * _e53)) * _e56) + _e58)));
    let _e63 = NoV_1;
    let _e64 = NoL_1;
    let _e65 = a2_;
    let _e66 = NoL_1;
    let _e69 = NoL_1;
    let _e71 = a2_;
    lambdaL = (_e63 * sqrt((((_e64 - (_e65 * _e66)) * _e69) + _e71)));
    let _e77 = lambdaV;
    let _e78 = lambdaL;
    v = (0.5f / (_e77 + _e78));
    let _e82 = v;
    return _e82;
}

fn F_Schlick(f0_: vec3<f32>, f90_: f32, VoH: f32) -> vec3<f32> {
    var f90_1: f32;
    var VoH_1: f32;

    f90_1 = f90_;
    VoH_1 = VoH;
    let _e45 = f90_1;
    let _e49 = VoH_1;
    let _e51 = pow5_((1f - _e49));
    return (f0_ + ((vec3(_e45) - f0_) * _e51));
}

fn F_Schlick_1(f0_1: f32, f90_2: f32, VoH_2: f32) -> f32 {
    var f0_2: f32;
    var f90_3: f32;
    var VoH_3: f32;

    f0_2 = f0_1;
    f90_3 = f90_2;
    VoH_3 = VoH_2;
    let _e46 = f0_2;
    let _e47 = f90_3;
    let _e48 = f0_2;
    let _e51 = VoH_3;
    let _e53 = pow5_((1f - _e51));
    return (_e46 + ((_e47 - _e48) * _e53));
}

fn fresnel(f0_3: vec3<f32>, LoH: f32) -> vec3<f32> {
    var f0_4: vec3<f32>;
    var LoH_1: f32;
    var f90_4: f32;

    f0_4 = f0_3;
    LoH_1 = LoH;
    let _e44 = f0_4;
    f90_4 = clamp(dot(_e44, vec3(16.5f)), 0f, 1f);
    let _e54 = f0_4;
    let _e55 = f90_4;
    let _e56 = LoH_1;
    let _e57 = F_Schlick(_e54, _e55, _e56);
    return _e57;
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
    let _e55 = roughness_5;
    let _e56 = NoH_3;
    let _e57 = D_GGX(_e55, _e56, h_1);
    D = _e57;
    let _e59 = roughness_5;
    let _e60 = NoV_3;
    let _e61 = NoL_3;
    let _e62 = V_SmithGGXCorrelated(_e59, _e60, _e61);
    V = _e62;
    let _e64 = f0_6;
    let _e65 = LoH_3;
    let _e66 = fresnel(_e64, _e65);
    F = _e66;
    let _e68 = specularIntensity_1;
    let _e69 = D;
    let _e71 = V;
    let _e73 = F;
    return (((_e68 * _e69) * _e71) * _e73);
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
    let _e50 = roughness_7;
    let _e52 = LoH_5;
    let _e54 = LoH_5;
    f90_5 = (0.5f + (((2f * _e50) * _e52) * _e54));
    let _e59 = f90_5;
    let _e60 = NoL_5;
    let _e61 = F_Schlick_1(1f, _e59, _e60);
    lightScatter = _e61;
    let _e64 = f90_5;
    let _e65 = NoV_5;
    let _e66 = F_Schlick_1(1f, _e64, _e65);
    viewScatter = _e66;
    let _e68 = lightScatter;
    let _e69 = viewScatter;
    return ((_e68 * _e69) * 0.31830987f);
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
    let _e62 = perceptual_roughness_1;
    let _e64 = c0_;
    let _e66 = c1_;
    r = ((vec4(_e62) * _e64) + _e66);
    let _e69 = r;
    let _e71 = r;
    let _e76 = NoV_7;
    let _e80 = r;
    let _e83 = r;
    a004_ = ((min((_e69.x * _e71.x), exp2((-9.28f * _e76))) * _e80.x) + _e83.y);
    let _e91 = a004_;
    let _e94 = r;
    AB = ((vec2<f32>(-1.04f, 1.04f) * vec2(_e91)) + _e94.zw);
    let _e98 = f0_8;
    let _e99 = AB;
    let _e103 = AB;
    return ((_e98 * vec3(_e99.x)) + vec3(_e103.y));
}

fn perceptualRoughnessToRoughness(perceptualRoughness: f32) -> f32 {
    var perceptualRoughness_1: f32;
    var clampedPerceptualRoughness: f32;

    perceptualRoughness_1 = perceptualRoughness;
    let _e42 = perceptualRoughness_1;
    clampedPerceptualRoughness = clamp(_e42, 0.089f, 1f);
    let _e47 = clampedPerceptualRoughness;
    let _e48 = clampedPerceptualRoughness;
    return (_e47 * _e48);
}

fn reinhard(color: vec3<f32>) -> vec3<f32> {
    var color_1: vec3<f32>;

    color_1 = color;
    let _e42 = color_1;
    let _e45 = color_1;
    return (_e42 / (vec3(1f) + _e45));
}

fn reinhard_extended(color_2: vec3<f32>, max_white: f32) -> vec3<f32> {
    var color_3: vec3<f32>;
    var max_white_1: f32;
    var numerator: vec3<f32>;

    color_3 = color_2;
    max_white_1 = max_white;
    let _e44 = color_3;
    let _e47 = color_3;
    let _e48 = max_white_1;
    let _e49 = max_white_1;
    numerator = (_e44 * (vec3(1f) + (_e47 / vec3((_e48 * _e49)))));
    let _e56 = numerator;
    let _e59 = color_3;
    return (_e56 / (vec3(1f) + _e59));
}

fn luminance(v_1: vec3<f32>) -> f32 {
    var v_2: vec3<f32>;

    v_2 = v_1;
    let _e42 = v_2;
    return dot(_e42, vec3<f32>(0.2126f, 0.7152f, 0.0722f));
}

fn change_luminance(c_in: vec3<f32>, l_out: f32) -> vec3<f32> {
    var c_in_1: vec3<f32>;
    var l_out_1: f32;
    var l_in: f32;

    c_in_1 = c_in;
    l_out_1 = l_out;
    let _e44 = c_in_1;
    let _e45 = luminance(_e44);
    l_in = _e45;
    let _e47 = c_in_1;
    let _e48 = l_out_1;
    let _e49 = l_in;
    return (_e47 * (_e48 / _e49));
}

fn reinhard_luminance(color_4: vec3<f32>) -> vec3<f32> {
    var color_5: vec3<f32>;
    var l_old: f32;
    var l_new: f32;

    color_5 = color_4;
    let _e42 = color_5;
    let _e43 = luminance(_e42);
    l_old = _e43;
    let _e45 = l_old;
    let _e47 = l_old;
    l_new = (_e45 / (1f + _e47));
    let _e51 = color_5;
    let _e52 = l_new;
    let _e53 = change_luminance(_e51, _e52);
    return _e53;
}

fn reinhard_extended_luminance(color_6: vec3<f32>, max_white_l: f32) -> vec3<f32> {
    var color_7: vec3<f32>;
    var max_white_l_1: f32;
    var l_old_1: f32;
    var numerator_1: f32;
    var l_new_1: f32;

    color_7 = color_6;
    max_white_l_1 = max_white_l;
    let _e44 = color_7;
    let _e45 = luminance(_e44);
    l_old_1 = _e45;
    let _e47 = l_old_1;
    let _e49 = l_old_1;
    let _e50 = max_white_l_1;
    let _e51 = max_white_l_1;
    numerator_1 = (_e47 * (1f + (_e49 / (_e50 * _e51))));
    let _e57 = numerator_1;
    let _e59 = l_old_1;
    l_new_1 = (_e57 / (1f + _e59));
    let _e63 = color_7;
    let _e64 = l_new_1;
    let _e65 = change_luminance(_e63, _e64);
    return _e65;
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
    let _e56 = light_1;
    let _e59 = v_WorldPosition_1;
    light_to_frag = (_e56.pos.xyz - _e59.xyz);
    let _e63 = light_to_frag;
    let _e64 = light_to_frag;
    distance_square = dot(_e63, _e64);
    let _e67 = distance_square;
    let _e68 = light_1;
    let _e71 = getDistanceAttenuation(_e67, _e68.lightParams.x);
    rangeAttenuation = _e71;
    let _e73 = roughness_9;
    a_1 = _e73;
    let _e75 = light_1;
    radius = _e75.lightParams.y;
    let _e79 = light_to_frag;
    let _e80 = R_1;
    let _e82 = R_1;
    let _e84 = light_to_frag;
    centerToRay = ((dot(_e79, _e80) * _e82) - _e84);
    let _e87 = light_to_frag;
    let _e88 = centerToRay;
    let _e89 = radius;
    let _e90 = centerToRay;
    let _e91 = centerToRay;
    closestPoint = (_e87 + (_e88 * clamp((_e89 * inverseSqrt(dot(_e90, _e91))), 0f, 1f)));
    let _e101 = closestPoint;
    let _e102 = closestPoint;
    LspecLengthInverse = inverseSqrt(dot(_e101, _e102));
    let _e106 = a_1;
    let _e107 = a_1;
    let _e108 = radius;
    let _e111 = LspecLengthInverse;
    normalizationFactor = (_e106 / clamp((_e107 + ((_e108 * 0.5f) * _e111)), 0f, 1f));
    let _e119 = normalizationFactor;
    let _e120 = normalizationFactor;
    specularIntensity_2 = (_e119 * _e120);
    let _e123 = closestPoint;
    let _e124 = LspecLengthInverse;
    L = (_e123 * _e124);
    let _e127 = L;
    let _e128 = V_2;
    H = normalize((_e127 + _e128));
    let _e132 = N_1;
    let _e133 = L;
    NoL_6 = clamp(dot(_e132, _e133), 0f, 1f);
    let _e139 = N_1;
    let _e140 = H;
    NoH_4 = clamp(dot(_e139, _e140), 0f, 1f);
    let _e146 = L;
    let _e147 = H;
    LoH_6 = clamp(dot(_e146, _e147), 0f, 1f);
    let _e153 = F0_1;
    let _e154 = roughness_9;
    let _e155 = H;
    let _e156 = NdotV_1;
    let _e157 = NoL_6;
    let _e158 = NoH_4;
    let _e159 = LoH_6;
    let _e160 = specularIntensity_2;
    let _e161 = specular(_e153, _e154, _e155, _e156, _e157, _e158, _e159, _e160);
    specular_1 = _e161;
    let _e163 = light_to_frag;
    L = normalize(_e163);
    let _e165 = L;
    let _e166 = V_2;
    H = normalize((_e165 + _e166));
    let _e169 = N_1;
    let _e170 = L;
    NoL_6 = clamp(dot(_e169, _e170), 0f, 1f);
    let _e175 = N_1;
    let _e176 = H;
    NoH_4 = clamp(dot(_e175, _e176), 0f, 1f);
    let _e181 = L;
    let _e182 = H;
    LoH_6 = clamp(dot(_e181, _e182), 0f, 1f);
    let _e187 = diffuseColor_1;
    let _e188 = roughness_9;
    let _e189 = NdotV_1;
    let _e190 = NoL_6;
    let _e191 = LoH_6;
    let _e192 = Fd_Burley(_e188, _e189, _e190, _e191);
    diffuse = (_e187 * _e192);
    let _e195 = diffuse;
    let _e196 = specular_1;
    let _e198 = light_1;
    let _e202 = rangeAttenuation;
    let _e203 = NoL_6;
    return (((_e195 + _e196) * _e198.color.xyz) * (_e202 * _e203));
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
    let _e56 = light_3;
    incident_light = _e56.direction.xyz;
    let _e60 = incident_light;
    let _e61 = view_1;
    half_vector = normalize((_e60 + _e61));
    let _e65 = normal_1;
    let _e66 = incident_light;
    NoL_7 = clamp(dot(_e65, _e66), 0f, 1f);
    let _e72 = normal_1;
    let _e73 = half_vector;
    NoH_5 = clamp(dot(_e72, _e73), 0f, 1f);
    let _e79 = incident_light;
    let _e80 = half_vector;
    LoH_7 = clamp(dot(_e79, _e80), 0f, 1f);
    let _e86 = diffuseColor_3;
    let _e87 = roughness_11;
    let _e88 = NdotV_3;
    let _e89 = NoL_7;
    let _e90 = LoH_7;
    let _e91 = Fd_Burley(_e87, _e88, _e89, _e90);
    diffuse_1 = (_e86 * _e91);
    let _e96 = F0_3;
    let _e97 = roughness_11;
    let _e98 = half_vector;
    let _e99 = NdotV_3;
    let _e100 = NoL_7;
    let _e101 = NoH_5;
    let _e102 = LoH_7;
    let _e103 = specularIntensity_3;
    let _e104 = specular(_e96, _e97, _e98, _e99, _e100, _e101, _e102, _e103);
    specular_2 = _e104;
    let _e106 = specular_2;
    let _e107 = diffuse_1;
    let _e109 = light_3;
    let _e113 = NoL_7;
    return (((_e106 + _e107) * _e109.color.xyz) * _e113);
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

    let _e40 = global_3.base_color;
    output_color = _e40;
    let _e42 = output_color;
    let _e43 = v_Uv_1;
    let _e44 = textureSample(StandardMaterial_base_color_texture, StandardMaterial_base_color_texture_sampler, _e43);
    output_color = (_e42 * _e44);
    let _e46 = v_Uv_1;
    let _e47 = textureSample(StandardMaterial_metallic_roughness_texture, StandardMaterial_metallic_roughness_texture_sampler, _e46);
    metallic_roughness = _e47;
    let _e49 = global_5.metallic;
    let _e50 = metallic_roughness;
    metallic = (_e49 * _e50.z);
    let _e54 = global_4.perceptual_roughness;
    let _e55 = metallic_roughness;
    perceptual_roughness_2 = (_e54 * _e55.y);
    let _e59 = perceptual_roughness_2;
    let _e60 = perceptualRoughnessToRoughness(_e59);
    roughness_12 = _e60;
    let _e62 = v_WorldNormal_1;
    N_2 = normalize(_e62);
    let _e65 = v_WorldTangent_1;
    T = normalize(_e65.xyz);
    let _e69 = N_2;
    let _e70 = T;
    let _e72 = v_WorldTangent_1;
    B = (cross(_e69, _e70) * _e72.w);
    let _e77 = gl_FrontFacing_1;
    if _e77 {
        let _e78 = N_2;
        local = _e78;
    } else {
        let _e79 = N_2;
        local = -(_e79);
    }
    let _e82 = local;
    N_2 = _e82;
    let _e83 = gl_FrontFacing_1;
    if _e83 {
        let _e84 = T;
        local_1 = _e84;
    } else {
        let _e85 = T;
        local_1 = -(_e85);
    }
    let _e88 = local_1;
    T = _e88;
    let _e89 = gl_FrontFacing_1;
    if _e89 {
        let _e90 = B;
        local_2 = _e90;
    } else {
        let _e91 = B;
        local_2 = -(_e91);
    }
    let _e94 = local_2;
    B = _e94;
    let _e95 = T;
    let _e96 = B;
    let _e97 = N_2;
    TBN = mat3x3<f32>(vec3<f32>(_e95.x, _e95.y, _e95.z), vec3<f32>(_e96.x, _e96.y, _e96.z), vec3<f32>(_e97.x, _e97.y, _e97.z));
    let _e112 = TBN;
    let _e113 = v_Uv_1;
    let _e114 = textureSample(StandardMaterial_normal_map, StandardMaterial_normal_map_sampler, _e113);
    N_2 = (_e112 * normalize(((_e114.xyz * 2f) - vec3(1f))));
    let _e123 = v_Uv_1;
    let _e124 = textureSample(StandardMaterial_occlusion_texture, StandardMaterial_occlusion_texture_sampler, _e123);
    occlusion = _e124.x;
    let _e127 = global_7.emissive;
    emissive = _e127;
    let _e129 = emissive;
    let _e131 = emissive;
    let _e133 = v_Uv_1;
    let _e134 = textureSample(StandardMaterial_emissive_texture, StandardMaterial_emissive_texture_sampler, _e133);
    let _e136 = (_e131.xyz * _e134.xyz);
    emissive.x = _e136.x;
    emissive.y = _e136.y;
    emissive.z = _e136.z;
    let _e143 = global_1.CameraPos;
    let _e145 = v_WorldPosition_1;
    V_3 = normalize((_e143.xyz - _e145.xyz));
    let _e150 = N_2;
    let _e151 = V_3;
    NdotV_4 = max(dot(_e150, _e151), 0.001f);
    let _e157 = global_6.reflectance;
    let _e159 = global_6.reflectance;
    let _e162 = metallic;
    let _e166 = output_color;
    let _e168 = metallic;
    F0_4 = (vec3((((0.16f * _e157) * _e159) * (1f - _e162))) + (_e166.xyz * vec3(_e168)));
    let _e173 = output_color;
    let _e176 = metallic;
    diffuseColor_4 = (_e173.xyz * vec3((1f - _e176)));
    let _e181 = V_3;
    let _e183 = N_2;
    R_4 = reflect(-(_e181), _e183);
    loop {
        let _e191 = i;
        let _e192 = global_2.NumLights;
        let _e196 = i;
        if !(((_e191 < i32(_e192.x)) && (_e196 < MAX_POINT_LIGHTS))) {
            break;
        }
        {
            let _e203 = light_accum;
            let _e204 = i;
            let _e206 = global_2.PointLights[_e204];
            let _e207 = roughness_12;
            let _e208 = NdotV_4;
            let _e209 = N_2;
            let _e210 = V_3;
            let _e211 = R_4;
            let _e212 = F0_4;
            let _e213 = diffuseColor_4;
            let _e214 = point_light(_e206, _e207, _e208, _e209, _e210, _e211, _e212, _e213);
            light_accum = (_e203 + _e214);
        }
        continuing {
            let _e200 = i;
            i = (_e200 + 1i);
        }
    }
    loop {
        let _e218 = i_1;
        let _e219 = global_2.NumLights;
        let _e223 = i_1;
        if !(((_e218 < i32(_e219.y)) && (_e223 < MAX_DIRECTIONAL_LIGHTS))) {
            break;
        }
        {
            let _e230 = light_accum;
            let _e231 = i_1;
            let _e233 = global_2.DirectionalLights[_e231];
            let _e234 = roughness_12;
            let _e235 = NdotV_4;
            let _e236 = N_2;
            let _e237 = V_3;
            let _e238 = R_4;
            let _e239 = F0_4;
            let _e240 = diffuseColor_4;
            let _e241 = dir_light(_e233, _e234, _e235, _e236, _e237, _e238, _e239, _e240);
            light_accum = (_e230 + _e241);
        }
        continuing {
            let _e227 = i_1;
            i_1 = (_e227 + 1i);
        }
    }
    let _e243 = diffuseColor_4;
    let _e245 = NdotV_4;
    let _e246 = EnvBRDFApprox(_e243, 1f, _e245);
    diffuse_ambient = _e246;
    let _e248 = F0_4;
    let _e249 = perceptual_roughness_2;
    let _e250 = NdotV_4;
    let _e251 = EnvBRDFApprox(_e248, _e249, _e250);
    specular_ambient = _e251;
    let _e253 = output_color;
    let _e255 = light_accum;
    output_color.x = _e255.x;
    output_color.y = _e255.y;
    output_color.z = _e255.z;
    let _e262 = output_color;
    let _e264 = output_color;
    let _e266 = diffuse_ambient;
    let _e267 = specular_ambient;
    let _e269 = global_2.AmbientColor;
    let _e272 = occlusion;
    let _e274 = (_e264.xyz + (((_e266 + _e267) * _e269.xyz) * _e272));
    output_color.x = _e274.x;
    output_color.y = _e274.y;
    output_color.z = _e274.z;
    let _e281 = output_color;
    let _e283 = output_color;
    let _e285 = emissive;
    let _e287 = output_color;
    let _e290 = (_e283.xyz + (_e285.xyz * _e287.w));
    output_color.x = _e290.x;
    output_color.y = _e290.y;
    output_color.z = _e290.z;
    let _e297 = output_color;
    let _e299 = output_color;
    let _e301 = reinhard_luminance(_e299.xyz);
    output_color.x = _e301.x;
    output_color.y = _e301.y;
    output_color.z = _e301.z;
    let _e308 = output_color;
    o_Target = _e308;
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
    let _e69 = o_Target;
    return FragmentOutput(_e69);
}
