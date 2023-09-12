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

const MAX_POINT_LIGHTS: i32 = 10;
const MAX_DIRECTIONAL_LIGHTS: i32 = 1;
const PI: f32 = 3.1415927;

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
var<private> gl_FrontFacing: bool;

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
    let _e56 = factor;
    let _e57 = factor;
    smoothFactor = clamp((1.0 - (_e56 * _e57)), 0.0, 1.0);
    let _e64 = smoothFactor;
    let _e65 = smoothFactor;
    attenuation = (_e64 * _e65);
    let _e68 = attenuation;
    let _e73 = distanceSquare_1;
    return ((_e68 * 1.0) / max(_e73, 0.001));
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
    oneMinusNoHSquared = (1.0 - (_e46 * _e47));
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
    d = ((_e63 * _e64) * (1.0 / PI));
    let _e70 = d;
    return _e70;
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
    let _e60 = NoV_1;
    let _e61 = a2_;
    let _e62 = NoV_1;
    let _e65 = NoV_1;
    let _e67 = a2_;
    lambdaV = (_e50 * sqrt((((_e60 - (_e61 * _e62)) * _e65) + _e67)));
    let _e72 = NoV_1;
    let _e73 = NoL_1;
    let _e74 = a2_;
    let _e75 = NoL_1;
    let _e78 = NoL_1;
    let _e80 = a2_;
    let _e82 = NoL_1;
    let _e83 = a2_;
    let _e84 = NoL_1;
    let _e87 = NoL_1;
    let _e89 = a2_;
    lambdaL = (_e72 * sqrt((((_e82 - (_e83 * _e84)) * _e87) + _e89)));
    let _e95 = lambdaV;
    let _e96 = lambdaL;
    v = (0.5 / (_e95 + _e96));
    let _e100 = v;
    return _e100;
}

fn F_Schlick(f0_: vec3<f32>, f90_: f32, VoH: f32) -> vec3<f32> {
    var f90_1: f32;
    var VoH_1: f32;

    f90_1 = f90_;
    VoH_1 = VoH;
    let _e45 = f90_1;
    let _e49 = VoH_1;
    let _e52 = VoH_1;
    let _e54 = pow5_((1.0 - _e52));
    return (f0_ + ((vec3(_e45) - f0_) * _e54));
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
    let _e54 = VoH_3;
    let _e56 = pow5_((1.0 - _e54));
    return (_e46 + ((_e47 - _e48) * _e56));
}

fn fresnel(f0_3: vec3<f32>, LoH: f32) -> vec3<f32> {
    var f0_4: vec3<f32>;
    var LoH_1: f32;
    var f90_4: f32;

    f0_4 = f0_3;
    LoH_1 = LoH;
    let _e49 = f0_4;
    let _e62 = f0_4;
    f90_4 = clamp(dot(_e62, vec3((50.0 * 0.33))), 0.0, 1.0);
    let _e75 = f0_4;
    let _e76 = f90_4;
    let _e77 = LoH_1;
    let _e78 = F_Schlick(_e75, _e76, _e77);
    return _e78;
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
    let _e57 = roughness_5;
    let _e58 = NoH_3;
    let _e59 = D_GGX(_e57, _e58, h_1);
    D = _e59;
    let _e64 = roughness_5;
    let _e65 = NoV_3;
    let _e66 = NoL_3;
    let _e67 = V_SmithGGXCorrelated(_e64, _e65, _e66);
    V = _e67;
    let _e71 = f0_6;
    let _e72 = LoH_3;
    let _e73 = fresnel(_e71, _e72);
    F = _e73;
    let _e75 = specularIntensity_1;
    let _e76 = D;
    let _e78 = V;
    let _e80 = F;
    return (((_e75 * _e76) * _e78) * _e80);
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
    f90_5 = (0.5 + (((2.0 * _e50) * _e52) * _e54));
    let _e62 = f90_5;
    let _e63 = NoL_5;
    let _e64 = F_Schlick_1(1.0, _e62, _e63);
    lightScatter = _e64;
    let _e70 = f90_5;
    let _e71 = NoV_5;
    let _e72 = F_Schlick_1(1.0, _e70, _e71);
    viewScatter = _e72;
    let _e74 = lightScatter;
    let _e75 = viewScatter;
    return ((_e74 * _e75) * (1.0 / PI));
}

fn EnvBRDFApprox(f0_7: vec3<f32>, perceptual_roughness: f32, NoV_6: f32) -> vec3<f32> {
    var f0_8: vec3<f32>;
    var perceptual_roughness_1: f32;
    var NoV_7: f32;
    var c0_: vec4<f32>;
    var c1_: vec4<f32>;
    var r: vec4<f32>;
    var a004_: f32;
    var AB: vec2<f32>;

    f0_8 = f0_7;
    perceptual_roughness_1 = perceptual_roughness;
    NoV_7 = NoV_6;
    c0_ = vec4<f32>(-(1.0), -(0.0275), -(0.572), 0.022);
    c1_ = vec4<f32>(1.0, 0.0425, 1.04, -(0.04));
    let _e62 = perceptual_roughness_1;
    let _e64 = c0_;
    let _e66 = c1_;
    r = ((vec4(_e62) * _e64) + _e66);
    let _e69 = r;
    let _e71 = r;
    let _e76 = NoV_7;
    let _e80 = NoV_7;
    let _e83 = r;
    let _e85 = r;
    let _e90 = NoV_7;
    let _e94 = NoV_7;
    let _e98 = r;
    let _e101 = r;
    a004_ = ((min((_e83.x * _e85.x), exp2((-(9.28) * _e94))) * _e98.x) + _e101.y);
    let _e109 = a004_;
    let _e112 = r;
    AB = ((vec2<f32>(-(1.04), 1.04) * vec2(_e109)) + _e112.zw);
    let _e116 = f0_8;
    let _e117 = AB;
    let _e121 = AB;
    return ((_e116 * vec3(_e117.x)) + vec3(_e121.y));
}

fn perceptualRoughnessToRoughness(perceptualRoughness: f32) -> f32 {
    var perceptualRoughness_1: f32;
    var clampedPerceptualRoughness: f32;

    perceptualRoughness_1 = perceptualRoughness;
    let _e45 = perceptualRoughness_1;
    clampedPerceptualRoughness = clamp(_e45, 0.089, 1.0);
    let _e50 = clampedPerceptualRoughness;
    let _e51 = clampedPerceptualRoughness;
    return (_e50 * _e51);
}

fn reinhard(color: vec3<f32>) -> vec3<f32> {
    var color_1: vec3<f32>;

    color_1 = color;
    let _e42 = color_1;
    let _e45 = color_1;
    return (_e42 / (vec3(1.0) + _e45));
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
    numerator = (_e44 * (vec3(1.0) + (_e47 / vec3((_e48 * _e49)))));
    let _e56 = numerator;
    let _e59 = color_3;
    return (_e56 / (vec3(1.0) + _e59));
}

fn luminance(v_1: vec3<f32>) -> f32 {
    var v_2: vec3<f32>;

    v_2 = v_1;
    let _e47 = v_2;
    return dot(_e47, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn change_luminance(c_in: vec3<f32>, l_out: f32) -> vec3<f32> {
    var c_in_1: vec3<f32>;
    var l_out_1: f32;
    var l_in: f32;

    c_in_1 = c_in;
    l_out_1 = l_out;
    let _e45 = c_in_1;
    let _e46 = luminance(_e45);
    l_in = _e46;
    let _e48 = c_in_1;
    let _e49 = l_out_1;
    let _e50 = l_in;
    return (_e48 * (_e49 / _e50));
}

fn reinhard_luminance(color_4: vec3<f32>) -> vec3<f32> {
    var color_5: vec3<f32>;
    var l_old: f32;
    var l_new: f32;

    color_5 = color_4;
    let _e43 = color_5;
    let _e44 = luminance(_e43);
    l_old = _e44;
    let _e46 = l_old;
    let _e48 = l_old;
    l_new = (_e46 / (1.0 + _e48));
    let _e54 = color_5;
    let _e55 = l_new;
    let _e56 = change_luminance(_e54, _e55);
    return _e56;
}

fn reinhard_extended_luminance(color_6: vec3<f32>, max_white_l: f32) -> vec3<f32> {
    var color_7: vec3<f32>;
    var max_white_l_1: f32;
    var l_old_1: f32;
    var numerator_1: f32;
    var l_new_1: f32;

    color_7 = color_6;
    max_white_l_1 = max_white_l;
    let _e45 = color_7;
    let _e46 = luminance(_e45);
    l_old_1 = _e46;
    let _e48 = l_old_1;
    let _e50 = l_old_1;
    let _e51 = max_white_l_1;
    let _e52 = max_white_l_1;
    numerator_1 = (_e48 * (1.0 + (_e50 / (_e51 * _e52))));
    let _e58 = numerator_1;
    let _e60 = l_old_1;
    l_new_1 = (_e58 / (1.0 + _e60));
    let _e66 = color_7;
    let _e67 = l_new_1;
    let _e68 = change_luminance(_e66, _e67);
    return _e68;
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
    let _e65 = light_to_frag;
    let _e66 = light_to_frag;
    distance_square = dot(_e65, _e66);
    let _e70 = light_1;
    let _e73 = distance_square;
    let _e74 = light_1;
    let _e77 = getDistanceAttenuation(_e73, _e74.lightParams.x);
    rangeAttenuation = _e77;
    let _e79 = roughness_9;
    a_1 = _e79;
    let _e81 = light_1;
    radius = _e81.lightParams.y;
    let _e87 = light_to_frag;
    let _e88 = R_1;
    let _e90 = R_1;
    let _e92 = light_to_frag;
    centerToRay = ((dot(_e87, _e88) * _e90) - _e92);
    let _e95 = light_to_frag;
    let _e96 = centerToRay;
    let _e97 = radius;
    let _e100 = centerToRay;
    let _e101 = centerToRay;
    let _e105 = centerToRay;
    let _e106 = centerToRay;
    let _e112 = radius;
    let _e115 = centerToRay;
    let _e116 = centerToRay;
    let _e120 = centerToRay;
    let _e121 = centerToRay;
    closestPoint = (_e95 + (_e96 * clamp((_e112 * inverseSqrt(dot(_e120, _e121))), 0.0, 1.0)));
    let _e133 = closestPoint;
    let _e134 = closestPoint;
    let _e138 = closestPoint;
    let _e139 = closestPoint;
    LspecLengthInverse = inverseSqrt(dot(_e138, _e139));
    let _e143 = a_1;
    let _e144 = a_1;
    let _e145 = radius;
    let _e148 = LspecLengthInverse;
    let _e153 = a_1;
    let _e154 = radius;
    let _e157 = LspecLengthInverse;
    normalizationFactor = (_e143 / clamp((_e153 + ((_e154 * 0.5) * _e157)), 0.0, 1.0));
    let _e165 = normalizationFactor;
    let _e166 = normalizationFactor;
    specularIntensity_2 = (_e165 * _e166);
    let _e169 = closestPoint;
    let _e170 = LspecLengthInverse;
    L = (_e169 * _e170);
    let _e173 = L;
    let _e174 = V_2;
    let _e176 = L;
    let _e177 = V_2;
    H = normalize((_e176 + _e177));
    let _e183 = N_1;
    let _e184 = L;
    let _e190 = N_1;
    let _e191 = L;
    NoL_6 = clamp(dot(_e190, _e191), 0.0, 1.0);
    let _e199 = N_1;
    let _e200 = H;
    let _e206 = N_1;
    let _e207 = H;
    NoH_4 = clamp(dot(_e206, _e207), 0.0, 1.0);
    let _e215 = L;
    let _e216 = H;
    let _e222 = L;
    let _e223 = H;
    LoH_6 = clamp(dot(_e222, _e223), 0.0, 1.0);
    let _e237 = F0_1;
    let _e238 = roughness_9;
    let _e239 = H;
    let _e240 = NdotV_1;
    let _e241 = NoL_6;
    let _e242 = NoH_4;
    let _e243 = LoH_6;
    let _e244 = specularIntensity_2;
    let _e245 = specular(_e237, _e238, _e239, _e240, _e241, _e242, _e243, _e244);
    specular_1 = _e245;
    let _e248 = light_to_frag;
    L = normalize(_e248);
    let _e250 = L;
    let _e251 = V_2;
    let _e253 = L;
    let _e254 = V_2;
    H = normalize((_e253 + _e254));
    let _e259 = N_1;
    let _e260 = L;
    let _e266 = N_1;
    let _e267 = L;
    NoL_6 = clamp(dot(_e266, _e267), 0.0, 1.0);
    let _e274 = N_1;
    let _e275 = H;
    let _e281 = N_1;
    let _e282 = H;
    NoH_4 = clamp(dot(_e281, _e282), 0.0, 1.0);
    let _e289 = L;
    let _e290 = H;
    let _e296 = L;
    let _e297 = H;
    LoH_6 = clamp(dot(_e296, _e297), 0.0, 1.0);
    let _e302 = diffuseColor_1;
    let _e307 = roughness_9;
    let _e308 = NdotV_1;
    let _e309 = NoL_6;
    let _e310 = LoH_6;
    let _e311 = Fd_Burley(_e307, _e308, _e309, _e310);
    diffuse = (_e302 * _e311);
    let _e314 = diffuse;
    let _e315 = specular_1;
    let _e317 = light_1;
    let _e321 = rangeAttenuation;
    let _e322 = NoL_6;
    return (((_e314 + _e315) * _e317.color.xyz) * (_e321 * _e322));
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
    var specularIntensity_3: f32;
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
    let _e63 = incident_light;
    let _e64 = view_1;
    half_vector = normalize((_e63 + _e64));
    let _e70 = normal_1;
    let _e71 = incident_light;
    let _e77 = normal_1;
    let _e78 = incident_light;
    NoL_7 = clamp(dot(_e77, _e78), 0.0, 1.0);
    let _e86 = normal_1;
    let _e87 = half_vector;
    let _e93 = normal_1;
    let _e94 = half_vector;
    NoH_5 = clamp(dot(_e93, _e94), 0.0, 1.0);
    let _e102 = incident_light;
    let _e103 = half_vector;
    let _e109 = incident_light;
    let _e110 = half_vector;
    LoH_7 = clamp(dot(_e109, _e110), 0.0, 1.0);
    let _e116 = diffuseColor_3;
    let _e121 = roughness_11;
    let _e122 = NdotV_3;
    let _e123 = NoL_7;
    let _e124 = LoH_7;
    let _e125 = Fd_Burley(_e121, _e122, _e123, _e124);
    diffuse_1 = (_e116 * _e125);
    specularIntensity_3 = 1.0;
    let _e138 = F0_3;
    let _e139 = roughness_11;
    let _e140 = half_vector;
    let _e141 = NdotV_3;
    let _e142 = NoL_7;
    let _e143 = NoH_5;
    let _e144 = LoH_7;
    let _e145 = specularIntensity_3;
    let _e146 = specular(_e138, _e139, _e140, _e141, _e142, _e143, _e144, _e145);
    specular_2 = _e146;
    let _e148 = specular_2;
    let _e149 = diffuse_1;
    let _e151 = light_3;
    let _e155 = NoL_7;
    return (((_e148 + _e149) * _e151.color.xyz) * _e155);
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
    var light_accum: vec3<f32>;
    var i: i32;
    var i_1: i32;
    var diffuse_ambient: vec3<f32>;
    var specular_ambient: vec3<f32>;

    let _e40 = global_3.base_color;
    output_color = _e40;
    let _e42 = output_color;
    let _e44 = v_Uv_1;
    let _e45 = textureSample(StandardMaterial_base_color_texture, StandardMaterial_base_color_texture_sampler, _e44);
    output_color = (_e42 * _e45);
    let _e48 = v_Uv_1;
    let _e49 = textureSample(StandardMaterial_metallic_roughness_texture, StandardMaterial_metallic_roughness_texture_sampler, _e48);
    metallic_roughness = _e49;
    let _e51 = global_5.metallic;
    let _e52 = metallic_roughness;
    metallic = (_e51 * _e52.z);
    let _e56 = global_4.perceptual_roughness;
    let _e57 = metallic_roughness;
    perceptual_roughness_2 = (_e56 * _e57.y);
    let _e62 = perceptual_roughness_2;
    let _e63 = perceptualRoughnessToRoughness(_e62);
    roughness_12 = _e63;
    let _e66 = v_WorldNormal_1;
    N_2 = normalize(_e66);
    let _e69 = v_WorldTangent_1;
    let _e71 = v_WorldTangent_1;
    T = normalize(_e71.xyz);
    let _e77 = N_2;
    let _e78 = T;
    let _e80 = v_WorldTangent_1;
    B = (cross(_e77, _e78) * _e80.w);
    let _e85 = gl_FrontFacing;
    if _e85 {
        let _e86 = N_2;
        local = _e86;
    } else {
        let _e87 = N_2;
        local = -(_e87);
    }
    let _e90 = local;
    N_2 = _e90;
    let _e91 = gl_FrontFacing;
    if _e91 {
        let _e92 = T;
        local_1 = _e92;
    } else {
        let _e93 = T;
        local_1 = -(_e93);
    }
    let _e96 = local_1;
    T = _e96;
    let _e97 = gl_FrontFacing;
    if _e97 {
        let _e98 = B;
        local_2 = _e98;
    } else {
        let _e99 = B;
        local_2 = -(_e99);
    }
    let _e102 = local_2;
    B = _e102;
    let _e103 = T;
    let _e104 = B;
    let _e105 = N_2;
    TBN = mat3x3<f32>(vec3<f32>(_e103.x, _e103.y, _e103.z), vec3<f32>(_e104.x, _e104.y, _e104.z), vec3<f32>(_e105.x, _e105.y, _e105.z));
    let _e120 = TBN;
    let _e122 = v_Uv_1;
    let _e123 = textureSample(StandardMaterial_normal_map, StandardMaterial_normal_map_sampler, _e122);
    let _e131 = v_Uv_1;
    let _e132 = textureSample(StandardMaterial_normal_map, StandardMaterial_normal_map_sampler, _e131);
    N_2 = (_e120 * normalize(((_e132.xyz * 2.0) - vec3(1.0))));
    let _e142 = v_Uv_1;
    let _e143 = textureSample(StandardMaterial_occlusion_texture, StandardMaterial_occlusion_texture_sampler, _e142);
    occlusion = _e143.x;
    let _e146 = global_7.emissive;
    emissive = _e146;
    let _e148 = emissive;
    let _e150 = emissive;
    let _e153 = v_Uv_1;
    let _e154 = textureSample(StandardMaterial_emissive_texture, StandardMaterial_emissive_texture_sampler, _e153);
    let _e156 = (_e150.xyz * _e154.xyz);
    emissive.x = _e156.x;
    emissive.y = _e156.y;
    emissive.z = _e156.z;
    let _e163 = global_1.CameraPos;
    let _e165 = v_WorldPosition_1;
    let _e168 = global_1.CameraPos;
    let _e170 = v_WorldPosition_1;
    V_3 = normalize((_e168.xyz - _e170.xyz));
    let _e177 = N_2;
    let _e178 = V_3;
    let _e183 = N_2;
    let _e184 = V_3;
    NdotV_4 = max(dot(_e183, _e184), 0.001);
    let _e190 = global_6.reflectance;
    let _e192 = global_6.reflectance;
    let _e195 = metallic;
    let _e199 = output_color;
    let _e201 = metallic;
    F0_4 = (vec3((((0.16 * _e190) * _e192) * (1.0 - _e195))) + (_e199.xyz * vec3(_e201)));
    let _e206 = output_color;
    let _e209 = metallic;
    diffuseColor_4 = (_e206.xyz * vec3((1.0 - _e209)));
    let _e214 = V_3;
    let _e217 = V_3;
    let _e219 = N_2;
    R_4 = reflect(-(_e217), _e219);
    light_accum = vec3(0.0);
    i = 0;
    loop {
        let _e227 = i;
        let _e228 = global_2.NumLights;
        let _e232 = i;
        if !(((_e227 < i32(_e228.x)) && (_e232 < MAX_POINT_LIGHTS))) {
            break;
        }
        {
            let _e239 = light_accum;
            let _e240 = i;
            let _e250 = i;
            let _e252 = global_2.PointLights[_e250];
            let _e253 = roughness_12;
            let _e254 = NdotV_4;
            let _e255 = N_2;
            let _e256 = V_3;
            let _e257 = R_4;
            let _e258 = F0_4;
            let _e259 = diffuseColor_4;
            let _e260 = point_light(_e252, _e253, _e254, _e255, _e256, _e257, _e258, _e259);
            light_accum = (_e239 + _e260);
        }
        continuing {
            let _e236 = i;
            i = (_e236 + 1);
        }
    }
    i_1 = 0;
    loop {
        let _e264 = i_1;
        let _e265 = global_2.NumLights;
        let _e269 = i_1;
        if !(((_e264 < i32(_e265.y)) && (_e269 < MAX_DIRECTIONAL_LIGHTS))) {
            break;
        }
        {
            let _e276 = light_accum;
            let _e277 = i_1;
            let _e287 = i_1;
            let _e289 = global_2.DirectionalLights[_e287];
            let _e290 = roughness_12;
            let _e291 = NdotV_4;
            let _e292 = N_2;
            let _e293 = V_3;
            let _e294 = R_4;
            let _e295 = F0_4;
            let _e296 = diffuseColor_4;
            let _e297 = dir_light(_e289, _e290, _e291, _e292, _e293, _e294, _e295, _e296);
            light_accum = (_e276 + _e297);
        }
        continuing {
            let _e273 = i_1;
            i_1 = (_e273 + 1);
        }
    }
    let _e302 = diffuseColor_4;
    let _e304 = NdotV_4;
    let _e305 = EnvBRDFApprox(_e302, 1.0, _e304);
    diffuse_ambient = _e305;
    let _e310 = F0_4;
    let _e311 = perceptual_roughness_2;
    let _e312 = NdotV_4;
    let _e313 = EnvBRDFApprox(_e310, _e311, _e312);
    specular_ambient = _e313;
    let _e315 = output_color;
    let _e317 = light_accum;
    output_color.x = _e317.x;
    output_color.y = _e317.y;
    output_color.z = _e317.z;
    let _e324 = output_color;
    let _e326 = output_color;
    let _e328 = diffuse_ambient;
    let _e329 = specular_ambient;
    let _e331 = global_2.AmbientColor;
    let _e334 = occlusion;
    let _e336 = (_e326.xyz + (((_e328 + _e329) * _e331.xyz) * _e334));
    output_color.x = _e336.x;
    output_color.y = _e336.y;
    output_color.z = _e336.z;
    let _e343 = output_color;
    let _e345 = output_color;
    let _e347 = emissive;
    let _e349 = output_color;
    let _e352 = (_e345.xyz + (_e347.xyz * _e349.w));
    output_color.x = _e352.x;
    output_color.y = _e352.y;
    output_color.z = _e352.z;
    let _e359 = output_color;
    let _e361 = output_color;
    let _e363 = output_color;
    let _e365 = reinhard_luminance(_e363.xyz);
    output_color.x = _e365.x;
    output_color.y = _e365.y;
    output_color.z = _e365.z;
    let _e372 = output_color;
    o_Target = _e372;
    return;
}

@fragment 
fn main(@location(0) v_WorldPosition: vec3<f32>, @location(1) v_WorldNormal: vec3<f32>, @location(2) v_Uv: vec2<f32>, @location(3) v_WorldTangent: vec4<f32>, @builtin(front_facing) param: bool) -> FragmentOutput {
    v_WorldPosition_1 = v_WorldPosition;
    v_WorldNormal_1 = v_WorldNormal;
    v_Uv_1 = v_Uv;
    v_WorldTangent_1 = v_WorldTangent;
    gl_FrontFacing = param;
    main_1();
    let _e72 = o_Target;
    return FragmentOutput(_e72);
}
