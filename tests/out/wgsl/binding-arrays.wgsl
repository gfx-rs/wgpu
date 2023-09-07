struct UniformIndex {
    index: u32,
}

struct FragmentIn {
    @location(0) @interpolate(flat) index: u32,
}

@group(0) @binding(0) 
var texture_array_unbounded: binding_array<texture_2d<f32>>;
@group(0) @binding(1) 
var texture_array_bounded: binding_array<texture_2d<f32>, 5>;
@group(0) @binding(2) 
var texture_array_2darray: binding_array<texture_2d_array<f32>, 5>;
@group(0) @binding(3) 
var texture_array_multisampled: binding_array<texture_multisampled_2d<f32>, 5>;
@group(0) @binding(4) 
var texture_array_depth: binding_array<texture_depth_2d, 5>;
@group(0) @binding(5) 
var texture_array_storage: binding_array<texture_storage_2d<rgba32float,write>, 5>;
@group(0) @binding(6) 
var samp: binding_array<sampler, 5>;
@group(0) @binding(7) 
var samp_comp: binding_array<sampler_comparison, 5>;
@group(0) @binding(8) 
var<uniform> uni: UniformIndex;

@fragment 
fn main(fragment_in: FragmentIn) -> @location(0) vec4<f32> {
    var u1_: u32;
    var u2_: vec2<u32>;
    var v1_: f32;
    var v4_: vec4<f32>;

    let uniform_index = uni.index;
    let non_uniform_index = fragment_in.index;
    u1_ = 0u;
    u2_ = vec2(0u);
    v1_ = 0.0;
    v4_ = vec4(0.0);
    let uv = vec2(0.0);
    let pix = vec2(0);
    let _e21 = textureDimensions(texture_array_unbounded[0]);
    let _e22 = u2_;
    u2_ = (_e22 + _e21);
    let _e26 = textureDimensions(texture_array_unbounded[uniform_index]);
    let _e27 = u2_;
    u2_ = (_e27 + _e26);
    let _e31 = textureDimensions(texture_array_unbounded[non_uniform_index]);
    let _e32 = u2_;
    u2_ = (_e32 + _e31);
    let _e38 = textureGather(0, texture_array_bounded[0], samp[0], uv);
    let _e39 = v4_;
    v4_ = (_e39 + _e38);
    let _e45 = textureGather(0, texture_array_bounded[uniform_index], samp[uniform_index], uv);
    let _e46 = v4_;
    v4_ = (_e46 + _e45);
    let _e52 = textureGather(0, texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv);
    let _e53 = v4_;
    v4_ = (_e53 + _e52);
    let _e60 = textureGatherCompare(texture_array_depth[0], samp_comp[0], uv, 0.0);
    let _e61 = v4_;
    v4_ = (_e61 + _e60);
    let _e68 = textureGatherCompare(texture_array_depth[uniform_index], samp_comp[uniform_index], uv, 0.0);
    let _e69 = v4_;
    v4_ = (_e69 + _e68);
    let _e76 = textureGatherCompare(texture_array_depth[non_uniform_index], samp_comp[non_uniform_index], uv, 0.0);
    let _e77 = v4_;
    v4_ = (_e77 + _e76);
    let _e82 = textureLoad(texture_array_unbounded[0], pix, 0);
    let _e83 = v4_;
    v4_ = (_e83 + _e82);
    let _e88 = textureLoad(texture_array_unbounded[uniform_index], pix, 0);
    let _e89 = v4_;
    v4_ = (_e89 + _e88);
    let _e94 = textureLoad(texture_array_unbounded[non_uniform_index], pix, 0);
    let _e95 = v4_;
    v4_ = (_e95 + _e94);
    let _e99 = textureNumLayers(texture_array_2darray[0]);
    let _e100 = u1_;
    u1_ = (_e100 + _e99);
    let _e104 = textureNumLayers(texture_array_2darray[uniform_index]);
    let _e105 = u1_;
    u1_ = (_e105 + _e104);
    let _e109 = textureNumLayers(texture_array_2darray[non_uniform_index]);
    let _e110 = u1_;
    u1_ = (_e110 + _e109);
    let _e114 = textureNumLevels(texture_array_bounded[0]);
    let _e115 = u1_;
    u1_ = (_e115 + _e114);
    let _e119 = textureNumLevels(texture_array_bounded[uniform_index]);
    let _e120 = u1_;
    u1_ = (_e120 + _e119);
    let _e124 = textureNumLevels(texture_array_bounded[non_uniform_index]);
    let _e125 = u1_;
    u1_ = (_e125 + _e124);
    let _e129 = textureNumSamples(texture_array_multisampled[0]);
    let _e130 = u1_;
    u1_ = (_e130 + _e129);
    let _e134 = textureNumSamples(texture_array_multisampled[uniform_index]);
    let _e135 = u1_;
    u1_ = (_e135 + _e134);
    let _e139 = textureNumSamples(texture_array_multisampled[non_uniform_index]);
    let _e140 = u1_;
    u1_ = (_e140 + _e139);
    let _e146 = textureSample(texture_array_bounded[0], samp[0], uv);
    let _e147 = v4_;
    v4_ = (_e147 + _e146);
    let _e153 = textureSample(texture_array_bounded[uniform_index], samp[uniform_index], uv);
    let _e154 = v4_;
    v4_ = (_e154 + _e153);
    let _e160 = textureSample(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv);
    let _e161 = v4_;
    v4_ = (_e161 + _e160);
    let _e168 = textureSampleBias(texture_array_bounded[0], samp[0], uv, 0.0);
    let _e169 = v4_;
    v4_ = (_e169 + _e168);
    let _e176 = textureSampleBias(texture_array_bounded[uniform_index], samp[uniform_index], uv, 0.0);
    let _e177 = v4_;
    v4_ = (_e177 + _e176);
    let _e184 = textureSampleBias(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv, 0.0);
    let _e185 = v4_;
    v4_ = (_e185 + _e184);
    let _e192 = textureSampleCompare(texture_array_depth[0], samp_comp[0], uv, 0.0);
    let _e193 = v1_;
    v1_ = (_e193 + _e192);
    let _e200 = textureSampleCompare(texture_array_depth[uniform_index], samp_comp[uniform_index], uv, 0.0);
    let _e201 = v1_;
    v1_ = (_e201 + _e200);
    let _e208 = textureSampleCompare(texture_array_depth[non_uniform_index], samp_comp[non_uniform_index], uv, 0.0);
    let _e209 = v1_;
    v1_ = (_e209 + _e208);
    let _e216 = textureSampleCompareLevel(texture_array_depth[0], samp_comp[0], uv, 0.0);
    let _e217 = v1_;
    v1_ = (_e217 + _e216);
    let _e224 = textureSampleCompareLevel(texture_array_depth[uniform_index], samp_comp[uniform_index], uv, 0.0);
    let _e225 = v1_;
    v1_ = (_e225 + _e224);
    let _e232 = textureSampleCompareLevel(texture_array_depth[non_uniform_index], samp_comp[non_uniform_index], uv, 0.0);
    let _e233 = v1_;
    v1_ = (_e233 + _e232);
    let _e239 = textureSampleGrad(texture_array_bounded[0], samp[0], uv, uv, uv);
    let _e240 = v4_;
    v4_ = (_e240 + _e239);
    let _e246 = textureSampleGrad(texture_array_bounded[uniform_index], samp[uniform_index], uv, uv, uv);
    let _e247 = v4_;
    v4_ = (_e247 + _e246);
    let _e253 = textureSampleGrad(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv, uv, uv);
    let _e254 = v4_;
    v4_ = (_e254 + _e253);
    let _e261 = textureSampleLevel(texture_array_bounded[0], samp[0], uv, 0.0);
    let _e262 = v4_;
    v4_ = (_e262 + _e261);
    let _e269 = textureSampleLevel(texture_array_bounded[uniform_index], samp[uniform_index], uv, 0.0);
    let _e270 = v4_;
    v4_ = (_e270 + _e269);
    let _e277 = textureSampleLevel(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv, 0.0);
    let _e278 = v4_;
    v4_ = (_e278 + _e277);
    let _e282 = v4_;
    textureStore(texture_array_storage[0], pix, _e282);
    let _e285 = v4_;
    textureStore(texture_array_storage[uniform_index], pix, _e285);
    let _e288 = v4_;
    textureStore(texture_array_storage[non_uniform_index], pix, _e288);
    let _e289 = u2_;
    let _e290 = u1_;
    let v2_ = vec2<f32>((_e289 + vec2(_e290)));
    let _e294 = v4_;
    let _e301 = v1_;
    return ((_e294 + vec4<f32>(v2_.x, v2_.y, v2_.x, v2_.y)) + vec4(_e301));
}
