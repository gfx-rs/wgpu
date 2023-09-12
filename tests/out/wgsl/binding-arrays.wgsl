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
    let _e22 = textureDimensions(texture_array_unbounded[0]);
    let _e23 = u2_;
    u2_ = (_e23 + _e22);
    let _e27 = textureDimensions(texture_array_unbounded[uniform_index]);
    let _e28 = u2_;
    u2_ = (_e28 + _e27);
    let _e32 = textureDimensions(texture_array_unbounded[non_uniform_index]);
    let _e33 = u2_;
    u2_ = (_e33 + _e32);
    let _e42 = textureGather(0, texture_array_bounded[0], samp[0], uv);
    let _e43 = v4_;
    v4_ = (_e43 + _e42);
    let _e50 = textureGather(0, texture_array_bounded[uniform_index], samp[uniform_index], uv);
    let _e51 = v4_;
    v4_ = (_e51 + _e50);
    let _e58 = textureGather(0, texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv);
    let _e59 = v4_;
    v4_ = (_e59 + _e58);
    let _e68 = textureGatherCompare(texture_array_depth[0], samp_comp[0], uv, 0.0);
    let _e69 = v4_;
    v4_ = (_e69 + _e68);
    let _e76 = textureGatherCompare(texture_array_depth[uniform_index], samp_comp[uniform_index], uv, 0.0);
    let _e77 = v4_;
    v4_ = (_e77 + _e76);
    let _e84 = textureGatherCompare(texture_array_depth[non_uniform_index], samp_comp[non_uniform_index], uv, 0.0);
    let _e85 = v4_;
    v4_ = (_e85 + _e84);
    let _e91 = textureLoad(texture_array_unbounded[0], pix, 0);
    let _e92 = v4_;
    v4_ = (_e92 + _e91);
    let _e97 = textureLoad(texture_array_unbounded[uniform_index], pix, 0);
    let _e98 = v4_;
    v4_ = (_e98 + _e97);
    let _e103 = textureLoad(texture_array_unbounded[non_uniform_index], pix, 0);
    let _e104 = v4_;
    v4_ = (_e104 + _e103);
    let _e109 = textureNumLayers(texture_array_2darray[0]);
    let _e110 = u1_;
    u1_ = (_e110 + _e109);
    let _e114 = textureNumLayers(texture_array_2darray[uniform_index]);
    let _e115 = u1_;
    u1_ = (_e115 + _e114);
    let _e119 = textureNumLayers(texture_array_2darray[non_uniform_index]);
    let _e120 = u1_;
    u1_ = (_e120 + _e119);
    let _e125 = textureNumLevels(texture_array_bounded[0]);
    let _e126 = u1_;
    u1_ = (_e126 + _e125);
    let _e130 = textureNumLevels(texture_array_bounded[uniform_index]);
    let _e131 = u1_;
    u1_ = (_e131 + _e130);
    let _e135 = textureNumLevels(texture_array_bounded[non_uniform_index]);
    let _e136 = u1_;
    u1_ = (_e136 + _e135);
    let _e141 = textureNumSamples(texture_array_multisampled[0]);
    let _e142 = u1_;
    u1_ = (_e142 + _e141);
    let _e146 = textureNumSamples(texture_array_multisampled[uniform_index]);
    let _e147 = u1_;
    u1_ = (_e147 + _e146);
    let _e151 = textureNumSamples(texture_array_multisampled[non_uniform_index]);
    let _e152 = u1_;
    u1_ = (_e152 + _e151);
    let _e160 = textureSample(texture_array_bounded[0], samp[0], uv);
    let _e161 = v4_;
    v4_ = (_e161 + _e160);
    let _e167 = textureSample(texture_array_bounded[uniform_index], samp[uniform_index], uv);
    let _e168 = v4_;
    v4_ = (_e168 + _e167);
    let _e174 = textureSample(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv);
    let _e175 = v4_;
    v4_ = (_e175 + _e174);
    let _e184 = textureSampleBias(texture_array_bounded[0], samp[0], uv, 0.0);
    let _e185 = v4_;
    v4_ = (_e185 + _e184);
    let _e192 = textureSampleBias(texture_array_bounded[uniform_index], samp[uniform_index], uv, 0.0);
    let _e193 = v4_;
    v4_ = (_e193 + _e192);
    let _e200 = textureSampleBias(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv, 0.0);
    let _e201 = v4_;
    v4_ = (_e201 + _e200);
    let _e210 = textureSampleCompare(texture_array_depth[0], samp_comp[0], uv, 0.0);
    let _e211 = v1_;
    v1_ = (_e211 + _e210);
    let _e218 = textureSampleCompare(texture_array_depth[uniform_index], samp_comp[uniform_index], uv, 0.0);
    let _e219 = v1_;
    v1_ = (_e219 + _e218);
    let _e226 = textureSampleCompare(texture_array_depth[non_uniform_index], samp_comp[non_uniform_index], uv, 0.0);
    let _e227 = v1_;
    v1_ = (_e227 + _e226);
    let _e236 = textureSampleCompareLevel(texture_array_depth[0], samp_comp[0], uv, 0.0);
    let _e237 = v1_;
    v1_ = (_e237 + _e236);
    let _e244 = textureSampleCompareLevel(texture_array_depth[uniform_index], samp_comp[uniform_index], uv, 0.0);
    let _e245 = v1_;
    v1_ = (_e245 + _e244);
    let _e252 = textureSampleCompareLevel(texture_array_depth[non_uniform_index], samp_comp[non_uniform_index], uv, 0.0);
    let _e253 = v1_;
    v1_ = (_e253 + _e252);
    let _e261 = textureSampleGrad(texture_array_bounded[0], samp[0], uv, uv, uv);
    let _e262 = v4_;
    v4_ = (_e262 + _e261);
    let _e268 = textureSampleGrad(texture_array_bounded[uniform_index], samp[uniform_index], uv, uv, uv);
    let _e269 = v4_;
    v4_ = (_e269 + _e268);
    let _e275 = textureSampleGrad(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv, uv, uv);
    let _e276 = v4_;
    v4_ = (_e276 + _e275);
    let _e285 = textureSampleLevel(texture_array_bounded[0], samp[0], uv, 0.0);
    let _e286 = v4_;
    v4_ = (_e286 + _e285);
    let _e293 = textureSampleLevel(texture_array_bounded[uniform_index], samp[uniform_index], uv, 0.0);
    let _e294 = v4_;
    v4_ = (_e294 + _e293);
    let _e301 = textureSampleLevel(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv, 0.0);
    let _e302 = v4_;
    v4_ = (_e302 + _e301);
    let _e307 = v4_;
    textureStore(texture_array_storage[0], pix, _e307);
    let _e310 = v4_;
    textureStore(texture_array_storage[uniform_index], pix, _e310);
    let _e313 = v4_;
    textureStore(texture_array_storage[non_uniform_index], pix, _e313);
    let _e314 = u2_;
    let _e315 = u1_;
    let v2_ = vec2<f32>((_e314 + vec2(_e315)));
    let _e319 = v4_;
    let _e326 = v1_;
    return ((_e319 + vec4<f32>(v2_.x, v2_.y, v2_.x, v2_.y)) + vec4(_e326));
}
