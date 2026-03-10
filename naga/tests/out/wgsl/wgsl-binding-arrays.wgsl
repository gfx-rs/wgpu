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
    var u1_: u32 = 0u;
    var u2_: vec2<u32> = vec2(0u);
    var v1_: f32 = 0f;
    var v4_: vec4<f32> = vec4(0f);

    let uniform_index = uni.index;
    let non_uniform_index = fragment_in.index;
    let uv = vec2(0f);
    let pix = vec2(0i);
    let _e19 = u2_;
    let _e22 = textureDimensions(texture_array_unbounded[0]);
    u2_ = (_e19 + _e22);
    let _e24 = u2_;
    let _e27 = textureDimensions(texture_array_unbounded[uniform_index]);
    u2_ = (_e24 + _e27);
    let _e29 = u2_;
    let _e32 = textureDimensions(texture_array_unbounded[non_uniform_index]);
    u2_ = (_e29 + _e32);
    let _e34 = v4_;
    let _e39 = textureGather(0, texture_array_bounded[0], samp[0], uv);
    v4_ = (_e34 + _e39);
    let _e41 = v4_;
    let _e46 = textureGather(0, texture_array_bounded[uniform_index], samp[uniform_index], uv);
    v4_ = (_e41 + _e46);
    let _e48 = v4_;
    let _e53 = textureGather(0, texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv);
    v4_ = (_e48 + _e53);
    let _e55 = v4_;
    let _e61 = textureGatherCompare(texture_array_depth[0], samp_comp[0], uv, 0f);
    v4_ = (_e55 + _e61);
    let _e63 = v4_;
    let _e69 = textureGatherCompare(texture_array_depth[uniform_index], samp_comp[uniform_index], uv, 0f);
    v4_ = (_e63 + _e69);
    let _e71 = v4_;
    let _e77 = textureGatherCompare(texture_array_depth[non_uniform_index], samp_comp[non_uniform_index], uv, 0f);
    v4_ = (_e71 + _e77);
    let _e79 = v4_;
    let _e83 = textureLoad(texture_array_unbounded[0], pix, 0i);
    v4_ = (_e79 + _e83);
    let _e85 = v4_;
    let _e89 = textureLoad(texture_array_unbounded[uniform_index], pix, 0i);
    v4_ = (_e85 + _e89);
    let _e91 = v4_;
    let _e95 = textureLoad(texture_array_unbounded[non_uniform_index], pix, 0i);
    v4_ = (_e91 + _e95);
    let _e97 = u1_;
    let _e100 = textureNumLayers(texture_array_2darray[0]);
    u1_ = (_e97 + _e100);
    let _e102 = u1_;
    let _e105 = textureNumLayers(texture_array_2darray[uniform_index]);
    u1_ = (_e102 + _e105);
    let _e107 = u1_;
    let _e110 = textureNumLayers(texture_array_2darray[non_uniform_index]);
    u1_ = (_e107 + _e110);
    let _e112 = u1_;
    let _e115 = textureNumLevels(texture_array_bounded[0]);
    u1_ = (_e112 + _e115);
    let _e117 = u1_;
    let _e120 = textureNumLevels(texture_array_bounded[uniform_index]);
    u1_ = (_e117 + _e120);
    let _e122 = u1_;
    let _e125 = textureNumLevels(texture_array_bounded[non_uniform_index]);
    u1_ = (_e122 + _e125);
    let _e127 = u1_;
    let _e130 = textureNumSamples(texture_array_multisampled[0]);
    u1_ = (_e127 + _e130);
    let _e132 = u1_;
    let _e135 = textureNumSamples(texture_array_multisampled[uniform_index]);
    u1_ = (_e132 + _e135);
    let _e137 = u1_;
    let _e140 = textureNumSamples(texture_array_multisampled[non_uniform_index]);
    u1_ = (_e137 + _e140);
    let _e142 = v4_;
    let _e147 = textureSample(texture_array_bounded[0], samp[0], uv);
    v4_ = (_e142 + _e147);
    let _e149 = v4_;
    let _e154 = textureSample(texture_array_bounded[uniform_index], samp[uniform_index], uv);
    v4_ = (_e149 + _e154);
    let _e156 = v4_;
    let _e161 = textureSample(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv);
    v4_ = (_e156 + _e161);
    let _e163 = v4_;
    let _e169 = textureSampleBias(texture_array_bounded[0], samp[0], uv, 0f);
    v4_ = (_e163 + _e169);
    let _e171 = v4_;
    let _e177 = textureSampleBias(texture_array_bounded[uniform_index], samp[uniform_index], uv, 0f);
    v4_ = (_e171 + _e177);
    let _e179 = v4_;
    let _e185 = textureSampleBias(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv, 0f);
    v4_ = (_e179 + _e185);
    let _e187 = v1_;
    let _e193 = textureSampleCompare(texture_array_depth[0], samp_comp[0], uv, 0f);
    v1_ = (_e187 + _e193);
    let _e195 = v1_;
    let _e201 = textureSampleCompare(texture_array_depth[uniform_index], samp_comp[uniform_index], uv, 0f);
    v1_ = (_e195 + _e201);
    let _e203 = v1_;
    let _e209 = textureSampleCompare(texture_array_depth[non_uniform_index], samp_comp[non_uniform_index], uv, 0f);
    v1_ = (_e203 + _e209);
    let _e211 = v1_;
    let _e217 = textureSampleCompareLevel(texture_array_depth[0], samp_comp[0], uv, 0f);
    v1_ = (_e211 + _e217);
    let _e219 = v1_;
    let _e225 = textureSampleCompareLevel(texture_array_depth[uniform_index], samp_comp[uniform_index], uv, 0f);
    v1_ = (_e219 + _e225);
    let _e227 = v1_;
    let _e233 = textureSampleCompareLevel(texture_array_depth[non_uniform_index], samp_comp[non_uniform_index], uv, 0f);
    v1_ = (_e227 + _e233);
    let _e235 = v4_;
    let _e240 = textureSampleGrad(texture_array_bounded[0], samp[0], uv, uv, uv);
    v4_ = (_e235 + _e240);
    let _e242 = v4_;
    let _e247 = textureSampleGrad(texture_array_bounded[uniform_index], samp[uniform_index], uv, uv, uv);
    v4_ = (_e242 + _e247);
    let _e249 = v4_;
    let _e254 = textureSampleGrad(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv, uv, uv);
    v4_ = (_e249 + _e254);
    let _e256 = v4_;
    let _e262 = textureSampleLevel(texture_array_bounded[0], samp[0], uv, 0f);
    v4_ = (_e256 + _e262);
    let _e264 = v4_;
    let _e270 = textureSampleLevel(texture_array_bounded[uniform_index], samp[uniform_index], uv, 0f);
    v4_ = (_e264 + _e270);
    let _e272 = v4_;
    let _e278 = textureSampleLevel(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv, 0f);
    v4_ = (_e272 + _e278);
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
