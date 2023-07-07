struct UniformIndex {
    index: u32,
}

struct FragmentIn {
    @location(0) @interpolate(flat) index: u32,
}

@group(0) @binding(0) 
var texture_array_unbounded: binding_array<texture_2d<f32>>;
@group(0) @binding(1) 
var texture_array_bounded: binding_array<texture_2d<f32>,5>;
@group(0) @binding(2) 
var texture_array_2darray: binding_array<texture_2d_array<f32>,5>;
@group(0) @binding(3) 
var texture_array_multisampled: binding_array<texture_multisampled_2d<f32>,5>;
@group(0) @binding(4) 
var texture_array_depth: binding_array<texture_depth_2d,5>;
@group(0) @binding(5) 
var texture_array_storage: binding_array<texture_storage_2d<rgba32float,write>,5>;
@group(0) @binding(6) 
var samp: binding_array<sampler,5>;
@group(0) @binding(7) 
var samp_comp: binding_array<sampler_comparison,5>;
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
    u2_ = vec2<u32>(0u);
    v1_ = 0.0;
    v4_ = vec4<f32>(0.0);
    let uv = vec2<f32>(0.0);
    let pix = vec2<i32>(0);
    let _e22 = textureDimensions(texture_array_unbounded[0]);
    let _e23 = u2_;
    u2_ = (_e23 + _e22);
    let _e27 = textureDimensions(texture_array_unbounded[uniform_index]);
    let _e28 = u2_;
    u2_ = (_e28 + _e27);
    let _e32 = textureDimensions(texture_array_unbounded[non_uniform_index]);
    let _e33 = u2_;
    u2_ = (_e33 + _e32);
    let _e41 = textureGather(0, texture_array_bounded[0], samp[0], uv);
    let _e42 = v4_;
    v4_ = (_e42 + _e41);
    let _e48 = textureGather(0, texture_array_bounded[uniform_index], samp[uniform_index], uv);
    let _e49 = v4_;
    v4_ = (_e49 + _e48);
    let _e55 = textureGather(0, texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv);
    let _e56 = v4_;
    v4_ = (_e56 + _e55);
    let _e65 = textureGatherCompare(texture_array_depth[0], samp_comp[0], uv, 0.0);
    let _e66 = v4_;
    v4_ = (_e66 + _e65);
    let _e73 = textureGatherCompare(texture_array_depth[uniform_index], samp_comp[uniform_index], uv, 0.0);
    let _e74 = v4_;
    v4_ = (_e74 + _e73);
    let _e81 = textureGatherCompare(texture_array_depth[non_uniform_index], samp_comp[non_uniform_index], uv, 0.0);
    let _e82 = v4_;
    v4_ = (_e82 + _e81);
    let _e88 = textureLoad(texture_array_unbounded[0], pix, 0);
    let _e89 = v4_;
    v4_ = (_e89 + _e88);
    let _e94 = textureLoad(texture_array_unbounded[uniform_index], pix, 0);
    let _e95 = v4_;
    v4_ = (_e95 + _e94);
    let _e100 = textureLoad(texture_array_unbounded[non_uniform_index], pix, 0);
    let _e101 = v4_;
    v4_ = (_e101 + _e100);
    let _e106 = textureNumLayers(texture_array_2darray[0]);
    let _e107 = u1_;
    u1_ = (_e107 + _e106);
    let _e111 = textureNumLayers(texture_array_2darray[uniform_index]);
    let _e112 = u1_;
    u1_ = (_e112 + _e111);
    let _e116 = textureNumLayers(texture_array_2darray[non_uniform_index]);
    let _e117 = u1_;
    u1_ = (_e117 + _e116);
    let _e122 = textureNumLevels(texture_array_bounded[0]);
    let _e123 = u1_;
    u1_ = (_e123 + _e122);
    let _e127 = textureNumLevels(texture_array_bounded[uniform_index]);
    let _e128 = u1_;
    u1_ = (_e128 + _e127);
    let _e132 = textureNumLevels(texture_array_bounded[non_uniform_index]);
    let _e133 = u1_;
    u1_ = (_e133 + _e132);
    let _e138 = textureNumSamples(texture_array_multisampled[0]);
    let _e139 = u1_;
    u1_ = (_e139 + _e138);
    let _e143 = textureNumSamples(texture_array_multisampled[uniform_index]);
    let _e144 = u1_;
    u1_ = (_e144 + _e143);
    let _e148 = textureNumSamples(texture_array_multisampled[non_uniform_index]);
    let _e149 = u1_;
    u1_ = (_e149 + _e148);
    let _e157 = textureSample(texture_array_bounded[0], samp[0], uv);
    let _e158 = v4_;
    v4_ = (_e158 + _e157);
    let _e164 = textureSample(texture_array_bounded[uniform_index], samp[uniform_index], uv);
    let _e165 = v4_;
    v4_ = (_e165 + _e164);
    let _e171 = textureSample(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv);
    let _e172 = v4_;
    v4_ = (_e172 + _e171);
    let _e181 = textureSampleBias(texture_array_bounded[0], samp[0], uv, 0.0);
    let _e182 = v4_;
    v4_ = (_e182 + _e181);
    let _e189 = textureSampleBias(texture_array_bounded[uniform_index], samp[uniform_index], uv, 0.0);
    let _e190 = v4_;
    v4_ = (_e190 + _e189);
    let _e197 = textureSampleBias(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv, 0.0);
    let _e198 = v4_;
    v4_ = (_e198 + _e197);
    let _e207 = textureSampleCompare(texture_array_depth[0], samp_comp[0], uv, 0.0);
    let _e208 = v1_;
    v1_ = (_e208 + _e207);
    let _e215 = textureSampleCompare(texture_array_depth[uniform_index], samp_comp[uniform_index], uv, 0.0);
    let _e216 = v1_;
    v1_ = (_e216 + _e215);
    let _e223 = textureSampleCompare(texture_array_depth[non_uniform_index], samp_comp[non_uniform_index], uv, 0.0);
    let _e224 = v1_;
    v1_ = (_e224 + _e223);
    let _e233 = textureSampleCompareLevel(texture_array_depth[0], samp_comp[0], uv, 0.0);
    let _e234 = v1_;
    v1_ = (_e234 + _e233);
    let _e241 = textureSampleCompareLevel(texture_array_depth[uniform_index], samp_comp[uniform_index], uv, 0.0);
    let _e242 = v1_;
    v1_ = (_e242 + _e241);
    let _e249 = textureSampleCompareLevel(texture_array_depth[non_uniform_index], samp_comp[non_uniform_index], uv, 0.0);
    let _e250 = v1_;
    v1_ = (_e250 + _e249);
    let _e258 = textureSampleGrad(texture_array_bounded[0], samp[0], uv, uv, uv);
    let _e259 = v4_;
    v4_ = (_e259 + _e258);
    let _e265 = textureSampleGrad(texture_array_bounded[uniform_index], samp[uniform_index], uv, uv, uv);
    let _e266 = v4_;
    v4_ = (_e266 + _e265);
    let _e272 = textureSampleGrad(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv, uv, uv);
    let _e273 = v4_;
    v4_ = (_e273 + _e272);
    let _e282 = textureSampleLevel(texture_array_bounded[0], samp[0], uv, 0.0);
    let _e283 = v4_;
    v4_ = (_e283 + _e282);
    let _e290 = textureSampleLevel(texture_array_bounded[uniform_index], samp[uniform_index], uv, 0.0);
    let _e291 = v4_;
    v4_ = (_e291 + _e290);
    let _e298 = textureSampleLevel(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv, 0.0);
    let _e299 = v4_;
    v4_ = (_e299 + _e298);
    let _e304 = v4_;
    textureStore(texture_array_storage[0], pix, _e304);
    let _e307 = v4_;
    textureStore(texture_array_storage[uniform_index], pix, _e307);
    let _e310 = v4_;
    textureStore(texture_array_storage[non_uniform_index], pix, _e310);
    let _e311 = u2_;
    let _e312 = u1_;
    let v2_ = vec2<f32>((_e311 + vec2<u32>(_e312)));
    let _e316 = v4_;
    let _e323 = v1_;
    return ((_e316 + vec4<f32>(v2_.x, v2_.y, v2_.x, v2_.y)) + vec4<f32>(_e323));
}
