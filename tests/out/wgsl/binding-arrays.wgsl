struct UniformIndex {
    index: u32,
}

struct FragmentIn {
    @location(0) index: u32,
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
    var i1_: i32 = 0;
    var i2_: vec2<i32>;
    var v1_: f32 = 0.0;
    var v4_: vec4<f32>;

    let uniform_index = uni.index;
    let non_uniform_index = fragment_in.index;
    i2_ = vec2<i32>(0);
    v4_ = vec4<f32>(0.0);
    let uv = vec2<f32>(0.0);
    let pix = vec2<i32>(0);
    let _e27 = i2_;
    let _e30 = textureDimensions(texture_array_unbounded[0]);
    i2_ = (_e27 + _e30);
    let _e32 = i2_;
    let _e34 = textureDimensions(texture_array_unbounded[uniform_index]);
    i2_ = (_e32 + _e34);
    let _e36 = i2_;
    let _e38 = textureDimensions(texture_array_unbounded[non_uniform_index]);
    i2_ = (_e36 + _e38);
    let _e40 = v4_;
    let _e45 = textureGather(0, texture_array_bounded[0], samp[0], uv);
    v4_ = (_e40 + _e45);
    let _e47 = v4_;
    let _e50 = textureGather(0, texture_array_bounded[uniform_index], samp[uniform_index], uv);
    v4_ = (_e47 + _e50);
    let _e52 = v4_;
    let _e55 = textureGather(0, texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv);
    v4_ = (_e52 + _e55);
    let _e57 = v4_;
    let _e63 = textureGatherCompare(texture_array_depth[0], samp_comp[0], uv, 0.0);
    v4_ = (_e57 + _e63);
    let _e65 = v4_;
    let _e69 = textureGatherCompare(texture_array_depth[uniform_index], samp_comp[uniform_index], uv, 0.0);
    v4_ = (_e65 + _e69);
    let _e71 = v4_;
    let _e75 = textureGatherCompare(texture_array_depth[non_uniform_index], samp_comp[non_uniform_index], uv, 0.0);
    v4_ = (_e71 + _e75);
    let _e77 = v4_;
    let _e81 = textureLoad(texture_array_unbounded[0], pix, 0);
    v4_ = (_e77 + _e81);
    let _e83 = v4_;
    let _e86 = textureLoad(texture_array_unbounded[uniform_index], pix, 0);
    v4_ = (_e83 + _e86);
    let _e88 = v4_;
    let _e91 = textureLoad(texture_array_unbounded[non_uniform_index], pix, 0);
    v4_ = (_e88 + _e91);
    let _e93 = i1_;
    let _e96 = textureNumLayers(texture_array_2darray[0]);
    i1_ = (_e93 + _e96);
    let _e98 = i1_;
    let _e100 = textureNumLayers(texture_array_2darray[uniform_index]);
    i1_ = (_e98 + _e100);
    let _e102 = i1_;
    let _e104 = textureNumLayers(texture_array_2darray[non_uniform_index]);
    i1_ = (_e102 + _e104);
    let _e106 = i1_;
    let _e109 = textureNumLevels(texture_array_bounded[0]);
    i1_ = (_e106 + _e109);
    let _e111 = i1_;
    let _e113 = textureNumLevels(texture_array_bounded[uniform_index]);
    i1_ = (_e111 + _e113);
    let _e115 = i1_;
    let _e117 = textureNumLevels(texture_array_bounded[non_uniform_index]);
    i1_ = (_e115 + _e117);
    let _e119 = i1_;
    let _e122 = textureNumSamples(texture_array_multisampled[0]);
    i1_ = (_e119 + _e122);
    let _e124 = i1_;
    let _e126 = textureNumSamples(texture_array_multisampled[uniform_index]);
    i1_ = (_e124 + _e126);
    let _e128 = i1_;
    let _e130 = textureNumSamples(texture_array_multisampled[non_uniform_index]);
    i1_ = (_e128 + _e130);
    let _e132 = v4_;
    let _e137 = textureSample(texture_array_bounded[0], samp[0], uv);
    v4_ = (_e132 + _e137);
    let _e139 = v4_;
    let _e142 = textureSample(texture_array_bounded[uniform_index], samp[uniform_index], uv);
    v4_ = (_e139 + _e142);
    let _e144 = v4_;
    let _e147 = textureSample(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv);
    v4_ = (_e144 + _e147);
    let _e149 = v4_;
    let _e155 = textureSampleBias(texture_array_bounded[0], samp[0], uv, 0.0);
    v4_ = (_e149 + _e155);
    let _e157 = v4_;
    let _e161 = textureSampleBias(texture_array_bounded[uniform_index], samp[uniform_index], uv, 0.0);
    v4_ = (_e157 + _e161);
    let _e163 = v4_;
    let _e167 = textureSampleBias(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv, 0.0);
    v4_ = (_e163 + _e167);
    let _e169 = v1_;
    let _e175 = textureSampleCompare(texture_array_depth[0], samp_comp[0], uv, 0.0);
    v1_ = (_e169 + _e175);
    let _e177 = v1_;
    let _e181 = textureSampleCompare(texture_array_depth[uniform_index], samp_comp[uniform_index], uv, 0.0);
    v1_ = (_e177 + _e181);
    let _e183 = v1_;
    let _e187 = textureSampleCompare(texture_array_depth[non_uniform_index], samp_comp[non_uniform_index], uv, 0.0);
    v1_ = (_e183 + _e187);
    let _e189 = v1_;
    let _e195 = textureSampleCompareLevel(texture_array_depth[0], samp_comp[0], uv, 0.0);
    v1_ = (_e189 + _e195);
    let _e197 = v1_;
    let _e201 = textureSampleCompareLevel(texture_array_depth[uniform_index], samp_comp[uniform_index], uv, 0.0);
    v1_ = (_e197 + _e201);
    let _e203 = v1_;
    let _e207 = textureSampleCompareLevel(texture_array_depth[non_uniform_index], samp_comp[non_uniform_index], uv, 0.0);
    v1_ = (_e203 + _e207);
    let _e209 = v4_;
    let _e214 = textureSampleGrad(texture_array_bounded[0], samp[0], uv, uv, uv);
    v4_ = (_e209 + _e214);
    let _e216 = v4_;
    let _e219 = textureSampleGrad(texture_array_bounded[uniform_index], samp[uniform_index], uv, uv, uv);
    v4_ = (_e216 + _e219);
    let _e221 = v4_;
    let _e224 = textureSampleGrad(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv, uv, uv);
    v4_ = (_e221 + _e224);
    let _e226 = v4_;
    let _e232 = textureSampleLevel(texture_array_bounded[0], samp[0], uv, 0.0);
    v4_ = (_e226 + _e232);
    let _e234 = v4_;
    let _e238 = textureSampleLevel(texture_array_bounded[uniform_index], samp[uniform_index], uv, 0.0);
    v4_ = (_e234 + _e238);
    let _e240 = v4_;
    let _e244 = textureSampleLevel(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv, 0.0);
    v4_ = (_e240 + _e244);
    let _e248 = v4_;
    textureStore(texture_array_storage[0], pix, _e248);
    let _e250 = v4_;
    textureStore(texture_array_storage[uniform_index], pix, _e250);
    let _e252 = v4_;
    textureStore(texture_array_storage[non_uniform_index], pix, _e252);
    let _e253 = i2_;
    let _e254 = i1_;
    let v2_ = vec2<f32>((_e253 + vec2<i32>(_e254)));
    let _e258 = v4_;
    let _e265 = v1_;
    return ((_e258 + vec4<f32>(v2_.x, v2_.y, v2_.x, v2_.y)) + vec4<f32>(_e265));
}
