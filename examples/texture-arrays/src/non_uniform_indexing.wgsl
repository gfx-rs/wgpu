struct FragmentInput {
    @location(0) tex_coord: vec2<f32>,
    @location(1) index: i32,
}

@group(0) @binding(0)
var texture_array_top: binding_array<texture_2d<f32>>;
@group(0) @binding(1)
var texture_array_bottom: binding_array<texture_2d<f32>>;
@group(0) @binding(2)
var sampler_array: binding_array<sampler>;

@fragment
fn non_uniform_main(fragment: FragmentInput) -> @location(0) vec4<f32> {
    var outval: vec3<f32>;
    if fragment.tex_coord.y <= 0.5 {
        outval = textureSampleLevel(
            texture_array_top[fragment.index],
            sampler_array[fragment.index],
            fragment.tex_coord,
            0.0
        ).rgb;
    } else {
        outval = textureSampleLevel(
            texture_array_bottom[fragment.index],
            sampler_array[fragment.index],
            fragment.tex_coord,
            0.0
        ).rgb;
    }

    return vec4<f32>(outval.x, outval.y, outval.z, 1.0);
}
