enable primitive_index;

struct ImmediateData {
    index: u32,
    double: vec2<f32>,
}

struct FragmentIn {
    @location(0) color: vec4<f32>,
    @builtin(primitive_index) primitive_index: u32,
}

var<immediate> im: ImmediateData;

@fragment
fn main(in: FragmentIn) -> @location(0) vec4<f32> {
    let _e4 = im.index;
    if (in.primitive_index == _e4) {
        return in.color;
    } else {
        let _e11 = (vec3(1f) - in.color.xyz);
        return vec4<f32>(_e11.x, _e11.y, _e11.z, in.color.w);
    }
}
