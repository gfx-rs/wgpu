enable f16;

@fragment
fn test_direct(
    @location(0) scalar_f16: f16,
    @location(1) scalar_f32: f32,
    @location(2) vec2_f16: vec2<f16>,
    @location(3) vec2_f32: vec2<f32>,
    @location(4) vec3_f16: vec3<f16>,
    @location(5) vec3_f32: vec3<f32>,
    @location(6) vec4_f16: vec4<f16>,
    @location(7) vec4_f32: vec4<f32>,
) -> F16IO {
    var output: F16IO;
    output.scalar_f16 = scalar_f16 + 1.0h;
    output.scalar_f32 = scalar_f32 + 1.0;
    output.vec2_f16 = vec2_f16 + vec2(1.0h);
    output.vec2_f32 = vec2_f32 + vec2(1.0);
    output.vec3_f16 = vec3_f16 + vec3(1.0h);
    output.vec3_f32 = vec3_f32 + vec3(1.0);
    output.vec4_f16 = vec4_f16 + vec4(1.0h);
    output.vec4_f32 = vec4_f32 + vec4(1.0);
    return output;
}

struct F16IO {
    @location(0) scalar_f16: f16,
    @location(1) scalar_f32: f32,
    @location(2) vec2_f16: vec2<f16>,
    @location(3) vec2_f32: vec2<f32>,
    @location(4) vec3_f16: vec3<f16>,
    @location(5) vec3_f32: vec3<f32>,
    @location(6) vec4_f16: vec4<f16>,
    @location(7) vec4_f32: vec4<f32>,
}

@fragment
fn test_struct(input: F16IO) -> F16IO {
    var output: F16IO;
    output.scalar_f16 = input.scalar_f16 + 1.0h;
    output.scalar_f32 = input.scalar_f32 + 1.0;
    output.vec2_f16 = input.vec2_f16 + vec2(1.0h);
    output.vec2_f32 = input.vec2_f32 + vec2(1.0);
    output.vec3_f16 = input.vec3_f16 + vec3(1.0h);
    output.vec3_f32 = input.vec3_f32 + vec3(1.0);
    output.vec4_f16 = input.vec4_f16 + vec4(1.0h);
    output.vec4_f32 = input.vec4_f32 + vec4(1.0);
    return output;
}

@fragment
fn test_copy_input(input_original: F16IO) -> F16IO {
    var input = input_original;
    var output: F16IO;
    output.scalar_f16 = input.scalar_f16 + 1.0h;
    output.scalar_f32 = input.scalar_f32 + 1.0;
    output.vec2_f16 = input.vec2_f16 + vec2(1.0h);
    output.vec2_f32 = input.vec2_f32 + vec2(1.0);
    output.vec3_f16 = input.vec3_f16 + vec3(1.0h);
    output.vec3_f32 = input.vec3_f32 + vec3(1.0);
    output.vec4_f16 = input.vec4_f16 + vec4(1.0h);
    output.vec4_f32 = input.vec4_f32 + vec4(1.0);
    return output;
}

@fragment
fn test_return_partial(input_original: F16IO) -> @location(0) f16 {
    var input = input_original;
    input.scalar_f16 = 0.0h;
    return input.scalar_f16;
}

@fragment
fn test_component_access(input: F16IO) -> F16IO {
    var output: F16IO;
    output.vec2_f16.x = input.vec2_f16.y;
    output.vec2_f16.y = input.vec2_f16.x;
    return output;
}