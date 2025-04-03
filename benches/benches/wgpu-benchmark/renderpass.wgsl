@group(0) @binding(0)
var tex_1: texture_2d<f32>;

@group(0) @binding(1)
var tex_2: texture_2d<f32>;

@group(0) @binding(2)
var tex_3: texture_2d<f32>;

@group(0) @binding(3)
var tex_4: texture_2d<f32>;

@group(0) @binding(4)
var tex_5: texture_2d<f32>;

@group(0) @binding(5)
var tex_6: texture_2d<f32>;

@group(0) @binding(6)
var tex_7: texture_2d<f32>;

@vertex
fn vs_main() -> @builtin(position) vec4f {
    return vec4f(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return textureLoad(tex_1, vec2u(0), 0) +
           textureLoad(tex_2, vec2u(0), 0) +
           textureLoad(tex_3, vec2u(0), 0) +
           textureLoad(tex_4, vec2u(0), 0) +
           textureLoad(tex_5, vec2u(0), 0) +
           textureLoad(tex_6, vec2u(0), 0) +
           textureLoad(tex_7, vec2u(0), 0); 
}
