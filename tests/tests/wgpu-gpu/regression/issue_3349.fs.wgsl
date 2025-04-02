struct ShaderData {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
}

@group(0) @binding(0)
var<uniform> data1: ShaderData;

var<push_constant> data2: ShaderData;

struct FsIn {
    @builtin(position) position: vec4f,
    @location(0) data1: vec4f,
    @location(1) data2: vec4f,
}

@fragment
fn fs_main(fs_in: FsIn) -> @location(0) vec4f {
    let floored = vec2u(floor(fs_in.position.xy));
    // We're outputting a 2x2 image, each pixel coming from a different source
    let serial = floored.x + floored.y * 2u;

    switch serial {
        // (0, 0) - uniform buffer from the vertex shader
        case 0u: {
            return fs_in.data1;
        }
        // (1, 0) - push constant from the vertex shader
        case 1u: {
            return fs_in.data2;
        }
        // (0, 1) - uniform buffer from the fragment shader
        case 2u: {
            return vec4f(data1.a, data1.b, data1.c, data1.d);
        }
        // (1, 1) - push constant from the fragment shader
        case 3u: {
            return vec4f(data2.a, data2.b, data2.c, data2.d);
        }
        default: {
            return vec4f(0.0);
        }
    }
}
