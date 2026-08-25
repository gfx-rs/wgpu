struct Params {
    // 0 = sRGB SDR, 1 = extended linear scRGB, 2 = HDR10 PQ, 3 = HLG,
    // 4 = encoded extended-range sRGB, 5 = encoded extended-range Display-P3
    mode: u32,
    // 1 if the shader must apply the sRGB OETF itself (non-sRGB SDR format)
    encode_srgb: u32,
}

@group(0) @binding(0) var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // Fullscreen triangle. Constructed in a single expression: a local
    // `var out: VertexOutput` would hit the same naga ArrayStride/Offset
    // issue described above (structs get explicit member offsets).
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    return VertexOutput(vec4f(x, y, 0.0, 1.0), vec2f(x, -y) * 0.5 + 0.5);
}

// Note: indexed lookups are written as if-chains rather than local
// `array<...>` values, because naga's SPIR-V backend decorates array types
// with ArrayStride, which is invalid on function-local variables
// (VUID-StandaloneSpirv-None-10684; see
// https://github.com/gfx-rs/wgpu/issues/7696).

fn staircase_nits(i: u32) -> f32 {
    if i == 0u { return 50.0; }
    if i == 1u { return 100.0; }
    if i == 2u { return 203.0; }
    if i == 3u { return 400.0; }
    if i == 4u { return 1000.0; }
    return 10000.0;
}

fn primary_secondary(i: u32) -> vec3f {
    if i == 0u { return vec3f(1.0, 0.0, 0.0); }
    if i == 1u { return vec3f(0.0, 1.0, 0.0); }
    if i == 2u { return vec3f(0.0, 0.0, 1.0); }
    if i == 3u { return vec3f(0.0, 1.0, 1.0); }
    if i == 4u { return vec3f(1.0, 0.0, 1.0); }
    return vec3f(1.0, 1.0, 0.0);
}

// The test pattern, in linear BT.709 with absolute luminance in nits.
fn pattern_nits(uv: vec2f) -> vec3f {
    let i = min(u32(uv.x * 6.0), 5u);
    if uv.y < 0.3333 {
        // Grayscale staircase.
        return vec3f(staircase_nits(i));
    } else if uv.y < 0.6667 {
        // BT.709 primaries and secondaries at 203 nits.
        return primary_secondary(i) * 203.0;
    } else {
        // Log gradient, 1 nit -> 10000 nits.
        return vec3f(pow(10.0, uv.x * 4.0));
    }
}

// Standard sRGB OETF, valid for inputs in [0, 1]. `pow` is undefined for
// negative inputs, so callers must clamp first (the SDR path does).
fn srgb_oetf(c: vec3f) -> vec3f {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3f(0.0031308));
}

// Extended sRGB OETF: the sRGB transfer function continued beyond [0, 1] with
// odd (point) symmetry through the origin, so values >1.0 (brighter than SDR
// reference white) and <0.0 (out-of-gamut) are encoded rather than clamped.
// This is what the `ExtendedSrgb` color space (browser HDR canvas, Vulkan
// EXTENDED_SRGB_NONLINEAR, Metal ExtendedSRGB) expects on the wire.
fn srgb_oetf_extended(c: vec3f) -> vec3f {
    let s = sign(c);
    let a = abs(c);
    let lo = a * 12.92;
    let hi = 1.055 * pow(a, vec3f(1.0 / 2.4)) - 0.055;
    return s * select(hi, lo, a <= vec3f(0.0031308));
}

// SMPTE ST 2084 (PQ) OETF; input is luminance normalized to 10000 nits.
fn pq_oetf(y: vec3f) -> vec3f {
    let m1 = 0.1593017578125;
    let m2 = 78.84375;
    let c1 = 0.8359375;
    let c2 = 18.8515625;
    let c3 = 18.6875;
    let yp = pow(max(y, vec3f(0.0)), vec3f(m1));
    return pow((c1 + c2 * yp) / (1.0 + c3 * yp), vec3f(m2));
}

// BT.2100 HLG OETF; input is scene luminance normalized to 1000-nit peak.
fn hlg_oetf(y: vec3f) -> vec3f {
    let a = 0.17883277;
    let b = 0.28466892;
    let c = 0.55991073;
    let lo = sqrt(3.0 * y);
    let hi = a * log(12.0 * y - b) + c;
    return select(hi, lo, y <= vec3f(1.0 / 12.0));
}

const BT709_TO_BT2020 = mat3x3f(
    vec3f(0.627402, 0.069095, 0.016394),
    vec3f(0.329292, 0.919544, 0.088028),
    vec3f(0.043306, 0.011360, 0.895578),
);

// Linear BT.709 (sRGB) -> linear Display-P3, both D65. Column-major.
const BT709_TO_DISPLAYP3 = mat3x3f(
    vec3f(0.8224621, 0.0331942, 0.0170826),
    vec3f(0.1775380, 0.9668058, 0.0723974),
    vec3f(0.0000000, 0.0000000, 0.9105199),
);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let nits = pattern_nits(in.uv);

    var out: vec3f;
    switch params.mode {
        case 1u: {
            // Extended linear scRGB: BT.709 primaries, linear, 1.0 = 80 nits.
            out = nits / 80.0;
        }
        case 4u: {
            // Encoded extended-range sRGB: BT.709 primaries, the sRGB OETF
            // extended beyond [0, 1]. Same normalization as scRGB
            // (1.0 = 80 nits), but sRGB-encoded rather than linear.
            out = srgb_oetf_extended(nits / 80.0);
        }
        case 5u: {
            // Encoded extended-range Display-P3: convert BT.709 -> P3 primaries
            // (linear), normalize like scRGB (1.0 = 80 nits), then apply the
            // extended sRGB OETF. The BT.709 test primaries look identical, just
            // carried in the wider P3 container.
            out = srgb_oetf_extended((BT709_TO_DISPLAYP3 * nits) / 80.0);
        }
        case 2u: {
            // HDR10: BT.2020 primaries, PQ-encoded absolute luminance.
            out = pq_oetf(BT709_TO_BT2020 * (nits / 10000.0));
        }
        case 3u: {
            // HLG: BT.2020 primaries, 1000-nit nominal peak.
            out = hlg_oetf(BT709_TO_BT2020 * (min(nits, vec3f(1000.0)) / 1000.0));
        }
        default: {
            // SDR sRGB: clip at 100 nits.
            out = clamp(nits / 100.0, vec3f(0.0), vec3f(1.0));
            if params.encode_srgb == 1u {
                out = srgb_oetf(out);
            }
        }
    }
    return vec4f(out, 1.0);
}
