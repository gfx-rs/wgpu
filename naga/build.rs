fn main() {
    cfg_aliases::cfg_aliases! {
        dot_out: { feature = "dot-out" },
        glsl_out: { feature = "glsl-out" },
        hlsl_out: { feature = "hlsl-out" },
        msl_out: { feature = "msl-out" },
        spv_out: { feature = "spv-out" },
        wgsl_out: { feature = "wgsl-out" },
    }
}
