// #9489: A `var` declaration inside a loop body without an explicit
// initializer must be re-zero-initialized on every iteration, just like
// one with an explicit initializer. Naga hoists all local variables to
// function scope and zero-initializes them once at function entry, so the
// per-iteration reset has to be lowered to an explicit store in the loop
// body (the same mechanism already used for explicit initializers).
//
// Without the fix, `acc_noinit` accumulates across iterations of the outer
// loop (a running prefix sum) while `acc_init` does not, even though the
// two are semantically equivalent per the WGSL spec.

@group(0) @binding(0) var<storage, read> input: array<f32, 64>;
@group(0) @binding(1) var<storage, read_write> output: array<f32, 8>;

@compute @workgroup_size(1)
fn main() {
    for (var t = 0u; t < 4u; t++) {
        var acc_noinit: vec4<f32>;                // no explicit initializer
        var acc_init: vec4<f32> = vec4<f32>();    // explicit initializer

        for (var d = 0u; d < 16u; d++) {
            let v = vec4<f32>(input[t * 16u + d]);
            acc_noinit += v;
            acc_init += v;
        }

        output[t * 2u] = acc_noinit.x;
        output[t * 2u + 1u] = acc_init.x;
    }
}
