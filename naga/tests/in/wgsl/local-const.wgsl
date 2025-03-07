const ga = 4;                  // AbstractInt with a value of 4.
const gb : i32 = 4;            // i32 with a value of 4.
const gc : u32 = 4;            // u32 with a value of 4.
const gd : f32 = 4;            // f32 with a value of 4.
const ge = vec3(ga, ga, ga);   // vec3 of AbstractInt with a value of (4, 4, 4).
const gf = 2.0;                // AbstractFloat with a value of 2.

fn const_in_fn() {
    const a = 4;                // AbstractInt with a value of 4.
    const b: i32 = 4;           // i32 with a value of 4.
    const c: u32 = 4;           // u32 with a value of 4.
    const d: f32 = 4;           // f32 with a value of 4.
    const e = vec3(a, a, a);    // vec3 of AbstractInt with a value of (4, 4, 4).
    const f = 2.0;              // AbstractFloat with a value of 2.
    // TODO: Make it per spec, currently not possible
    // because naga does not support automatic conversions
    // of Abstract types

    // Check that we can access global constants
    const ag = ga;
    const bg = gb;
    const cg = gc;
    const dg = gd;
    const eg = ge;
    const fg = gf;
}
