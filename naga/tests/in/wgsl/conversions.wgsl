// Conversion of initializer expressions
const ic00: vec2<u32> = vec2();
const ic01: vec4i = vec4();
const ic02: vec4i = vec4(1);
const ic03: vec4u = vec4();
const ic04: vec4u = vec4(1);
const ic05: vec4f = vec4();
const ic06: vec4f = vec4(1);
const ic07: vec2i = vec2(1, 1);
const ic08: vec3i = vec3(1, 1, 1);
const ic09: vec4i = vec4(1, 1, 1, 1);
const ic10: vec2u = vec2(1, 1);
const ic11: vec3u = vec3(1, 1, 1);
const ic12: vec4u = vec4(1, 1, 1, 1);
const ic13: vec2f = vec2(1, 1);
const ic14: vec3f = vec3(1, 1, 1);
const ic15: vec4f = vec4(1, 1, 1, 1);
const ic16: vec2f = vec2(1.0, 1.0);
const ic17: vec3f = vec3(1.0, 1.0, 1.0);
const ic18: vec4f = vec4(1.0, 1.0, 1.0, 1.0);
const ic19: vec2f = vec2(1, 1) + vec2(1.0, 1.0);
const ic20: mat2x2f = mat2x2(vec2(), vec2());
const ic21: array<u32, 4> = array(1, 2, 3, 4);

// Conversion by value constructors
//let vc0 = i32(1.0);  // https://github.com/gfx-rs/wgpu/issues/7312
// etc. (also create the locals versions below)

@compute @workgroup_size(1)
fn main() {
    const ic00: vec2<u32> = vec2();
    const ic01: vec4i = vec4();
    const ic02: vec4i = vec4(1);
    const ic03: vec4u = vec4();
    const ic04: vec4u = vec4(1);
    const ic05: vec4f = vec4();
    const ic06: vec4f = vec4(1);
    const ic07: vec2i = vec2(1, 1);
    const ic08: vec3i = vec3(1, 1, 1);
    const ic09: vec4i = vec4(1, 1, 1, 1);
    const ic10: vec2u = vec2(1, 1);
    const ic11: vec3u = vec3(1, 1, 1);
    const ic12: vec4u = vec4(1, 1, 1, 1);
    const ic13: vec2f = vec2(1, 1);
    const ic14: vec3f = vec3(1, 1, 1);
    const ic15: vec4f = vec4(1, 1, 1, 1);
    const ic16: vec2f = vec2(1.0, 1.0);
    const ic17: vec3f = vec3(1.0, 1.0, 1.0);
    const ic18: vec4f = vec4(1.0, 1.0, 1.0, 1.0);
    const ic19: vec2f = vec2(1, 1) + vec2(1.0, 1.0);
    const ic20: mat2x2f = mat2x2(vec2(), vec2());
    const ic21: array<u32, 4> = array(1, 2, 3, 4);

    let lc00: vec2<u32> = vec2();
    let lc01: vec4i = vec4();
    let lc02: vec4i = vec4(1);
    let lc03: vec4u = vec4();
    let lc04: vec4u = vec4(1);
    let lc05: vec4f = vec4();
    let lc06: vec4f = vec4(1);
    let lc07: vec2i = vec2(1, 1);
    let lc08: vec3i = vec3(1, 1, 1);
    let lc09: vec4i = vec4(1, 1, 1, 1);
    let lc10: vec2u = vec2(1, 1);
    let lc11: vec3u = vec3(1, 1, 1);
    let lc12: vec4u = vec4(1, 1, 1, 1);
    let lc13: vec2f = vec2(1, 1);
    let lc14: vec3f = vec3(1, 1, 1);
    let lc15: vec4f = vec4(1, 1, 1, 1);
    let lc16: vec2f = vec2(1.0, 1.0);
    let lc17: vec3f = vec3(1.0, 1.0, 1.0);
    let lc18: vec4f = vec4(1.0, 1.0, 1.0, 1.0);
    let lc19: vec2f = vec2(1, 1) + vec2(1.0, 1.0);
    let lc20: mat2x2f = mat2x2(vec2(), vec2());
    let lc21: array<u32, 4> = array(1, 2, 3, 4);

    var vc00: vec2<u32> = vec2();
    var vc01: vec4i = vec4();
    var vc02: vec4i = vec4(1);
    var vc03: vec4u = vec4();
    var vc04: vec4u = vec4(1);
    var vc05: vec4f = vec4();
    var vc06: vec4f = vec4(1);
    var vc07: vec2i = vec2(1, 1);
    var vc08: vec3i = vec3(1, 1, 1);
    var vc09: vec4i = vec4(1, 1, 1, 1);
    var vc10: vec2u = vec2(1, 1);
    var vc11: vec3u = vec3(1, 1, 1);
    var vc12: vec4u = vec4(1, 1, 1, 1);
    var vc13: vec2f = vec2(1, 1);
    var vc14: vec3f = vec3(1, 1, 1);
    var vc15: vec4f = vec4(1, 1, 1, 1);
    var vc16: vec2f = vec2(1.0, 1.0);
    var vc17: vec3f = vec3(1.0, 1.0, 1.0);
    var vc18: vec4f = vec4(1.0, 1.0, 1.0, 1.0);
    var vc19: vec2f = vec2(1, 1) + vec2(1.0, 1.0);
    var vc20: mat2x2f = mat2x2(vec2(), vec2());
    var vc21: array<u32, 4> = array(1, 2, 3, 4);
}
