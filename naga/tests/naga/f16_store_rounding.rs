/*! `FPRoundingMode RTE` decorations on direct f16 storage narrowings. */

#![cfg(all(feature = "wgsl-in", spv_out))]

use naga::back::spv;
use naga::proc::BoundsCheckPolicy;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use rspirv::dr::{load_words, Operand};
use rspirv::spirv;

#[derive(Debug)]
struct NarrowScan {
    narrows: Vec<u32>,
    fp_rounding_decorates: Vec<(u32, spirv::FPRoundingMode)>,
}

fn scan(words: &[u32]) -> NarrowScan {
    let module = load_words(words).expect("Produced invalid SPIR-V");
    let bound = module.header.as_ref().unwrap().bound as usize;
    let mut float_widths = vec![0u32; bound];
    let mut vec_components = vec![0u32; bound];
    for inst in &module.types_global_values {
        match (inst.class.opcode, inst.result_id, inst.operands.as_slice()) {
            (spirv::Op::TypeFloat, Some(id), [Operand::LiteralBit32(width)]) => {
                float_widths[id as usize] = *width;
            }
            (spirv::Op::TypeVector, Some(id), [Operand::IdRef(comp), ..]) => {
                vec_components[id as usize] = *comp;
            }
            _ => {}
        }
    }

    let mut narrows = Vec::new();
    for func in &module.functions {
        for block in &func.blocks {
            for inst in &block.instructions {
                if inst.class.opcode == spirv::Op::FConvert {
                    let mut ty = inst.result_type.unwrap() as usize;
                    while vec_components[ty] != 0 {
                        ty = vec_components[ty] as usize;
                    }
                    if float_widths[ty] == 16 {
                        narrows.push(inst.result_id.unwrap());
                    }
                }
            }
        }
    }

    let mut fp_rounding_decorates = Vec::new();
    for inst in &module.annotations {
        if inst.class.opcode == spirv::Op::Decorate {
            if let [Operand::IdRef(target), Operand::Decoration(spirv::Decoration::FPRoundingMode), Operand::FPRoundingMode(mode)] =
                inst.operands.as_slice()
            {
                fp_rounding_decorates.push((*target, *mode));
            }
        }
    }

    NarrowScan {
        narrows,
        fp_rounding_decorates,
    }
}

fn emit_words(case: &Case) -> Vec<u32> {
    let module = naga::front::wgsl::parse_str(case.wgsl).unwrap();
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap();
    let options = spv::Options {
        lang_version: case.lang_version,
        flags: case.flags,
        bounds_check_policies: naga::proc::BoundsCheckPolicies {
            buffer: case.buffer,
            ..Default::default()
        },
        ..Default::default()
    };
    spv::write_vec(&module, &info, &options, None).unwrap()
}

fn validate_spirv(words: &[u32]) {
    if cfg!(miri) {
        return;
    }

    use std::io::Write as _;
    use std::process::{Command, Stdio};

    // `vulkan1.2` accepts both SPIR-V 1.0 + `SPV_KHR_16bit_storage` and the
    // SPIR-V 1.3 case (which `vulkan1.0` would reject on version alone).
    let mut child = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.2", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect(
            "Failed to execute spirv-val. It can be installed \
            by installing the Vulkan SDK and adding it to your path.",
        );

    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&bytes)
        .expect("failed to write SPIR-V module to spirv-val stdin");

    let output = child
        .wait_with_output()
        .expect("failed to wait for spirv-val");
    assert!(
        output.status.success(),
        "spirv-val rejected naga output:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[derive(Clone, Copy, Debug)]
enum Expectation {
    Decorated,
    UndecoratedNarrow,
    NoF16Narrow,
}

#[derive(Debug)]
struct Case {
    name: &'static str,
    wgsl: &'static str,
    lang_version: (u8, u8),
    buffer: BoundsCheckPolicy,
    flags: spv::WriterFlags,
    expected: Expectation,
}

impl Case {
    const fn new(name: &'static str, wgsl: &'static str, expected: Expectation) -> Self {
        Self {
            name,
            wgsl,
            lang_version: (1, 0),
            buffer: BoundsCheckPolicy::Unchecked,
            flags: spv::WriterFlags::empty(),
            expected,
        }
    }

    const fn with_lang_version(mut self, lang_version: (u8, u8)) -> Self {
        self.lang_version = lang_version;
        self
    }

    const fn with_buffer_policy(mut self, buffer: BoundsCheckPolicy) -> Self {
        self.buffer = buffer;
        self
    }

    const fn with_flags(mut self, flags: spv::WriterFlags) -> Self {
        self.flags = flags;
        self
    }
}

fn check_cases(cases: &[Case]) {
    for case in cases {
        let words = emit_words(case);
        let scan = scan(&words);
        match case.expected {
            Expectation::Decorated => {
                assert!(
                    !scan.narrows.is_empty(),
                    "{}: no f16 narrowing was emitted",
                    case.name,
                );
                assert_eq!(
                    scan.fp_rounding_decorates.len(),
                    scan.narrows.len(),
                    "{}: FPRoundingMode decoration count mismatch: {:?} vs narrows {:?}",
                    case.name,
                    scan.fp_rounding_decorates,
                    scan.narrows,
                );
                for &id in &scan.narrows {
                    assert!(
                        scan.fp_rounding_decorates
                            .contains(&(id, spirv::FPRoundingMode::RTE)),
                        "{}: narrowing %{id} missing `FPRoundingMode RTE` in {:?}",
                        case.name,
                        scan.fp_rounding_decorates,
                    );
                }
            }
            Expectation::UndecoratedNarrow => {
                assert!(
                    !scan.narrows.is_empty(),
                    "{}: no f16 narrowing was emitted",
                    case.name,
                );
                assert!(
                    scan.fp_rounding_decorates.is_empty(),
                    "{}: unexpected FPRoundingMode decorations: {:?}",
                    case.name,
                    scan.fp_rounding_decorates,
                );
            }
            Expectation::NoF16Narrow => {
                assert!(
                    scan.narrows.is_empty(),
                    "{}: unexpected f16 narrowings: {:?}",
                    case.name,
                    scan.narrows,
                );
                assert!(
                    scan.fp_rounding_decorates.is_empty(),
                    "{}: unexpected FPRoundingMode decorations: {:?}",
                    case.name,
                    scan.fp_rounding_decorates,
                );
            }
        }
        validate_spirv(&words);
    }
}

const STORAGE_SCALAR: &str = "
enable f16;
@group(0) @binding(0) var<storage, read> i: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f16>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    o[g.x] = f16(i[g.x]);
}
";

const STORAGE_VEC_COMPONENT: &str = "
enable f16;
@group(0) @binding(0) var<storage, read> i: array<f32>;
@group(0) @binding(1) var<storage, read_write> ov: array<vec4<f16>>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    ov[g.x][g.y] = f16(i[g.x]);
}
";

#[test]
fn direct_storage_narrowings_are_decorated() {
    check_cases(&[
        Case::new(
            "f32 scalar unchecked",
            STORAGE_SCALAR,
            Expectation::Decorated,
        ),
        Case::new(
            "f32 scalar SPIR-V 1.3",
            STORAGE_SCALAR,
            Expectation::Decorated,
        )
        .with_lang_version((1, 3)),
        Case::new(
            "f32 scalar bounds-checked",
            STORAGE_SCALAR,
            Expectation::Decorated,
        )
        .with_buffer_policy(BoundsCheckPolicy::ReadZeroSkipWrite),
        Case::new(
            "f32 vec2 unchecked",
            "
enable f16;
@group(0) @binding(0) var<storage, read> v: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> ov: array<vec2<f16>>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    ov[g.x] = vec2<f16>(v[g.x]);
}
",
            Expectation::Decorated,
        ),
        Case::new(
            "f64 scalar unchecked",
            "
enable f16;
enable wgpu_f64;
@group(0) @binding(0) var<storage, read> d: array<f64>;
@group(0) @binding(1) var<storage, read_write> o: array<f16>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    o[g.x] = f16(d[g.x]);
}
",
            Expectation::Decorated,
        ),
        Case::new(
            "f64 vec2 unchecked",
            "
enable f16;
enable wgpu_f64;
@group(0) @binding(0) var<storage, read> dv: array<vec2<f64>>;
@group(0) @binding(1) var<storage, read_write> ov: array<vec2<f16>>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    ov[g.x] = vec2<f16>(dv[g.x]);
}
",
            Expectation::Decorated,
        ),
        Case::new(
            "vec4 component ValuePointer unchecked",
            STORAGE_VEC_COMPONENT,
            Expectation::Decorated,
        ),
        Case::new(
            "vec4 component ValuePointer bounds-checked",
            STORAGE_VEC_COMPONENT,
            Expectation::Decorated,
        )
        .with_buffer_policy(BoundsCheckPolicy::ReadZeroSkipWrite),
        Case::new(
            "single-use let with DEBUG flags",
            "
enable f16;
@group(0) @binding(0) var<storage, read> i: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f16>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let narrowed = f16(i[g.x]);
    o[g.x] = narrowed;
}
",
            Expectation::Decorated,
        )
        .with_flags(spv::WriterFlags::DEBUG),
        Case::new(
            "helper function let storage pointer",
            "
enable f16;
@group(0) @binding(0) var<storage, read> i: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f16>;
fn store_item(idx: u32, x: f32) {
    let p: ptr<storage, f16, read_write> = &o[idx];
    *p = f16(x);
}
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    store_item(g.x, i[g.x]);
}
",
            Expectation::Decorated,
        ),
    ]);
}

#[test]
fn non_qualifying_f16_narrowings_are_not_decorated() {
    check_cases(&[
        Case::new(
            "reused narrowing",
            "
enable f16;
@group(0) @binding(0) var<storage, read> i: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f16>;
@compute @workgroup_size(1)
fn main() {
    let h = f16(i[0]);
    o[0] = h;
    o[1] = h;
}
",
            Expectation::UndecoratedNarrow,
        ),
        Case::new(
            "narrowing with arithmetic use",
            "
enable f16;
@group(0) @binding(0) var<storage, read> i: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f16>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    o[g.x] = f16(i[g.x]) + f16(1.0);
}
",
            Expectation::UndecoratedNarrow,
        ),
        Case::new(
            "workgroup store",
            "
enable f16;
var<workgroup> t: f16;
@group(0) @binding(0) var<storage, read> i: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f16>;
@compute @workgroup_size(1)
fn main() {
    t = f16(i[0]);
    o[0] = t;
}
",
            Expectation::UndecoratedNarrow,
        ),
        Case::new(
            "private store",
            "
enable f16;
var<private> p: f16;
@group(0) @binding(0) var<storage, read> i: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f16>;
@compute @workgroup_size(1)
fn main() {
    p = f16(i[0]);
    o[0] = p;
}
",
            Expectation::UndecoratedNarrow,
        ),
        Case::new(
            "function local store",
            "
enable f16;
@group(0) @binding(0) var<storage, read> i: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f16>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    var h = f16(i[g.x]);
    o[g.x] = h;
}
",
            Expectation::UndecoratedNarrow,
        ),
        // `f16(h)` / `vec2<f16>(h)` alias `h`'s SSA ID in `write_as_expression`,
        // so the store must not decorate `h` when `h` is also used in arithmetic.
        Case::new(
            "scalar identity cast of reused narrowing",
            "
enable f16;
@group(0) @binding(0) var<storage, read> i: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f16>;
@compute @workgroup_size(1)
fn main() {
    let h = f16(i[0]);
    o[0] = f16(h);
    o[1] = h * 2.0h;
}
",
            Expectation::UndecoratedNarrow,
        ),
        Case::new(
            "vector identity cast of reused narrowing",
            "
enable f16;
@group(0) @binding(0) var<storage, read> v: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> ov: array<vec2<f16>>;
@compute @workgroup_size(1)
fn main() {
    let h = vec2<f16>(v[0]);
    ov[0] = vec2<f16>(h);
    ov[1] = h * vec2<f16>(2.0h);
}
",
            Expectation::UndecoratedNarrow,
        ),
        Case::new(
            "per-component vector compose of f16 narrows",
            "
enable f16;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> ov: array<vec2<f16>>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    ov[g.x] = vec2<f16>(f16(a[g.x]), f16(b[g.x]));
}
",
            Expectation::UndecoratedNarrow,
        ),
        Case::new(
            "matrix f32 to f16 storage store",
            "
enable f16;
@group(0) @binding(0) var<storage, read> im: array<mat2x2<f32>>;
@group(0) @binding(1) var<storage, read_write> om: array<mat2x2<f16>>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    om[g.x] = mat2x2<f16>(im[g.x]);
}
",
            Expectation::UndecoratedNarrow,
        ),
        Case::new(
            "emitted narrowing with discarded tail store",
            "
enable f16;
@group(0) @binding(0) var<storage, read> i: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f16>;
@compute @workgroup_size(1)
fn main() {
    let h = f16(i[0]);
    {
        return;
    }
    o[0] = h;
}
",
            Expectation::UndecoratedNarrow,
        ),
    ]);
}

#[test]
fn non_f16_narrowing_conversions_are_not_decorated() {
    check_cases(&[
        Case::new(
            "f64 and f16 to f32 conversions",
            "
enable f16;
enable wgpu_f64;
@group(0) @binding(0) var<storage, read> d: array<f64>;
@group(0) @binding(1) var<storage, read> h: array<f16>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    o[g.x * 2u] = f32(d[g.x]);
    o[g.x * 2u + 1u] = f32(h[g.x]);
}
",
            Expectation::NoF16Narrow,
        ),
        Case::new(
            "integer to f16 conversions",
            "
enable f16;
@group(0) @binding(0) var<storage, read> si: array<i32>;
@group(0) @binding(1) var<storage, read> ui: array<u32>;
@group(0) @binding(2) var<storage, read> sv: array<vec2<i32>>;
@group(0) @binding(3) var<storage, read_write> o: array<f16>;
@group(0) @binding(4) var<storage, read_write> ov: array<vec2<f16>>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    o[g.x * 2u] = f16(si[g.x]);
    o[g.x * 2u + 1u] = f16(ui[g.x]);
    ov[g.x] = vec2<f16>(sv[g.x]);
}
",
            Expectation::NoF16Narrow,
        ),
        Case::new(
            "float to 16-bit integer conversions",
            "
enable wgpu_int16;
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> oi: array<i16>;
@group(0) @binding(3) var<storage, read_write> ou: array<u16>;
@group(0) @binding(4) var<storage, read_write> ov: array<vec2<i16>>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    oi[g.x] = i16(x[g.x]);
    ou[g.x] = u16(x[g.x]);
    ov[g.x] = vec2<i16>(v[g.x]);
}
",
            Expectation::NoF16Narrow,
        ),
        Case::new(
            "unreachable store after return",
            "
enable f16;
@group(0) @binding(0) var<storage, read> i: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f16>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    return;
    o[g.x] = f16(i[g.x]);
}
",
            Expectation::NoF16Narrow,
        ),
    ]);
}
