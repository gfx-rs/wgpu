#![cfg(all(feature = "wgsl-in", feature = "wgsl-out", spv_out))]

use naga::{
    back::spv,
    valid::{Capabilities, ValidationFlags, Validator},
};
use rspirv::binary::Disassemble;

const CONVERT: &str = r#"
enable wgpu_bfloat16;
@group(0) @binding(0) var<storage> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed: array<bf16>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
const tie_even = bf16(1.00390625f);
@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = bf16(src[id.x]);
    packed[id.x] = x;
    dst[id.x] = f32(x) + f32(tie_even);
}
"#;

const PRODUCT: &str = r#"
enable wgpu_bfloat16;
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage> src: array<bf16>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(32)
fn main() {
    let a = coopLoadT<coop_mat16x16<bf16, A>>(&src[0], 16u);
    let b = coopLoadT<coop_mat16x16<bf16, B>>(&src[256], 16u);
    let c = coop_mat16x16<f32, C>();
    let d = coopMultiplyAdd(a, b, c);
    coopStoreT(d, &dst[0], 16u);
}
"#;

const TEMPORARIES: &str = r#"
enable wgpu_bfloat16;
@group(0) @binding(0) var<storage> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
var<workgroup> tile: array<bf16, 32>;
fn narrow(v: f32) -> bf16 { return bf16(v); }
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) index: u32) {
    var temporary = vec2<bf16>(narrow(src[index]), bf16(0));
    tile[index] = select(temporary.x, temporary.y, index == 0u);
    workgroupBarrier();
    dst[index] = f32(tile[(index + 1u) % 32u]);
}
"#;

fn validate(source: &str, caps: Capabilities) -> (naga::Module, naga::valid::ModuleInfo) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
    let info = Validator::new(ValidationFlags::all(), caps)
        .validate(&module)
        .unwrap();
    (module, info)
}

#[test]
fn storage_conversion_and_cooperative_product() {
    for (name, source) in [
        ("convert", CONVERT),
        ("product", PRODUCT),
        ("temporaries", TEMPORARIES),
    ] {
        let (module, info) = validate(source, Capabilities::all());
        let options = spv::Options {
            lang_version: (1, 3),
            ..Default::default()
        };
        let words = spv::write_vec(&module, &info, &options, None).unwrap();
        let assembly = rspirv::dr::load_words(&words).unwrap().disassemble();
        assert!(
            assembly.contains("OpTypeFloat 16 BFloat16KHR"),
            "{assembly}"
        );
        assert!(assembly.contains("OpCapability BFloat16TypeKHR"));
        assert!(!assembly.contains("OpCapability Float16\n"));
        assert!(!assembly.contains("BFloat16DotProductKHR"));
        if name == "convert" {
            assert_eq!(assembly.matches("OpFConvert").count(), 2);
            let value = module
                .constants
                .iter()
                .find(|(_, c)| c.name.as_deref() == Some("tie_even"))
                .unwrap()
                .1;
            assert!(
                matches!(module.global_expressions[value.init], naga::Expression::Literal(naga::Literal::BF16(v)) if v.to_bits() == 0x3f80)
            );
        } else if name == "product" {
            assert!(assembly.contains("OpCapability BFloat16CooperativeMatrixKHR"));
            assert!(assembly.contains("OpCooperativeMatrixMulAddKHR"));
        }
        let wgsl =
            naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
                .unwrap();
        validate(&wgsl, Capabilities::all());
        if let Some(directory) = std::env::var_os("NAGA_BF16_SPV_DIR") {
            let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
            std::fs::write(
                std::path::Path::new(&directory).join(format!("{name}.spv")),
                bytes,
            )
            .unwrap();
        }
    }
    let mixed = PRODUCT.replace(
        "let b = coopLoadT<coop_mat16x16<bf16, B>>(&src[256], 16u);",
        "let b = coop_mat16x16<f32, B>();",
    );
    let module = naga::front::wgsl::parse_str(&mixed).unwrap();
    assert!(Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .is_err());
}

#[test]
fn capabilities_and_unsupported_operations_are_rejected() {
    assert!(naga::front::wgsl::parse_str(&CONVERT.replace("enable wgpu_bfloat16;", "")).is_err());
    validate(
        PRODUCT,
        Capabilities::SHADER_BFLOAT16 | Capabilities::COOPERATIVE_MATRIX,
    );
    for (source, missing) in [
        (CONVERT, Capabilities::SHADER_BFLOAT16),
        (PRODUCT, Capabilities::SHADER_BFLOAT16),
        (PRODUCT, Capabilities::COOPERATIVE_MATRIX),
    ] {
        let module = naga::front::wgsl::parse_str(source).unwrap();
        assert!(
            Validator::new(ValidationFlags::all(), Capabilities::all() - missing)
                .validate(&module)
                .is_err(),
            "accepted without {missing:?}"
        );
    }
    for expr in [
        "x + x",
        "x - x",
        "x * x",
        "x / x",
        "x % x",
        "-x",
        "abs(x)",
        "min(x, x)",
        "sqrt(x)",
        "bf16(1u)",
        "bitcast<bf16>(1u)",
    ] {
        let source = CONVERT.replace("packed[id.x] = x;", &format!("packed[id.x] = {expr};"));
        if let Ok(module) = naga::front::wgsl::parse_str(&source) {
            assert!(
                Validator::new(ValidationFlags::all(), Capabilities::all())
                    .validate(&module)
                    .is_err(),
                "accepted {expr}"
            );
        }
    }
    for expr in ["x == x", "x < x", "isNan(x)"] {
        let source = CONVERT.replace(
            "packed[id.x] = x;",
            &format!("if {expr} {{ packed[id.x] = x; }}"),
        );
        if let Ok(module) = naga::front::wgsl::parse_str(&source) {
            assert!(
                Validator::new(ValidationFlags::all(), Capabilities::all())
                    .validate(&module)
                    .is_err(),
                "accepted {expr}"
            );
        }
    }
    let source = PRODUCT.replace("coopMultiplyAdd(a, b, c)", "a + a");
    if let Ok(module) = naga::front::wgsl::parse_str(&source) {
        assert!(Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .is_err());
    }
}

/// Matrices that only come from loads leave no cooperative matrix type in the arena.
#[test]
fn loaded_cooperative_matrices_use_vulkan_memory_model() {
    let source = PRODUCT.replace(
        "let c = coop_mat16x16<f32, C>();",
        "let c = coopLoadT<coop_mat16x16<f32, C>>(&dst[0], 16u);",
    );
    let (module, info) = validate(&source, Capabilities::all());
    let options = spv::Options {
        lang_version: (1, 3),
        ..Default::default()
    };
    let words = spv::write_vec(&module, &info, &options, None).unwrap();
    let assembly = rspirv::dr::load_words(&words).unwrap().disassemble();
    assert!(
        assembly.contains("OpCapability VulkanMemoryModel"),
        "{assembly}"
    );
    assert!(
        assembly.contains("OpMemoryModel Logical Vulkan"),
        "{assembly}"
    );
}

#[test]
fn spirv_1_0_declares_16bit_storage() {
    let (module, info) = validate(CONVERT, Capabilities::all());
    let words = spv::write_vec(&module, &info, &spv::Options::default(), None).unwrap();
    let assembly = rspirv::dr::load_words(&words).unwrap().disassemble();
    assert!(
        assembly.contains("OpExtension \"SPV_KHR_16bit_storage\""),
        "{assembly}"
    );
}

#[cfg(msl_out)]
#[test]
fn msl_bf16_requires_3_1() {
    let source = r#"
enable wgpu_bfloat16;
@group(0) @binding(0) var<storage> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    dst[id.x] = f32(bf16(src[id.x]));
}
"#;
    let product = PRODUCT
        .replace("16x16", "8x8")
        .replace("16u", "8u")
        .replace("256", "64");
    for source in [source, CONVERT, TEMPORARIES, &product] {
        let (module, info) = validate(source, Capabilities::all());
        for version in [(2, 4), (3, 0), (3, 1)] {
            let options = naga::back::msl::Options {
                lang_version: version,
                ..Default::default()
            };
            let result =
                naga::back::msl::write_string(&module, &info, &options, &Default::default());
            if version < (3, 1) {
                assert!(matches!(
                    result,
                    Err(naga::back::msl::Error::CapabilityNotSupported(
                        Capabilities::SHADER_BFLOAT16
                    ))
                ));
            } else {
                let (msl, info) = result.unwrap();
                assert!(info.entry_point_names.iter().all(Result::is_ok));
                assert!(msl.contains("bfloat"));
                if source == product {
                    assert!(msl.contains("simdgroup_bfloat8x8"));
                    assert!(msl.contains("simdgroup_float8x8"));
                    assert!(msl.contains("simdgroup_multiply_accumulate"));
                }
            }
        }
    }
}

#[test]
fn abstract_numbers_convert_to_bf16() {
    let source = CONVERT.replace(
        "packed[id.x] = x;",
        "let one: bf16 = 1.0;
    packed[id.x] = select(vec2<bf16>(x, one), vec2<bf16>(0.5, 2), id.x == 0u).y;",
    );
    validate(&source, Capabilities::all());
}
