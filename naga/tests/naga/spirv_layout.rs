#![cfg(all(feature = "wgsl-in", spv_out))]

use naga::back::spv::{self, ZeroInitializeWorkgroupMemoryMode as Init};
use std::{
    io::Write as _,
    process::{Command, Stdio},
};

fn check(version: (u8, u8)) {
    let source = include_str!("../in/wgsl/workgroup-shared-layout.wgsl");
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    for mode in [Init::None, Init::Polyfill, Init::Native] {
        let options = spv::Options {
            lang_version: version,
            zero_initialize_workgroup_memory: mode,
            ..Default::default()
        };
        let words = spv::write_vec(&module, &info, &options, None).unwrap();
        validate(&words, version, mode);
        check_buffer_abi(&words);
    }
}

#[test]
fn shared_layout_10() {
    check((1, 0));
}
#[test]
fn shared_layout_11() {
    check((1, 1));
}
#[test]
fn shared_layout_12() {
    check((1, 2));
}
#[test]
fn shared_layout_13() {
    check((1, 3));
}
#[test]
fn shared_layout_14() {
    check((1, 4));
}
#[test]
fn shared_layout_15() {
    check((1, 5));
}
#[test]
fn shared_layout_16() {
    check((1, 6));
}

fn check_large_array(version: (u8, u8)) {
    let source = "var<workgroup> work: array<u32, 65536>;
@group(0) @binding(0) var<storage, read_write> output: array<u32, 65536>;
@compute @workgroup_size(1) fn main() { output = work; }";
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    let options = spv::Options {
        lang_version: version,
        zero_initialize_workgroup_memory: Init::None,
        flags: spv::WriterFlags::empty(),
        ..Default::default()
    };
    let words = spv::write_vec(&module, &info, &options, None).unwrap();
    assert!(
        words.len() < 1000,
        "array copies must not unroll all elements"
    );
    validate(&words, version, Init::None);
}
#[test]
fn large_array_copy_10() {
    check_large_array((1, 0));
}
#[test]
fn large_array_copy_13() {
    check_large_array((1, 3));
}

fn validate(words: &[u32], version: (u8, u8), mode: Init) {
    let mut child = Command::new("spirv-val")
        .args([
            "--target-env",
            if version <= (1, 3) {
                "vulkan1.1"
            } else {
                "vulkan1.3"
            },
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("install SPIRV-Tools to run SPIR-V tests");
    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    child.stdin.take().unwrap().write_all(&bytes).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "SPIR-V {version:?}, {mode:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn check_buffer_abi(words: &[u32]) {
    use rspirv::dr::Operand;
    use spirv::{Decoration, Op};
    let module = rspirv::dr::load_words(words).unwrap();
    let mut offsets = Vec::new();
    let mut strides = Vec::new();
    let mut matrix_strides = Vec::new();
    for instruction in module.annotations {
        match (instruction.class.opcode, instruction.operands.as_slice()) {
            (
                Op::Decorate,
                [_, Operand::Decoration(Decoration::ArrayStride), Operand::LiteralBit32(stride)],
            ) => strides.push(*stride),
            (
                Op::MemberDecorate,
                [_, _, Operand::Decoration(Decoration::Offset), Operand::LiteralBit32(offset)],
            ) => offsets.push(*offset),
            (
                Op::MemberDecorate,
                [_, _, Operand::Decoration(Decoration::MatrixStride), Operand::LiteralBit32(stride)],
            ) => matrix_strides.push(*stride),
            _ => {}
        }
    }
    offsets.sort_unstable();
    strides.sort_unstable();
    assert_eq!(offsets, [0, 0, 0, 0, 0, 0, 8, 32, 32]);
    assert_eq!(strides, [16, 64, 64]);
    assert_eq!(matrix_strides, [8]);
}
