/*!
Test SPIR-V backend options that work around driver bugs.
*/

#![cfg(all(feature = "wgsl-in", spv_out))]

use naga::back::spv;

fn module(source: &str, options: &spv::Options) -> rspirv::dr::Module {
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    let words = spv::write_vec(&module, &info, options, None).unwrap();
    rspirv::dr::load_words(words).unwrap()
}

fn ops(source: &str, buffer_sizes: Option<spv::BufferSizes>) -> Vec<spirv::Op> {
    let options = spv::Options {
        buffer_sizes,
        ..spv::Options::default()
    };
    module(source, &options)
        .all_inst_iter()
        .map(|instruction| instruction.class.opcode)
        .collect()
}

fn sizes(bindings: &[(u32, u32)]) -> Option<spv::BufferSizes> {
    Some(spv::BufferSizes {
        push_constant_size: 128,
        offsets: bindings
            .iter()
            .map(|&(binding, offset)| (naga::ResourceBinding { group: 0, binding }, offset))
            .collect(),
    })
}

const SOURCE: &str = "
struct Data { header: vec4<u32>, values: array<u32> }
@group(0) @binding(0) var<storage, read_write> data: Data;
@group(0) @binding(1) var<storage> other: array<f32>;
@compute @workgroup_size(1)
fn main() {
    data.header.x = arrayLength(&data.values) + arrayLength(&other);
}
";

#[test]
fn replaces_array_length() {
    let ops = ops(SOURCE, sizes(&[(0, 124), (1, 120)]));
    assert!(!ops.contains(&spirv::Op::ArrayLength));
    assert!(ops.contains(&spirv::Op::UDiv));
}

#[test]
fn keeps_array_length_without_size() {
    let ops = ops(SOURCE, sizes(&[(0, 124)]));
    assert_eq!(
        ops.iter()
            .filter(|&&op| op == spirv::Op::ArrayLength)
            .count(),
        1
    );
}

#[test]
fn shares_block_with_immediates() {
    let source = format!("var<immediate> im: vec4<u32>;\n{SOURCE}");
    let ops = ops(&source, sizes(&[(0, 124), (1, 120)]));
    assert!(!ops.contains(&spirv::Op::ArrayLength));
    assert_eq!(
        ops.iter().filter(|&&op| op == spirv::Op::Variable).count(),
        3
    );
}

#[test]
fn keeps_array_length_for_size_in_immediates() {
    let source = format!("var<immediate> im: vec4<u32>;\n{SOURCE}");
    let ops = ops(&source, sizes(&[(0, 124), (1, 12)]));
    assert_eq!(
        ops.iter()
            .filter(|&&op| op == spirv::Op::ArrayLength)
            .count(),
        1
    );
}

fn num_workgroups_inputs(source: &str) -> usize {
    let mut options = spv::Options::default();
    options.flags |= spv::WriterFlags::FORCE_NUM_WORKGROUPS;
    module(source, &options)
        .annotations
        .iter()
        .filter(|instruction| {
            instruction.operands.get(2)
                == Some(&rspirv::dr::Operand::BuiltIn(spirv::BuiltIn::NumWorkgroups))
        })
        .count()
}

#[test]
fn force_num_workgroups() {
    assert_eq!(
        num_workgroups_inputs("@compute @workgroup_size(1) fn main() {}"),
        1
    );
    assert_eq!(
        num_workgroups_inputs(
            "@compute @workgroup_size(1) fn main(@builtin(num_workgroups) n: vec3u) {}"
        ),
        1
    );
}
