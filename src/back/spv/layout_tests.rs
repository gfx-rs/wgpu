use crate::back::spv::test_framework::*;
use crate::back::spv::{helpers, Instruction, LogicalLayout, PhysicalLayout};
use crate::Header;
use spirv::*;

#[test]
fn test_physical_layout_in_words() {
    let header = Header {
        generator: 0,
        version: (1, 2, 3),
    };
    let bound = 5;

    let mut output = vec![];
    let mut layout = PhysicalLayout::new(&header);
    layout.bound = bound;

    layout.in_words(&mut output);

    assert_eq!(output[0], spirv::MAGIC_NUMBER);
    assert_eq!(
        output[1],
        to_word(&[header.version.0, header.version.1, header.version.2, 1])
    );
    assert_eq!(output[2], 0);
    assert_eq!(output[3], bound);
    assert_eq!(output[4], 0);
}

#[test]
fn test_logical_layout_in_words() {
    let mut output = vec![];
    let mut layout = LogicalLayout::default();
    let layout_vectors = 11;
    let mut instructions = Vec::with_capacity(layout_vectors);

    let vector_names = &[
        "Capabilities",
        "Extensions",
        "External Instruction Imports",
        "Memory Model",
        "Entry Points",
        "Execution Modes",
        "Debugs",
        "Annotations",
        "Declarations",
        "Function Declarations",
        "Function Definitions",
    ];

    for i in 0..layout_vectors {
        let mut dummy_instruction = Instruction::new(Op::Constant);
        dummy_instruction.set_type((i + 1) as u32);
        dummy_instruction.set_result((i + 2) as u32);
        dummy_instruction.add_operand((i + 3) as u32);
        dummy_instruction.add_operands(helpers::string_to_words(
            format!("This is the vector: {}", vector_names[i]).as_str(),
        ));
        instructions.push(dummy_instruction);
    }

    instructions[0].to_words(&mut layout.capabilities);
    instructions[1].to_words(&mut layout.extensions);
    instructions[2].to_words(&mut layout.ext_inst_imports);
    instructions[3].to_words(&mut layout.memory_model);
    instructions[4].to_words(&mut layout.entry_points);
    instructions[5].to_words(&mut layout.execution_modes);
    instructions[6].to_words(&mut layout.debugs);
    instructions[7].to_words(&mut layout.annotations);
    instructions[8].to_words(&mut layout.declarations);
    instructions[9].to_words(&mut layout.function_declarations);
    instructions[10].to_words(&mut layout.function_definitions);

    layout.in_words(&mut output);

    let mut index: usize = 0;
    for instruction in instructions {
        let wc = instruction.wc as usize;
        let instruction_output = &output[index..index + wc];
        validate_instruction(instruction_output, &instruction);
        index += wc;
    }
}

#[test]
fn test_instruction_set_type() {
    let ty = 1;
    let mut instruction = Instruction::new(Op::Constant);
    assert_eq!(instruction.wc, 1);

    instruction.set_type(ty);
    assert_eq!(instruction.type_id.unwrap(), ty);
    assert_eq!(instruction.wc, 2);
}

#[test]
#[should_panic]
fn test_instruction_set_type_twice() {
    let ty = 1;
    let mut instruction = Instruction::new(Op::Constant);
    instruction.set_type(ty);
    instruction.set_type(ty);
}

#[test]
fn test_instruction_set_result() {
    let result = 1;
    let mut instruction = Instruction::new(Op::Constant);
    assert_eq!(instruction.wc, 1);

    instruction.set_result(result);
    assert_eq!(instruction.result_id.unwrap(), result);
    assert_eq!(instruction.wc, 2);
}

#[test]
#[should_panic]
fn test_instruction_set_result_twice() {
    let result = 1;
    let mut instruction = Instruction::new(Op::Constant);
    instruction.set_result(result);
    instruction.set_result(result);
}

#[test]
fn test_instruction_add_operand() {
    let operand = 1;
    let mut instruction = Instruction::new(Op::Constant);
    assert_eq!(instruction.operands.len(), 0);
    assert_eq!(instruction.wc, 1);

    instruction.add_operand(operand);
    assert_eq!(instruction.operands.len(), 1);
    assert_eq!(instruction.wc, 2);
}

#[test]
fn test_instruction_add_operands() {
    let operands = vec![1, 2, 3];
    let mut instruction = Instruction::new(Op::Constant);
    assert_eq!(instruction.operands.len(), 0);
    assert_eq!(instruction.wc, 1);

    instruction.add_operands(operands);
    assert_eq!(instruction.operands.len(), 3);
    assert_eq!(instruction.wc, 4);
}

#[test]
fn test_instruction_to_words() {
    let ty = 1;
    let result = 2;
    let operand = 3;
    let mut instruction = Instruction::new(Op::Constant);
    instruction.set_type(ty);
    instruction.set_result(result);
    instruction.add_operand(operand);

    let mut output = vec![];
    instruction.to_words(&mut output);
    validate_instruction(output.as_slice(), &instruction);
}

fn to_word(bytes: &[u8]) -> Word {
    ((bytes[0] as u32) << 16) | ((bytes[1] as u32) << 8) | bytes[2] as u32
}
