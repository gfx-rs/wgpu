use crate::back::spv::Instruction;
use spirv::{Op, Word};

pub(super) struct SpecRequirements {
    pub(super) op: Op,
    pub(super) wc: u32,
    pub(super) type_id: bool,
    pub(super) result_id: bool,
    pub(super) operands: bool,
}

/// Basic validation for checking if the instruction complies to the spec requirements
pub(super) fn validate_spec_requirements(
    requirements: SpecRequirements,
    instruction: &Instruction,
) {
    assert_eq!(requirements.op, instruction.op);

    // Pass the assert if the minimum referred wc in the spec is met
    assert!(instruction.wc >= requirements.wc);

    if requirements.type_id {
        assert!(instruction.type_id.is_some());
    }

    if requirements.result_id {
        assert!(instruction.result_id.is_some());
    }

    if requirements.operands {
        assert!(!instruction.operands.is_empty());
    }
}

pub(super) fn validate_instruction(words: &[Word], instruction: &Instruction) {
    let mut inst_index = 0;
    let (wc, op) = ((words[inst_index] >> 16) as u16, words[inst_index] as u16);
    inst_index += 1;

    assert_eq!(wc, words.len() as u16);
    assert_eq!(op, instruction.op as u16);

    if instruction.type_id.is_some() {
        assert_eq!(words[inst_index], instruction.type_id.unwrap());
        inst_index += 1;
    }

    if instruction.result_id.is_some() {
        assert_eq!(words[inst_index], instruction.result_id.unwrap());
        inst_index += 1;
    }

    let mut op_index = 0;
    for i in inst_index..wc as usize {
        assert_eq!(words[i as usize], instruction.operands[op_index]);
        op_index += 1;
    }
}
