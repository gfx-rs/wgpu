pub(super) fn validate_instruction(
    words: &[spirv::Word],
    instruction: &crate::back::spv::Instruction,
) {
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
