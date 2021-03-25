use super::{Instruction, LogicalLayout, PhysicalLayout};
use spirv::{Op, Word, MAGIC_NUMBER};
use std::iter;

// https://github.com/KhronosGroup/SPIRV-Headers/pull/195
const GENERATOR: Word = 28;

impl PhysicalLayout {
    pub(super) fn new(version: Word) -> Self {
        PhysicalLayout {
            magic_number: MAGIC_NUMBER,
            version,
            generator: GENERATOR,
            bound: 0,
            instruction_schema: 0x0u32,
        }
    }

    pub(super) fn in_words(&self, sink: &mut impl Extend<Word>) {
        sink.extend(iter::once(self.magic_number));
        sink.extend(iter::once(self.version));
        sink.extend(iter::once(self.generator));
        sink.extend(iter::once(self.bound));
        sink.extend(iter::once(self.instruction_schema));
    }
}

impl LogicalLayout {
    pub(super) fn in_words(&self, sink: &mut impl Extend<Word>) {
        sink.extend(self.capabilities.iter().cloned());
        sink.extend(self.extensions.iter().cloned());
        sink.extend(self.ext_inst_imports.iter().cloned());
        sink.extend(self.memory_model.iter().cloned());
        sink.extend(self.entry_points.iter().cloned());
        sink.extend(self.execution_modes.iter().cloned());
        sink.extend(self.debugs.iter().cloned());
        sink.extend(self.annotations.iter().cloned());
        sink.extend(self.declarations.iter().cloned());
        sink.extend(self.function_declarations.iter().cloned());
        sink.extend(self.function_definitions.iter().cloned());
    }
}

impl Instruction {
    pub(super) fn new(op: Op) -> Self {
        Instruction {
            op,
            wc: 1, // Always start at 1 for the first word (OP + WC),
            type_id: None,
            result_id: None,
            operands: vec![],
        }
    }

    #[allow(clippy::panic)]
    pub(super) fn set_type(&mut self, id: Word) {
        assert!(self.type_id.is_none(), "Type can only be set once");
        self.type_id = Some(id);
        self.wc += 1;
    }

    #[allow(clippy::panic)]
    pub(super) fn set_result(&mut self, id: Word) {
        assert!(self.result_id.is_none(), "Result can only be set once");
        self.result_id = Some(id);
        self.wc += 1;
    }

    pub(super) fn add_operand(&mut self, operand: Word) {
        self.operands.push(operand);
        self.wc += 1;
    }

    pub(super) fn add_operands(&mut self, operands: Vec<Word>) {
        for operand in operands.into_iter() {
            self.add_operand(operand)
        }
    }

    pub(super) fn to_words(&self, sink: &mut impl Extend<Word>) {
        sink.extend(Some(self.wc << 16 | self.op as u32));
        sink.extend(self.type_id);
        sink.extend(self.result_id);
        sink.extend(self.operands.iter().cloned());
    }
}

impl Instruction {
    #[cfg(test)]
    fn validate(&self, words: &[Word]) {
        let mut inst_index = 0;
        let (wc, op) = ((words[inst_index] >> 16) as u16, words[inst_index] as u16);
        inst_index += 1;

        assert_eq!(wc, words.len() as u16);
        assert_eq!(op, self.op as u16);

        if self.type_id.is_some() {
            assert_eq!(words[inst_index], self.type_id.unwrap());
            inst_index += 1;
        }

        if self.result_id.is_some() {
            assert_eq!(words[inst_index], self.result_id.unwrap());
            inst_index += 1;
        }

        let mut op_index = 0;
        for i in inst_index..wc as usize {
            assert_eq!(words[i], self.operands[op_index]);
            op_index += 1;
        }
    }
}

#[test]
fn test_physical_layout_in_words() {
    let bound = 5;
    let version = 0x10203;

    let mut output = vec![];
    let mut layout = PhysicalLayout::new(version);
    layout.bound = bound;

    layout.in_words(&mut output);

    assert_eq!(&output, &[MAGIC_NUMBER, version, GENERATOR, bound, 0,]);
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
        dummy_instruction.add_operands(super::helpers::string_to_words(
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
        instruction.validate(&output[index..index + wc]);
        index += wc;
    }
}
