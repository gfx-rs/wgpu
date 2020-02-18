use crate::storage::Storage;

#[derive(Debug)]
pub enum ParseError {
    InvalidHeader,
    InvalidWordCount,
    UnexpectedInstruction(ModuleState, spirv::Op),
    UnknownInstruction,
    InvalidOperandCount(spirv::Op, u16),
    IncompleteData,
}

struct Instruction {
    op: spirv::Op,
    wc: u16,
}

impl Instruction {
    fn expect(&self, count: u16) -> Result<(), ParseError> {
        if self.wc == count {
            Ok(())
        } else {
            Err(ParseError::InvalidOperandCount(self.op, self.wc))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum ModuleState {
    Empty,
    Capability,
    Extension,
    ExtInstImport,
    MemoryModel,
    EntryPoint,
    ExecutionMode,
    Source,
    Name,
    ModuleProcessed,
    Annotation,
    Type,
    FunctionDecl,
    FunctionDef,
}

pub struct Parser<I> {
    data: I,
    state: ModuleState,
}

impl<I: Iterator<Item = u32>> Parser<I> {
    pub fn new(data: I) -> Self {
        Parser {
            data,
            state: ModuleState::Empty,
        }
    }

    fn next(&mut self) -> Result<u32, ParseError> {
        self.data.next().ok_or(ParseError::IncompleteData)
    }

    fn next_inst(&mut self) -> Result<Instruction, ParseError> {
        const LAST_KNOWN_OPCODE: spirv::Op = spirv::Op::MemberDecorateStringGOOGLE;

        let word = self.next()?;
        let (wc, opcode) = ((word >> 16) as u16, (word & 0xffff) as u16);
        if wc == 0 {
            return Err(ParseError::InvalidWordCount);
        }
        if opcode > LAST_KNOWN_OPCODE as u16 {
            return Err(ParseError::UnknownInstruction);
        }

        Ok(Instruction {
            op: unsafe {
                std::mem::transmute(opcode as u32)
            },
            wc,
        })
    }

    fn switch(&mut self, state: ModuleState, inst: &Instruction, word_count: u16) -> Result<(), ParseError> {
        if state < self.state {
            return Err(ParseError::UnexpectedInstruction(self.state, inst.op))
        } else {
            self.state = state;
            inst.expect(word_count)
        }
    }

    pub fn parse(&mut self) -> Result<super::Module, ParseError> {
        let header = {
            if self.next()? != spirv::MAGIC_NUMBER {
                return Err(ParseError::InvalidHeader);
            }
            let version_raw = self.next()?.to_le_bytes();
            let generator = self.next()?;
            let _bound = self.next()?;
            let _schema = self.next()?;
            super::Header {
                version: (version_raw[2], version_raw[1], version_raw[0]),
                generator,
            }
        };

        while let Ok(inst) = self.next_inst() {
            use spirv::Op;
            match inst.op {
                Op::Capability => {
                    self.switch(ModuleState::Capability, &inst, 2)?;
                    let _capability = self.next()?;
                }
                Op::MemoryModel => {
                    self.switch(ModuleState::MemoryModel, &inst, 3)?;
                    let _addressing_model = self.next()?;
                    let _memory_model = self.next()?;
                },
                _ => return Err(ParseError::UnexpectedInstruction(self.state, inst.op))
                //TODO
            }
        }

        Ok(super::Module {
            header,
            struct_declarations: Storage::new(),
            functions: Storage::new(),
            entry_points: Vec::new(),
        })
    }
}

pub fn parse_u8_slice(data: &[u8]) -> Result<super::Module, ParseError> {
    use std::convert::TryInto;

    if data.len() % 4 != 0 {
        return Err(ParseError::IncompleteData);
    }

    let words = data
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()));
    Parser::new(words).parse()
}

#[cfg(test)]
mod test {
    #[test]
    fn parse() {
        let bin = vec![
            // Magic number.           Version number: 1.0.
            0x03, 0x02, 0x23, 0x07,    0x00, 0x00, 0x01, 0x00,
            // Generator number: 0.    Bound: 0.
            0x00, 0x00, 0x00, 0x00,    0x00, 0x00, 0x00, 0x00,
            // Reserved word: 0.
            0x00, 0x00, 0x00, 0x00,
            // OpMemoryModel.          Logical.
            0x0e, 0x00, 0x03, 0x00,    0x00, 0x00, 0x00, 0x00,
            // GLSL450.
            0x01, 0x00, 0x00, 0x00,
        ];
        let _ = super::parse_u8_slice(&bin).unwrap();
    }
}
