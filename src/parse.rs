use crate::storage::{Storage, Token};

use std::{
    collections::HashMap,
    hash::BuildHasherDefault,
};

const LAST_KNOWN_OPCODE: spirv::Op = spirv::Op::MemberDecorateStringGOOGLE;
const LAST_KNOWN_CAPABILITY: spirv::Capability = spirv::Capability::VulkanMemoryModelDeviceScopeKHR;
const LAST_KNOWN_EXEC_MODEL: spirv::ExecutionModel = spirv::ExecutionModel::Kernel;

pub const SUPPORTED_CAPABILITIES: &[spirv::Capability] = &[
    spirv::Capability::Shader,
];
pub const SUPPORTED_EXTENSIONS: &[&str] = &[
];
pub const SUPPORTED_EXT_SETS: &[&str] = &[
    "GLSL.std.450",
];

#[derive(Debug)]
pub enum ParseError {
    InvalidHeader,
    InvalidWordCount,
    UnknownInstruction(u16),
    UnsupportedInstruction(ModuleState, spirv::Op),
    UnknownCapability(u32),
    UnsupportedCapability(spirv::Capability),
    UnsupportedExtension(String),
    UnsupportedExtSet(String),
    UnsupportedExecModel(u32),
    InvalidOperandCount(spirv::Op, u16),
    InvalidOperand,
    InvalidId(spirv::Word),
    BadString,
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

trait Lookup {
    type Target;
    fn lookup(&self, key: spirv::Word) -> Result<&Self::Target, ParseError>;
}

type FastHashMap<K, T> = HashMap<K, T, BuildHasherDefault<fxhash::FxHasher>>;

impl<T> Lookup for FastHashMap<spirv::Word, T> {
    type Target = T;
    fn lookup(&self, key: spirv::Word) -> Result<&T, ParseError> {
        self.get(&key)
            .ok_or(ParseError::InvalidId(key))
    }
}

type MemberIndex = u32;

pub struct Parser<I> {
    data: I,
    state: ModuleState,
    temp_bytes: Vec<u8>,
    future_names: FastHashMap<spirv::Word, String>,
    future_member_names: FastHashMap<(spirv::Word, MemberIndex), String>,
    lookup_function: FastHashMap<spirv::Word, Token<super::Function>>,
}

impl<I: Iterator<Item = u32>> Parser<I> {
    pub fn new(data: I) -> Self {
        Parser {
            data,
            state: ModuleState::Empty,
            temp_bytes: Vec::new(),
            future_names: HashMap::default(),
            future_member_names: HashMap::default(),
            lookup_function: HashMap::default(),
        }
    }

    fn next(&mut self) -> Result<u32, ParseError> {
        self.data.next().ok_or(ParseError::IncompleteData)
    }

    fn next_inst(&mut self) -> Result<Instruction, ParseError> {
        let word = self.next()?;
        let (wc, opcode) = ((word >> 16) as u16, (word & 0xffff) as u16);
        if wc == 0 {
            return Err(ParseError::InvalidWordCount);
        }
        if opcode > LAST_KNOWN_OPCODE as u16 {
            return Err(ParseError::UnknownInstruction(opcode));
        }

        Ok(Instruction {
            op: unsafe {
                std::mem::transmute(opcode as u32)
            },
            wc,
        })
    }

    fn next_string(&mut self, mut count: u16) -> Result<(String, u16), ParseError>{
        self.temp_bytes.clear();
        loop {
            if count == 0 {
                return Err(ParseError::BadString);
            }
            count -= 1;
            let chars = self.next()?.to_le_bytes();
            let pos = chars.iter().position(|&c| c  == 0).unwrap_or(4);
            self.temp_bytes.extend_from_slice(&chars[.. pos]);
            if pos < 4 {
                break
            }
        }
        std::str::from_utf8(&self.temp_bytes)
            .map(|s| (s.to_owned(), count))
            .map_err(|_| ParseError::BadString)
    }

    fn switch(&mut self, state: ModuleState, op: spirv::Op) -> Result<(), ParseError> {
        if state < self.state {
            return Err(ParseError::UnsupportedInstruction(self.state, op))
        } else {
            self.state = state;
            Ok(())
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
        let mut raw_entry_points = Vec::new();

        while let Ok(inst) = self.next_inst() {
            use spirv::Op;
            match inst.op {
                Op::Capability => {
                    self.switch(ModuleState::Capability, inst.op)?;
                    inst.expect(2)?;
                    let capability = self.next()?;
                    if capability > LAST_KNOWN_CAPABILITY as u32 {
                        return Err(ParseError::UnknownCapability(capability));
                    }
                    let cap = unsafe {
                        std::mem::transmute(capability)
                    };
                    if !SUPPORTED_CAPABILITIES.contains(&cap) {
                        return Err(ParseError::UnsupportedCapability(cap));
                    }
                }
                Op::Extension => {
                    self.switch(ModuleState::Extension, inst.op)?;
                    let (name, left) = self.next_string(inst.wc - 1)?;
                    if left != 0 {
                        return Err(ParseError::InvalidOperand);
                    }
                    if !SUPPORTED_EXTENSIONS.contains(&name.as_str()) {
                        return Err(ParseError::UnsupportedExtension(name.to_owned()));
                    }
                }
                Op::ExtInstImport => {
                    self.switch(ModuleState::Extension, inst.op)?;
                    let _result = self.next()?;
                    let (name, left) = self.next_string(inst.wc - 2)?;
                    if left != 0 {
                        return Err(ParseError::InvalidOperand)
                    }
                    if !SUPPORTED_EXT_SETS.contains(&name.as_str()) {
                        return Err(ParseError::UnsupportedExtSet(name.to_owned()));
                    }
                }
                Op::MemoryModel => {
                    self.switch(ModuleState::MemoryModel, inst.op)?;
                    inst.expect(3)?;
                    let _addressing_model = self.next()?;
                    let _memory_model = self.next()?;
                }
                Op::EntryPoint => {
                    self.switch(ModuleState::EntryPoint, inst.op)?;
                    let exec_model = self.next()?;
                    if exec_model > LAST_KNOWN_EXEC_MODEL as u32 {
                        return Err(ParseError::UnsupportedExecModel(exec_model));
                    }
                    let function_id = self.next()?;
                    let (name, left) = self.next_string(inst.wc - 3)?;
                    for _ in 0 .. left {
                        let _var = self.next()?; //TODO
                    }
                    raw_entry_points.push((
                        unsafe {
                            std::mem::transmute(exec_model)
                        },
                        name.to_owned(),
                        function_id,
                    ));
                }
                Op::Source => {
                    self.switch(ModuleState::Source, inst.op)?;
                    for _ in 1 .. inst.wc {
                        let _ = self.next()?;
                    }
                }
                Op::Name => {
                    self.switch(ModuleState::Name, inst.op)?;
                    if inst.wc < 3 {
                        return Err(ParseError::InvalidOperandCount(inst.op, inst.wc));
                    }
                    let id = self.next()?;
                    let (name, left) = self.next_string(inst.wc - 2)?;
                    if left != 0 {
                        return Err(ParseError::InvalidOperand);
                    }
                    self.future_names.insert(id, name.to_owned());
                }
                Op::MemberName => {
                    self.switch(ModuleState::Name, inst.op)?;
                    if inst.wc < 4 {
                        return Err(ParseError::InvalidOperandCount(inst.op, inst.wc));
                    }
                    let id = self.next()?;
                    let member = self.next()?;
                    let (name, left) = self.next_string(inst.wc - 3)?;
                    if left != 0 {
                        return Err(ParseError::InvalidOperand);
                    }
                    self.future_member_names.insert((id, member), name.to_owned());
                }
                _ => return Err(ParseError::UnsupportedInstruction(self.state, inst.op))
                //TODO
            }
        }

        let mut entry_points = Vec::with_capacity(raw_entry_points.len());
        for (exec_model, name, fun_id) in raw_entry_points {
            entry_points.push(super::EntryPoint {
                exec_model,
                name,
                function: *self.lookup_function.lookup(fun_id)?,
            });
        }

        Ok(super::Module {
            header,
            struct_declarations: Storage::new(),
            functions: Storage::new(),
            entry_points,
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
