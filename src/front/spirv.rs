use crate::{
    storage::{Storage, Token},
    FastHashMap,
};

use std::convert::TryInto;

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
    UnsupportedType(crate::Type),
    UnsupportedExecModel(u32),
    InvalidOperandCount(spirv::Op, u16),
    InvalidOperand,
    InvalidId(spirv::Word),
    InvalidTypeWidth(spirv::Word),
    InvalidSign(spirv::Word),
    InvalidScalar(spirv::Word),
    InvalidVectorSize(spirv::Word),
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

    fn expect_at_least(&self, count: u16) -> Result<(), ParseError> {
        if self.wc >= count {
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

impl<T> Lookup for FastHashMap<spirv::Word, T> {
    type Target = T;
    fn lookup(&self, key: spirv::Word) -> Result<&T, ParseError> {
        self.get(&key)
            .ok_or(ParseError::InvalidId(key))
    }
}

fn map_vector_size(word: spirv::Word) -> Result<crate::VectorSize, ParseError> {
    match word {
        2 => Ok(crate::VectorSize::Bi),
        3 => Ok(crate::VectorSize::Tri),
        4 => Ok(crate::VectorSize::Quad),
        _ => Err(ParseError::InvalidVectorSize(word))
    }
}

type MemberIndex = u32;

#[derive(Debug, Default)]
struct Decoration {
    name: Option<String>,
}

#[derive(Debug, Default)]
struct MemberDecoration {
    name: Option<String>,
}

pub struct Parser<I> {
    data: I,
    state: ModuleState,
    temp_bytes: Vec<u8>,
    future_decor: FastHashMap<spirv::Word, Decoration>,
    future_member_decor: FastHashMap<(spirv::Word, MemberIndex), MemberDecoration>,
    lookup_type: FastHashMap<spirv::Word, crate::Type>,
    lookup_constant: FastHashMap<spirv::Word, crate::Constant>,
    lookup_function: FastHashMap<spirv::Word, Token<crate::Function>>,
}

impl<I: Iterator<Item = u32>> Parser<I> {
    pub fn new(data: I) -> Self {
        Parser {
            data,
            state: ModuleState::Empty,
            temp_bytes: Vec::new(),
            future_decor: FastHashMap::default(),
            future_member_decor: FastHashMap::default(),
            lookup_type: FastHashMap::default(),
            lookup_constant: FastHashMap::default(),
            lookup_function: FastHashMap::default(),
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

    pub fn parse(&mut self) -> Result<crate::Module, ParseError> {
        let mut module = crate::Module {
            header: {
                if self.next()? != spirv::MAGIC_NUMBER {
                    return Err(ParseError::InvalidHeader);
                }
                let version_raw = self.next()?.to_le_bytes();
                let generator = self.next()?;
                let _bound = self.next()?;
                let _schema = self.next()?;
                crate::Header {
                    version: (version_raw[2], version_raw[1], version_raw[0]),
                    generator,
                }
            },
            array_declarations: Storage::new(),
            struct_declarations: Storage::new(),
            functions: Storage::new(),
            entry_points: Vec::new(),
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
                        let _var = self.next()?; //TODO: in/out variables
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
                    inst.expect_at_least(3)?;
                    let id = self.next()?;
                    let (name, left) = self.next_string(inst.wc - 2)?;
                    if left != 0 {
                        return Err(ParseError::InvalidOperand);
                    }
                    self.future_decor
                        .entry(id)
                        .or_default()
                        .name = Some(name.to_owned());
                }
                Op::MemberName => {
                    self.switch(ModuleState::Name, inst.op)?;
                    inst.expect_at_least(4)?;
                    let id = self.next()?;
                    let member = self.next()?;
                    let (name, left) = self.next_string(inst.wc - 3)?;
                    if left != 0 {
                        return Err(ParseError::InvalidOperand);
                    }
                    self.future_member_decor
                        .entry((id, member))
                        .or_default()
                        .name = Some(name.to_owned());
                }
                Op::Decorate => {
                    self.switch(ModuleState::Annotation, inst.op)?;
                    inst.expect_at_least(3)?;
                    let _id = self.next()?;
                    let _decoration = self.next()?;
                    for _ in 3 .. inst.wc {
                        let _var = self.next()?; //TODO
                    }
                }
                Op::MemberDecorate => {
                    self.switch(ModuleState::Annotation, inst.op)?;
                    inst.expect_at_least(4)?;
                    let _id = self.next()?;
                    let _member = self.next()?;
                    let _decoration = self.next()?;
                    for _ in 4 .. inst.wc {
                        let _var = self.next()?; //TODO
                    }
                }
                Op::TypeVoid => {
                    self.switch(ModuleState::Type, inst.op)?;
                    inst.expect(2)?;
                    let id = self.next()?;
                    self.lookup_type.insert(id, crate::Type::Void);
                }
                Op::TypeInt => {
                    self.switch(ModuleState::Type, inst.op)?;
                    inst.expect(4)?;
                    let id = self.next()?;
                    let width = self.next()?;
                    let sign = self.next()?;
                    self.lookup_type.insert(id, crate::Type::Scalar {
                        kind: match sign {
                            0 => crate::ScalarKind::Uint,
                            1 => crate::ScalarKind::Sint,
                            _ => return Err(ParseError::InvalidSign(sign)),
                        },
                        width: width
                            .try_into()
                            .map_err(|_| ParseError::InvalidTypeWidth(width))?,
                    });
                }
                Op::TypeFloat => {
                    self.switch(ModuleState::Type, inst.op)?;
                    inst.expect(3)?;
                    let id = self.next()?;
                    let width = self.next()?;
                    self.lookup_type.insert(id, crate::Type::Scalar {
                        kind: crate::ScalarKind::Float,
                        width: width
                            .try_into()
                            .map_err(|_| ParseError::InvalidTypeWidth(width))?,
                    });
                }
                Op::TypeVector => {
                    self.switch(ModuleState::Type, inst.op)?;
                    inst.expect(4)?;
                    let id = self.next()?;
                    let type_id = self.next()?;
                    let (kind, width) = match self.lookup_type.lookup(type_id)? {
                        &crate::Type::Scalar { ref kind, width } => (kind.clone(), width),
                        _ => return Err(ParseError::InvalidScalar(type_id)),
                    };
                    let component_count = self.next()?;
                    self.lookup_type.insert(id, crate::Type::Vector {
                        size: map_vector_size(component_count)?,
                        kind,
                        width,
                    });
                }
                Op::TypeFunction => {
                    self.switch(ModuleState::Type, inst.op)?;
                    inst.expect_at_least(3)?;
                    let id = self.next()?;
                    let ret_type_id = self.next()?;
                    let mut parameter_types = Vec::new();
                    for _ in 3 .. inst.wc {
                        let type_id = self.next()?;
                        let ty = self.lookup_type.lookup(type_id)?.clone();
                        parameter_types.push(ty);
                    }
                    let token = module.functions.append(crate::Function {
                        name: None,
                        return_type: self.lookup_type.lookup(ret_type_id)?.clone(),
                        parameter_types,
                        body: Vec::new(),
                    });
                    self.lookup_function.insert(id, token);
                }
                Op::TypeArray => {
                    self.switch(ModuleState::Type, inst.op)?;
                    inst.expect(4)?;
                    let id = self.next()?;
                    let type_id = self.next()?;
                    let length = self.next()?;
                    let decl = crate::ArrayDeclaration {
                        base: self.lookup_type.lookup(type_id)?.clone(),
                        length,
                    };
                    let ty = crate::Type::Array(module.array_declarations.append(decl));
                    self.lookup_type.insert(id, ty);
                }
                Op::TypeStruct => {
                    self.switch(ModuleState::Type, inst.op)?;
                    inst.expect_at_least(2)?;
                    let id = self.next()?;
                    let mut decl = crate::StructDeclaration {
                        name: self.future_decor
                            .remove(&id)
                            .and_then(|dec| dec.name),
                        members: Vec::with_capacity(inst.wc as usize - 2),
                    };
                    for i in 0 .. inst.wc as u32 - 2 {
                        let type_id = self.next()?;
                        let ty = self.lookup_type.lookup(type_id)?.clone();
                        decl.members.push(crate::StructMember {
                            name: self.future_member_decor
                                .remove(&(id, i))
                                .and_then(|dec| dec.name),
                            ty,
                        });
                    }
                    let ty = crate::Type::Struct(module.struct_declarations.append(decl));
                    self.lookup_type.insert(id, ty);
                }
                Op::Constant => {
                    self.switch(ModuleState::Type, inst.op)?;
                    inst.expect_at_least(3)?;
                    let result_type_id = self.next()?;
                    let id = self.next()?;
                    let constant = match *self.lookup_type.lookup(result_type_id)? {
                        crate::Type::Scalar { kind: crate::ScalarKind::Uint, width } => {
                            let low = self.next()?;
                            let high = if width > 32 {
                                inst.expect(4)?;
                                self.next()?
                            } else {
                                0
                            };
                            crate::Constant::Uint(((high as u64) << 32) | low as u64)
                        }
                        crate::Type::Scalar { kind: crate::ScalarKind::Sint, width } => {
                            let low = self.next()?;
                            let high = if width < 32 {
                                return Err(ParseError::InvalidTypeWidth(width as u32));
                            } else if width > 32 {
                                inst.expect(4)?;
                                self.next()?
                            } else {
                                !0
                            };
                            crate::Constant::Sint(unsafe {
                                std::mem::transmute(((high as u64) << 32) | low as u64)
                            })
                        }
                        crate::Type::Scalar { kind: crate::ScalarKind::Float, width } => {
                            let low = self.next()?;
                            let extended = if width < 32 {
                                return Err(ParseError::InvalidTypeWidth(width as u32));
                            } else if width > 32 {
                                inst.expect(4)?;
                                let high = self.next()?;
                                unsafe {
                                    std::mem::transmute(((high as u64) << 32) | low as u64)
                                }
                            } else {
                                unsafe {
                                    std::mem::transmute::<_, f32>(low) as f64
                                }
                            };
                            crate::Constant::Float(extended)
                        }
                        ref other => return Err(ParseError::UnsupportedType(other.clone()))
                    };
                    self.lookup_constant.insert(id, constant);
                }
                _ => return Err(ParseError::UnsupportedInstruction(self.state, inst.op))
                //TODO
            }
        }

        if !self.future_decor.is_empty() {
            log::warn!("Unused item decorations: {:?}", self.future_decor);
            self.future_decor.clear();
        }
        if !self.future_member_decor.is_empty() {
            log::warn!("Unused member decorations: {:?}", self.future_member_decor);
            self.future_member_decor.clear();
        }

        module.entry_points.reserve(raw_entry_points.len());
        for (exec_model, name, fun_id) in raw_entry_points {
            module.entry_points.push(crate::EntryPoint {
                exec_model,
                name,
                function: *self.lookup_function.lookup(fun_id)?,
            });
        }

        Ok(module)
    }
}

pub fn parse_u8_slice(data: &[u8]) -> Result<crate::Module, ParseError> {
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
