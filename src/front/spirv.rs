use crate::{
    storage::{Storage, Token},
    FastHashMap,
};

use std::convert::TryInto;

const LAST_KNOWN_OPCODE: spirv::Op = spirv::Op::MemberDecorateStringGOOGLE;
const LAST_KNOWN_CAPABILITY: spirv::Capability = spirv::Capability::VulkanMemoryModelDeviceScopeKHR;
const LAST_KNOWN_EXECUTION_MODEL: spirv::ExecutionModel = spirv::ExecutionModel::Kernel;
const LAST_KNOWN_STORAGE_CLASS: spirv::StorageClass = spirv::StorageClass::StorageBuffer;

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
    UnknownCapability(u32),
    UnsupportedInstruction(ModuleState, spirv::Op),
    UnsupportedCapability(spirv::Capability),
    UnsupportedExtension(String),
    UnsupportedExtSet(String),
    UnsupportedType(crate::Type),
    UnsupportedExecutionModel(u32),
    UnsupportedStorageClass(u32),
    UnsupportedFunctionControl(u32),
    InvalidParameter(spirv::Op),
    InvalidOperandCount(spirv::Op, u16),
    InvalidOperand,
    InvalidId(spirv::Word),
    InvalidTypeWidth(spirv::Word),
    InvalidSign(spirv::Word),
    InvalidInnerType(spirv::Word),
    InvalidVectorSize(spirv::Word),
    InvalidVariableClass(spirv::StorageClass),
    InvalidAccessType(spirv::Word),
    InvalidAccessIndex(crate::Expression),
    InvalidLoadType(spirv::Word),
    WrongFunctionResultType(spirv::Word),
    WrongFunctionParameterType(spirv::Word),
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
    Function,
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

fn map_storage_class(word: spirv::Word) -> Result<spirv::StorageClass, ParseError> {
    if word > LAST_KNOWN_STORAGE_CLASS as u32 {
        Err(ParseError::UnsupportedStorageClass(word))
    } else {
        Ok(unsafe { std::mem::transmute(word) })
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

#[derive(Debug)]
struct LookupFunctionType {
    parameter_type_ids: Vec<spirv::Word>,
    return_type_id: spirv::Word,
}

#[derive(Debug)]
struct EntryPoint {
    exec_model: spirv::ExecutionModel,
    name: String,
    function_id: spirv::Word,
    variable_ids: Vec<spirv::Word>,
}

#[derive(Debug)]
struct LookupType {
    value: crate::Type,
    base_id: Option<spirv::Word>,
}

#[derive(Debug)]
struct LookupConstant {
    value: crate::Constant,
    type_id: spirv::Word,
}

#[derive(Debug)]
struct LookupVariable {
    token: Token<crate::GlobalVariable>,
    type_id: spirv::Word,
}

#[derive(Clone, Debug)]
struct LookupExpression {
    token: Token<crate::Expression>,
    type_id: spirv::Word,
}

pub struct Parser<I> {
    data: I,
    state: ModuleState,
    temp_bytes: Vec<u8>,
    future_decor: FastHashMap<spirv::Word, Decoration>,
    future_member_decor: FastHashMap<(spirv::Word, MemberIndex), MemberDecoration>,
    lookup_member_type_id: FastHashMap<(spirv::Word, MemberIndex), spirv::Word>,
    lookup_type: FastHashMap<spirv::Word, LookupType>,
    lookup_constant: FastHashMap<spirv::Word, LookupConstant>,
    lookup_variable: FastHashMap<spirv::Word, LookupVariable>,
    lookup_expression: FastHashMap<spirv::Word, LookupExpression>,
    lookup_function_type: FastHashMap<spirv::Word, LookupFunctionType>,
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
            lookup_member_type_id: FastHashMap::default(),
            lookup_type: FastHashMap::default(),
            lookup_constant: FastHashMap::default(),
            lookup_variable: FastHashMap::default(),
            lookup_expression: FastHashMap::default(),
            lookup_function_type: FastHashMap::default(),
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

    fn next_block(&mut self, fun: &mut crate::Function) -> Result<(), ParseError> {
        loop {
            use spirv::Op;
            let inst = self.next_inst()?;
            log::debug!("\t\t{:?} [{}]", inst.op, inst.wc);
            match inst.op {
                Op::AccessChain => {
                    struct AccessExpression {
                        base_token: Token<crate::Expression>,
                        type_id: spirv::Word,
                    }
                    inst.expect_at_least(4)?;
                    let result_type_id = self.next()?;
                    let id = self.next()?;
                    let base_id = self.next()?;
                    log::trace!("\t\t\tlooking up expr {:?}", base_id);
                    let mut acex = {
                        let expr = self.lookup_expression.lookup(base_id)?;
                        let ptr_type = self.lookup_type.lookup(expr.type_id)?;
                        AccessExpression {
                            base_token: expr.token,
                            type_id: ptr_type.base_id.unwrap(),
                        }
                    };
                    for _ in 4 .. inst.wc {
                        let access_id = self.next()?;
                        log::trace!("\t\t\tlooking up expr {:?}", access_id);
                        let index_expr = self.lookup_expression.lookup(access_id)?.clone();
                        match self.lookup_type.lookup(index_expr.type_id)?.value {
                            crate::Type::Scalar { kind: crate::ScalarKind::Uint, .. } |
                            crate::Type::Scalar { kind: crate::ScalarKind::Sint, .. } => (),
                            ref other => return Err(ParseError::UnsupportedType(other.clone())),
                        }
                        log::trace!("\t\t\tlooking up type {:?}", acex.type_id);
                        let ty = self.lookup_type.lookup(acex.type_id)?;
                        acex = match ty.value {
                            crate::Type::Struct(_) => {
                                let index = match fun.expressions[index_expr.token] {
                                    crate::Expression::Constant(crate::Constant::Sint(v)) => v as u32,
                                    crate::Expression::Constant(crate::Constant::Uint(v)) => v as u32,
                                    ref other => return Err(ParseError::InvalidAccessIndex(other.clone()))
                                };
                                AccessExpression {
                                    base_token: fun.expressions.append(crate::Expression::AccessMember {
                                        base: acex.base_token,
                                        index,
                                    }),
                                    type_id: *self.lookup_member_type_id
                                        .get(&(acex.type_id, index))
                                        .ok_or(ParseError::InvalidAccessType(acex.type_id))?,
                                }
                            }
                            crate::Type::Array { .. } |
                            crate::Type::Vector { .. } |
                            crate::Type::Matrix { .. } => {
                                AccessExpression {
                                    base_token: fun.expressions.append(crate::Expression::Access {
                                        base: acex.base_token,
                                        index: index_expr.token,
                                    }),
                                    type_id: ty.base_id
                                        .ok_or(ParseError::InvalidAccessType(acex.type_id))?,
                                }
                            }
                            ref other => return Err(ParseError::UnsupportedType(other.clone())),
                        };
                    }

                    self.lookup_expression.insert(id, LookupExpression {
                        token: acex.base_token,
                        type_id: result_type_id,
                    });
                }
                Op::Load => {
                    inst.expect_at_least(4)?;
                    let result_type_id = self.next()?;
                    let id = self.next()?;
                    let pointer_id = self.next()?;
                    if inst.wc != 4 {
                        inst.expect(5)?;
                        let _memory_access = self.next()?;
                    }
                    let base_expr = self.lookup_expression.lookup(pointer_id)?;
                    let base_type = self.lookup_type.lookup(base_expr.type_id)?;
                    if base_type.base_id != Some(result_type_id) {
                        return Err(ParseError::InvalidLoadType(result_type_id));
                    }
                    match base_type.value {
                        crate::Type::Pointer { .. } => (),
                        ref other => return Err(ParseError::UnsupportedType(other.clone())),
                    }
                    let expr = crate::Expression::Load {
                        pointer: base_expr.token,
                    };
                    self.lookup_expression.insert(id, LookupExpression {
                        token: fun.expressions.append(expr),
                        type_id: result_type_id,
                    });
                }
                _ => return Err(ParseError::UnsupportedInstruction(self.state, inst.op)),
            }
        }
    }

    fn make_expression_storage(&mut self) -> Storage<crate::Expression> {
        let mut expressions = Storage::new();
        assert!(self.lookup_expression.is_empty());
        // register global variables
        for (&id, var) in self.lookup_variable.iter() {
            self.lookup_expression.insert(id, LookupExpression {
                type_id: var.type_id,
                token: expressions.append(crate::Expression::GlobalVariable(var.token)),
            });
        }
        // register constants
        for (&id, con) in self.lookup_constant.iter() {
            self.lookup_expression.insert(id, LookupExpression {
                type_id: con.type_id,
                token: expressions.append(crate::Expression::Constant(con.value.clone())),
            });
        }
        // done
        expressions
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
            struct_declarations: Storage::new(),
            global_variables: Storage::new(),
            functions: Storage::new(),
            entry_points: Vec::new(),
        };
        let mut entry_points = Vec::new();

        while let Ok(inst) = self.next_inst() {
            use spirv::Op;
            log::debug!("\t{:?} [{}]", inst.op, inst.wc);
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
                    if exec_model > LAST_KNOWN_EXECUTION_MODEL as u32 {
                        return Err(ParseError::UnsupportedExecutionModel(exec_model));
                    }
                    let function_id = self.next()?;
                    let (name, left) = self.next_string(inst.wc - 3)?;
                    let ep = EntryPoint {
                        exec_model: unsafe {
                            std::mem::transmute(exec_model)
                        },
                        name: name.to_owned(),
                        function_id,
                        variable_ids: self.data
                            .by_ref()
                            .take(left as usize)
                            .collect(),
                    };
                    entry_points.push(ep);
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
                    self.lookup_type.insert(id, LookupType {
                        value: crate::Type::Void,
                        base_id: None,
                    });
                }
                Op::TypeInt => {
                    self.switch(ModuleState::Type, inst.op)?;
                    inst.expect(4)?;
                    let id = self.next()?;
                    let width = self.next()?;
                    let sign = self.next()?;
                    self.lookup_type.insert(id, LookupType {
                        value: crate::Type::Scalar {
                            kind: match sign {
                                0 => crate::ScalarKind::Uint,
                                1 => crate::ScalarKind::Sint,
                                _ => return Err(ParseError::InvalidSign(sign)),
                            },
                            width: width
                                .try_into()
                                .map_err(|_| ParseError::InvalidTypeWidth(width))?,
                        },
                        base_id: None,
                    });
                }
                Op::TypeFloat => {
                    self.switch(ModuleState::Type, inst.op)?;
                    inst.expect(3)?;
                    let id = self.next()?;
                    let width = self.next()?;
                    self.lookup_type.insert(id, LookupType {
                        value: crate::Type::Scalar {
                            kind: crate::ScalarKind::Float,
                            width: width
                                .try_into()
                                .map_err(|_| ParseError::InvalidTypeWidth(width))?,
                        },
                        base_id: None,
                    });
                }
                Op::TypeVector => {
                    self.switch(ModuleState::Type, inst.op)?;
                    inst.expect(4)?;
                    let id = self.next()?;
                    let type_id = self.next()?;
                    let (kind, width) = match self.lookup_type.lookup(type_id)?.value {
                        crate::Type::Scalar { kind, width } => (kind, width),
                        _ => return Err(ParseError::InvalidInnerType(type_id)),
                    };
                    let component_count = self.next()?;
                    self.lookup_type.insert(id, LookupType {
                        value: crate::Type::Vector {
                            size: map_vector_size(component_count)?,
                            kind,
                            width,
                        },
                        base_id: Some(type_id),
                    });
                }
                Op::TypeMatrix => {
                    self.switch(ModuleState::Type, inst.op)?;
                    inst.expect(4)?;
                    let id = self.next()?;
                    let vector_type_id = self.next()?;
                    let num_columns = self.next()?;
                    let (rows, kind, width) = match self.lookup_type.lookup(vector_type_id)?.value {
                        crate::Type::Vector { size, kind, width } => (size, kind, width),
                        _ => return Err(ParseError::InvalidInnerType(vector_type_id)),
                    };
                    self.lookup_type.insert(id, LookupType {
                        value: crate::Type::Matrix {
                            columns: map_vector_size(num_columns)?,
                            rows,
                            kind,
                            width,
                        },
                        base_id: Some(vector_type_id),
                    });
                }
                Op::TypeFunction => {
                    self.switch(ModuleState::Type, inst.op)?;
                    inst.expect_at_least(3)?;
                    let id = self.next()?;
                    let return_type_id = self.next()?;
                    let parameter_type_ids = self.data
                        .by_ref()
                        .take(inst.wc as usize - 3)
                        .collect();
                    self.lookup_function_type.insert(id, LookupFunctionType {
                        parameter_type_ids,
                        return_type_id,
                    });
                }
                Op::TypePointer => {
                    self.switch(ModuleState::Type, inst.op)?;
                    inst.expect(4)?;
                    let id = self.next()?;
                    let storage = self.next()?;
                    let type_id = self.next()?;
                    let value = crate::Type::Pointer {
                        base: Box::new(self.lookup_type.lookup(type_id)?.value.clone()),
                        class: map_storage_class(storage)?,
                    };
                    self.lookup_type.insert(id, LookupType {
                        value,
                        base_id: Some(type_id),
                    });
                }
                Op::TypeArray => {
                    self.switch(ModuleState::Type, inst.op)?;
                    inst.expect(4)?;
                    let id = self.next()?;
                    let type_id = self.next()?;
                    let length = self.next()?;
                    let value = crate::Type::Array {
                        base: Box::new(self.lookup_type.lookup(type_id)?.value.clone()),
                        length,
                    };
                    self.lookup_type.insert(id, LookupType {
                        value,
                        base_id: Some(type_id),
                    });
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
                        let ty = self.lookup_type.lookup(type_id)?.value.clone();
                        self.lookup_member_type_id.insert((id, i), type_id);
                        decl.members.push(crate::StructMember {
                            name: self.future_member_decor
                                .remove(&(id, i))
                                .and_then(|dec| dec.name),
                            ty,
                        });
                    }
                    let value = crate::Type::Struct(module.struct_declarations.append(decl));
                    self.lookup_type.insert(id, LookupType {
                        value,
                        base_id: None,
                    });
                }
                Op::Constant => {
                    self.switch(ModuleState::Type, inst.op)?;
                    inst.expect_at_least(3)?;
                    let type_id = self.next()?;
                    let id = self.next()?;
                    let value = match self.lookup_type.lookup(type_id)?.value {
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
                    self.lookup_constant.insert(id, LookupConstant {
                        value,
                        type_id,
                    });
                }
                Op::Variable => {
                    self.switch(ModuleState::Type, inst.op)?;
                    inst.expect_at_least(4)?;
                    let type_id = self.next()?;
                    let id = self.next()?;
                    let storage = self.next()?;
                    if inst.wc != 4 {
                        inst.expect(5)?;
                        let _init = self.next()?; //TODO
                    }
                    let var = crate::GlobalVariable {
                        name: self.future_decor
                            .remove(&id)
                            .and_then(|dec| dec.name),
                        class: map_storage_class(storage)?,
                        ty: self.lookup_type.lookup(type_id)?.value.clone(),
                    };
                    let token = module.global_variables.append(var);
                    self.lookup_variable.insert(id, LookupVariable {
                        token,
                        type_id,
                    });
                }
                Op::Function => {
                    self.switch(ModuleState::Function, inst.op)?;
                    inst.expect(5)?;
                    let result_type = self.next()?;
                    let fun_id = self.next()?;
                    let fun_control = self.next()?;
                    let fun_type = self.next()?;
                    let mut fun = {
                        let ft = self.lookup_function_type.lookup(fun_type)?.clone();
                        if ft.return_type_id != result_type {
                            return Err(ParseError::WrongFunctionResultType(result_type))
                        }
                        crate::Function {
                            name: None,
                            control: spirv::FunctionControl::from_bits(fun_control)
                                .ok_or(ParseError::UnsupportedFunctionControl(fun_control))?,
                            parameter_types: Vec::with_capacity(ft.parameter_type_ids.len()),
                            return_type: self.lookup_type.lookup(result_type)?.value.clone(),
                            expressions: self.make_expression_storage(),
                            body: Vec::new(),
                        }
                    };
                    // read parameters
                    for i in 0 .. fun.parameter_types.capacity() {
                        match self.next_inst()? {
                            Instruction { op: Op::FunctionParameter, wc: 3 } => {
                                let type_id = self.next()?;
                                let _id = self.next()?;
                                //Note: we redo the lookup in order to work around `self` borrowing
                                if type_id != self.lookup_function_type
                                    .lookup(fun_type)?
                                    .parameter_type_ids[i]
                                {
                                    return Err(ParseError::WrongFunctionParameterType(type_id))
                                }
                                let ty = self.lookup_type.lookup(type_id)?.value.clone();
                                fun.parameter_types.push(ty);
                            }
                            Instruction { op, .. } => return Err(ParseError::InvalidParameter(op)),
                        }
                    }
                    // read body
                    loop {
                        let fun_inst = self.next_inst()?;
                        log::debug!("\t\t{:?}", fun_inst.op);
                        match fun_inst.op {
                            Op::Label => {
                                fun_inst.expect(2)?;
                                let _id = self.next()?;
                                self.next_block(&mut fun)?;
                                break
                            }
                            Op::FunctionEnd => {
                                fun_inst.expect(1)?;
                                break
                            }
                            _ => return Err(ParseError::UnsupportedInstruction(self.state, fun_inst.op))
                        }
                    }
                    // done
                    let token = module.functions.append(fun);
                    self.lookup_function.insert(fun_id, token);
                    self.lookup_expression.clear();
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

        module.entry_points.reserve(entry_points.len());
        for raw in entry_points {
            let mut ep = crate::EntryPoint {
                exec_model: raw.exec_model,
                name: raw.name,
                function: *self.lookup_function.lookup(raw.function_id)?,
                inputs: Vec::new(),
                outputs: Vec::new(),
            };
            for var_id in raw.variable_ids {
                let token = self.lookup_variable.lookup(var_id)?.token;
                match module.global_variables[token].class {
                    spirv::StorageClass::Input => ep.inputs.push(token),
                    spirv::StorageClass::Output => ep.outputs.push(token),
                    other => return Err(ParseError::InvalidVariableClass(other))
                }
            }
            module.entry_points.push(ep);
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
