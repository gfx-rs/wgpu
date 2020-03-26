/*! SPIR-V frontend

## ID lookups

Our IR links to everything with `Handle`, while SPIR-V uses IDs.
In order to keep track of the associations, the parser has many lookup tables.
There map `spirv::Word` into a specific IR handle, plus potentially a bit of
extra info, such as the related SPIR-V type ID.
TODO: would be nice to find ways that avoid looking up as much

!*/

use crate::{
    arena::{Arena, Handle},
    FastHashMap, FastHashSet,
};

use num_traits::cast::FromPrimitive;
use std::convert::TryInto;

pub const SUPPORTED_CAPABILITIES: &[spirv::Capability] = &[
    spirv::Capability::Shader,
];
pub const SUPPORTED_EXTENSIONS: &[&str] = &[
];
pub const SUPPORTED_EXT_SETS: &[&str] = &[
    "GLSL.std.450",
];

#[derive(Debug)]
pub enum Error {
    InvalidHeader,
    InvalidWordCount,
    UnknownInstruction(u16),
    UnknownCapability(u32),
    UnsupportedInstruction(ModuleState, spirv::Op),
    UnsupportedCapability(spirv::Capability),
    UnsupportedExtension(String),
    UnsupportedExtSet(String),
    UnsupportedType(Handle<crate::Type>),
    UnsupportedExecutionModel(u32),
    UnsupportedStorageClass(u32),
    UnsupportedFunctionControl(u32),
    UnsupportedDim(u32),
    InvalidParameter(spirv::Op),
    InvalidOperandCount(spirv::Op, u16),
    InvalidOperand,
    InvalidDecoration(spirv::Word),
    InvalidId(spirv::Word),
    InvalidTypeWidth(spirv::Word),
    InvalidSign(spirv::Word),
    InvalidInnerType(spirv::Word),
    InvalidVectorSize(spirv::Word),
    InvalidVariableClass(spirv::StorageClass),
    InvalidAccessType(spirv::Word),
    InvalidAccessIndex(Handle<crate::Expression>),
    InvalidLoadType(spirv::Word),
    InvalidStoreType(spirv::Word),
    InvalidBinding(spirv::Word),
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
    fn expect(&self, count: u16) -> Result<(), Error> {
        if self.wc == count {
            Ok(())
        } else {
            Err(Error::InvalidOperandCount(self.op, self.wc))
        }
    }

    fn expect_at_least(&self, count: u16) -> Result<(), Error> {
        if self.wc >= count {
            Ok(())
        } else {
            Err(Error::InvalidOperandCount(self.op, self.wc))
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

trait LookupHelper {
    type Target;
    fn lookup(&self, key: spirv::Word) -> Result<&Self::Target, Error>;
}

impl<T> LookupHelper for FastHashMap<spirv::Word, T> {
    type Target = T;
    fn lookup(&self, key: spirv::Word) -> Result<&T, Error> {
        self.get(&key).ok_or(Error::InvalidId(key))
    }
}

fn map_vector_size(word: spirv::Word) -> Result<crate::VectorSize, Error> {
    match word {
        2 => Ok(crate::VectorSize::Bi),
        3 => Ok(crate::VectorSize::Tri),
        4 => Ok(crate::VectorSize::Quad),
        _ => Err(Error::InvalidVectorSize(word)),
    }
}

fn map_storage_class(word: spirv::Word) -> Result<spirv::StorageClass, Error> {
    spirv::StorageClass::from_u32(word).ok_or(Error::UnsupportedStorageClass(word))
}

type MemberIndex = u32;

#[derive(Debug, Default)]
struct Decoration {
    name: Option<String>,
    built_in: Option<spirv::BuiltIn>,
    location: Option<spirv::Word>,
    desc_set: Option<spirv::Word>,
    desc_index: Option<spirv::Word>,
}

impl Decoration {
    fn get_binding(&self) -> Option<crate::Binding> {
        //TODO: validate this better
        match *self {
            Decoration {
                built_in: Some(built_in),
                location: None,
                desc_set: None,
                desc_index: None,
                ..
            } => Some(crate::Binding::BuiltIn(built_in)),
            Decoration {
                built_in: None,
                location: Some(loc),
                desc_set: None,
                desc_index: None,
                ..
            } => Some(crate::Binding::Location(loc)),
            Decoration {
                built_in: None,
                location: None,
                desc_set: Some(set),
                desc_index: Some(binding),
                ..
            } => Some(crate::Binding::Descriptor { set, binding }),
            _ => None,
        }
    }
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
    handle: Handle<crate::Type>,
    base_id: Option<spirv::Word>,
}

#[derive(Debug)]
struct LookupConstant {
    handle: Handle<crate::Constant>,
    type_id: spirv::Word,
}

#[derive(Debug)]
struct LookupVariable {
    handle: Handle<crate::GlobalVariable>,
    type_id: spirv::Word,
}

#[derive(Clone, Debug)]
struct LookupExpression {
    handle: Handle<crate::Expression>,
    type_id: spirv::Word,
}

#[derive(Clone, Debug)]
struct LookupSampledImage {
    image: Handle<crate::Expression>,
    sampler: Handle<crate::Expression>,
}

pub struct Parser<I> {
    data: I,
    state: ModuleState,
    temp_bytes: Vec<u8>,
    future_decor: FastHashMap<spirv::Word, Decoration>,
    future_member_decor: FastHashMap<(spirv::Word, MemberIndex), Decoration>,
    lookup_member_type_id: FastHashMap<(spirv::Word, MemberIndex), spirv::Word>,
    lookup_type: FastHashMap<spirv::Word, LookupType>,
    lookup_void_type: FastHashSet<spirv::Word>,
    lookup_constant: FastHashMap<spirv::Word, LookupConstant>,
    lookup_variable: FastHashMap<spirv::Word, LookupVariable>,
    lookup_expression: FastHashMap<spirv::Word, LookupExpression>,
    lookup_sampled_image: FastHashMap<spirv::Word, LookupSampledImage>,
    lookup_function_type: FastHashMap<spirv::Word, LookupFunctionType>,
    lookup_function: FastHashMap<spirv::Word, Handle<crate::Function>>,
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
            lookup_void_type: FastHashSet::default(),
            lookup_constant: FastHashMap::default(),
            lookup_variable: FastHashMap::default(),
            lookup_expression: FastHashMap::default(),
            lookup_sampled_image: FastHashMap::default(),
            lookup_function_type: FastHashMap::default(),
            lookup_function: FastHashMap::default(),
        }
    }

    fn next(&mut self) -> Result<u32, Error> {
        self.data.next().ok_or(Error::IncompleteData)
    }

    fn next_inst(&mut self) -> Result<Instruction, Error> {
        let word = self.next()?;
        let (wc, opcode) = ((word >> 16) as u16, (word & 0xffff) as u16);
        if wc == 0 {
            return Err(Error::InvalidWordCount);
        }
        let op = spirv::Op::from_u16(opcode).ok_or(Error::UnknownInstruction(opcode))?;

        Ok(Instruction { op, wc })
    }

    fn next_string(&mut self, mut count: u16) -> Result<(String, u16), Error>{
        self.temp_bytes.clear();
        loop {
            if count == 0 {
                return Err(Error::BadString);
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
            .map_err(|_| Error::BadString)
    }

    fn next_decoration(
        &mut self,
        inst: Instruction,
        base_words: u16,
        dec: &mut Decoration,
    ) -> Result<(), Error> {
        let raw = self.next()?;
        let dec_typed = spirv::Decoration::from_u32(raw).ok_or(Error::InvalidDecoration(raw))?;
        log::trace!("\t\t{:?}", dec_typed);
        match dec_typed {
            spirv::Decoration::BuiltIn => {
                inst.expect(base_words + 2)?;
                let raw = self.next()?;
                let built_in = spirv::BuiltIn::from_u32(raw);
                if built_in.is_none() {
                    log::warn!("Unknown built in {:?}", raw);
                }
                dec.built_in = built_in;
            }
            spirv::Decoration::Location => {
                inst.expect(base_words + 2)?;
                dec.location = Some(self.next()?);
            }
            spirv::Decoration::DescriptorSet => {
                inst.expect(base_words + 2)?;
                dec.desc_set = Some(self.next()?);
            }
            spirv::Decoration::Binding => {
                inst.expect(base_words + 2)?;
                dec.desc_index = Some(self.next()?);
            }
            other => {
                log::warn!("Unknown decoration {:?}", other);
                for _ in base_words + 1 .. inst.wc {
                    let _var = self.next()?;
                }
            }
        }
        Ok(())
    }

    fn next_block(
        &mut self,
        fun: &mut crate::Function,
        type_arena: &Arena<crate::Type>,
        const_arena: &Arena<crate::Constant>,
    ) -> Result<(), Error> {
        loop {
            use spirv::Op;
            let inst = self.next_inst()?;
            log::debug!("\t\t{:?} [{}]", inst.op, inst.wc);
            match inst.op {
                Op::AccessChain => {
                    struct AccessExpression {
                        base_handle: Handle<crate::Expression>,
                        type_id: spirv::Word,
                    }
                    inst.expect_at_least(4)?;
                    let result_type_id = self.next()?;
                    let result_id = self.next()?;
                    let base_id = self.next()?;
                    log::trace!("\t\t\tlooking up expr {:?}", base_id);
                    let mut acex = {
                        let expr = self.lookup_expression.lookup(base_id)?;
                        let ptr_type = self.lookup_type.lookup(expr.type_id)?;
                        AccessExpression {
                            base_handle: expr.handle,
                            type_id: ptr_type.base_id.unwrap(),
                        }
                    };
                    for _ in 4 .. inst.wc {
                        let access_id = self.next()?;
                        log::trace!("\t\t\tlooking up expr {:?}", access_id);
                        let index_expr = self.lookup_expression.lookup(access_id)?.clone();
                        let index_type_handle = self.lookup_type.lookup(index_expr.type_id)?.handle;
                        match type_arena[index_type_handle].inner {
                            crate::TypeInner::Scalar { kind: crate::ScalarKind::Uint, .. } |
                            crate::TypeInner::Scalar { kind: crate::ScalarKind::Sint, .. } => (),
                            _ => return Err(Error::UnsupportedType(index_type_handle)),
                        }
                        log::trace!("\t\t\tlooking up type {:?}", acex.type_id);
                        let type_lookup = self.lookup_type.lookup(acex.type_id)?;
                        acex = match type_arena[type_lookup.handle].inner {
                            crate::TypeInner::Struct { .. } => {
                                let index = match fun.expressions[index_expr.handle] {
                                    crate::Expression::Constant(const_handle) => {
                                        match const_arena[const_handle].inner {
                                            crate::ConstantInner::Uint(v) => v as u32,
                                            crate::ConstantInner::Sint(v) => v as u32,
                                            _ => return Err(Error::InvalidAccessIndex(index_expr.handle)),
                                        }
                                    }
                                    _ => return Err(Error::InvalidAccessIndex(index_expr.handle))
                                };
                                AccessExpression {
                                    base_handle: fun.expressions.append(crate::Expression::AccessIndex {
                                        base: acex.base_handle,
                                        index,
                                    }),
                                    type_id: *self.lookup_member_type_id
                                        .get(&(acex.type_id, index))
                                        .ok_or(Error::InvalidAccessType(acex.type_id))?,
                                }
                            }
                            crate::TypeInner::Array { .. } |
                            crate::TypeInner::Vector { .. } |
                            crate::TypeInner::Matrix { .. } => {
                                AccessExpression {
                                    base_handle: fun.expressions.append(crate::Expression::Access {
                                        base: acex.base_handle,
                                        index: index_expr.handle,
                                    }),
                                    type_id: type_lookup.base_id
                                        .ok_or(Error::InvalidAccessType(acex.type_id))?,
                                }
                            }
                            _ => return Err(Error::UnsupportedType(type_lookup.handle)),
                        };
                    }

                    self.lookup_expression.insert(result_id, LookupExpression {
                        handle: acex.base_handle,
                        type_id: result_type_id,
                    });
                }
                Op::CompositeExtract => {
                    inst.expect_at_least(4)?;
                    let result_type_id = self.next()?;
                    let result_id = self.next()?;
                    let base_id = self.next()?;
                    log::trace!("\t\t\tlooking up expr {:?}", base_id);
                    let mut lexp = {
                        let expr = self.lookup_expression.lookup(base_id)?;
                        LookupExpression {
                            handle: expr.handle,
                            type_id: expr.type_id,
                        }
                    };
                    for _ in 4 .. inst.wc {
                        let index = self.next()?;
                        log::trace!("\t\t\tlooking up type {:?}", lexp.type_id);
                        let type_lookup = self.lookup_type.lookup(lexp.type_id)?;
                        let type_id = match type_arena[type_lookup.handle].inner {
                            crate::TypeInner::Struct { .. } => {
                                *self.lookup_member_type_id
                                    .get(&(lexp.type_id, index))
                                    .ok_or(Error::InvalidAccessType(lexp.type_id))?
                            }
                            crate::TypeInner::Array { .. } |
                            crate::TypeInner::Vector { .. } |
                            crate::TypeInner::Matrix { .. } => {
                                type_lookup.base_id
                                    .ok_or(Error::InvalidAccessType(lexp.type_id))?
                            }
                            _ => return Err(Error::UnsupportedType(type_lookup.handle)),
                        };
                        lexp = LookupExpression {
                            handle: fun.expressions.append(crate::Expression::AccessIndex {
                                base: lexp.handle,
                                index,
                            }),
                            type_id,
                        };
                    }

                    self.lookup_expression.insert(result_id, LookupExpression {
                        handle: lexp.handle,
                        type_id: result_type_id,
                    });
                }
                Op::CompositeConstruct => {
                    inst.expect_at_least(3)?;
                    let result_type_id = self.next()?;
                    let id = self.next()?;
                    let mut components = Vec::with_capacity(inst.wc as usize  - 2);
                    for _ in 3 .. inst.wc {
                        let comp_id = self.next()?;
                        log::trace!("\t\t\tlooking up expr {:?}", comp_id);
                        let lexp = self.lookup_expression.lookup(comp_id)?;
                        components.push(lexp.handle);
                    }
                    let expr = crate::Expression::Compose {
                        ty: self.lookup_type.lookup(result_type_id)?.handle,
                        components,
                    };
                    self.lookup_expression.insert(id, LookupExpression {
                        handle: fun.expressions.append(expr),
                        type_id: result_type_id,
                    });
                }
                Op::Load => {
                    inst.expect_at_least(4)?;
                    let result_type_id = self.next()?;
                    let result_id = self.next()?;
                    let pointer_id = self.next()?;
                    if inst.wc != 4 {
                        inst.expect(5)?;
                        let _memory_access = self.next()?;
                    }
                    let base_expr = self.lookup_expression.lookup(pointer_id)?;
                    let base_type = self.lookup_type.lookup(base_expr.type_id)?;
                    if base_type.base_id != Some(result_type_id) {
                        return Err(Error::InvalidLoadType(result_type_id));
                    }
                    match type_arena[base_type.handle].inner {
                        crate::TypeInner::Pointer { .. } => (),
                        _ => return Err(Error::UnsupportedType(base_type.handle)),
                    }
                    let expr = crate::Expression::Load {
                        pointer: base_expr.handle,
                    };
                    self.lookup_expression.insert(result_id, LookupExpression {
                        handle: fun.expressions.append(expr),
                        type_id: result_type_id,
                    });
                }
                Op::Store => {
                    inst.expect_at_least(3)?;
                    let pointer_id = self.next()?;
                    let value_id = self.next()?;
                    if inst.wc != 3 {
                        inst.expect(4)?;
                        let _memory_access = self.next()?;
                    }
                    let base_expr = self.lookup_expression.lookup(pointer_id)?;
                    let base_type = self.lookup_type.lookup(base_expr.type_id)?;
                    match type_arena[base_type.handle].inner {
                        crate::TypeInner::Pointer { .. } => (),
                        _ => return Err(Error::UnsupportedType(base_type.handle)),
                    };
                    let value_expr = self.lookup_expression.lookup(value_id)?;
                    if base_type.base_id != Some(value_expr.type_id) {
                        return Err(Error::InvalidStoreType(value_expr.type_id));
                    }
                    fun.body.push(crate::Statement::Store {
                        pointer: base_expr.handle,
                        value: value_expr.handle,
                    })
                }
                Op::Return => {
                    inst.expect(1)?;
                    fun.body.push(crate::Statement::Return { value: None });
                    break
                }
                Op::VectorTimesScalar => {
                    inst.expect(5)?;
                    let result_type_id = self.next()?;
                    let result_type_loookup = self.lookup_type.lookup(result_type_id)?;
                    let (res_size, res_width) = match type_arena[result_type_loookup.handle].inner {
                        crate::TypeInner::Vector { size, kind: crate::ScalarKind::Float, width } => (size, width),
                        _ => return Err(Error::UnsupportedType(result_type_loookup.handle)),
                    };
                    let result_id = self.next()?;
                    let vector_id = self.next()?;
                    let scalar_id = self.next()?;
                    let vector_lexp = self.lookup_expression.lookup(vector_id)?;
                    let vector_type_lookup = self.lookup_type.lookup(vector_lexp.type_id)?;
                    match type_arena[vector_type_lookup.handle].inner {
                        crate::TypeInner::Vector { size, kind: crate::ScalarKind::Float, width } if size == res_size && width == res_width => (),
                        _ => return Err(Error::UnsupportedType(vector_type_lookup.handle)),
                    };
                    let scalar_lexp = self.lookup_expression.lookup(scalar_id)?.clone();
                    let scalar_type_lookup = self.lookup_type.lookup(scalar_lexp.type_id)?;
                    match type_arena[scalar_type_lookup.handle].inner {
                        crate::TypeInner::Scalar { kind: crate::ScalarKind::Float, width } if width == res_width => (),
                        _ => return Err(Error::UnsupportedType(scalar_type_lookup.handle)),
                    };
                    let expr = crate::Expression::Binary {
                        op: crate::BinaryOperator::Multiply,
                        left: vector_lexp.handle,
                        right: scalar_lexp.handle,
                    };
                    self.lookup_expression.insert(result_id, LookupExpression {
                        handle: fun.expressions.append(expr),
                        type_id: result_type_id,
                    });
                }
                Op::MatrixTimesVector => {
                    inst.expect(5)?;
                    let result_type_id = self.next()?;
                    let result_type_loookup = self.lookup_type.lookup(result_type_id)?;
                    let (res_size, res_width) = match type_arena[result_type_loookup.handle].inner {
                        crate::TypeInner::Vector { size, kind: crate::ScalarKind::Float, width } => (size, width),
                        _ => return Err(Error::UnsupportedType(result_type_loookup.handle)),
                    };
                    let result_id = self.next()?;
                    let matrix_id = self.next()?;
                    let vector_id = self.next()?;
                    let matrix_lexp = self.lookup_expression.lookup(matrix_id)?;
                    let matrix_type_lookup = self.lookup_type.lookup(matrix_lexp.type_id)?;
                    let columns = match type_arena[matrix_type_lookup.handle].inner {
                        crate::TypeInner::Matrix { columns, rows, kind: crate::ScalarKind::Float, width } if rows == res_size && width == res_width => columns,
                        _ => return Err(Error::UnsupportedType(matrix_type_lookup.handle)),
                    };
                    let vector_lexp = self.lookup_expression.lookup(vector_id)?.clone();
                    let vector_type_lookup = self.lookup_type.lookup(vector_lexp.type_id)?;
                    match type_arena[vector_type_lookup.handle].inner {
                        crate::TypeInner::Vector { size, kind: crate::ScalarKind::Float, width } if size == columns && width == res_width => (),
                        _ => return Err(Error::UnsupportedType(vector_type_lookup.handle)),
                    };
                    let expr = crate::Expression::Binary {
                        op: crate::BinaryOperator::Multiply,
                        left: matrix_lexp.handle,
                        right: vector_lexp.handle,
                    };
                    self.lookup_expression.insert(result_id, LookupExpression {
                        handle: fun.expressions.append(expr),
                        type_id: result_type_id,
                    });
                }
                Op::SampledImage => {
                    inst.expect(5)?;
                    let _result_type_id = self.next()?;
                    let result_id = self.next()?;
                    let image_id = self.next()?;
                    let sampler_id = self.next()?;
                    let image_lexp = self.lookup_expression.lookup(image_id)?;
                    let sampler_lexp = self.lookup_expression.lookup(sampler_id)?;
                    //TODO: compare the result type
                    self.lookup_sampled_image.insert(result_id, LookupSampledImage {
                        image: image_lexp.handle,
                        sampler: sampler_lexp.handle,
                    });
                }
                Op::ImageSampleImplicitLod => {
                    inst.expect_at_least(5)?;
                    let result_type_id = self.next()?;
                    let result_id = self.next()?;
                    let sampled_image_id = self.next()?;
                    let coordinate_id = self.next()?;
                    let si_lexp = self.lookup_sampled_image.lookup(sampled_image_id)?;
                    let coord_lexp = self.lookup_expression.lookup(coordinate_id)?;
                    let coord_type_lookup = self.lookup_type.lookup(coord_lexp.type_id)?;
                    match type_arena[coord_type_lookup.handle].inner {
                        crate::TypeInner::Scalar { kind: crate::ScalarKind::Float, .. } |
                        crate::TypeInner::Vector { kind: crate::ScalarKind::Float, .. } => (),
                        _ => return Err(Error::UnsupportedType(coord_type_lookup.handle)),
                    }
                    //TODO: compare the result type
                    let expr = crate::Expression::ImageSample {
                        image: si_lexp.image,
                        sampler: si_lexp.sampler,
                        coordinate: coord_lexp.handle,
                    };
                    self.lookup_expression.insert(result_id, LookupExpression {
                        handle: fun.expressions.append(expr),
                        type_id: result_type_id,
                    });
                }
                _ => return Err(Error::UnsupportedInstruction(self.state, inst.op)),
            }
        }
        Ok(())
    }

    fn make_expression_storage(&mut self) -> Arena<crate::Expression> {
        let mut expressions = Arena::new();
        assert!(self.lookup_expression.is_empty());
        // register global variables
        for (&id, var) in self.lookup_variable.iter() {
            self.lookup_expression.insert(id, LookupExpression {
                type_id: var.type_id,
                handle: expressions.append(crate::Expression::GlobalVariable(var.handle)),
            });
        }
        // register constants
        for (&id, con) in self.lookup_constant.iter() {
            self.lookup_expression.insert(id, LookupExpression {
                type_id: con.type_id,
                handle: expressions.append(crate::Expression::Constant(con.handle)),
            });
        }
        // done
        expressions
    }

    fn switch(&mut self, state: ModuleState, op: spirv::Op) -> Result<(), Error> {
        if state < self.state {
            Err(Error::UnsupportedInstruction(self.state, op))
        } else {
            self.state = state;
            Ok(())
        }
    }

    pub fn parse(&mut self) -> Result<crate::Module, Error> {
        let mut module = crate::Module::from_header({
            if self.next()? != spirv::MAGIC_NUMBER {
                return Err(Error::InvalidHeader);
            }
            let version_raw = self.next()?.to_le_bytes();
            let generator = self.next()?;
            let _bound = self.next()?;
            let _schema = self.next()?;
            crate::Header {
                version: (version_raw[2], version_raw[1], version_raw[0]),
                generator,
            }
        });
        let mut entry_points = Vec::new();

        while let Ok(inst) = self.next_inst() {
            use spirv::Op;
            log::debug!("\t{:?} [{}]", inst.op, inst.wc);
            match inst.op {
                Op::Capability => self.parse_capability(inst),
                Op::Extension => self.parse_extension(inst),
                Op::ExtInstImport => self.parse_ext_inst_import(inst),
                Op::MemoryModel => self.parse_memory_model(inst),
                Op::EntryPoint => self.parse_entry_point(inst, &mut entry_points),
                Op::ExecutionMode => self.parse_execution_mode(inst),
                Op::Source => self.parse_source(inst),
                Op::SourceExtension => self.parse_source_extension(inst),
                Op::Name => self.parse_name(inst),
                Op::MemberName => self.parse_member_name(inst),
                Op::Decorate => self.parse_decorate(inst),
                Op::MemberDecorate => self.parse_member_decorate(inst),
                Op::TypeVoid => self.parse_type_void(inst),
                Op::TypeInt => self.parse_type_int(inst, &mut module),
                Op::TypeFloat => self.parse_type_float(inst, &mut module),
                Op::TypeVector => self.parse_type_vector(inst, &mut module),
                Op::TypeMatrix => self.parse_type_matrix(inst, &mut module),
                Op::TypeFunction => self.parse_type_function(inst),
                Op::TypePointer => self.parse_type_pointer(inst, &mut module),
                Op::TypeArray => self.parse_type_array(inst, &mut module),
                Op::TypeRuntimeArray => self.parse_type_runtime_array(inst, &mut module),
                Op::TypeStruct => self.parse_type_struct(inst, &mut module),
                Op::TypeImage => self.parse_type_image(inst, &mut module),
                Op::TypeSampledImage => self.parse_type_sampled_image(inst),
                Op::TypeSampler => self.parse_type_sampler(inst, &mut module),
                Op::Constant | Op::SpecConstant => self.parse_constant(inst, &mut module),
                Op::ConstantComposite => self.parse_composite_constant(inst, &mut module),
                Op::Variable => self.parse_variable(inst, &mut module),
                Op::Function => self.parse_function(inst, &mut module),
                _ => Err(Error::UnsupportedInstruction(self.state, inst.op)), //TODO
            }?;
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
            module.entry_points.push(crate::EntryPoint {
                exec_model: raw.exec_model,
                name: raw.name,
                function: *self.lookup_function.lookup(raw.function_id)?,
            });
        }

        Ok(module)
    }

    fn parse_capability(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::Capability, inst.op)?;
        inst.expect(2)?;
        let capability = self.next()?;
        let cap =
            spirv::Capability::from_u32(capability).ok_or(Error::UnknownCapability(capability))?;
        if !SUPPORTED_CAPABILITIES.contains(&cap) {
            return Err(Error::UnsupportedCapability(cap));
        }
        Ok(())
    }

    fn parse_extension(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::Extension, inst.op)?;
        inst.expect_at_least(2)?;
        let (name, left) = self.next_string(inst.wc - 1)?;
        if left != 0 {
            return Err(Error::InvalidOperand);
        }
        if !SUPPORTED_EXTENSIONS.contains(&name.as_str()) {
            return Err(Error::UnsupportedExtension(name.to_owned()));
        }
        Ok(())
    }

    fn parse_ext_inst_import(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::Extension, inst.op)?;
        inst.expect_at_least(3)?;
        let _result = self.next()?;
        let (name, left) = self.next_string(inst.wc - 2)?;
        if left != 0 {
            return Err(Error::InvalidOperand);
        }
        if !SUPPORTED_EXT_SETS.contains(&name.as_str()) {
            return Err(Error::UnsupportedExtSet(name.to_owned()));
        }
        Ok(())
    }

    fn parse_memory_model(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::MemoryModel, inst.op)?;
        inst.expect(3)?;
        let _addressing_model = self.next()?;
        let _memory_model = self.next()?;
        Ok(())
    }

    fn parse_entry_point(
        &mut self,
        inst: Instruction,
        entry_points: &mut Vec<EntryPoint>,
    ) -> Result<(), Error> {
        self.switch(ModuleState::EntryPoint, inst.op)?;
        inst.expect_at_least(4)?;
        let exec_model = self.next()?;
        let exec_model = spirv::ExecutionModel::from_u32(exec_model)
            .ok_or(Error::UnsupportedExecutionModel(exec_model))?;
        let function_id = self.next()?;
        let (name, left) = self.next_string(inst.wc - 3)?;
        let ep = EntryPoint {
            exec_model,
            name: name.to_owned(),
            function_id,
            variable_ids: self.data
                .by_ref()
                .take(left as usize)
                .collect(),
        };
        entry_points.push(ep);
        Ok(())
    }

    fn parse_execution_mode(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::ExecutionMode, inst.op)?;
        inst.expect_at_least(3)?;
        let _ep_id = self.next()?;
        let _mode = self.next()?;
        for _ in 3..inst.wc {
            let _ = self.next()?; //TODO
        }
        Ok(())
    }

    fn parse_source(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::Source, inst.op)?;
        for _ in 1 .. inst.wc {
            let _ = self.next()?;
        }
        Ok(())
    }

    fn parse_source_extension(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::Source, inst.op)?;
        inst.expect_at_least(2)?;
        let (_name, _) = self.next_string(inst.wc - 1)?;
        Ok(())
    }

    fn parse_name(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::Name, inst.op)?;
        inst.expect_at_least(3)?;
        let id = self.next()?;
        let (name, left) = self.next_string(inst.wc - 2)?;
        if left != 0 {
            return Err(Error::InvalidOperand);
        }
        self.future_decor
            .entry(id)
            .or_default()
            .name = Some(name.to_owned());
        Ok(())
    }

    fn parse_member_name(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::Name, inst.op)?;
        inst.expect_at_least(4)?;
        let id = self.next()?;
        let member = self.next()?;
        let (name, left) = self.next_string(inst.wc - 3)?;
        if left != 0 {
            return Err(Error::InvalidOperand);
        }
        self.future_member_decor
            .entry((id, member))
            .or_default()
            .name = Some(name.to_owned());
        Ok(())
    }

    fn parse_decorate(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::Annotation, inst.op)?;
        inst.expect_at_least(3)?;
        let id = self.next()?;
        let mut dec = self.future_decor
            .remove(&id)
            .unwrap_or_default();
        self.next_decoration(inst, 2, &mut dec)?;
        self.future_decor.insert(id, dec);
        Ok(())
    }

    fn parse_member_decorate(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::Annotation, inst.op)?;
        inst.expect_at_least(4)?;
        let id = self.next()?;
        let member = self.next()?;
        let mut dec = self.future_member_decor
            .remove(&(id, member))
            .unwrap_or_default();
        self.next_decoration(inst, 3, &mut dec)?;
        self.future_member_decor.insert((id, member), dec);
        Ok(())
    }

    fn parse_type_void(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect(2)?;
        let id = self.next()?;
        self.lookup_void_type.insert(id);
        Ok(())
    }

    fn parse_type_int(
        &mut self,
        inst: Instruction,
        module: &mut crate::Module,
    ) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect(4)?;
        let id = self.next()?;
        let width = self.next()?;
        let sign = self.next()?;
        let inner = crate::TypeInner::Scalar {
            kind: match sign {
                0 => crate::ScalarKind::Uint,
                1 => crate::ScalarKind::Sint,
                _ => return Err(Error::InvalidSign(sign)),
            },
            width: width
                .try_into()
                .map_err(|_| Error::InvalidTypeWidth(width))?,
        };
        self.lookup_type.insert(id, LookupType {
            handle: module.types.append(crate::Type {
                name: self.future_decor
                    .remove(&id)
                    .and_then(|dec| dec.name),
                inner,
            }),
            base_id: None,
        });
        Ok(())
    }

    fn parse_type_float(
        &mut self,
        inst: Instruction,
        module: &mut crate::Module,
    ) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect(3)?;
        let id = self.next()?;
        let width = self.next()?;
        let inner = crate::TypeInner::Scalar {
            kind: crate::ScalarKind::Float,
            width: width
                .try_into()
                .map_err(|_| Error::InvalidTypeWidth(width))?,
        };
        self.lookup_type.insert(id, LookupType {
            handle: module.types.append(crate::Type {
                name: self.future_decor
                    .remove(&id)
                    .and_then(|dec| dec.name),
                inner,
            }),
            base_id: None,
        });
        Ok(())
    }

    fn parse_type_vector(
        &mut self,
        inst: Instruction,
        module: &mut crate::Module,
    ) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect(4)?;
        let id = self.next()?;
        let type_id = self.next()?;
        let type_lookup = self.lookup_type.lookup(type_id)?;
        let (kind, width) = match module.types[type_lookup.handle].inner {
            crate::TypeInner::Scalar { kind, width } => (kind, width),
            _ => return Err(Error::InvalidInnerType(type_id)),
        };
        let component_count = self.next()?;
        let inner = crate::TypeInner::Vector {
            size: map_vector_size(component_count)?,
            kind,
            width,
        };
        self.lookup_type.insert(id, LookupType {
            handle: module.types.append(crate::Type {
                name: self.future_decor
                    .remove(&id)
                    .and_then(|dec| dec.name),
                inner,
            }),
            base_id: Some(type_id),
        });
        Ok(())
    }

    fn parse_type_matrix(
        &mut self,
        inst: Instruction,
        module: &mut crate::Module,
    ) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect(4)?;
        let id = self.next()?;
        let vector_type_id = self.next()?;
        let num_columns = self.next()?;
        let vector_type_lookup = self.lookup_type.lookup(vector_type_id)?;
        let inner = match module.types[vector_type_lookup.handle].inner {
            crate::TypeInner::Vector { size, kind, width } => crate::TypeInner::Matrix {
                columns: map_vector_size(num_columns)?,
                rows: size,
                kind,
                width,
            },
            _ => return Err(Error::InvalidInnerType(vector_type_id)),
        };
        self.lookup_type.insert(id, LookupType {
            handle: module.types.append(crate::Type {
                name: self.future_decor
                    .remove(&id)
                    .and_then(|dec| dec.name),
                inner,
            }),
            base_id: Some(vector_type_id),
        });
        Ok(())
    }

    fn parse_type_function(&mut self, inst: Instruction) -> Result<(), Error> {
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
        Ok(())
    }

    fn parse_type_pointer(
        &mut self,
        inst: Instruction,
        module: &mut crate::Module,
    ) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect(4)?;
        let id = self.next()?;
        let storage = self.next()?;
        let type_id = self.next()?;
        let inner = crate::TypeInner::Pointer {
            base: self.lookup_type.lookup(type_id)?.handle,
            class: map_storage_class(storage)?,
        };
        self.lookup_type.insert(id, LookupType {
            handle: module.types.append(crate::Type {
                name: self.future_decor
                    .remove(&id)
                    .and_then(|dec| dec.name),
                inner,
            }),
            base_id: Some(type_id),
        });
        Ok(())
    }

    fn parse_type_array(
        &mut self,
        inst: Instruction,
        module: &mut crate::Module,
    ) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect(4)?;
        let id = self.next()?;
        let type_id = self.next()?;
        let length = self.next()?;
        let inner = crate::TypeInner::Array {
            base: self.lookup_type.lookup(type_id)?.handle,
            size: crate::ArraySize::Static(length),
        };
        self.lookup_type.insert(id, LookupType {
            handle: module.types.append(crate::Type {
                name: self.future_decor
                    .remove(&id)
                    .and_then(|dec| dec.name),
                inner,
            }),
            base_id: Some(type_id),
        });
        Ok(())
    }

    fn parse_type_runtime_array(
        &mut self,
        inst: Instruction,
        module: &mut crate::Module,
    ) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect(4)?;
        let id = self.next()?;
        let type_id = self.next()?;
        let inner = crate::TypeInner::Array {
            base: self.lookup_type.lookup(type_id)?.handle,
            size: crate::ArraySize::Dynamic,
        };
        self.lookup_type.insert(id, LookupType {
            handle: module.types.append(crate::Type {
                name: self.future_decor.remove(&id).and_then(|dec| dec.name),
                inner,
            }),
            base_id: Some(type_id),
        });
        Ok(())
    }

    fn parse_type_struct(
        &mut self,
        inst: Instruction,
        module: &mut crate::Module,
    ) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect_at_least(2)?;
        let id = self.next()?;
        let mut members = Vec::with_capacity(inst.wc as usize - 2);
        for i in 0 .. u32::from(inst.wc) - 2 {
            let type_id = self.next()?;
            let ty = self.lookup_type.lookup(type_id)?.handle;
            self.lookup_member_type_id.insert((id, i), type_id);
            let decor = self.future_member_decor
                .remove(&(id, i))
                .unwrap_or_default();
            let binding = decor.get_binding();
            members.push(crate::StructMember {
                name: decor.name,
                binding,
                ty,
            });
        }
        let inner = crate::TypeInner::Struct {
            members
        };
        self.lookup_type.insert(id, LookupType {
            handle: module.types.append(crate::Type {
                name: self.future_decor.remove(&id).and_then(|dec| dec.name),
                inner,
            }),
            base_id: None,
        });
        Ok(())
    }

    fn parse_type_image(
        &mut self,
        inst: Instruction,
        module: &mut crate::Module,
    ) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect_at_least(9)?;

        let id = self.next()?;
        let sample_type_id = self.next()?;
        let dim = self.next()?;
        let mut flags = crate::ImageFlags::empty();
        let _is_depth = self.next()?;
        if self.next()? != 0 {
            flags |= crate::ImageFlags::ARRAYED;
        }
        if self.next()? != 0 {
            flags |= crate::ImageFlags::MULTISAMPLED;
        }
        let is_sampled = self.next()?;
        if is_sampled != 0 {
            flags |= crate::ImageFlags::SAMPLED;
        }
        let _format = self.next()?;
        if inst.wc > 9 {
            inst.expect(10)?;
            let access = self.next()?;
            if access == 0 || access == 2 {
                flags |= crate::ImageFlags::CAN_LOAD;
            }
            if access == 1 || access == 2 {
                flags |= crate::ImageFlags::CAN_STORE;
            }
        };

        let decor = self.future_decor
            .remove(&id)
            .unwrap_or_default();

        let inner = crate::TypeInner::Image {
            base: self.lookup_type.lookup(sample_type_id)?.handle,
            dim: spirv::Dim::from_u32(dim).ok_or(Error::UnsupportedDim(dim))?,
            flags,
        };
        self.lookup_type.insert(id, LookupType {
            handle: module.types.append(crate::Type {
                name: decor.name,
                inner,
            }),
            base_id: Some(sample_type_id),
        });
        Ok(())
    }

    fn parse_type_sampled_image(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect(3)?;
        let id = self.next()?;
        let image_id = self.next()?;
        self.lookup_type.insert(id, LookupType {
            handle: self.lookup_type.lookup(image_id)?.handle,
            base_id: Some(image_id),
        });
        Ok(())
    }

    fn parse_type_sampler(
        &mut self,
        inst: Instruction,
        module: &mut crate::Module,
    ) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect(2)?;
        let id = self.next()?;
        let decor = self.future_decor
            .remove(&id)
            .unwrap_or_default();
        let inner = crate::TypeInner::Sampler;
        self.lookup_type.insert(id, LookupType {
            handle: module.types.append(crate::Type {
                name: decor.name,
                inner,
            }),
            base_id: None,
        });
        Ok(())
    }

    fn parse_constant(
        &mut self,
        inst: Instruction,
        module: &mut crate::Module,
    ) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect_at_least(3)?;
        let type_id = self.next()?;
        let id = self.next()?;
        let type_lookup = self.lookup_type.lookup(type_id)?;
        let ty = type_lookup.handle;
        let inner = match module.types[type_lookup.handle].inner {
            crate::TypeInner::Scalar { kind: crate::ScalarKind::Uint, width } => {
                let low = self.next()?;
                let high = if width > 32 {
                    inst.expect(4)?;
                    self.next()?
                } else {
                    0
                };
                crate::ConstantInner::Uint((u64::from(high) << 32) | u64::from(low))
            }
            crate::TypeInner::Scalar { kind: crate::ScalarKind::Sint, width } => {
                let low = self.next()?;
                let high = if width < 32 {
                    return Err(Error::InvalidTypeWidth(u32::from(width)));
                } else if width > 32 {
                    inst.expect(4)?;
                    self.next()?
                } else {
                    !0
                };
                crate::ConstantInner::Sint(((u64::from(high) << 32) | u64::from(low)) as i64)
            }
            crate::TypeInner::Scalar { kind: crate::ScalarKind::Float, width } => {
                let low = self.next()?;
                let extended = match width {
                    32 => f64::from(f32::from_bits(low)),
                    64 => {
                        inst.expect(4)?;
                        let high = self.next()?;
                        f64::from_bits((u64::from(high) << 32) | u64::from(low))
                    }
                    _ => return Err(Error::InvalidTypeWidth(u32::from(width))),
                };
                crate::ConstantInner::Float(extended)
            }
            _ => return Err(Error::UnsupportedType(type_lookup.handle))
        };
        self.lookup_constant.insert(id, LookupConstant {
            handle: module.constants.append(crate::Constant {
                name: self.future_decor
                    .remove(&id)
                    .and_then(|dec| dec.name),
                specialization: None, //TODO
                inner,
                ty,
            }),
            type_id,
        });
        Ok(())
    }

    fn parse_composite_constant(
        &mut self,
        inst: Instruction,
        module: &mut crate::Module,
    ) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect_at_least(3)?;
        let type_id = self.next()?;
        let type_lookup = self.lookup_type.lookup(type_id)?;
        let ty = type_lookup.handle;

        let id = self.next()?;

        let constituents_count = inst.wc - 3;
        let mut constituents = Vec::with_capacity(constituents_count as usize);
        for _ in 0..constituents_count {
            let constituent_id = self.next()?;
            let constant = self.lookup_constant.lookup(constituent_id)?;
            constituents.push(constant.handle);
        }

        self.lookup_constant.insert(id, LookupConstant {
            handle: module.constants.append(crate::Constant {
                name: self.future_decor
                    .remove(&id)
                    .and_then(|dec|dec.name),
                specialization: None,
                inner: crate::ConstantInner::Composite(constituents),
                ty
            }),
            type_id
        });

        Ok(())
    }

    fn parse_variable(
        &mut self,
        inst: Instruction,
        module: &mut crate::Module,
    ) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect_at_least(4)?;
        let type_id = self.next()?;
        let id = self.next()?;
        let storage = self.next()?;
        if inst.wc != 4 {
            inst.expect(5)?;
            let _init = self.next()?; //TODO
        }
        let lookup_type = self.lookup_type.lookup(type_id)?;
        let dec = self.future_decor
            .remove(&id)
            .ok_or(Error::InvalidBinding(id))?;
        let binding = match module.types[lookup_type.handle].inner {
            crate::TypeInner::Pointer { base, class: spirv::StorageClass::Input } |
            crate::TypeInner::Pointer { base, class: spirv::StorageClass::Output } => {
                match module.types[base].inner {
                    crate::TypeInner::Struct { ref members } => {
                        // we don't expect binding decoration on I/O structs,
                        // but we do expect them on all of the members
                        for member in members {
                            if member.binding.is_none() {
                                log::warn!("Struct {:?} member {:?} doesn't have a binding", base, member);
                                return Err(Error::InvalidBinding(id));
                            }
                        }
                        None
                    }
                    _ => {
                        Some(dec
                            .get_binding()
                            .ok_or(Error::InvalidBinding(id))?
                        )
                   }
                }
            }
            _ => {
                Some(dec
                    .get_binding()
                    .ok_or(Error::InvalidBinding(id))?
                )
            }
        };
        let var = crate::GlobalVariable {
            name: dec.name,
            class: map_storage_class(storage)?,
            binding,
            ty: lookup_type.handle,
        };
        self.lookup_variable.insert(id, LookupVariable {
            handle: module.global_variables.append(var),
            type_id,
        });
        Ok(())
    }

    fn parse_function(
        &mut self,
        inst: Instruction,
        module: &mut crate::Module,
    ) -> Result<(), Error> {
        self.switch(ModuleState::Function, inst.op)?;
        inst.expect(5)?;
        let result_type = self.next()?;
        let fun_id = self.next()?;
        let fun_control = self.next()?;
        let fun_type = self.next()?;
        let mut fun = {
            let ft = self.lookup_function_type.lookup(fun_type)?;
            if ft.return_type_id != result_type {
                return Err(Error::WrongFunctionResultType(result_type))
            }
            crate::Function {
                name: self.future_decor
                    .remove(&fun_id)
                    .and_then(|dec| dec.name),
                control: spirv::FunctionControl::from_bits(fun_control)
                    .ok_or(Error::UnsupportedFunctionControl(fun_control))?,
                parameter_types: Vec::with_capacity(ft.parameter_type_ids.len()),
                return_type: if self.lookup_void_type.contains(&result_type) {
                    None
                } else {
                    Some(self.lookup_type.lookup(result_type)?.handle)
                },
                global_usage: Vec::new(),
                local_variables: Arena::new(),
                expressions: self.make_expression_storage(),
                body: Vec::new(),
            }
        };
        // read parameters
        for i in 0..fun.parameter_types.capacity() {
            match self.next_inst()? {
                Instruction { op: spirv::Op::FunctionParameter, wc: 3 } => {
                    let type_id = self.next()?;
                    let _id = self.next()?;
                    //Note: we redo the lookup in order to work around `self` borrowing
                    if type_id != self.lookup_function_type
                        .lookup(fun_type)?
                        .parameter_type_ids[i]
                    {
                        return Err(Error::WrongFunctionParameterType(type_id))
                    }
                    let ty = self.lookup_type.lookup(type_id)?.handle;
                    fun.parameter_types.push(ty);
                }
                Instruction { op, .. } => return Err(Error::InvalidParameter(op)),
            }
        }
        // read body
        loop {
            let fun_inst = self.next_inst()?;
            log::debug!("\t\t{:?}", fun_inst.op);
            match fun_inst.op {
                spirv::Op::Label => {
                    fun_inst.expect(2)?;
                    let _id = self.next()?;
                    self.next_block(&mut fun, &module.types, &module.constants)?;
                }
                spirv::Op::FunctionEnd => {
                    fun_inst.expect(1)?;
                    break
                }
                _ => return Err(Error::UnsupportedInstruction(self.state, fun_inst.op))
            }
        }
        // done
        fun.global_usage = crate::GlobalUse::scan(&fun.expressions, &fun.body, &module.global_variables);
        let handle = module.functions.append(fun);
        self.lookup_function.insert(fun_id, handle);
        self.lookup_expression.clear();
        self.lookup_sampled_image.clear();
        Ok(())
    }
}

pub fn parse_u8_slice(data: &[u8]) -> Result<crate::Module, Error> {
    if data.len() % 4 != 0 {
        return Err(Error::IncompleteData);
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
