/*! SPIR-V frontend

## ID lookups

Our IR links to everything with `Handle`, while SPIR-V uses IDs.
In order to keep track of the associations, the parser has many lookup tables.
There map `spv::Word` into a specific IR handle, plus potentially a bit of
extra info, such as the related SPIR-V type ID.
TODO: would be nice to find ways that avoid looking up as much

!*/
#![allow(dead_code)]

mod convert;
mod error;
mod flow;
mod function;
#[cfg(all(test, feature = "serialize"))]
mod rosetta;

use convert::*;
use error::Error;
use flow::*;
use function::*;

use crate::{
    arena::{Arena, Handle},
    FastHashMap, FastHashSet,
};

use num_traits::cast::FromPrimitive;
use std::{convert::TryInto, num::NonZeroU32, path::PathBuf};

pub const SUPPORTED_CAPABILITIES: &[spirv::Capability] = &[
    spirv::Capability::Shader,
    spirv::Capability::CullDistance,
    spirv::Capability::StorageImageExtendedFormats,
];
pub const SUPPORTED_EXTENSIONS: &[&str] = &[];
pub const SUPPORTED_EXT_SETS: &[&str] = &["GLSL.std.450"];

#[derive(Copy, Clone)]
pub struct Instruction {
    op: spirv::Op,
    wc: u16,
}

impl Instruction {
    fn expect(self, count: u16) -> Result<(), Error> {
        if self.wc == count {
            Ok(())
        } else {
            Err(Error::InvalidOperandCount(self.op, self.wc))
        }
    }

    fn expect_at_least(self, count: u16) -> Result<(), Error> {
        if self.wc >= count {
            Ok(())
        } else {
            Err(Error::InvalidOperandCount(self.op, self.wc))
        }
    }
}
/// OpPhi instruction.
#[derive(Clone, Default, Debug)]
struct PhiInstruction {
    /// SPIR-V's ID.
    id: u32,

    /// Tuples of (variable, parent).
    variables: Vec<(u32, u32)>,
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

//TODO: this method may need to be gone, depending on whether
// WGSL allows treating images and samplers as expressions and pass them around.
fn reach_global_type(
    mut expr_handle: Handle<crate::Expression>,
    expressions: &Arena<crate::Expression>,
    globals: &Arena<crate::GlobalVariable>,
) -> Option<Handle<crate::Type>> {
    loop {
        expr_handle = match expressions[expr_handle] {
            crate::Expression::Load { pointer } => pointer,
            crate::Expression::GlobalVariable(var) => return Some(globals[var].ty),
            _ => return None,
        };
    }
}

fn check_sample_coordinates(
    ty: &crate::Type,
    expect_kind: crate::ScalarKind,
    dim: crate::ImageDimension,
    is_array: bool,
) -> bool {
    let base_count = match dim {
        crate::ImageDimension::D1 => 1,
        crate::ImageDimension::D2 => 2,
        crate::ImageDimension::D3 | crate::ImageDimension::Cube => 3,
    };
    let extra_count = if is_array { 1 } else { 0 };
    let count = base_count + extra_count;
    match ty.inner {
        crate::TypeInner::Scalar { kind, width: _ } => count == 1 && kind == expect_kind,
        crate::TypeInner::Vector {
            size,
            kind,
            width: _,
        } => size as u8 == count && kind == expect_kind,
        _ => false,
    }
}

type MemberIndex = u32;

#[derive(Debug, Default)]
struct Block {
    buffer: bool,
}

bitflags::bitflags! {
    #[derive(Default)]
    struct DecorationFlags: u32 {
        const NON_READABLE = 0x1;
        const NON_WRITABLE = 0x2;
    }
}

#[derive(Debug, Default)]
struct Decoration {
    name: Option<String>,
    built_in: Option<crate::BuiltIn>,
    location: Option<spirv::Word>,
    desc_set: Option<spirv::Word>,
    desc_index: Option<spirv::Word>,
    block: Option<Block>,
    offset: Option<spirv::Word>,
    array_stride: Option<NonZeroU32>,
    interpolation: Option<crate::Interpolation>,
    flags: DecorationFlags,
}

impl Decoration {
    fn debug_name(&self) -> &str {
        match self.name {
            Some(ref name) => name.as_str(),
            None => "?",
        }
    }

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
                desc_set: Some(group),
                desc_index: Some(binding),
                ..
            } => Some(crate::Binding::Resource { group, binding }),
            _ => None,
        }
    }

    fn get_origin(&self) -> Result<crate::MemberOrigin, Error> {
        match *self {
            Decoration {
                location: Some(_), ..
            }
            | Decoration {
                desc_set: Some(_), ..
            }
            | Decoration {
                desc_index: Some(_),
                ..
            } => Err(Error::MissingDecoration(spirv::Decoration::Offset)),
            Decoration {
                built_in: Some(built_in),
                offset: None,
                ..
            } => Ok(crate::MemberOrigin::BuiltIn(built_in)),
            Decoration {
                built_in: None,
                offset: Some(offset),
                ..
            } => Ok(crate::MemberOrigin::Offset(offset)),
            _ => Ok(crate::MemberOrigin::Empty),
        }
    }
}

bitflags::bitflags! {
    /// Flags describing sampling method.
    pub struct SamplingFlags: u32 {
        /// Regular sampling.
        const REGULAR = 0x1;
        /// Comparison sampling.
        const COMPARISON = 0x2;
    }
}

#[derive(Debug)]
struct LookupFunctionType {
    parameter_type_ids: Vec<spirv::Word>,
    return_type_id: spirv::Word,
}

#[derive(Debug)]
struct EntryPoint {
    stage: crate::ShaderStage,
    name: String,
    early_depth_test: Option<crate::EarlyDepthTest>,
    workgroup_size: [u32; 3],
    function_id: spirv::Word,
    variable_ids: Vec<spirv::Word>,
}

#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
enum DeferredSource {
    EntryPoint(crate::ShaderStage, String),
    Function(Handle<crate::Function>),
}
struct DeferredFunctionCall {
    source: DeferredSource,
    expr_handle: Handle<crate::Expression>,
    dst_id: spirv::Word,
}

#[derive(Clone, Debug)]
pub struct Assignment {
    to: Handle<crate::Expression>,
    value: Handle<crate::Expression>,
}

#[derive(Clone, Debug, Default)]
pub struct Options {
    pub flow_graph_dump_prefix: Option<PathBuf>,
}

pub struct Parser<I> {
    data: I,
    state: ModuleState,
    temp_bytes: Vec<u8>,
    ext_glsl_id: Option<spirv::Word>,
    future_decor: FastHashMap<spirv::Word, Decoration>,
    future_member_decor: FastHashMap<(spirv::Word, MemberIndex), Decoration>,
    lookup_member_type_id: FastHashMap<(Handle<crate::Type>, MemberIndex), spirv::Word>,
    handle_sampling: FastHashMap<Handle<crate::Type>, SamplingFlags>,
    lookup_type: FastHashMap<spirv::Word, LookupType>,
    lookup_void_type: FastHashSet<spirv::Word>,
    lookup_storage_buffer_types: FastHashSet<Handle<crate::Type>>,
    // Lookup for samplers and sampled images, storing flags on how they are used.
    lookup_constant: FastHashMap<spirv::Word, LookupConstant>,
    lookup_variable: FastHashMap<spirv::Word, LookupVariable>,
    lookup_expression: FastHashMap<spirv::Word, LookupExpression>,
    lookup_sampled_image: FastHashMap<spirv::Word, LookupSampledImage>,
    lookup_function_type: FastHashMap<spirv::Word, LookupFunctionType>,
    lookup_function: FastHashMap<spirv::Word, Handle<crate::Function>>,
    lookup_entry_point: FastHashMap<spirv::Word, EntryPoint>,
    deferred_function_calls: Vec<DeferredFunctionCall>,
    options: Options,
}

impl<I: Iterator<Item = u32>> Parser<I> {
    pub fn new(data: I, options: &Options) -> Self {
        Parser {
            data,
            state: ModuleState::Empty,
            temp_bytes: Vec::new(),
            ext_glsl_id: None,
            future_decor: FastHashMap::default(),
            future_member_decor: FastHashMap::default(),
            handle_sampling: FastHashMap::default(),
            lookup_member_type_id: FastHashMap::default(),
            lookup_type: FastHashMap::default(),
            lookup_void_type: FastHashSet::default(),
            lookup_storage_buffer_types: FastHashSet::default(),
            lookup_constant: FastHashMap::default(),
            lookup_variable: FastHashMap::default(),
            lookup_expression: FastHashMap::default(),
            lookup_sampled_image: FastHashMap::default(),
            lookup_function_type: FastHashMap::default(),
            lookup_function: FastHashMap::default(),
            lookup_entry_point: FastHashMap::default(),
            deferred_function_calls: Vec::new(),
            options: options.clone(),
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

    fn next_string(&mut self, mut count: u16) -> Result<(String, u16), Error> {
        self.temp_bytes.clear();
        loop {
            if count == 0 {
                return Err(Error::BadString);
            }
            count -= 1;
            let chars = self.next()?.to_le_bytes();
            let pos = chars.iter().position(|&c| c == 0).unwrap_or(4);
            self.temp_bytes.extend_from_slice(&chars[..pos]);
            if pos < 4 {
                break;
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
        log::trace!("\t\t{}: {:?}", dec.debug_name(), dec_typed);
        match dec_typed {
            spirv::Decoration::BuiltIn => {
                inst.expect(base_words + 2)?;
                let raw = self.next()?;
                match map_builtin(raw) {
                    Ok(built_in) => dec.built_in = Some(built_in),
                    Err(_e) => log::warn!("Unsupported builtin {}", raw),
                };
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
            spirv::Decoration::Block => {
                dec.block = Some(Block { buffer: false });
            }
            spirv::Decoration::BufferBlock => {
                dec.block = Some(Block { buffer: true });
            }
            spirv::Decoration::Offset => {
                inst.expect(base_words + 2)?;
                dec.offset = Some(self.next()?);
            }
            spirv::Decoration::ArrayStride => {
                inst.expect(base_words + 2)?;
                dec.array_stride = NonZeroU32::new(self.next()?);
            }
            spirv::Decoration::NoPerspective => {
                dec.interpolation = Some(crate::Interpolation::Linear);
            }
            spirv::Decoration::Flat => {
                dec.interpolation = Some(crate::Interpolation::Flat);
            }
            spirv::Decoration::Patch => {
                dec.interpolation = Some(crate::Interpolation::Patch);
            }
            spirv::Decoration::Centroid => {
                dec.interpolation = Some(crate::Interpolation::Centroid);
            }
            spirv::Decoration::Sample => {
                dec.interpolation = Some(crate::Interpolation::Sample);
            }
            spirv::Decoration::NonReadable => {
                dec.flags |= DecorationFlags::NON_READABLE;
            }
            spirv::Decoration::NonWritable => {
                dec.flags |= DecorationFlags::NON_WRITABLE;
            }
            other => {
                log::warn!("Unknown decoration {:?}", other);
                for _ in base_words + 1..inst.wc {
                    let _var = self.next()?;
                }
            }
        }
        Ok(())
    }

    fn parse_expr_unary_op(
        &mut self,
        expressions: &mut Arena<crate::Expression>,
        op: crate::UnaryOperator,
    ) -> Result<(), Error> {
        let result_type_id = self.next()?;
        let result_id = self.next()?;
        let p_id = self.next()?;

        let p_lexp = self.lookup_expression.lookup(p_id)?;

        let expr = crate::Expression::Unary {
            op,
            expr: p_lexp.handle,
        };
        self.lookup_expression.insert(
            result_id,
            LookupExpression {
                handle: expressions.append(expr),
                type_id: result_type_id,
            },
        );
        Ok(())
    }

    fn parse_expr_binary_op(
        &mut self,
        expressions: &mut Arena<crate::Expression>,
        op: crate::BinaryOperator,
    ) -> Result<(), Error> {
        let result_type_id = self.next()?;
        let result_id = self.next()?;
        let p1_id = self.next()?;
        let p2_id = self.next()?;

        let p1_lexp = self.lookup_expression.lookup(p1_id)?;
        let p2_lexp = self.lookup_expression.lookup(p2_id)?;

        let expr = crate::Expression::Binary {
            op,
            left: p1_lexp.handle,
            right: p2_lexp.handle,
        };
        self.lookup_expression.insert(
            result_id,
            LookupExpression {
                handle: expressions.append(expr),
                type_id: result_type_id,
            },
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn next_block(
        &mut self,
        block_id: spirv::Word,
        expressions: &mut Arena<crate::Expression>,
        local_arena: &mut Arena<crate::LocalVariable>,
        type_arena: &Arena<crate::Type>,
        const_arena: &Arena<crate::Constant>,
        global_arena: &Arena<crate::GlobalVariable>,
        local_function_calls: &mut FastHashMap<Handle<crate::Expression>, spirv::Word>,
    ) -> Result<ControlFlowNode, Error> {
        let mut assignments = Vec::new();
        let mut phis = Vec::new();
        let mut merge = None;
        let terminator = loop {
            use spirv::Op;
            let inst = self.next_inst()?;
            log::debug!("\t\t{:?} [{}]", inst.op, inst.wc);

            match inst.op {
                Op::Variable => {
                    inst.expect_at_least(4)?;
                    let result_type_id = self.next()?;
                    let result_id = self.next()?;
                    let storage = self.next()?;
                    match spirv::StorageClass::from_u32(storage) {
                        Some(spirv::StorageClass::Function) => (),
                        Some(class) => return Err(Error::InvalidVariableClass(class)),
                        None => return Err(Error::UnsupportedStorageClass(storage)),
                    }
                    let init = if inst.wc > 4 {
                        inst.expect(5)?;
                        let init_id = self.next()?;
                        let lconst = self.lookup_constant.lookup(init_id)?;
                        Some(lconst.handle)
                    } else {
                        None
                    };
                    let name = self
                        .future_decor
                        .remove(&result_id)
                        .and_then(|decor| decor.name);
                    if let Some(ref name) = name {
                        log::debug!("\t\t\tid={} name={}", result_id, name);
                    }
                    let var_handle = local_arena.append(crate::LocalVariable {
                        name,
                        ty: self.lookup_type.lookup(result_type_id)?.handle,
                        init,
                    });
                    self.lookup_expression.insert(
                        result_id,
                        LookupExpression {
                            handle: expressions
                                .append(crate::Expression::LocalVariable(var_handle)),
                            type_id: result_type_id,
                        },
                    );
                }
                Op::Phi => {
                    inst.expect_at_least(3)?;

                    let result_type_id = self.next()?;
                    let result_id = self.next()?;

                    let name = format!("phi_{}", result_id);
                    let var_handle = local_arena.append(crate::LocalVariable {
                        name: Some(name),
                        ty: self.lookup_type.lookup(result_type_id)?.handle,
                        init: None,
                    });
                    self.lookup_expression.insert(
                        result_id,
                        LookupExpression {
                            handle: expressions
                                .append(crate::Expression::LocalVariable(var_handle)),
                            type_id: result_type_id,
                        },
                    );

                    let mut phi = PhiInstruction::default();
                    phi.id = result_id;
                    for _ in 0..(inst.wc - 3) / 2 {
                        phi.variables.push((self.next()?, self.next()?));
                    }

                    phis.push(phi);
                }
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
                        AccessExpression {
                            base_handle: expr.handle,
                            type_id: expr.type_id,
                        }
                    };
                    for _ in 4..inst.wc {
                        let access_id = self.next()?;
                        log::trace!("\t\t\tlooking up index expr {:?}", access_id);
                        let index_expr = self.lookup_expression.lookup(access_id)?.clone();
                        let index_type_handle = self.lookup_type.lookup(index_expr.type_id)?.handle;
                        match type_arena[index_type_handle].inner {
                            crate::TypeInner::Scalar {
                                kind: crate::ScalarKind::Uint,
                                ..
                            }
                            | crate::TypeInner::Scalar {
                                kind: crate::ScalarKind::Sint,
                                ..
                            } => (),
                            _ => return Err(Error::UnsupportedType(index_type_handle)),
                        }
                        log::trace!("\t\t\tlooking up type {:?}", acex.type_id);
                        let type_lookup = self.lookup_type.lookup(acex.type_id)?;
                        acex = match type_arena[type_lookup.handle].inner {
                            crate::TypeInner::Struct { .. } => {
                                let index = match expressions[index_expr.handle] {
                                    crate::Expression::Constant(const_handle) => {
                                        match const_arena[const_handle].inner {
                                            crate::ConstantInner::Uint(v) => v as u32,
                                            crate::ConstantInner::Sint(v) => v as u32,
                                            _ => {
                                                return Err(Error::InvalidAccess(index_expr.handle))
                                            }
                                        }
                                    }
                                    _ => return Err(Error::InvalidAccess(index_expr.handle)),
                                };
                                AccessExpression {
                                    base_handle: expressions.append(
                                        crate::Expression::AccessIndex {
                                            base: acex.base_handle,
                                            index,
                                        },
                                    ),
                                    type_id: *self
                                        .lookup_member_type_id
                                        .get(&(type_lookup.handle, index))
                                        .ok_or(Error::InvalidAccessType(acex.type_id))?,
                                }
                            }
                            crate::TypeInner::Array { .. }
                            | crate::TypeInner::Vector { .. }
                            | crate::TypeInner::Matrix { .. } => AccessExpression {
                                base_handle: expressions.append(crate::Expression::Access {
                                    base: acex.base_handle,
                                    index: index_expr.handle,
                                }),
                                type_id: type_lookup
                                    .base_id
                                    .ok_or(Error::InvalidAccessType(acex.type_id))?,
                            },
                            _ => return Err(Error::UnsupportedType(type_lookup.handle)),
                        };
                    }

                    let lookup_expression = LookupExpression {
                        handle: acex.base_handle,
                        type_id: result_type_id,
                    };
                    self.lookup_expression.insert(result_id, lookup_expression);
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
                    for _ in 4..inst.wc {
                        let index = self.next()?;
                        log::trace!("\t\t\tlooking up type {:?}", lexp.type_id);
                        let type_lookup = self.lookup_type.lookup(lexp.type_id)?;
                        let type_id = match type_arena[type_lookup.handle].inner {
                            crate::TypeInner::Struct { .. } => *self
                                .lookup_member_type_id
                                .get(&(type_lookup.handle, index))
                                .ok_or(Error::InvalidAccessType(lexp.type_id))?,
                            crate::TypeInner::Array { .. }
                            | crate::TypeInner::Vector { .. }
                            | crate::TypeInner::Matrix { .. } => type_lookup
                                .base_id
                                .ok_or(Error::InvalidAccessType(lexp.type_id))?,
                            _ => return Err(Error::UnsupportedType(type_lookup.handle)),
                        };
                        lexp = LookupExpression {
                            handle: expressions.append(crate::Expression::AccessIndex {
                                base: lexp.handle,
                                index,
                            }),
                            type_id,
                        };
                    }

                    self.lookup_expression.insert(
                        result_id,
                        LookupExpression {
                            handle: lexp.handle,
                            type_id: result_type_id,
                        },
                    );
                }
                Op::CompositeConstruct => {
                    inst.expect_at_least(3)?;
                    let result_type_id = self.next()?;
                    let id = self.next()?;
                    let mut components = Vec::with_capacity(inst.wc as usize - 2);
                    for _ in 3..inst.wc {
                        let comp_id = self.next()?;
                        log::trace!("\t\t\tlooking up expr {:?}", comp_id);
                        let lexp = self.lookup_expression.lookup(comp_id)?;
                        components.push(lexp.handle);
                    }
                    let expr = crate::Expression::Compose {
                        ty: self.lookup_type.lookup(result_type_id)?.handle,
                        components,
                    };
                    self.lookup_expression.insert(
                        id,
                        LookupExpression {
                            handle: expressions.append(expr),
                            type_id: result_type_id,
                        },
                    );
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
                    let base_expr = self.lookup_expression.lookup(pointer_id)?.clone();
                    self.lookup_expression.insert(
                        result_id,
                        LookupExpression {
                            handle: base_expr.handle, // pass-through pointers
                            type_id: result_type_id,
                        },
                    );
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
                    let value_expr = self.lookup_expression.lookup(value_id)?;
                    assignments.push(Assignment {
                        to: base_expr.handle,
                        value: value_expr.handle,
                    });
                }
                // Arithmetic Instructions +, -, *, /, %
                Op::SNegate | Op::FNegate => {
                    inst.expect(4)?;
                    self.parse_expr_unary_op(expressions, crate::UnaryOperator::Negate)?;
                }
                Op::IAdd | Op::FAdd => {
                    inst.expect(5)?;
                    self.parse_expr_binary_op(expressions, crate::BinaryOperator::Add)?;
                }
                Op::ISub | Op::FSub => {
                    inst.expect(5)?;
                    self.parse_expr_binary_op(expressions, crate::BinaryOperator::Subtract)?;
                }
                Op::IMul | Op::FMul => {
                    inst.expect(5)?;
                    self.parse_expr_binary_op(expressions, crate::BinaryOperator::Multiply)?;
                }
                Op::SDiv | Op::UDiv | Op::FDiv => {
                    inst.expect(5)?;
                    self.parse_expr_binary_op(expressions, crate::BinaryOperator::Divide)?;
                }
                Op::UMod | Op::FMod | Op::SRem | Op::FRem => {
                    inst.expect(5)?;
                    self.parse_expr_binary_op(expressions, crate::BinaryOperator::Modulo)?;
                }
                Op::VectorTimesScalar
                | Op::VectorTimesMatrix
                | Op::MatrixTimesScalar
                | Op::MatrixTimesVector
                | Op::MatrixTimesMatrix => {
                    inst.expect(5)?;
                    self.parse_expr_binary_op(expressions, crate::BinaryOperator::Multiply)?;
                }
                Op::Transpose => {
                    inst.expect(4)?;
                    let result_type_id = self.next()?;
                    let result_id = self.next()?;
                    let matrix_id = self.next()?;
                    let matrix_lexp = self.lookup_expression.lookup(matrix_id)?;
                    let expr = crate::Expression::Transpose(matrix_lexp.handle);
                    self.lookup_expression.insert(
                        result_id,
                        LookupExpression {
                            handle: expressions.append(expr),
                            type_id: result_type_id,
                        },
                    );
                }
                Op::Dot => {
                    inst.expect(5)?;
                    let result_type_id = self.next()?;
                    let result_id = self.next()?;
                    let left_id = self.next()?;
                    let right_id = self.next()?;
                    let left_lexp = self.lookup_expression.lookup(left_id)?;
                    let right_lexp = self.lookup_expression.lookup(right_id)?;
                    let expr = crate::Expression::DotProduct(left_lexp.handle, right_lexp.handle);
                    self.lookup_expression.insert(
                        result_id,
                        LookupExpression {
                            handle: expressions.append(expr),
                            type_id: result_type_id,
                        },
                    );
                }
                // Bitwise instructions
                Op::Not => {
                    inst.expect(4)?;
                    self.parse_expr_unary_op(expressions, crate::UnaryOperator::Not)?;
                }
                Op::BitwiseOr => {
                    inst.expect(5)?;
                    self.parse_expr_binary_op(expressions, crate::BinaryOperator::InclusiveOr)?;
                }
                Op::BitwiseXor => {
                    inst.expect(5)?;
                    self.parse_expr_binary_op(expressions, crate::BinaryOperator::ExclusiveOr)?;
                }
                Op::BitwiseAnd => {
                    inst.expect(5)?;
                    self.parse_expr_binary_op(expressions, crate::BinaryOperator::And)?;
                }
                Op::ShiftRightLogical => {
                    inst.expect(5)?;
                    //TODO: convert input and result to usigned
                    self.parse_expr_binary_op(expressions, crate::BinaryOperator::ShiftRight)?;
                }
                Op::ShiftRightArithmetic => {
                    inst.expect(5)?;
                    //TODO: convert input and result to signed
                    self.parse_expr_binary_op(expressions, crate::BinaryOperator::ShiftRight)?;
                }
                Op::ShiftLeftLogical => {
                    inst.expect(5)?;
                    self.parse_expr_binary_op(expressions, crate::BinaryOperator::ShiftLeft)?;
                }
                // Sampling
                Op::SampledImage => {
                    inst.expect(5)?;
                    let _result_type_id = self.next()?;
                    let result_id = self.next()?;
                    let image_id = self.next()?;
                    let sampler_id = self.next()?;
                    let image_lexp = self.lookup_expression.lookup(image_id)?;
                    let sampler_lexp = self.lookup_expression.lookup(sampler_id)?;
                    //TODO: compare the result type
                    self.lookup_sampled_image.insert(
                        result_id,
                        LookupSampledImage {
                            image: image_lexp.handle,
                            sampler: sampler_lexp.handle,
                        },
                    );
                }
                Op::ImageSampleImplicitLod | Op::ImageSampleExplicitLod => {
                    inst.expect_at_least(5)?;
                    let result_type_id = self.next()?;
                    let result_id = self.next()?;
                    let sampled_image_id = self.next()?;
                    let coordinate_id = self.next()?;
                    let si_lexp = self.lookup_sampled_image.lookup(sampled_image_id)?.clone();
                    let coord_lexp = self.lookup_expression.lookup(coordinate_id)?.clone();
                    let coord_type_handle = self.lookup_type.lookup(coord_lexp.type_id)?.handle;

                    let sampler_type_handle =
                        reach_global_type(si_lexp.sampler, &expressions, global_arena)
                            .ok_or(Error::InvalidSamplerExpression(si_lexp.sampler))?;
                    let image_type_handle =
                        reach_global_type(si_lexp.image, &expressions, global_arena)
                            .ok_or(Error::InvalidImageExpression(si_lexp.image))?;
                    log::debug!(
                        "\t\t\tImage {:?} with sampler {:?}",
                        image_type_handle,
                        sampler_type_handle
                    );
                    *self.handle_sampling.get_mut(&sampler_type_handle).unwrap() |=
                        SamplingFlags::REGULAR;
                    *self.handle_sampling.get_mut(&image_type_handle).unwrap() |=
                        SamplingFlags::REGULAR;
                    match type_arena[sampler_type_handle].inner {
                        crate::TypeInner::Sampler { comparison: false } => (),
                        _ => return Err(Error::InvalidSampleSampler(sampler_type_handle)),
                    };
                    match type_arena[image_type_handle].inner {
                        //TODO: compare the result type
                        crate::TypeInner::Image {
                            dim,
                            arrayed,
                            class:
                                crate::ImageClass::Sampled {
                                    kind: _,
                                    multi: false,
                                },
                        } => {
                            if !check_sample_coordinates(
                                &type_arena[coord_type_handle],
                                crate::ScalarKind::Float,
                                dim,
                                arrayed,
                            ) {
                                return Err(Error::InvalidSampleCoordinates(coord_type_handle));
                            }
                        }
                        _ => return Err(Error::InvalidSampleImage(image_type_handle)),
                    };

                    let mut level = crate::SampleLevel::Auto;
                    let mut base_wc = 5;
                    if base_wc < inst.wc {
                        let image_ops = self.next()?;
                        base_wc += 1;
                        let mask = spirv::ImageOperands::from_bits_truncate(image_ops);
                        if mask.contains(spirv::ImageOperands::BIAS) {
                            let bias_expr = self.next()?;
                            let bias_handle = self.lookup_expression.lookup(bias_expr)?.handle;
                            level = crate::SampleLevel::Bias(bias_handle);
                            base_wc += 1;
                        }
                        if mask.contains(spirv::ImageOperands::LOD) {
                            let lod_expr = self.next()?;
                            let lod_handle = self.lookup_expression.lookup(lod_expr)?.handle;
                            level = crate::SampleLevel::Exact(lod_handle);
                            base_wc += 1;
                        }
                        for _ in base_wc..inst.wc {
                            self.next()?;
                        }
                    }

                    let expr = crate::Expression::ImageSample {
                        image: si_lexp.image,
                        sampler: si_lexp.sampler,
                        coordinate: coord_lexp.handle,
                        level,
                        depth_ref: None,
                    };
                    self.lookup_expression.insert(
                        result_id,
                        LookupExpression {
                            handle: expressions.append(expr),
                            type_id: result_type_id,
                        },
                    );
                }
                Op::ImageSampleDrefImplicitLod => {
                    inst.expect_at_least(6)?;
                    let result_type_id = self.next()?;
                    let result_id = self.next()?;
                    let sampled_image_id = self.next()?;
                    let coordinate_id = self.next()?;
                    let dref_id = self.next()?;

                    let si_lexp = self.lookup_sampled_image.lookup(sampled_image_id)?;
                    let coord_lexp = self.lookup_expression.lookup(coordinate_id)?;
                    let coord_type_handle = self.lookup_type.lookup(coord_lexp.type_id)?.handle;
                    let sampler_type_handle =
                        reach_global_type(si_lexp.sampler, &expressions, global_arena)
                            .ok_or(Error::InvalidSamplerExpression(si_lexp.sampler))?;
                    let image_type_handle =
                        reach_global_type(si_lexp.image, &expressions, global_arena)
                            .ok_or(Error::InvalidImageExpression(si_lexp.image))?;
                    *self.handle_sampling.get_mut(&sampler_type_handle).unwrap() |=
                        SamplingFlags::COMPARISON;
                    *self.handle_sampling.get_mut(&image_type_handle).unwrap() |=
                        SamplingFlags::COMPARISON;
                    match type_arena[sampler_type_handle].inner {
                        crate::TypeInner::Sampler { comparison: true } => (),
                        _ => return Err(Error::InvalidSampleSampler(sampler_type_handle)),
                    };
                    match type_arena[image_type_handle].inner {
                        //TODO: compare the result type
                        crate::TypeInner::Image {
                            dim,
                            arrayed,
                            class: crate::ImageClass::Depth,
                        } => {
                            if !check_sample_coordinates(
                                &type_arena[coord_type_handle],
                                crate::ScalarKind::Float,
                                dim,
                                arrayed,
                            ) {
                                return Err(Error::InvalidSampleCoordinates(coord_type_handle));
                            }
                        }
                        _ => return Err(Error::InvalidSampleImage(image_type_handle)),
                    };

                    let dref_lexp = self.lookup_expression.lookup(dref_id)?;
                    let dref_type_handle = self.lookup_type.lookup(dref_lexp.type_id)?.handle;
                    match type_arena[dref_type_handle].inner {
                        crate::TypeInner::Scalar {
                            kind: crate::ScalarKind::Float,
                            width: _,
                        } => (),
                        _ => return Err(Error::InvalidDepthReference(dref_type_handle)),
                    }

                    let expr = crate::Expression::ImageSample {
                        image: si_lexp.image,
                        sampler: si_lexp.sampler,
                        coordinate: coord_lexp.handle,
                        level: crate::SampleLevel::Auto,
                        depth_ref: Some(dref_lexp.handle),
                    };
                    self.lookup_expression.insert(
                        result_id,
                        LookupExpression {
                            handle: expressions.append(expr),
                            type_id: result_type_id,
                        },
                    );
                }
                Op::Select => {
                    inst.expect(6)?;
                    let result_type_id = self.next()?;
                    let result_id = self.next()?;
                    let condition = self.next()?;
                    let o1_id = self.next()?;
                    let o2_id = self.next()?;

                    let cond_lexp = self.lookup_expression.lookup(condition)?;
                    let o1_lexp = self.lookup_expression.lookup(o1_id)?;
                    let o2_lexp = self.lookup_expression.lookup(o2_id)?;

                    let expr = crate::Expression::Select {
                        condition: cond_lexp.handle,
                        accept: o1_lexp.handle,
                        reject: o2_lexp.handle,
                    };
                    self.lookup_expression.insert(
                        result_id,
                        LookupExpression {
                            handle: expressions.append(expr),
                            type_id: result_type_id,
                        },
                    );
                }
                Op::VectorShuffle => {
                    inst.expect_at_least(5)?;
                    let result_type_id = self.next()?;
                    let result_id = self.next()?;
                    let v1_id = self.next()?;
                    let v2_id = self.next()?;

                    let v1_lexp = self.lookup_expression.lookup(v1_id)?;
                    let v1_lty = self.lookup_type.lookup(v1_lexp.type_id)?;
                    let n1 = match type_arena[v1_lty.handle].inner {
                        crate::TypeInner::Vector { size, .. } => size as u8,
                        _ => return Err(Error::InvalidInnerType(v1_lexp.type_id)),
                    };
                    let v1_handle = v1_lexp.handle;
                    let v2_lexp = self.lookup_expression.lookup(v2_id)?;
                    let v2_lty = self.lookup_type.lookup(v2_lexp.type_id)?;
                    let n2 = match type_arena[v2_lty.handle].inner {
                        crate::TypeInner::Vector { size, .. } => size as u8,
                        _ => return Err(Error::InvalidInnerType(v2_lexp.type_id)),
                    };
                    let v2_handle = v2_lexp.handle;

                    let mut components = Vec::with_capacity(inst.wc as usize - 5);
                    for _ in 0..components.capacity() {
                        let index = self.next()?;
                        let expr = if index < n1 as u32 {
                            crate::Expression::AccessIndex {
                                base: v1_handle,
                                index,
                            }
                        } else if index < n1 as u32 + n2 as u32 {
                            crate::Expression::AccessIndex {
                                base: v2_handle,
                                index: index - n1 as u32,
                            }
                        } else {
                            return Err(Error::InvalidAccessIndex(index));
                        };
                        components.push(expressions.append(expr));
                    }
                    let expr = crate::Expression::Compose {
                        ty: self.lookup_type.lookup(result_type_id)?.handle,
                        components,
                    };
                    self.lookup_expression.insert(
                        result_id,
                        LookupExpression {
                            handle: expressions.append(expr),
                            type_id: result_type_id,
                        },
                    );
                }
                Op::Bitcast
                | Op::ConvertSToF
                | Op::ConvertUToF
                | Op::ConvertFToU
                | Op::ConvertFToS => {
                    inst.expect_at_least(4)?;
                    let result_type_id = self.next()?;
                    let result_id = self.next()?;
                    let value_id = self.next()?;

                    let value_lexp = self.lookup_expression.lookup(value_id)?;
                    let ty_lookup = self.lookup_type.lookup(result_type_id)?;
                    let kind = type_arena[ty_lookup.handle]
                        .inner
                        .scalar_kind()
                        .ok_or(Error::InvalidAsType(ty_lookup.handle))?;

                    let expr = crate::Expression::As {
                        expr: value_lexp.handle,
                        kind,
                        convert: inst.op != Op::Bitcast,
                    };
                    self.lookup_expression.insert(
                        result_id,
                        LookupExpression {
                            handle: expressions.append(expr),
                            type_id: result_type_id,
                        },
                    );
                }
                Op::FunctionCall => {
                    inst.expect_at_least(4)?;
                    let result_type_id = self.next()?;
                    let result_id = self.next()?;
                    let func_id = self.next()?;

                    let mut arguments = Vec::with_capacity(inst.wc as usize - 4);
                    for _ in 0..arguments.capacity() {
                        let arg_id = self.next()?;
                        arguments.push(self.lookup_expression.lookup(arg_id)?.handle);
                    }
                    let expr = crate::Expression::Call {
                        // will be replaced by `Local()` after all the functions are parsed
                        origin: crate::FunctionOrigin::External(String::new()),
                        arguments,
                    };
                    let expr_handle = expressions.append(expr);
                    local_function_calls.insert(expr_handle, func_id);
                    self.lookup_expression.insert(
                        result_id,
                        LookupExpression {
                            handle: expr_handle,
                            type_id: result_type_id,
                        },
                    );
                }
                Op::ExtInst => {
                    let base_wc = 5;
                    inst.expect_at_least(base_wc)?;
                    let result_type_id = self.next()?;
                    let result_id = self.next()?;
                    let set_id = self.next()?;
                    if Some(set_id) != self.ext_glsl_id {
                        return Err(Error::UnsupportedExtInstSet(set_id));
                    }
                    let inst_id = self.next()?;
                    let name = match spirv::GLOp::from_u32(inst_id) {
                        Some(spirv::GLOp::FAbs) | Some(spirv::GLOp::SAbs) => {
                            inst.expect(base_wc + 1)?;
                            "abs"
                        }
                        Some(spirv::GLOp::FSign) | Some(spirv::GLOp::SSign) => {
                            inst.expect(base_wc + 1)?;
                            "sign"
                        }
                        Some(spirv::GLOp::Floor) => {
                            inst.expect(base_wc + 1)?;
                            "floor"
                        }
                        Some(spirv::GLOp::Ceil) => {
                            inst.expect(base_wc + 1)?;
                            "ceil"
                        }
                        Some(spirv::GLOp::Fract) => {
                            inst.expect(base_wc + 1)?;
                            "fract"
                        }
                        Some(spirv::GLOp::Sin) => {
                            inst.expect(base_wc + 1)?;
                            "sin"
                        }
                        Some(spirv::GLOp::Cos) => {
                            inst.expect(base_wc + 1)?;
                            "cos"
                        }
                        Some(spirv::GLOp::Tan) => {
                            inst.expect(base_wc + 1)?;
                            "tan"
                        }
                        Some(spirv::GLOp::Atan2) => {
                            inst.expect(base_wc + 2)?;
                            "atan2"
                        }
                        Some(spirv::GLOp::Pow) => {
                            inst.expect(base_wc + 2)?;
                            "pow"
                        }
                        Some(spirv::GLOp::MatrixInverse) => {
                            inst.expect(base_wc + 1)?;
                            "inverse"
                        }
                        Some(spirv::GLOp::FMix) => {
                            inst.expect(base_wc + 3)?;
                            "mix"
                        }
                        Some(spirv::GLOp::Step) => {
                            inst.expect(base_wc + 2)?;
                            "step"
                        }
                        Some(spirv::GLOp::SmoothStep) => {
                            inst.expect(base_wc + 3)?;
                            "smoothstep"
                        }
                        Some(spirv::GLOp::FMin) => {
                            inst.expect(base_wc + 2)?;
                            "min"
                        }
                        Some(spirv::GLOp::FMax) => {
                            inst.expect(base_wc + 2)?;
                            "max"
                        }
                        Some(spirv::GLOp::FClamp) => {
                            inst.expect(base_wc + 3)?;
                            "clamp"
                        }
                        Some(spirv::GLOp::Length) => {
                            inst.expect(base_wc + 1)?;
                            "length"
                        }
                        Some(spirv::GLOp::Distance) => {
                            inst.expect(base_wc + 2)?;
                            "distance"
                        }
                        Some(spirv::GLOp::Cross) => {
                            inst.expect(base_wc + 2)?;
                            "cross"
                        }
                        Some(spirv::GLOp::Normalize) => {
                            inst.expect(base_wc + 1)?;
                            "normalize"
                        }
                        Some(spirv::GLOp::Reflect) => {
                            inst.expect(base_wc + 2)?;
                            "reflect"
                        }
                        _ => return Err(Error::UnsupportedExtInst(inst_id)),
                    };

                    let mut arguments = Vec::with_capacity((inst.wc - base_wc) as usize);
                    for _ in 0..arguments.capacity() {
                        let arg_id = self.next()?;
                        arguments.push(self.lookup_expression.lookup(arg_id)?.handle);
                    }
                    let expr = crate::Expression::Call {
                        origin: crate::FunctionOrigin::External(name.to_string()),
                        arguments,
                    };
                    self.lookup_expression.insert(
                        result_id,
                        LookupExpression {
                            handle: expressions.append(expr),
                            type_id: result_type_id,
                        },
                    );
                }
                // Relational and Logical Instructions
                Op::LogicalNot => {
                    inst.expect(4)?;
                    self.parse_expr_unary_op(expressions, crate::UnaryOperator::Not)?;
                }
                op if inst.op >= Op::IEqual && inst.op <= Op::FUnordGreaterThanEqual => {
                    inst.expect(5)?;
                    self.parse_expr_binary_op(expressions, map_binary_operator(op)?)?;
                }
                Op::Kill => {
                    inst.expect(1)?;
                    break Terminator::Kill;
                }
                Op::Unreachable => {
                    inst.expect(1)?;
                    break Terminator::Unreachable;
                }
                Op::Return => {
                    inst.expect(1)?;
                    break Terminator::Return { value: None };
                }
                Op::ReturnValue => {
                    inst.expect(2)?;
                    let value_id = self.next()?;
                    let value_lexp = self.lookup_expression.lookup(value_id)?;
                    break Terminator::Return {
                        value: Some(value_lexp.handle),
                    };
                }
                Op::Branch => {
                    inst.expect(2)?;
                    let target_id = self.next()?;
                    break Terminator::Branch { target_id };
                }
                Op::BranchConditional => {
                    inst.expect_at_least(4)?;

                    let condition_id = self.next()?;
                    let condition = self.lookup_expression.lookup(condition_id)?.handle;

                    let true_id = self.next()?;
                    let false_id = self.next()?;

                    break Terminator::BranchConditional {
                        condition,
                        true_id,
                        false_id,
                    };
                }
                Op::Switch => {
                    inst.expect_at_least(3)?;

                    let selector = self.next()?;
                    let selector = self.lookup_expression[&selector].handle;
                    let default = self.next()?;

                    let mut targets = Vec::new();
                    for _ in 0..(inst.wc - 3) / 2 {
                        let literal = self.next()?;
                        let target = self.next()?;
                        targets.push((literal as i32, target));
                    }

                    break Terminator::Switch {
                        selector,
                        default,
                        targets,
                    };
                }
                Op::SelectionMerge => {
                    inst.expect(3)?;
                    let merge_block_id = self.next()?;
                    // TODO: Selection Control Mask
                    let _selection_control = self.next()?;
                    let continue_block_id = None;
                    merge = Some(MergeInstruction {
                        merge_block_id,
                        continue_block_id,
                    });
                }
                Op::LoopMerge => {
                    inst.expect_at_least(4)?;
                    let merge_block_id = self.next()?;
                    let continue_block_id = Some(self.next()?);

                    // TODO: Loop Control Parameters
                    for _ in 0..inst.wc - 3 {
                        self.next()?;
                    }

                    merge = Some(MergeInstruction {
                        merge_block_id,
                        continue_block_id,
                    });
                }
                _ => return Err(Error::UnsupportedInstruction(self.state, inst.op)),
            }
        };

        let mut block = Vec::new();
        for assignment in assignments.iter() {
            block.push(crate::Statement::Store {
                pointer: assignment.to,
                value: assignment.value,
            });
        }

        Ok(ControlFlowNode {
            id: block_id,
            ty: None,
            phis,
            block,
            terminator,
            merge,
        })
    }

    fn make_expression_storage(&mut self) -> Arena<crate::Expression> {
        let mut expressions = Arena::new();
        #[allow(clippy::panic)]
        {
            assert!(self.lookup_expression.is_empty());
        }
        // register global variables
        for (&id, var) in self.lookup_variable.iter() {
            let handle = expressions.append(crate::Expression::GlobalVariable(var.handle));
            self.lookup_expression.insert(
                id,
                LookupExpression {
                    type_id: var.type_id,
                    handle,
                },
            );
        }
        // register constants
        for (&id, con) in self.lookup_constant.iter() {
            let handle = expressions.append(crate::Expression::Constant(con.handle));
            self.lookup_expression.insert(
                id,
                LookupExpression {
                    type_id: con.type_id,
                    handle,
                },
            );
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

    pub fn parse(mut self) -> Result<crate::Module, Error> {
        let mut module = {
            if self.next()? != spirv::MAGIC_NUMBER {
                return Err(Error::InvalidHeader);
            }
            let _version_raw = self.next()?.to_le_bytes();
            let _generator = self.next()?;
            let _bound = self.next()?;
            let _schema = self.next()?;
            crate::Module::generate_empty()
        };

        while let Ok(inst) = self.next_inst() {
            use spirv::Op;
            log::debug!("\t{:?} [{}]", inst.op, inst.wc);
            match inst.op {
                Op::Capability => self.parse_capability(inst),
                Op::Extension => self.parse_extension(inst),
                Op::ExtInstImport => self.parse_ext_inst_import(inst),
                Op::MemoryModel => self.parse_memory_model(inst),
                Op::EntryPoint => self.parse_entry_point(inst),
                Op::ExecutionMode => self.parse_execution_mode(inst),
                Op::Source => self.parse_source(inst),
                Op::SourceExtension => self.parse_source_extension(inst),
                Op::Name => self.parse_name(inst),
                Op::MemberName => self.parse_member_name(inst),
                Op::Decorate => self.parse_decorate(inst),
                Op::MemberDecorate => self.parse_member_decorate(inst),
                Op::TypeVoid => self.parse_type_void(inst),
                Op::TypeBool => self.parse_type_bool(inst, &mut module),
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
                Op::Variable => self.parse_global_variable(inst, &mut module),
                Op::Function => self.parse_function(inst, &mut module),
                _ => Err(Error::UnsupportedInstruction(self.state, inst.op)), //TODO
            }?;
        }

        // Check all the images and samplers to have consistent comparison property.
        for (handle, flags) in self.handle_sampling.drain() {
            if !flags.contains(SamplingFlags::COMPARISON) {
                continue;
            }
            if flags == SamplingFlags::all() {
                return Err(Error::InconsistentComparisonSampling(handle));
            }
            let ty = module.types.get_mut(handle);
            match ty.inner {
                crate::TypeInner::Sampler { ref mut comparison } => {
                    #[allow(clippy::panic)]
                    {
                        assert!(!*comparison)
                    };
                    *comparison = true;
                }
                _ => {
                    return Err(Error::UnexpectedComparisonType(handle));
                }
            }
        }

        for dfc in self.deferred_function_calls.drain(..) {
            let dst_handle = *self.lookup_function.lookup(dfc.dst_id)?;
            let fun = match dfc.source {
                DeferredSource::Function(fun_handle) => module.functions.get_mut(fun_handle),
                DeferredSource::EntryPoint(stage, name) => {
                    &mut module
                        .entry_points
                        .get_mut(&(stage, name))
                        .unwrap()
                        .function
                }
            };
            match *fun.expressions.get_mut(dfc.expr_handle) {
                crate::Expression::Call {
                    ref mut origin,
                    arguments: _,
                } => *origin = crate::FunctionOrigin::Local(dst_handle),
                _ => unreachable!(),
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
            return Err(Error::UnsupportedExtension(name));
        }
        Ok(())
    }

    fn parse_ext_inst_import(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::Extension, inst.op)?;
        inst.expect_at_least(3)?;
        let result_id = self.next()?;
        let (name, left) = self.next_string(inst.wc - 2)?;
        if left != 0 {
            return Err(Error::InvalidOperand);
        }
        if !SUPPORTED_EXT_SETS.contains(&name.as_str()) {
            return Err(Error::UnsupportedExtSet(name));
        }
        self.ext_glsl_id = Some(result_id);
        Ok(())
    }

    fn parse_memory_model(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::MemoryModel, inst.op)?;
        inst.expect(3)?;
        let _addressing_model = self.next()?;
        let _memory_model = self.next()?;
        Ok(())
    }

    fn parse_entry_point(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::EntryPoint, inst.op)?;
        inst.expect_at_least(4)?;
        let exec_model = self.next()?;
        let exec_model = spirv::ExecutionModel::from_u32(exec_model)
            .ok_or(Error::UnsupportedExecutionModel(exec_model))?;
        let function_id = self.next()?;
        let (name, left) = self.next_string(inst.wc - 3)?;
        let ep = EntryPoint {
            stage: match exec_model {
                spirv::ExecutionModel::Vertex => crate::ShaderStage::Vertex,
                spirv::ExecutionModel::Fragment => crate::ShaderStage::Fragment,
                spirv::ExecutionModel::GLCompute => crate::ShaderStage::Compute,
                _ => return Err(Error::UnsupportedExecutionModel(exec_model as u32)),
            },
            name,
            early_depth_test: None,
            workgroup_size: [0; 3],
            function_id,
            variable_ids: self.data.by_ref().take(left as usize).collect(),
        };
        self.lookup_entry_point.insert(function_id, ep);
        Ok(())
    }

    fn parse_execution_mode(&mut self, inst: Instruction) -> Result<(), Error> {
        use spirv::ExecutionMode;

        self.switch(ModuleState::ExecutionMode, inst.op)?;
        inst.expect_at_least(3)?;

        let ep_id = self.next()?;
        let mode_id = self.next()?;
        let args: Vec<spirv::Word> = self.data.by_ref().take(inst.wc as usize - 3).collect();

        let ep = self
            .lookup_entry_point
            .get_mut(&ep_id)
            .ok_or(Error::InvalidId(ep_id))?;
        let mode = spirv::ExecutionMode::from_u32(mode_id)
            .ok_or(Error::UnsupportedExecutionMode(mode_id))?;

        match mode {
            ExecutionMode::EarlyFragmentTests => {
                if ep.early_depth_test.is_none() {
                    ep.early_depth_test = Some(crate::EarlyDepthTest { conservative: None });
                }
            }
            ExecutionMode::DepthUnchanged => {
                ep.early_depth_test = Some(crate::EarlyDepthTest {
                    conservative: Some(crate::ConservativeDepth::Unchanged),
                });
            }
            ExecutionMode::DepthGreater => {
                ep.early_depth_test = Some(crate::EarlyDepthTest {
                    conservative: Some(crate::ConservativeDepth::GreaterEqual),
                });
            }
            ExecutionMode::DepthLess => {
                ep.early_depth_test = Some(crate::EarlyDepthTest {
                    conservative: Some(crate::ConservativeDepth::LessEqual),
                });
            }
            ExecutionMode::DepthReplacing => {
                // Ignored because it can be deduced from the IR.
            }
            ExecutionMode::OriginUpperLeft => {
                // Ignored because the other option (OriginLowerLeft) is not valid in Vulkan mode.
            }
            ExecutionMode::LocalSize => {
                ep.workgroup_size = [args[0], args[1], args[2]];
            }
            _ => {
                return Err(Error::UnsupportedExecutionMode(mode_id));
            }
        }

        Ok(())
    }

    fn parse_source(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::Source, inst.op)?;
        for _ in 1..inst.wc {
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
        self.future_decor.entry(id).or_default().name = Some(name);
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
            .name = Some(name);
        Ok(())
    }

    fn parse_decorate(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::Annotation, inst.op)?;
        inst.expect_at_least(3)?;
        let id = self.next()?;
        let mut dec = self.future_decor.remove(&id).unwrap_or_default();
        self.next_decoration(inst, 2, &mut dec)?;
        self.future_decor.insert(id, dec);
        Ok(())
    }

    fn parse_member_decorate(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::Annotation, inst.op)?;
        inst.expect_at_least(4)?;
        let id = self.next()?;
        let member = self.next()?;

        let mut dec = self
            .future_member_decor
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

    fn parse_type_bool(
        &mut self,
        inst: Instruction,
        module: &mut crate::Module,
    ) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect(2)?;
        let id = self.next()?;
        let inner = crate::TypeInner::Scalar {
            kind: crate::ScalarKind::Bool,
            width: 1,
        };
        self.lookup_type.insert(
            id,
            LookupType {
                handle: module.types.append(crate::Type {
                    name: self.future_decor.remove(&id).and_then(|dec| dec.name),
                    inner,
                }),
                base_id: None,
            },
        );
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
            width: map_width(width)?,
        };
        self.lookup_type.insert(
            id,
            LookupType {
                handle: module.types.append(crate::Type {
                    name: self.future_decor.remove(&id).and_then(|dec| dec.name),
                    inner,
                }),
                base_id: None,
            },
        );
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
            width: map_width(width)?,
        };
        self.lookup_type.insert(
            id,
            LookupType {
                handle: module.types.append(crate::Type {
                    name: self.future_decor.remove(&id).and_then(|dec| dec.name),
                    inner,
                }),
                base_id: None,
            },
        );
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
        self.lookup_type.insert(
            id,
            LookupType {
                handle: module.types.append(crate::Type {
                    name: self.future_decor.remove(&id).and_then(|dec| dec.name),
                    inner,
                }),
                base_id: Some(type_id),
            },
        );
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
            crate::TypeInner::Vector { size, width, .. } => crate::TypeInner::Matrix {
                columns: map_vector_size(num_columns)?,
                rows: size,
                width,
            },
            _ => return Err(Error::InvalidInnerType(vector_type_id)),
        };
        self.lookup_type.insert(
            id,
            LookupType {
                handle: module.types.append(crate::Type {
                    name: self.future_decor.remove(&id).and_then(|dec| dec.name),
                    inner,
                }),
                base_id: Some(vector_type_id),
            },
        );
        Ok(())
    }

    fn parse_type_function(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect_at_least(3)?;
        let id = self.next()?;
        let return_type_id = self.next()?;
        let parameter_type_ids = self.data.by_ref().take(inst.wc as usize - 3).collect();
        self.lookup_function_type.insert(
            id,
            LookupFunctionType {
                parameter_type_ids,
                return_type_id,
            },
        );
        Ok(())
    }

    fn parse_type_pointer(
        &mut self,
        inst: Instruction,
        _module: &mut crate::Module,
    ) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect(4)?;
        let id = self.next()?;
        let _storage = self.next()?;
        let type_id = self.next()?;
        let type_lookup = self.lookup_type.lookup(type_id)?.clone();
        self.lookup_type.insert(id, type_lookup); // don't register pointers in the IR
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
        let length_id = self.next()?;
        let length_const = self.lookup_constant.lookup(length_id)?;

        let decor = self.future_decor.remove(&id);
        let inner = crate::TypeInner::Array {
            base: self.lookup_type.lookup(type_id)?.handle,
            size: crate::ArraySize::Constant(length_const.handle),
            stride: decor.as_ref().and_then(|dec| dec.array_stride),
        };
        self.lookup_type.insert(
            id,
            LookupType {
                handle: module.types.append(crate::Type {
                    name: decor.and_then(|dec| dec.name),
                    inner,
                }),
                base_id: Some(type_id),
            },
        );
        Ok(())
    }

    fn parse_type_runtime_array(
        &mut self,
        inst: Instruction,
        module: &mut crate::Module,
    ) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect(3)?;
        let id = self.next()?;
        let type_id = self.next()?;

        let decor = self.future_decor.remove(&id);
        let inner = crate::TypeInner::Array {
            base: self.lookup_type.lookup(type_id)?.handle,
            size: crate::ArraySize::Dynamic,
            stride: decor.as_ref().and_then(|dec| dec.array_stride),
        };
        self.lookup_type.insert(
            id,
            LookupType {
                handle: module.types.append(crate::Type {
                    name: decor.and_then(|dec| dec.name),
                    inner,
                }),
                base_id: Some(type_id),
            },
        );
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
        let parent_decor = self.future_decor.remove(&id);
        let is_buffer_block = parent_decor
            .as_ref()
            .map_or(false, |decor| match decor.block {
                Some(Block { buffer }) => buffer,
                _ => false,
            });

        let mut members = Vec::with_capacity(inst.wc as usize - 2);
        let mut member_type_ids = Vec::with_capacity(members.capacity());
        for i in 0..u32::from(inst.wc) - 2 {
            let type_id = self.next()?;
            member_type_ids.push(type_id);
            let ty = self.lookup_type.lookup(type_id)?.handle;
            let decor = self
                .future_member_decor
                .remove(&(id, i))
                .unwrap_or_default();
            let origin = decor.get_origin()?;
            members.push(crate::StructMember {
                name: decor.name,
                origin,
                ty,
            });
        }
        let inner = crate::TypeInner::Struct { members };
        let ty_handle = module.types.append(crate::Type {
            name: parent_decor.and_then(|dec| dec.name),
            inner,
        });

        if is_buffer_block {
            self.lookup_storage_buffer_types.insert(ty_handle);
        }
        for (i, type_id) in member_type_ids.into_iter().enumerate() {
            self.lookup_member_type_id
                .insert((ty_handle, i as u32), type_id);
        }
        self.lookup_type.insert(
            id,
            LookupType {
                handle: ty_handle,
                base_id: None,
            },
        );
        Ok(())
    }

    fn parse_type_image(
        &mut self,
        inst: Instruction,
        module: &mut crate::Module,
    ) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect(9)?;

        let id = self.next()?;
        let sample_type_id = self.next()?;
        let dim = self.next()?;
        let _is_depth = self.next()?;
        let is_array = self.next()? != 0;
        let is_msaa = self.next()? != 0;
        let _is_sampled = self.next()?;
        let format = self.next()?;

        let base_handle = self.lookup_type.lookup(sample_type_id)?.handle;
        let kind = module.types[base_handle]
            .inner
            .scalar_kind()
            .ok_or(Error::InvalidImageBaseType(base_handle))?;

        let class = if format != 0 {
            crate::ImageClass::Storage(map_image_format(format)?)
        } else {
            crate::ImageClass::Sampled {
                kind,
                multi: is_msaa,
            }
        };

        let decor = self.future_decor.remove(&id).unwrap_or_default();

        let inner = crate::TypeInner::Image {
            class,
            dim: map_image_dim(dim)?,
            arrayed: is_array,
        };
        let handle = module.types.append(crate::Type {
            name: decor.name,
            inner,
        });
        log::debug!("\t\ttracking {:?} for sampling properties", handle);
        self.handle_sampling.insert(handle, SamplingFlags::empty());
        self.lookup_type.insert(
            id,
            LookupType {
                handle,
                base_id: Some(sample_type_id),
            },
        );
        Ok(())
    }

    fn parse_type_sampled_image(&mut self, inst: Instruction) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect(3)?;
        let id = self.next()?;
        let image_id = self.next()?;
        self.lookup_type.insert(
            id,
            LookupType {
                handle: self.lookup_type.lookup(image_id)?.handle,
                base_id: Some(image_id),
            },
        );
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
        let decor = self.future_decor.remove(&id).unwrap_or_default();
        // The comparison bit is temporary, will be overwritten based on the
        // accumulated sampling flags at the end.
        let inner = crate::TypeInner::Sampler { comparison: false };
        let handle = module.types.append(crate::Type {
            name: decor.name,
            inner,
        });
        log::debug!("\t\ttracking {:?} for sampling properties", handle);
        self.handle_sampling.insert(handle, SamplingFlags::empty());
        self.lookup_type.insert(
            id,
            LookupType {
                handle,
                base_id: None,
            },
        );
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
        let inner = match module.types[ty].inner {
            crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Uint,
                width,
            } => {
                let low = self.next()?;
                let high = if width > 4 {
                    inst.expect(4)?;
                    self.next()?
                } else {
                    0
                };
                crate::ConstantInner::Uint((u64::from(high) << 32) | u64::from(low))
            }
            crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Sint,
                width,
            } => {
                use std::cmp::Ordering;
                let low = self.next()?;
                let high = match width.cmp(&4) {
                    Ordering::Less => return Err(Error::InvalidTypeWidth(u32::from(width))),
                    Ordering::Greater => {
                        inst.expect(4)?;
                        self.next()?
                    }
                    Ordering::Equal => 0,
                };
                crate::ConstantInner::Sint(((u64::from(high) << 32) | u64::from(low)) as i64)
            }
            crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Float,
                width,
            } => {
                let low = self.next()?;
                let extended = match width {
                    4 => f64::from(f32::from_bits(low)),
                    8 => {
                        inst.expect(4)?;
                        let high = self.next()?;
                        f64::from_bits((u64::from(high) << 32) | u64::from(low))
                    }
                    _ => return Err(Error::InvalidTypeWidth(u32::from(width))),
                };
                crate::ConstantInner::Float(extended)
            }
            _ => return Err(Error::UnsupportedType(type_lookup.handle)),
        };
        self.lookup_constant.insert(
            id,
            LookupConstant {
                handle: module.constants.append(crate::Constant {
                    name: self.future_decor.remove(&id).and_then(|dec| dec.name),
                    specialization: None, //TODO
                    inner,
                    ty,
                }),
                type_id,
            },
        );
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

        self.lookup_constant.insert(
            id,
            LookupConstant {
                handle: module.constants.append(crate::Constant {
                    name: self.future_decor.remove(&id).and_then(|dec| dec.name),
                    specialization: None,
                    inner: crate::ConstantInner::Composite(constituents),
                    ty,
                }),
                type_id,
            },
        );

        Ok(())
    }

    fn parse_global_variable(
        &mut self,
        inst: Instruction,
        module: &mut crate::Module,
    ) -> Result<(), Error> {
        self.switch(ModuleState::Type, inst.op)?;
        inst.expect_at_least(4)?;
        let type_id = self.next()?;
        let id = self.next()?;
        let storage_class = self.next()?;
        let init = if inst.wc > 4 {
            inst.expect(5)?;
            let init_id = self.next()?;
            let lconst = self.lookup_constant.lookup(init_id)?;
            Some(lconst.handle)
        } else {
            None
        };
        let lookup_type = self.lookup_type.lookup(type_id)?;
        let dec = self
            .future_decor
            .remove(&id)
            .ok_or(Error::InvalidBinding(id))?;

        let class = {
            use spirv::StorageClass as Sc;
            match Sc::from_u32(storage_class) {
                Some(Sc::Function) => crate::StorageClass::Function,
                Some(Sc::Input) => crate::StorageClass::Input,
                Some(Sc::Output) => crate::StorageClass::Output,
                Some(Sc::Private) => crate::StorageClass::Private,
                Some(Sc::UniformConstant) => crate::StorageClass::Handle,
                Some(Sc::StorageBuffer) => crate::StorageClass::Storage,
                Some(Sc::Uniform) => {
                    if self
                        .lookup_storage_buffer_types
                        .contains(&lookup_type.handle)
                    {
                        crate::StorageClass::Storage
                    } else {
                        crate::StorageClass::Uniform
                    }
                }
                Some(Sc::Workgroup) => crate::StorageClass::WorkGroup,
                Some(Sc::PushConstant) => crate::StorageClass::PushConstant,
                _ => return Err(Error::UnsupportedStorageClass(storage_class)),
            }
        };

        let binding = match (class, &module.types[lookup_type.handle].inner) {
            (crate::StorageClass::Input, &crate::TypeInner::Struct { .. })
            | (crate::StorageClass::Output, &crate::TypeInner::Struct { .. }) => None,
            _ => Some(dec.get_binding().ok_or(Error::InvalidBinding(id))?),
        };
        let is_storage = match module.types[lookup_type.handle].inner {
            crate::TypeInner::Struct { .. } => class == crate::StorageClass::Storage,
            crate::TypeInner::Image {
                class: crate::ImageClass::Storage(_),
                ..
            } => true,
            _ => false,
        };

        let storage_access = if is_storage {
            let mut access = crate::StorageAccess::all();
            if dec.flags.contains(DecorationFlags::NON_READABLE) {
                access ^= crate::StorageAccess::LOAD;
            }
            if dec.flags.contains(DecorationFlags::NON_WRITABLE) {
                access ^= crate::StorageAccess::STORE;
            }
            access
        } else {
            crate::StorageAccess::empty()
        };

        let var = crate::GlobalVariable {
            name: dec.name,
            class,
            binding,
            ty: lookup_type.handle,
            init,
            interpolation: dec.interpolation,
            storage_access,
        };
        self.lookup_variable.insert(
            id,
            LookupVariable {
                handle: module.global_variables.append(var),
                type_id,
            },
        );
        Ok(())
    }
}

pub fn parse_u8_slice(data: &[u8], options: &Options) -> Result<crate::Module, Error> {
    if data.len() % 4 != 0 {
        return Err(Error::IncompleteData);
    }

    let words = data
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()));
    Parser::new(words, options).parse()
}

#[cfg(test)]
mod test {
    #[test]
    fn parse() {
        let bin = vec![
            // Magic number.           Version number: 1.0.
            0x03, 0x02, 0x23, 0x07, 0x00, 0x00, 0x01, 0x00,
            // Generator number: 0.    Bound: 0.
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Reserved word: 0.
            0x00, 0x00, 0x00, 0x00, // OpMemoryModel.          Logical.
            0x0e, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, // GLSL450.
            0x01, 0x00, 0x00, 0x00,
        ];
        let _ = super::parse_u8_slice(&bin, &Default::default()).unwrap();
    }
}
