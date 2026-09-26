//! Lowering of workgroup memory zero initialization into IR.
//!
//! See [`zero_initialize_workgroup_memory`].

use alloc::{borrow::Cow, boxed::Box, vec::Vec};

use thiserror::Error;

use crate::{
    proc::{Emitter, GlobalCtx, IndexableLength},
    valid::{Capabilities, ModuleInfo, ValidationError, ValidationFlags, Validator},
    AddressSpace, Arena, BinaryOperator, Binding, Block, BuiltIn, Expression, FunctionArgument,
    GlobalVariable, Handle, Literal, LocalVariable, Module, Scalar, ShaderStage, Span, Statement,
    Type, TypeInner, UniqueArena, WithSpan,
};

#[derive(Error, Debug, Clone)]
pub enum WorkgroupInitError {
    #[error("workgroup variable has an array size that is not a constant")]
    UnresolvedArraySize,
    #[error("workgroup variable has more than u32::MAX elements")]
    TooManyElements,
    #[error(transparent)]
    ValidationError(#[from] Box<WithSpan<ValidationError>>),
}

/// Zero initialize workgroup memory with IR statements at the start of each
/// compute, task, and mesh entry point.
///
/// Arrays are initialized by all invocations of the workgroup, each storing
/// a strided subset of the elements, instead of by a single invocation. The
/// initialization is followed by a workgroup control barrier.
///
/// `module` must be valid, and overrides must have been processed with
/// [`process_overrides`], so that workgroup sizes and array sizes are known.
///
/// Backends should not additionally zero initialize workgroup memory for the
/// returned module.
///
/// If no entry point uses workgroup memory, this returns the inputs unchanged.
/// Otherwise, it returns the lowered module and its validation results.
///
/// [`process_overrides`]: super::pipeline_constants::process_overrides
pub fn zero_initialize_workgroup_memory<'a>(
    module: Cow<'a, Module>,
    module_info: Cow<'a, ModuleInfo>,
) -> Result<(Cow<'a, Module>, Cow<'a, ModuleInfo>), WorkgroupInitError> {
    let targets = module
        .entry_points
        .iter()
        .enumerate()
        .map(|(index, ep)| {
            let fun_info = module_info.get_entry_point(index);
            module
                .global_variables
                .iter()
                .filter(|&(handle, var)| {
                    let initialized_space = match var.space {
                        AddressSpace::WorkGroup => true,
                        AddressSpace::TaskPayload => ep.stage == ShaderStage::Task,
                        _ => false,
                    };
                    initialized_space && !fun_info[handle].is_empty()
                })
                .map(|(handle, _)| handle)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let is_target = |(ep, vars): (&crate::EntryPoint, &Vec<_>)| {
        matches!(
            ep.stage,
            ShaderStage::Compute | ShaderStage::Task | ShaderStage::Mesh
        ) && !vars.is_empty()
    };
    if !module.entry_points.iter().zip(&targets).any(is_target) {
        return Ok((module, module_info));
    }

    let mut module = module.into_owned();
    for (index, vars) in targets.into_iter().enumerate() {
        if is_target((&module.entry_points[index], &vars)) {
            lower_entry_point(&mut module, index, &vars)?;
        }
    }

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let module_info = validator.validate_resolved_overrides(&module)?;
    Ok((Cow::Owned(module), Cow::Owned(module_info)))
}

#[derive(Clone, Copy)]
enum Step {
    Element(u32),
    Member(u32),
}

/// A part of a variable that is initialized with a single store per element.
struct Leaf {
    steps: Vec<Step>,
    ty: Handle<Type>,
    count: u32,
}

fn contains_array(types: &UniqueArena<Type>, ty: Handle<Type>) -> bool {
    match types[ty].inner {
        TypeInner::Array { .. } => true,
        TypeInner::Struct { ref members, .. } => {
            members.iter().any(|m| contains_array(types, m.ty))
        }
        _ => false,
    }
}

fn collect_leaves(
    gctx: GlobalCtx,
    ty: Handle<Type>,
    steps: &mut Vec<Step>,
    count: u32,
    leaves: &mut Vec<Leaf>,
) -> Result<(), WorkgroupInitError> {
    let types = gctx.types;
    let inner = &types[ty].inner;
    let is_leaf = match *inner {
        TypeInner::Array { .. } | TypeInner::Struct { .. } => {
            inner.is_constructible(types) && !contains_array(types, ty)
        }
        _ => true,
    };
    if is_leaf {
        leaves.push(Leaf {
            steps: steps.clone(),
            ty,
            count,
        });
        return Ok(());
    }

    match *inner {
        TypeInner::Array { base, size, .. } => {
            let Ok(IndexableLength::Known(len)) = size.resolve(gctx) else {
                return Err(WorkgroupInitError::UnresolvedArraySize);
            };
            let count = count
                .checked_mul(len)
                .ok_or(WorkgroupInitError::TooManyElements)?;
            steps.push(Step::Element(len));
            collect_leaves(gctx, base, steps, count, leaves)?;
            steps.pop();
        }
        TypeInner::Struct { ref members, .. } => {
            for (index, member) in members.iter().enumerate() {
                steps.push(Step::Member(index as u32));
                collect_leaves(gctx, member.ty, steps, count, leaves)?;
                steps.pop();
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}

struct Builder<'a> {
    expressions: &'a mut Arena<Expression>,
    emitter: Emitter,
}

impl Builder<'_> {
    fn pre_emitted(&mut self, expr: Expression) -> Handle<Expression> {
        self.expressions.append(expr, Span::UNDEFINED)
    }

    fn start(&mut self) {
        self.emitter.start(self.expressions);
    }

    fn finish(&mut self, block: &mut Block) {
        block.extend(self.emitter.finish(self.expressions));
    }

    fn emitted(&mut self, expr: Expression) -> Handle<Expression> {
        debug_assert!(self.emitter.is_running() && !expr.needs_pre_emit());
        self.expressions.append(expr, Span::UNDEFINED)
    }

    fn binary(
        &mut self,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emitted(Expression::Binary { op, left, right })
    }

    fn u32(&mut self, value: u32) -> Handle<Expression> {
        self.pre_emitted(Expression::Literal(Literal::U32(value)))
    }

    /// Emit a pointer to the leaf element with flat index `index`.
    fn leaf_pointer(
        &mut self,
        block: &mut Block,
        global: Handle<GlobalVariable>,
        steps: &[Step],
        index: Option<Handle<Expression>>,
    ) -> Handle<Expression> {
        // Literals must be appended outside of the emit range.
        let mut divisors = Vec::new();
        for step in steps.iter().rev() {
            if let Step::Element(len) = *step {
                divisors.push(self.u32(len));
            }
        }
        let mut pointer = self.pre_emitted(Expression::GlobalVariable(global));
        let mut indices = Vec::new();
        if index.is_none() {
            let zero = self.u32(0);
            indices.resize(divisors.len(), zero);
        }

        self.start();
        if let Some(mut remaining) = index {
            let last = divisors.len().saturating_sub(1);
            for (i, &len) in divisors.iter().enumerate() {
                if i == last {
                    indices.push(remaining);
                } else {
                    indices.push(self.binary(BinaryOperator::Modulo, remaining, len));
                    remaining = self.binary(BinaryOperator::Divide, remaining, len);
                }
            }
        }
        for step in steps {
            pointer = match *step {
                Step::Element(_) => {
                    let index = indices.pop().expect("one index per array step");
                    self.emitted(Expression::Access {
                        base: pointer,
                        index,
                    })
                }
                Step::Member(index) => self.emitted(Expression::AccessIndex {
                    base: pointer,
                    index,
                }),
            };
        }
        self.finish(block);
        pointer
    }
}

fn lower_entry_point(
    module: &mut Module,
    ep_index: usize,
    vars: &[Handle<GlobalVariable>],
) -> Result<(), WorkgroupInitError> {
    let u32_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::U32),
        },
        Span::UNDEFINED,
    );

    let mut leaves = Vec::new();
    for &var in vars {
        let mut var_leaves = Vec::new();
        collect_leaves(
            module.to_ctx(),
            module.global_variables[var].ty,
            &mut Vec::new(),
            1,
            &mut var_leaves,
        )?;
        leaves.extend(var_leaves.into_iter().map(|leaf| (var, leaf)));
    }

    // Atomics can't be zero constructed, so store a zero of their scalar type.
    let mut zero_types = Vec::with_capacity(leaves.len());
    for &(_, ref leaf) in &leaves {
        let ty = match module.types[leaf.ty].inner {
            TypeInner::Atomic(scalar) => module.types.insert(
                Type {
                    name: None,
                    inner: TypeInner::Scalar(scalar),
                },
                Span::UNDEFINED,
            ),
            _ => leaf.ty,
        };
        zero_types.push(ty);
    }

    let ep = &mut module.entry_points[ep_index];
    let invocations = ep
        .workgroup_size
        .iter()
        .try_fold(1u32, |acc, &n| acc.checked_mul(n))
        .ok_or(WorkgroupInitError::TooManyElements)?;
    let function = &mut ep.function;

    let existing_index = function.arguments.iter().enumerate().find_map(|(i, arg)| {
        let is_index = |binding: &Option<Binding>| {
            *binding == Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex))
        };
        if is_index(&arg.binding) {
            return Some((i, None));
        }
        match module.types[arg.ty].inner {
            TypeInner::Struct { ref members, .. } => members
                .iter()
                .position(|m| is_index(&m.binding))
                .map(|m| (i, Some(m as u32))),
            _ => None,
        }
    });
    let (arg_index, member) = existing_index.unwrap_or_else(|| {
        function.arguments.push(FunctionArgument {
            name: Some("local_invocation_index".into()),
            ty: u32_ty,
            binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)),
        });
        (function.arguments.len() - 1, None)
    });

    // The loop index must not wrap when the stride is added to it.
    if leaves
        .iter()
        .any(|&(_, ref leaf)| leaf.count.checked_add(invocations).is_none())
    {
        return Err(WorkgroupInitError::TooManyElements);
    }
    let needs_loop = leaves.iter().any(|&(_, ref leaf)| leaf.count > invocations);
    let index_var = needs_loop.then(|| {
        function.local_variables.append(
            LocalVariable {
                name: Some("zero_init_index".into()),
                ty: u32_ty,
                init: None,
            },
            Span::UNDEFINED,
        )
    });

    let mut builder = Builder {
        expressions: &mut function.expressions,
        emitter: Emitter::default(),
    };
    let mut prologue = Block::new();

    let argument = builder.pre_emitted(Expression::FunctionArgument(arg_index as u32));
    let local_index = match member {
        Some(member) => {
            builder.start();
            let access = builder.emitted(Expression::AccessIndex {
                base: argument,
                index: member,
            });
            builder.finish(&mut prologue);
            access
        }
        None => argument,
    };

    let mut single = Block::new();
    for (&(var, ref leaf), &zero_ty) in leaves.iter().zip(&zero_types) {
        let zero = builder.pre_emitted(Expression::ZeroValue(zero_ty));
        if leaf.count == 1 {
            let pointer = builder.leaf_pointer(&mut single, var, &leaf.steps, None);
            single.push(
                Statement::Store {
                    pointer,
                    value: zero,
                },
                Span::UNDEFINED,
            );
        } else if leaf.count <= invocations {
            let count = builder.u32(leaf.count);
            builder.start();
            let in_bounds = builder.binary(BinaryOperator::Less, local_index, count);
            builder.finish(&mut prologue);

            let mut accept = Block::new();
            let pointer = builder.leaf_pointer(&mut accept, var, &leaf.steps, Some(local_index));
            accept.push(
                Statement::Store {
                    pointer,
                    value: zero,
                },
                Span::UNDEFINED,
            );
            prologue.push(
                Statement::If {
                    condition: in_bounds,
                    accept,
                    reject: Block::new(),
                },
                Span::UNDEFINED,
            );
        } else {
            // `local_index < invocations < count`, so the first iteration is
            // always in bounds.
            let index_var = index_var.expect("a loop needs the index variable");
            let index_pointer = builder.pre_emitted(Expression::LocalVariable(index_var));
            prologue.push(
                Statement::Store {
                    pointer: index_pointer,
                    value: local_index,
                },
                Span::UNDEFINED,
            );

            let mut body = Block::new();
            builder.start();
            let index = builder.emitted(Expression::Load {
                pointer: index_pointer,
            });
            builder.finish(&mut body);
            let pointer = builder.leaf_pointer(&mut body, var, &leaf.steps, Some(index));
            body.push(
                Statement::Store {
                    pointer,
                    value: zero,
                },
                Span::UNDEFINED,
            );

            let mut continuing = Block::new();
            let stride = builder.u32(invocations);
            let count = builder.u32(leaf.count);
            builder.start();
            let index = builder.emitted(Expression::Load {
                pointer: index_pointer,
            });
            let next = builder.binary(BinaryOperator::Add, index, stride);
            builder.finish(&mut continuing);
            continuing.push(
                Statement::Store {
                    pointer: index_pointer,
                    value: next,
                },
                Span::UNDEFINED,
            );
            builder.start();
            let done = builder.binary(BinaryOperator::GreaterEqual, next, count);
            builder.finish(&mut continuing);

            prologue.push(
                Statement::Loop {
                    body,
                    continuing,
                    break_if: Some(done),
                },
                Span::UNDEFINED,
            );
        }
    }

    if !single.is_empty() {
        let zero = builder.u32(0);
        builder.start();
        let is_first = builder.binary(BinaryOperator::Equal, local_index, zero);
        builder.finish(&mut prologue);
        prologue.push(
            Statement::If {
                condition: is_first,
                accept: single,
                reject: Block::new(),
            },
            Span::UNDEFINED,
        );
    }

    prologue.push(
        Statement::ControlBarrier(crate::Barrier::WORK_GROUP),
        Span::UNDEFINED,
    );

    prologue.extend_block(core::mem::take(&mut function.body));
    function.body = prologue;
    Ok(())
}
