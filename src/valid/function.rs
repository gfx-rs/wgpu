#[cfg(feature = "validate")]
use crate::arena::{Arena, UniqueArena};
use crate::arena::{BadHandle, Handle};

use super::{
    analyzer::{UniformityDisruptor, UniformityRequirements},
    ExpressionError, FunctionInfo, ModuleInfo,
};
use crate::span::WithSpan;
#[cfg(feature = "validate")]
use crate::span::{AddSpan as _, MapErrWithSpan as _};

#[cfg(feature = "validate")]
use bit_set::BitSet;

#[derive(Clone, Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum CallError {
    #[error(transparent)]
    BadHandle(#[from] BadHandle),
    #[error("The callee is declared after the caller")]
    ForwardDeclaredFunction,
    #[error("Argument {index} expression is invalid")]
    Argument {
        index: usize,
        #[source]
        error: ExpressionError,
    },
    #[error("Result expression {0:?} has already been introduced earlier")]
    ResultAlreadyInScope(Handle<crate::Expression>),
    #[error("Result value is invalid")]
    ResultValue(#[source] ExpressionError),
    #[error("Requires {required} arguments, but {seen} are provided")]
    ArgumentCount { required: usize, seen: usize },
    #[error("Argument {index} value {seen_expression:?} doesn't match the type {required:?}")]
    ArgumentType {
        index: usize,
        required: Handle<crate::Type>,
        seen_expression: Handle<crate::Expression>,
    },
    #[error("The emitted expression doesn't match the call")]
    ExpressionMismatch(Option<Handle<crate::Expression>>),
}

#[derive(Clone, Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum AtomicError {
    #[error("Pointer {0:?} to atomic is invalid.")]
    InvalidPointer(Handle<crate::Expression>),
    #[error("Operand {0:?} has invalid type.")]
    InvalidOperand(Handle<crate::Expression>),
    #[error("Result expression {0:?} has already been introduced earlier")]
    ResultAlreadyInScope(Handle<crate::Expression>),
    #[error("Result type for {0:?} doesn't match the statement")]
    ResultTypeMismatch(Handle<crate::Expression>),
}

#[derive(Clone, Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum LocalVariableError {
    #[error("Local variable has a type {0:?} that can't be stored in a local variable.")]
    InvalidType(Handle<crate::Type>),
    #[error("Initializer doesn't match the variable type")]
    InitializerType,
}

#[derive(Clone, Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum FunctionError {
    #[error(transparent)]
    BadHandle(#[from] BadHandle),
    #[error("Expression {handle:?} is invalid")]
    Expression {
        handle: Handle<crate::Expression>,
        #[source]
        error: ExpressionError,
    },
    #[error("Expression {0:?} can't be introduced - it's already in scope")]
    ExpressionAlreadyInScope(Handle<crate::Expression>),
    #[error("Local variable {handle:?} '{name}' is invalid")]
    LocalVariable {
        handle: Handle<crate::LocalVariable>,
        name: String,
        #[source]
        error: LocalVariableError,
    },
    #[error("Argument '{name}' at index {index} has a type that can't be passed into functions.")]
    InvalidArgumentType { index: usize, name: String },
    #[error("The function's given return type cannot be returned from functions")]
    NonConstructibleReturnType,
    #[error("Argument '{name}' at index {index} is a pointer of space {space:?}, which can't be passed into functions.")]
    InvalidArgumentPointerSpace {
        index: usize,
        name: String,
        space: crate::AddressSpace,
    },
    #[error("There are instructions after `return`/`break`/`continue`")]
    InstructionsAfterReturn,
    #[error("The `break` is used outside of a `loop` or `switch` context")]
    BreakOutsideOfLoopOrSwitch,
    #[error("The `continue` is used outside of a `loop` context")]
    ContinueOutsideOfLoop,
    #[error("The `return` is called within a `continuing` block")]
    InvalidReturnSpot,
    #[error("The `return` value {0:?} does not match the function return value")]
    InvalidReturnType(Option<Handle<crate::Expression>>),
    #[error("The `if` condition {0:?} is not a boolean scalar")]
    InvalidIfType(Handle<crate::Expression>),
    #[error("The `switch` value {0:?} is not an integer scalar")]
    InvalidSwitchType(Handle<crate::Expression>),
    #[error("Multiple `switch` cases for {0:?} are present")]
    ConflictingSwitchCase(i32),
    #[error("The `switch` is missing a `default` case")]
    MissingDefaultCase,
    #[error("Multiple `default` cases are present")]
    MultipleDefaultCases,
    #[error("The last `switch` case contains a `falltrough`")]
    LastCaseFallTrough,
    #[error("The pointer {0:?} doesn't relate to a valid destination for a store")]
    InvalidStorePointer(Handle<crate::Expression>),
    #[error("The value {0:?} can not be stored")]
    InvalidStoreValue(Handle<crate::Expression>),
    #[error("Store of {value:?} into {pointer:?} doesn't have matching types")]
    InvalidStoreTypes {
        pointer: Handle<crate::Expression>,
        value: Handle<crate::Expression>,
    },
    #[error("Image store parameters are invalid")]
    InvalidImageStore(#[source] ExpressionError),
    #[error("Call to {function:?} is invalid")]
    InvalidCall {
        function: Handle<crate::Function>,
        #[source]
        error: CallError,
    },
    #[error("Atomic operation is invalid")]
    InvalidAtomic(#[from] AtomicError),
    #[error(
        "Required uniformity of control flow for {0:?} in {1:?} is not fulfilled because of {2:?}"
    )]
    NonUniformControlFlow(
        UniformityRequirements,
        Handle<crate::Expression>,
        UniformityDisruptor,
    ),
}

bitflags::bitflags! {
    #[repr(transparent)]
    struct ControlFlowAbility: u8 {
        /// The control can return out of this block.
        const RETURN = 0x1;
        /// The control can break.
        const BREAK = 0x2;
        /// The control can continue.
        const CONTINUE = 0x4;
    }
}

#[cfg(feature = "validate")]
struct BlockInfo {
    stages: super::ShaderStages,
    finished: bool,
}

#[cfg(feature = "validate")]
struct BlockContext<'a> {
    abilities: ControlFlowAbility,
    info: &'a FunctionInfo,
    expressions: &'a Arena<crate::Expression>,
    types: &'a UniqueArena<crate::Type>,
    global_vars: &'a Arena<crate::GlobalVariable>,
    functions: &'a Arena<crate::Function>,
    prev_infos: &'a [FunctionInfo],
    return_type: Option<Handle<crate::Type>>,
}

#[cfg(feature = "validate")]
impl<'a> BlockContext<'a> {
    fn new(
        fun: &'a crate::Function,
        module: &'a crate::Module,
        info: &'a FunctionInfo,
        prev_infos: &'a [FunctionInfo],
    ) -> Self {
        Self {
            abilities: ControlFlowAbility::RETURN,
            info,
            expressions: &fun.expressions,
            types: &module.types,
            global_vars: &module.global_variables,
            functions: &module.functions,
            prev_infos,
            return_type: fun.result.as_ref().map(|fr| fr.ty),
        }
    }

    const fn with_abilities(&self, abilities: ControlFlowAbility) -> Self {
        BlockContext { abilities, ..*self }
    }

    fn get_expression(
        &self,
        handle: Handle<crate::Expression>,
    ) -> Result<&'a crate::Expression, FunctionError> {
        Ok(self.expressions.try_get(handle)?)
    }

    fn resolve_type_impl(
        &self,
        handle: Handle<crate::Expression>,
        valid_expressions: &BitSet,
    ) -> Result<&crate::TypeInner, WithSpan<ExpressionError>> {
        if handle.index() >= self.expressions.len() {
            Err(ExpressionError::DoesntExist.with_span())
        } else if !valid_expressions.contains(handle.index()) {
            Err(ExpressionError::NotInScope.with_span_handle(handle, self.expressions))
        } else {
            Ok(self.info[handle].ty.inner_with(self.types))
        }
    }

    fn resolve_type(
        &self,
        handle: Handle<crate::Expression>,
        valid_expressions: &BitSet,
    ) -> Result<&crate::TypeInner, WithSpan<FunctionError>> {
        self.resolve_type_impl(handle, valid_expressions)
            .map_err_inner(|error| FunctionError::Expression { handle, error }.with_span())
    }

    fn resolve_pointer_type(
        &self,
        handle: Handle<crate::Expression>,
    ) -> Result<&crate::TypeInner, FunctionError> {
        if handle.index() >= self.expressions.len() {
            Err(FunctionError::Expression {
                handle,
                error: ExpressionError::DoesntExist,
            })
        } else {
            Ok(self.info[handle].ty.inner_with(self.types))
        }
    }
}

impl super::Validator {
    #[cfg(feature = "validate")]
    fn validate_call(
        &mut self,
        function: Handle<crate::Function>,
        arguments: &[Handle<crate::Expression>],
        result: Option<Handle<crate::Expression>>,
        context: &BlockContext,
    ) -> Result<super::ShaderStages, WithSpan<CallError>> {
        let fun = context
            .functions
            .try_get(function)
            .map_err(CallError::BadHandle)
            .map_err(WithSpan::new)?;
        if fun.arguments.len() != arguments.len() {
            return Err(CallError::ArgumentCount {
                required: fun.arguments.len(),
                seen: arguments.len(),
            }
            .with_span());
        }
        for (index, (arg, &expr)) in fun.arguments.iter().zip(arguments).enumerate() {
            let ty = context
                .resolve_type_impl(expr, &self.valid_expression_set)
                .map_err_inner(|error| {
                    CallError::Argument { index, error }.with_span_handle(expr, context.expressions)
                })?;
            let arg_inner = &context.types[arg.ty].inner;
            if !ty.equivalent(arg_inner, context.types) {
                return Err(CallError::ArgumentType {
                    index,
                    required: arg.ty,
                    seen_expression: expr,
                }
                .with_span_handle(expr, context.expressions));
            }
        }

        if let Some(expr) = result {
            if self.valid_expression_set.insert(expr.index()) {
                self.valid_expression_list.push(expr);
            } else {
                return Err(CallError::ResultAlreadyInScope(expr)
                    .with_span_handle(expr, context.expressions));
            }
            match context.expressions[expr] {
                crate::Expression::CallResult(callee)
                    if fun.result.is_some() && callee == function => {}
                _ => {
                    return Err(CallError::ExpressionMismatch(result)
                        .with_span_handle(expr, context.expressions))
                }
            }
        } else if fun.result.is_some() {
            return Err(CallError::ExpressionMismatch(result).with_span());
        }

        let callee_info = &context.prev_infos[function.index()];
        Ok(callee_info.available_stages)
    }

    #[cfg(feature = "validate")]
    fn validate_atomic(
        &mut self,
        pointer: Handle<crate::Expression>,
        fun: &crate::AtomicFunction,
        value: Handle<crate::Expression>,
        result: Handle<crate::Expression>,
        context: &BlockContext,
    ) -> Result<(), WithSpan<FunctionError>> {
        let pointer_inner = context.resolve_type(pointer, &self.valid_expression_set)?;
        let (ptr_kind, ptr_width) = match *pointer_inner {
            crate::TypeInner::Pointer { base, .. } => match context.types[base].inner {
                crate::TypeInner::Atomic { kind, width } => (kind, width),
                ref other => {
                    log::error!("Atomic pointer to type {:?}", other);
                    return Err(AtomicError::InvalidPointer(pointer)
                        .with_span_handle(pointer, context.expressions)
                        .into_other());
                }
            },
            ref other => {
                log::error!("Atomic on type {:?}", other);
                return Err(AtomicError::InvalidPointer(pointer)
                    .with_span_handle(pointer, context.expressions)
                    .into_other());
            }
        };

        let value_inner = context.resolve_type(value, &self.valid_expression_set)?;
        match *value_inner {
            crate::TypeInner::Scalar { width, kind } if kind == ptr_kind && width == ptr_width => {}
            ref other => {
                log::error!("Atomic operand type {:?}", other);
                return Err(AtomicError::InvalidOperand(value)
                    .with_span_handle(value, context.expressions)
                    .into_other());
            }
        }

        if let crate::AtomicFunction::Exchange { compare: Some(cmp) } = *fun {
            if context.resolve_type(cmp, &self.valid_expression_set)? != value_inner {
                log::error!("Atomic exchange comparison has a different type from the value");
                return Err(AtomicError::InvalidOperand(cmp)
                    .with_span_handle(cmp, context.expressions)
                    .into_other());
            }
        }

        if self.valid_expression_set.insert(result.index()) {
            self.valid_expression_list.push(result);
        } else {
            return Err(AtomicError::ResultAlreadyInScope(result)
                .with_span_handle(result, context.expressions)
                .into_other());
        }
        match context.expressions[result] {
            //TODO: support atomic result with comparison
            crate::Expression::AtomicResult {
                kind,
                width,
                comparison: false,
            } if kind == ptr_kind && width == ptr_width => {}
            _ => {
                return Err(AtomicError::ResultTypeMismatch(result)
                    .with_span_handle(result, context.expressions)
                    .into_other())
            }
        }
        Ok(())
    }

    #[cfg(feature = "validate")]
    fn validate_block_impl(
        &mut self,
        statements: &crate::Block,
        context: &BlockContext,
    ) -> Result<BlockInfo, WithSpan<FunctionError>> {
        use crate::{Statement as S, TypeInner as Ti};
        let mut finished = false;
        let mut stages = super::ShaderStages::all();
        for (statement, &span) in statements.span_iter() {
            if finished {
                return Err(FunctionError::InstructionsAfterReturn
                    .with_span_static(span, "instructions after return"));
            }
            match *statement {
                S::Emit(ref range) => {
                    for handle in range.clone() {
                        if self.valid_expression_set.insert(handle.index()) {
                            self.valid_expression_list.push(handle);
                        } else {
                            return Err(FunctionError::ExpressionAlreadyInScope(handle)
                                .with_span_handle(handle, context.expressions));
                        }
                    }
                }
                S::Block(ref block) => {
                    let info = self.validate_block(block, context)?;
                    stages &= info.stages;
                    finished = info.finished;
                }
                S::If {
                    condition,
                    ref accept,
                    ref reject,
                } => {
                    match *context.resolve_type(condition, &self.valid_expression_set)? {
                        Ti::Scalar {
                            kind: crate::ScalarKind::Bool,
                            width: _,
                        } => {}
                        _ => {
                            return Err(FunctionError::InvalidIfType(condition)
                                .with_span_handle(condition, context.expressions))
                        }
                    }
                    stages &= self.validate_block(accept, context)?.stages;
                    stages &= self.validate_block(reject, context)?.stages;
                }
                S::Switch {
                    selector,
                    ref cases,
                } => {
                    match *context.resolve_type(selector, &self.valid_expression_set)? {
                        Ti::Scalar {
                            kind: crate::ScalarKind::Uint,
                            width: _,
                        } => {}
                        Ti::Scalar {
                            kind: crate::ScalarKind::Sint,
                            width: _,
                        } => {}
                        _ => {
                            return Err(FunctionError::InvalidSwitchType(selector)
                                .with_span_handle(selector, context.expressions))
                        }
                    }
                    self.select_cases.clear();
                    let mut default = false;
                    for case in cases {
                        match case.value {
                            crate::SwitchValue::Integer(value) => {
                                if !self.select_cases.insert(value) {
                                    return Err(FunctionError::ConflictingSwitchCase(value)
                                        .with_span_static(
                                            case.body
                                                .span_iter()
                                                .next()
                                                .map_or(Default::default(), |(_, s)| *s),
                                            "conflicting switch arm here",
                                        ));
                                }
                            }
                            crate::SwitchValue::Default => {
                                if default {
                                    return Err(FunctionError::MultipleDefaultCases
                                        .with_span_static(
                                            case.body
                                                .span_iter()
                                                .next()
                                                .map_or(Default::default(), |(_, s)| *s),
                                            "duplicated switch arm here",
                                        ));
                                }
                                default = true
                            }
                        }
                    }
                    if !default {
                        return Err(FunctionError::MissingDefaultCase
                            .with_span_static(span, "missing default case"));
                    }
                    if let Some(case) = cases.last() {
                        if case.fall_through {
                            return Err(FunctionError::LastCaseFallTrough.with_span_static(
                                case.body
                                    .span_iter()
                                    .next()
                                    .map_or(Default::default(), |(_, s)| *s),
                                "bad switch arm here",
                            ));
                        }
                    }
                    let pass_through_abilities = context.abilities
                        & (ControlFlowAbility::RETURN | ControlFlowAbility::CONTINUE);
                    let sub_context =
                        context.with_abilities(pass_through_abilities | ControlFlowAbility::BREAK);
                    for case in cases {
                        stages &= self.validate_block(&case.body, &sub_context)?.stages;
                    }
                }
                S::Loop {
                    ref body,
                    ref continuing,
                    break_if,
                } => {
                    // special handling for block scoping is needed here,
                    // because the continuing{} block inherits the scope
                    let base_expression_count = self.valid_expression_list.len();
                    let pass_through_abilities = context.abilities & ControlFlowAbility::RETURN;
                    stages &= self
                        .validate_block_impl(
                            body,
                            &context.with_abilities(
                                pass_through_abilities
                                    | ControlFlowAbility::BREAK
                                    | ControlFlowAbility::CONTINUE,
                            ),
                        )?
                        .stages;
                    stages &= self
                        .validate_block_impl(
                            continuing,
                            &context.with_abilities(ControlFlowAbility::empty()),
                        )?
                        .stages;

                    if let Some(condition) = break_if {
                        match *context.resolve_type(condition, &self.valid_expression_set)? {
                            Ti::Scalar {
                                kind: crate::ScalarKind::Bool,
                                width: _,
                            } => {}
                            _ => {
                                return Err(FunctionError::InvalidIfType(condition)
                                    .with_span_handle(condition, context.expressions))
                            }
                        }
                    }

                    for handle in self.valid_expression_list.drain(base_expression_count..) {
                        self.valid_expression_set.remove(handle.index());
                    }
                }
                S::Break => {
                    if !context.abilities.contains(ControlFlowAbility::BREAK) {
                        return Err(FunctionError::BreakOutsideOfLoopOrSwitch
                            .with_span_static(span, "invalid break"));
                    }
                    finished = true;
                }
                S::Continue => {
                    if !context.abilities.contains(ControlFlowAbility::CONTINUE) {
                        return Err(FunctionError::ContinueOutsideOfLoop
                            .with_span_static(span, "invalid continue"));
                    }
                    finished = true;
                }
                S::Return { value } => {
                    if !context.abilities.contains(ControlFlowAbility::RETURN) {
                        return Err(FunctionError::InvalidReturnSpot
                            .with_span_static(span, "invalid return"));
                    }
                    let value_ty = value
                        .map(|expr| context.resolve_type(expr, &self.valid_expression_set))
                        .transpose()?;
                    let expected_ty = context.return_type.map(|ty| &context.types[ty].inner);
                    // We can't return pointers, but it seems best not to embed that
                    // assumption here, so use `TypeInner::equivalent` for comparison.
                    let okay = match (value_ty, expected_ty) {
                        (None, None) => true,
                        (Some(value_inner), Some(expected_inner)) => {
                            value_inner.equivalent(expected_inner, context.types)
                        }
                        (_, _) => false,
                    };

                    if !okay {
                        log::error!(
                            "Returning {:?} where {:?} is expected",
                            value_ty,
                            expected_ty
                        );
                        if let Some(handle) = value {
                            return Err(FunctionError::InvalidReturnType(value)
                                .with_span_handle(handle, context.expressions));
                        } else {
                            return Err(FunctionError::InvalidReturnType(value)
                                .with_span_static(span, "invalid return"));
                        }
                    }
                    finished = true;
                }
                S::Kill => {
                    finished = true;
                }
                S::Barrier(_) => {
                    stages &= super::ShaderStages::COMPUTE;
                }
                S::Store { pointer, value } => {
                    let mut current = pointer;
                    loop {
                        let _ = context
                            .resolve_pointer_type(current)
                            .map_err(|e| e.with_span())?;
                        match context.expressions[current] {
                            crate::Expression::Access { base, .. }
                            | crate::Expression::AccessIndex { base, .. } => current = base,
                            crate::Expression::LocalVariable(_)
                            | crate::Expression::GlobalVariable(_)
                            | crate::Expression::FunctionArgument(_) => break,
                            _ => {
                                return Err(FunctionError::InvalidStorePointer(current)
                                    .with_span_handle(pointer, context.expressions))
                            }
                        }
                    }

                    let value_ty = context.resolve_type(value, &self.valid_expression_set)?;
                    match *value_ty {
                        Ti::Image { .. } | Ti::Sampler { .. } => {
                            return Err(FunctionError::InvalidStoreValue(value)
                                .with_span_handle(value, context.expressions));
                        }
                        _ => {}
                    }

                    let pointer_ty = context
                        .resolve_pointer_type(pointer)
                        .map_err(|e| e.with_span())?;

                    let good = match *pointer_ty {
                        Ti::Pointer { base, space: _ } => match context.types[base].inner {
                            Ti::Atomic { kind, width } => *value_ty == Ti::Scalar { kind, width },
                            ref other => value_ty == other,
                        },
                        Ti::ValuePointer {
                            size: Some(size),
                            kind,
                            width,
                            space: _,
                        } => *value_ty == Ti::Vector { size, kind, width },
                        Ti::ValuePointer {
                            size: None,
                            kind,
                            width,
                            space: _,
                        } => *value_ty == Ti::Scalar { kind, width },
                        _ => false,
                    };
                    if !good {
                        return Err(FunctionError::InvalidStoreTypes { pointer, value }
                            .with_span()
                            .with_handle(pointer, context.expressions)
                            .with_handle(value, context.expressions));
                    }

                    if let Some(space) = pointer_ty.pointer_space() {
                        if !space.access().contains(crate::StorageAccess::STORE) {
                            return Err(FunctionError::InvalidStorePointer(pointer)
                                .with_span_static(
                                    context.expressions.get_span(pointer),
                                    "writing to this location is not permitted",
                                ));
                        }
                    }
                }
                S::ImageStore {
                    image,
                    coordinate,
                    array_index,
                    value,
                } => {
                    //Note: this code uses a lot of `FunctionError::InvalidImageStore`,
                    // and could probably be refactored.
                    let var = match *context.get_expression(image).map_err(|e| e.with_span())? {
                        crate::Expression::GlobalVariable(var_handle) => {
                            &context.global_vars[var_handle]
                        }
                        // We're looking at a binding index situation, so punch through the index and look at the global behind it.
                        crate::Expression::Access { base, .. }
                        | crate::Expression::AccessIndex { base, .. } => {
                            match *context.get_expression(base).map_err(|e| e.with_span())? {
                                crate::Expression::GlobalVariable(var_handle) => {
                                    &context.global_vars[var_handle]
                                }
                                _ => {
                                    return Err(FunctionError::InvalidImageStore(
                                        ExpressionError::ExpectedGlobalVariable,
                                    )
                                    .with_span_handle(image, context.expressions))
                                }
                            }
                        }
                        _ => {
                            return Err(FunctionError::InvalidImageStore(
                                ExpressionError::ExpectedGlobalVariable,
                            )
                            .with_span_handle(image, context.expressions))
                        }
                    };

                    // Punch through a binding array to get the underlying type
                    let global_ty = match context.types[var.ty].inner {
                        Ti::BindingArray { base, .. } => &context.types[base].inner,
                        ref inner => inner,
                    };

                    let value_ty = match *global_ty {
                        Ti::Image {
                            class,
                            arrayed,
                            dim,
                        } => {
                            match context
                                .resolve_type(coordinate, &self.valid_expression_set)?
                                .image_storage_coordinates()
                            {
                                Some(coord_dim) if coord_dim == dim => {}
                                _ => {
                                    return Err(FunctionError::InvalidImageStore(
                                        ExpressionError::InvalidImageCoordinateType(
                                            dim, coordinate,
                                        ),
                                    )
                                    .with_span_handle(coordinate, context.expressions));
                                }
                            };
                            if arrayed != array_index.is_some() {
                                return Err(FunctionError::InvalidImageStore(
                                    ExpressionError::InvalidImageArrayIndex,
                                )
                                .with_span_handle(coordinate, context.expressions));
                            }
                            if let Some(expr) = array_index {
                                match *context.resolve_type(expr, &self.valid_expression_set)? {
                                    Ti::Scalar {
                                        kind: crate::ScalarKind::Sint,
                                        width: _,
                                    } => {}
                                    _ => {
                                        return Err(FunctionError::InvalidImageStore(
                                            ExpressionError::InvalidImageArrayIndexType(expr),
                                        )
                                        .with_span_handle(expr, context.expressions));
                                    }
                                }
                            }
                            match class {
                                crate::ImageClass::Storage { format, .. } => {
                                    crate::TypeInner::Vector {
                                        kind: format.into(),
                                        size: crate::VectorSize::Quad,
                                        width: 4,
                                    }
                                }
                                _ => {
                                    return Err(FunctionError::InvalidImageStore(
                                        ExpressionError::InvalidImageClass(class),
                                    )
                                    .with_span_handle(image, context.expressions));
                                }
                            }
                        }
                        _ => {
                            return Err(FunctionError::InvalidImageStore(
                                ExpressionError::ExpectedImageType(var.ty),
                            )
                            .with_span()
                            .with_handle(var.ty, context.types)
                            .with_handle(image, context.expressions))
                        }
                    };

                    if *context.resolve_type(value, &self.valid_expression_set)? != value_ty {
                        return Err(FunctionError::InvalidStoreValue(value)
                            .with_span_handle(value, context.expressions));
                    }
                }
                S::Call {
                    function,
                    ref arguments,
                    result,
                } => match self.validate_call(function, arguments, result, context) {
                    Ok(callee_stages) => stages &= callee_stages,
                    Err(error) => {
                        return Err(error.and_then(|error| {
                            FunctionError::InvalidCall { function, error }
                                .with_span_static(span, "invalid function call")
                        }))
                    }
                },
                S::Atomic {
                    pointer,
                    ref fun,
                    value,
                    result,
                } => {
                    self.validate_atomic(pointer, fun, value, result, context)?;
                }
            }
        }
        Ok(BlockInfo { stages, finished })
    }

    #[cfg(feature = "validate")]
    fn validate_block(
        &mut self,
        statements: &crate::Block,
        context: &BlockContext,
    ) -> Result<BlockInfo, WithSpan<FunctionError>> {
        let base_expression_count = self.valid_expression_list.len();
        let info = self.validate_block_impl(statements, context)?;
        for handle in self.valid_expression_list.drain(base_expression_count..) {
            self.valid_expression_set.remove(handle.index());
        }
        Ok(info)
    }

    #[cfg(feature = "validate")]
    fn validate_local_var(
        &self,
        var: &crate::LocalVariable,
        types: &UniqueArena<crate::Type>,
        constants: &Arena<crate::Constant>,
    ) -> Result<(), LocalVariableError> {
        log::debug!("var {:?}", var);
        let type_info = self
            .types
            .get(var.ty.index())
            .ok_or(LocalVariableError::InvalidType(var.ty))?;
        if !type_info
            .flags
            .contains(super::TypeFlags::DATA | super::TypeFlags::SIZED)
        {
            return Err(LocalVariableError::InvalidType(var.ty));
        }

        if let Some(const_handle) = var.init {
            match constants[const_handle].inner {
                crate::ConstantInner::Scalar { width, ref value } => {
                    let ty_inner = crate::TypeInner::Scalar {
                        width,
                        kind: value.scalar_kind(),
                    };
                    if types[var.ty].inner != ty_inner {
                        return Err(LocalVariableError::InitializerType);
                    }
                }
                crate::ConstantInner::Composite { ty, components: _ } => {
                    if ty != var.ty {
                        return Err(LocalVariableError::InitializerType);
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn validate_function(
        &mut self,
        fun: &crate::Function,
        module: &crate::Module,
        mod_info: &ModuleInfo,
    ) -> Result<FunctionInfo, WithSpan<FunctionError>> {
        #[cfg_attr(not(feature = "validate"), allow(unused_mut))]
        let mut info = mod_info.process_function(fun, module, self.flags, self.capabilities)?;

        #[cfg(feature = "validate")]
        for (var_handle, var) in fun.local_variables.iter() {
            self.validate_local_var(var, &module.types, &module.constants)
                .map_err(|error| {
                    FunctionError::LocalVariable {
                        handle: var_handle,
                        name: var.name.clone().unwrap_or_default(),
                        error,
                    }
                    .with_span_handle(var.ty, &module.types)
                    .with_handle(var_handle, &fun.local_variables)
                })?;
        }

        #[cfg(feature = "validate")]
        for (index, argument) in fun.arguments.iter().enumerate() {
            let ty = module.types.get_handle(argument.ty).map_err(|err| {
                FunctionError::from(err).with_span_handle(argument.ty, &module.types)
            })?;
            match ty.inner.pointer_space() {
                Some(
                    crate::AddressSpace::Private
                    | crate::AddressSpace::Function
                    | crate::AddressSpace::WorkGroup,
                )
                | None => {}
                Some(other) => {
                    return Err(FunctionError::InvalidArgumentPointerSpace {
                        index,
                        name: argument.name.clone().unwrap_or_default(),
                        space: other,
                    }
                    .with_span_handle(argument.ty, &module.types))
                }
            }
            // Check for the least informative error last.
            if !self.types[argument.ty.index()]
                .flags
                .contains(super::TypeFlags::ARGUMENT)
            {
                return Err(FunctionError::InvalidArgumentType {
                    index,
                    name: argument.name.clone().unwrap_or_default(),
                }
                .with_span_handle(argument.ty, &module.types));
            }
        }

        #[cfg(feature = "validate")]
        if let Some(ref result) = fun.result {
            if !self.types[result.ty.index()]
                .flags
                .contains(super::TypeFlags::CONSTRUCTIBLE)
            {
                return Err(FunctionError::NonConstructibleReturnType
                    .with_span_handle(result.ty, &module.types));
            }
        }

        self.valid_expression_set.clear();
        self.valid_expression_list.clear();
        for (handle, expr) in fun.expressions.iter() {
            if expr.needs_pre_emit() {
                self.valid_expression_set.insert(handle.index());
            }
            #[cfg(feature = "validate")]
            if self.flags.contains(super::ValidationFlags::EXPRESSIONS) {
                match self.validate_expression(
                    handle,
                    expr,
                    fun,
                    module,
                    &info,
                    &mod_info.functions,
                ) {
                    Ok(stages) => info.available_stages &= stages,
                    Err(error) => {
                        return Err(FunctionError::Expression { handle, error }
                            .with_span_handle(handle, &fun.expressions))
                    }
                }
            }
        }

        #[cfg(feature = "validate")]
        if self.flags.contains(super::ValidationFlags::BLOCKS) {
            let stages = self
                .validate_block(
                    &fun.body,
                    &BlockContext::new(fun, module, &info, &mod_info.functions),
                )?
                .stages;
            info.available_stages &= stages;
        }
        Ok(info)
    }
}
