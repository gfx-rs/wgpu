//! Implementation of `Validator::validate_module_handles`.

use crate::{
    arena::{BadHandle, BadRangeError},
    Handle,
};

#[cfg(feature = "validate")]
use crate::{Arena, UniqueArena};

#[cfg(feature = "validate")]
use super::ValidationError;

#[cfg(feature = "validate")]
use std::{convert::TryInto, hash::Hash, num::NonZeroU32};

#[cfg(feature = "validate")]
impl super::Validator {
    /// Validates that all handles within `module` are:
    ///
    /// * Valid, in the sense that they contain indices within each arena structure inside the
    /// [`crate::Module`] type.
    /// * No arena contents contain any items that have forward dependencies; that is, the value
    ///     associated with a handle only may contain references to handles in the same arena that
    ///     were constructed before it.
    ///
    /// By validating the above conditions, we free up subsequent logic to assume that handle
    /// accesses are infallible.
    ///
    /// # Errors
    ///
    /// Errors returned by this method are intentionally sparse, for simplicity of implementation.
    /// It is expected that only buggy frontends or fuzzers should ever emit IR that fails this
    /// validation pass.
    pub(super) fn validate_module_handles(module: &crate::Module) -> Result<(), ValidationError> {
        let &crate::Module {
            ref constants,
            ref entry_points,
            ref functions,
            ref global_variables,
            ref types,
            ref special_types,
            ref const_expressions,
        } = module;

        // NOTE: Types being first is important. All other forms of validation depend on this.
        for (this_handle, ty) in types.iter() {
            match ty.inner {
                crate::TypeInner::Scalar { .. }
                | crate::TypeInner::Vector { .. }
                | crate::TypeInner::Matrix { .. }
                | crate::TypeInner::ValuePointer { .. }
                | crate::TypeInner::Atomic { .. }
                | crate::TypeInner::Image { .. }
                | crate::TypeInner::Sampler { .. }
                | crate::TypeInner::AccelerationStructure
                | crate::TypeInner::RayQuery => (),
                crate::TypeInner::Pointer { base, space: _ } => {
                    this_handle.check_dep(base)?;
                }
                crate::TypeInner::Array { base, .. }
                | crate::TypeInner::BindingArray { base, .. } => {
                    this_handle.check_dep(base)?;
                }
                crate::TypeInner::Struct {
                    ref members,
                    span: _,
                } => {
                    this_handle.check_dep_iter(members.iter().map(|m| m.ty))?;
                }
            }
        }

        for handle_and_expr in const_expressions.iter() {
            Self::validate_const_expression_handles(handle_and_expr, constants, types)?;
        }

        let validate_type = |handle| Self::validate_type_handle(handle, types);
        let validate_const_expr =
            |handle| Self::validate_expression_handle(handle, const_expressions);

        for (_handle, constant) in constants.iter() {
            let &crate::Constant {
                name: _,
                r#override: _,
                ty,
                init,
            } = constant;
            validate_type(ty)?;
            validate_const_expr(init)?;
        }

        for (_handle, global_variable) in global_variables.iter() {
            let &crate::GlobalVariable {
                name: _,
                space: _,
                binding: _,
                ty,
                init,
            } = global_variable;
            validate_type(ty)?;
            if let Some(init_expr) = init {
                validate_const_expr(init_expr)?;
            }
        }

        let validate_function = |function_handle, function: &_| -> Result<_, InvalidHandleError> {
            let &crate::Function {
                name: _,
                ref arguments,
                ref result,
                ref local_variables,
                ref expressions,
                ref named_expressions,
                ref body,
            } = function;

            for arg in arguments.iter() {
                let &crate::FunctionArgument {
                    name: _,
                    ty,
                    binding: _,
                } = arg;
                validate_type(ty)?;
            }

            if let &Some(crate::FunctionResult { ty, binding: _ }) = result {
                validate_type(ty)?;
            }

            for (_handle, local_variable) in local_variables.iter() {
                let &crate::LocalVariable { name: _, ty, init } = local_variable;
                validate_type(ty)?;
                if let Some(init) = init {
                    Self::validate_expression_handle(init, expressions)?;
                }
            }

            for handle in named_expressions.keys().copied() {
                Self::validate_expression_handle(handle, expressions)?;
            }

            for handle_and_expr in expressions.iter() {
                Self::validate_expression_handles(
                    handle_and_expr,
                    constants,
                    const_expressions,
                    types,
                    local_variables,
                    global_variables,
                    functions,
                    function_handle,
                )?;
            }

            Self::validate_block_handles(body, expressions, functions)?;

            Ok(())
        };

        for entry_point in entry_points.iter() {
            validate_function(None, &entry_point.function)?;
        }

        for (function_handle, function) in functions.iter() {
            validate_function(Some(function_handle), function)?;
        }

        if let Some(ty) = special_types.ray_desc {
            validate_type(ty)?;
        }
        if let Some(ty) = special_types.ray_intersection {
            validate_type(ty)?;
        }

        Ok(())
    }

    fn validate_type_handle(
        handle: Handle<crate::Type>,
        types: &UniqueArena<crate::Type>,
    ) -> Result<(), InvalidHandleError> {
        handle.check_valid_for_uniq(types).map(|_| ())
    }

    fn validate_constant_handle(
        handle: Handle<crate::Constant>,
        constants: &Arena<crate::Constant>,
    ) -> Result<(), InvalidHandleError> {
        handle.check_valid_for(constants).map(|_| ())
    }

    fn validate_expression_handle(
        handle: Handle<crate::Expression>,
        expressions: &Arena<crate::Expression>,
    ) -> Result<(), InvalidHandleError> {
        handle.check_valid_for(expressions).map(|_| ())
    }

    fn validate_function_handle(
        handle: Handle<crate::Function>,
        functions: &Arena<crate::Function>,
    ) -> Result<(), InvalidHandleError> {
        handle.check_valid_for(functions).map(|_| ())
    }

    fn validate_const_expression_handles(
        (handle, expression): (Handle<crate::Expression>, &crate::Expression),
        constants: &Arena<crate::Constant>,
        types: &UniqueArena<crate::Type>,
    ) -> Result<(), InvalidHandleError> {
        let validate_constant = |handle| Self::validate_constant_handle(handle, constants);
        let validate_type = |handle| Self::validate_type_handle(handle, types);

        match *expression {
            crate::Expression::Literal(_) => {}
            crate::Expression::Constant(constant) => {
                validate_constant(constant)?;
                handle.check_dep(constants[constant].init)?;
            }
            crate::Expression::ZeroValue(ty) => {
                validate_type(ty)?;
            }
            crate::Expression::Compose { ty, ref components } => {
                validate_type(ty)?;
                handle.check_dep_iter(components.iter().copied())?;
            }
            _ => {}
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_expression_handles(
        (handle, expression): (Handle<crate::Expression>, &crate::Expression),
        constants: &Arena<crate::Constant>,
        const_expressions: &Arena<crate::Expression>,
        types: &UniqueArena<crate::Type>,
        local_variables: &Arena<crate::LocalVariable>,
        global_variables: &Arena<crate::GlobalVariable>,
        functions: &Arena<crate::Function>,
        // The handle of the current function or `None` if it's an entry point
        current_function: Option<Handle<crate::Function>>,
    ) -> Result<(), InvalidHandleError> {
        let validate_constant = |handle| Self::validate_constant_handle(handle, constants);
        let validate_const_expr =
            |handle| Self::validate_expression_handle(handle, const_expressions);
        let validate_type = |handle| Self::validate_type_handle(handle, types);

        match *expression {
            crate::Expression::Access { base, index } => {
                handle.check_dep(base)?.check_dep(index)?;
            }
            crate::Expression::AccessIndex { base, .. } => {
                handle.check_dep(base)?;
            }
            crate::Expression::Splat { value, .. } => {
                handle.check_dep(value)?;
            }
            crate::Expression::Swizzle { vector, .. } => {
                handle.check_dep(vector)?;
            }
            crate::Expression::Literal(_) => {}
            crate::Expression::Constant(constant) => {
                validate_constant(constant)?;
            }
            crate::Expression::ZeroValue(ty) => {
                validate_type(ty)?;
            }
            crate::Expression::Compose { ty, ref components } => {
                validate_type(ty)?;
                handle.check_dep_iter(components.iter().copied())?;
            }
            crate::Expression::FunctionArgument(_arg_idx) => (),
            crate::Expression::GlobalVariable(global_variable) => {
                global_variable.check_valid_for(global_variables)?;
            }
            crate::Expression::LocalVariable(local_variable) => {
                local_variable.check_valid_for(local_variables)?;
            }
            crate::Expression::Load { pointer } => {
                handle.check_dep(pointer)?;
            }
            crate::Expression::ImageSample {
                image,
                sampler,
                gather: _,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
            } => {
                if let Some(offset) = offset {
                    validate_const_expr(offset)?;
                }

                handle
                    .check_dep(image)?
                    .check_dep(sampler)?
                    .check_dep(coordinate)?
                    .check_dep_opt(array_index)?;

                match level {
                    crate::SampleLevel::Auto | crate::SampleLevel::Zero => (),
                    crate::SampleLevel::Exact(expr) => {
                        handle.check_dep(expr)?;
                    }
                    crate::SampleLevel::Bias(expr) => {
                        handle.check_dep(expr)?;
                    }
                    crate::SampleLevel::Gradient { x, y } => {
                        handle.check_dep(x)?.check_dep(y)?;
                    }
                };

                handle.check_dep_opt(depth_ref)?;
            }
            crate::Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => {
                handle
                    .check_dep(image)?
                    .check_dep(coordinate)?
                    .check_dep_opt(array_index)?
                    .check_dep_opt(sample)?
                    .check_dep_opt(level)?;
            }
            crate::Expression::ImageQuery { image, query } => {
                handle.check_dep(image)?;
                match query {
                    crate::ImageQuery::Size { level } => {
                        handle.check_dep_opt(level)?;
                    }
                    crate::ImageQuery::NumLevels
                    | crate::ImageQuery::NumLayers
                    | crate::ImageQuery::NumSamples => (),
                };
            }
            crate::Expression::Unary {
                op: _,
                expr: operand,
            } => {
                handle.check_dep(operand)?;
            }
            crate::Expression::Binary { op: _, left, right } => {
                handle.check_dep(left)?.check_dep(right)?;
            }
            crate::Expression::Select {
                condition,
                accept,
                reject,
            } => {
                handle
                    .check_dep(condition)?
                    .check_dep(accept)?
                    .check_dep(reject)?;
            }
            crate::Expression::Derivative { expr: argument, .. } => {
                handle.check_dep(argument)?;
            }
            crate::Expression::Relational { fun: _, argument } => {
                handle.check_dep(argument)?;
            }
            crate::Expression::Math {
                fun: _,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                handle
                    .check_dep(arg)?
                    .check_dep_opt(arg1)?
                    .check_dep_opt(arg2)?
                    .check_dep_opt(arg3)?;
            }
            crate::Expression::As {
                expr: input,
                kind: _,
                convert: _,
            } => {
                handle.check_dep(input)?;
            }
            crate::Expression::CallResult(function) => {
                Self::validate_function_handle(function, functions)?;
                if let Some(handle) = current_function {
                    handle.check_dep(function)?;
                }
            }
            crate::Expression::AtomicResult { .. }
            | crate::Expression::RayQueryProceedResult
            | crate::Expression::WorkGroupUniformLoadResult { .. } => (),
            crate::Expression::ArrayLength(array) => {
                handle.check_dep(array)?;
            }
            crate::Expression::RayQueryGetIntersection {
                query,
                committed: _,
            } => {
                handle.check_dep(query)?;
            }
        }
        Ok(())
    }

    fn validate_block_handles(
        block: &crate::Block,
        expressions: &Arena<crate::Expression>,
        functions: &Arena<crate::Function>,
    ) -> Result<(), InvalidHandleError> {
        let validate_block = |block| Self::validate_block_handles(block, expressions, functions);
        let validate_expr = |handle| Self::validate_expression_handle(handle, expressions);
        let validate_expr_opt = |handle_opt| {
            if let Some(handle) = handle_opt {
                validate_expr(handle)?;
            }
            Ok(())
        };

        block.iter().try_for_each(|stmt| match *stmt {
            crate::Statement::Emit(ref expr_range) => {
                expr_range.check_valid_for(expressions)?;
                Ok(())
            }
            crate::Statement::Block(ref block) => {
                validate_block(block)?;
                Ok(())
            }
            crate::Statement::If {
                condition,
                ref accept,
                ref reject,
            } => {
                validate_expr(condition)?;
                validate_block(accept)?;
                validate_block(reject)?;
                Ok(())
            }
            crate::Statement::Switch {
                selector,
                ref cases,
            } => {
                validate_expr(selector)?;
                for &crate::SwitchCase {
                    value: _,
                    ref body,
                    fall_through: _,
                } in cases
                {
                    validate_block(body)?;
                }
                Ok(())
            }
            crate::Statement::Loop {
                ref body,
                ref continuing,
                break_if,
            } => {
                validate_block(body)?;
                validate_block(continuing)?;
                validate_expr_opt(break_if)?;
                Ok(())
            }
            crate::Statement::Return { value } => validate_expr_opt(value),
            crate::Statement::Store { pointer, value } => {
                validate_expr(pointer)?;
                validate_expr(value)?;
                Ok(())
            }
            crate::Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                validate_expr(image)?;
                validate_expr(coordinate)?;
                validate_expr_opt(array_index)?;
                validate_expr(value)?;
                Ok(())
            }
            crate::Statement::Atomic {
                pointer,
                fun,
                value,
                result,
            } => {
                validate_expr(pointer)?;
                match fun {
                    crate::AtomicFunction::Add
                    | crate::AtomicFunction::Subtract
                    | crate::AtomicFunction::And
                    | crate::AtomicFunction::ExclusiveOr
                    | crate::AtomicFunction::InclusiveOr
                    | crate::AtomicFunction::Min
                    | crate::AtomicFunction::Max => (),
                    crate::AtomicFunction::Exchange { compare } => validate_expr_opt(compare)?,
                };
                validate_expr(value)?;
                validate_expr(result)?;
                Ok(())
            }
            crate::Statement::WorkGroupUniformLoad { pointer, result } => {
                validate_expr(pointer)?;
                validate_expr(result)?;
                Ok(())
            }
            crate::Statement::Call {
                function,
                ref arguments,
                result,
            } => {
                Self::validate_function_handle(function, functions)?;
                for arg in arguments.iter().copied() {
                    validate_expr(arg)?;
                }
                validate_expr_opt(result)?;
                Ok(())
            }
            crate::Statement::RayQuery { query, ref fun } => {
                validate_expr(query)?;
                match *fun {
                    crate::RayQueryFunction::Initialize {
                        acceleration_structure,
                        descriptor,
                    } => {
                        validate_expr(acceleration_structure)?;
                        validate_expr(descriptor)?;
                    }
                    crate::RayQueryFunction::Proceed { result } => {
                        validate_expr(result)?;
                    }
                    crate::RayQueryFunction::Terminate => {}
                }
                Ok(())
            }
            crate::Statement::Break
            | crate::Statement::Continue
            | crate::Statement::Kill
            | crate::Statement::Barrier(_) => Ok(()),
        })
    }
}

#[cfg(feature = "validate")]
impl From<BadHandle> for ValidationError {
    fn from(source: BadHandle) -> Self {
        Self::InvalidHandle(source.into())
    }
}

#[cfg(feature = "validate")]
impl From<FwdDepError> for ValidationError {
    fn from(source: FwdDepError) -> Self {
        Self::InvalidHandle(source.into())
    }
}

#[cfg(feature = "validate")]
impl From<BadRangeError> for ValidationError {
    fn from(source: BadRangeError) -> Self {
        Self::InvalidHandle(source.into())
    }
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum InvalidHandleError {
    #[error(transparent)]
    BadHandle(#[from] BadHandle),
    #[error(transparent)]
    ForwardDependency(#[from] FwdDepError),
    #[error(transparent)]
    BadRange(#[from] BadRangeError),
}

#[derive(Clone, Debug, thiserror::Error)]
#[error(
    "{subject:?} of kind {subject_kind:?} depends on {depends_on:?} of kind {depends_on_kind}, \
    which has not been processed yet"
)]
pub struct FwdDepError {
    // This error is used for many `Handle` types, but there's no point in making this generic, so
    // we just flatten them all to `Handle<()>` here.
    subject: Handle<()>,
    subject_kind: &'static str,
    depends_on: Handle<()>,
    depends_on_kind: &'static str,
}

#[cfg(feature = "validate")]
impl<T> Handle<T> {
    /// Check that `self` is valid within `arena` using [`Arena::check_contains_handle`].
    pub(self) fn check_valid_for(self, arena: &Arena<T>) -> Result<(), InvalidHandleError> {
        arena.check_contains_handle(self)?;
        Ok(())
    }

    /// Check that `self` is valid within `arena` using [`UniqueArena::check_contains_handle`].
    pub(self) fn check_valid_for_uniq(
        self,
        arena: &UniqueArena<T>,
    ) -> Result<(), InvalidHandleError>
    where
        T: Eq + Hash,
    {
        arena.check_contains_handle(self)?;
        Ok(())
    }

    /// Check that `depends_on` was constructed before `self` by comparing handle indices.
    ///
    /// If `self` is a valid handle (i.e., it has been validated using [`Self::check_valid_for`])
    /// and this function returns [`Ok`], then it may be assumed that `depends_on` is also valid.
    /// In [`naga`](crate)'s current arena-based implementation, this is useful for validating
    /// recursive definitions of arena-based values in linear time.
    ///
    /// # Errors
    ///
    /// If `depends_on`'s handle is from the same [`Arena`] as `self'`s, but not constructed earlier
    /// than `self`'s, this function returns an error.
    pub(self) fn check_dep(self, depends_on: Self) -> Result<Self, FwdDepError> {
        if depends_on < self {
            Ok(self)
        } else {
            let erase_handle_type = |handle: Handle<_>| {
                Handle::new(NonZeroU32::new((handle.index() + 1).try_into().unwrap()).unwrap())
            };
            Err(FwdDepError {
                subject: erase_handle_type(self),
                subject_kind: std::any::type_name::<T>(),
                depends_on: erase_handle_type(depends_on),
                depends_on_kind: std::any::type_name::<T>(),
            })
        }
    }

    /// Like [`Self::check_dep`], except for [`Option`]al handle values.
    pub(self) fn check_dep_opt(self, depends_on: Option<Self>) -> Result<Self, FwdDepError> {
        self.check_dep_iter(depends_on.into_iter())
    }

    /// Like [`Self::check_dep`], except for [`Iterator`]s over handle values.
    pub(self) fn check_dep_iter(
        self,
        depends_on: impl Iterator<Item = Self>,
    ) -> Result<Self, FwdDepError> {
        for handle in depends_on {
            self.check_dep(handle)?;
        }
        Ok(self)
    }
}

#[cfg(feature = "validate")]
impl<T> crate::arena::Range<T> {
    pub(self) fn check_valid_for(&self, arena: &Arena<T>) -> Result<(), BadRangeError> {
        arena.check_contains_range(self)
    }
}

#[test]
#[cfg(feature = "validate")]
fn constant_deps() {
    use crate::{Constant, Expression, Literal, Span, Type, TypeInner};

    let nowhere = Span::default();

    let mut types = UniqueArena::new();
    let mut const_exprs = Arena::new();
    let mut fun_exprs = Arena::new();
    let mut constants = Arena::new();

    let i32_handle = types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar {
                kind: crate::ScalarKind::Sint,
                width: 4,
            },
        },
        nowhere,
    );

    // Construct a self-referential constant by misusing a handle to
    // fun_exprs as a constant initializer.
    let fun_expr = fun_exprs.append(Expression::Literal(Literal::I32(42)), nowhere);
    let self_referential_const = constants.append(
        Constant {
            name: None,
            r#override: crate::Override::None,
            ty: i32_handle,
            init: fun_expr,
        },
        nowhere,
    );
    let _self_referential_expr =
        const_exprs.append(Expression::Constant(self_referential_const), nowhere);

    for handle_and_expr in const_exprs.iter() {
        assert!(super::Validator::validate_const_expression_handles(
            handle_and_expr,
            &constants,
            &types,
        )
        .is_err());
    }
}
