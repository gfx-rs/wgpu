//! [`Module`] helpers for "upgrading" atomics in the SPIR-V (and eventually GLSL) frontends.
use std::sync::{atomic::AtomicUsize, Arc};

use crate::{
    Constant, Expression, Function, GlobalVariable, Handle, LocalVariable, Module, Override,
    StructMember, Type, TypeInner,
};

#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error("bad handle: {0}")]
    MissingHandle(crate::arena::BadHandle),
    #[error("no function context")]
    NoFunction,
}

impl From<Error> for crate::front::spv::Error {
    fn from(source: Error) -> Self {
        crate::front::spv::Error::AtomicUpgradeError(source)
    }
}

impl From<crate::arena::BadHandle> for Error {
    fn from(value: crate::arena::BadHandle) -> Self {
        Error::MissingHandle(value)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum AtomicOpInst {
    AtomicIIncrement,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AtomicOp {
    pub instruction: AtomicOpInst,
    /// Handle to the pointer's type in the module
    pub pointer_type_handle: Handle<Type>,
    /// Handle to the pointer expression in the module/function
    pub pointer_handle: Handle<Expression>,
}

#[derive(Default)]
struct Padding {
    padding: Arc<AtomicUsize>,
    current: usize,
}

impl std::fmt::Display for Padding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for _ in 0..self.current {
            f.write_str("  ")?;
        }
        Ok(())
    }
}

impl Drop for Padding {
    fn drop(&mut self) {
        let _ = self
            .padding
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Padding {
    fn debug(&self, msg: impl std::fmt::Display, t: impl std::fmt::Debug) {
        format!("{msg} {t:#?}")
            .split('\n')
            .for_each(|ln| log::trace!("{self}{ln}"));
    }

    fn inc_padding(&self) -> Padding {
        let current = self
            .padding
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Padding {
            padding: self.padding.clone(),
            current,
        }
    }
}

struct UpgradeState<'a> {
    padding: Padding,
    module: &'a mut Module,
}

impl<'a> UpgradeState<'a> {
    fn inc_padding(&self) -> Padding {
        self.padding.inc_padding()
    }

    /// Upgrade the type, recursing until we reach the leaves.
    /// At the leaves, replace scalars with atomic scalars.
    fn upgrade_type(&mut self, type_handle: Handle<Type>) -> Result<Handle<Type>, Error> {
        let padding = self.inc_padding();
        padding.debug("upgrading type: ", type_handle);
        let type_ = self
            .module
            .types
            .get_handle(type_handle)
            .map_err(Error::MissingHandle)?
            .clone();
        padding.debug("type: ", &type_);

        let new_inner = match type_.inner {
            TypeInner::Scalar(scalar) => {
                log::trace!("{padding}hit the scalar leaf, replacing with an atomic");
                TypeInner::Atomic(scalar)
            }
            TypeInner::Pointer { base, space } => TypeInner::Pointer {
                base: self.upgrade_type(base)?,
                space,
            },
            TypeInner::Array { base, size, stride } => TypeInner::Array {
                base: self.upgrade_type(base)?,
                size,
                stride,
            },
            TypeInner::Struct { members, span } => TypeInner::Struct {
                members: {
                    let mut new_members = vec![];
                    for member in members.iter().cloned() {
                        let StructMember {
                            name,
                            ty,
                            binding,
                            offset,
                        } = member;
                        new_members.push(StructMember {
                            name,
                            ty: self.upgrade_type(ty)?,
                            binding,
                            offset,
                        });
                    }
                    new_members
                },
                span,
            },
            TypeInner::BindingArray { base, size } => TypeInner::BindingArray {
                base: self.upgrade_type(base)?,
                size,
            },
            n => n,
        };

        let new_type_handle = self.module.types.insert(
            Type {
                name: type_.name,
                inner: new_inner,
            },
            self.module.types.get_span(type_handle),
        );
        padding.debug("new_type: ", new_type_handle);
        Ok(new_type_handle)
    }

    fn upgrade_global_variable(
        &mut self,
        handle: Handle<GlobalVariable>,
    ) -> Result<Handle<GlobalVariable>, Error> {
        let padding = self.inc_padding();
        padding.debug("upgrading global variable: ", handle);

        let var = self.module.global_variables.try_get(handle)?.clone();
        padding.debug("global variable:", &var);

        let new_var = GlobalVariable {
            name: var.name.clone(),
            space: var.space,
            binding: var.binding,
            ty: self.upgrade_type(var.ty)?,
            init: self.upgrade_opt_expression(None, var.init)?,
        };
        if new_var != var {
            padding.debug("new global variable: ", &new_var);
            let span = self.module.global_variables.get_span(handle);
            let new_handle = self.module.global_variables.append(new_var, span);
            padding.debug("new global variable handle: ", new_handle);
            Ok(new_handle)
        } else {
            Ok(handle)
        }
    }

    fn upgrade_local_variable(
        &mut self,
        fn_handle: Handle<Function>,
        handle: Handle<LocalVariable>,
    ) -> Result<Handle<LocalVariable>, Error> {
        let padding = self.inc_padding();
        padding.debug("upgrading local variable: ", handle);

        let (var, span) = {
            let f = self.module.functions.try_get(fn_handle)?;
            let var = f.local_variables.try_get(handle)?.clone();
            let span = f.local_variables.get_span(handle);
            (var, span)
        };
        padding.debug("local variable:", &var);

        let new_var = LocalVariable {
            name: var.name.clone(),
            ty: self.upgrade_type(var.ty)?,
            init: self.upgrade_opt_expression(Some(fn_handle), var.init)?,
        };
        if new_var != var {
            padding.debug("new local variable: ", &new_var);
            let f = self.module.functions.get_mut(fn_handle);
            let new_handle = f.local_variables.append(new_var, span);
            padding.debug("new local variable handle: ", new_handle);
            Ok(new_handle)
        } else {
            Ok(handle)
        }
    }

    fn upgrade_opt_expression(
        &mut self,
        maybe_fn_handle: Option<Handle<Function>>,
        maybe_handle: Option<Handle<Expression>>,
    ) -> Result<Option<Handle<Expression>>, Error> {
        Ok(if let Some(h) = maybe_handle {
            Some(self.upgrade_expression(maybe_fn_handle, h)?)
        } else {
            None
        })
    }

    fn upgrade_constant(&mut self, handle: Handle<Constant>) -> Result<Handle<Constant>, Error> {
        let padding = self.inc_padding();
        padding.debug("upgrading const: ", handle);

        let constant = self.module.constants.try_get(handle)?.clone();
        padding.debug("constant: ", &constant);

        let new_constant = Constant {
            name: constant.name.clone(),
            ty: self.upgrade_type(constant.ty)?,
            init: self.upgrade_expression(None, constant.init)?,
        };

        if constant != new_constant {
            padding.debug("inserting new constant: ", &new_constant);
            let new_handle = self
                .module
                .constants
                .append(new_constant, self.module.constants.get_span(handle));
            padding.debug("new constant handle: ", &new_handle);
            Ok(new_handle)
        } else {
            Ok(handle)
        }
    }

    fn upgrade_override(&mut self, handle: Handle<Override>) -> Result<Handle<Override>, Error> {
        let padding = self.inc_padding();
        padding.debug("upgrading override: ", handle);

        let o = self.module.overrides.try_get(handle)?.clone();
        padding.debug("override: ", &o);

        let new_o = Override {
            name: o.name.clone(),
            id: o.id,
            ty: self.upgrade_type(o.ty)?,
            init: self.upgrade_opt_expression(None, o.init)?,
        };

        if o != new_o {
            padding.debug("inserting new override: ", &new_o);
            let new_handle = self
                .module
                .overrides
                .append(new_o, self.module.overrides.get_span(handle));
            padding.debug("new override handle: ", &new_handle);
            Ok(new_handle)
        } else {
            Ok(handle)
        }
    }

    // Upgrade the expression, recursing int we reach...
    fn upgrade_expression(
        &mut self,
        maybe_fn_handle: Option<Handle<Function>>,
        handle: Handle<Expression>,
    ) -> Result<Handle<Expression>, Error> {
        let padding = self.inc_padding();
        padding.debug("upgrading expr: ", handle);
        let expr = if let Some(fh) = maybe_fn_handle {
            let function = self.module.functions.try_get(fh)?;
            function.expressions.try_get(handle)?.clone()
        } else {
            self.module.global_expressions.try_get(handle)?.clone()
        };

        padding.debug("expr: ", &expr);
        let new_expr = match expr.clone() {
            l @ Expression::Literal(_) => l,
            Expression::Constant(h) => Expression::Constant(self.upgrade_constant(h)?),
            Expression::Override(h) => Expression::Override(self.upgrade_override(h)?),
            Expression::ZeroValue(ty) => Expression::ZeroValue(self.upgrade_type(ty)?),
            Expression::Compose { ty, components } => Expression::Compose {
                ty: self.upgrade_type(ty)?,
                components: {
                    let mut new_components = vec![];
                    for component in components.into_iter() {
                        new_components.push(self.upgrade_expression(maybe_fn_handle, component)?);
                    }
                    new_components
                },
            },
            Expression::Access { base, index } => Expression::Access {
                base: self.upgrade_expression(maybe_fn_handle, base)?,
                index: self.upgrade_expression(maybe_fn_handle, index)?,
            },
            Expression::AccessIndex { base, index } => Expression::AccessIndex {
                base: self.upgrade_expression(maybe_fn_handle, base)?,
                index,
            },
            Expression::Splat { size, value } => Expression::Splat {
                size,
                value: self.upgrade_expression(maybe_fn_handle, value)?,
            },
            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => Expression::Swizzle {
                size,
                vector: self.upgrade_expression(maybe_fn_handle, vector)?,
                pattern,
            },
            f @ Expression::FunctionArgument(_) => f,
            Expression::GlobalVariable(var) => {
                Expression::GlobalVariable(self.upgrade_global_variable(var)?)
            }
            Expression::LocalVariable(var) => Expression::LocalVariable(
                self.upgrade_local_variable(maybe_fn_handle.ok_or(Error::NoFunction)?, var)?,
            ),
            Expression::Load { pointer } => Expression::Load {
                pointer: self.upgrade_expression(maybe_fn_handle, pointer)?,
            },
            Expression::ImageSample {
                image,
                sampler,
                gather,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
            } => Expression::ImageSample {
                image: self.upgrade_expression(maybe_fn_handle, image)?,
                sampler: self.upgrade_expression(maybe_fn_handle, sampler)?,
                gather,
                coordinate: self.upgrade_expression(maybe_fn_handle, coordinate)?,
                array_index: self.upgrade_opt_expression(maybe_fn_handle, array_index)?,
                offset: self.upgrade_opt_expression(maybe_fn_handle, offset)?,
                level: match level {
                    crate::SampleLevel::Exact(h) => {
                        crate::SampleLevel::Exact(self.upgrade_expression(maybe_fn_handle, h)?)
                    }
                    crate::SampleLevel::Bias(h) => {
                        crate::SampleLevel::Bias(self.upgrade_expression(maybe_fn_handle, h)?)
                    }
                    crate::SampleLevel::Gradient { x, y } => crate::SampleLevel::Gradient {
                        x: self.upgrade_expression(maybe_fn_handle, x)?,
                        y: self.upgrade_expression(maybe_fn_handle, y)?,
                    },
                    n => n,
                },
                depth_ref: self.upgrade_opt_expression(maybe_fn_handle, depth_ref)?,
            },
            Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => Expression::ImageLoad {
                image: self.upgrade_expression(maybe_fn_handle, image)?,
                coordinate: self.upgrade_expression(maybe_fn_handle, coordinate)?,
                array_index: self.upgrade_opt_expression(maybe_fn_handle, array_index)?,
                sample: self.upgrade_opt_expression(maybe_fn_handle, sample)?,
                level: self.upgrade_opt_expression(maybe_fn_handle, level)?,
            },
            Expression::ImageQuery { image, query } => Expression::ImageQuery {
                image: self.upgrade_expression(maybe_fn_handle, image)?,
                query: match query {
                    crate::ImageQuery::Size { level } => crate::ImageQuery::Size {
                        level: self.upgrade_opt_expression(maybe_fn_handle, level)?,
                    },
                    n => n,
                },
            },
            Expression::Unary { op, expr } => Expression::Unary {
                op,
                expr: self.upgrade_expression(maybe_fn_handle, expr)?,
            },
            Expression::Binary { op, left, right } => Expression::Binary {
                op,
                left: self.upgrade_expression(maybe_fn_handle, left)?,
                right: self.upgrade_expression(maybe_fn_handle, right)?,
            },
            Expression::Select {
                condition,
                accept,
                reject,
            } => Expression::Select {
                condition: self.upgrade_expression(maybe_fn_handle, condition)?,
                accept: self.upgrade_expression(maybe_fn_handle, accept)?,
                reject: self.upgrade_expression(maybe_fn_handle, reject)?,
            },
            Expression::Derivative { axis, ctrl, expr } => Expression::Derivative {
                axis,
                ctrl,
                expr: self.upgrade_expression(maybe_fn_handle, expr)?,
            },
            Expression::Relational { fun, argument } => Expression::Relational {
                fun,
                argument: self.upgrade_expression(maybe_fn_handle, argument)?,
            },
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => Expression::Math {
                fun,
                arg: self.upgrade_expression(maybe_fn_handle, arg)?,
                arg1: self.upgrade_opt_expression(maybe_fn_handle, arg1)?,
                arg2: self.upgrade_opt_expression(maybe_fn_handle, arg2)?,
                arg3: self.upgrade_opt_expression(maybe_fn_handle, arg3)?,
            },
            Expression::As {
                expr,
                kind,
                convert,
            } => Expression::As {
                expr: self.upgrade_expression(maybe_fn_handle, expr)?,
                kind,
                convert,
            },
            c @ Expression::CallResult(_) => c,
            a @ Expression::AtomicResult { .. } => a,
            Expression::WorkGroupUniformLoadResult { ty } => {
                Expression::WorkGroupUniformLoadResult {
                    ty: self.upgrade_type(ty)?,
                }
            }
            Expression::ArrayLength(h) => {
                Expression::ArrayLength(self.upgrade_expression(maybe_fn_handle, h)?)
            }
            r @ Expression::RayQueryProceedResult => r,
            Expression::RayQueryGetIntersection { query, committed } => {
                Expression::RayQueryGetIntersection {
                    query: self.upgrade_expression(maybe_fn_handle, query)?,
                    committed,
                }
            }
            s @ Expression::SubgroupBallotResult => s,
            Expression::SubgroupOperationResult { ty } => Expression::SubgroupOperationResult {
                ty: self.upgrade_type(ty)?,
            },
        };

        if new_expr != expr {
            padding.debug("inserting new expr: ", &new_expr);
            let arena = if let Some(fh) = maybe_fn_handle {
                let f = self.module.functions.get_mut(fh);
                &mut f.expressions
            } else {
                &mut self.module.global_expressions
            };
            let span = arena.get_span(handle);
            let new_handle = arena.append(new_expr, span);
            padding.debug("new expr handle: ", new_handle);
            Ok(new_handle)
        } else {
            Ok(handle)
        }
    }
}

impl Module {
    /// Upgrade all atomics given.
    pub(crate) fn upgrade_atomics(
        &mut self,
        ops: impl IntoIterator<Item = AtomicOp>,
    ) -> Result<(), Error> {
        let mut state = UpgradeState {
            padding: Default::default(),
            module: self,
        };

        for op in ops.into_iter() {
            let padding = state.inc_padding();
            padding.debug("op: ", op);

            // Find the expression's enclosing function, if any
            let mut maybe_fn_handle = None;
            for (fn_handle, function) in state.module.functions.iter() {
                log::trace!("function: {fn_handle:?}");
                if function.expressions.try_get(op.pointer_handle).is_ok() {
                    log::trace!("  is op's function");
                    maybe_fn_handle = Some(fn_handle);
                    break;
                }
            }

            padding.debug("upgrading the pointer type:", op.pointer_type_handle);
            let _new_pointer_type_handle = state.upgrade_type(op.pointer_type_handle)?;

            padding.debug("upgrading the pointer expression", op.pointer_handle);
            let _new_pointer_handle =
                state.upgrade_expression(maybe_fn_handle, op.pointer_handle)?;
        }

        Ok(())
    }
}
