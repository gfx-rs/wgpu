//! [`Module`] helpers for "upgrading" atomics in the SPIR-V (and eventually GLSL) frontends.
use std::sync::{atomic::AtomicUsize, Arc};

use crate::{
    Expression, Function, GlobalVariable, Handle, LocalVariable, Module, StructMember, Type,
    TypeInner,
};

#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error("bad handle: {0}")]
    MissingHandle(crate::arena::BadHandle),
    #[error("no function context")]
    NoFunction,
    #[error("encountered an unsupported expression")]
    Unsupported,
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

/// Information about some [`Atomic`][as] statement, for upgrading types.
///
/// SPIR-V doesn't have atomic types like Naga IR's [`Atomic`][at], it
/// just has atomic instructions that operate on pointers to ordinary
/// scalar values, so to build Naga IR from SPIR-V input, we must
/// observe which variables/arguments/fields the SPIR-V applies atomic
/// instructions to, and then update their types after the fact.
///
/// This type describes some [`Atomic`][as] statement we've generated,
/// along with enough information for us to find the items whose types
/// we need to upgrade.
///
/// [at]: crate::TypeInner::Atomic
/// [as]: crate::Statement::Atomic
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AtomicOp {
    pub instruction: AtomicOpInst,
    /// The type of the [`Atomic`] statement's [`pointer`] operand.
    ///
    /// [`Atomic`]: crate::Statement::Atomic
    /// [`pointer`]: crate::Statement::Atomic::pointer
    pub pointer_type_handle: Handle<Type>,
    /// Handle to the pointer expression in the module/function
    pub pointer_handle: Handle<Expression>,
}

#[derive(Clone, Default)]
struct Padding(Arc<AtomicUsize>);

impl std::fmt::Display for Padding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for _ in 0..self.0.load(std::sync::atomic::Ordering::Relaxed) {
            f.write_str("  ")?;
        }
        Ok(())
    }
}

impl Drop for Padding {
    fn drop(&mut self) {
        let _ = self.0.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Padding {
    fn trace(&self, msg: impl std::fmt::Display, t: impl std::fmt::Debug) {
        format!("{msg} {t:#?}")
            .split('\n')
            .for_each(|ln| log::trace!("{self}{ln}"));
    }

    fn debug(&self, msg: impl std::fmt::Display, t: impl std::fmt::Debug) {
        format!("{msg} {t:#?}")
            .split('\n')
            .for_each(|ln| log::debug!("{self}{ln}"));
    }

    fn inc_padding(&self) -> Padding {
        let _ = self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.clone()
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
            binding: var.binding.clone(),
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
            Expression::AccessIndex { base, index } => Expression::AccessIndex {
                base: self.upgrade_expression(maybe_fn_handle, base)?,
                index,
            },
            Expression::GlobalVariable(var) => {
                Expression::GlobalVariable(self.upgrade_global_variable(var)?)
            }
            lv @ Expression::LocalVariable(_) => lv,
            _ => {
                return Err(Error::Unsupported);
            }
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
