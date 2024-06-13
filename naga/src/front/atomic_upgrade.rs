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
            Expression::LocalVariable(var) => Expression::LocalVariable(
                self.upgrade_local_variable(maybe_fn_handle.ok_or(Error::NoFunction)?, var)?,
            ),
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
