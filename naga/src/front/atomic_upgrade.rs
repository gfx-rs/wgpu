//! [`Module`] helpers for "upgrading" atomics in the SPIR-V (and eventually GLSL) frontends.
use std::sync::{atomic::AtomicUsize, Arc};

use crate::{Expression, Function, GlobalVariable, Handle, Module, StructMember, Type, TypeInner};

#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error("encountered an unsupported expression")]
    Unsupported,
}

impl From<Error> for crate::front::spv::Error {
    fn from(source: Error) -> Self {
        crate::front::spv::Error::AtomicUpgradeError(source)
    }
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
    fn upgrade_type(&mut self, ty: Handle<Type>) -> Result<Handle<Type>, Error> {
        let padding = self.inc_padding();
        padding.trace("upgrading type: ", ty);

        let r#type = self.module.types[ty].clone();

        let inner = match r#type.inner.clone() {
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

        let new_type = Type {
            name: r#type.name.clone(),
            inner,
        };
        let new_ty = if let Some(prev_ty) = self.module.types.get(&new_type) {
            padding.trace("type exists: ", prev_ty);
            prev_ty
        } else {
            padding.debug("ty: ", ty);
            padding.debug("from: ", &r#type);
            padding.debug("to:   ", &new_type);

            let new_ty = self
                .module
                .types
                .insert(new_type, self.module.types.get_span(ty));
            padding.debug("new ty: ", new_ty);
            new_ty
        };
        Ok(new_ty)
    }

    fn upgrade_global_variable(&mut self, handle: Handle<GlobalVariable>) -> Result<(), Error> {
        let padding = self.inc_padding();
        padding.trace("upgrading global variable: ", handle);

        let var = self.module.global_variables[handle].clone();

        let new_var = GlobalVariable {
            name: var.name.clone(),
            space: var.space,
            binding: var.binding.clone(),
            ty: self.upgrade_type(var.ty)?,
            init: self.upgrade_opt_expression(None, var.init)?,
        };
        if new_var != var {
            padding.debug("upgrading global variable: ", handle);
            padding.debug("from:     ", &var);
            padding.debug("to:       ", &new_var);
            self.module.global_variables[handle] = new_var;
        }
        Ok(())
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
        padding.trace("upgrading expr: ", handle);

        let expr = if let Some(fh) = maybe_fn_handle {
            let function = &self.module.functions[fh];
            function.expressions[handle].clone()
        } else {
            self.module.global_expressions[handle].clone()
        };

        let new_expr = match expr.clone() {
            Expression::AccessIndex { base, index } => Expression::AccessIndex {
                base: self.upgrade_expression(maybe_fn_handle, base)?,
                index,
            },
            Expression::GlobalVariable(var) => {
                self.upgrade_global_variable(var)?;
                Expression::GlobalVariable(var)
            }
            lv @ Expression::LocalVariable(_) => lv,
            _ => {
                return Err(Error::Unsupported);
            }
        };

        if new_expr != expr {
            padding.debug("upgrading expr: ", handle);
            padding.debug("from: ", &expr);
            padding.debug("to:   ", &new_expr);
            let arena = if let Some(fh) = maybe_fn_handle {
                let f = self.module.functions.get_mut(fh);
                &mut f.expressions
            } else {
                &mut self.module.global_expressions
            };
            let span = arena.get_span(handle);
            let new_handle = arena.append(new_expr, span);
            padding.debug("new expr: ", new_handle);
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
        global_var_handles: impl IntoIterator<Item = Handle<GlobalVariable>>,
    ) -> Result<(), Error> {
        let mut state = UpgradeState {
            padding: Default::default(),
            module: self,
        };

        for handle in global_var_handles {
            state.upgrade_global_variable(handle)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {

    #[test]
    fn atomic_i_inc() {
        let _ = env_logger::builder().is_test(true).try_init();
        let bytes = include_bytes!("../../tests/in/spv/atomic_i_increment.spv");
        let m = crate::front::spv::parse_u8_slice(bytes, &Default::default()).unwrap();
        let mut validator = crate::valid::Validator::new(
            crate::valid::ValidationFlags::empty(),
            Default::default(),
        );
        let info = match validator.validate(&m) {
            Err(e) => {
                log::error!("{}", e.emit_to_string(""));
                return;
            }
            Ok(i) => i,
        };
        let wgsl =
            crate::back::wgsl::write_string(&m, &info, crate::back::wgsl::WriterFlags::empty())
                .unwrap();
        log::info!("atomic_i_increment:\n{wgsl}");

        let m = match crate::front::wgsl::parse_str(&wgsl) {
            Ok(m) => m,
            Err(e) => {
                log::error!("{}", e.emit_to_string(&wgsl));
                panic!("invalid module");
            }
        };
        let mut validator =
            crate::valid::Validator::new(crate::valid::ValidationFlags::all(), Default::default());
        if let Err(e) = validator.validate(&m) {
            log::error!("{}", e.emit_to_string(&wgsl));
            panic!("invalid generated wgsl");
        }
    }
}
