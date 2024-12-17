//! Upgrade the types of scalars observed to be accessed as atomics to [`Atomic`] types.
//!
//! In SPIR-V, atomic operations can be applied to any scalar value, but in Naga
//! IR atomic operations can only be applied to values of type [`Atomic`]. Naga
//! IR's restriction matches Metal Shading Language and WGSL, so we don't want
//! to relax that. Instead, when the SPIR-V front end observes a value being
//! accessed using atomic instructions, it promotes the value's type from
//! [`Scalar`] to [`Atomic`]. This module implements `Module::upgrade_atomics`,
//! the function that makes that change.
//!
//! Atomics can only appear in global variables in the [`Storage`] and
//! [`Workgroup`] address spaces. These variables can either have `Atomic` types
//! themselves, or be [`Array`]s of such, or be [`Struct`]s containing such.
//! So we only need to change the types of globals and struct fields.
//!
//! Naga IR [`Load`] expressions and [`Store`] statements can operate directly
//! on [`Atomic`] values, retrieving and depositing ordinary [`Scalar`] values,
//! so changing the types doesn't have much effect on the code that operates on
//! those values.
//!
//! Future work:
//!
//! - Atomics in structs are not implemented yet.
//!
//! - The GLSL front end could use this transformation as well.
//!
//! [`Atomic`]: TypeInner::Atomic
//! [`Scalar`]: TypeInner::Scalar
//! [`Storage`]: crate::AddressSpace::Storage
//! [`WorkGroup`]: crate::AddressSpace::WorkGroup
//! [`Array`]: TypeInner::Array
//! [`Struct`]: TypeInner::Struct
//! [`Load`]: crate::Expression::Load
//! [`Store`]: crate::Statement::Store
use std::sync::{atomic::AtomicUsize, Arc};

use crate::{GlobalVariable, Handle, Module, Type, TypeInner};

#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error("encountered an unsupported expression")]
    Unsupported,
    #[error("upgrading structs of more than one member is not yet implemented")]
    MultiMemberStruct,
    #[error("encountered unsupported global initializer in an atomic variable")]
    GlobalInitUnsupported,
    #[error("expected to find a global variable")]
    GlobalVariableMissing,
    #[error("atomic compare exchange requires a scalar base type")]
    CompareExchangeNonScalarBaseType,
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

impl UpgradeState<'_> {
    fn inc_padding(&self) -> Padding {
        self.padding.inc_padding()
    }

    /// Upgrade the type, recursing until we reach the leaves.
    /// At the leaves, replace scalars with atomic scalars.
    fn upgrade_type(&mut self, ty: Handle<Type>) -> Result<Handle<Type>, Error> {
        let padding = self.inc_padding();
        padding.trace("upgrading type: ", ty);

        let inner = match self.module.types[ty].inner {
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
            TypeInner::Struct { ref members, span } => {
                // In the future we should have to figure out which member needs
                // upgrading, but for now we'll only cover the single-member
                // case.
                let &[crate::StructMember {
                    ref name,
                    ty,
                    ref binding,
                    offset,
                }] = &members[..]
                else {
                    return Err(Error::MultiMemberStruct);
                };

                // Take our own clones of these values now, so that
                // `upgrade_type` can mutate the module.
                let name = name.clone();
                let binding = binding.clone();
                let upgraded_member_type = self.upgrade_type(ty)?;
                TypeInner::Struct {
                    members: vec![crate::StructMember {
                        name,
                        ty: upgraded_member_type,
                        binding,
                        offset,
                    }],
                    span,
                }
            }
            TypeInner::BindingArray { base, size } => TypeInner::BindingArray {
                base: self.upgrade_type(base)?,
                size,
            },
            _ => return Ok(ty),
        };

        // Now that we've upgraded any subtypes, re-borrow a reference to our
        // type and update its `inner`.
        let r#type = &self.module.types[ty];
        let span = self.module.types.get_span(ty);
        let new_type = Type {
            name: r#type.name.clone(),
            inner,
        };
        padding.debug("ty: ", ty);
        padding.debug("from: ", r#type);
        padding.debug("to:   ", &new_type);
        let new_handle = self.module.types.insert(new_type, span);
        Ok(new_handle)
    }

    fn upgrade_global_variable(&mut self, handle: Handle<GlobalVariable>) -> Result<(), Error> {
        let padding = self.inc_padding();
        padding.trace("upgrading global variable: ", handle);

        let var = &self.module.global_variables[handle];

        if var.init.is_some() {
            return Err(Error::GlobalInitUnsupported);
        }

        let var_ty = var.ty;
        let new_ty = self.upgrade_type(var.ty)?;
        if new_ty != var_ty {
            padding.debug("upgrading global variable: ", handle);
            padding.debug("from ty: ", var_ty);
            padding.debug("to ty:   ", new_ty);
            self.module.global_variables[handle].ty = new_ty;
        }
        Ok(())
    }
}

impl Module {
    /// Upgrade `global_var_handles` to have [`Atomic`] leaf types.
    ///
    /// [`Atomic`]: TypeInner::Atomic
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
