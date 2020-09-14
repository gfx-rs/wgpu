use super::typifier::{ResolveContext, ResolveError, Typifier};
use crate::arena::Handle;

#[derive(Debug)]
pub struct Validator {
    //Note: this is a bit tricky: some of the front-ends as well as backends
    // already have to use the typifier, so the work here is redundant in a way.
    typifier: Typifier,
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("The type width is not supported")]
    InvalidTypeWidth(crate::ScalarKind, crate::Bytes),
    #[error("The type handle can not be resolved")]
    UnresolvedType(Handle<crate::Type>),
    #[error("Expression type can't be resolved")]
    Resolve(Handle<crate::Function>, ResolveError),
    #[error("There are instructions after `return`/`break`/`continue`")]
    InvalidControlFlowExitTail,
}

impl Validator {
    /// Construct a new validator instance.
    pub fn new() -> Self {
        Validator {
            typifier: Typifier::new(),
        }
    }

    /// Check the given module to be valid.
    pub fn validate(&mut self, module: &crate::Module) -> Result<(), ValidationError> {
        // check the types
        for (handle, ty) in module.types.iter() {
            use crate::TypeInner as Ti;
            match ty.inner {
                Ti::Scalar { kind, width }
                | Ti::Vector { kind, width, .. }
                | Ti::Matrix { kind, width, .. } => {
                    let expected = match kind {
                        crate::ScalarKind::Bool => 1,
                        _ => 4,
                    };
                    if width != expected {
                        return Err(ValidationError::InvalidTypeWidth(kind, width));
                    }
                }
                Ti::Pointer { base, class: _ } => {
                    if base >= handle {
                        return Err(ValidationError::UnresolvedType(base));
                    }
                }
                Ti::Array { base, .. } => {
                    if base >= handle {
                        return Err(ValidationError::UnresolvedType(base));
                    }
                }
                Ti::Struct { ref members } => {
                    //TODO: check that offsets are not intersecting?
                    for member in members {
                        if member.ty >= handle {
                            return Err(ValidationError::UnresolvedType(member.ty));
                        }
                    }
                }
                Ti::Image { .. } => {}
                Ti::Sampler { comparison: _ } => {}
            }
        }

        // check the type resolution of expressions
        for (fun_handle, fun) in module.functions.iter() {
            let resolve_ctx = ResolveContext {
                constants: &module.constants,
                global_vars: &module.global_variables,
                local_vars: &fun.local_variables,
                functions: &module.functions,
                parameter_types: &fun.parameter_types,
            };
            if let Err(e) = self
                .typifier
                .resolve_all(&fun.expressions, &module.types, &resolve_ctx)
            {
                return Err(ValidationError::Resolve(fun_handle, e));
            }
        }

        Ok(())
    }
}
