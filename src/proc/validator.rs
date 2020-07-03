use crate::arena::Handle;

pub struct Validator {}

#[derive(Clone, Debug)]
pub enum ValidationError {
    /// The type width is not supported.
    InvalidTypeWidth(crate::ScalarKind, crate::Bytes),
    /// A type handle can not be resolved.
    UnresolvedType(Handle<crate::Type>),
    /// There are instructions after `return`/`break`/`continue`.
    InvalidControlFlowExitTail,
}

impl Validator {
    /// Construct a new validator instance.
    pub fn new() -> Self {
        Validator {}
    }

    /// Check the given module to be valid.
    pub fn validate(&mut self, module: &crate::Module) -> Result<(), ValidationError> {
        for (handle, ty) in module.types.iter() {
            use crate::TypeInner as Ti;
            match ty.inner {
                Ti::Scalar { kind, width }
                | Ti::Vector { kind, width, .. }
                | Ti::Matrix { kind, width, .. } => {
                    if width != 4 {
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
                Ti::Image { .. } | Ti::DepthImage { .. } => {}
                Ti::Sampler { comparison: _ } => {}
            }
        }

        Ok(())
    }
}
