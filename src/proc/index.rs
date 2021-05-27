//! Definitions for index bounds checking.

use super::ProcError;

impl crate::TypeInner {
    /// Return the length of a subscriptable type.
    ///
    /// The `self` parameter should be a handle to a vector, matrix, or array
    /// type, a pointer to one of those, or a value pointer. Arrays may be
    /// fixed-size, dynamically sized, or sized by a specializable constant.
    ///
    /// The value returned is appropriate for bounds checks on subscripting.
    ///
    /// Return an error if `self` does not describe a subscriptable type at all.
    pub fn indexable_length(&self, module: &crate::Module) -> Result<IndexableLength, ProcError> {
        use crate::TypeInner as Ti;
        let known_length = match *self {
            Ti::Vector { size, .. } => size as _,
            Ti::Matrix { columns, .. } => columns as _,
            Ti::Array { size, .. } => {
                return size.to_indexable_length(module);
            }
            Ti::ValuePointer {
                size: Some(size), ..
            } => size as _,
            Ti::Pointer { base, .. } => {
                // When assigning types to expressions, ResolveContext::Resolve
                // does a separate sub-match here instead of a full recursion,
                // so we'll do the same.
                let base_inner = &module.types[base].inner;
                match *base_inner {
                    Ti::Vector { size, .. } => size as _,
                    Ti::Matrix { columns, .. } => columns as _,
                    Ti::Array { size, .. } => return size.to_indexable_length(module),
                    _ => return Err(ProcError::TypeNotIndexable),
                }
            }
            _ => return Err(ProcError::TypeNotIndexable),
        };
        Ok(IndexableLength::Known(known_length))
    }
}

/// The number of elements in an indexable type.
///
/// This summarizes the length of vectors, matrices, and arrays in a way that is
/// convenient for indexing and bounds-checking code.
pub enum IndexableLength {
    /// Values of this type always have the given number of elements.
    Known(u32),

    /// The value of the given specializable constant is the number of elements.
    /// (Non-specializable constants are reported as `Known`.)
    Specializable(crate::Handle<crate::Constant>),

    /// The number of elements is determined at runtime.
    Dynamic,
}

impl crate::ArraySize {
    pub fn to_indexable_length(self, module: &crate::Module) -> Result<IndexableLength, ProcError> {
        use crate::Constant as K;
        Ok(match self {
            Self::Constant(k) => match module.constants[k] {
                K {
                    specialization: Some(_),
                    ..
                } => IndexableLength::Specializable(k),
                ref unspecialized => {
                    let length = unspecialized
                        .to_array_length()
                        .ok_or(ProcError::InvalidArraySizeConstant(k))?;
                    IndexableLength::Known(length)
                }
            },
            Self::Dynamic => IndexableLength::Dynamic,
        })
    }
}
