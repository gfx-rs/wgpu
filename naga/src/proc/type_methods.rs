//! Methods on [`TypeInner`], [`Scalar`], and [`ScalarKind`].
//!
//! [`TypeInner`]: crate::TypeInner
//! [`Scalar`]: crate::Scalar
//! [`ScalarKind`]: crate::ScalarKind

use super::TypeResolution;

impl crate::ScalarKind {
    pub const fn is_numeric(self) -> bool {
        match self {
            crate::ScalarKind::Sint
            | crate::ScalarKind::Uint
            | crate::ScalarKind::Float
            | crate::ScalarKind::AbstractInt
            | crate::ScalarKind::AbstractFloat => true,
            crate::ScalarKind::Bool => false,
        }
    }
}

impl crate::Scalar {
    pub const I32: Self = Self {
        kind: crate::ScalarKind::Sint,
        width: 4,
    };
    pub const U32: Self = Self {
        kind: crate::ScalarKind::Uint,
        width: 4,
    };
    pub const F32: Self = Self {
        kind: crate::ScalarKind::Float,
        width: 4,
    };
    pub const F64: Self = Self {
        kind: crate::ScalarKind::Float,
        width: 8,
    };
    pub const I64: Self = Self {
        kind: crate::ScalarKind::Sint,
        width: 8,
    };
    pub const U64: Self = Self {
        kind: crate::ScalarKind::Uint,
        width: 8,
    };
    pub const BOOL: Self = Self {
        kind: crate::ScalarKind::Bool,
        width: crate::BOOL_WIDTH,
    };
    pub const ABSTRACT_INT: Self = Self {
        kind: crate::ScalarKind::AbstractInt,
        width: crate::ABSTRACT_WIDTH,
    };
    pub const ABSTRACT_FLOAT: Self = Self {
        kind: crate::ScalarKind::AbstractFloat,
        width: crate::ABSTRACT_WIDTH,
    };

    pub const fn is_abstract(self) -> bool {
        match self.kind {
            crate::ScalarKind::AbstractInt | crate::ScalarKind::AbstractFloat => true,
            crate::ScalarKind::Sint
            | crate::ScalarKind::Uint
            | crate::ScalarKind::Float
            | crate::ScalarKind::Bool => false,
        }
    }

    /// Construct a float `Scalar` with the given width.
    ///
    /// This is especially common when dealing with
    /// `TypeInner::Matrix`, where the scalar kind is implicit.
    pub const fn float(width: crate::Bytes) -> Self {
        Self {
            kind: crate::ScalarKind::Float,
            width,
        }
    }

    pub const fn to_inner_scalar(self) -> crate::TypeInner {
        crate::TypeInner::Scalar(self)
    }

    pub const fn to_inner_vector(self, size: crate::VectorSize) -> crate::TypeInner {
        crate::TypeInner::Vector { size, scalar: self }
    }

    pub const fn to_inner_atomic(self) -> crate::TypeInner {
        crate::TypeInner::Atomic(self)
    }
}

const POINTER_SPAN: u32 = 4;

impl crate::TypeInner {
    /// Return the scalar type of `self`.
    ///
    /// If `inner` is a scalar, vector, or matrix type, return
    /// its scalar type. Otherwise, return `None`.
    pub const fn scalar(&self) -> Option<crate::Scalar> {
        use crate::TypeInner as Ti;
        match *self {
            Ti::Scalar(scalar) | Ti::Vector { scalar, .. } => Some(scalar),
            Ti::Matrix { scalar, .. } => Some(scalar),
            _ => None,
        }
    }

    pub fn scalar_kind(&self) -> Option<crate::ScalarKind> {
        self.scalar().map(|scalar| scalar.kind)
    }

    /// Returns the scalar width in bytes
    pub fn scalar_width(&self) -> Option<u8> {
        self.scalar().map(|scalar| scalar.width)
    }

    pub const fn pointer_space(&self) -> Option<crate::AddressSpace> {
        match *self {
            Self::Pointer { space, .. } => Some(space),
            Self::ValuePointer { space, .. } => Some(space),
            _ => None,
        }
    }

    pub fn is_atomic_pointer(&self, types: &crate::UniqueArena<crate::Type>) -> bool {
        match *self {
            crate::TypeInner::Pointer { base, .. } => match types[base].inner {
                crate::TypeInner::Atomic { .. } => true,
                _ => false,
            },
            _ => false,
        }
    }

    /// Get the size of this type.
    pub fn size(&self, _gctx: super::GlobalCtx) -> u32 {
        match *self {
            Self::Scalar(scalar) | Self::Atomic(scalar) => scalar.width as u32,
            Self::Vector { size, scalar } => size as u32 * scalar.width as u32,
            // matrices are treated as arrays of aligned columns
            Self::Matrix {
                columns,
                rows,
                scalar,
            } => super::Alignment::from(rows) * scalar.width as u32 * columns as u32,
            Self::Pointer { .. } | Self::ValuePointer { .. } => POINTER_SPAN,
            Self::Array {
                base: _,
                size,
                stride,
            } => {
                let count = match size {
                    crate::ArraySize::Constant(count) => count.get(),
                    // any struct member or array element needing a size at pipeline-creation time
                    // must have a creation-fixed footprint
                    crate::ArraySize::Pending(_) => 0,
                    // A dynamically-sized array has to have at least one element
                    crate::ArraySize::Dynamic => 1,
                };
                count * stride
            }
            Self::Struct { span, .. } => span,
            Self::Image { .. }
            | Self::Sampler { .. }
            | Self::AccelerationStructure
            | Self::RayQuery
            | Self::BindingArray { .. } => 0,
        }
    }

    /// Return the canonical form of `self`, or `None` if it's already in
    /// canonical form.
    ///
    /// Certain types have multiple representations in `TypeInner`. This
    /// function converts all forms of equivalent types to a single
    /// representative of their class, so that simply applying `Eq` to the
    /// result indicates whether the types are equivalent, as far as Naga IR is
    /// concerned.
    pub fn canonical_form(
        &self,
        types: &crate::UniqueArena<crate::Type>,
    ) -> Option<crate::TypeInner> {
        use crate::TypeInner as Ti;
        match *self {
            Ti::Pointer { base, space } => match types[base].inner {
                Ti::Scalar(scalar) => Some(Ti::ValuePointer {
                    size: None,
                    scalar,
                    space,
                }),
                Ti::Vector { size, scalar } => Some(Ti::ValuePointer {
                    size: Some(size),
                    scalar,
                    space,
                }),
                _ => None,
            },
            _ => None,
        }
    }

    /// Compare `self` and `rhs` as types.
    ///
    /// This is mostly the same as `<TypeInner as Eq>::eq`, but it treats
    /// `ValuePointer` and `Pointer` types as equivalent.
    ///
    /// When you know that one side of the comparison is never a pointer, it's
    /// fine to not bother with canonicalization, and just compare `TypeInner`
    /// values with `==`.
    pub fn equivalent(
        &self,
        rhs: &crate::TypeInner,
        types: &crate::UniqueArena<crate::Type>,
    ) -> bool {
        let left = self.canonical_form(types);
        let right = rhs.canonical_form(types);
        left.as_ref().unwrap_or(self) == right.as_ref().unwrap_or(rhs)
    }

    pub fn is_dynamically_sized(&self, types: &crate::UniqueArena<crate::Type>) -> bool {
        use crate::TypeInner as Ti;
        match *self {
            Ti::Array { size, .. } => size == crate::ArraySize::Dynamic,
            Ti::Struct { ref members, .. } => members
                .last()
                .map(|last| types[last.ty].inner.is_dynamically_sized(types))
                .unwrap_or(false),
            _ => false,
        }
    }

    pub fn components(&self) -> Option<u32> {
        Some(match *self {
            Self::Vector { size, .. } => size as u32,
            Self::Matrix { columns, .. } => columns as u32,
            Self::Array {
                size: crate::ArraySize::Constant(len),
                ..
            } => len.get(),
            Self::Struct { ref members, .. } => members.len() as u32,
            _ => return None,
        })
    }

    pub fn component_type(&self, index: usize) -> Option<TypeResolution> {
        Some(match *self {
            Self::Vector { scalar, .. } => TypeResolution::Value(crate::TypeInner::Scalar(scalar)),
            Self::Matrix { rows, scalar, .. } => {
                TypeResolution::Value(crate::TypeInner::Vector { size: rows, scalar })
            }
            Self::Array {
                base,
                size: crate::ArraySize::Constant(_),
                ..
            } => TypeResolution::Handle(base),
            Self::Struct { ref members, .. } => TypeResolution::Handle(members[index].ty),
            _ => return None,
        })
    }
}
