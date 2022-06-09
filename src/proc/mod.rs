/*!
[`Module`](super::Module) processing functionality.
*/

pub mod index;
mod layouter;
mod namer;
mod terminator;
mod typifier;

use std::cmp::PartialEq;

pub use index::{BoundsCheckPolicies, BoundsCheckPolicy, IndexableLength, IndexableLengthError};
pub use layouter::{Alignment, LayoutError, LayoutErrorInner, Layouter, TypeLayout};
pub use namer::{EntryPointIndex, NameKey, Namer};
pub use terminator::ensure_block_returns;
pub use typifier::{ResolveContext, ResolveError, TypeResolution};

impl From<super::StorageFormat> for super::ScalarKind {
    fn from(format: super::StorageFormat) -> Self {
        use super::{ScalarKind as Sk, StorageFormat as Sf};
        match format {
            Sf::R8Unorm => Sk::Float,
            Sf::R8Snorm => Sk::Float,
            Sf::R8Uint => Sk::Uint,
            Sf::R8Sint => Sk::Sint,
            Sf::R16Uint => Sk::Uint,
            Sf::R16Sint => Sk::Sint,
            Sf::R16Float => Sk::Float,
            Sf::Rg8Unorm => Sk::Float,
            Sf::Rg8Snorm => Sk::Float,
            Sf::Rg8Uint => Sk::Uint,
            Sf::Rg8Sint => Sk::Sint,
            Sf::R32Uint => Sk::Uint,
            Sf::R32Sint => Sk::Sint,
            Sf::R32Float => Sk::Float,
            Sf::Rg16Uint => Sk::Uint,
            Sf::Rg16Sint => Sk::Sint,
            Sf::Rg16Float => Sk::Float,
            Sf::Rgba8Unorm => Sk::Float,
            Sf::Rgba8Snorm => Sk::Float,
            Sf::Rgba8Uint => Sk::Uint,
            Sf::Rgba8Sint => Sk::Sint,
            Sf::Rgb10a2Unorm => Sk::Float,
            Sf::Rg11b10Float => Sk::Float,
            Sf::Rg32Uint => Sk::Uint,
            Sf::Rg32Sint => Sk::Sint,
            Sf::Rg32Float => Sk::Float,
            Sf::Rgba16Uint => Sk::Uint,
            Sf::Rgba16Sint => Sk::Sint,
            Sf::Rgba16Float => Sk::Float,
            Sf::Rgba32Uint => Sk::Uint,
            Sf::Rgba32Sint => Sk::Sint,
            Sf::Rgba32Float => Sk::Float,
        }
    }
}

impl super::ScalarValue {
    pub const fn scalar_kind(&self) -> super::ScalarKind {
        match *self {
            Self::Uint(_) => super::ScalarKind::Uint,
            Self::Sint(_) => super::ScalarKind::Sint,
            Self::Float(_) => super::ScalarKind::Float,
            Self::Bool(_) => super::ScalarKind::Bool,
        }
    }
}

impl super::ScalarKind {
    pub const fn is_numeric(self) -> bool {
        match self {
            crate::ScalarKind::Sint | crate::ScalarKind::Uint | crate::ScalarKind::Float => true,
            crate::ScalarKind::Bool => false,
        }
    }
}

pub const POINTER_SPAN: u32 = 4;

impl super::TypeInner {
    pub const fn scalar_kind(&self) -> Option<super::ScalarKind> {
        match *self {
            super::TypeInner::Scalar { kind, .. } | super::TypeInner::Vector { kind, .. } => {
                Some(kind)
            }
            super::TypeInner::Matrix { .. } => Some(super::ScalarKind::Float),
            _ => None,
        }
    }

    pub const fn pointer_space(&self) -> Option<crate::AddressSpace> {
        match *self {
            Self::Pointer { space, .. } => Some(space),
            Self::ValuePointer { space, .. } => Some(space),
            _ => None,
        }
    }

    pub fn try_size(
        &self,
        constants: &super::Arena<super::Constant>,
    ) -> Result<u32, crate::arena::BadHandle> {
        Ok(match *self {
            Self::Scalar { kind: _, width } | Self::Atomic { kind: _, width } => width as u32,
            Self::Vector {
                size,
                kind: _,
                width,
            } => size as u32 * width as u32,
            // matrices are treated as arrays of aligned columns
            Self::Matrix {
                columns,
                rows,
                width,
            } => Alignment::from(rows) * width as u32 * columns as u32,
            Self::Pointer { .. } | Self::ValuePointer { .. } => POINTER_SPAN,
            Self::Array {
                base: _,
                size,
                stride,
            } => {
                let count = match size {
                    super::ArraySize::Constant(handle) => {
                        let constant = constants.try_get(handle)?;
                        constant.to_array_length().unwrap_or(1)
                    }
                    // A dynamically-sized array has to have at least one element
                    super::ArraySize::Dynamic => 1,
                };
                count * stride
            }
            Self::Struct { span, .. } => span,
            Self::Image { .. } | Self::Sampler { .. } | Self::BindingArray { .. } => 0,
        })
    }

    /// Get the size of this type. Panics if the `constants` doesn't contain
    /// a referenced handle. This may not happen in a properly validated IR module.
    pub fn size(&self, constants: &super::Arena<super::Constant>) -> u32 {
        self.try_size(constants).unwrap()
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
                Ti::Scalar { kind, width } => Some(Ti::ValuePointer {
                    size: None,
                    kind,
                    width,
                    space,
                }),
                Ti::Vector { size, kind, width } => Some(Ti::ValuePointer {
                    size: Some(size),
                    kind,
                    width,
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
}

impl super::AddressSpace {
    pub fn access(self) -> crate::StorageAccess {
        use crate::StorageAccess as Sa;
        match self {
            crate::AddressSpace::Function
            | crate::AddressSpace::Private
            | crate::AddressSpace::WorkGroup => Sa::LOAD | Sa::STORE,
            crate::AddressSpace::Uniform => Sa::LOAD,
            crate::AddressSpace::Storage { access } => access,
            crate::AddressSpace::Handle => Sa::LOAD,
            crate::AddressSpace::PushConstant => Sa::LOAD,
        }
    }
}

impl super::MathFunction {
    pub const fn argument_count(&self) -> usize {
        match *self {
            // comparison
            Self::Abs => 1,
            Self::Min => 2,
            Self::Max => 2,
            Self::Clamp => 3,
            // trigonometry
            Self::Cos => 1,
            Self::Cosh => 1,
            Self::Sin => 1,
            Self::Sinh => 1,
            Self::Tan => 1,
            Self::Tanh => 1,
            Self::Acos => 1,
            Self::Asin => 1,
            Self::Atan => 1,
            Self::Atan2 => 2,
            Self::Asinh => 1,
            Self::Acosh => 1,
            Self::Atanh => 1,
            Self::Radians => 1,
            Self::Degrees => 1,
            // decomposition
            Self::Ceil => 1,
            Self::Floor => 1,
            Self::Round => 1,
            Self::Fract => 1,
            Self::Trunc => 1,
            Self::Modf => 2,
            Self::Frexp => 2,
            Self::Ldexp => 2,
            // exponent
            Self::Exp => 1,
            Self::Exp2 => 1,
            Self::Log => 1,
            Self::Log2 => 1,
            Self::Pow => 2,
            // geometry
            Self::Dot => 2,
            Self::Outer => 2,
            Self::Cross => 2,
            Self::Distance => 2,
            Self::Length => 1,
            Self::Normalize => 1,
            Self::FaceForward => 3,
            Self::Reflect => 2,
            Self::Refract => 3,
            // computational
            Self::Sign => 1,
            Self::Fma => 3,
            Self::Mix => 3,
            Self::Step => 2,
            Self::SmoothStep => 3,
            Self::Sqrt => 1,
            Self::InverseSqrt => 1,
            Self::Inverse => 1,
            Self::Transpose => 1,
            Self::Determinant => 1,
            // bits
            Self::CountOneBits => 1,
            Self::ReverseBits => 1,
            Self::ExtractBits => 3,
            Self::InsertBits => 4,
            Self::FindLsb => 1,
            Self::FindMsb => 1,
            // data packing
            Self::Pack4x8snorm => 1,
            Self::Pack4x8unorm => 1,
            Self::Pack2x16snorm => 1,
            Self::Pack2x16unorm => 1,
            Self::Pack2x16float => 1,
            // data unpacking
            Self::Unpack4x8snorm => 1,
            Self::Unpack4x8unorm => 1,
            Self::Unpack2x16snorm => 1,
            Self::Unpack2x16unorm => 1,
            Self::Unpack2x16float => 1,
        }
    }
}

impl crate::Expression {
    /// Returns true if the expression is considered emitted at the start of a function.
    pub const fn needs_pre_emit(&self) -> bool {
        match *self {
            Self::Constant(_)
            | Self::FunctionArgument(_)
            | Self::GlobalVariable(_)
            | Self::LocalVariable(_) => true,
            _ => false,
        }
    }

    /// Return true if this expression is a dynamic array index, for [`Access`].
    ///
    /// This method returns true if this expression is a dynamically computed
    /// index, and as such can only be used to index matrices and arrays when
    /// they appear behind a pointer. See the documentation for [`Access`] for
    /// details.
    ///
    /// Note, this does not check the _type_ of the given expression. It's up to
    /// the caller to establish that the `Access` expression is well-typed
    /// through other means, like [`ResolveContext`].
    ///
    /// [`Access`]: crate::Expression::Access
    /// [`ResolveContext`]: crate::proc::ResolveContext
    pub fn is_dynamic_index(&self, module: &crate::Module) -> bool {
        if let Self::Constant(handle) = *self {
            let constant = &module.constants[handle];
            constant.specialization.is_some()
        } else {
            true
        }
    }
}

impl crate::Function {
    /// Return the global variable being accessed by the expression `pointer`.
    ///
    /// Assuming that `pointer` is a series of `Access` and `AccessIndex`
    /// expressions that ultimately access some part of a `GlobalVariable`,
    /// return a handle for that global.
    ///
    /// If the expression does not ultimately access a global variable, return
    /// `None`.
    pub fn originating_global(
        &self,
        mut pointer: crate::Handle<crate::Expression>,
    ) -> Option<crate::Handle<crate::GlobalVariable>> {
        loop {
            pointer = match self.expressions[pointer] {
                crate::Expression::Access { base, .. } => base,
                crate::Expression::AccessIndex { base, .. } => base,
                crate::Expression::GlobalVariable(handle) => return Some(handle),
                crate::Expression::LocalVariable(_) => return None,
                crate::Expression::FunctionArgument(_) => return None,
                // There are no other expressions that produce pointer values.
                _ => unreachable!(),
            }
        }
    }
}

impl crate::SampleLevel {
    pub const fn implicit_derivatives(&self) -> bool {
        match *self {
            Self::Auto | Self::Bias(_) => true,
            Self::Zero | Self::Exact(_) | Self::Gradient { .. } => false,
        }
    }
}

impl crate::Constant {
    /// Interpret this constant as an array length, and return it as a `u32`.
    ///
    /// Ignore any specialization available for this constant; return its
    /// unspecialized value.
    ///
    /// If the constant has an inappropriate kind (non-scalar or non-integer) or
    /// value (negative, out of range for u32), return `None`. This usually
    /// indicates an error, but only the caller has enough information to report
    /// the error helpfully: in back ends, it's a validation error, but in front
    /// ends, it may indicate ill-formed input (for example, a SPIR-V
    /// `OpArrayType` referring to an inappropriate `OpConstant`). So we return
    /// `Option` and let the caller sort things out.
    pub(crate) fn to_array_length(&self) -> Option<u32> {
        use std::convert::TryInto;
        match self.inner {
            crate::ConstantInner::Scalar { value, width: _ } => match value {
                crate::ScalarValue::Uint(value) => value.try_into().ok(),
                // Accept a signed integer size to avoid
                // requiring an explicit uint
                // literal. Type inference should make
                // this unnecessary.
                crate::ScalarValue::Sint(value) => value.try_into().ok(),
                _ => None,
            },
            // caught by type validation
            crate::ConstantInner::Composite { .. } => None,
        }
    }
}

impl crate::Binding {
    pub const fn to_built_in(&self) -> Option<crate::BuiltIn> {
        match *self {
            crate::Binding::BuiltIn(built_in) => Some(built_in),
            Self::Location { .. } => None,
        }
    }
}

//TODO: should we use an existing crate for hashable floats?
impl PartialEq for crate::ScalarValue {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::Uint(a), Self::Uint(b)) => a == b,
            (Self::Sint(a), Self::Sint(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a.to_bits() == b.to_bits(),
            (Self::Bool(a), Self::Bool(b)) => a == b,
            _ => false,
        }
    }
}
impl Eq for crate::ScalarValue {}
impl std::hash::Hash for crate::ScalarValue {
    fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
        match *self {
            Self::Sint(v) => v.hash(hasher),
            Self::Uint(v) => v.hash(hasher),
            Self::Float(v) => v.to_bits().hash(hasher),
            Self::Bool(v) => v.hash(hasher),
        }
    }
}

impl super::SwizzleComponent {
    pub const XYZW: [Self; 4] = [Self::X, Self::Y, Self::Z, Self::W];

    pub const fn index(&self) -> u32 {
        match *self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
            Self::W => 3,
        }
    }
    pub const fn from_index(idx: u32) -> Self {
        match idx {
            0 => Self::X,
            1 => Self::Y,
            2 => Self::Z,
            _ => Self::W,
        }
    }
}

impl super::ImageClass {
    pub const fn is_multisampled(self) -> bool {
        match self {
            crate::ImageClass::Sampled { multi, .. } | crate::ImageClass::Depth { multi } => multi,
            crate::ImageClass::Storage { .. } => false,
        }
    }

    pub const fn is_mipmapped(self) -> bool {
        match self {
            crate::ImageClass::Sampled { multi, .. } | crate::ImageClass::Depth { multi } => !multi,
            crate::ImageClass::Storage { .. } => false,
        }
    }
}

#[test]
fn test_matrix_size() {
    let constants = crate::Arena::new();
    assert_eq!(
        crate::TypeInner::Matrix {
            columns: crate::VectorSize::Tri,
            rows: crate::VectorSize::Tri,
            width: 4
        }
        .size(&constants),
        48,
    );
}
