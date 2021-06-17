//! Module processing functionality.

mod interpolator;
mod layouter;
mod namer;
mod terminator;
mod typifier;

pub use layouter::{Alignment, InvalidBaseType, Layouter, TypeLayout};
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
    pub fn scalar_kind(&self) -> super::ScalarKind {
        match *self {
            Self::Uint(_) => super::ScalarKind::Uint,
            Self::Sint(_) => super::ScalarKind::Sint,
            Self::Float(_) => super::ScalarKind::Float,
            Self::Bool(_) => super::ScalarKind::Bool,
        }
    }
}

pub const POINTER_SPAN: u32 = 4;

impl super::TypeInner {
    pub fn scalar_kind(&self) -> Option<super::ScalarKind> {
        match *self {
            super::TypeInner::Scalar { kind, .. } | super::TypeInner::Vector { kind, .. } => {
                Some(kind)
            }
            super::TypeInner::Matrix { .. } => Some(super::ScalarKind::Float),
            _ => None,
        }
    }

    pub fn span(&self, constants: &super::Arena<super::Constant>) -> u32 {
        match *self {
            Self::Scalar { kind: _, width } => width as u32,
            Self::Vector {
                size,
                kind: _,
                width,
            } => (size as u8 * width) as u32,
            Self::Matrix {
                columns,
                rows,
                width,
            } => (columns as u8 * rows as u8 * width) as u32,
            Self::Pointer { .. } | Self::ValuePointer { .. } => POINTER_SPAN,
            Self::Array {
                base: _,
                size,
                stride,
            } => {
                let count = match size {
                    super::ArraySize::Constant(handle) => {
                        // Bad array lengths will be caught during validation.
                        constants[handle].to_array_length().unwrap_or(1)
                    }
                    // A dynamically-sized array has to have at least one element
                    super::ArraySize::Dynamic => 1,
                };
                count * stride
            }
            Self::Struct { span, .. } => span,
            Self::Image { .. } | Self::Sampler { .. } => 0,
        }
    }
}

impl super::MathFunction {
    pub fn argument_count(&self) -> usize {
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
        }
    }
}

impl crate::Expression {
    /// Returns true if the expression is considered emitted at the start of a function.
    pub fn needs_pre_emit(&self) -> bool {
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

impl crate::SampleLevel {
    pub fn implicit_derivatives(&self) -> bool {
        match *self {
            Self::Auto | Self::Bias(_) => true,
            Self::Zero | Self::Exact(_) | Self::Gradient { .. } => false,
        }
    }
}

impl crate::Constant {
    pub fn to_array_length(&self) -> Option<u32> {
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
    pub fn to_built_in(&self) -> Option<crate::BuiltIn> {
        match *self {
            Self::BuiltIn(bi) => Some(bi),
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

    pub fn index(&self) -> u32 {
        match *self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
            Self::W => 3,
        }
    }
    pub fn from_index(idx: u32) -> Self {
        match idx {
            0 => Self::X,
            1 => Self::Y,
            2 => Self::Z,
            _ => Self::W,
        }
    }
}
