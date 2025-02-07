/*!
[`Module`](super::Module) processing functionality.
*/

mod constant_evaluator;
mod emitter;
pub mod index;
mod layouter;
mod namer;
mod terminator;
mod type_methods;
mod typifier;

pub use constant_evaluator::{
    ConstantEvaluator, ConstantEvaluatorError, ExpressionKind, ExpressionKindTracker,
};
pub use emitter::Emitter;
pub use index::{BoundsCheckPolicies, BoundsCheckPolicy, IndexableLength, IndexableLengthError};
pub use layouter::{Alignment, LayoutError, LayoutErrorInner, Layouter, TypeLayout};
pub use namer::{EntryPointIndex, NameKey, Namer};
pub use terminator::ensure_block_returns;
pub use typifier::{ResolveContext, ResolveError, TypeResolution};

impl From<super::StorageFormat> for super::Scalar {
    fn from(format: super::StorageFormat) -> Self {
        use super::{ScalarKind as Sk, StorageFormat as Sf};
        let kind = match format {
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
            Sf::Bgra8Unorm => Sk::Float,
            Sf::Rgb10a2Uint => Sk::Uint,
            Sf::Rgb10a2Unorm => Sk::Float,
            Sf::Rg11b10Ufloat => Sk::Float,
            Sf::R64Uint => Sk::Uint,
            Sf::Rg32Uint => Sk::Uint,
            Sf::Rg32Sint => Sk::Sint,
            Sf::Rg32Float => Sk::Float,
            Sf::Rgba16Uint => Sk::Uint,
            Sf::Rgba16Sint => Sk::Sint,
            Sf::Rgba16Float => Sk::Float,
            Sf::Rgba32Uint => Sk::Uint,
            Sf::Rgba32Sint => Sk::Sint,
            Sf::Rgba32Float => Sk::Float,
            Sf::R16Unorm => Sk::Float,
            Sf::R16Snorm => Sk::Float,
            Sf::Rg16Unorm => Sk::Float,
            Sf::Rg16Snorm => Sk::Float,
            Sf::Rgba16Unorm => Sk::Float,
            Sf::Rgba16Snorm => Sk::Float,
        };
        let width = match format {
            Sf::R64Uint => 8,
            _ => 4,
        };
        super::Scalar { kind, width }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HashableLiteral {
    F64(u64),
    F32(u32),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    Bool(bool),
    AbstractInt(i64),
    AbstractFloat(u64),
}

impl From<crate::Literal> for HashableLiteral {
    fn from(l: crate::Literal) -> Self {
        match l {
            crate::Literal::F64(v) => Self::F64(v.to_bits()),
            crate::Literal::F32(v) => Self::F32(v.to_bits()),
            crate::Literal::U32(v) => Self::U32(v),
            crate::Literal::I32(v) => Self::I32(v),
            crate::Literal::U64(v) => Self::U64(v),
            crate::Literal::I64(v) => Self::I64(v),
            crate::Literal::Bool(v) => Self::Bool(v),
            crate::Literal::AbstractInt(v) => Self::AbstractInt(v),
            crate::Literal::AbstractFloat(v) => Self::AbstractFloat(v.to_bits()),
        }
    }
}

impl crate::Literal {
    pub const fn new(value: u8, scalar: crate::Scalar) -> Option<Self> {
        match (value, scalar.kind, scalar.width) {
            (value, crate::ScalarKind::Float, 8) => Some(Self::F64(value as _)),
            (value, crate::ScalarKind::Float, 4) => Some(Self::F32(value as _)),
            (value, crate::ScalarKind::Uint, 4) => Some(Self::U32(value as _)),
            (value, crate::ScalarKind::Sint, 4) => Some(Self::I32(value as _)),
            (value, crate::ScalarKind::Uint, 8) => Some(Self::U64(value as _)),
            (value, crate::ScalarKind::Sint, 8) => Some(Self::I64(value as _)),
            (1, crate::ScalarKind::Bool, crate::BOOL_WIDTH) => Some(Self::Bool(true)),
            (0, crate::ScalarKind::Bool, crate::BOOL_WIDTH) => Some(Self::Bool(false)),
            _ => None,
        }
    }

    pub const fn zero(scalar: crate::Scalar) -> Option<Self> {
        Self::new(0, scalar)
    }

    pub const fn one(scalar: crate::Scalar) -> Option<Self> {
        Self::new(1, scalar)
    }

    pub const fn width(&self) -> crate::Bytes {
        match *self {
            Self::F64(_) | Self::I64(_) | Self::U64(_) => 8,
            Self::F32(_) | Self::U32(_) | Self::I32(_) => 4,
            Self::Bool(_) => crate::BOOL_WIDTH,
            Self::AbstractInt(_) | Self::AbstractFloat(_) => crate::ABSTRACT_WIDTH,
        }
    }
    pub const fn scalar(&self) -> crate::Scalar {
        match *self {
            Self::F64(_) => crate::Scalar::F64,
            Self::F32(_) => crate::Scalar::F32,
            Self::U32(_) => crate::Scalar::U32,
            Self::I32(_) => crate::Scalar::I32,
            Self::U64(_) => crate::Scalar::U64,
            Self::I64(_) => crate::Scalar::I64,
            Self::Bool(_) => crate::Scalar::BOOL,
            Self::AbstractInt(_) => crate::Scalar::ABSTRACT_INT,
            Self::AbstractFloat(_) => crate::Scalar::ABSTRACT_FLOAT,
        }
    }
    pub const fn scalar_kind(&self) -> crate::ScalarKind {
        self.scalar().kind
    }
    pub const fn ty_inner(&self) -> crate::TypeInner {
        crate::TypeInner::Scalar(self.scalar())
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
            Self::Saturate => 1,
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
            Self::Modf => 1,
            Self::Frexp => 1,
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
            Self::QuantizeToF16 => 1,
            // bits
            Self::CountTrailingZeros => 1,
            Self::CountLeadingZeros => 1,
            Self::CountOneBits => 1,
            Self::ReverseBits => 1,
            Self::ExtractBits => 3,
            Self::InsertBits => 4,
            Self::FirstTrailingBit => 1,
            Self::FirstLeadingBit => 1,
            // data packing
            Self::Pack4x8snorm => 1,
            Self::Pack4x8unorm => 1,
            Self::Pack2x16snorm => 1,
            Self::Pack2x16unorm => 1,
            Self::Pack2x16float => 1,
            Self::Pack4xI8 => 1,
            Self::Pack4xU8 => 1,
            // data unpacking
            Self::Unpack4x8snorm => 1,
            Self::Unpack4x8unorm => 1,
            Self::Unpack2x16snorm => 1,
            Self::Unpack2x16unorm => 1,
            Self::Unpack2x16float => 1,
            Self::Unpack4xI8 => 1,
            Self::Unpack4xU8 => 1,
        }
    }
}

impl crate::Expression {
    /// Returns true if the expression is considered emitted at the start of a function.
    pub const fn needs_pre_emit(&self) -> bool {
        match *self {
            Self::Literal(_)
            | Self::Constant(_)
            | Self::Override(_)
            | Self::ZeroValue(_)
            | Self::FunctionArgument(_)
            | Self::GlobalVariable(_)
            | Self::LocalVariable(_) => true,
            _ => false,
        }
    }

    /// Return true if this expression is a dynamic array/vector/matrix index,
    /// for [`Access`].
    ///
    /// This method returns true if this expression is a dynamically computed
    /// index, and as such can only be used to index matrices when they appear
    /// behind a pointer. See the documentation for [`Access`] for details.
    ///
    /// Note, this does not check the _type_ of the given expression. It's up to
    /// the caller to establish that the `Access` expression is well-typed
    /// through other means, like [`ResolveContext`].
    ///
    /// [`Access`]: crate::Expression::Access
    /// [`ResolveContext`]: crate::proc::ResolveContext
    pub const fn is_dynamic_index(&self) -> bool {
        match *self {
            Self::Literal(_) | Self::ZeroValue(_) | Self::Constant(_) => false,
            _ => true,
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

impl crate::Binding {
    pub const fn to_built_in(&self) -> Option<crate::BuiltIn> {
        match *self {
            crate::Binding::BuiltIn(built_in) => Some(built_in),
            Self::Location { .. } => None,
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

    pub const fn is_depth(self) -> bool {
        matches!(self, crate::ImageClass::Depth { .. })
    }
}

impl crate::Module {
    pub const fn to_ctx(&self) -> GlobalCtx<'_> {
        GlobalCtx {
            types: &self.types,
            constants: &self.constants,
            overrides: &self.overrides,
            global_expressions: &self.global_expressions,
        }
    }
}

#[derive(Debug)]
pub(super) enum U32EvalError {
    NonConst,
    Negative,
}

#[derive(Clone, Copy)]
pub struct GlobalCtx<'a> {
    pub types: &'a crate::UniqueArena<crate::Type>,
    pub constants: &'a crate::Arena<crate::Constant>,
    pub overrides: &'a crate::Arena<crate::Override>,
    pub global_expressions: &'a crate::Arena<crate::Expression>,
}

impl GlobalCtx<'_> {
    /// Try to evaluate the expression in `self.global_expressions` using its `handle` and return it as a `u32`.
    #[allow(dead_code)]
    pub(super) fn eval_expr_to_u32(
        &self,
        handle: crate::Handle<crate::Expression>,
    ) -> Result<u32, U32EvalError> {
        self.eval_expr_to_u32_from(handle, self.global_expressions)
    }

    /// Try to evaluate the expression in the `arena` using its `handle` and return it as a `u32`.
    pub(super) fn eval_expr_to_u32_from(
        &self,
        handle: crate::Handle<crate::Expression>,
        arena: &crate::Arena<crate::Expression>,
    ) -> Result<u32, U32EvalError> {
        match self.eval_expr_to_literal_from(handle, arena) {
            Some(crate::Literal::U32(value)) => Ok(value),
            Some(crate::Literal::I32(value)) => {
                value.try_into().map_err(|_| U32EvalError::Negative)
            }
            _ => Err(U32EvalError::NonConst),
        }
    }

    /// Try to evaluate the expression in the `arena` using its `handle` and return it as a `bool`.
    #[allow(dead_code)]
    pub(super) fn eval_expr_to_bool_from(
        &self,
        handle: crate::Handle<crate::Expression>,
        arena: &crate::Arena<crate::Expression>,
    ) -> Option<bool> {
        match self.eval_expr_to_literal_from(handle, arena) {
            Some(crate::Literal::Bool(value)) => Some(value),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn eval_expr_to_literal(
        &self,
        handle: crate::Handle<crate::Expression>,
    ) -> Option<crate::Literal> {
        self.eval_expr_to_literal_from(handle, self.global_expressions)
    }

    fn eval_expr_to_literal_from(
        &self,
        handle: crate::Handle<crate::Expression>,
        arena: &crate::Arena<crate::Expression>,
    ) -> Option<crate::Literal> {
        fn get(
            gctx: GlobalCtx,
            handle: crate::Handle<crate::Expression>,
            arena: &crate::Arena<crate::Expression>,
        ) -> Option<crate::Literal> {
            match arena[handle] {
                crate::Expression::Literal(literal) => Some(literal),
                crate::Expression::ZeroValue(ty) => match gctx.types[ty].inner {
                    crate::TypeInner::Scalar(scalar) => crate::Literal::zero(scalar),
                    _ => None,
                },
                _ => None,
            }
        }
        match arena[handle] {
            crate::Expression::Constant(c) => {
                get(*self, self.constants[c].init, self.global_expressions)
            }
            _ => get(*self, handle, arena),
        }
    }
}

/// Return an iterator over the individual components assembled by a
/// `Compose` expression.
///
/// Given `ty` and `components` from an `Expression::Compose`, return an
/// iterator over the components of the resulting value.
///
/// Normally, this would just be an iterator over `components`. However,
/// `Compose` expressions can concatenate vectors, in which case the i'th
/// value being composed is not generally the i'th element of `components`.
/// This function consults `ty` to decide if this concatenation is occurring,
/// and returns an iterator that produces the components of the result of
/// the `Compose` expression in either case.
pub fn flatten_compose<'arenas>(
    ty: crate::Handle<crate::Type>,
    components: &'arenas [crate::Handle<crate::Expression>],
    expressions: &'arenas crate::Arena<crate::Expression>,
    types: &'arenas crate::UniqueArena<crate::Type>,
) -> impl Iterator<Item = crate::Handle<crate::Expression>> + 'arenas {
    // Returning `impl Iterator` is a bit tricky. We may or may not
    // want to flatten the components, but we have to settle on a
    // single concrete type to return. This function returns a single
    // iterator chain that handles both the flattening and
    // non-flattening cases.
    let (size, is_vector) = if let crate::TypeInner::Vector { size, .. } = types[ty].inner {
        (size as usize, true)
    } else {
        (components.len(), false)
    };

    /// Flatten `Compose` expressions if `is_vector` is true.
    fn flatten_compose<'c>(
        component: &'c crate::Handle<crate::Expression>,
        is_vector: bool,
        expressions: &'c crate::Arena<crate::Expression>,
    ) -> &'c [crate::Handle<crate::Expression>] {
        if is_vector {
            if let crate::Expression::Compose {
                ty: _,
                components: ref subcomponents,
            } = expressions[*component]
            {
                return subcomponents;
            }
        }
        std::slice::from_ref(component)
    }

    /// Flatten `Splat` expressions if `is_vector` is true.
    fn flatten_splat<'c>(
        component: &'c crate::Handle<crate::Expression>,
        is_vector: bool,
        expressions: &'c crate::Arena<crate::Expression>,
    ) -> impl Iterator<Item = crate::Handle<crate::Expression>> {
        let mut expr = *component;
        let mut count = 1;
        if is_vector {
            if let crate::Expression::Splat { size, value } = expressions[expr] {
                expr = value;
                count = size as usize;
            }
        }
        std::iter::repeat(expr).take(count)
    }

    // Expressions like `vec4(vec3(vec2(6, 7), 8), 9)` require us to
    // flatten up to two levels of `Compose` expressions.
    //
    // Expressions like `vec4(vec3(1.0), 1.0)` require us to flatten
    // `Splat` expressions. Fortunately, the operand of a `Splat` must
    // be a scalar, so we can stop there.
    components
        .iter()
        .flat_map(move |component| flatten_compose(component, is_vector, expressions))
        .flat_map(move |component| flatten_compose(component, is_vector, expressions))
        .flat_map(move |component| flatten_splat(component, is_vector, expressions))
        .take(size)
}

#[test]
fn test_matrix_size() {
    let module = crate::Module::default();
    assert_eq!(
        crate::TypeInner::Matrix {
            columns: crate::VectorSize::Tri,
            rows: crate::VectorSize::Tri,
            scalar: crate::Scalar::F32,
        }
        .size(module.to_ctx()),
        48,
    );
}
