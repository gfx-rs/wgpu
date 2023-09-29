/*!
[`Module`](super::Module) processing functionality.
*/

mod constant_evaluator;
mod emitter;
pub mod index;
mod layouter;
mod namer;
mod terminator;
mod typifier;

pub use constant_evaluator::{
    ConstantEvaluator, ConstantEvaluatorError, ExpressionConstnessTracker,
};
pub use emitter::Emitter;
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
            Sf::Bgra8Unorm => Sk::Float,
            Sf::Rgb10a2Uint => Sk::Uint,
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
            Sf::R16Unorm => Sk::Float,
            Sf::R16Snorm => Sk::Float,
            Sf::Rg16Unorm => Sk::Float,
            Sf::Rg16Snorm => Sk::Float,
            Sf::Rgba16Unorm => Sk::Float,
            Sf::Rgba16Snorm => Sk::Float,
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

impl PartialEq for crate::Literal {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::F64(a), Self::F64(b)) => a.to_bits() == b.to_bits(),
            (Self::F32(a), Self::F32(b)) => a.to_bits() == b.to_bits(),
            (Self::U32(a), Self::U32(b)) => a == b,
            (Self::I32(a), Self::I32(b)) => a == b,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            _ => false,
        }
    }
}
impl Eq for crate::Literal {}
impl std::hash::Hash for crate::Literal {
    fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
        match *self {
            Self::F64(v) => {
                hasher.write_u8(0);
                v.to_bits().hash(hasher);
            }
            Self::F32(v) => {
                hasher.write_u8(1);
                v.to_bits().hash(hasher);
            }
            Self::U32(v) => {
                hasher.write_u8(2);
                v.hash(hasher);
            }
            Self::I32(v) => {
                hasher.write_u8(3);
                v.hash(hasher);
            }
            Self::Bool(v) => {
                hasher.write_u8(4);
                v.hash(hasher);
            }
        }
    }
}

impl crate::Literal {
    pub const fn new(value: u8, kind: crate::ScalarKind, width: crate::Bytes) -> Option<Self> {
        match (value, kind, width) {
            (value, crate::ScalarKind::Float, 8) => Some(Self::F64(value as _)),
            (value, crate::ScalarKind::Float, 4) => Some(Self::F32(value as _)),
            (value, crate::ScalarKind::Uint, 4) => Some(Self::U32(value as _)),
            (value, crate::ScalarKind::Sint, 4) => Some(Self::I32(value as _)),
            (1, crate::ScalarKind::Bool, 4) => Some(Self::Bool(true)),
            (0, crate::ScalarKind::Bool, 4) => Some(Self::Bool(false)),
            _ => None,
        }
    }

    pub const fn zero(kind: crate::ScalarKind, width: crate::Bytes) -> Option<Self> {
        Self::new(0, kind, width)
    }

    pub const fn one(kind: crate::ScalarKind, width: crate::Bytes) -> Option<Self> {
        Self::new(1, kind, width)
    }

    pub const fn width(&self) -> crate::Bytes {
        match *self {
            Self::F64(_) => 8,
            Self::F32(_) | Self::U32(_) | Self::I32(_) => 4,
            Self::Bool(_) => 1,
        }
    }
    pub const fn scalar_kind(&self) -> crate::ScalarKind {
        match *self {
            Self::F64(_) | Self::F32(_) => crate::ScalarKind::Float,
            Self::U32(_) => crate::ScalarKind::Uint,
            Self::I32(_) => crate::ScalarKind::Sint,
            Self::Bool(_) => crate::ScalarKind::Bool,
        }
    }
    pub const fn ty_inner(&self) -> crate::TypeInner {
        crate::TypeInner::Scalar {
            kind: self.scalar_kind(),
            width: self.width(),
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

    pub const fn scalar_width(&self) -> Option<u8> {
        // Multiply by 8 to get the bit width
        match *self {
            super::TypeInner::Scalar { width, .. } | super::TypeInner::Vector { width, .. } => {
                Some(width * 8)
            }
            super::TypeInner::Matrix { width, .. } => Some(width * 8),
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
    pub fn size(&self, _gctx: GlobalCtx) -> u32 {
        match *self {
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
                    super::ArraySize::Constant(count) => count.get(),
                    // A dynamically-sized array has to have at least one element
                    super::ArraySize::Dynamic => 1,
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
            Self::Vector { kind, width, .. } => {
                TypeResolution::Value(crate::TypeInner::Scalar { kind, width })
            }
            Self::Matrix { rows, width, .. } => TypeResolution::Value(crate::TypeInner::Vector {
                size: rows,
                kind: crate::ScalarKind::Float,
                width,
            }),
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
            // bits
            Self::CountTrailingZeros => 1,
            Self::CountLeadingZeros => 1,
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
            Self::Literal(_)
            | Self::Constant(_)
            | Self::ZeroValue(_)
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
            !matches!(constant.r#override, crate::Override::None)
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
}

impl crate::Module {
    pub const fn to_ctx(&self) -> GlobalCtx<'_> {
        GlobalCtx {
            types: &self.types,
            constants: &self.constants,
            const_expressions: &self.const_expressions,
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
    pub const_expressions: &'a crate::Arena<crate::Expression>,
}

impl GlobalCtx<'_> {
    /// Try to evaluate the expression in `self.const_expressions` using its `handle` and return it as a `u32`.
    #[allow(dead_code)]
    pub(super) fn eval_expr_to_u32(
        &self,
        handle: crate::Handle<crate::Expression>,
    ) -> Result<u32, U32EvalError> {
        self.eval_expr_to_u32_from(handle, self.const_expressions)
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

    pub(crate) fn eval_expr_to_literal(
        &self,
        handle: crate::Handle<crate::Expression>,
    ) -> Option<crate::Literal> {
        self.eval_expr_to_literal_from(handle, self.const_expressions)
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
                    crate::TypeInner::Scalar { kind, width } => crate::Literal::zero(kind, width),
                    _ => None,
                },
                _ => None,
            }
        }
        match arena[handle] {
            crate::Expression::Constant(c) => {
                get(*self, self.constants[c].init, self.const_expressions)
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
/// This function consults `ty` to decide if this concatenation is occuring,
/// and returns an iterator that produces the components of the result of
/// the `Compose` expression in either case.
pub fn flatten_compose<'arenas>(
    ty: crate::Handle<crate::Type>,
    components: &'arenas [crate::Handle<crate::Expression>],
    expressions: &'arenas crate::Arena<crate::Expression>,
    types: &'arenas crate::UniqueArena<crate::Type>,
) -> impl Iterator<Item = crate::Handle<crate::Expression>> + 'arenas {
    // Returning `impl Iterator` is a bit tricky. We may or may not want to
    // flatten the components, but we have to settle on a single concrete
    // type to return. The below is a single iterator chain that handles
    // both the flattening and non-flattening cases.
    let (size, is_vector) = if let crate::TypeInner::Vector { size, .. } = types[ty].inner {
        (size as usize, true)
    } else {
        (components.len(), false)
    };

    fn flattener<'c>(
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

    // Expressions like `vec4(vec3(vec2(6, 7), 8), 9)` require us to flatten
    // two levels.
    components
        .iter()
        .flat_map(move |component| flattener(component, is_vector, expressions))
        .flat_map(move |component| flattener(component, is_vector, expressions))
        .take(size)
        .cloned()
}

#[test]
fn test_matrix_size() {
    let module = crate::Module::default();
    assert_eq!(
        crate::TypeInner::Matrix {
            columns: crate::VectorSize::Tri,
            rows: crate::VectorSize::Tri,
            width: 4
        }
        .size(module.to_ctx()),
        48,
    );
}
