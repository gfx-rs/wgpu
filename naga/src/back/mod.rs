/*!
Backend functions that export shader [`Module`](super::Module)s into binary and text formats.
*/
#![allow(dead_code)] // can be dead if none of the enabled backends need it

#[cfg(feature = "dot-out")]
pub mod dot;
#[cfg(feature = "glsl-out")]
pub mod glsl;
#[cfg(feature = "hlsl-out")]
pub mod hlsl;
#[cfg(feature = "msl-out")]
pub mod msl;
#[cfg(feature = "spv-out")]
pub mod spv;
#[cfg(feature = "wgsl-out")]
pub mod wgsl;

#[cfg(any(
    feature = "hlsl-out",
    feature = "msl-out",
    feature = "spv-out",
    feature = "glsl-out"
))]
pub mod pipeline_constants;

/// Names of vector components.
pub const COMPONENTS: &[char] = &['x', 'y', 'z', 'w'];
/// Indent for backends.
pub const INDENT: &str = "    ";
/// Prefix used for baking.
pub const BAKE_PREFIX: &str = "_e";

/// Expressions that need baking.
pub type NeedBakeExpressions = crate::FastHashSet<crate::Handle<crate::Expression>>;

/// Specifies the values of pipeline-overridable constants in the shader module.
///
/// If an `@id` attribute was specified on the declaration,
/// the key must be the pipeline constant ID as a decimal ASCII number; if not,
/// the key must be the constant's identifier name.
///
/// The value may represent any of WGSL's concrete scalar types.
pub type PipelineConstants = std::collections::HashMap<String, f64>;

/// Indentation level.
#[derive(Clone, Copy)]
pub struct Level(pub usize);

impl Level {
    const fn next(&self) -> Self {
        Level(self.0 + 1)
    }
}

impl std::fmt::Display for Level {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        (0..self.0).try_for_each(|_| formatter.write_str(INDENT))
    }
}

/// Whether we're generating an entry point or a regular function.
///
/// Backend languages often require different code for a [`Function`]
/// depending on whether it represents an [`EntryPoint`] or not.
/// Backends can pass common code one of these values to select the
/// right behavior.
///
/// These values also carry enough information to find the `Function`
/// in the [`Module`]: the `Handle` for a regular function, or the
/// index into [`Module::entry_points`] for an entry point.
///
/// [`Function`]: crate::Function
/// [`EntryPoint`]: crate::EntryPoint
/// [`Module`]: crate::Module
/// [`Module::entry_points`]: crate::Module::entry_points
pub enum FunctionType {
    /// A regular function.
    Function(crate::Handle<crate::Function>),
    /// An [`EntryPoint`], and its index in [`Module::entry_points`].
    ///
    /// [`EntryPoint`]: crate::EntryPoint
    /// [`Module::entry_points`]: crate::Module::entry_points
    EntryPoint(crate::proc::EntryPointIndex),
}

impl FunctionType {
    /// Returns true if the function is an entry point for a compute shader.
    pub fn is_compute_entry_point(&self, module: &crate::Module) -> bool {
        match *self {
            FunctionType::EntryPoint(index) => {
                module.entry_points[index as usize].stage == crate::ShaderStage::Compute
            }
            FunctionType::Function(_) => false,
        }
    }
}

/// Helper structure that stores data needed when writing the function
pub struct FunctionCtx<'a> {
    /// The current function being written
    pub ty: FunctionType,
    /// Analysis about the function
    pub info: &'a crate::valid::FunctionInfo,
    /// The expression arena of the current function being written
    pub expressions: &'a crate::Arena<crate::Expression>,
    /// Map of expressions that have associated variable names
    pub named_expressions: &'a crate::NamedExpressions,
}

impl FunctionCtx<'_> {
    /// Helper method that resolves a type of a given expression.
    pub fn resolve_type<'a>(
        &'a self,
        handle: crate::Handle<crate::Expression>,
        types: &'a crate::UniqueArena<crate::Type>,
    ) -> &'a crate::TypeInner {
        self.info[handle].ty.inner_with(types)
    }

    /// Helper method that generates a [`NameKey`](crate::proc::NameKey) for a local in the current function
    pub const fn name_key(
        &self,
        local: crate::Handle<crate::LocalVariable>,
    ) -> crate::proc::NameKey {
        match self.ty {
            FunctionType::Function(handle) => crate::proc::NameKey::FunctionLocal(handle, local),
            FunctionType::EntryPoint(idx) => crate::proc::NameKey::EntryPointLocal(idx, local),
        }
    }

    /// Helper method that generates a [`NameKey`](crate::proc::NameKey) for a function argument.
    ///
    /// # Panics
    /// - If the function arguments are less or equal to `arg`
    pub const fn argument_key(&self, arg: u32) -> crate::proc::NameKey {
        match self.ty {
            FunctionType::Function(handle) => crate::proc::NameKey::FunctionArgument(handle, arg),
            FunctionType::EntryPoint(ep_index) => {
                crate::proc::NameKey::EntryPointArgument(ep_index, arg)
            }
        }
    }

    /// Returns true if the given expression points to a fixed-function pipeline input.
    pub fn is_fixed_function_input(
        &self,
        mut expression: crate::Handle<crate::Expression>,
        module: &crate::Module,
    ) -> Option<crate::BuiltIn> {
        let ep_function = match self.ty {
            FunctionType::Function(_) => return None,
            FunctionType::EntryPoint(ep_index) => &module.entry_points[ep_index as usize].function,
        };
        let mut built_in = None;
        loop {
            match self.expressions[expression] {
                crate::Expression::FunctionArgument(arg_index) => {
                    return match ep_function.arguments[arg_index as usize].binding {
                        Some(crate::Binding::BuiltIn(bi)) => Some(bi),
                        _ => built_in,
                    };
                }
                crate::Expression::AccessIndex { base, index } => {
                    match *self.resolve_type(base, &module.types) {
                        crate::TypeInner::Struct { ref members, .. } => {
                            if let Some(crate::Binding::BuiltIn(bi)) =
                                members[index as usize].binding
                            {
                                built_in = Some(bi);
                            }
                        }
                        _ => return None,
                    }
                    expression = base;
                }
                _ => return None,
            }
        }
    }
}

impl crate::Expression {
    /// Returns the ref count, upon reaching which this expression
    /// should be considered for baking.
    ///
    /// Note: we have to cache any expressions that depend on the control flow,
    /// or otherwise they may be moved into a non-uniform control flow, accidentally.
    /// See the [module-level documentation][emit] for details.
    ///
    /// [emit]: index.html#expression-evaluation-time
    pub const fn bake_ref_count(&self) -> usize {
        match *self {
            // accesses are never cached, only loads are
            crate::Expression::Access { .. } | crate::Expression::AccessIndex { .. } => usize::MAX,
            // sampling may use the control flow, and image ops look better by themselves
            crate::Expression::ImageSample { .. } | crate::Expression::ImageLoad { .. } => 1,
            // derivatives use the control flow
            crate::Expression::Derivative { .. } => 1,
            // TODO: We need a better fix for named `Load` expressions
            // More info - https://github.com/gfx-rs/naga/pull/914
            // And https://github.com/gfx-rs/naga/issues/910
            crate::Expression::Load { .. } => 1,
            // cache expressions that are referenced multiple times
            _ => 2,
        }
    }
}

/// Helper function that returns the string corresponding to the [`BinaryOperator`](crate::BinaryOperator)
pub const fn binary_operation_str(op: crate::BinaryOperator) -> &'static str {
    use crate::BinaryOperator as Bo;
    match op {
        Bo::Add => "+",
        Bo::Subtract => "-",
        Bo::Multiply => "*",
        Bo::Divide => "/",
        Bo::Modulo => "%",
        Bo::Equal => "==",
        Bo::NotEqual => "!=",
        Bo::Less => "<",
        Bo::LessEqual => "<=",
        Bo::Greater => ">",
        Bo::GreaterEqual => ">=",
        Bo::And => "&",
        Bo::ExclusiveOr => "^",
        Bo::InclusiveOr => "|",
        Bo::LogicalAnd => "&&",
        Bo::LogicalOr => "||",
        Bo::ShiftLeft => "<<",
        Bo::ShiftRight => ">>",
    }
}

/// Helper function that returns the string corresponding to the [`VectorSize`](crate::VectorSize)
const fn vector_size_str(size: crate::VectorSize) -> &'static str {
    match size {
        crate::VectorSize::Bi => "2",
        crate::VectorSize::Tri => "3",
        crate::VectorSize::Quad => "4",
    }
}

impl crate::TypeInner {
    /// Returns true if this is a handle to a type rather than the type directly.
    pub const fn is_handle(&self) -> bool {
        match *self {
            crate::TypeInner::Image { .. } | crate::TypeInner::Sampler { .. } => true,
            _ => false,
        }
    }
}

impl crate::Statement {
    /// Returns true if the statement directly terminates the current block.
    ///
    /// Used to decide whether case blocks require a explicit `break`.
    pub const fn is_terminator(&self) -> bool {
        match *self {
            crate::Statement::Break
            | crate::Statement::Continue
            | crate::Statement::Return { .. }
            | crate::Statement::Kill => true,
            _ => false,
        }
    }
}

bitflags::bitflags! {
    /// Ray flags, for a [`RayDesc`]'s `flags` field.
    ///
    /// Note that these exactly correspond to the SPIR-V "Ray Flags" mask, and
    /// the SPIR-V backend passes them directly through to the
    /// [`OpRayQueryInitializeKHR`][op] instruction. (We have to choose something, so
    /// we might as well make one back end's life easier.)
    ///
    /// [`RayDesc`]: crate::Module::generate_ray_desc_type
    /// [op]: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpRayQueryInitializeKHR
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct RayFlag: u32 {
        const OPAQUE = 0x01;
        const NO_OPAQUE = 0x02;
        const TERMINATE_ON_FIRST_HIT = 0x04;
        const SKIP_CLOSEST_HIT_SHADER = 0x08;
        const CULL_BACK_FACING = 0x10;
        const CULL_FRONT_FACING = 0x20;
        const CULL_OPAQUE = 0x40;
        const CULL_NO_OPAQUE = 0x80;
        const SKIP_TRIANGLES = 0x100;
        const SKIP_AABBS = 0x200;
    }
}

/// The intersection test to use for ray queries.
#[repr(u32)]
pub enum RayIntersectionType {
    Triangle = 1,
    BoundingBox = 4,
}
