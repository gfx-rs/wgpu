//! Functions which export shader modules into binary and text formats.

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

#[allow(dead_code)]
const COMPONENTS: &[char] = &['x', 'y', 'z', 'w'];
#[allow(dead_code)]
const INDENT: &str = "    ";
#[allow(dead_code)]
const BAKE_PREFIX: &str = "_e";

#[derive(Clone, Copy)]
#[allow(dead_code)]
struct Level(usize);

#[allow(dead_code)]
impl Level {
    fn next(&self) -> Self {
        Level(self.0 + 1)
    }
}

#[allow(dead_code)]
impl std::fmt::Display for Level {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        (0..self.0).try_for_each(|_| formatter.write_str(INDENT))
    }
}

/// Stores the current function type (either a regular function or an entry point)
///
/// Also stores data needed to identify it (handle for a regular function or index for an entry point)
#[allow(dead_code)]
enum FunctionType {
    /// A regular function and it's handle
    Function(crate::Handle<crate::Function>),
    /// A entry point and it's index
    EntryPoint(crate::proc::EntryPointIndex),
}

/// Helper structure that stores data needed when writing the function
#[allow(dead_code)]
struct FunctionCtx<'a> {
    /// The current function being written
    ty: FunctionType,
    /// Analysis about the function
    info: &'a crate::valid::FunctionInfo,
    /// The expression arena of the current function being written
    expressions: &'a crate::Arena<crate::Expression>,
    /// Map of expressions that have associated variable names
    named_expressions: &'a crate::NamedExpressions,
}

#[allow(dead_code)]
impl<'a> FunctionCtx<'_> {
    /// Helper method that generates a [`NameKey`](crate::proc::NameKey) for a local in the current function
    fn name_key(&self, local: crate::Handle<crate::LocalVariable>) -> crate::proc::NameKey {
        match self.ty {
            FunctionType::Function(handle) => crate::proc::NameKey::FunctionLocal(handle, local),
            FunctionType::EntryPoint(idx) => crate::proc::NameKey::EntryPointLocal(idx, local),
        }
    }

    /// Helper method that generates a [`NameKey`](crate::proc::NameKey) for a function argument.
    ///
    /// # Panics
    /// - If the function arguments are less or equal to `arg`
    fn argument_key(&self, arg: u32) -> crate::proc::NameKey {
        match self.ty {
            FunctionType::Function(handle) => crate::proc::NameKey::FunctionArgument(handle, arg),
            FunctionType::EntryPoint(ep_index) => {
                crate::proc::NameKey::EntryPointArgument(ep_index, arg)
            }
        }
    }

    // Returns true if the given expression points to a fixed-function pipeline input.
    fn is_fixed_function_input(
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
                    match *self.info[base].ty.inner_with(&module.types) {
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
    /// or otherwise they may be moved into a non-uniform contol flow, accidentally.
    /// See the [module-level documentation][emit] for details.
    ///
    /// [emit]: index.html#expression-evaluation-time
    #[allow(dead_code)]
    fn bake_ref_count(&self) -> usize {
        match *self {
            // accesses are never cached, only loads are
            crate::Expression::Access { .. } | crate::Expression::AccessIndex { .. } => !0,
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
/// # Notes
/// Used by `glsl-out`, `msl-out`, `wgsl-out`, `hlsl-out`.
#[allow(dead_code)]
fn binary_operation_str(op: crate::BinaryOperator) -> &'static str {
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
/// # Notes
/// Used by `msl-out`, `wgsl-out`, `hlsl-out`.
#[allow(dead_code)]
fn vector_size_str(size: crate::VectorSize) -> &'static str {
    match size {
        crate::VectorSize::Bi => "2",
        crate::VectorSize::Tri => "3",
        crate::VectorSize::Quad => "4",
    }
}

impl crate::TypeInner {
    #[allow(unused)]
    fn is_handle(&self) -> bool {
        match *self {
            crate::TypeInner::Image { .. } | crate::TypeInner::Sampler { .. } => true,
            _ => false,
        }
    }
}

impl crate::Statement {
    /// Returns true if the statement directly terminates the current block
    ///
    /// Used to decided wether case blocks require a explicit `break`
    pub fn is_terminator(&self) -> bool {
        match *self {
            crate::Statement::Break
            | crate::Statement::Continue
            | crate::Statement::Return { .. }
            | crate::Statement::Kill => true,
            _ => false,
        }
    }
}
