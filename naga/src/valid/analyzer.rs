/*! Module analyzer.

Figures out the following properties:
  - control flow uniformity
  - texture/sampler pairs
  - expression reference counts
!*/

use super::{ExpressionError, FunctionError, ModuleInfo, ShaderStages, ValidationFlags};
use crate::span::{AddSpan as _, WithSpan};
use crate::{
    arena::{Arena, Handle},
    proc::{ResolveContext, TypeResolution},
};
use std::ops;

pub type NonUniformResult = Option<Handle<crate::Expression>>;

// Remove this once we update our uniformity analysis and
// add support for the `derivative_uniformity` diagnostic
const DISABLE_UNIFORMITY_REQ_FOR_FRAGMENT_STAGE: bool = true;

bitflags::bitflags! {
    /// Kinds of expressions that require uniform control flow.
    #[cfg_attr(feature = "serialize", derive(serde::Serialize))]
    #[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct UniformityRequirements: u8 {
        const WORK_GROUP_BARRIER = 0x1;
        const DERIVATIVE = if DISABLE_UNIFORMITY_REQ_FOR_FRAGMENT_STAGE { 0 } else { 0x2 };
        const IMPLICIT_LEVEL = if DISABLE_UNIFORMITY_REQ_FOR_FRAGMENT_STAGE { 0 } else { 0x4 };
    }
}

/// Uniform control flow characteristics.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(test, derive(PartialEq))]
pub struct Uniformity {
    /// A child expression with non-uniform result.
    ///
    /// This means, when the relevant invocations are scheduled on a compute unit,
    /// they have to use vector registers to store an individual value
    /// per invocation.
    ///
    /// Whenever the control flow is conditioned on such value,
    /// the hardware needs to keep track of the mask of invocations,
    /// and process all branches of the control flow.
    ///
    /// Any operations that depend on non-uniform results also produce non-uniform.
    pub non_uniform_result: NonUniformResult,
    /// If this expression requires uniform control flow, store the reason here.
    pub requirements: UniformityRequirements,
}

impl Uniformity {
    const fn new() -> Self {
        Uniformity {
            non_uniform_result: None,
            requirements: UniformityRequirements::empty(),
        }
    }
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct ExitFlags: u8 {
        /// Control flow may return from the function, which makes all the
        /// subsequent statements within the current function (only!)
        /// to be executed in a non-uniform control flow.
        const MAY_RETURN = 0x1;
        /// Control flow may be killed. Anything after `Statement::Kill` is
        /// considered inside non-uniform context.
        const MAY_KILL = 0x2;
    }
}

/// Uniformity characteristics of a function.
#[cfg_attr(test, derive(Debug, PartialEq))]
struct FunctionUniformity {
    result: Uniformity,
    exit: ExitFlags,
}

impl ops::BitOr for FunctionUniformity {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        FunctionUniformity {
            result: Uniformity {
                non_uniform_result: self
                    .result
                    .non_uniform_result
                    .or(other.result.non_uniform_result),
                requirements: self.result.requirements | other.result.requirements,
            },
            exit: self.exit | other.exit,
        }
    }
}

impl FunctionUniformity {
    const fn new() -> Self {
        FunctionUniformity {
            result: Uniformity::new(),
            exit: ExitFlags::empty(),
        }
    }

    /// Returns a disruptor based on the stored exit flags, if any.
    const fn exit_disruptor(&self) -> Option<UniformityDisruptor> {
        if self.exit.contains(ExitFlags::MAY_RETURN) {
            Some(UniformityDisruptor::Return)
        } else if self.exit.contains(ExitFlags::MAY_KILL) {
            Some(UniformityDisruptor::Discard)
        } else {
            None
        }
    }
}

bitflags::bitflags! {
    /// Indicates how a global variable is used.
    #[cfg_attr(feature = "serialize", derive(serde::Serialize))]
    #[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct GlobalUse: u8 {
        /// Data will be read from the variable.
        const READ = 0x1;
        /// Data will be written to the variable.
        const WRITE = 0x2;
        /// The information about the data is queried.
        const QUERY = 0x4;
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct SamplingKey {
    pub image: Handle<crate::GlobalVariable>,
    pub sampler: Handle<crate::GlobalVariable>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct ExpressionInfo {
    pub uniformity: Uniformity,
    pub ref_count: usize,
    assignable_global: Option<Handle<crate::GlobalVariable>>,
    pub ty: TypeResolution,
}

impl ExpressionInfo {
    const fn new() -> Self {
        ExpressionInfo {
            uniformity: Uniformity::new(),
            ref_count: 0,
            assignable_global: None,
            // this doesn't matter at this point, will be overwritten
            ty: TypeResolution::Value(crate::TypeInner::Scalar {
                kind: crate::ScalarKind::Bool,
                width: 0,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
enum GlobalOrArgument {
    Global(Handle<crate::GlobalVariable>),
    Argument(u32),
}

impl GlobalOrArgument {
    fn from_expression(
        expression_arena: &Arena<crate::Expression>,
        expression: Handle<crate::Expression>,
    ) -> Result<GlobalOrArgument, ExpressionError> {
        Ok(match expression_arena[expression] {
            crate::Expression::GlobalVariable(var) => GlobalOrArgument::Global(var),
            crate::Expression::FunctionArgument(i) => GlobalOrArgument::Argument(i),
            crate::Expression::Access { base, .. }
            | crate::Expression::AccessIndex { base, .. } => match expression_arena[base] {
                crate::Expression::GlobalVariable(var) => GlobalOrArgument::Global(var),
                _ => return Err(ExpressionError::ExpectedGlobalOrArgument),
            },
            _ => return Err(ExpressionError::ExpectedGlobalOrArgument),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
struct Sampling {
    image: GlobalOrArgument,
    sampler: GlobalOrArgument,
}

#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct FunctionInfo {
    /// Validation flags.
    #[allow(dead_code)]
    flags: ValidationFlags,
    /// Set of shader stages where calling this function is valid.
    pub available_stages: ShaderStages,
    /// Uniformity characteristics.
    pub uniformity: Uniformity,
    /// Function may kill the invocation.
    pub may_kill: bool,

    /// All pairs of (texture, sampler) globals that may be used together in
    /// sampling operations by this function and its callees. This includes
    /// pairings that arise when this function passes textures and samplers as
    /// arguments to its callees.
    ///
    /// This table does not include uses of textures and samplers passed as
    /// arguments to this function itself, since we do not know which globals
    /// those will be. However, this table *is* exhaustive when computed for an
    /// entry point function: entry points never receive textures or samplers as
    /// arguments, so all an entry point's sampling can be reported in terms of
    /// globals.
    ///
    /// The GLSL back end uses this table to construct reflection info that
    /// clients need to construct texture-combined sampler values.
    pub sampling_set: crate::FastHashSet<SamplingKey>,

    /// How this function and its callees use this module's globals.
    ///
    /// This is indexed by `Handle<GlobalVariable>` indices. However,
    /// `FunctionInfo` implements `std::ops::Index<Handle<GlobalVariable>>`,
    /// so you can simply index this struct with a global handle to retrieve
    /// its usage information.
    global_uses: Box<[GlobalUse]>,

    /// Information about each expression in this function's body.
    ///
    /// This is indexed by `Handle<Expression>` indices. However, `FunctionInfo`
    /// implements `std::ops::Index<Handle<Expression>>`, so you can simply
    /// index this struct with an expression handle to retrieve its
    /// `ExpressionInfo`.
    expressions: Box<[ExpressionInfo]>,

    /// All (texture, sampler) pairs that may be used together in sampling
    /// operations by this function and its callees, whether they are accessed
    /// as globals or passed as arguments.
    ///
    /// Participants are represented by [`GlobalVariable`] handles whenever
    /// possible, and otherwise by indices of this function's arguments.
    ///
    /// When analyzing a function call, we combine this data about the callee
    /// with the actual arguments being passed to produce the callers' own
    /// `sampling_set` and `sampling` tables.
    ///
    /// [`GlobalVariable`]: crate::GlobalVariable
    sampling: crate::FastHashSet<Sampling>,

    /// Indicates that the function is using dual source blending.
    pub dual_source_blending: bool,
}

impl FunctionInfo {
    pub const fn global_variable_count(&self) -> usize {
        self.global_uses.len()
    }
    pub const fn expression_count(&self) -> usize {
        self.expressions.len()
    }
    pub fn dominates_global_use(&self, other: &Self) -> bool {
        for (self_global_uses, other_global_uses) in
            self.global_uses.iter().zip(other.global_uses.iter())
        {
            if !self_global_uses.contains(*other_global_uses) {
                return false;
            }
        }
        true
    }
}

impl ops::Index<Handle<crate::GlobalVariable>> for FunctionInfo {
    type Output = GlobalUse;
    fn index(&self, handle: Handle<crate::GlobalVariable>) -> &GlobalUse {
        &self.global_uses[handle.index()]
    }
}

impl ops::Index<Handle<crate::Expression>> for FunctionInfo {
    type Output = ExpressionInfo;
    fn index(&self, handle: Handle<crate::Expression>) -> &ExpressionInfo {
        &self.expressions[handle.index()]
    }
}

/// Disruptor of the uniform control flow.
#[derive(Clone, Copy, Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum UniformityDisruptor {
    #[error("Expression {0:?} produced non-uniform result, and control flow depends on it")]
    Expression(Handle<crate::Expression>),
    #[error("There is a Return earlier in the control flow of the function")]
    Return,
    #[error("There is a Discard earlier in the entry point across all called functions")]
    Discard,
}

impl FunctionInfo {
    /// Adds a value-type reference to an expression.
    #[must_use]
    fn add_ref_impl(
        &mut self,
        handle: Handle<crate::Expression>,
        global_use: GlobalUse,
    ) -> NonUniformResult {
        let info = &mut self.expressions[handle.index()];
        info.ref_count += 1;
        // mark the used global as read
        if let Some(global) = info.assignable_global {
            self.global_uses[global.index()] |= global_use;
        }
        info.uniformity.non_uniform_result
    }

    /// Adds a value-type reference to an expression.
    #[must_use]
    fn add_ref(&mut self, handle: Handle<crate::Expression>) -> NonUniformResult {
        self.add_ref_impl(handle, GlobalUse::READ)
    }

    /// Adds a potentially assignable reference to an expression.
    /// These are destinations for `Store` and `ImageStore` statements,
    /// which can transit through `Access` and `AccessIndex`.
    #[must_use]
    fn add_assignable_ref(
        &mut self,
        handle: Handle<crate::Expression>,
        assignable_global: &mut Option<Handle<crate::GlobalVariable>>,
    ) -> NonUniformResult {
        let info = &mut self.expressions[handle.index()];
        info.ref_count += 1;
        // propagate the assignable global up the chain, till it either hits
        // a value-type expression, or the assignment statement.
        if let Some(global) = info.assignable_global {
            if let Some(_old) = assignable_global.replace(global) {
                unreachable!()
            }
        }
        info.uniformity.non_uniform_result
    }

    /// Inherit information from a called function.
    fn process_call(
        &mut self,
        callee: &Self,
        arguments: &[Handle<crate::Expression>],
        expression_arena: &Arena<crate::Expression>,
    ) -> Result<FunctionUniformity, WithSpan<FunctionError>> {
        self.sampling_set
            .extend(callee.sampling_set.iter().cloned());
        for sampling in callee.sampling.iter() {
            // If the callee was passed the texture or sampler as an argument,
            // we may now be able to determine which globals those referred to.
            let image_storage = match sampling.image {
                GlobalOrArgument::Global(var) => GlobalOrArgument::Global(var),
                GlobalOrArgument::Argument(i) => {
                    let handle = arguments[i as usize];
                    GlobalOrArgument::from_expression(expression_arena, handle).map_err(
                        |source| {
                            FunctionError::Expression { handle, source }
                                .with_span_handle(handle, expression_arena)
                        },
                    )?
                }
            };

            let sampler_storage = match sampling.sampler {
                GlobalOrArgument::Global(var) => GlobalOrArgument::Global(var),
                GlobalOrArgument::Argument(i) => {
                    let handle = arguments[i as usize];
                    GlobalOrArgument::from_expression(expression_arena, handle).map_err(
                        |source| {
                            FunctionError::Expression { handle, source }
                                .with_span_handle(handle, expression_arena)
                        },
                    )?
                }
            };

            // If we've managed to pin both the image and sampler down to
            // specific globals, record that in our `sampling_set`. Otherwise,
            // record as much as we do know in our own `sampling` table, for our
            // callers to sort out.
            match (image_storage, sampler_storage) {
                (GlobalOrArgument::Global(image), GlobalOrArgument::Global(sampler)) => {
                    self.sampling_set.insert(SamplingKey { image, sampler });
                }
                (image, sampler) => {
                    self.sampling.insert(Sampling { image, sampler });
                }
            }
        }

        // Inherit global use from our callees.
        for (mine, other) in self.global_uses.iter_mut().zip(callee.global_uses.iter()) {
            *mine |= *other;
        }

        Ok(FunctionUniformity {
            result: callee.uniformity.clone(),
            exit: if callee.may_kill {
                ExitFlags::MAY_KILL
            } else {
                ExitFlags::empty()
            },
        })
    }

    /// Compute the [`ExpressionInfo`] for `handle`.
    ///
    /// Replace the dummy entry in [`self.expressions`] for `handle`
    /// with a real `ExpressionInfo` value describing that expression.
    ///
    /// This function is called as part of a forward sweep through the
    /// arena, so we can assume that all earlier expressions in the
    /// arena already have valid info. Since expressions only depend
    /// on earlier expressions, this includes all our subexpressions.
    ///
    /// Adjust the reference counts on all expressions we use.
    ///
    /// Also populate the [`sampling_set`], [`sampling`] and
    /// [`global_uses`] fields of `self`.
    ///
    /// [`self.expressions`]: FunctionInfo::expressions
    /// [`sampling_set`]: FunctionInfo::sampling_set
    /// [`sampling`]: FunctionInfo::sampling
    /// [`global_uses`]: FunctionInfo::global_uses
    #[allow(clippy::or_fun_call)]
    fn process_expression(
        &mut self,
        handle: Handle<crate::Expression>,
        expression_arena: &Arena<crate::Expression>,
        other_functions: &[FunctionInfo],
        resolve_context: &ResolveContext,
        capabilities: super::Capabilities,
    ) -> Result<(), ExpressionError> {
        use crate::{Expression as E, SampleLevel as Sl};

        let expression = &expression_arena[handle];
        let mut assignable_global = None;
        let uniformity = match *expression {
            E::Access { base, index } => {
                let base_ty = self[base].ty.inner_with(resolve_context.types);

                // build up the caps needed if this is indexed non-uniformly
                let mut needed_caps = super::Capabilities::empty();
                let is_binding_array = match *base_ty {
                    crate::TypeInner::BindingArray {
                        base: array_element_ty_handle,
                        ..
                    } => {
                        // these are nasty aliases, but these idents are too long and break rustfmt
                        let ub_st = super::Capabilities::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING;
                        let st_sb = super::Capabilities::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
                        let sampler = super::Capabilities::SAMPLER_NON_UNIFORM_INDEXING;

                        // We're a binding array, so lets use the type of _what_ we are array of to determine if we can non-uniformly index it.
                        let array_element_ty =
                            &resolve_context.types[array_element_ty_handle].inner;

                        needed_caps |= match *array_element_ty {
                            // If we're an image, use the appropriate limit.
                            crate::TypeInner::Image { class, .. } => match class {
                                crate::ImageClass::Storage { .. } => ub_st,
                                _ => st_sb,
                            },
                            crate::TypeInner::Sampler { .. } => sampler,
                            // If we're anything but an image, assume we're a buffer and use the address space.
                            _ => {
                                if let E::GlobalVariable(global_handle) = expression_arena[base] {
                                    let global = &resolve_context.global_vars[global_handle];
                                    match global.space {
                                        crate::AddressSpace::Uniform => ub_st,
                                        crate::AddressSpace::Storage { .. } => st_sb,
                                        _ => unreachable!(),
                                    }
                                } else {
                                    unreachable!()
                                }
                            }
                        };

                        true
                    }
                    _ => false,
                };

                if self[index].uniformity.non_uniform_result.is_some()
                    && !capabilities.contains(needed_caps)
                    && is_binding_array
                {
                    return Err(ExpressionError::MissingCapabilities(needed_caps));
                }

                Uniformity {
                    non_uniform_result: self
                        .add_assignable_ref(base, &mut assignable_global)
                        .or(self.add_ref(index)),
                    requirements: UniformityRequirements::empty(),
                }
            }
            E::AccessIndex { base, .. } => Uniformity {
                non_uniform_result: self.add_assignable_ref(base, &mut assignable_global),
                requirements: UniformityRequirements::empty(),
            },
            // always uniform
            E::Splat { size: _, value } => Uniformity {
                non_uniform_result: self.add_ref(value),
                requirements: UniformityRequirements::empty(),
            },
            E::Swizzle { vector, .. } => Uniformity {
                non_uniform_result: self.add_ref(vector),
                requirements: UniformityRequirements::empty(),
            },
            E::Literal(_) | E::Constant(_) | E::ZeroValue(_) => Uniformity::new(),
            E::Compose { ref components, .. } => {
                let non_uniform_result = components
                    .iter()
                    .fold(None, |nur, &comp| nur.or(self.add_ref(comp)));
                Uniformity {
                    non_uniform_result,
                    requirements: UniformityRequirements::empty(),
                }
            }
            // depends on the builtin or interpolation
            E::FunctionArgument(index) => {
                let arg = &resolve_context.arguments[index as usize];
                let uniform = match arg.binding {
                    Some(crate::Binding::BuiltIn(built_in)) => match built_in {
                        // per-polygon built-ins are uniform
                        crate::BuiltIn::FrontFacing
                        // per-work-group built-ins are uniform
                        | crate::BuiltIn::WorkGroupId
                        | crate::BuiltIn::WorkGroupSize
                        | crate::BuiltIn::NumWorkGroups => true,
                        _ => false,
                    },
                    // only flat inputs are uniform
                    Some(crate::Binding::Location {
                        interpolation: Some(crate::Interpolation::Flat),
                        ..
                    }) => true,
                    _ => false,
                };
                Uniformity {
                    non_uniform_result: if uniform { None } else { Some(handle) },
                    requirements: UniformityRequirements::empty(),
                }
            }
            // depends on the address space
            E::GlobalVariable(gh) => {
                use crate::AddressSpace as As;
                assignable_global = Some(gh);
                let var = &resolve_context.global_vars[gh];
                let uniform = match var.space {
                    // local data is non-uniform
                    As::Function | As::Private => false,
                    // workgroup memory is exclusively accessed by the group
                    As::WorkGroup => true,
                    // uniform data
                    As::Uniform | As::PushConstant => true,
                    // storage data is only uniform when read-only
                    As::Storage { access } => !access.contains(crate::StorageAccess::STORE),
                    As::Handle => false,
                };
                Uniformity {
                    non_uniform_result: if uniform { None } else { Some(handle) },
                    requirements: UniformityRequirements::empty(),
                }
            }
            E::LocalVariable(_) => Uniformity {
                non_uniform_result: Some(handle),
                requirements: UniformityRequirements::empty(),
            },
            E::Load { pointer } => Uniformity {
                non_uniform_result: self.add_ref(pointer),
                requirements: UniformityRequirements::empty(),
            },
            E::ImageSample {
                image,
                sampler,
                gather: _,
                coordinate,
                array_index,
                offset: _,
                level,
                depth_ref,
            } => {
                let image_storage = GlobalOrArgument::from_expression(expression_arena, image)?;
                let sampler_storage = GlobalOrArgument::from_expression(expression_arena, sampler)?;

                match (image_storage, sampler_storage) {
                    (GlobalOrArgument::Global(image), GlobalOrArgument::Global(sampler)) => {
                        self.sampling_set.insert(SamplingKey { image, sampler });
                    }
                    _ => {
                        self.sampling.insert(Sampling {
                            image: image_storage,
                            sampler: sampler_storage,
                        });
                    }
                }

                // "nur" == "Non-Uniform Result"
                let array_nur = array_index.and_then(|h| self.add_ref(h));
                let level_nur = match level {
                    Sl::Auto | Sl::Zero => None,
                    Sl::Exact(h) | Sl::Bias(h) => self.add_ref(h),
                    Sl::Gradient { x, y } => self.add_ref(x).or(self.add_ref(y)),
                };
                let dref_nur = depth_ref.and_then(|h| self.add_ref(h));
                Uniformity {
                    non_uniform_result: self
                        .add_ref(image)
                        .or(self.add_ref(sampler))
                        .or(self.add_ref(coordinate))
                        .or(array_nur)
                        .or(level_nur)
                        .or(dref_nur),
                    requirements: if level.implicit_derivatives() {
                        UniformityRequirements::IMPLICIT_LEVEL
                    } else {
                        UniformityRequirements::empty()
                    },
                }
            }
            E::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => {
                let array_nur = array_index.and_then(|h| self.add_ref(h));
                let sample_nur = sample.and_then(|h| self.add_ref(h));
                let level_nur = level.and_then(|h| self.add_ref(h));
                Uniformity {
                    non_uniform_result: self
                        .add_ref(image)
                        .or(self.add_ref(coordinate))
                        .or(array_nur)
                        .or(sample_nur)
                        .or(level_nur),
                    requirements: UniformityRequirements::empty(),
                }
            }
            E::ImageQuery { image, query } => {
                let query_nur = match query {
                    crate::ImageQuery::Size { level: Some(h) } => self.add_ref(h),
                    _ => None,
                };
                Uniformity {
                    non_uniform_result: self.add_ref_impl(image, GlobalUse::QUERY).or(query_nur),
                    requirements: UniformityRequirements::empty(),
                }
            }
            E::Unary { expr, .. } => Uniformity {
                non_uniform_result: self.add_ref(expr),
                requirements: UniformityRequirements::empty(),
            },
            E::Binary { left, right, .. } => Uniformity {
                non_uniform_result: self.add_ref(left).or(self.add_ref(right)),
                requirements: UniformityRequirements::empty(),
            },
            E::Select {
                condition,
                accept,
                reject,
            } => Uniformity {
                non_uniform_result: self
                    .add_ref(condition)
                    .or(self.add_ref(accept))
                    .or(self.add_ref(reject)),
                requirements: UniformityRequirements::empty(),
            },
            // explicit derivatives require uniform
            E::Derivative { expr, .. } => Uniformity {
                //Note: taking a derivative of a uniform doesn't make it non-uniform
                non_uniform_result: self.add_ref(expr),
                requirements: UniformityRequirements::DERIVATIVE,
            },
            E::Relational { argument, .. } => Uniformity {
                non_uniform_result: self.add_ref(argument),
                requirements: UniformityRequirements::empty(),
            },
            E::Math {
                fun: _,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                let arg1_nur = arg1.and_then(|h| self.add_ref(h));
                let arg2_nur = arg2.and_then(|h| self.add_ref(h));
                let arg3_nur = arg3.and_then(|h| self.add_ref(h));
                Uniformity {
                    non_uniform_result: self.add_ref(arg).or(arg1_nur).or(arg2_nur).or(arg3_nur),
                    requirements: UniformityRequirements::empty(),
                }
            }
            E::As { expr, .. } => Uniformity {
                non_uniform_result: self.add_ref(expr),
                requirements: UniformityRequirements::empty(),
            },
            E::CallResult(function) => other_functions[function.index()].uniformity.clone(),
            E::AtomicResult { .. } | E::RayQueryProceedResult => Uniformity {
                non_uniform_result: Some(handle),
                requirements: UniformityRequirements::empty(),
            },
            E::WorkGroupUniformLoadResult { .. } => Uniformity {
                // The result of WorkGroupUniformLoad is always uniform by definition
                non_uniform_result: None,
                // The call is what cares about uniformity, not the expression
                // This expression is never emitted, so this requirement should never be used anyway?
                requirements: UniformityRequirements::empty(),
            },
            E::ArrayLength(expr) => Uniformity {
                non_uniform_result: self.add_ref_impl(expr, GlobalUse::QUERY),
                requirements: UniformityRequirements::empty(),
            },
            E::RayQueryGetIntersection {
                query,
                committed: _,
            } => Uniformity {
                non_uniform_result: self.add_ref(query),
                requirements: UniformityRequirements::empty(),
            },
            E::SubgroupBallotResult => Uniformity {
                non_uniform_result: None, // FIXME
                requirements: UniformityRequirements::empty(),
            },
            E::SubgroupOperationResult { ty } => Uniformity {
                non_uniform_result: None, // FIXME
                requirements: UniformityRequirements::empty(),
            },
        };

        let ty = resolve_context.resolve(expression, |h| Ok(&self[h].ty))?;
        self.expressions[handle.index()] = ExpressionInfo {
            uniformity,
            ref_count: 0,
            assignable_global,
            ty,
        };
        Ok(())
    }

    /// Analyzes the uniformity requirements of a block (as a sequence of statements).
    /// Returns the uniformity characteristics at the *function* level, i.e.
    /// whether or not the function requires to be called in uniform control flow,
    /// and whether the produced result is not disrupting the control flow.
    ///
    /// The parent control flow is uniform if `disruptor.is_none()`.
    ///
    /// Returns a `NonUniformControlFlow` error if any of the expressions in the block
    /// require uniformity, but the current flow is non-uniform.
    #[allow(clippy::or_fun_call)]
    fn process_block(
        &mut self,
        statements: &crate::Block,
        other_functions: &[FunctionInfo],
        mut disruptor: Option<UniformityDisruptor>,
        expression_arena: &Arena<crate::Expression>,
    ) -> Result<FunctionUniformity, WithSpan<FunctionError>> {
        use crate::Statement as S;

        let mut combined_uniformity = FunctionUniformity::new();
        for statement in statements {
            let uniformity = match *statement {
                S::Emit(ref range) => {
                    let mut requirements = UniformityRequirements::empty();
                    for expr in range.clone() {
                        let req = self.expressions[expr.index()].uniformity.requirements;
                        #[cfg(feature = "validate")]
                        if self
                            .flags
                            .contains(super::ValidationFlags::CONTROL_FLOW_UNIFORMITY)
                            && !req.is_empty()
                        {
                            if let Some(cause) = disruptor {
                                return Err(FunctionError::NonUniformControlFlow(req, expr, cause)
                                    .with_span_handle(expr, expression_arena));
                            }
                        }
                        requirements |= req;
                    }
                    FunctionUniformity {
                        result: Uniformity {
                            non_uniform_result: None,
                            requirements,
                        },
                        exit: ExitFlags::empty(),
                    }
                }
                S::Break | S::Continue => FunctionUniformity::new(),
                S::Kill => FunctionUniformity {
                    result: Uniformity::new(),
                    exit: if disruptor.is_some() {
                        ExitFlags::MAY_KILL
                    } else {
                        ExitFlags::empty()
                    },
                },
                S::Barrier(_) => FunctionUniformity {
                    result: Uniformity {
                        non_uniform_result: None,
                        requirements: UniformityRequirements::WORK_GROUP_BARRIER,
                    },
                    exit: ExitFlags::empty(),
                },
                S::WorkGroupUniformLoad { pointer, .. } => {
                    let _condition_nur = self.add_ref(pointer);

                    // Don't check that this call occurs in uniform control flow until Naga implements WGSL's standard
                    // uniformity analysis (https://github.com/gfx-rs/naga/issues/1744).
                    // The uniformity analysis Naga uses now is less accurate than the one in the WGSL standard,
                    // causing Naga to reject correct uses of `workgroupUniformLoad` in some interesting programs.

                    /* #[cfg(feature = "validate")]
                    if self
                        .flags
                        .contains(super::ValidationFlags::CONTROL_FLOW_UNIFORMITY)
                    {
                        let condition_nur = self.add_ref(pointer);
                        let this_disruptor =
                            disruptor.or(condition_nur.map(UniformityDisruptor::Expression));
                        if let Some(cause) = this_disruptor {
                            return Err(FunctionError::NonUniformWorkgroupUniformLoad(cause)
                                .with_span_static(*span, "WorkGroupUniformLoad"));
                        }
                    } */
                    FunctionUniformity {
                        result: Uniformity {
                            non_uniform_result: None,
                            requirements: UniformityRequirements::WORK_GROUP_BARRIER,
                        },
                        exit: ExitFlags::empty(),
                    }
                }
                S::Block(ref b) => {
                    self.process_block(b, other_functions, disruptor, expression_arena)?
                }
                S::If {
                    condition,
                    ref accept,
                    ref reject,
                } => {
                    let condition_nur = self.add_ref(condition);
                    let branch_disruptor =
                        disruptor.or(condition_nur.map(UniformityDisruptor::Expression));
                    let accept_uniformity = self.process_block(
                        accept,
                        other_functions,
                        branch_disruptor,
                        expression_arena,
                    )?;
                    let reject_uniformity = self.process_block(
                        reject,
                        other_functions,
                        branch_disruptor,
                        expression_arena,
                    )?;
                    accept_uniformity | reject_uniformity
                }
                S::Switch {
                    selector,
                    ref cases,
                } => {
                    let selector_nur = self.add_ref(selector);
                    let branch_disruptor =
                        disruptor.or(selector_nur.map(UniformityDisruptor::Expression));
                    let mut uniformity = FunctionUniformity::new();
                    let mut case_disruptor = branch_disruptor;
                    for case in cases.iter() {
                        let case_uniformity = self.process_block(
                            &case.body,
                            other_functions,
                            case_disruptor,
                            expression_arena,
                        )?;
                        case_disruptor = if case.fall_through {
                            case_disruptor.or(case_uniformity.exit_disruptor())
                        } else {
                            branch_disruptor
                        };
                        uniformity = uniformity | case_uniformity;
                    }
                    uniformity
                }
                S::Loop {
                    ref body,
                    ref continuing,
                    break_if,
                } => {
                    let body_uniformity =
                        self.process_block(body, other_functions, disruptor, expression_arena)?;
                    let continuing_disruptor = disruptor.or(body_uniformity.exit_disruptor());
                    let continuing_uniformity = self.process_block(
                        continuing,
                        other_functions,
                        continuing_disruptor,
                        expression_arena,
                    )?;
                    if let Some(expr) = break_if {
                        let _ = self.add_ref(expr);
                    }
                    body_uniformity | continuing_uniformity
                }
                S::Return { value } => FunctionUniformity {
                    result: Uniformity {
                        non_uniform_result: value.and_then(|expr| self.add_ref(expr)),
                        requirements: UniformityRequirements::empty(),
                    },
                    exit: if disruptor.is_some() {
                        ExitFlags::MAY_RETURN
                    } else {
                        ExitFlags::empty()
                    },
                },
                // Here and below, the used expressions are already emitted,
                // and their results do not affect the function return value,
                // so we can ignore their non-uniformity.
                S::Store { pointer, value } => {
                    let _ = self.add_ref_impl(pointer, GlobalUse::WRITE);
                    let _ = self.add_ref(value);
                    FunctionUniformity::new()
                }
                S::ImageStore {
                    image,
                    coordinate,
                    array_index,
                    value,
                } => {
                    let _ = self.add_ref_impl(image, GlobalUse::WRITE);
                    if let Some(expr) = array_index {
                        let _ = self.add_ref(expr);
                    }
                    let _ = self.add_ref(coordinate);
                    let _ = self.add_ref(value);
                    FunctionUniformity::new()
                }
                S::Call {
                    function,
                    ref arguments,
                    result: _,
                } => {
                    for &argument in arguments {
                        let _ = self.add_ref(argument);
                    }
                    let info = &other_functions[function.index()];
                    //Note: the result is validated by the Validator, not here
                    self.process_call(info, arguments, expression_arena)?
                }
                S::Atomic {
                    pointer,
                    ref fun,
                    value,
                    result: _,
                } => {
                    let _ = self.add_ref_impl(pointer, GlobalUse::WRITE);
                    let _ = self.add_ref(value);
                    if let crate::AtomicFunction::Exchange { compare: Some(cmp) } = *fun {
                        let _ = self.add_ref(cmp);
                    }
                    FunctionUniformity::new()
                }
                S::RayQuery { query, ref fun } => {
                    let _ = self.add_ref(query);
                    if let crate::RayQueryFunction::Initialize {
                        acceleration_structure,
                        descriptor,
                    } = *fun
                    {
                        let _ = self.add_ref(acceleration_structure);
                        let _ = self.add_ref(descriptor);
                    }
                    FunctionUniformity::new()
                }
                S::SubgroupBallot { result: _ } => FunctionUniformity::new(),
                S::SubgroupCollectiveOperation {
                    ref op,
                    ref collective_op,
                    argument,
                    result: _,
                } => {
                    let _ = self.add_ref(argument);
                    FunctionUniformity::new()
                }
                S::SubgroupBroadcast {
                    ref mode,
                    argument,
                    result,
                } => {
                    let _ = self.add_ref(argument);
                    if let crate::BroadcastMode::Index(expr) = *mode {
                        let _ = self.add_ref(expr);
                    }
                    FunctionUniformity::new()
                }
            };

            disruptor = disruptor.or(uniformity.exit_disruptor());
            combined_uniformity = combined_uniformity | uniformity;
        }
        Ok(combined_uniformity)
    }
}

impl ModuleInfo {
    /// Populates `self.const_expression_types`
    pub(super) fn process_const_expression(
        &mut self,
        handle: Handle<crate::Expression>,
        resolve_context: &ResolveContext,
        gctx: crate::proc::GlobalCtx,
    ) -> Result<(), super::ConstExpressionError> {
        self.const_expression_types[handle.index()] =
            resolve_context.resolve(&gctx.const_expressions[handle], |h| Ok(&self[h]))?;
        Ok(())
    }

    /// Builds the `FunctionInfo` based on the function, and validates the
    /// uniform control flow if required by the expressions of this function.
    pub(super) fn process_function(
        &self,
        fun: &crate::Function,
        module: &crate::Module,
        flags: ValidationFlags,
        capabilities: super::Capabilities,
    ) -> Result<FunctionInfo, WithSpan<FunctionError>> {
        let mut info = FunctionInfo {
            flags,
            available_stages: ShaderStages::all(),
            uniformity: Uniformity::new(),
            may_kill: false,
            sampling_set: crate::FastHashSet::default(),
            global_uses: vec![GlobalUse::empty(); module.global_variables.len()].into_boxed_slice(),
            expressions: vec![ExpressionInfo::new(); fun.expressions.len()].into_boxed_slice(),
            sampling: crate::FastHashSet::default(),
            dual_source_blending: false,
        };
        let resolve_context =
            ResolveContext::with_locals(module, &fun.local_variables, &fun.arguments);

        for (handle, _) in fun.expressions.iter() {
            if let Err(source) = info.process_expression(
                handle,
                &fun.expressions,
                &self.functions,
                &resolve_context,
                capabilities,
            ) {
                return Err(FunctionError::Expression { handle, source }
                    .with_span_handle(handle, &fun.expressions));
            }
        }

        for (_, expr) in fun.local_variables.iter() {
            if let Some(init) = expr.init {
                let _ = info.add_ref(init);
            }
        }

        let uniformity = info.process_block(&fun.body, &self.functions, None, &fun.expressions)?;
        info.uniformity = uniformity.result;
        info.may_kill = uniformity.exit.contains(ExitFlags::MAY_KILL);

        Ok(info)
    }

    pub fn get_entry_point(&self, index: usize) -> &FunctionInfo {
        &self.entry_points[index]
    }
}

#[test]
#[cfg(feature = "validate")]
fn uniform_control_flow() {
    use crate::{Expression as E, Statement as S};

    let mut type_arena = crate::UniqueArena::new();
    let ty = type_arena.insert(
        crate::Type {
            name: None,
            inner: crate::TypeInner::Vector {
                size: crate::VectorSize::Bi,
                kind: crate::ScalarKind::Float,
                width: 4,
            },
        },
        Default::default(),
    );
    let mut global_var_arena = Arena::new();
    let non_uniform_global = global_var_arena.append(
        crate::GlobalVariable {
            name: None,
            init: None,
            ty,
            space: crate::AddressSpace::Handle,
            binding: None,
        },
        Default::default(),
    );
    let uniform_global = global_var_arena.append(
        crate::GlobalVariable {
            name: None,
            init: None,
            ty,
            binding: None,
            space: crate::AddressSpace::Uniform,
        },
        Default::default(),
    );

    let mut expressions = Arena::new();
    // checks the uniform control flow
    let constant_expr = expressions.append(E::Literal(crate::Literal::U32(0)), Default::default());
    // checks the non-uniform control flow
    let derivative_expr = expressions.append(
        E::Derivative {
            axis: crate::DerivativeAxis::X,
            ctrl: crate::DerivativeControl::None,
            expr: constant_expr,
        },
        Default::default(),
    );
    let emit_range_constant_derivative = expressions.range_from(0);
    let non_uniform_global_expr =
        expressions.append(E::GlobalVariable(non_uniform_global), Default::default());
    let uniform_global_expr =
        expressions.append(E::GlobalVariable(uniform_global), Default::default());
    let emit_range_globals = expressions.range_from(2);

    // checks the QUERY flag
    let query_expr = expressions.append(E::ArrayLength(uniform_global_expr), Default::default());
    // checks the transitive WRITE flag
    let access_expr = expressions.append(
        E::AccessIndex {
            base: non_uniform_global_expr,
            index: 1,
        },
        Default::default(),
    );
    let emit_range_query_access_globals = expressions.range_from(2);

    let mut info = FunctionInfo {
        flags: ValidationFlags::all(),
        available_stages: ShaderStages::all(),
        uniformity: Uniformity::new(),
        may_kill: false,
        sampling_set: crate::FastHashSet::default(),
        global_uses: vec![GlobalUse::empty(); global_var_arena.len()].into_boxed_slice(),
        expressions: vec![ExpressionInfo::new(); expressions.len()].into_boxed_slice(),
        sampling: crate::FastHashSet::default(),
        dual_source_blending: false,
    };
    let resolve_context = ResolveContext {
        constants: &Arena::new(),
        types: &type_arena,
        special_types: &crate::SpecialTypes::default(),
        global_vars: &global_var_arena,
        local_vars: &Arena::new(),
        functions: &Arena::new(),
        arguments: &[],
    };
    for (handle, _) in expressions.iter() {
        info.process_expression(
            handle,
            &expressions,
            &[],
            &resolve_context,
            super::Capabilities::empty(),
        )
        .unwrap();
    }
    assert_eq!(info[non_uniform_global_expr].ref_count, 1);
    assert_eq!(info[uniform_global_expr].ref_count, 1);
    assert_eq!(info[query_expr].ref_count, 0);
    assert_eq!(info[access_expr].ref_count, 0);
    assert_eq!(info[non_uniform_global], GlobalUse::empty());
    assert_eq!(info[uniform_global], GlobalUse::QUERY);

    let stmt_emit1 = S::Emit(emit_range_globals.clone());
    let stmt_if_uniform = S::If {
        condition: uniform_global_expr,
        accept: crate::Block::new(),
        reject: vec![
            S::Emit(emit_range_constant_derivative.clone()),
            S::Store {
                pointer: constant_expr,
                value: derivative_expr,
            },
        ]
        .into(),
    };
    assert_eq!(
        info.process_block(
            &vec![stmt_emit1, stmt_if_uniform].into(),
            &[],
            None,
            &expressions
        ),
        Ok(FunctionUniformity {
            result: Uniformity {
                non_uniform_result: None,
                requirements: UniformityRequirements::DERIVATIVE,
            },
            exit: ExitFlags::empty(),
        }),
    );
    assert_eq!(info[constant_expr].ref_count, 2);
    assert_eq!(info[uniform_global], GlobalUse::READ | GlobalUse::QUERY);

    let stmt_emit2 = S::Emit(emit_range_globals.clone());
    let stmt_if_non_uniform = S::If {
        condition: non_uniform_global_expr,
        accept: vec![
            S::Emit(emit_range_constant_derivative),
            S::Store {
                pointer: constant_expr,
                value: derivative_expr,
            },
        ]
        .into(),
        reject: crate::Block::new(),
    };
    {
        let block_info = info.process_block(
            &vec![stmt_emit2, stmt_if_non_uniform].into(),
            &[],
            None,
            &expressions,
        );
        if DISABLE_UNIFORMITY_REQ_FOR_FRAGMENT_STAGE {
            assert_eq!(info[derivative_expr].ref_count, 2);
        } else {
            assert_eq!(
                block_info,
                Err(FunctionError::NonUniformControlFlow(
                    UniformityRequirements::DERIVATIVE,
                    derivative_expr,
                    UniformityDisruptor::Expression(non_uniform_global_expr)
                )
                .with_span()),
            );
            assert_eq!(info[derivative_expr].ref_count, 1);
        }
    }
    assert_eq!(info[non_uniform_global], GlobalUse::READ);

    let stmt_emit3 = S::Emit(emit_range_globals);
    let stmt_return_non_uniform = S::Return {
        value: Some(non_uniform_global_expr),
    };
    assert_eq!(
        info.process_block(
            &vec![stmt_emit3, stmt_return_non_uniform].into(),
            &[],
            Some(UniformityDisruptor::Return),
            &expressions
        ),
        Ok(FunctionUniformity {
            result: Uniformity {
                non_uniform_result: Some(non_uniform_global_expr),
                requirements: UniformityRequirements::empty(),
            },
            exit: ExitFlags::MAY_RETURN,
        }),
    );
    assert_eq!(info[non_uniform_global_expr].ref_count, 3);

    // Check that uniformity requirements reach through a pointer
    let stmt_emit4 = S::Emit(emit_range_query_access_globals);
    let stmt_assign = S::Store {
        pointer: access_expr,
        value: query_expr,
    };
    let stmt_return_pointer = S::Return {
        value: Some(access_expr),
    };
    let stmt_kill = S::Kill;
    assert_eq!(
        info.process_block(
            &vec![stmt_emit4, stmt_assign, stmt_kill, stmt_return_pointer].into(),
            &[],
            Some(UniformityDisruptor::Discard),
            &expressions
        ),
        Ok(FunctionUniformity {
            result: Uniformity {
                non_uniform_result: Some(non_uniform_global_expr),
                requirements: UniformityRequirements::empty(),
            },
            exit: ExitFlags::all(),
        }),
    );
    assert_eq!(info[non_uniform_global], GlobalUse::READ | GlobalUse::WRITE);
}
