/*! Module analyzer.

Figures out the following properties:
  - control flow uniformity
  - texture/sampler pairs
  - expression reference counts
!*/

use super::{CallError, ExpressionError, FunctionError, ModuleInfo, ShaderStages, ValidationFlags};
use crate::{
    arena::{Arena, Handle},
    proc::{ResolveContext, TypeResolution},
};
use std::ops;

pub type NonUniformResult = Option<Handle<crate::Expression>>;

bitflags::bitflags! {
    /// Kinds of expressions that require uniform control flow.
    #[cfg_attr(feature = "serialize", derive(serde::Serialize))]
    #[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
    pub struct UniformityRequirements: u8 {
        const WORK_GROUP_BARRIER = 0x1;
        const DERIVATIVE = 0x2;
        const IMPLICIT_LEVEL = 0x4;
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
    fn new() -> Self {
        Uniformity {
            non_uniform_result: None,
            requirements: UniformityRequirements::empty(),
        }
    }
}

bitflags::bitflags! {
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
    fn new() -> Self {
        FunctionUniformity {
            result: Uniformity::new(),
            exit: ExitFlags::empty(),
        }
    }

    /// Returns a disruptor based on the stored exit flags, if any.
    fn exit_disruptor(&self) -> Option<UniformityDisruptor> {
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
    fn new() -> Self {
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

#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct FunctionInfo {
    /// Validation flags.
    flags: ValidationFlags,
    /// Set of shader stages where calling this function is valid.
    pub available_stages: ShaderStages,
    /// Uniformity characteristics.
    pub uniformity: Uniformity,
    /// Function may kill the invocation.
    pub may_kill: bool,
    /// Set of image-sampler pais used with sampling.
    pub sampling_set: crate::FastHashSet<SamplingKey>,
    /// Vector of global variable usages.
    ///
    /// Each item corresponds to a global variable in the module.
    global_uses: Box<[GlobalUse]>,
    /// Vector of expression infos.
    ///
    /// Each item corresponds to an expression in the function.
    expressions: Box<[ExpressionInfo]>,
}

impl FunctionInfo {
    pub fn global_variable_count(&self) -> usize {
        self.global_uses.len()
    }
    pub fn expression_count(&self) -> usize {
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
    fn process_call(&mut self, info: &Self) -> FunctionUniformity {
        for key in info.sampling_set.iter() {
            self.sampling_set.insert(key.clone());
        }
        for (mine, other) in self.global_uses.iter_mut().zip(info.global_uses.iter()) {
            *mine |= *other;
        }
        FunctionUniformity {
            result: info.uniformity.clone(),
            exit: if info.may_kill {
                ExitFlags::MAY_KILL
            } else {
                ExitFlags::empty()
            },
        }
    }

    /// Computes the expression info and stores it in `self.expressions`.
    /// Also, bumps the reference counts on dependent expressions.
    #[allow(clippy::or_fun_call)]
    fn process_expression(
        &mut self,
        handle: Handle<crate::Expression>,
        expression: &crate::Expression,
        expression_arena: &Arena<crate::Expression>,
        other_functions: &[FunctionInfo],
        type_arena: &Arena<crate::Type>,
        resolve_context: &ResolveContext,
    ) -> Result<(), ExpressionError> {
        use crate::{Expression as E, SampleLevel as Sl};

        let mut assignable_global = None;
        let uniformity = match *expression {
            E::Access { base, index } => Uniformity {
                non_uniform_result: self
                    .add_assignable_ref(base, &mut assignable_global)
                    .or(self.add_ref(index)),
                requirements: UniformityRequirements::empty(),
            },
            E::AccessIndex { base, .. } => Uniformity {
                non_uniform_result: self.add_assignable_ref(base, &mut assignable_global),
                requirements: UniformityRequirements::empty(),
            },
            // always uniform
            E::Constant(_) => Uniformity::new(),
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
                        | crate::BuiltIn::WorkGroupSize => true,
                        _ => false,
                    },
                    // only flat inputs are uniform
                    Some(crate::Binding::Location(_, Some(crate::Interpolation::Flat))) => true,
                    _ => false,
                };
                Uniformity {
                    non_uniform_result: if uniform { None } else { Some(handle) },
                    requirements: UniformityRequirements::empty(),
                }
            }
            // depends on the storage class
            E::GlobalVariable(gh) => {
                use crate::StorageClass as Sc;
                assignable_global = Some(gh);
                let var = &resolve_context.global_vars[gh];
                let uniform = match var.class {
                    // local data is non-uniform
                    Sc::Function | Sc::Private => false,
                    // workgroup memory is exclusively accessed by the group
                    Sc::WorkGroup => true,
                    // uniform data
                    Sc::Uniform | Sc::PushConstant => true,
                    // storage data is only uniform when read-only
                    Sc::Handle | Sc::Storage => {
                        !var.storage_access.contains(crate::StorageAccess::STORE)
                    }
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
                coordinate,
                array_index,
                offset: _,
                level,
                depth_ref,
            } => {
                self.sampling_set.insert(SamplingKey {
                    image: match expression_arena[image] {
                        crate::Expression::GlobalVariable(var) => var,
                        _ => return Err(ExpressionError::ExpectedGlobalVariable),
                    },
                    sampler: match expression_arena[sampler] {
                        crate::Expression::GlobalVariable(var) => var,
                        _ => return Err(ExpressionError::ExpectedGlobalVariable),
                    },
                });
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
                index,
            } => {
                let array_nur = array_index.and_then(|h| self.add_ref(h));
                let index_nur = index.and_then(|h| self.add_ref(h));
                Uniformity {
                    non_uniform_result: self
                        .add_ref(image)
                        .or(self.add_ref(coordinate))
                        .or(array_nur)
                        .or(index_nur),
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
                arg, arg1, arg2, ..
            } => {
                let arg1_nur = arg1.and_then(|h| self.add_ref(h));
                let arg2_nur = arg2.and_then(|h| self.add_ref(h));
                Uniformity {
                    non_uniform_result: self.add_ref(arg).or(arg1_nur).or(arg2_nur),
                    requirements: UniformityRequirements::empty(),
                }
            }
            E::As { expr, .. } => Uniformity {
                non_uniform_result: self.add_ref(expr),
                requirements: UniformityRequirements::empty(),
            },
            E::Call(function) => {
                let fun = other_functions
                    .get(function.index())
                    .ok_or(ExpressionError::CallToUndeclaredFunction(function))?;
                self.process_call(fun).result
            }
            E::ArrayLength(expr) => Uniformity {
                non_uniform_result: self.add_ref_impl(expr, GlobalUse::QUERY),
                requirements: UniformityRequirements::empty(),
            },
        };

        let ty =
            resolve_context.resolve(expression, type_arena, |h| &self.expressions[h.index()].ty)?;
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
        statements: &[crate::Statement],
        other_functions: &[FunctionInfo],
        mut disruptor: Option<UniformityDisruptor>,
    ) -> Result<FunctionUniformity, FunctionError> {
        use crate::Statement as S;

        let mut combined_uniformity = FunctionUniformity::new();
        for statement in statements {
            let uniformity = match *statement {
                S::Emit(ref range) => {
                    let mut requirements = UniformityRequirements::empty();
                    for expr in range.clone() {
                        let req = self.expressions[expr.index()].uniformity.requirements;
                        if self
                            .flags
                            .contains(super::ValidationFlags::CONTROL_FLOW_UNIFORMITY)
                            && !req.is_empty()
                        {
                            if let Some(cause) = disruptor {
                                return Err(FunctionError::NonUniformControlFlow(req, expr, cause));
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
                    exit: ExitFlags::MAY_KILL,
                },
                S::Block(ref b) => self.process_block(b, other_functions, disruptor)?,
                S::If {
                    condition,
                    ref accept,
                    ref reject,
                } => {
                    let condition_nur = self.add_ref(condition);
                    let branch_disruptor =
                        disruptor.or(condition_nur.map(UniformityDisruptor::Expression));
                    let accept_uniformity =
                        self.process_block(accept, other_functions, branch_disruptor)?;
                    let reject_uniformity =
                        self.process_block(reject, other_functions, branch_disruptor)?;
                    accept_uniformity | reject_uniformity
                }
                S::Switch {
                    selector,
                    ref cases,
                    ref default,
                } => {
                    let selector_nur = self.add_ref(selector);
                    let branch_disruptor =
                        disruptor.or(selector_nur.map(UniformityDisruptor::Expression));
                    let mut uniformity = FunctionUniformity::new();
                    let mut case_disruptor = branch_disruptor;
                    for case in cases.iter() {
                        let case_uniformity =
                            self.process_block(&case.body, other_functions, case_disruptor)?;
                        case_disruptor = if case.fall_through {
                            case_disruptor.or(case_uniformity.exit_disruptor())
                        } else {
                            branch_disruptor
                        };
                        uniformity = uniformity | case_uniformity;
                    }
                    // using the disruptor inherited from the last fall-through chain
                    let default_exit =
                        self.process_block(default, other_functions, case_disruptor)?;
                    uniformity | default_exit
                }
                S::Loop {
                    ref body,
                    ref continuing,
                } => {
                    let body_uniformity = self.process_block(body, other_functions, disruptor)?;
                    let continuing_disruptor = disruptor.or(body_uniformity.exit_disruptor());
                    let continuing_uniformity =
                        self.process_block(continuing, other_functions, continuing_disruptor)?;
                    body_uniformity | continuing_uniformity
                }
                S::Return { value } => FunctionUniformity {
                    result: Uniformity {
                        non_uniform_result: value.and_then(|expr| self.add_ref(expr)),
                        requirements: UniformityRequirements::empty(),
                    },
                    //TODO: if we are in the uniform control flow, should this still be an exit flag?
                    exit: ExitFlags::MAY_RETURN,
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
                    let info = other_functions.get(function.index()).ok_or(
                        FunctionError::InvalidCall {
                            function,
                            error: CallError::ForwardDeclaredFunction,
                        },
                    )?;
                    //Note: the result is validated by the Validator, not here
                    self.process_call(info)
                }
            };

            disruptor = disruptor.or(uniformity.exit_disruptor());
            combined_uniformity = combined_uniformity | uniformity;
        }
        Ok(combined_uniformity)
    }
}

impl ModuleInfo {
    /// Builds the `FunctionInfo` based on the function, and validates the
    /// uniform control flow if required by the expressions of this function.
    pub(super) fn process_function(
        &self,
        fun: &crate::Function,
        module: &crate::Module,
        flags: ValidationFlags,
    ) -> Result<FunctionInfo, FunctionError> {
        let mut info = FunctionInfo {
            flags,
            available_stages: ShaderStages::all(),
            uniformity: Uniformity::new(),
            may_kill: false,
            sampling_set: crate::FastHashSet::default(),
            global_uses: vec![GlobalUse::empty(); module.global_variables.len()].into_boxed_slice(),
            expressions: vec![ExpressionInfo::new(); fun.expressions.len()].into_boxed_slice(),
        };
        let resolve_context = ResolveContext {
            constants: &module.constants,
            global_vars: &module.global_variables,
            local_vars: &fun.local_variables,
            functions: &module.functions,
            arguments: &fun.arguments,
        };

        for (handle, expr) in fun.expressions.iter() {
            if let Err(error) = info.process_expression(
                handle,
                expr,
                &fun.expressions,
                &self.functions,
                &module.types,
                &resolve_context,
            ) {
                return Err(FunctionError::Expression { handle, error });
            }
        }

        let uniformity = info.process_block(&fun.body, &self.functions, None)?;
        info.uniformity = uniformity.result;
        info.may_kill = uniformity.exit.contains(ExitFlags::MAY_KILL);

        Ok(info)
    }

    pub fn get_entry_point(&self, index: usize) -> &FunctionInfo {
        &self.entry_points[index]
    }
}

#[test]
fn uniform_control_flow() {
    use crate::{Expression as E, Statement as S};

    let mut constant_arena = Arena::new();
    let constant = constant_arena.append(crate::Constant {
        name: None,
        specialization: None,
        inner: crate::ConstantInner::Scalar {
            width: 4,
            value: crate::ScalarValue::Uint(0),
        },
    });
    let mut type_arena = Arena::new();
    let ty = type_arena.append(crate::Type {
        name: None,
        inner: crate::TypeInner::Vector {
            size: crate::VectorSize::Bi,
            kind: crate::ScalarKind::Float,
            width: 4,
        },
    });
    let mut global_var_arena = Arena::new();
    let non_uniform_global = global_var_arena.append(crate::GlobalVariable {
        name: None,
        init: None,
        ty,
        class: crate::StorageClass::Handle,
        binding: None,
        storage_access: crate::StorageAccess::STORE,
    });
    let uniform_global = global_var_arena.append(crate::GlobalVariable {
        name: None,
        init: None,
        ty,
        binding: None,
        class: crate::StorageClass::Uniform,
        storage_access: crate::StorageAccess::empty(),
    });

    let mut expressions = Arena::new();
    // checks the uniform control flow
    let constant_expr = expressions.append(E::Constant(constant));
    // checks the non-uniform control flow
    let derivative_expr = expressions.append(E::Derivative {
        axis: crate::DerivativeAxis::X,
        expr: constant_expr,
    });
    let emit_range_constant_derivative = expressions.range_from(0);
    let non_uniform_global_expr = expressions.append(E::GlobalVariable(non_uniform_global));
    let uniform_global_expr = expressions.append(E::GlobalVariable(uniform_global));
    let emit_range_globals = expressions.range_from(2);

    // checks the QUERY flag
    let query_expr = expressions.append(E::ArrayLength(uniform_global_expr));
    // checks the transitive WRITE flag
    let access_expr = expressions.append(E::AccessIndex {
        base: non_uniform_global_expr,
        index: 1,
    });
    let emit_range_query_access_globals = expressions.range_from(2);

    let mut info = FunctionInfo {
        flags: ValidationFlags::all(),
        available_stages: ShaderStages::all(),
        uniformity: Uniformity::new(),
        may_kill: false,
        sampling_set: crate::FastHashSet::default(),
        global_uses: vec![GlobalUse::empty(); global_var_arena.len()].into_boxed_slice(),
        expressions: vec![ExpressionInfo::new(); expressions.len()].into_boxed_slice(),
    };
    let resolve_context = ResolveContext {
        constants: &constant_arena,
        global_vars: &global_var_arena,
        local_vars: &Arena::new(),
        functions: &Arena::new(),
        arguments: &[],
    };
    for (handle, expression) in expressions.iter() {
        info.process_expression(
            handle,
            expression,
            &expressions,
            &[],
            &type_arena,
            &resolve_context,
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
        accept: Vec::new(),
        reject: vec![
            S::Emit(emit_range_constant_derivative.clone()),
            S::Store {
                pointer: constant_expr,
                value: derivative_expr,
            },
        ],
    };
    assert_eq!(
        info.process_block(&[stmt_emit1, stmt_if_uniform], &[], None),
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
            S::Emit(emit_range_constant_derivative.clone()),
            S::Store {
                pointer: constant_expr,
                value: derivative_expr,
            },
        ],
        reject: Vec::new(),
    };
    assert_eq!(
        info.process_block(&[stmt_emit2, stmt_if_non_uniform], &[], None),
        Err(FunctionError::NonUniformControlFlow(
            UniformityRequirements::DERIVATIVE,
            derivative_expr,
            UniformityDisruptor::Expression(non_uniform_global_expr)
        )),
    );
    assert_eq!(info[derivative_expr].ref_count, 1);
    assert_eq!(info[non_uniform_global], GlobalUse::READ);

    let stmt_emit3 = S::Emit(emit_range_globals);
    let stmt_return_non_uniform = S::Return {
        value: Some(non_uniform_global_expr),
    };
    assert_eq!(
        info.process_block(
            &[stmt_emit3, stmt_return_non_uniform],
            &[],
            Some(UniformityDisruptor::Return)
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
            &[stmt_emit4, stmt_assign, stmt_kill, stmt_return_pointer],
            &[],
            Some(UniformityDisruptor::Discard)
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
