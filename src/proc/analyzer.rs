/*! Module analyzer.

Figures out the following properties:
  - control flow uniformity
  - texture/sampler pairs
  - expression reference counts
!*/

use crate::arena::{Arena, Handle};
use std::ops;

bitflags::bitflags! {
    #[derive(Default)]
    #[cfg_attr(feature = "serialize", derive(serde::Serialize))]
    #[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
    pub struct ControlFlags: u8 {
        /// The result (of an expression) is not dynamically uniform.
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
        const NON_UNIFORM_RESULT = 0x1;
        /// Uniform control flow is required by the code.
        ///
        /// Some operations can only be done within uniform control flow:
        /// derivatives and auto-level image sampling in fragment shaders,
        /// and group barriers in compute shaders.
        ///
        /// This flag bubbles up from child expressions to parents.
        const REQUIRE_UNIFORM = 0x2;
        /// The code may exit the control flow.
        ///
        /// `Kill` and `Return` operations have an effect on all the other
        /// expressions/statements that are evaluated after them. They act as
        /// pervasive control flow branching. Thus, even after the remaining
        /// merge together, we are blocked from considering the merged flow uniform.
        const MAY_EXIT = 0x4;
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

#[derive(Clone, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct ExpressionInfo {
    pub control_flags: ControlFlags,
    pub ref_count: usize,
    assignable_global: Option<Handle<crate::GlobalVariable>>,
}

#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct FunctionInfo {
    /// Accumulated control flags of this function.
    pub control_flags: ControlFlags,
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

#[derive(Clone, Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum AnalysisError {
    #[error("Expression {0:?} is not a global variable!")]
    ExpectedGlobalVariable(crate::Expression),
    //TODO: add more information here!
    #[error("Required uniformity of control flow is not fulfilled")]
    NonUniformControlFlow,
}

impl FunctionInfo {
    /// Adds a value-type reference to an expression.
    #[must_use]
    fn add_ref_impl(
        &mut self,
        handle: Handle<crate::Expression>,
        global_use: GlobalUse,
    ) -> ControlFlags {
        let info = &mut self.expressions[handle.index()];
        info.ref_count += 1;
        // mark the used global as read
        if let Some(global) = info.assignable_global {
            self.global_uses[global.index()] |= global_use;
        }
        info.control_flags
    }

    /// Adds a value-type reference to an expression.
    #[must_use]
    fn add_ref(&mut self, handle: Handle<crate::Expression>) -> ControlFlags {
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
    ) -> ControlFlags {
        let info = &mut self.expressions[handle.index()];
        info.ref_count += 1;
        // propagate the assignable global up the chain, till it either hits
        // a value-type expression, or the assignment statement.
        if let Some(global) = info.assignable_global {
            if let Some(_old) = assignable_global.replace(global) {
                unreachable!()
            }
        }
        info.control_flags
    }

    /// Inherit information from a called function.
    fn process_call(
        &mut self,
        info: &Self,
        arguments: &[Handle<crate::Expression>],
    ) -> ControlFlags {
        for key in info.sampling_set.iter() {
            self.sampling_set.insert(key.clone());
        }
        for (mine, other) in self.global_uses.iter_mut().zip(info.global_uses.iter()) {
            *mine |= *other;
        }
        let mut flags = info.control_flags;
        for &argument in arguments {
            flags |= self.add_ref(argument);
        }
        flags
    }

    /// Computes the control flags of a given expression, and store them
    /// in `self.expressions`. Also, bumps the reference counts on
    /// dependent expressions.
    fn process_expression(
        &mut self,
        handle: Handle<crate::Expression>,
        expression_arena: &Arena<crate::Expression>,
        global_var_arena: &Arena<crate::GlobalVariable>,
        other_functions: &[FunctionInfo],
    ) -> Result<(), AnalysisError> {
        use crate::{Expression as E, SampleLevel as Sl};

        let mut assignable_global = None;
        let control_flags = match expression_arena[handle] {
            E::Access { base, index } => {
                self.add_assignable_ref(base, &mut assignable_global) | self.add_ref(index)
            }
            E::AccessIndex { base, .. } => self.add_assignable_ref(base, &mut assignable_global),
            E::Constant(_) => ControlFlags::empty(),
            E::Compose { ref components, .. } => {
                let mut accum = ControlFlags::empty();
                for &comp in components {
                    accum |= self.add_ref(comp);
                }
                accum
            }
            E::FunctionArgument(_) => ControlFlags::NON_UNIFORM_RESULT, //TODO?
            E::GlobalVariable(handle) => {
                assignable_global = Some(handle);
                let var = &global_var_arena[handle];
                let uniform = if let Some(crate::Binding::BuiltIn(built_in)) = var.binding {
                    match built_in {
                        // per-polygon built-ins are uniform
                        crate::BuiltIn::FrontFacing
                        // per-work-group built-ins are uniform
                        | crate::BuiltIn::WorkGroupId
                        | crate::BuiltIn::WorkGroupSize => true,
                        _ => false,
                    }
                } else {
                    use crate::StorageClass as Sc;
                    match var.class {
                        // only flat inputs are uniform
                        Sc::Input => var.interpolation == Some(crate::Interpolation::Flat),
                        Sc::Output | Sc::Function | Sc::Private | Sc::WorkGroup => false,
                        // uniform data
                        Sc::Uniform | Sc::PushConstant => true,
                        // storage data is only uniform when read-only
                        Sc::Handle | Sc::Storage => {
                            !var.storage_access.contains(crate::StorageAccess::STORE)
                        }
                    }
                };
                if uniform {
                    ControlFlags::empty()
                } else {
                    ControlFlags::NON_UNIFORM_RESULT
                }
            }
            E::LocalVariable(_) => {
                ControlFlags::NON_UNIFORM_RESULT //TODO?
            }
            E::Load { pointer } => self.add_ref(pointer),
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
                        ref other => {
                            return Err(AnalysisError::ExpectedGlobalVariable(other.clone()))
                        }
                    },
                    sampler: match expression_arena[sampler] {
                        crate::Expression::GlobalVariable(var) => var,
                        ref other => {
                            return Err(AnalysisError::ExpectedGlobalVariable(other.clone()))
                        }
                    },
                });
                let array_flags = match array_index {
                    Some(h) => self.add_ref(h),
                    None => ControlFlags::empty(),
                };
                let level_flags = match level {
                    // implicit derivatives for LOD require uniform
                    Sl::Auto => ControlFlags::REQUIRE_UNIFORM,
                    Sl::Zero => ControlFlags::empty(),
                    Sl::Exact(h) | Sl::Bias(h) => self.add_ref(h),
                    Sl::Gradient { x, y } => self.add_ref(x) | self.add_ref(y),
                };
                let dref_flags = match depth_ref {
                    Some(h) => self.add_ref(h),
                    None => ControlFlags::empty(),
                };
                self.add_ref(image)
                    | self.add_ref(sampler)
                    | self.add_ref(coordinate)
                    | array_flags
                    | level_flags
                    | dref_flags
            }
            E::ImageLoad {
                image,
                coordinate,
                array_index,
                index,
            } => {
                let array_flags = match array_index {
                    Some(h) => self.add_ref(h),
                    None => ControlFlags::empty(),
                };
                let index_flags = match index {
                    Some(h) => self.add_ref(h),
                    None => ControlFlags::empty(),
                };
                self.add_ref(image) | self.add_ref(coordinate) | array_flags | index_flags
            }
            E::ImageQuery { image, query } => {
                let query_flags = match query {
                    crate::ImageQuery::Size { level: Some(h) } => self.add_ref(h),
                    _ => ControlFlags::empty(),
                };
                self.add_ref_impl(image, GlobalUse::QUERY) | query_flags
            }
            E::Unary { expr, .. } => self.add_ref(expr),
            E::Binary { left, right, .. } => self.add_ref(left) | self.add_ref(right),
            E::Select {
                condition,
                accept,
                reject,
            } => self.add_ref(condition) | self.add_ref(accept) | self.add_ref(reject),
            // explicit derivatives require uniform
            E::Derivative { expr, .. } => ControlFlags::REQUIRE_UNIFORM | self.add_ref(expr),
            E::Relational { argument, .. } => self.add_ref(argument),
            E::Math {
                arg, arg1, arg2, ..
            } => {
                let arg1_flags = match arg1 {
                    Some(h) => self.add_ref(h),
                    None => ControlFlags::empty(),
                };
                let arg2_flags = match arg2 {
                    Some(h) => self.add_ref(h),
                    None => ControlFlags::empty(),
                };
                self.add_ref(arg) | arg1_flags | arg2_flags
            }
            E::As { expr, .. } => self.add_ref(expr),
            E::Call {
                function,
                ref arguments,
            } => self.process_call(&other_functions[function.index()], arguments),
            E::ArrayLength(expr) => self.add_ref_impl(expr, GlobalUse::QUERY),
        };

        self.expressions[handle.index()] = ExpressionInfo {
            control_flags,
            ref_count: 0,
            assignable_global,
        };
        Ok(())
    }

    /// Computes the control flags on the block (as a sequence of statements),
    /// and returns them. The parent control flow is uniform if `is_uniform` is true.
    ///
    /// Returns a `NonUniformControlFlow` error if any of the expressions in the block
    /// have `ControlFlags::REQUIRE_UNIFORM` flag, but the current flow is non-uniform.
    fn process_block(
        &mut self,
        statements: &[crate::Statement],
        other_functions: &[FunctionInfo],
        mut is_uniform: bool,
    ) -> Result<ControlFlags, AnalysisError> {
        use crate::Statement as S;
        let mut block_flags = ControlFlags::empty();
        for statement in statements {
            let flags = match *statement {
                S::Break | S::Continue => ControlFlags::empty(),
                S::Kill => ControlFlags::MAY_EXIT,
                S::Block(ref b) => self.process_block(b, other_functions, is_uniform)?,
                S::If {
                    condition,
                    ref accept,
                    ref reject,
                } => {
                    let flags = self.add_ref(condition);
                    if flags.contains(ControlFlags::REQUIRE_UNIFORM) && !is_uniform {
                        log::warn!("If condition {:?} needs uniformity", condition);
                        return Err(AnalysisError::NonUniformControlFlow);
                    }
                    let branch_uniform =
                        is_uniform && !flags.contains(ControlFlags::NON_UNIFORM_RESULT);
                    flags
                        | self.process_block(accept, other_functions, branch_uniform)?
                        | self.process_block(reject, other_functions, branch_uniform)?
                }
                S::Switch {
                    selector,
                    ref cases,
                    ref default,
                } => {
                    let mut flags = self.add_ref(selector);
                    if flags.contains(ControlFlags::REQUIRE_UNIFORM) && !is_uniform {
                        log::warn!("Switch selector {:?} needs uniformity", selector);
                        return Err(AnalysisError::NonUniformControlFlow);
                    }
                    let branch_uniform =
                        is_uniform && !flags.contains(ControlFlags::NON_UNIFORM_RESULT);
                    let mut still_uniform = branch_uniform;
                    for case in cases.iter() {
                        let case_flags =
                            self.process_block(&case.body, other_functions, still_uniform)?;
                        flags |= case_flags;
                        if case.fall_through {
                            still_uniform &= !case_flags.contains(ControlFlags::MAY_EXIT);
                        } else {
                            still_uniform = branch_uniform;
                        }
                    }
                    flags | self.process_block(default, other_functions, still_uniform)?
                }
                S::Loop {
                    ref body,
                    ref continuing,
                } => {
                    let flags = self.process_block(body, other_functions, is_uniform)?;
                    let still_uniform = is_uniform && !flags.contains(ControlFlags::MAY_EXIT);
                    self.process_block(continuing, other_functions, still_uniform)?
                }
                S::Return { value } => {
                    let flags = match value {
                        Some(expr) => self.add_ref(expr),
                        None => ControlFlags::empty(),
                    };
                    ControlFlags::MAY_EXIT | flags
                }
                S::Store { pointer, value } => {
                    self.add_ref_impl(pointer, GlobalUse::WRITE) | self.add_ref(value)
                }
                S::ImageStore {
                    image,
                    coordinate,
                    array_index,
                    value,
                } => {
                    let array_flags = match array_index {
                        Some(expr) => self.add_ref(expr),
                        None => ControlFlags::empty(),
                    };
                    array_flags
                        | self.add_ref_impl(image, GlobalUse::WRITE)
                        | self.add_ref(coordinate)
                        | self.add_ref(value)
                }
                S::Call {
                    function,
                    ref arguments,
                } => self.process_call(&other_functions[function.index()], arguments),
            };

            if flags.contains(ControlFlags::REQUIRE_UNIFORM) && !is_uniform {
                return Err(AnalysisError::NonUniformControlFlow);
            }
            is_uniform &= !flags.contains(ControlFlags::MAY_EXIT);
            block_flags |= flags;
        }
        Ok(block_flags)
    }
}

#[derive(Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct Analysis {
    functions: Vec<FunctionInfo>,
    entry_points: crate::FastHashMap<(crate::ShaderStage, String), FunctionInfo>,
}

impl Analysis {
    /// Builds the `FunctionInfo` based on the function, and validates the
    /// uniform control flow if required by the expressions of this function.
    fn process_function(
        &self,
        fun: &crate::Function,
        global_var_arena: &Arena<crate::GlobalVariable>,
    ) -> Result<FunctionInfo, AnalysisError> {
        let mut info = FunctionInfo {
            control_flags: ControlFlags::empty(),
            sampling_set: crate::FastHashSet::default(),
            global_uses: vec![GlobalUse::empty(); global_var_arena.len()].into_boxed_slice(),
            expressions: vec![ExpressionInfo::default(); fun.expressions.len()].into_boxed_slice(),
        };

        for (handle, _) in fun.expressions.iter() {
            info.process_expression(handle, &fun.expressions, global_var_arena, &self.functions)?;
        }

        info.control_flags = info.process_block(&fun.body, &self.functions, true)?;

        Ok(info)
    }

    /// Analyze a module and return the `Analysis`, if successful.
    pub fn new(module: &crate::Module) -> Result<Self, AnalysisError> {
        let mut this = Analysis {
            functions: Vec::with_capacity(module.functions.len()),
            entry_points: crate::FastHashMap::default(),
        };
        for (_, fun) in module.functions.iter() {
            let info = this.process_function(fun, &module.global_variables)?;
            this.functions.push(info);
        }

        for (key, ep) in module.entry_points.iter() {
            let info = this.process_function(&ep.function, &module.global_variables)?;
            this.entry_points.insert(key.clone(), info);
        }

        Ok(this)
    }

    pub fn get_entry_point(&self, stage: crate::ShaderStage, name: &str) -> &FunctionInfo {
        let (_, info) = self
            .entry_points
            .iter()
            .find(|(key, _)| key.0 == stage && key.1 == name)
            .unwrap();
        info
    }
}

impl ops::Index<Handle<crate::Function>> for Analysis {
    type Output = FunctionInfo;
    fn index(&self, handle: Handle<crate::Function>) -> &FunctionInfo {
        &self.functions[handle.index()]
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
        inner: crate::TypeInner::Scalar {
            kind: crate::ScalarKind::Float,
            width: 4,
        },
    });
    let mut global_var_arena = Arena::new();
    let non_uniform_global = global_var_arena.append(crate::GlobalVariable {
        name: None,
        init: None,
        ty,
        binding: Some(crate::Binding::BuiltIn(crate::BuiltIn::VertexIndex)),
        class: crate::StorageClass::Input,
        interpolation: None,
        storage_access: crate::StorageAccess::empty(),
    });
    let uniform_global = global_var_arena.append(crate::GlobalVariable {
        name: None,
        init: None,
        ty,
        binding: Some(crate::Binding::Location(0)),
        class: crate::StorageClass::Input,
        interpolation: Some(crate::Interpolation::Flat),
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
    let non_uniform_global_expr = expressions.append(E::GlobalVariable(non_uniform_global));
    let uniform_global_expr = expressions.append(E::GlobalVariable(uniform_global));
    // checks the QUERY flag
    let query_expr = expressions.append(E::ArrayLength(uniform_global_expr));
    // checks the transitive WRITE flag
    let access_expr = expressions.append(E::AccessIndex {
        base: non_uniform_global_expr,
        index: 1,
    });

    let mut info = FunctionInfo {
        control_flags: ControlFlags::empty(),
        sampling_set: crate::FastHashSet::default(),
        global_uses: vec![GlobalUse::empty(); global_var_arena.len()].into_boxed_slice(),
        expressions: vec![ExpressionInfo::default(); expressions.len()].into_boxed_slice(),
    };
    for (handle, _) in expressions.iter() {
        info.process_expression(handle, &expressions, &global_var_arena, &[])
            .unwrap();
    }
    assert_eq!(info[non_uniform_global_expr].ref_count, 1);
    assert_eq!(info[uniform_global_expr].ref_count, 1);
    assert_eq!(info[query_expr].ref_count, 0);
    assert_eq!(info[access_expr].ref_count, 0);
    assert_eq!(info[non_uniform_global], GlobalUse::empty());
    assert_eq!(info[uniform_global], GlobalUse::QUERY);

    let stmt_if_uniform = S::If {
        condition: uniform_global_expr,
        accept: Vec::new(),
        reject: vec![S::Store {
            pointer: constant_expr,
            value: derivative_expr,
        }],
    };
    assert_eq!(
        info.process_block(&[stmt_if_uniform], &[], true),
        Ok(ControlFlags::REQUIRE_UNIFORM),
    );
    assert_eq!(info[constant_expr].ref_count, 2);
    assert_eq!(info[uniform_global], GlobalUse::READ | GlobalUse::QUERY);

    let stmt_if_non_uniform = S::If {
        condition: non_uniform_global_expr,
        accept: vec![S::Store {
            pointer: constant_expr,
            value: derivative_expr,
        }],
        reject: Vec::new(),
    };
    assert_eq!(
        info.process_block(&[stmt_if_non_uniform], &[], true),
        Err(AnalysisError::NonUniformControlFlow),
    );
    assert_eq!(info[derivative_expr].ref_count, 2);
    assert_eq!(info[non_uniform_global], GlobalUse::READ);

    let stmt_return_non_uniform = S::Return {
        value: Some(non_uniform_global_expr),
    };
    assert_eq!(
        info.process_block(&[stmt_return_non_uniform], &[], false),
        Ok(ControlFlags::NON_UNIFORM_RESULT | ControlFlags::MAY_EXIT),
    );
    assert_eq!(info[non_uniform_global_expr].ref_count, 3);

    let stmt_assign = S::Store {
        pointer: access_expr,
        value: query_expr,
    };
    assert_eq!(
        info.process_block(&[stmt_assign], &[], false),
        Ok(ControlFlags::NON_UNIFORM_RESULT),
    );
    assert_eq!(info[non_uniform_global], GlobalUse::READ | GlobalUse::WRITE);
}
