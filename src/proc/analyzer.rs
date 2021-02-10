/*! Module analyzer.

Figures out the following properties:
  - control flow uniformity
  - texture/sampler pairs
  - expression reference counts
!*/

use crate::arena::{Arena, Handle};

bitflags::bitflags! {
    #[derive(Default)]
    pub struct ControlFlags: u8 {
        /// The result is not dynamically uniform.
        const NON_UNIFORM_RESULT = 0x1;
        /// Uniform control flow is required by the code.
        const REQUIRE_UNIFORM = 0x2;
        /// The code may exit the control flow.
        const MAY_EXIT = 0x4;
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SamplingKey {
    pub image: Handle<crate::GlobalVariable>,
    pub sampler: Handle<crate::GlobalVariable>,
}

#[derive(Clone, Default)]
pub struct ExpressionInfo {
    pub control_flags: ControlFlags,
    pub ref_count: usize,
}

pub struct FunctionInfo {
    pub control_flags: ControlFlags,
    pub sampling_set: crate::FastHashSet<SamplingKey>,
    pub expressions: Box<[ExpressionInfo]>,
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum AnalysisError {
    #[error("Expression {0:?} is not a global variable!")]
    ExpectedGlobalVariable(crate::Expression),
    //TODO: add more information here!
    #[error("Required uniformity of control flow is not fulfilled")]
    NonUniformControlFlow,
}

impl FunctionInfo {
    #[must_use]
    fn add_ref(&mut self, handle: Handle<crate::Expression>) -> ControlFlags {
        let info = &mut self.expressions[handle.index()];
        info.ref_count += 1;
        info.control_flags
    }

    fn process_expression(
        &mut self,
        handle: Handle<crate::Expression>,
        expression_arena: &Arena<crate::Expression>,
        global_var_arena: &Arena<crate::GlobalVariable>,
        other_functions: &[FunctionInfo],
    ) -> Result<(), AnalysisError> {
        use crate::{Expression as E, SampleLevel as Sl};

        let control_flags = match expression_arena[handle] {
            E::Access { base, index } => self.add_ref(base) | self.add_ref(index),
            E::AccessIndex { base, .. } => self.add_ref(base),
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
                let var = &global_var_arena[handle];
                match var.binding {
                    Some(crate::Binding::BuiltIn(built_in)) => match built_in {
                        crate::BuiltIn::FrontFacing
                        | crate::BuiltIn::WorkGroupId
                        | crate::BuiltIn::WorkGroupSize => ControlFlags::empty(),
                        _ => ControlFlags::NON_UNIFORM_RESULT,
                    },
                    _ if var.class == crate::StorageClass::Input
                        && var.interpolation == Some(crate::Interpolation::Flat) =>
                    {
                        ControlFlags::empty()
                    }
                    _ => ControlFlags::NON_UNIFORM_RESULT,
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
                self.add_ref(image) | query_flags
            }
            E::Unary { expr, .. } => self.add_ref(expr),
            E::Binary { left, right, .. } => self.add_ref(left) | self.add_ref(right),
            E::Select {
                condition,
                accept,
                reject,
            } => self.add_ref(condition) | self.add_ref(accept) | self.add_ref(reject),
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
            } => {
                let other_info = &other_functions[function.index()];
                for key in other_info.sampling_set.iter() {
                    self.sampling_set.insert(key.clone());
                }
                let mut accum = other_info.control_flags;
                for &argument in arguments {
                    accum |= self.add_ref(argument);
                }
                accum
            }
            E::ArrayLength(expr) => self.add_ref(expr),
        };

        self.expressions[handle.index()] = ExpressionInfo {
            control_flags,
            ref_count: 0,
        };
        Ok(())
    }

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
                S::Store { pointer, value } => self.add_ref(pointer) | self.add_ref(value),
                S::Call {
                    function,
                    ref arguments,
                } => {
                    let mut flags = other_functions[function.index()].control_flags;
                    for &argument in arguments {
                        flags |= self.add_ref(argument);
                    }
                    flags
                }
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
pub struct Analysis {
    functions: Vec<FunctionInfo>,
    entry_points: crate::FastHashMap<(crate::ShaderStage, String), FunctionInfo>,
}

impl Analysis {
    fn process_function(
        &self,
        fun: &crate::Function,
        global_var_arena: &Arena<crate::GlobalVariable>,
    ) -> Result<FunctionInfo, AnalysisError> {
        let mut info = FunctionInfo {
            control_flags: ControlFlags::empty(),
            sampling_set: crate::FastHashSet::default(),
            expressions: vec![ExpressionInfo::default(); fun.expressions.len()].into_boxed_slice(),
        };

        for (handle, _) in fun.expressions.iter() {
            info.process_expression(handle, &fun.expressions, global_var_arena, &self.functions)?;
        }

        info.control_flags = info.process_block(&fun.body, &self.functions, true)?;

        Ok(info)
    }

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
}
