use super::{HandleMap, HandleSet, ModuleMap};
use crate::arena::{Arena, Handle, UniqueArena};

pub struct ExpressionTracer<'tracer> {
    pub types: &'tracer UniqueArena<crate::Type>,
    pub constants: &'tracer Arena<crate::Constant>,

    /// The arena in which we are currently tracing expressions.
    pub expressions: &'tracer Arena<crate::Expression>,

    /// The used map for `types`.
    pub types_used: &'tracer mut HandleSet<crate::Type>,

    /// The used map for `constants`.
    pub constants_used: &'tracer mut HandleSet<crate::Constant>,

    /// The used set for `arena`.
    ///
    /// This points to whatever arena holds the expressions we are
    /// currently tracing: either a function's expression arena, or
    /// the module's constant expression arena.
    pub expressions_used: &'tracer mut HandleSet<crate::Expression>,

    /// The constant expression arena and its used map, if we haven't
    /// switched to tracing constant expressions yet.
    pub const_expressions: Option<(
        &'tracer Arena<crate::Expression>,
        &'tracer mut HandleSet<crate::Expression>,
    )>,
}

impl<'tracer> ExpressionTracer<'tracer> {
    pub fn trace_expression(&mut self, expr: Handle<crate::Expression>) {
        log::trace!(
            "entering trace_expression of {}",
            if self.const_expressions.is_some() {
                "function expressions"
            } else {
                "const expressions"
            }
        );
        let mut work_list = vec![expr];
        while let Some(expr) = work_list.pop() {
            // If we've already seen this expression, no need to trace further.
            if !self.expressions_used.insert(expr) {
                continue;
            }
            log::trace!("tracing new expression {:?}", expr);

            use crate::Expression as Ex;
            match self.expressions[expr] {
                // Expressions that do not contain handles that need to be traced.
                Ex::Literal(_)
                | Ex::FunctionArgument(_)
                | Ex::GlobalVariable(_)
                | Ex::LocalVariable(_)
                | Ex::CallResult(_)
                | Ex::SubgroupBallotResult
                | Ex::RayQueryProceedResult => {}

                Ex::Constant(handle) => {
                    self.constants_used.insert(handle);
                    let constant = &self.constants[handle];
                    self.trace_type(constant.ty);
                    self.trace_const_expression(constant.init);
                }
                Ex::ZeroValue(ty) => self.trace_type(ty),
                Ex::Compose { ty, ref components } => {
                    self.trace_type(ty);
                    work_list.extend(components);
                }
                Ex::Access { base, index } => work_list.extend([base, index]),
                Ex::AccessIndex { base, index: _ } => work_list.push(base),
                Ex::Splat { size: _, value } => work_list.push(value),
                Ex::Swizzle {
                    size: _,
                    vector,
                    pattern: _,
                } => work_list.push(vector),
                Ex::Load { pointer } => work_list.push(pointer),
                Ex::ImageSample {
                    image,
                    sampler,
                    gather: _,
                    coordinate,
                    array_index,
                    offset,
                    ref level,
                    depth_ref,
                } => {
                    work_list.push(image);
                    work_list.push(sampler);
                    work_list.push(coordinate);
                    work_list.extend(array_index);
                    if let Some(offset) = offset {
                        self.trace_const_expression(offset);
                    }
                    use crate::SampleLevel as Sl;
                    match *level {
                        Sl::Auto | Sl::Zero => {}
                        Sl::Exact(expr) | Sl::Bias(expr) => work_list.push(expr),
                        Sl::Gradient { x, y } => work_list.extend([x, y]),
                    }
                    work_list.extend(depth_ref);
                }
                Ex::ImageLoad {
                    image,
                    coordinate,
                    array_index,
                    sample,
                    level,
                } => {
                    work_list.push(image);
                    work_list.push(coordinate);
                    work_list.extend(array_index);
                    work_list.extend(sample);
                    work_list.extend(level);
                }
                Ex::ImageQuery { image, ref query } => {
                    work_list.push(image);
                    use crate::ImageQuery as Iq;
                    match *query {
                        Iq::Size { level } => work_list.extend(level),
                        Iq::NumLevels | Iq::NumLayers | Iq::NumSamples => {}
                    }
                }
                Ex::Unary { op: _, expr } => work_list.push(expr),
                Ex::Binary { op: _, left, right } => work_list.extend([left, right]),
                Ex::Select {
                    condition,
                    accept,
                    reject,
                } => work_list.extend([condition, accept, reject]),
                Ex::Derivative {
                    axis: _,
                    ctrl: _,
                    expr,
                } => work_list.push(expr),
                Ex::Relational { fun: _, argument } => work_list.push(argument),
                Ex::Math {
                    fun: _,
                    arg,
                    arg1,
                    arg2,
                    arg3,
                } => {
                    work_list.push(arg);
                    work_list.extend(arg1);
                    work_list.extend(arg2);
                    work_list.extend(arg3);
                }
                Ex::As {
                    expr,
                    kind: _,
                    convert: _,
                } => work_list.push(expr),
                Ex::AtomicResult { ty, comparison: _ } => self.trace_type(ty),
                Ex::WorkGroupUniformLoadResult { ty } => self.trace_type(ty),
                Ex::ArrayLength(expr) => work_list.push(expr),
                Ex::SubgroupOperationResult { ty } => self.trace_type(ty),
                Ex::RayQueryGetIntersection {
                    query,
                    committed: _,
                } => work_list.push(query),
            }
        }
    }

    fn trace_type(&mut self, ty: Handle<crate::Type>) {
        let mut types_used = super::types::TypeTracer {
            types: self.types,
            types_used: self.types_used,
        };
        types_used.trace_type(ty);
    }

    pub fn as_const_expression(&mut self) -> ExpressionTracer {
        match self.const_expressions {
            Some((ref mut exprs, ref mut exprs_used)) => ExpressionTracer {
                expressions: exprs,
                expressions_used: exprs_used,
                types: self.types,
                constants: self.constants,
                types_used: self.types_used,
                constants_used: self.constants_used,
                const_expressions: None,
            },
            None => ExpressionTracer {
                types: self.types,
                constants: self.constants,
                expressions: self.expressions,
                types_used: self.types_used,
                constants_used: self.constants_used,
                expressions_used: self.expressions_used,
                const_expressions: None,
            },
        }
    }

    fn trace_const_expression(&mut self, const_expr: Handle<crate::Expression>) {
        self.as_const_expression().trace_expression(const_expr);
    }
}

impl ModuleMap {
    /// Fix up all handles in `expr`.
    ///
    /// Use the expression handle remappings in `operand_map`, and all
    /// other mappings from `self`.
    pub fn adjust_expression(
        &self,
        expr: &mut crate::Expression,
        operand_map: &HandleMap<crate::Expression>,
    ) {
        let adjust = |expr: &mut Handle<crate::Expression>| {
            operand_map.adjust(expr);
        };

        use crate::Expression as Ex;
        match *expr {
            // Expressions that do not contain handles that need to be adjusted.
            Ex::Literal(_)
            | Ex::FunctionArgument(_)
            | Ex::GlobalVariable(_)
            | Ex::LocalVariable(_)
            | Ex::CallResult(_)
            | Ex::SubgroupBallotResult
            | Ex::RayQueryProceedResult => {}

            // Expressions that contain handles that need to be adjusted.
            Ex::Constant(ref mut constant) => self.constants.adjust(constant),
            Ex::ZeroValue(ref mut ty) => self.types.adjust(ty),
            Ex::Compose {
                ref mut ty,
                ref mut components,
            } => {
                self.types.adjust(ty);
                for component in components {
                    adjust(component);
                }
            }
            Ex::Access {
                ref mut base,
                ref mut index,
            } => {
                adjust(base);
                adjust(index);
            }
            Ex::AccessIndex {
                ref mut base,
                index: _,
            } => adjust(base),
            Ex::Splat {
                size: _,
                ref mut value,
            } => adjust(value),
            Ex::Swizzle {
                size: _,
                ref mut vector,
                pattern: _,
            } => adjust(vector),
            Ex::Load { ref mut pointer } => adjust(pointer),
            Ex::ImageSample {
                ref mut image,
                ref mut sampler,
                gather: _,
                ref mut coordinate,
                ref mut array_index,
                ref mut offset,
                ref mut level,
                ref mut depth_ref,
            } => {
                adjust(image);
                adjust(sampler);
                adjust(coordinate);
                operand_map.adjust_option(array_index);
                if let Some(ref mut offset) = *offset {
                    self.const_expressions.adjust(offset);
                }
                self.adjust_sample_level(level, operand_map);
                operand_map.adjust_option(depth_ref);
            }
            Ex::ImageLoad {
                ref mut image,
                ref mut coordinate,
                ref mut array_index,
                ref mut sample,
                ref mut level,
            } => {
                adjust(image);
                adjust(coordinate);
                operand_map.adjust_option(array_index);
                operand_map.adjust_option(sample);
                operand_map.adjust_option(level);
            }
            Ex::ImageQuery {
                ref mut image,
                ref mut query,
            } => {
                adjust(image);
                self.adjust_image_query(query, operand_map);
            }
            Ex::Unary {
                op: _,
                ref mut expr,
            } => adjust(expr),
            Ex::Binary {
                op: _,
                ref mut left,
                ref mut right,
            } => {
                adjust(left);
                adjust(right);
            }
            Ex::Select {
                ref mut condition,
                ref mut accept,
                ref mut reject,
            } => {
                adjust(condition);
                adjust(accept);
                adjust(reject);
            }
            Ex::Derivative {
                axis: _,
                ctrl: _,
                ref mut expr,
            } => adjust(expr),
            Ex::Relational {
                fun: _,
                ref mut argument,
            } => adjust(argument),
            Ex::Math {
                fun: _,
                ref mut arg,
                ref mut arg1,
                ref mut arg2,
                ref mut arg3,
            } => {
                adjust(arg);
                operand_map.adjust_option(arg1);
                operand_map.adjust_option(arg2);
                operand_map.adjust_option(arg3);
            }
            Ex::As {
                ref mut expr,
                kind: _,
                convert: _,
            } => adjust(expr),
            Ex::AtomicResult {
                ref mut ty,
                comparison: _,
            } => self.types.adjust(ty),
            Ex::WorkGroupUniformLoadResult { ref mut ty } => self.types.adjust(ty),
            Ex::SubgroupOperationResult { ref mut ty } => self.types.adjust(ty),
            Ex::ArrayLength(ref mut expr) => adjust(expr),
            Ex::RayQueryGetIntersection {
                ref mut query,
                committed: _,
            } => adjust(query),
        }
    }

    fn adjust_sample_level(
        &self,
        level: &mut crate::SampleLevel,
        operand_map: &HandleMap<crate::Expression>,
    ) {
        let adjust = |expr: &mut Handle<crate::Expression>| operand_map.adjust(expr);

        use crate::SampleLevel as Sl;
        match *level {
            Sl::Auto | Sl::Zero => {}
            Sl::Exact(ref mut expr) => adjust(expr),
            Sl::Bias(ref mut expr) => adjust(expr),
            Sl::Gradient {
                ref mut x,
                ref mut y,
            } => {
                adjust(x);
                adjust(y);
            }
        }
    }

    fn adjust_image_query(
        &self,
        query: &mut crate::ImageQuery,
        operand_map: &HandleMap<crate::Expression>,
    ) {
        use crate::ImageQuery as Iq;

        match *query {
            Iq::Size { ref mut level } => operand_map.adjust_option(level),
            Iq::NumLevels | Iq::NumLayers | Iq::NumSamples => {}
        }
    }
}
