use super::{HandleMap, HandleSet, ModuleMap};
use crate::arena::{Arena, Handle};

pub struct ExpressionTracer<'tracer> {
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

    /// The used set for the module's `const_expressions` arena.
    ///
    /// If `None`, we are already tracing the constant expressions,
    /// and `expressions_used` already refers to their handle set.
    pub const_expressions_used: Option<&'tracer mut HandleSet<crate::Expression>>,
}

impl<'tracer> ExpressionTracer<'tracer> {
    /// Propagate usage through `self.expressions`, starting with `self.expressions_used`.
    ///
    /// Treat `self.expressions_used` as the initial set of "known
    /// live" expressions, and follow through to identify all
    /// transitively used expressions.
    ///
    /// Mark types, constants, and constant expressions used directly
    /// by `self.expressions` as used. Items used indirectly are not
    /// marked.
    ///
    /// [fe]: crate::Function::expressions
    /// [ce]: crate::Module::const_expressions
    pub fn trace_expressions(&mut self) {
        log::trace!(
            "entering trace_expression of {}",
            if self.const_expressions_used.is_some() {
                "function expressions"
            } else {
                "const expressions"
            }
        );

        // We don't need recursion or a work list. Because an
        // expression may only refer to other expressions that precede
        // it in the arena, it suffices to make a single pass over the
        // arena from back to front, marking the referents of used
        // expressions as used themselves.
        for (handle, expr) in self.expressions.iter().rev() {
            // If this expression isn't used, it doesn't matter what it uses.
            if !self.expressions_used.contains(handle) {
                continue;
            }

            log::trace!("tracing new expression {:?}", expr);

            use crate::Expression as Ex;
            match *expr {
                // Expressions that do not contain handles that need to be traced.
                Ex::Literal(_)
                | Ex::FunctionArgument(_)
                | Ex::GlobalVariable(_)
                | Ex::LocalVariable(_)
                | Ex::CallResult(_)
                | Ex::RayQueryProceedResult => {}

                Ex::Constant(handle) => {
                    self.constants_used.insert(handle);
                    // Constants and expressions are mutually recursive, which
                    // complicates our nice one-pass algorithm. However, since
                    // constants don't refer to each other, we can get around
                    // this by looking *through* each constant and marking its
                    // initializer as used. Since `expr` refers to the constant,
                    // and the constant refers to the initializer, it must
                    // precede `expr` in the arena.
                    let init = self.constants[handle].init;
                    match self.const_expressions_used {
                        Some(ref mut used) => used.insert(init),
                        None => self.expressions_used.insert(init),
                    }
                }
                Ex::ZeroValue(ty) => self.types_used.insert(ty),
                Ex::Compose { ty, ref components } => {
                    self.types_used.insert(ty);
                    self.expressions_used
                        .insert_iter(components.iter().cloned());
                }
                Ex::Access { base, index } => self.expressions_used.insert_iter([base, index]),
                Ex::AccessIndex { base, index: _ } => self.expressions_used.insert(base),
                Ex::Splat { size: _, value } => self.expressions_used.insert(value),
                Ex::Swizzle {
                    size: _,
                    vector,
                    pattern: _,
                } => self.expressions_used.insert(vector),
                Ex::Load { pointer } => self.expressions_used.insert(pointer),
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
                    self.expressions_used
                        .insert_iter([image, sampler, coordinate]);
                    self.expressions_used.insert_iter(array_index);
                    match self.const_expressions_used {
                        Some(ref mut used) => used.insert_iter(offset),
                        None => self.expressions_used.insert_iter(offset),
                    }
                    use crate::SampleLevel as Sl;
                    match *level {
                        Sl::Auto | Sl::Zero => {}
                        Sl::Exact(expr) | Sl::Bias(expr) => self.expressions_used.insert(expr),
                        Sl::Gradient { x, y } => self.expressions_used.insert_iter([x, y]),
                    }
                    self.expressions_used.insert_iter(depth_ref);
                }
                Ex::ImageLoad {
                    image,
                    coordinate,
                    array_index,
                    sample,
                    level,
                } => {
                    self.expressions_used.insert(image);
                    self.expressions_used.insert(coordinate);
                    self.expressions_used.insert_iter(array_index);
                    self.expressions_used.insert_iter(sample);
                    self.expressions_used.insert_iter(level);
                }
                Ex::ImageQuery { image, ref query } => {
                    self.expressions_used.insert(image);
                    use crate::ImageQuery as Iq;
                    match *query {
                        Iq::Size { level } => self.expressions_used.insert_iter(level),
                        Iq::NumLevels | Iq::NumLayers | Iq::NumSamples => {}
                    }
                }
                Ex::Unary { op: _, expr } => self.expressions_used.insert(expr),
                Ex::Binary { op: _, left, right } => {
                    self.expressions_used.insert_iter([left, right]);
                }
                Ex::Select {
                    condition,
                    accept,
                    reject,
                } => self
                    .expressions_used
                    .insert_iter([condition, accept, reject]),
                Ex::Derivative {
                    axis: _,
                    ctrl: _,
                    expr,
                } => self.expressions_used.insert(expr),
                Ex::Relational { fun: _, argument } => self.expressions_used.insert(argument),
                Ex::Math {
                    fun: _,
                    arg,
                    arg1,
                    arg2,
                    arg3,
                } => {
                    self.expressions_used.insert(arg);
                    self.expressions_used.insert_iter(arg1);
                    self.expressions_used.insert_iter(arg2);
                    self.expressions_used.insert_iter(arg3);
                }
                Ex::As {
                    expr,
                    kind: _,
                    convert: _,
                } => self.expressions_used.insert(expr),
                Ex::AtomicResult { ty, comparison: _ } => self.types_used.insert(ty),
                Ex::WorkGroupUniformLoadResult { ty } => self.types_used.insert(ty),
                Ex::ArrayLength(expr) => self.expressions_used.insert(expr),
                Ex::RayQueryGetIntersection {
                    query,
                    committed: _,
                } => self.expressions_used.insert(query),
            }
        }
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
