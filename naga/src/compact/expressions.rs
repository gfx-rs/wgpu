use alloc::vec::Vec;

use super::{HandleMap, HandleSet, ModuleMap};
use crate::arena::{Arena, Handle};

pub struct ExpressionTracer<'tracer> {
    pub constants: &'tracer Arena<crate::Constant>,
    pub overrides: &'tracer Arena<crate::Override>,

    /// The arena in which we are currently tracing expressions.
    pub expressions: &'tracer Arena<crate::Expression>,

    /// The used map for `types`.
    pub types_used: &'tracer mut HandleSet<crate::Type>,

    /// The used map for global variables.
    pub global_variables_used: &'tracer mut HandleSet<crate::GlobalVariable>,

    /// The used map for `constants`.
    pub constants_used: &'tracer mut HandleSet<crate::Constant>,

    /// The used map for `overrides`.
    pub overrides_used: &'tracer mut HandleSet<crate::Override>,

    /// The used set for `arena`.
    ///
    /// This points to whatever arena holds the expressions we are
    /// currently tracing: either a function's expression arena, or
    /// the module's constant expression arena.
    pub expressions_used: &'tracer mut HandleSet<crate::Expression>,

    /// The used set for the module's `global_expressions` arena.
    ///
    /// If `None`, we are already tracing the constant expressions,
    /// and `expressions_used` already refers to their handle set.
    pub global_expressions_used: Option<&'tracer mut HandleSet<crate::Expression>>,

    /// The index of the first expression that [`trace_expressions`]'s
    /// back-to-front pass has already visited.
    ///
    /// Marking an expression at or after this index as used means that some
    /// expression refers to a later expression: a forward reference. See
    /// [`ForwardRefs`].
    ///
    /// [`trace_expressions`]: ExpressionTracer::trace_expressions
    pub visited_from: usize,

    /// Forward references found while tracing.
    pub forward_refs: &'tracer mut ForwardRefs,
}

/// Forward references found while tracing an expression arena.
///
/// A valid module has no forward references: every expression refers only to
/// expressions that precede it in the arena. But compaction also accepts
/// function expression arenas that violate this rule, as long as the
/// function's statements are otherwise valid, and reorders their expressions
/// to make them valid again. See [`Reorder`].
///
/// [`Reorder`]: super::reorder::Reorder
#[derive(Default)]
pub struct ForwardRefs {
    /// Whether any used expression refers to a later expression.
    pub seen: bool,

    /// Expressions marked as used after the back-to-front pass had already
    /// visited them. They still need to be traced.
    deferred: Vec<Handle<crate::Expression>>,
}

impl ExpressionTracer<'_> {
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
    /// [ce]: crate::Module::global_expressions
    pub fn trace_expressions(&mut self) {
        log::trace!(
            "entering trace_expression of {}",
            if self.global_expressions_used.is_some() {
                "function expressions"
            } else {
                "const expressions"
            }
        );

        // In a valid module, an expression may only refer to other
        // expressions that precede it in the arena, so it suffices to make a
        // single pass over the arena from back to front, marking the referents
        // of used expressions as used themselves.
        for (handle, expr) in self.expressions.iter().rev() {
            // If this expression isn't used, it doesn't matter what it uses.
            if !self.expressions_used.contains(handle) {
                continue;
            }

            log::trace!("tracing new expression {expr:?}");
            self.visited_from = handle.index() + 1;
            self.trace_expression(expr);
        }

        // Function expression arenas may contain forward references, though.
        // An expression that only became used after the pass had visited it
        // has not been traced yet, so trace it now. Everything has been
        // visited at this point, so any expression newly marked from here on
        // is deferred as well.
        self.visited_from = 0;
        while let Some(handle) = self.forward_refs.deferred.pop() {
            log::trace!("tracing forward-referenced expression {handle:?}");
            self.trace_expression(&self.expressions[handle]);
        }
    }

    /// Mark `handle` as used, deferring it if the back-to-front pass in
    /// [`trace_expressions`] has already visited it.
    ///
    /// [`trace_expressions`]: ExpressionTracer::trace_expressions
    fn mark(&mut self, handle: Handle<crate::Expression>) {
        let newly_used = self.expressions_used.insert(handle);
        if handle.index() >= self.visited_from {
            self.forward_refs.seen = true;
            if newly_used {
                self.forward_refs.deferred.push(handle);
            }
        }
    }

    fn mark_iter(&mut self, iter: impl IntoIterator<Item = Handle<crate::Expression>>) {
        for handle in iter {
            self.mark(handle);
        }
    }

    pub fn trace_expression(&mut self, expr: &crate::Expression) {
        use crate::Expression as Ex;
        match *expr {
            // Expressions that do not contain handles that need to be traced.
            Ex::Literal(_)
            | Ex::FunctionArgument(_)
            | Ex::LocalVariable(_)
            | Ex::SubgroupBallotResult
            | Ex::RayQueryProceedResult => {}

            // Expressions can refer to constants and overrides, which can refer
            // in turn to expressions, which complicates our nice one-pass
            // algorithm. But since constants and overrides don't refer to each
            // other directly, only via expressions, we can get around this by
            // looking *through* each constant/override and marking its
            // initializer expression as used immediately. Since `expr` refers
            // to the constant/override, which then refers to the initializer,
            // the initializer must precede `expr` in the arena, so we know we
            // have yet to visit the initializer, so it's not too late to mark
            // it.
            Ex::Constant(handle) => {
                self.constants_used.insert(handle);
                let constant = &self.constants[handle];
                self.types_used.insert(constant.ty);
                match self.global_expressions_used {
                    Some(ref mut used) => used.insert(constant.init),
                    None => self.expressions_used.insert(constant.init),
                };
            }
            Ex::Override(handle) => {
                self.overrides_used.insert(handle);
                let r#override = &self.overrides[handle];
                self.types_used.insert(r#override.ty);
                if let Some(init) = r#override.init {
                    match self.global_expressions_used {
                        Some(ref mut used) => used.insert(init),
                        None => self.expressions_used.insert(init),
                    };
                }
            }
            Ex::ZeroValue(ty) => {
                self.types_used.insert(ty);
            }
            Ex::Compose { ty, ref components } => {
                self.types_used.insert(ty);
                self.mark_iter(components.iter().cloned());
            }
            Ex::Access { base, index } => self.mark_iter([base, index]),
            Ex::AccessIndex { base, index: _ } => {
                self.mark(base);
            }
            Ex::Splat { size: _, value } => {
                self.mark(value);
            }
            Ex::Swizzle {
                size: _,
                vector,
                pattern: _,
            } => {
                self.mark(vector);
            }
            Ex::GlobalVariable(handle) => {
                self.global_variables_used.insert(handle);
            }
            Ex::Load { pointer } => {
                self.mark(pointer);
            }
            Ex::ImageSample {
                image,
                sampler,
                gather: _,
                coordinate,
                array_index,
                offset,
                ref level,
                depth_ref,
                clamp_to_edge: _,
            } => {
                self.mark_iter([image, sampler, coordinate]);
                self.mark_iter(array_index);
                self.mark_iter(offset);
                use crate::SampleLevel as Sl;
                match *level {
                    Sl::Auto | Sl::Zero => {}
                    Sl::Exact(expr) | Sl::Bias(expr) => {
                        self.mark(expr);
                    }
                    Sl::Gradient { x, y } => self.mark_iter([x, y]),
                }
                self.mark_iter(depth_ref);
            }
            Ex::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => {
                self.mark(image);
                self.mark(coordinate);
                self.mark_iter(array_index);
                self.mark_iter(sample);
                self.mark_iter(level);
            }
            Ex::ImageQuery { image, ref query } => {
                self.mark(image);
                use crate::ImageQuery as Iq;
                match *query {
                    Iq::Size { level } => self.mark_iter(level),
                    Iq::NumLevels | Iq::NumLayers | Iq::NumSamples => {}
                }
            }
            Ex::RayQueryVertexPositions {
                query,
                committed: _,
            } => {
                self.mark(query);
            }
            Ex::Unary { op: _, expr } => {
                self.mark(expr);
            }
            Ex::Binary { op: _, left, right } => {
                self.mark_iter([left, right]);
            }
            Ex::Select {
                condition,
                accept,
                reject,
            } => self.mark_iter([condition, accept, reject]),
            Ex::Derivative {
                axis: _,
                ctrl: _,
                expr,
            } => {
                self.mark(expr);
            }
            Ex::Relational { fun: _, argument } => {
                self.mark(argument);
            }
            Ex::Math {
                fun: _,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                self.mark(arg);
                self.mark_iter(arg1);
                self.mark_iter(arg2);
                self.mark_iter(arg3);
            }
            Ex::As {
                expr,
                kind: _,
                convert: _,
            } => {
                self.mark(expr);
            }
            Ex::ArrayLength(expr) => {
                self.mark(expr);
            }
            // `CallResult` expressions do contain a function handle, but any used
            // `CallResult` expression should have an associated `ir::Statement::Call`
            // that we will trace.
            Ex::CallResult(_) => {}
            Ex::AtomicResult { ty, comparison: _ }
            | Ex::WorkGroupUniformLoadResult { ty }
            | Ex::SubgroupOperationResult { ty } => {
                self.types_used.insert(ty);
            }
            Ex::RayQueryGetIntersection {
                query,
                committed: _,
            } => {
                self.mark(query);
            }
            Ex::CooperativeLoad { ref data, .. } => {
                self.mark(data.pointer);
                self.mark(data.stride);
            }
            Ex::CooperativeMultiplyAdd { a, b, c } => {
                self.mark(a);
                self.mark(b);
                self.mark(c);
            }
        }
    }
}

/// Call `f` on each expression that `expr` refers to directly.
pub fn for_each_operand(expr: &crate::Expression, mut f: impl FnMut(Handle<crate::Expression>)) {
    use crate::Expression as Ex;
    match *expr {
        // Expressions that do not refer to other expressions.
        Ex::Literal(_)
        | Ex::Constant(_)
        | Ex::Override(_)
        | Ex::ZeroValue(_)
        | Ex::FunctionArgument(_)
        | Ex::GlobalVariable(_)
        | Ex::LocalVariable(_)
        | Ex::CallResult(_)
        | Ex::AtomicResult { .. }
        | Ex::WorkGroupUniformLoadResult { .. }
        | Ex::SubgroupBallotResult
        | Ex::SubgroupOperationResult { .. }
        | Ex::RayQueryProceedResult => {}

        Ex::Compose {
            ty: _,
            ref components,
        } => components.iter().copied().for_each(f),
        Ex::Access { base, index } => {
            f(base);
            f(index);
        }
        Ex::AccessIndex { base, index: _ } => f(base),
        Ex::Splat { size: _, value } => f(value),
        Ex::Swizzle {
            size: _,
            vector,
            pattern: _,
        } => f(vector),
        Ex::Load { pointer } => f(pointer),
        Ex::ImageSample {
            image,
            sampler,
            gather: _,
            coordinate,
            array_index,
            offset,
            ref level,
            depth_ref,
            clamp_to_edge: _,
        } => {
            f(image);
            f(sampler);
            f(coordinate);
            array_index.into_iter().for_each(&mut f);
            offset.into_iter().for_each(&mut f);
            use crate::SampleLevel as Sl;
            match *level {
                Sl::Auto | Sl::Zero => {}
                Sl::Exact(expr) | Sl::Bias(expr) => f(expr),
                Sl::Gradient { x, y } => {
                    f(x);
                    f(y);
                }
            }
            depth_ref.into_iter().for_each(f);
        }
        Ex::ImageLoad {
            image,
            coordinate,
            array_index,
            sample,
            level,
        } => {
            f(image);
            f(coordinate);
            array_index.into_iter().for_each(&mut f);
            sample.into_iter().for_each(&mut f);
            level.into_iter().for_each(f);
        }
        Ex::ImageQuery { image, ref query } => {
            f(image);
            use crate::ImageQuery as Iq;
            match *query {
                Iq::Size { level } => level.into_iter().for_each(f),
                Iq::NumLevels | Iq::NumLayers | Iq::NumSamples => {}
            }
        }
        Ex::Unary { op: _, expr } => f(expr),
        Ex::Binary { op: _, left, right } => {
            f(left);
            f(right);
        }
        Ex::Select {
            condition,
            accept,
            reject,
        } => {
            f(condition);
            f(accept);
            f(reject);
        }
        Ex::Derivative {
            axis: _,
            ctrl: _,
            expr,
        } => f(expr),
        Ex::Relational { fun: _, argument } => f(argument),
        Ex::Math {
            fun: _,
            arg,
            arg1,
            arg2,
            arg3,
        } => {
            f(arg);
            arg1.into_iter().for_each(&mut f);
            arg2.into_iter().for_each(&mut f);
            arg3.into_iter().for_each(f);
        }
        Ex::As {
            expr,
            kind: _,
            convert: _,
        } => f(expr),
        Ex::ArrayLength(expr) => f(expr),
        Ex::RayQueryGetIntersection {
            query,
            committed: _,
        }
        | Ex::RayQueryVertexPositions {
            query,
            committed: _,
        } => f(query),
        Ex::CooperativeLoad { ref data, .. } => {
            f(data.pointer);
            f(data.stride);
        }
        Ex::CooperativeMultiplyAdd { a, b, c } => {
            f(a);
            f(b);
            f(c);
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
            | Ex::LocalVariable(_)
            | Ex::SubgroupBallotResult
            | Ex::RayQueryProceedResult => {}

            // Expressions that contain handles that need to be adjusted.
            Ex::Constant(ref mut constant) => self.constants.adjust(constant),
            Ex::Override(ref mut r#override) => self.overrides.adjust(r#override),
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
            Ex::GlobalVariable(ref mut handle) => self.globals.adjust(handle),
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
                clamp_to_edge: _,
            } => {
                adjust(image);
                adjust(sampler);
                adjust(coordinate);
                operand_map.adjust_option(array_index);
                operand_map.adjust_option(offset);
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
            Ex::CallResult(ref mut function) => {
                self.functions.adjust(function);
            }
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
            Ex::RayQueryVertexPositions {
                ref mut query,
                committed: _,
            } => adjust(query),
            Ex::CooperativeLoad { ref mut data, .. } => {
                adjust(&mut data.pointer);
                adjust(&mut data.stride);
            }
            Ex::CooperativeMultiplyAdd {
                ref mut a,
                ref mut b,
                ref mut c,
            } => {
                adjust(a);
                adjust(b);
                adjust(c);
            }
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
