//! Helpers for passes that build statements and the expressions they use,
//! and for passes that rewrite loads into new expressions.

use core::mem;

use crate::{
    arena::{Arena, Range},
    Block, Expression, FastHashSet, Handle, Span, Statement,
};

/// Appends expressions to a function, and `Emit` statements that evaluate
/// them to a block.
pub(super) struct Emitting<'a> {
    pub(super) expressions: &'a mut Arena<Expression>,
    pub(super) block: &'a mut Block,
    /// The first expression not yet covered by an `Emit`.
    pub(super) start: usize,
    pub(super) span: Span,
}

impl<'a> Emitting<'a> {
    pub(super) const fn new(
        expressions: &'a mut Arena<Expression>,
        block: &'a mut Block,
        span: Span,
    ) -> Self {
        Self {
            start: expressions.len(),
            expressions,
            block,
            span,
        }
    }

    /// Append `expr`. The next [`flush`](Self::flush) emits it.
    pub(super) fn expr(&mut self, expr: Expression) -> Handle<Expression> {
        self.expressions.append(expr, self.span)
    }

    /// Append `expr`, which must not be covered by an `Emit`, such as a literal
    /// or a pointer to a variable.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "no pass on this branch pre-emits expressions")
    )]
    pub(super) fn pre_emit(&mut self, expr: Expression) -> Handle<Expression> {
        self.flush();
        let handle = self.expressions.append(expr, self.span);
        self.start = self.expressions.len();
        handle
    }

    /// Emit the pending expressions, then push `statement`.
    pub(super) fn push(&mut self, statement: Statement) {
        self.flush();
        self.block.push(statement, self.span);
    }

    /// Emit the expressions appended since the last `Emit`.
    pub(super) fn flush(&mut self) {
        if self.start < self.expressions.len() {
            let range = self.expressions.range_from(self.start);
            self.block.push(Statement::Emit(range), self.span);
            self.start = self.expressions.len();
        }
    }
}

/// Rewrite the `loads` in `block` and the blocks nested in it.
///
/// `check` is called on every statement first. `rewrite` is called on each
/// load, with an [`Emitting`] whose expressions are evaluated just before the
/// load. It returns whether the load's handle must still be emitted.
pub(super) fn rewrite_loads<E>(
    block: &mut Block,
    expressions: &mut Arena<Expression>,
    loads: &FastHashSet<Handle<Expression>>,
    check: &mut impl FnMut(&Statement) -> Result<(), E>,
    rewrite: &mut impl FnMut(&mut Emitting, Handle<Expression>) -> Result<bool, E>,
) -> Result<(), E> {
    for (mut statement, span) in mem::take(block).span_into_iter() {
        check(&statement)?;
        match statement {
            Statement::Emit(ref range) => {
                if range.clone().any(|handle| loads.contains(&handle)) {
                    let range = range.clone();
                    split_emit(block, expressions, loads, range, span, rewrite)?;
                    continue;
                }
            }
            Statement::Block(ref mut inner) => {
                rewrite_loads(inner, expressions, loads, check, rewrite)?;
            }
            Statement::If {
                ref mut accept,
                ref mut reject,
                ..
            } => {
                rewrite_loads(accept, expressions, loads, check, rewrite)?;
                rewrite_loads(reject, expressions, loads, check, rewrite)?;
            }
            Statement::Switch { ref mut cases, .. } => {
                for case in cases {
                    rewrite_loads(&mut case.body, expressions, loads, check, rewrite)?;
                }
            }
            Statement::Loop {
                ref mut body,
                ref mut continuing,
                ..
            } => {
                rewrite_loads(body, expressions, loads, check, rewrite)?;
                rewrite_loads(continuing, expressions, loads, check, rewrite)?;
            }
            _ => {}
        }
        block.push(statement, span);
    }
    Ok(())
}

/// Push `range` to `block`, rewriting the loads in it. Each load's new
/// expressions get an `Emit` of their own, just before the load.
fn split_emit<E>(
    block: &mut Block,
    expressions: &mut Arena<Expression>,
    loads: &FastHashSet<Handle<Expression>>,
    range: Range<Expression>,
    span: Span,
    rewrite: &mut impl FnMut(&mut Emitting, Handle<Expression>) -> Result<bool, E>,
) -> Result<(), E> {
    let mut start = range.index_range().start;
    for load in range.clone().filter(|handle| loads.contains(handle)) {
        let at = load.index() as u32;
        push_emit(block, expressions, start..at, span);
        let mut out = Emitting::new(expressions, block, expressions.get_span(load));
        let emit_load = rewrite(&mut out, load)?;
        out.flush();
        start = if emit_load { at } else { at + 1 };
    }
    push_emit(block, expressions, start..range.index_range().end, span);
    Ok(())
}

pub(super) fn push_emit(
    block: &mut Block,
    expressions: &Arena<Expression>,
    range: core::ops::Range<u32>,
    span: Span,
) {
    if !range.is_empty() {
        block.push(
            Statement::Emit(Range::from_index_range(range, expressions)),
            span,
        );
    }
}

/// Call `f` on the expressions `statement` uses directly. These include every
/// pointer a statement can take, but not the operands nested in its function
/// payload, such as a ray query's.
pub(super) fn for_each_direct_operand(
    statement: &Statement,
    mut f: impl FnMut(Handle<Expression>),
) {
    match *statement {
        Statement::If { condition, .. } => f(condition),
        Statement::Switch { selector, .. } => f(selector),
        Statement::Loop { break_if, .. } => break_if.into_iter().for_each(f),
        Statement::Return { value } => value.into_iter().for_each(f),
        Statement::Store { pointer, value } => {
            f(pointer);
            f(value);
        }
        Statement::ImageStore {
            image,
            coordinate,
            array_index,
            value,
        }
        | Statement::ImageAtomic {
            image,
            coordinate,
            array_index,
            value,
            ..
        } => {
            f(image);
            f(coordinate);
            array_index.into_iter().for_each(&mut f);
            f(value);
        }
        Statement::Atomic { pointer, value, .. } => {
            f(pointer);
            f(value);
        }
        Statement::WorkGroupUniformLoad { pointer, .. } => f(pointer),
        Statement::Call { ref arguments, .. } | Statement::DebugPrintf { ref arguments, .. } => {
            arguments.iter().copied().for_each(f)
        }
        Statement::RayQuery { query, .. } => f(query),
        Statement::HitObject { hit_object, .. } => f(hit_object),
        Statement::SubgroupBallot { predicate, .. } => predicate.into_iter().for_each(f),
        Statement::SubgroupGather { argument, .. }
        | Statement::SubgroupCollectiveOperation { argument, .. }
        | Statement::SubgroupBallotFindBit { argument, .. } => f(argument),
        Statement::CooperativeStore { target, .. } => f(target),
        Statement::Emit(_)
        | Statement::Block(_)
        | Statement::Break
        | Statement::Continue
        | Statement::Kill
        | Statement::ControlBarrier(_)
        | Statement::MemoryBarrier(_)
        | Statement::RayPipelineFunction(_) => {}
    }
}

/// The base of `expr`, an access.
pub(super) fn access_base(expr: &Expression) -> Handle<Expression> {
    match *expr {
        Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => base,
        _ => unreachable!("only accesses are rewritten as value accesses"),
    }
}

/// `expr`, an access, with its base replaced by `base`.
pub(super) fn with_base(expr: &Expression, base: Handle<Expression>) -> Expression {
    match *expr {
        Expression::Access { index, .. } => Expression::Access { base, index },
        Expression::AccessIndex { index, .. } => Expression::AccessIndex { base, index },
        _ => unreachable!("only accesses are rewritten as value accesses"),
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::String;

    use super::Emitting;
    use crate::valid::{Capabilities, ValidationFlags, Validator};
    use crate::{
        BinaryOperator, Block, EntryPoint, Expression, Function, Literal, LocalVariable, Module,
        Scalar, ShaderStage, Span, Statement, Type, TypeInner,
    };

    #[test]
    fn builds_valid_body() {
        let mut module = Module::default();
        let f32 = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::F32),
            },
            Span::UNDEFINED,
        );
        let mut function = Function::default();
        let local = function.local_variables.append(
            LocalVariable {
                name: None,
                ty: f32,
                init: None,
            },
            Span::UNDEFINED,
        );

        let mut body = Block::new();
        let mut out = Emitting::new(&mut function.expressions, &mut body, Span::UNDEFINED);
        let one = out.pre_emit(Expression::Literal(Literal::F32(1.0)));
        let sum = out.expr(Expression::Binary {
            op: BinaryOperator::Add,
            left: one,
            right: one,
        });
        // Pre-emitted expressions stay out of the pending `Emit` range.
        let pointer = out.pre_emit(Expression::LocalVariable(local));
        let doubled = out.expr(Expression::Binary {
            op: BinaryOperator::Multiply,
            left: sum,
            right: sum,
        });
        out.push(Statement::Store {
            pointer,
            value: doubled,
        });
        out.push(Statement::Return { value: None });
        function.body = body;

        let emitted: alloc::vec::Vec<_> = function
            .body
            .iter()
            .filter_map(|statement| match *statement {
                Statement::Emit(ref range) => Some(range.index_range()),
                _ => None,
            })
            .collect();
        assert_eq!(emitted, [1..2, 3..4]);

        module.entry_points.push(EntryPoint {
            name: String::from("main"),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [1, 1, 1],
            workgroup_size_overrides: None,
            function,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap();
    }
}
