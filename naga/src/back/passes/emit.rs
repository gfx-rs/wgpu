//! A helper for passes that build statements and the expressions they use.

use crate::{arena::Arena, Block, Expression, Handle, Span, Statement};

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
