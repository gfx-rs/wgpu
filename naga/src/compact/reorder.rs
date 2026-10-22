//! Reordering a function's expressions so that every expression precedes its
//! users.
//!
//! Naga requires that an [`Expression`] appear in its arena before every
//! expression that refers to it (see [`Function::expressions`]). Passes that
//! rewrite a function body, such as inlining, find it convenient to append new
//! expressions at the end of the arena and let existing expressions refer to
//! them, producing *forward references*. Compaction accepts such arenas and
//! renumbers the function's expressions so that the rule holds again.
//!
//! The new order is determined by walking the function body: expressions are
//! placed in the order in which they are emitted, and any expression an
//! emitted expression depends on that has not been placed already is placed
//! just before the [`Emit`] statement's range. This keeps the expressions
//! covered by each [`Emit`] statement contiguous, and free of expressions that
//! must not be emitted, so the statement can be adjusted to cover the
//! renumbered expressions.
//!
//! [`Emit`]: crate::Statement::Emit
//! [`Function::expressions`]: crate::Function::expressions

use alloc::{vec, vec::Vec};

use super::expressions::for_each_operand;
use super::handle_set_map::HandleMap;
use crate::arena::{Arena, Handle, HandleSet, Range};
use crate::{Expression, FastHashMap, Function, Statement};

type Index = crate::non_max_u32::NonMaxU32;

/// The post-compaction order of a function's expressions, when compaction has
/// to reorder them.
pub struct Reorder {
    /// Pre-compaction handles of the used expressions, in post-compaction
    /// order.
    order: Vec<Handle<Expression>>,

    /// The post-compaction index range covered by each [`Emit`] statement,
    /// keyed by the bounds of its pre-compaction range.
    ///
    /// [`Emit`]: crate::Statement::Emit
    emit_ranges: FastHashMap<(u32, u32), (u32, u32)>,
}

impl Reorder {
    /// Choose a post-compaction order for the used expressions of `function`.
    ///
    /// Return a map from pre-compaction to post-compaction handles, and the
    /// `Reorder` needed to rebuild the arena and adjust `Emit` statements.
    pub fn new(function: &Function, used: &HandleSet<Expression>) -> (HandleMap<Expression>, Self) {
        let mut builder = Builder {
            expressions: &function.expressions,
            used,
            map: HandleMap::all_unused(function.expressions.len()),
            order: Vec::with_capacity(function.expressions.len()),
            visiting: HandleSet::for_arena(&function.expressions),
            stack: Vec::new(),
            emit_ranges: FastHashMap::default(),
        };

        builder.walk(&function.body);

        // Anything used only outside `Emit` statements: local variable
        // initializers, named expressions, pre-emitted expressions referred to
        // only by statements, and so on.
        for handle in used.iter() {
            builder.place_with_operands(handle);
        }

        let reorder = Reorder {
            order: builder.order,
            emit_ranges: builder.emit_ranges,
        };
        (builder.map, reorder)
    }

    /// Adjust `range`, from an `Emit` statement in the pre-compaction
    /// function, to cover the same expressions in the compacted function.
    pub fn adjust_range(&self, range: &mut Range<Expression>, compacted_arena: &Arena<Expression>) {
        let old = range.index_range();
        let &(start, end) = self
            .emit_ranges
            .get(&(old.start, old.end))
            .expect("every `Emit` statement was visited when choosing the order");
        *range = Range::from_index_range(start..end, compacted_arena);
    }

    /// Replace the contents of `expressions` with the used expressions in
    /// their new order, applying `adjust` to each one as it is moved.
    pub fn rebuild(
        &self,
        expressions: &mut Arena<Expression>,
        mut adjust: impl FnMut(&mut Expression),
    ) {
        let mut old: Vec<Option<_>> = expressions
            .drain()
            .map(|(_, expr, span)| Some((expr, span)))
            .collect();
        for &handle in &self.order {
            let (mut expr, span) = old[handle.index()].take().unwrap();
            adjust(&mut expr);
            expressions.append(expr, span);
        }
    }
}

struct Builder<'a> {
    expressions: &'a Arena<Expression>,
    used: &'a HandleSet<Expression>,

    /// The new index of each expression placed so far.
    map: HandleMap<Expression>,

    /// Pre-compaction handles in post-compaction order.
    order: Vec<Handle<Expression>>,

    /// Expressions in `stack` that are waiting for their operands to be
    /// placed. Used to detect cycles, which a valid module cannot have.
    visiting: HandleSet<Expression>,

    /// Scratch stack for `place_with_operands`.
    stack: Vec<Handle<Expression>>,

    emit_ranges: FastHashMap<(u32, u32), (u32, u32)>,
}

impl Builder<'_> {
    /// Visit the `Emit` statements in `body` and its nested blocks, in order.
    fn walk(&mut self, body: &[Statement]) {
        // A stack of the statement lists we are in the middle of visiting.
        // Nested blocks are pushed in reverse so that they are visited in
        // order.
        let mut pending = vec![body.iter()];
        while let Some(statements) = pending.last_mut() {
            let Some(statement) = statements.next() else {
                pending.pop();
                continue;
            };
            match *statement {
                Statement::Emit(ref range) => self.emit(range),
                Statement::Block(ref block) => pending.push(block.iter()),
                Statement::If {
                    condition: _,
                    ref accept,
                    ref reject,
                } => {
                    pending.push(reject.iter());
                    pending.push(accept.iter());
                }
                Statement::Switch {
                    selector: _,
                    ref cases,
                } => {
                    for case in cases.iter().rev() {
                        pending.push(case.body.iter());
                    }
                }
                Statement::Loop {
                    ref body,
                    ref continuing,
                    break_if: _,
                } => {
                    pending.push(continuing.iter());
                    pending.push(body.iter());
                }
                // No other statements contain blocks. Their operands are either
                // emitted before them, in which case they have been placed
                // already, or are pre-emitted expressions with no operands of
                // their own, which can be placed anywhere.
                Statement::Break
                | Statement::Continue
                | Statement::Return { .. }
                | Statement::Kill
                | Statement::ControlBarrier(_)
                | Statement::MemoryBarrier(_)
                | Statement::Store { .. }
                | Statement::ImageStore { .. }
                | Statement::Atomic { .. }
                | Statement::ImageAtomic { .. }
                | Statement::WorkGroupUniformLoad { .. }
                | Statement::Call { .. }
                | Statement::RayQuery { .. }
                | Statement::SubgroupBallot { .. }
                | Statement::SubgroupGather { .. }
                | Statement::SubgroupCollectiveOperation { .. }
                | Statement::CooperativeStore { .. }
                | Statement::RayPipelineFunction(_)
                | Statement::DebugPrintf { .. } => {}
            }
        }
    }

    /// Place the used expressions in `range`, and record the new range.
    fn emit(&mut self, range: &Range<Expression>) {
        let old = range.index_range();
        let key = (old.start, old.end);
        // The same range may be emitted in two different blocks, for example
        // in both branches of an `If`. Both refer to the same expressions.
        if self.emit_ranges.contains_key(&key) {
            return;
        }
        // An `Emit` must cover exactly the expressions it evaluates. Anything
        // else the range's expressions depend on that has not been placed yet
        // (pre-emitted expressions like literals and local variables, or the
        // results of statements like `Call`) must be placed before the range.
        let expressions = self.expressions;
        let in_range = |handle: Handle<Expression>| old.contains(&(handle.index() as u32));
        for handle in range.clone() {
            if self.used.contains(handle) {
                for_each_operand(&expressions[handle], |operand| {
                    if !self.map.used(operand) && !in_range(operand) {
                        self.place_with_operands(operand);
                    }
                });
            }
        }

        let start = self.order.len() as u32;
        for handle in range.clone() {
            if self.used.contains(handle) {
                self.place_with_operands(handle);
            }
        }
        let end = self.order.len() as u32;
        self.emit_ranges.insert(key, (start, end));
    }

    /// Place `root`, after placing any of its operands that have not been
    /// placed yet, and their operands, and so on.
    fn place_with_operands(&mut self, root: Handle<Expression>) {
        debug_assert!(self.used.contains(root));
        if self.map.used(root) {
            return;
        }
        let expressions = self.expressions;
        self.stack.push(root);
        while let Some(&handle) = self.stack.last() {
            if self.map.used(handle) {
                self.stack.pop();
                continue;
            }
            let depth = self.stack.len();
            for_each_operand(&expressions[handle], |operand| {
                if !self.map.used(operand) {
                    assert!(
                        !self.visiting.contains(operand),
                        "expression {operand:?} depends on itself"
                    );
                    self.stack.push(operand);
                }
            });
            if self.stack.len() == depth {
                // All operands are placed, so `handle` can follow them.
                self.stack.pop();
                self.visiting.remove(handle);
                self.place(handle);
            } else {
                self.visiting.insert(handle);
            }
        }
    }

    fn place(&mut self, handle: Handle<Expression>) {
        let index = Index::new(self.order.len() as u32).unwrap();
        self.map.insert(handle, index);
        self.order.push(handle);
    }
}

#[cfg(test)]
mod tests {
    use alloc::{boxed::Box, string::String, vec::Vec};

    use crate::compact::{compact, KeepUnused, ModuleTracer};
    use crate::valid::{Capabilities, ModuleInfo, ValidationError, ValidationFlags, Validator};
    use crate::{
        BinaryOperator, EntryPoint, Expression, Function, Handle, Literal, LocalVariable, Module,
        Range, Scalar, ShaderStage, Span, Statement, Type, TypeInner, WithSpan,
    };

    fn validate(module: &Module) -> Result<ModuleInfo, Box<WithSpan<ValidationError>>> {
        Validator::new(ValidationFlags::all(), Capabilities::all()).validate(module)
    }

    fn scalar_type(module: &mut Module, scalar: Scalar) -> Handle<Type> {
        module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(scalar),
            },
            Span::default(),
        )
    }

    fn local(function: &mut Function, ty: Handle<Type>) -> Handle<LocalVariable> {
        function.local_variables.append(
            LocalVariable {
                name: None,
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    fn add(function: &mut Function, expr: Expression) -> Handle<Expression> {
        function.expressions.append(expr, Span::default())
    }

    /// The expression handle with index `index`.
    fn h(index: u32) -> Handle<Expression> {
        Handle::new(super::Index::new(index).unwrap())
    }

    fn emit(first: u32, end: u32) -> Statement {
        Statement::Emit(Range::new_from_bounds(h(first), h(end - 1)))
    }

    fn entry_point(function: Function) -> EntryPoint {
        EntryPoint {
            name: String::from("main"),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [1, 1, 1],
            workgroup_size_overrides: None,
            function,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        }
    }

    fn expressions(function: &Function) -> Vec<Expression> {
        function
            .expressions
            .iter()
            .map(|(_, e)| e.clone())
            .collect()
    }

    fn emit_range(statement: &Statement) -> core::ops::Range<u32> {
        match *statement {
            Statement::Emit(ref range) => range.index_range(),
            ref other => panic!("expected `Emit`, found {other:?}"),
        }
    }

    fn has_forward_references(module: &Module, function: &Function) -> bool {
        let mut tracer = ModuleTracer::new(module);
        let mut tracer = tracer.as_function(function);
        tracer.trace();
        tracer.forward_refs.seen
    }

    /// A `Load` that refers to a `LocalVariable` expression appended after
    /// it, as an inlining pass would produce.
    #[test]
    fn forward_reference_is_reordered() {
        let mut module = Module::default();
        let f32 = scalar_type(&mut module, Scalar::F32);
        let mut f = Function::default();
        let v = local(&mut f, f32);

        // e0
        let one = add(&mut f, Expression::Literal(Literal::F32(1.0)));
        // e1, refers forward to e3
        let load = add(&mut f, Expression::Load { pointer: h(3) });
        // e2
        let sum = add(
            &mut f,
            Expression::Binary {
                op: BinaryOperator::Add,
                left: load,
                right: one,
            },
        );
        // e3
        let pointer = add(&mut f, Expression::LocalVariable(v));

        f.body.push(emit(1, 3), Span::default());
        f.body.push(
            Statement::Store {
                pointer,
                value: sum,
            },
            Span::default(),
        );
        f.body
            .push(Statement::Return { value: None }, Span::default());
        module.entry_points.push(entry_point(f));

        assert!(validate(&module).is_err());
        assert!(has_forward_references(
            &module,
            &module.entry_points[0].function
        ));

        compact(&mut module, KeepUnused::No);

        validate(&module).expect("compacted module should be valid");
        let f = &module.entry_points[0].function;
        assert_eq!(
            expressions(f),
            [
                Expression::LocalVariable(v),
                Expression::Literal(Literal::F32(1.0)),
                Expression::Load { pointer: h(0) },
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: h(2),
                    right: h(1),
                },
            ]
        );
        assert_eq!(emit_range(&f.body[0]), 2..4);
        assert!(matches!(
            f.body[1],
            Statement::Store { pointer, value } if pointer == h(0) && value == h(3)
        ));
    }

    /// An expression reached only through a forward reference must still be
    /// traced, so that the expressions *it* refers to are kept.
    #[test]
    fn forward_reference_keeps_transitive_uses_alive() {
        let mut module = Module::default();
        let f32 = scalar_type(&mut module, Scalar::F32);
        let vec2 = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector {
                    size: crate::VectorSize::Bi,
                    scalar: Scalar::F32,
                },
            },
            Span::default(),
        );
        let mut f = Function::default();
        let v = local(&mut f, vec2);
        let out = local(&mut f, f32);

        // e0, refers forward to e2
        let load = add(&mut f, Expression::Load { pointer: h(2) });
        // e1, unused
        add(&mut f, Expression::Literal(Literal::F32(2.0)));
        // e2, refers forward to e3
        add(
            &mut f,
            Expression::AccessIndex {
                base: h(3),
                index: 0,
            },
        );
        // e3
        add(&mut f, Expression::LocalVariable(v));
        // e4
        let out_pointer = add(&mut f, Expression::LocalVariable(out));

        f.body.push(emit(2, 3), Span::default());
        f.body.push(emit(0, 1), Span::default());
        f.body.push(
            Statement::Store {
                pointer: out_pointer,
                value: load,
            },
            Span::default(),
        );
        f.body
            .push(Statement::Return { value: None }, Span::default());
        module.entry_points.push(entry_point(f));

        compact(&mut module, KeepUnused::No);

        validate(&module).expect("compacted module should be valid");
        let f = &module.entry_points[0].function;
        assert_eq!(
            expressions(f),
            [
                Expression::LocalVariable(v),
                Expression::AccessIndex {
                    base: h(0),
                    index: 0
                },
                Expression::Load { pointer: h(1) },
                Expression::LocalVariable(out),
            ]
        );
        assert_eq!(emit_range(&f.body[0]), 1..2);
        assert_eq!(emit_range(&f.body[1]), 2..3);
    }

    /// A pre-emitted expression inside an `Emit` range that an earlier `Emit`
    /// range depends on is placed before its first user, and the later range
    /// is adjusted to exclude it.
    #[test]
    fn dependency_pulled_out_of_later_emit_range() {
        let mut module = Module::default();
        let f32 = scalar_type(&mut module, Scalar::F32);
        let mut f = Function::default();
        let v = local(&mut f, f32);

        // e0, refers forward to e2
        let sum = add(
            &mut f,
            Expression::Binary {
                op: BinaryOperator::Add,
                left: h(2),
                right: h(2),
            },
        );
        // e1, unused
        add(&mut f, Expression::Literal(Literal::F32(2.0)));
        // e2
        add(&mut f, Expression::Literal(Literal::F32(1.0)));
        // e3
        let pointer = add(&mut f, Expression::LocalVariable(v));

        f.body.push(emit(0, 1), Span::default());
        f.body.push(emit(1, 3), Span::default());
        f.body.push(
            Statement::Store {
                pointer,
                value: sum,
            },
            Span::default(),
        );
        f.body
            .push(Statement::Return { value: None }, Span::default());
        module.entry_points.push(entry_point(f));

        compact(&mut module, KeepUnused::No);

        validate(&module).expect("compacted module should be valid");
        let f = &module.entry_points[0].function;
        assert_eq!(
            expressions(f),
            [
                Expression::Literal(Literal::F32(1.0)),
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: h(0),
                    right: h(0),
                },
                Expression::LocalVariable(v),
            ]
        );
        assert_eq!(emit_range(&f.body[0]), 1..2);
        assert_eq!(emit_range(&f.body[1]), 2..2);
    }

    /// The same range emitted in both branches of an `If` must map to the
    /// same compacted range.
    #[test]
    fn same_range_emitted_in_two_blocks() {
        let mut module = Module::default();
        let f32 = scalar_type(&mut module, Scalar::F32);
        let mut f = Function::default();
        let v = local(&mut f, f32);

        // e0
        let condition = add(&mut f, Expression::Literal(Literal::Bool(true)));
        // e1, refers forward to e3
        let sum = add(
            &mut f,
            Expression::Binary {
                op: BinaryOperator::Add,
                left: h(3),
                right: h(3),
            },
        );
        // e2
        let pointer = add(&mut f, Expression::LocalVariable(v));
        // e3
        add(&mut f, Expression::Literal(Literal::F32(1.0)));

        let branch = || {
            let mut block = crate::Block::new();
            block.push(emit(1, 2), Span::default());
            block.push(
                Statement::Store {
                    pointer,
                    value: sum,
                },
                Span::default(),
            );
            block
        };
        f.body.push(
            Statement::If {
                condition,
                accept: branch(),
                reject: branch(),
            },
            Span::default(),
        );
        f.body
            .push(Statement::Return { value: None }, Span::default());
        module.entry_points.push(entry_point(f));

        compact(&mut module, KeepUnused::No);

        validate(&module).expect("compacted module should be valid");
        let f = &module.entry_points[0].function;
        let (accept, reject) = match f.body[0] {
            Statement::If {
                ref accept,
                ref reject,
                ..
            } => (accept, reject),
            ref other => panic!("expected `If`, found {other:?}"),
        };
        assert_eq!(emit_range(&accept[0]), 1..2);
        assert_eq!(emit_range(&reject[0]), 1..2);
    }

    /// Forward references in a function that is only kept because of
    /// `KeepUnused::Yes`.
    #[test]
    fn forward_reference_in_unused_function() {
        let mut module = Module::default();
        let f32 = scalar_type(&mut module, Scalar::F32);
        let mut f = Function {
            name: Some(String::from("f")),
            result: Some(crate::FunctionResult {
                ty: f32,
                binding: None,
            }),
            ..Function::default()
        };

        // e0, refers forward to e1
        let product = add(
            &mut f,
            Expression::Binary {
                op: BinaryOperator::Multiply,
                left: h(1),
                right: h(1),
            },
        );
        // e1
        add(&mut f, Expression::Literal(Literal::F32(3.0)));

        f.body.push(emit(0, 1), Span::default());
        f.body.push(
            Statement::Return {
                value: Some(product),
            },
            Span::default(),
        );
        module.functions.append(f, Span::default());

        let mut dropped = module.clone();
        compact(&mut dropped, KeepUnused::No);
        assert!(dropped.functions.is_empty());

        compact(&mut module, KeepUnused::Yes);
        validate(&module).expect("compacted module should be valid");
        let (_, f) = module.functions.iter().next().unwrap();
        assert_eq!(
            expressions(f),
            [
                Expression::Literal(Literal::F32(3.0)),
                Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: h(0),
                    right: h(0),
                },
            ]
        );
        assert_eq!(emit_range(&f.body[0]), 1..2);
    }

    /// A module without forward references takes the ordinary path, which
    /// preserves the order of the expressions.
    #[cfg(feature = "wgsl-in")]
    #[test]
    fn no_forward_references_preserves_order() {
        let mut module = crate::front::wgsl::parse_str(
            "
            @compute @workgroup_size(1)
            fn main() {
                var a = 1.0;
                let unused = 2.0;
                a = a + 3.0;
            }
            ",
        )
        .unwrap();
        assert!(!has_forward_references(
            &module,
            &module.entry_points[0].function
        ));

        let before = expressions(&module.entry_points[0].function);
        compact(&mut module, KeepUnused::No);
        validate(&module).expect("compacted module should be valid");
        let after = expressions(&module.entry_points[0].function);

        // Compaction only removes expressions; it never moves them. So each
        // expression's kind should appear in the same order as before.
        let kinds = |exprs: &[Expression]| {
            exprs
                .iter()
                .map(core::mem::discriminant)
                .collect::<Vec<_>>()
        };
        let mut remaining = kinds(&before).into_iter();
        for kind in kinds(&after) {
            assert!(remaining.any(|k| k == kind), "expression was reordered");
        }
    }
}
