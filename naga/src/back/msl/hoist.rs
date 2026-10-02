/*!
Finding temporaries that must be declared outside a loop.

Naga IR's [`Loop`] statement runs its `continuing` block after its `body`, but
MSL has no equivalent construct, so [`Writer::put_block`] emits the `continuing`
block at the *top* of a `while` loop, skipped on the first iteration:

```ignore
bool loop_init = true;
while(true) {
    if (!loop_init) {
        <continuing>
        if (<break_if>) { break; }
    }
    loop_init = false;
    <body>
}
```

Expressions emitted in `body` are in scope in `continuing`, but the temporaries
holding their values are declared *below* the generated `continuing` code, so
they can't be named there. Worse, falling back to re-evaluating such an
expression in place would read whatever the body left behind at the end of the
iteration, which is not what the expression means.

So when a loop has a `continuing` block, we declare the temporaries the
`continuing` block needs ahead of the `while` loop, and have the body merely
assign to them.

[`Loop`]: crate::Statement::Loop
[`Writer::put_block`]: super::writer::Writer::put_block
*/

use core::fmt::Write;

use super::{writer::StatementContext, BackendResult, Writer};
use crate::{
    arena::{Arena, Handle, HandleSet},
    back,
    compact::ModuleTracer,
    proc::TypeResolution,
};

impl<W: Write> Writer<W> {
    /// Declare the temporaries that a loop's `continuing` block needs ahead of
    /// the loop.
    ///
    /// Record them in [`Self::hoisted_loop_expressions`], so that `continuing` can
    /// name them and the body assigns to them rather than declaring them. Each
    /// entry is moved to [`Self::named_expressions`] when the body writes the
    /// expression's definition, which it always does, since every candidate
    /// comes from a statement in `body`.
    ///
    /// See [the module documentation](self) for why this is needed.
    pub(super) fn hoist_continuing_dependencies(
        &mut self,
        level: back::Level,
        body: &crate::Block,
        continuing: &crate::Block,
        break_if: Option<Handle<crate::Expression>>,
        context: &StatementContext,
    ) -> BackendResult {
        let context = &context.expression;
        let expressions = &context.function.expressions;

        // The candidates are the definitions that get a temporary at all.
        //
        // Skipping the rest is deliberate: `continuing` re-evaluates those in
        // place, and that is sound, because every expression whose value
        // depends on *when* it is evaluated already gets a temporary. Loads,
        // samples and derivatives have a `bake_ref_count` of one, so any use at
        // all bakes them, and statement results have no inline form. What's
        // left to re-evaluate is arithmetic and address computation.

        let mut candidates = HandleSet::for_arena(expressions);
        for handle in definitions(expressions, body).iter() {
            if self.bakes_expression(handle, context)
                && fits_in_temporary(context.module, context.info, handle)
            {
                candidates.insert(handle);
            }
        }
        if candidates.is_empty() {
            return Ok(());
        }

        let used = expressions_used(
            context.module,
            context.function,
            continuing,
            break_if,
            &candidates,
        );

        for handle in candidates.iter() {
            if !used.contains(handle) {
                continue;
            }
            let Some(name) = self.baked_name(handle, context) else {
                continue;
            };

            write!(self.out, "{level}")?;
            self.put_baked_type(handle, context)?;
            writeln!(self.out, " {name} = {{}};")?;

            self.hoisted_loop_expressions.insert(handle, name);
        }

        Ok(())
    }
}

/// Return the expressions that `block` itself computes values for, in
/// increasing handle order.
fn definitions(
    expressions: &Arena<crate::Expression>,
    block: &crate::Block,
) -> HandleSet<crate::Expression> {
    use crate::Statement as St;

    let mut defined = HandleSet::for_arena(expressions);
    for statement in block.iter() {
        match *statement {
            St::Emit(ref range) => defined.insert_iter(range.clone()),
            St::Atomic {
                result: Some(result),
                ..
            }
            | St::Call {
                result: Some(result),
                ..
            }
            | St::RayQuery {
                fun: crate::RayQueryFunction::Proceed { result },
                ..
            }
            | St::WorkGroupUniformLoad { result, .. }
            | St::SubgroupBallot { result, .. }
            | St::SubgroupGather { result, .. }
            | St::SubgroupCollectiveOperation { result, .. } => {
                defined.insert(result);
            }
            _ => {}
        }
    }
    defined
}

/// Return the expressions of `function` that the code generated for `block`
/// and `extra_roots` will refer to, directly or indirectly.
///
/// This is wider than what `block` needs the values of, because we do no dead
/// code elimination: [`Writer::put_block`] writes out a definition for every
/// expression it bakes, whether or not anything reads it, and that definition
/// refers to the expression's operands. Hence
/// [`trace_block_including_emits`] rather than [`trace_block`]. Miss those and
/// a dead definition would leave its operands un-hoisted, and `continuing`
/// would re-evaluate them in place - or, for a statement result, reach the
/// `unreachable!` in [`Writer::put_expression`], which has no inline form to
/// fall back on.
///
/// The search does not continue past an expression in `stop_at`: such an
/// expression is marked if something refers to it, but the search does not
/// follow it to its operands. Anything reachable only through it is therefore
/// left out.
///
/// [`trace_block_including_emits`]: crate::compact::functions::FunctionTracer::trace_block_including_emits
/// [`trace_block`]: crate::compact::functions::FunctionTracer::trace_block
/// [`Writer::put_block`]: super::writer::Writer::put_block
/// [`Writer::put_expression`]: super::writer::Writer::put_expression
fn expressions_used(
    module: &crate::Module,
    function: &crate::Function,
    block: &crate::Block,
    extra_roots: impl IntoIterator<Item = Handle<crate::Expression>>,
    stop_at: &HandleSet<crate::Expression>,
) -> HandleSet<crate::Expression> {
    let mut module_tracer = ModuleTracer::new(module);

    let mut function_tracer = module_tracer.as_function(function);
    function_tracer.trace_block_including_emits(block);
    function_tracer.expressions_used.insert_iter(extra_roots);

    let mut expression_tracer = function_tracer.as_expression();
    for (handle, expression) in function.expressions.iter().rev() {
        if expression_tracer.expressions_used.contains(handle) && !stop_at.contains(handle) {
            expression_tracer.trace_expression(expression);
        }
    }

    function_tracer.expressions_used
}

/// Whether `expression` is the result of a statement.
///
/// There is no way to write these inline, so the statement that produces one
/// always stores it in a temporary, whether or not anything asked for one.
pub(super) const fn is_statement_result(expression: &crate::Expression) -> bool {
    use crate::Expression as Ex;
    matches!(
        *expression,
        Ex::CallResult(_)
            | Ex::AtomicResult { .. }
            | Ex::WorkGroupUniformLoadResult { .. }
            | Ex::SubgroupBallotResult
            | Ex::SubgroupOperationResult { .. }
            | Ex::RayQueryProceedResult
    )
}

/// Whether a value of `handle`'s type can be held in a default-initialized
/// local temporary.
///
/// Anything this rejects simply doesn't get hoisted.
fn fits_in_temporary(
    module: &crate::Module,
    info: &crate::valid::FunctionInfo,
    handle: Handle<crate::Expression>,
) -> bool {
    use crate::TypeInner as Ti;
    match info[handle].ty {
        TypeResolution::Handle(ty) => matches!(
            module.types[ty].inner,
            Ti::Scalar(_)
                | Ti::Vector { .. }
                | Ti::Matrix { .. }
                | Ti::Array {
                    size: crate::ArraySize::Constant(_),
                    ..
                }
                | Ti::Struct { .. }
                | Ti::CooperativeMatrix { .. }
        ),
        TypeResolution::Value(
            Ti::Scalar(_) | Ti::Vector { .. } | Ti::Matrix { .. } | Ti::CooperativeMatrix { .. },
        ) => true,
        TypeResolution::Value(_) => false,
    }
}
