mod adjust;

use alloc::{vec, vec::Vec};
use core::unreachable;

use crate::{
    non_max_u32::NonMaxU32, AddressSpace, Arena, Block, Expression, Function, Handle,
    LocalVariable, Module, Span, Statement, SwitchCase, Type, TypeInner,
};
use nt::FastHashMap;

/// Which functions should be inlined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum InlineStrategy {
    /// This is safe for certain backends.
    Never,
    /// Only inline functions that need to be inlined for shaders using unrestricted-pointer-parameters
    /// to be compiled.
    ///
    /// If you are compiling to a language other than HLSL (TODO: is it that or SPIR-V?),
    /// you should skip this pass entirely.
    #[default]
    OnlyNeededSpirv,
    /// Inline all functions.
    All,
}

struct InlineState<'a> {
    pub module: &'a mut Module,
    /// Collect all functions that may need to be inlined at the start, but inline them lazily, keeping
    /// track of which have already been inlined.
    pub funcs_needing_inline: Vec<bool>,
    pub inlined_funcs: FastHashMap<Handle<Function>, Function>,
}

pub fn type_needs_unrestricted_pointer_params(module: &Module, ty: Handle<Type>) -> bool {
    match module.types[ty].inner {
        TypeInner::Pointer { space, .. } | TypeInner::ValuePointer { space, .. } => {
            !matches!(space, AddressSpace::Function | AddressSpace::Private)
        }
        TypeInner::Struct { ref members, .. } => members
            .iter()
            .any(|m| type_needs_unrestricted_pointer_params(module, m.ty)),
        TypeInner::Array { base, .. } => type_needs_unrestricted_pointer_params(module, base),
        _ => false,
    }
}

/// Needs a compaction pass to run after it. The compaction pass will reorder expressions and statements within blocks,
/// and will remove functions that were inlined at all call sites so they don't appear.
pub fn inline(module: &mut Module, strategy: InlineStrategy) {
    if strategy == InlineStrategy::Never {
        return;
    }
    let funcs_needing_inline: Vec<bool>;
    if strategy == InlineStrategy::All {
        funcs_needing_inline = vec![true; module.functions.len()];
    } else {
        funcs_needing_inline = module
            .functions
            .iter()
            .map(|(_handle, func)| {
                let args = &func.arguments;
                let needs_inline = args
                    .iter()
                    .any(|arg| type_needs_unrestricted_pointer_params(module, r#arg.ty));
                needs_inline
            })
            .collect();
    };
    let mut state = InlineState {
        module,
        funcs_needing_inline,
        inlined_funcs: Default::default(),
    };
    for i in 0..state.module.entry_points.len() {
        // Take these out so we can modify them while using the rest of the module without using unsafe code.
        let mut function: Function = core::mem::take(&mut state.module.entry_points[i].function);

        state.inline_all_calls_full_function(&mut function, false);

        state.module.entry_points[i].function = function;
    }
}

impl InlineState<'_> {
    /// Unrestricted pointer parameters also allows new behavior which only needs to be inlined at some callsites.
    fn expression_needs_inlining(
        &self,
        arena: &Arena<Expression>,
        expr: Handle<Expression>,
    ) -> bool {
        let raw_expr = &arena[expr];

        // TODO: Verify this check
        match raw_expr {
            &Expression::Access { .. } | &Expression::AccessIndex { .. } => true,
            &Expression::Compose { ref components, .. } => components
                .iter()
                .any(|&expr| self.expression_needs_inlining(arena, expr)),
            &Expression::Select { accept, reject, .. } => {
                self.expression_needs_inlining(arena, accept)
                    || self.expression_needs_inlining(arena, reject)
            }
            _ => false,
        }
    }

    fn calls_inlinable_functions(&self, arena: &Arena<Expression>, block: &Block) -> bool {
        for st in &block.body {
            match *st {
                Statement::Call {
                    function,
                    ref arguments,
                    ..
                } => {
                    if self.funcs_needing_inline[function.index()]
                        || arguments
                            .iter()
                            .any(|&a| self.expression_needs_inlining(arena, a))
                    {
                        return true;
                    }
                }
                Statement::Block(ref b) => {
                    if self.calls_inlinable_functions(arena, b) {
                        return true;
                    }
                }
                Statement::If {
                    condition: _,
                    accept: ref b1,
                    reject: ref b2,
                }
                | Statement::Loop {
                    body: ref b1,
                    continuing: ref b2,
                    break_if: _,
                } => {
                    if self.calls_inlinable_functions(arena, b1)
                        || self.calls_inlinable_functions(arena, b2)
                    {
                        return true;
                    }
                }
                Statement::Switch {
                    selector: _,
                    ref cases,
                } => {
                    if cases
                        .iter()
                        .any(|e| self.calls_inlinable_functions(arena, &e.body))
                    {
                        return true;
                    }
                }
                _ => (),
            }
        }
        false
    }

    fn inline_all_calls_full_function(
        &mut self,
        func: &mut Function,
        prepare_for_self_inline: bool,
    ) {
        let bool_type = self.module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(crate::Scalar::BOOL),
            },
            Span::UNDEFINED,
        );
        // Variable that is referenced in `if(is_done_var) {break}` statements emitted after
        // switches and loops, in which the break is captured.
        let is_done_var = func.local_variables.append(
            LocalVariable {
                name: None,
                ty: bool_type,
                init: None,
            },
            Span::UNDEFINED,
        );
        let is_done_var_ptr = func
            .expressions
            .append(Expression::LocalVariable(is_done_var), Span::UNDEFINED);

        let false_val = func.expressions.append(
            Expression::Literal(crate::Literal::Bool(false)),
            Span::UNDEFINED,
        );

        let inline = self.inline_all_calls(
            &mut func.expressions,
            &mut func.local_variables,
            &mut func.body,
            is_done_var_ptr,
            prepare_for_self_inline,
        );
        if prepare_for_self_inline {
            let mut loop_block = Block::new();
            let inner_block = inline.unwrap_or_else(|| std::mem::take(&mut func.body));
            let full_span = Span::total_span(inner_block.span_info.iter().copied());
            loop_block.push(
                Statement::Loop {
                    body: inner_block,
                    continuing: Block::new(),
                    break_if: Some(false_val),
                },
                full_span,
            );
            // TODO: Figure out how to transfer the return value
            func.body = loop_block;
        } else {
            if let Some(inline) = inline {
                func.body = inline;
            }
        }
    }

    /// Inline all function calls in a function, after each of those have been recursively inlined and then prepared.
    ///
    /// Also optionally prepare the function for itself being inlined in a caller. This involves transforming the
    /// control flow.
    fn inline_all_calls(
        &mut self,
        expressions: &mut Arena<Expression>,
        local_variables: &mut Arena<LocalVariable>,
        old_body: &Block,
        is_done_var_ptr: Handle<Expression>,
        prepare_for_self_inline: bool,
    ) -> Option<Block> {
        if !prepare_for_self_inline && !self.calls_inlinable_functions(&expressions, &old_body) {
            // If it doesn't need to be inlined itself, and doesn't need any of its calls inlined,
            // skip.
            return None;
        }

        // Reconstruct the statements from scratch, though we may
        let mut new_block = Block::new();

        let false_val = expressions.append(
            Expression::Literal(crate::Literal::Bool(false)),
            Span::UNDEFINED,
        );

        // Correct here to use precomputed length, so it doesn't look at statements added
        // in this pass.
        for i in 0..old_body.body.len() {
            let st = &old_body.body[i];
            let span = old_body.span_info[i];

            let mut inlined_block = move |block: &Block| {
                self.inline_all_calls(expressions, local_variables, block, is_done_var_ptr, true)
                    .unwrap_or_else(|| block.clone())
            };
            match st {
                Statement::Call {
                    function: handle,
                    result,
                    ref arguments,
                } => {
                    let result = *result;
                    let needs_inline = self.funcs_needing_inline[handle.index()]
                        || arguments
                            .iter()
                            .any(|&a| self.expression_needs_inlining(&expressions, a));
                    if !needs_inline {
                        new_block.push(st.clone(), span);
                        continue;
                    }
                    if !self.inlined_funcs.contains_key(handle) {
                        let mut function = self.module.functions[*handle].clone();

                        self.inline_all_calls_full_function(&mut function, true);

                        self.inlined_funcs.insert(*handle, function);
                    }
                    let prepared = self.inlined_funcs.get(handle).unwrap();
                    let call_result_var = prepared.result.as_ref().map(|r| {
                        let var = local_variables.append(
                            LocalVariable {
                                name: None,
                                ty: r.ty,
                                init: None,
                            },
                            span,
                        );
                        expressions[result.unwrap()] = Expression::LocalVariable(var);
                        var
                    });
                    new_block.push(
                        Statement::Store {
                            pointer: is_done_var_ptr,
                            value: false_val,
                        },
                        span,
                    );
                    let expr_offset = expressions.len() as u32;
                    let local_variable_offset = local_variables.len() as u32;
                    let statement_offset = old_body.body.len();
                    for (_, expr, span) in prepared.expressions.iter_span() {
                        expressions.append(expr.clone(), *span);
                    }
                    for (_, var, span) in prepared.local_variables.iter_span() {
                        local_variables.append(var.clone(), *span);
                    }
                    new_block.body.extend_from_slice(&prepared.body.body);
                    new_block
                        .span_info
                        .extend_from_slice(&prepared.body.span_info);

                    let adjust_info = adjust::AdjustInfo {
                        function_args: arguments,
                        function_return: result,
                        expressions,
                        statements: &mut old_body.body[statement_offset..],
                        expr_offset,
                        local_variable_offset,
                    };
                    adjust_info.adjust_all();

                    // TODO: should we `Emit` the call result expression which is now a LocalVariable expression?
                }
                Statement::If {
                    accept,
                    reject,
                    condition,
                } => new_block.push(
                    Statement::If {
                        condition: *condition,
                        accept: inlined_block(accept),
                        reject: inlined_block(reject),
                    },
                    span,
                ),
                Statement::Switch { selector, cases } => new_block.push(
                    Statement::Switch {
                        selector: *selector,
                        cases: cases
                            .iter()
                            .map(|case| SwitchCase {
                                value: case.value,
                                fall_through: case.fall_through,
                                body: inlined_block(&case.body),
                            })
                            .collect(),
                    },
                    span,
                ),
                Statement::Block(old) => new_block.push(Statement::Block(inlined_block(old)), span),
                Statement::Loop {
                    body,
                    continuing,
                    break_if,
                } => {
                    new_block.push(
                        Statement::Loop {
                            body: inlined_block(body),
                            continuing: inlined_block(continuing),
                            break_if: *break_if,
                        },
                        span,
                    );
                    if prepare_for_self_inline {
                        // TODO: If the is_done_var is true, break again
                    }
                }
                Statement::Return { value } if prepare_for_self_inline => {
                    if let Some(v) = value {
                        todo!()
                    }
                }
                _ => new_block.push(st.clone(), span),
            }
        }
        Some(new_block)
    }
}
