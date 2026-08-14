mod adjust;

use core::unreachable;

use crate::{
    AddressSpace, Block, Expression, Function, Handle, LocalVariable, Module, Span, Statement,
    Type, TypeInner,
};
use nt::FastHashMap;

/// Which functions should be inlined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum InlineStrategy {
    /// Only inline functions that need to be inlined for shaders using unrestricted_pointer_parameters
    /// to be compiled.
    #[default]
    PointerParametersOnly,
    /// Inline all functions.
    All,
}

struct InlineState<'a> {
    pub module: &'a mut Module,
    /// Collect all functions that may need to be inlined, but inline them lazily, keeping
    /// track of which have already been inlined.
    pub funcs_needing_inline: FastHashMap<Handle<Function>, Option<Function>>,
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

pub fn inline(module: &mut Module, strategy: InlineStrategy) {
    let mut funcs_needing_inline: FastHashMap<Handle<Function>, Option<Function>>;
    if strategy == InlineStrategy::All {
        funcs_needing_inline = module.functions.iter().map(|e| (e.0, None)).collect();
    } else {
        funcs_needing_inline = Default::default();
        for (handle, func) in module.functions.iter() {
            let args = &func.arguments;
            let needs_inline = args
                .iter()
                .any(|arg| type_needs_unrestricted_pointer_params(module, r#arg.ty));
            if needs_inline {
                funcs_needing_inline.insert(handle, None);
            }
        }
    };
    let mut state = InlineState {
        module,
        funcs_needing_inline,
    };
    for i in 0..state.module.entry_points.len() {
        // Take these out so we can modify them while using the rest of the module without using unsafe code.
        let mut function = core::mem::take(&mut state.module.entry_points[i].function);

        state.inline_all_calls(&mut function, false);

        state.module.entry_points[i].function = function;
    }
}

impl InlineState<'_> {
    fn calls_inlinable_functions(&self, block: &Block) -> bool {
        for st in &block.body {
            match *st {
                Statement::Call { function, .. } => {
                    if self.funcs_needing_inline.contains_key(&function) {
                        return true;
                    }
                }
                Statement::Block(ref b) => {
                    if self.calls_inlinable_functions(b) {
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
                    if self.calls_inlinable_functions(b1) || self.calls_inlinable_functions(b2) {
                        return true;
                    }
                }
                Statement::Switch {
                    selector: _,
                    ref cases,
                } => {
                    if cases
                        .iter()
                        .any(|e| self.calls_inlinable_functions(&e.body))
                    {
                        return true;
                    }
                }
                _ => (),
            }
        }
        false
    }

    /// Inline all function calls in a function, after each of those have been recursively inlined and then prepared.
    ///
    /// Also optionally prepare the function for itself being inlined in a caller. This involves transforming the
    /// control flow.
    fn inline_all_calls(&mut self, function: &mut Function, prepare_for_self_inline: bool) {
        if !prepare_for_self_inline {
            // If it doesn't need to be inlined itself, and doesn't need any of its calls inlined,
            // skip.
            if !self.calls_inlinable_functions(&function.body) {
                return;
            }
        }

        // Reconstruct the statements from scratch, though we may
        let mut new_block = Block::new();
        let bool_type = self.module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(crate::Scalar::BOOL),
            },
            Span::UNDEFINED,
        );
        // Variable that is referenced in `if(is_done_var) {break}` statements emitted after
        // switches and loops, in which the break is captured.
        let is_done_var = function.local_variables.append(
            LocalVariable {
                name: None,
                ty: bool_type,
                init: None,
            },
            Span::UNDEFINED,
        );
        let is_done_var_ptr = function
            .expressions
            .append(Expression::LocalVariable(is_done_var), Span::UNDEFINED);
        let false_val = function.expressions.append(
            Expression::Literal(crate::Literal::Bool(false)),
            Span::UNDEFINED,
        );
        let true_val = function.expressions.append(
            Expression::Literal(crate::Literal::Bool(true)),
            Span::UNDEFINED,
        );
        for (st, span) in function
            .body
            .body
            .into_iter()
            .zip(function.body.span_info.into_iter())
        {
            match st {
                Statement::Call {
                    function: handle,
                    ref result,
                    ref arguments,
                } => {
                    if matches!(self.funcs_needing_inline.get(&handle), Some(None)) {
                        let function = self.module.functions[handle].clone();

                        let function = self.inline_all_calls(function, true);

                        self.funcs_needing_inline.insert(handle, Some(function));
                    }
                    let Some(prepared) = self.funcs_needing_inline.get(&handle) else {
                        new_block.push(st, span);
                        continue;
                    };
                    let prepared = prepared.as_ref().unwrap();
                    let call_result_var = prepared.result.as_ref().map(|r| {
                        let var = function.local_variables.append(
                            LocalVariable {
                                name: None,
                                ty: r.ty,
                                init: None,
                            },
                            span,
                        );
                        function.expressions[result.unwrap()] = Expression::LocalVariable(var);
                        var
                    });
                    new_block.push(
                        Statement::Store {
                            pointer: is_done_var_ptr,
                            value: false_val,
                        },
                        span,
                    );
                    let expr_offset = function.expressions.len() as u32;
                    let local_variable_offset = function.local_variables.len() as u32;
                    let statement_offset = function.body.body.len();
                    for (_, expr, span) in prepared.expressions.iter_span() {
                        function.expressions.append(expr.clone(), *span);
                    }
                    for (_, var, span) in prepared.local_variables.iter_span() {
                        function.local_variables.append(var.clone(), *span);
                    }
                    function.body.body.extend_from_slice(&prepared.body.body);
                    function
                        .body
                        .span_info
                        .extend_from_slice(&prepared.body.span_info);

                    let adjust_info = adjust::AdjustInfo {
                        function_args: arguments,
                        function_return: *result,
                        expressions: &mut function.expressions,
                        statements: &mut function.body.body[statement_offset..],
                        expr_offset,
                        local_variable_offset,
                    };
                    adjust_info.adjust_all();

                    // TODO: should we `Emit` the call result expression which is now a LocalVariable expression?
                }
                Statement::Return { value } if prepare_for_self_inline => {
                    if let Some(v) = value {
                        todo!()
                    }
                }
                _ => new_block.push(st, span),
            }
        }
        // TODO: wrap in for loop in some cases
        function.body = new_block;
    }
}
