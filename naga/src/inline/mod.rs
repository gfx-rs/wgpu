use core::unreachable;

use crate::{
    valid::{FunctionInfo, ModuleInfo},
    Block, Function, Handle, Module, Span, Statement, Type, UniqueArena,
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

struct InlineStackItem {
    pub func: Handle<Function>,
    pub statement_index: usize,
}

struct InlineState<'a> {
    pub module: &'a mut Module,
    pub info: &'a mut ModuleInfo,
    /// Collect all functions that may need to be inlined, but inline them lazily, keeping
    /// track of which have already been inlined.
    pub funcs_needing_inline: FastHashMap<Handle<Function>, bool>,
}

/// Perform the inlining pass on a module. It may leave some unused variables and blocks behind,
/// so should be used with the compaction pass.
pub fn inline(module: &mut Module, info: &mut ModuleInfo, strategy: InlineStrategy) {
    let mut funcs_needing_inline: FastHashMap<Handle<Function>, bool>;
    if strategy == InlineStrategy::All {
        funcs_needing_inline = module.functions.iter().map(|e| (e.0, false)).collect();
    } else {
        funcs_needing_inline = Default::default();
        for (handle, func) in module.functions.iter() {
            let args = &func.arguments;
            let needs_inline = args
                .iter()
                .any(|arg| parameter_needs_unrestricted_pointer_params(&module.types, r#arg.ty));
            if needs_inline {
                funcs_needing_inline.insert(handle, false);
            }
        }
    };
    let mut state = InlineState {
        module,
        info,
        funcs_needing_inline,
    };
    for i in 0..state.module.entry_points.len() {
        // Take these out so we can modify them while using the rest of the module without using unsafe code.
        let function = core::mem::take(&mut state.module.entry_points[i].function);
        let mut function_info = std::mem::take(&mut state.info.entry_points[i]);

        let function = state.inline_all_calls(function, &mut function_info, false);

        state.module.entry_points[i].function = function;
        state.info.entry_points[i] = function_info;
    }
}

fn parameter_needs_unrestricted_pointer_params(
    arena: &UniqueArena<Type>,
    r#type: Handle<Type>,
) -> bool {
    core::todo!()
}

impl InlineState<'_> {
    /// Inline all function calls in a function, after each of those have been recursively inlined and then prepared.
    fn inline_all_calls(
        &mut self,
        mut function: Function,
        info: &mut FunctionInfo,
        prepare_for_self_inline: bool,
    ) -> Function {
        if !prepare_for_self_inline {
            // If it doesn't need to be inlined itself, and doesn't need any of its calls inlined,
            // skip.
            'a: {
                for st in &function.body {
                    if let &Statement::Call { function, .. } = st {
                        if self.funcs_needing_inline.contains_key(&function) {
                            break 'a;
                        }
                    }
                }
                return function;
            }
        }

        let mut new_block = Block::new();
        let bool_type = self.module.types.insert(
            Type {
                name: None,
                inner: crate::TypeInner::Scalar(crate::Scalar::BOOL),
            },
            Span::UNDEFINED,
        );
        let mut is_done_var = function.local_variables.append(
            crate::LocalVariable {
                name: None,
                ty: bool_type,
                init: None,
            },
            Span::UNDEFINED,
        );
        let is_done_var_ptr = function.expressions.append(
            crate::Expression::LocalVariable(is_done_var),
            Span::UNDEFINED,
        );
        let false_val = function.expressions.append(
            crate::Expression::Literal(crate::Literal::Bool(false)),
            Span::UNDEFINED,
        );
        let true_val = function.expressions.append(
            crate::Expression::Literal(crate::Literal::Bool(true)),
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
                    if self.funcs_needing_inline.get(&handle) == Some(&false) {
                        self.funcs_needing_inline.insert(handle, true);

                        let function = core::mem::take(&mut self.module.functions[handle]);
                        let mut info = std::mem::take(&mut self.info.functions[handle.index()]);

                        let function = self.inline_all_calls(function, &mut info, true);

                        self.module.functions[handle] = function;
                        self.info.functions[handle.index()] = info;
                    }
                    if self.funcs_needing_inline.contains_key(&handle) {
                        new_block.push(
                            Statement::Store {
                                pointer: is_done_var_ptr,
                                value: false_val,
                            },
                            span,
                        );
                        // Copy-paste the function body inside of a for-loop.
                        // We will have to reuse adjust_body from the compact module to realign all local variable and expression indices.
                        new_block.push(
                            Statement::Loop {
                                body: todo!(),
                                continuing: Block::default(),
                                break_if: Some(true_val),
                            },
                            span,
                        );
                    } else {
                        new_block.push(st, span);
                    }
                }
                _ => new_block.push(st, span),
            }
        }
        function.body = new_block;
        // Update the expressions in info
        function
    }
}
