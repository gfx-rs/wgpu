use super::arena::HandleSet;
use super::expressions::ForwardRefs;
use super::reorder::Reorder;
use super::{FunctionMap, HandleMap, ModuleMap};

pub struct FunctionTracer<'a> {
    pub function: &'a crate::Function,
    pub constants: &'a crate::Arena<crate::Constant>,
    pub overrides: &'a crate::Arena<crate::Override>,

    pub functions_pending: &'a mut HandleSet<crate::Function>,
    pub functions_used: &'a mut HandleSet<crate::Function>,
    pub types_used: &'a mut HandleSet<crate::Type>,
    pub global_variables_used: &'a mut HandleSet<crate::GlobalVariable>,
    pub constants_used: &'a mut HandleSet<crate::Constant>,
    pub overrides_used: &'a mut HandleSet<crate::Override>,
    pub global_expressions_used: &'a mut HandleSet<crate::Expression>,

    /// Function-local expressions used.
    pub expressions_used: HandleSet<crate::Expression>,

    /// Forward references among the function's expressions.
    pub forward_refs: ForwardRefs,
}

impl FunctionTracer<'_> {
    pub fn trace_call(&mut self, function: crate::Handle<crate::Function>) {
        if !self.functions_used.contains(function) {
            self.functions_used.insert(function);
            self.functions_pending.insert(function);
        }
    }

    pub fn trace(&mut self) {
        for argument in self.function.arguments.iter() {
            self.types_used.insert(argument.ty);
        }

        if let Some(ref result) = self.function.result {
            self.types_used.insert(result.ty);
        }

        for (_, local) in self.function.local_variables.iter() {
            self.types_used.insert(local.ty);
            if let Some(init) = local.init {
                self.expressions_used.insert(init);
            }
        }

        // Treat named expressions as alive, for the sake of our test suite,
        // which uses `let blah = expr;` to exercise lots of things.
        for (&value, _name) in &self.function.named_expressions {
            self.expressions_used.insert(value);
        }

        self.trace_block(&self.function.body);

        // Given that `trace_block` has marked the expressions used
        // directly by statements, walk the arena to find all
        // expressions used, directly or indirectly.
        self.as_expression().trace_expressions();
    }

    const fn as_expression(&mut self) -> super::expressions::ExpressionTracer<'_> {
        super::expressions::ExpressionTracer {
            constants: self.constants,
            overrides: self.overrides,
            expressions: &self.function.expressions,

            types_used: self.types_used,
            global_variables_used: self.global_variables_used,
            constants_used: self.constants_used,
            overrides_used: self.overrides_used,
            expressions_used: &mut self.expressions_used,
            global_expressions_used: Some(&mut self.global_expressions_used),
            visited_from: usize::MAX,
            forward_refs: &mut self.forward_refs,
        }
    }
}

impl From<FunctionTracer<'_>> for FunctionMap {
    fn from(used: FunctionTracer) -> Self {
        if used.forward_refs.seen {
            let (expressions, reorder) = Reorder::new(used.function, &used.expressions_used);
            FunctionMap {
                expressions,
                reorder: Some(reorder),
            }
        } else {
            FunctionMap {
                expressions: HandleMap::from_set(used.expressions_used),
                reorder: None,
            }
        }
    }
}

impl FunctionMap {
    pub fn compact(
        &self,
        function: &mut crate::Function,
        module_map: &ModuleMap,
        reuse: &mut crate::NamedExpressions,
    ) {
        assert!(reuse.is_empty());

        for argument in function.arguments.iter_mut() {
            module_map.types.adjust(&mut argument.ty);
        }

        if let Some(ref mut result) = function.result {
            module_map.types.adjust(&mut result.ty);
        }

        for (_, local) in function.local_variables.iter_mut() {
            log::trace!("adjusting local variable {:?}", local.name);
            module_map.types.adjust(&mut local.ty);
            if let Some(ref mut init) = local.init {
                self.expressions.adjust(init);
            }
        }

        match self.reorder {
            None => {
                // Drop unused expressions, reusing existing storage.
                function.expressions.retain_mut(|handle, expr| {
                    if self.expressions.used(handle) {
                        module_map.adjust_expression(expr, &self.expressions);
                        true
                    } else {
                        false
                    }
                });
            }
            Some(ref reorder) => {
                log::trace!("reordering expressions of {:?}", function.name);
                reorder.rebuild(&mut function.expressions, |expr| {
                    module_map.adjust_expression(expr, &self.expressions);
                });
            }
        }

        // Adjust named expressions.
        for (mut handle, name) in function.named_expressions.drain(..) {
            self.expressions.adjust(&mut handle);
            reuse.insert(handle, name);
        }
        core::mem::swap(&mut function.named_expressions, reuse);
        assert!(reuse.is_empty());

        // Adjust statements.
        self.adjust_body(function, &module_map.functions);
    }
}
