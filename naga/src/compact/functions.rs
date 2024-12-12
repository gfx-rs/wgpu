use super::arena::HandleSet;
use super::{FunctionMap, ModuleMap};

pub struct FunctionTracer<'a> {
    pub function: &'a crate::Function,
    pub constants: &'a crate::Arena<crate::Constant>,

    pub types_used: &'a mut HandleSet<crate::Type>,
    pub constants_used: &'a mut HandleSet<crate::Constant>,
    pub global_expressions_used: &'a mut HandleSet<crate::Expression>,

    /// Function-local expressions used.
    pub expressions_used: HandleSet<crate::Expression>,
}

impl FunctionTracer<'_> {
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

    fn as_expression(&mut self) -> super::expressions::ExpressionTracer {
        super::expressions::ExpressionTracer {
            constants: self.constants,
            expressions: &self.function.expressions,

            types_used: self.types_used,
            constants_used: self.constants_used,
            expressions_used: &mut self.expressions_used,
            global_expressions_used: Some(&mut self.global_expressions_used),
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

        // Drop unused expressions, reusing existing storage.
        function.expressions.retain_mut(|handle, expr| {
            if self.expressions.used(handle) {
                module_map.adjust_expression(expr, &self.expressions);
                true
            } else {
                false
            }
        });

        // Adjust named expressions.
        for (mut handle, name) in function.named_expressions.drain(..) {
            self.expressions.adjust(&mut handle);
            reuse.insert(handle, name);
        }
        std::mem::swap(&mut function.named_expressions, reuse);
        assert!(reuse.is_empty());

        // Adjust statements.
        self.adjust_body(function);
    }
}
