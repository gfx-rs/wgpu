use super::handle_set_map::HandleSet;
use super::{FunctionMap, ModuleMap};
use crate::arena::Handle;

pub(super) struct FunctionTracer<'a> {
    pub(super) module: &'a crate::Module,
    pub(super) function: &'a crate::Function,

    pub(super) types_used: &'a mut HandleSet<crate::Type>,
    pub(super) constants_used: &'a mut HandleSet<crate::Constant>,
    pub(super) const_expressions_used: &'a mut HandleSet<crate::Expression>,

    /// Function-local expressions used.
    pub(super) expressions_used: HandleSet<crate::Expression>,
}

impl<'a> FunctionTracer<'a> {
    pub fn trace(&mut self) {
        for argument in self.function.arguments.iter() {
            self.trace_type(argument.ty);
        }

        if let Some(ref result) = self.function.result {
            self.trace_type(result.ty);
        }

        for (_, local) in self.function.local_variables.iter() {
            self.trace_type(local.ty);
            if let Some(init) = local.init {
                // TEST: try changing this to trace_expression
                self.trace_const_expression(init);
            }
        }

        // Treat named expressions as alive, for the sake of our test suite,
        // which uses `let blah = expr;` to exercise lots of things.
        for (value, _name) in &self.function.named_expressions {
            self.trace_expression(*value);
        }

        self.trace_block(&self.function.body);
    }

    pub fn trace_type(&mut self, ty: Handle<crate::Type>) {
        self.as_type().trace_type(ty)
    }

    pub fn trace_expression(&mut self, expr: Handle<crate::Expression>) {
        self.as_expression().trace_expression(expr);
    }

    pub fn trace_const_expression(&mut self, expr: Handle<crate::Expression>) {
        self.as_expression()
            .as_const_expression()
            .trace_expression(expr);
    }

    /*
        pub fn trace_const_expression(&mut self, const_expr: Handle<crate::Expression>) {
            self.as_expression().as_const_expression().trace_expression(const_expr);
    }
        */

    fn as_type(&mut self) -> super::types::TypeTracer {
        super::types::TypeTracer {
            types: &self.module.types,
            types_used: self.types_used,
        }
    }

    fn as_expression(&mut self) -> super::expressions::ExpressionTracer {
        super::expressions::ExpressionTracer {
            types: &self.module.types,
            constants: &self.module.constants,
            expressions: &self.function.expressions,

            types_used: self.types_used,
            constants_used: self.constants_used,
            expressions_used: &mut self.expressions_used,
            const_expressions: Some((
                &self.module.const_expressions,
                &mut self.const_expressions_used,
            )),
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
                module_map.const_expressions.adjust(init);
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
        self.adjust_block(&mut function.body);
    }
}
