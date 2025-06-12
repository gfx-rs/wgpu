use core::{cell::Cell, mem};
use std::vec::Vec;

use crate::{ir, valid, FastHashMap, FastHashSet, Handle};

enum VariableState {
    Uninitialized,
    Initialized,
    Written,
    ReRead,
}

impl VariableState {
    fn write(&mut self) {
        match self {
            VariableState::Uninitialized => *self = VariableState::Initialized,
            VariableState::Initialized => *self = VariableState::Written,
            VariableState::ReRead => {}
            VariableState::Written => {}
        }
    }

    fn read(&mut self) {
        match self {
            VariableState::Uninitialized => unreachable!(),
            VariableState::Initialized => {}
            VariableState::Written => *self = VariableState::ReRead,
            VariableState::ReRead => {}
        }
    }
}

struct LoadBaker<'a> {
    pub module: &'a ir::Module,
    pub function: &'a ir::Function,
    pub function_info: &'a valid::FunctionInfo,

    requires_bake: FastHashSet<Handle<ir::Expression>>,

    states: FastHashMap<Handle<ir::Expression>, (usize, VariableState)>,

    loop_levels: Vec<usize>,
}

impl<'a> LoadBaker<'a> {
    pub fn new(
        module: &'a ir::Module,
        function: &'a ir::Function,
        function_info: &'a valid::FunctionInfo,
    ) -> Self {
        let requires_bake = FastHashSet::default();
        let states = FastHashMap::default();

        Self {
            module,
            function,
            function_info,
            requires_bake,
            states,
            loop_levels: Vec::new(),
        }
    }

    pub fn evaluate(&mut self) {
        let current_depth = ScopedDepth::new();
        self.evaluate_block(&current_depth, &self.function.body);
    }

    fn evaluate_block(&mut self, depth: &ScopedDepth, block: &'a ir::Block) {
        let _guard = depth.enter();
        for statement in block {
            match *statement {
                ir::Statement::Store { pointer, value } => {
                    self.evaluate_expression(depth, value);
                    self.register_write(depth, pointer);
                }
                ir::Statement::Emit(ref range) => {
                    for expression in range.clone() {
                        self.evaluate_expression(depth, expression);
                    }
                }
                ir::Statement::Block(ref block) => {
                    self.evaluate_block(&depth, block);
                }
                ir::Statement::Loop {
                    ref body,
                    ref continuing,
                    break_if,
                } => {
                    self.loop_levels.push(depth.get());

                    self.evaluate_block(&depth, body);
                    self.evaluate_block(&depth, continuing);
                    if let Some(condition) = break_if {
                        self.evaluate_expression(depth, condition);
                    }

                    self.loop_levels.pop();
                }
                _ => {}
            }
        }
    }

    fn evaluate_expression(&mut self, depth: &ScopedDepth, expression: Handle<ir::Expression>) {
        let expression = &self.function.expressions[expression];

        match *expression {
            ir::Expression::Load { pointer } => {
                let exp = &self.function.expressions[pointer];

                let is_local_variable = matches!(exp, ir::Expression::LocalVariable(_));

                if is_local_variable {
                    self.register_load(depth, pointer);
                }
            }
            _ => {}
        }
    }

    fn register_load(&mut self, depth: &ScopedDepth, pointer: Handle<ir::Expression>) {
        // Register the load as happening at the depth of the variable.
        if let Some((_, state)) = self.states.get_mut(&pointer) {
            state.read();
            return;
        }
    }

    fn register_write(&mut self, depth: &ScopedDepth, pointer: Handle<ir::Expression>) {}
}

struct ScopedDepth {
    current: Cell<usize>,
}

impl ScopedDepth {
    pub fn new() -> Self {
        Self {
            current: Cell::new(0),
        }
    }

    pub fn enter(&self) -> DepthGuard<'_> {
        self.current.set(self.current.get() + 1);
        DepthGuard {
            manager: self,
            depth: self.current.get(),
        }
    }

    pub fn get(&self) -> usize {
        self.current.get()
    }
}

struct DepthGuard<'a> {
    manager: &'a ScopedDepth,
    depth: usize,
}

impl Drop for DepthGuard<'_> {
    fn drop(&mut self) {
        self.manager.current.set(self.manager.current.get() - 1);
    }
}
