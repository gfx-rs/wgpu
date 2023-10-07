use super::functions::FunctionTracer;
use super::FunctionMap;
use crate::arena::Handle;

impl FunctionTracer<'_> {
    pub fn trace_block(&mut self, block: &[crate::Statement]) {
        let mut worklist: Vec<&[crate::Statement]> = vec![block];
        while let Some(last) = worklist.pop() {
            for stmt in last {
                use crate::Statement as St;
                match *stmt {
                    St::Emit(ref _range) => {
                        // If we come across a statement that actually uses an
                        // expression in this range, it'll get traced from
                        // there. But since evaluating expressions has no
                        // effect, we don't need to assume that everything
                        // emitted is live.
                    }
                    St::Block(ref block) => worklist.push(block),
                    St::If {
                        condition,
                        ref accept,
                        ref reject,
                    } => {
                        self.trace_expression(condition);
                        worklist.push(accept);
                        worklist.push(reject);
                    }
                    St::Switch {
                        selector,
                        ref cases,
                    } => {
                        self.trace_expression(selector);
                        for case in cases {
                            worklist.push(&case.body);
                        }
                    }
                    St::Loop {
                        ref body,
                        ref continuing,
                        break_if,
                    } => {
                        if let Some(break_if) = break_if {
                            self.trace_expression(break_if);
                        }
                        worklist.push(body);
                        worklist.push(continuing);
                    }
                    St::Return { value: Some(value) } => self.trace_expression(value),
                    St::Store { pointer, value } => {
                        self.trace_expression(pointer);
                        self.trace_expression(value);
                    }
                    St::ImageStore {
                        image,
                        coordinate,
                        array_index,
                        value,
                    } => {
                        self.trace_expression(image);
                        self.trace_expression(coordinate);
                        if let Some(array_index) = array_index {
                            self.trace_expression(array_index);
                        }
                        self.trace_expression(value);
                    }
                    St::Atomic {
                        pointer,
                        ref fun,
                        value,
                        result,
                    } => {
                        self.trace_expression(pointer);
                        self.trace_atomic_function(fun);
                        self.trace_expression(value);
                        self.trace_expression(result);
                    }
                    St::WorkGroupUniformLoad { pointer, result } => {
                        self.trace_expression(pointer);
                        self.trace_expression(result);
                    }
                    St::Call {
                        function: _,
                        ref arguments,
                        result,
                    } => {
                        for expr in arguments {
                            self.trace_expression(*expr);
                        }
                        if let Some(result) = result {
                            self.trace_expression(result);
                        }
                    }
                    St::RayQuery { query, ref fun } => {
                        self.trace_expression(query);
                        self.trace_ray_query_function(fun);
                    }

                    // Trivial statements.
                    St::Break
                    | St::Continue
                    | St::Kill
                    | St::Barrier(_)
                    | St::Return { value: None } => {}
                }
            }
        }
    }

    fn trace_atomic_function(&mut self, fun: &crate::AtomicFunction) {
        use crate::AtomicFunction as Af;
        match *fun {
            Af::Exchange {
                compare: Some(expr),
            } => self.trace_expression(expr),
            Af::Exchange { compare: None }
            | Af::Add
            | Af::Subtract
            | Af::And
            | Af::ExclusiveOr
            | Af::InclusiveOr
            | Af::Min
            | Af::Max => {}
        }
    }

    fn trace_ray_query_function(&mut self, fun: &crate::RayQueryFunction) {
        use crate::RayQueryFunction as Qf;
        match *fun {
            Qf::Initialize {
                acceleration_structure,
                descriptor,
            } => {
                self.trace_expression(acceleration_structure);
                self.trace_expression(descriptor);
            }
            Qf::Proceed { result } => self.trace_expression(result),
            Qf::Terminate => {}
        }
    }
}

impl FunctionMap {
    pub fn adjust_body(&self, function: &mut crate::Function) {
        let block = &mut function.body;
        let mut worklist: Vec<&mut [crate::Statement]> = vec![block];
        let adjust = |handle: &mut Handle<crate::Expression>| {
            self.expressions.adjust(handle);
        };
        while let Some(last) = worklist.pop() {
            for stmt in last {
                use crate::Statement as St;
                match *stmt {
                    St::Emit(ref mut range) => {
                        self.expressions.adjust_range(range, &function.expressions);
                    }
                    St::Block(ref mut block) => worklist.push(block),
                    St::If {
                        ref mut condition,
                        ref mut accept,
                        ref mut reject,
                    } => {
                        adjust(condition);
                        worklist.push(accept);
                        worklist.push(reject);
                    }
                    St::Switch {
                        ref mut selector,
                        ref mut cases,
                    } => {
                        adjust(selector);
                        for case in cases {
                            worklist.push(&mut case.body);
                        }
                    }
                    St::Loop {
                        ref mut body,
                        ref mut continuing,
                        ref mut break_if,
                    } => {
                        if let Some(ref mut break_if) = *break_if {
                            adjust(break_if);
                        }
                        worklist.push(body);
                        worklist.push(continuing);
                    }
                    St::Return {
                        value: Some(ref mut value),
                    } => adjust(value),
                    St::Store {
                        ref mut pointer,
                        ref mut value,
                    } => {
                        adjust(pointer);
                        adjust(value);
                    }
                    St::ImageStore {
                        ref mut image,
                        ref mut coordinate,
                        ref mut array_index,
                        ref mut value,
                    } => {
                        adjust(image);
                        adjust(coordinate);
                        if let Some(ref mut array_index) = *array_index {
                            adjust(array_index);
                        }
                        adjust(value);
                    }
                    St::Atomic {
                        ref mut pointer,
                        ref mut fun,
                        ref mut value,
                        ref mut result,
                    } => {
                        adjust(pointer);
                        self.adjust_atomic_function(fun);
                        adjust(value);
                        adjust(result);
                    }
                    St::WorkGroupUniformLoad {
                        ref mut pointer,
                        ref mut result,
                    } => {
                        adjust(pointer);
                        adjust(result);
                    }
                    St::Call {
                        function: _,
                        ref mut arguments,
                        ref mut result,
                    } => {
                        for expr in arguments {
                            adjust(expr);
                        }
                        if let Some(ref mut result) = *result {
                            adjust(result);
                        }
                    }
                    St::RayQuery {
                        ref mut query,
                        ref mut fun,
                    } => {
                        adjust(query);
                        self.adjust_ray_query_function(fun);
                    }

                    // Trivial statements.
                    St::Break
                    | St::Continue
                    | St::Kill
                    | St::Barrier(_)
                    | St::Return { value: None } => {}
                }
            }
        }
    }

    fn adjust_atomic_function(&self, fun: &mut crate::AtomicFunction) {
        use crate::AtomicFunction as Af;
        match *fun {
            Af::Exchange {
                compare: Some(ref mut expr),
            } => {
                self.expressions.adjust(expr);
            }
            Af::Exchange { compare: None }
            | Af::Add
            | Af::Subtract
            | Af::And
            | Af::ExclusiveOr
            | Af::InclusiveOr
            | Af::Min
            | Af::Max => {}
        }
    }

    fn adjust_ray_query_function(&self, fun: &mut crate::RayQueryFunction) {
        use crate::RayQueryFunction as Qf;
        match *fun {
            Qf::Initialize {
                ref mut acceleration_structure,
                ref mut descriptor,
            } => {
                self.expressions.adjust(acceleration_structure);
                self.expressions.adjust(descriptor);
            }
            Qf::Proceed { ref mut result } => {
                self.expressions.adjust(result);
            }
            Qf::Terminate => {}
        }
    }
}
