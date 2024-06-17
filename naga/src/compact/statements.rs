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
                        self.expressions_used.insert(condition);
                        worklist.push(accept);
                        worklist.push(reject);
                    }
                    St::Switch {
                        selector,
                        ref cases,
                    } => {
                        self.expressions_used.insert(selector);
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
                            self.expressions_used.insert(break_if);
                        }
                        worklist.push(body);
                        worklist.push(continuing);
                    }
                    St::Return { value: Some(value) } => {
                        self.expressions_used.insert(value);
                    }
                    St::Store { pointer, value } => {
                        self.expressions_used.insert(pointer);
                        self.expressions_used.insert(value);
                    }
                    St::ImageStore {
                        image,
                        coordinate,
                        array_index,
                        value,
                    } => {
                        self.expressions_used.insert(image);
                        self.expressions_used.insert(coordinate);
                        if let Some(array_index) = array_index {
                            self.expressions_used.insert(array_index);
                        }
                        self.expressions_used.insert(value);
                    }
                    St::Atomic {
                        pointer,
                        ref fun,
                        value,
                        result,
                    } => {
                        self.expressions_used.insert(pointer);
                        self.trace_atomic_function(fun);
                        self.expressions_used.insert(value);
                        if let Some(result) = result {
                            self.expressions_used.insert(result);
                        }
                    }
                    St::WorkGroupUniformLoad { pointer, result } => {
                        self.expressions_used.insert(pointer);
                        self.expressions_used.insert(result);
                    }
                    St::Call {
                        function: _,
                        ref arguments,
                        result,
                    } => {
                        for expr in arguments {
                            self.expressions_used.insert(*expr);
                        }
                        if let Some(result) = result {
                            self.expressions_used.insert(result);
                        }
                    }
                    St::RayQuery { query, ref fun } => {
                        self.expressions_used.insert(query);
                        self.trace_ray_query_function(fun);
                    }
                    St::SubgroupBallot { result, predicate } => {
                        if let Some(predicate) = predicate {
                            self.expressions_used.insert(predicate)
                        }
                        self.expressions_used.insert(result)
                    }
                    St::SubgroupCollectiveOperation {
                        op: _,
                        collective_op: _,
                        argument,
                        result,
                    } => {
                        self.expressions_used.insert(argument);
                        self.expressions_used.insert(result)
                    }
                    St::SubgroupGather {
                        mode,
                        argument,
                        result,
                    } => {
                        match mode {
                            crate::GatherMode::BroadcastFirst => {}
                            crate::GatherMode::Broadcast(index)
                            | crate::GatherMode::Shuffle(index)
                            | crate::GatherMode::ShuffleDown(index)
                            | crate::GatherMode::ShuffleUp(index)
                            | crate::GatherMode::ShuffleXor(index) => {
                                self.expressions_used.insert(index)
                            }
                        }
                        self.expressions_used.insert(argument);
                        self.expressions_used.insert(result)
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
            } => {
                self.expressions_used.insert(expr);
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

    fn trace_ray_query_function(&mut self, fun: &crate::RayQueryFunction) {
        use crate::RayQueryFunction as Qf;
        match *fun {
            Qf::Initialize {
                acceleration_structure,
                descriptor,
            } => {
                self.expressions_used.insert(acceleration_structure);
                self.expressions_used.insert(descriptor);
            }
            Qf::Proceed { result } => {
                self.expressions_used.insert(result);
            }
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
                        if let Some(ref mut result) = *result {
                            adjust(result);
                        }
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
                    St::SubgroupBallot {
                        ref mut result,
                        ref mut predicate,
                    } => {
                        if let Some(ref mut predicate) = *predicate {
                            adjust(predicate);
                        }
                        adjust(result);
                    }
                    St::SubgroupCollectiveOperation {
                        op: _,
                        collective_op: _,
                        ref mut argument,
                        ref mut result,
                    } => {
                        adjust(argument);
                        adjust(result);
                    }
                    St::SubgroupGather {
                        ref mut mode,
                        ref mut argument,
                        ref mut result,
                    } => {
                        match *mode {
                            crate::GatherMode::BroadcastFirst => {}
                            crate::GatherMode::Broadcast(ref mut index)
                            | crate::GatherMode::Shuffle(ref mut index)
                            | crate::GatherMode::ShuffleDown(ref mut index)
                            | crate::GatherMode::ShuffleUp(ref mut index)
                            | crate::GatherMode::ShuffleXor(ref mut index) => adjust(index),
                        }
                        adjust(argument);
                        adjust(result);
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
