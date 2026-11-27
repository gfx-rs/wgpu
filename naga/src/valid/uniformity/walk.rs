//! Builds the uniformity graph of a function by walking its IR.
//!
//! This follows the rules of the WGSL spec's uniformity analysis. Each
//! expression is evaluated by the [`Statement::Emit`] that covers it, so the
//! control flow node of that statement stands in for the control flow node of
//! the spec's expression rules. Expressions that are never emitted are
//! constant, so they are uniform wherever they are used.

use alloc::{vec, vec::Vec};

use super::graph::{
    Graph, NodeId, NodeKind, Violation, CF_START, MAY_BE_NON_UNIFORM, REQUIRED_TO_BE_UNIFORM,
};
use super::tags::{
    expression_intrinsic, statement_intrinsic, FunctionSummary, Intrinsic, IntrinsicResult,
    ParamDependency, ParamSummary, PointerOut, Requirement, Triggers,
};
use super::{FunctionKind, NonUniformCause, Operation, RequirementTarget, UniformityError};
use crate::diagnostic_filter::{DiagnosticFilterNode, Severity, StandardFilterableTriggeringRule};
use crate::span::WithSpan;
use crate::valid::FunctionInfo;
use crate::{Block, Expression, Function, Handle, Module, Span, Statement};

bitflags::bitflags! {
    /// A WGSL spec statement behavior.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Behavior: u8 {
        const NEXT = 1 << 0;
        const BREAK = 1 << 1;
        const CONTINUE = 1 << 2;
        const RETURN = 1 << 3;
    }
}

/// Flow-sensitive facts at a point in a function.
///
/// Everything that must be joined where control flow merges belongs here.
#[derive(Clone, Debug)]
struct State {
    /// The value of each local variable, followed by the contents of each
    /// `function` address space pointer parameter.
    slots: Vec<NodeId>,
}

/// The states that leave a loop or switch statement other than by falling
/// through.
struct Exits {
    is_loop: bool,
    breaks: Vec<State>,
    continues: Vec<State>,
}

enum Root {
    Slot(usize),
    Global(Handle<crate::GlobalVariable>),
    PrivateArgument,
    Unknown,
}

const SEVERITIES: [Severity; 3] = [Severity::Error, Severity::Warning, Severity::Info];

const fn severity_index(severity: Severity) -> Option<usize> {
    match severity {
        Severity::Error => Some(0),
        Severity::Warning => Some(1),
        Severity::Info => Some(2),
        Severity::Off => None,
    }
}

pub(super) struct Analyzer<'a> {
    module: &'a Module,
    fun: &'a Function,
    info: &'a FunctionInfo,
    summaries: &'a [FunctionSummary],
    kind: FunctionKind,
    graph: Graph,
    /// The value node of each expression that has been evaluated.
    expr_nodes: Vec<Option<NodeId>>,
    /// `param_i`.
    params: Vec<NodeId>,
    /// `param_i_contents`, for `function` address space pointer parameters.
    param_contents: Vec<Option<NodeId>>,
    /// `Value_return_i_contents`, for `function` address space pointer parameters.
    return_contents: Vec<Option<NodeId>>,
    /// The [`State::slots`] index of each `function` address space pointer parameter.
    arg_slots: Vec<Option<usize>>,
    /// `Value_return`.
    value_return: Option<NodeId>,
    exits: Vec<Exits>,
    /// The potential-trigger-set of each `RequiredToBeUniform.S` node.
    triggers: [Triggers; 3],
}

impl<'a> Analyzer<'a> {
    pub fn new(
        module: &'a Module,
        fun: &'a Function,
        info: &'a FunctionInfo,
        summaries: &'a [FunctionSummary],
        kind: FunctionKind,
    ) -> Self {
        let mut graph = Graph::new();
        let mut params = Vec::new();
        let mut param_contents = Vec::new();
        let mut return_contents = Vec::new();
        let mut arg_slots = Vec::new();
        let mut value_return = None;
        if let FunctionKind::Regular = kind {
            let mut next_slot = fun.local_variables.len();
            for arg in fun.arguments.iter() {
                params.push(graph.add(NodeKind::Plain));
                let space = module.types[arg.ty].inner.pointer_space();
                if space == Some(crate::AddressSpace::Function) {
                    param_contents.push(Some(graph.add(NodeKind::Plain)));
                    return_contents.push(Some(graph.add(NodeKind::Plain)));
                    arg_slots.push(Some(next_slot));
                    next_slot += 1;
                } else {
                    param_contents.push(None);
                    return_contents.push(None);
                    arg_slots.push(None);
                }
            }
            if fun.result.is_some() {
                value_return = Some(graph.add(NodeKind::Plain));
            }
        }
        graph.begin_interior();
        Analyzer {
            module,
            fun,
            info,
            summaries,
            kind,
            graph,
            expr_nodes: vec![None; fun.expressions.len()],
            params,
            param_contents,
            return_contents,
            arg_slots,
            value_return,
            exits: Vec::new(),
            triggers: [Triggers::empty(); 3],
        }
    }

    pub fn run(mut self) -> Result<FunctionSummary, WithSpan<UniformityError>> {
        let mut slots = vec![CF_START; self.fun.local_variables.len()];
        for (i, slot) in self.arg_slots.iter().enumerate() {
            if let (&Some(_), Some(contents)) = (slot, self.param_contents[i]) {
                slots.push(contents);
            }
        }
        let mut state = State { slots };
        let fun = self.fun;
        let (_, behavior) = self.block(&fun.body, CF_START, &mut state);
        if behavior.contains(Behavior::NEXT) {
            self.record_return_contents(&state);
        }
        self.summarize()
    }

    fn summarize(&self) -> Result<FunctionSummary, WithSpan<UniformityError>> {
        let len = self.graph.len();
        let mut summary = FunctionSummary {
            params: vec![ParamSummary::default(); self.params.len()],
            ..FunctionSummary::default()
        };

        let mut visited = vec![false; len];
        for (index, severity) in SEVERITIES.into_iter().enumerate() {
            let (seen, violation) = self
                .graph
                .reach(REQUIRED_TO_BE_UNIFORM[index], &mut visited);
            if let Some(violation) = violation {
                severity.report_diag(self.error(violation), |e, level| {
                    log::log!(level, "{}", e.as_inner());
                })?;
                continue;
            }
            let requirement = Requirement {
                severity,
                triggers: self.triggers[index],
            };
            if seen[CF_START.index()] {
                summary.call_site.get_or_insert(requirement);
            }
            for (i, param) in summary.params.iter_mut().enumerate() {
                if seen[self.params[i].index()] {
                    param.required.get_or_insert(requirement);
                }
                if self.param_contents[i].is_some_and(|c| seen[c.index()]) {
                    param.contents_required.get_or_insert(requirement);
                }
            }
        }

        if let Some(value_return) = self.value_return {
            let (seen, _) = self.graph.reach(value_return, &mut vec![false; len]);
            summary.result_may_be_non_uniform = seen[MAY_BE_NON_UNIFORM.index()];
            for (i, param) in summary.params.iter_mut().enumerate() {
                param.result_depends_on_value = seen[self.params[i].index()];
                // The spec only sets `ParameterReturnContentsRequiredToBeUniform`
                // when `param_i` is reachable, but the tag's description says
                // the contents of a pointer parameter must also be uniform.
                param.result_depends_on_contents =
                    self.param_contents[i].is_some_and(|c| seen[c.index()]);
            }
        }

        for (i, return_contents) in self.return_contents.iter().enumerate() {
            let Some(return_contents) = *return_contents else {
                continue;
            };
            let (seen, _) = self.graph.reach(return_contents, &mut vec![false; len]);
            let mut out = PointerOut {
                may_be_non_uniform: seen[MAY_BE_NON_UNIFORM.index()],
                depends_on_call_site: seen[CF_START.index()],
                depends_on: Vec::new(),
            };
            for (j, &param) in self.params.iter().enumerate() {
                let j_u32 = j as u32;
                if seen[param.index()] {
                    out.depends_on.push(ParamDependency::Value(j_u32));
                }
                if self.param_contents[j].is_some_and(|c| seen[c.index()]) {
                    out.depends_on.push(ParamDependency::Contents(j_u32));
                }
            }
            summary.params[i].pointer_out = Some(out);
        }

        Ok(summary)
    }

    fn error(&self, violation: Violation<'_>) -> WithSpan<UniformityError> {
        let cause = violation
            .cause
            .map_or(NonUniformCause::Unknown, |(cause, _)| cause.clone());
        let (op, target, requirement_span) = match violation.requirement {
            Some((op, target, span)) => (op.clone(), target.clone(), span),
            None => (
                Operation::Unknown,
                RequirementTarget::ControlFlow,
                Span::UNDEFINED,
            ),
        };
        let (error, label) = match target {
            RequirementTarget::ControlFlow => (
                UniformityError::NonUniformControlFlow { op, cause },
                "requires uniform control flow",
            ),
            RequirementTarget::Argument(index) => (
                UniformityError::NonUniformArgument { op, index, cause },
                "requires a uniform argument",
            ),
            RequirementTarget::ArgumentContents(index) => (
                UniformityError::NonUniformArgumentContents { op, index, cause },
                "requires an argument that points to a uniform value",
            ),
        };
        let mut error = WithSpan::new(error).with_span(requirement_span, label);
        if let Some(span) = violation.condition {
            error = error.with_span(span, "control flow depends on this value");
        }
        if let Some((_, span)) = violation.cause {
            if Some(span) != violation.condition {
                error = error.with_span(span, "this value may be non-uniform");
            }
        }
        error
    }

    fn severity(&self, triggers: Triggers) -> Severity {
        let rule = if triggers.contains(Triggers::DERIVATIVE_UNIFORMITY) {
            StandardFilterableTriggeringRule::DerivativeUniformity
        } else if triggers.contains(Triggers::SUBGROUP_UNIFORMITY) {
            StandardFilterableTriggeringRule::SubgroupUniformity
        } else {
            return Severity::Error;
        };
        DiagnosticFilterNode::search(
            self.fun.diagnostic_filter_leaf,
            &self.module.diagnostic_filters,
            rule,
        )
    }

    fn require(
        &mut self,
        requirement: Requirement,
        op: &Operation,
        target: RequirementTarget,
        span: Span,
        node: NodeId,
    ) {
        let Some(index) = severity_index(requirement.severity) else {
            return;
        };
        let kind = NodeKind::Requirement {
            op: op.clone(),
            target,
            span,
        };
        let requirement_node = self.graph.node_to(kind, [node]);
        self.graph
            .edge(REQUIRED_TO_BE_UNIFORM[index], requirement_node);
        self.triggers[index] |= requirement.triggers;
    }

    fn source(&mut self, cause: NonUniformCause, span: Span) -> NodeId {
        self.graph
            .node_to(NodeKind::Source { cause, span }, [MAY_BE_NON_UNIFORM])
    }

    fn span(&self, expr: Handle<Expression>) -> Span {
        self.fun.expressions.get_span(expr)
    }

    /// The node an expression's value depends on, other than the control flow
    /// where it is used. `None` means the value is constant.
    fn base_node(&mut self, expr: Handle<Expression>) -> Option<NodeId> {
        if let Some(node) = self.expr_nodes[expr.index()] {
            return Some(node);
        }
        let Expression::FunctionArgument(index) = self.fun.expressions[expr] else {
            return None;
        };
        let node = match self.kind {
            FunctionKind::Regular => Some(self.params[index as usize]),
            FunctionKind::EntryPoint(stage) => self
                .entry_point_input_cause(index as usize, stage)
                .map(|cause| self.source(cause, self.span(expr))),
        };
        self.expr_nodes[expr.index()] = node;
        node
    }

    /// The value of `expr`, used where control flow is `cf`.
    fn value_at(&mut self, expr: Handle<Expression>, cf: NodeId) -> NodeId {
        match self.base_node(expr) {
            Some(node) => self.graph.join(&[node, cf]),
            None => cf,
        }
    }

    fn condition(&mut self, expr: Handle<Expression>, cf: NodeId) -> NodeId {
        let kind = NodeKind::Condition {
            span: self.span(expr),
        };
        let base = self.base_node(expr);
        self.graph.node_to(kind, [cf].into_iter().chain(base))
    }

    /// The spec's rules for formal parameters don't cover entry points, which
    /// have no callers. Its overview lists user-defined inputs and most
    /// built-in values as sources of non-uniformity, and it doesn't analyze
    /// structure members separately.
    fn entry_point_input_cause(
        &self,
        index: usize,
        stage: crate::ShaderStage,
    ) -> Option<NonUniformCause> {
        let builtin_cause = |builtin| {
            use crate::BuiltIn as Bi;
            let uniform = match builtin {
                Bi::WorkGroupId | Bi::NumWorkGroups | Bi::WorkGroupSize | Bi::NumSubgroups => true,
                Bi::SubgroupSize => matches!(
                    stage,
                    crate::ShaderStage::Compute
                        | crate::ShaderStage::Task
                        | crate::ShaderStage::Mesh
                ),
                _ => false,
            };
            (!uniform).then_some(NonUniformCause::BuiltIn(builtin))
        };
        let binding_cause = |binding: Option<&crate::Binding>| match binding {
            Some(&crate::Binding::BuiltIn(builtin)) => builtin_cause(builtin),
            _ => Some(NonUniformCause::UserDefinedInput),
        };
        let arg = &self.fun.arguments[index];
        match (arg.binding.as_ref(), &self.module.types[arg.ty].inner) {
            (None, &crate::TypeInner::Struct { ref members, .. }) => members
                .iter()
                .find_map(|member| binding_cause(member.binding.as_ref())),
            (binding, _) => binding_cause(binding),
        }
    }

    fn root(&self, mut pointer: Handle<Expression>) -> (Root, bool) {
        let mut full = true;
        loop {
            match self.fun.expressions[pointer] {
                Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                    full = false;
                    pointer = base;
                }
                Expression::LocalVariable(local) => return (Root::Slot(local.index()), full),
                Expression::FunctionArgument(index) => {
                    let root = match self.arg_slots.get(index as usize) {
                        Some(&Some(slot)) => Root::Slot(slot),
                        _ => Root::PrivateArgument,
                    };
                    return (root, full);
                }
                Expression::GlobalVariable(global) => return (Root::Global(global), full),
                _ => return (Root::Unknown, full),
            }
        }
    }

    /// The node for the value stored where `pointer` points. `None` means the
    /// stored value is uniform.
    fn contents(&mut self, pointer: Handle<Expression>, state: &State) -> Option<NodeId> {
        let span = self.span(pointer);
        let cause = match self.root(pointer).0 {
            Root::Slot(slot) => return Some(state.slots[slot]),
            Root::Global(global) => {
                use crate::AddressSpace as As;
                let read_only = match self.module.global_variables[global].space {
                    As::Uniform | As::Handle | As::Immediate => true,
                    As::Storage { access } => !access.contains(crate::StorageAccess::STORE),
                    As::Function
                    | As::Private
                    | As::WorkGroup
                    | As::TaskPayload
                    | As::RayPayload
                    | As::IncomingRayPayload => false,
                };
                if read_only {
                    return None;
                }
                NonUniformCause::MutableGlobal
            }
            Root::PrivateArgument => NonUniformCause::PrivatePointerArgument,
            Root::Unknown => NonUniformCause::MutableGlobal,
        };
        Some(self.source(cause, span))
    }

    fn is_read_write_storage_image(&self, image: Handle<Expression>) -> bool {
        matches!(
            *self.info[image].ty.inner_with(&self.module.types),
            crate::TypeInner::Image {
                class: crate::ImageClass::Storage { access, .. },
                ..
            } if access.contains(crate::StorageAccess::STORE)
        )
    }

    fn join_states(&mut self, states: &[State]) -> State {
        let slots = (0..states[0].slots.len())
            .map(|i| {
                let targets: Vec<NodeId> = states.iter().map(|s| s.slots[i]).collect();
                self.graph.join(&targets)
            })
            .collect();
        State { slots }
    }

    fn record_return_contents(&mut self, state: &State) {
        for i in 0..self.return_contents.len() {
            if let (Some(node), Some(slot)) = (self.return_contents[i], self.arg_slots[i]) {
                self.graph.edge(node, state.slots[slot]);
            }
        }
    }

    fn block(&mut self, block: &Block, mut cf: NodeId, state: &mut State) -> (NodeId, Behavior) {
        let mut behavior = Behavior::NEXT;
        for (statement, &span) in block.span_iter() {
            // Statically unreachable code does not contribute to the analysis.
            if !behavior.contains(Behavior::NEXT) {
                break;
            }
            let (next_cf, statement_behavior) = self.statement(statement, span, cf, state);
            cf = next_cf;
            behavior = (behavior - Behavior::NEXT) | statement_behavior;
        }
        (cf, behavior)
    }

    fn statement(
        &mut self,
        statement: &Statement,
        span: Span,
        cf: NodeId,
        state: &mut State,
    ) -> (NodeId, Behavior) {
        match *statement {
            Statement::Emit(ref range) => {
                for expr in range.clone() {
                    self.expression(expr, cf, state);
                }
            }
            Statement::Block(ref block) => return self.block(block, cf, state),
            Statement::If {
                condition,
                ref accept,
                ref reject,
            } => {
                let condition = self.condition(condition, cf);
                let mut accept_state = state.clone();
                let (accept_cf, accept_behavior) = self.block(accept, condition, &mut accept_state);
                let mut reject_state = state.clone();
                let (reject_cf, reject_behavior) = self.block(reject, condition, &mut reject_state);
                let mut outs = Vec::new();
                for (behavior, out) in [
                    (accept_behavior, accept_state),
                    (reject_behavior, reject_state),
                ] {
                    if behavior.contains(Behavior::NEXT) {
                        outs.push(out);
                    }
                }
                if !outs.is_empty() {
                    *state = self.join_states(&outs);
                }
                let behavior = accept_behavior | reject_behavior;
                if behavior == Behavior::NEXT {
                    return (cf, behavior);
                }
                let cf_end = self.graph.node_to(NodeKind::Plain, [accept_cf, reject_cf]);
                return (cf_end, behavior);
            }
            Statement::Switch {
                selector,
                ref cases,
            } => return self.switch(selector, cases, cf, state),
            Statement::Loop {
                ref body,
                ref continuing,
                break_if,
            } => return self.loop_(body, continuing, break_if, cf, state),
            Statement::Break => {
                if let Some(exits) = self.exits.last_mut() {
                    exits.breaks.push(state.clone());
                }
                return (cf, Behavior::BREAK);
            }
            Statement::Continue => {
                if let Some(exits) = self.exits.iter_mut().rev().find(|e| e.is_loop) {
                    exits.continues.push(state.clone());
                }
                return (cf, Behavior::CONTINUE);
            }
            Statement::Return { value } => {
                if let (Some(value), Some(value_return)) = (value, self.value_return) {
                    let value = self.value_at(value, cf);
                    self.graph.edge(value_return, value);
                }
                self.record_return_contents(state);
                return (cf, Behavior::RETURN);
            }
            Statement::Store { pointer, value } => {
                if let (Root::Slot(slot), full) = self.root(pointer) {
                    let mut targets = vec![self.value_at(value, cf)];
                    if !full {
                        targets.push(state.slots[slot]);
                        targets.extend(self.base_node(pointer));
                    }
                    state.slots[slot] = self.graph.join(&targets);
                }
            }
            Statement::Call {
                function,
                ref arguments,
                result,
            } => self.call(function, arguments, result, span, cf, state),
            // `discard` demotes the invocation to a helper invocation, which
            // keeps participating in derivatives.
            Statement::Kill => {}
            Statement::MemoryBarrier(_)
            | Statement::ImageStore { .. }
            | Statement::ImageAtomic { .. }
            | Statement::RayPipelineFunction(_)
            | Statement::HitObject { .. }
            | Statement::DebugPrintf { .. } => {}
            Statement::ControlBarrier(_)
            | Statement::Atomic { .. }
            | Statement::WorkGroupUniformLoad { .. }
            | Statement::RayQuery { .. }
            | Statement::SubgroupBallot { .. }
            | Statement::SubgroupGather { .. }
            | Statement::SubgroupCollectiveOperation { .. }
            | Statement::SubgroupBallotFindBit { .. }
            | Statement::CooperativeStore { .. } => {
                if let Some(intrinsic) = statement_intrinsic(statement) {
                    self.intrinsic(&intrinsic, span, cf);
                    if let Some(result) = intrinsic.result_expr {
                        let node = match intrinsic.result {
                            IntrinsicResult::Uniform => cf,
                            IntrinsicResult::Operands | IntrinsicResult::NonUniform => {
                                self.source(NonUniformCause::Result(intrinsic.op), span)
                            }
                        };
                        self.expr_nodes[result.index()] = Some(node);
                    }
                }
            }
        }
        (cf, Behavior::NEXT)
    }

    /// Add the requirements of `intrinsic` at control flow `cf`.
    fn intrinsic(&mut self, intrinsic: &Intrinsic, span: Span, cf: NodeId) {
        if let Some(triggers) = intrinsic.call_site {
            let requirement = Requirement {
                severity: self.severity(triggers),
                triggers,
            };
            let target = RequirementTarget::ControlFlow;
            self.require(requirement, &intrinsic.op, target, span, cf);
        }
        for &(index, operand, triggers) in intrinsic.uniform_operands.iter() {
            let requirement = Requirement {
                severity: self.severity(triggers),
                triggers,
            };
            let value = self.value_at(operand, cf);
            let target = RequirementTarget::Argument(index);
            self.require(requirement, &intrinsic.op, target, span, value);
        }
    }

    fn expression(&mut self, handle: Handle<Expression>, cf: NodeId, state: &State) {
        let fun = self.fun;
        let expr = &fun.expressions[handle];
        match *expr {
            // Constant, or a memory view that is uniform.
            Expression::Literal(_)
            | Expression::Constant(_)
            | Expression::Override(_)
            | Expression::ZeroValue(_)
            | Expression::GlobalVariable(_)
            | Expression::LocalVariable(_)
            // Their nodes come from elsewhere.
            | Expression::FunctionArgument(_)
            | Expression::CallResult(_)
            | Expression::AtomicResult { .. }
            | Expression::WorkGroupUniformLoadResult { .. }
            | Expression::RayQueryProceedResult
            | Expression::SubgroupBallotResult
            | Expression::SubgroupOperationResult { .. } => return,
            _ => {}
        }
        let span = self.span(handle);
        let intrinsic = expression_intrinsic(expr, |image| self.is_read_write_storage_image(image));
        let result = match intrinsic {
            Some(ref intrinsic) => {
                self.intrinsic(intrinsic, span, cf);
                intrinsic.result
            }
            None => IntrinsicResult::Operands,
        };
        let node = match result {
            IntrinsicResult::Uniform => cf,
            IntrinsicResult::NonUniform => {
                let op = intrinsic.map_or(Operation::Unknown, |i| i.op);
                self.source(NonUniformCause::Result(op), span)
            }
            IntrinsicResult::Operands => {
                let mut operands = Vec::new();
                push_operands(expr, &mut operands);
                let mut targets = vec![cf];
                for operand in operands {
                    targets.extend(self.base_node(operand));
                }
                // The spec makes a built-in function result depend on the
                // memory behind its pointer arguments. For `arrayLength` this
                // makes the length of a `read_write` buffer non-uniform, which
                // might not be intended.
                let pointer = match *expr {
                    Expression::Load { pointer } | Expression::ArrayLength(pointer) => {
                        Some(pointer)
                    }
                    Expression::CooperativeLoad { ref data, .. } => Some(data.pointer),
                    _ => None,
                };
                if let Some(pointer) = pointer {
                    targets.extend(self.contents(pointer, state));
                }
                self.graph.join(&targets)
            }
        };
        self.expr_nodes[handle.index()] = Some(node);
    }

    fn switch(
        &mut self,
        selector: Handle<Expression>,
        cases: &[crate::SwitchCase],
        cf: NodeId,
        state: &mut State,
    ) -> (NodeId, Behavior) {
        let selector = self.condition(selector, cf);
        self.exits.push(Exits {
            is_loop: false,
            breaks: Vec::new(),
            continues: Vec::new(),
        });
        let entry = state.clone();
        let mut behavior = Behavior::empty();
        let mut case_cfs = Vec::new();
        let mut outs = Vec::new();
        let mut fall_through: Option<(State, NodeId)> = None;
        for case in cases {
            let (mut case_state, case_cf) = match fall_through.take() {
                Some((previous, previous_cf)) => (
                    self.join_states(&[entry.clone(), previous]),
                    self.graph.join(&[selector, previous_cf]),
                ),
                None => (entry.clone(), selector),
            };
            let (case_cf, case_behavior) = self.block(&case.body, case_cf, &mut case_state);
            case_cfs.push(case_cf);
            if case.fall_through && case_behavior.contains(Behavior::NEXT) {
                behavior |= case_behavior - Behavior::NEXT;
                fall_through = Some((case_state, case_cf));
            } else {
                behavior |= case_behavior;
                if case_behavior.contains(Behavior::NEXT) {
                    outs.push(case_state);
                }
            }
        }
        if let Some((case_state, _)) = fall_through {
            behavior |= Behavior::NEXT;
            outs.push(case_state);
        }
        let exits = self.exits.pop().expect("pushed above");
        outs.extend(exits.breaks);
        if behavior.contains(Behavior::BREAK) {
            behavior = (behavior | Behavior::NEXT) - Behavior::BREAK;
        }
        if !outs.is_empty() {
            *state = self.join_states(&outs);
        }
        if behavior == Behavior::NEXT {
            return (cf, behavior);
        }
        (self.graph.join(&case_cfs), behavior)
    }

    fn loop_(
        &mut self,
        body: &Block,
        continuing: &Block,
        break_if: Option<Handle<Expression>>,
        cf: NodeId,
        state: &mut State,
    ) -> (NodeId, Behavior) {
        // `CF'`, and the value of each slot at the start of each iteration.
        let cf_iteration = self.graph.node_to(NodeKind::Plain, [cf]);
        let entry = state.clone();
        let heads: Vec<NodeId> = state
            .slots
            .iter()
            .map(|&slot| self.graph.node_to(NodeKind::Plain, [slot]))
            .collect();
        state.slots.clone_from(&heads);

        self.exits.push(Exits {
            is_loop: true,
            breaks: Vec::new(),
            continues: Vec::new(),
        });
        let (body_cf, body_behavior) = self.block(body, cf_iteration, state);
        let iterates = body_behavior.intersects(Behavior::NEXT | Behavior::CONTINUE);

        let mut continuing_behavior = Behavior::empty();
        let mut outs = Vec::new();
        if iterates {
            let exits = self.exits.last_mut().expect("pushed above");
            let mut ins = core::mem::take(&mut exits.continues);
            if body_behavior.contains(Behavior::NEXT) {
                ins.push(state.clone());
            }
            let mut continuing_state = self.join_states(&ins);
            let (mut continuing_cf, behavior) =
                self.block(continuing, body_cf, &mut continuing_state);
            continuing_behavior = behavior;
            if behavior.contains(Behavior::NEXT) {
                if let Some(condition) = break_if {
                    continuing_cf = self.condition(condition, continuing_cf);
                    continuing_behavior |= Behavior::BREAK;
                    outs.push(continuing_state.clone());
                }
                for (&head, &slot) in heads.iter().zip(continuing_state.slots.iter()) {
                    self.graph.edge(head, slot);
                }
                self.graph.edge(cf_iteration, continuing_cf);
            }
        }
        let exits = self.exits.pop().expect("pushed above");
        outs.extend(exits.breaks);
        *state = if outs.is_empty() {
            entry
        } else {
            self.join_states(&outs)
        };

        let both = body_behavior | continuing_behavior;
        let behavior = if body_behavior == Behavior::RETURN {
            Behavior::RETURN
        } else if !both.contains(Behavior::BREAK) {
            both - (Behavior::CONTINUE | Behavior::NEXT)
        } else {
            (both | Behavior::NEXT) - (Behavior::BREAK | Behavior::CONTINUE)
        };
        let cf_out = if !body_behavior.contains(Behavior::RETURN) {
            cf
        } else if iterates {
            cf_iteration
        } else {
            body_cf
        };
        (cf_out, behavior)
    }

    fn call(
        &mut self,
        function: Handle<Function>,
        arguments: &[Handle<Expression>],
        result: Option<Handle<Expression>>,
        span: Span,
        cf: NodeId,
        state: &mut State,
    ) {
        let name = self.module.functions[function]
            .name
            .clone()
            .unwrap_or_default();
        let op = Operation::Call { name: name.clone() };
        let conservative;
        let summaries = self.summaries;
        let summary = match summaries.get(function.index()) {
            Some(summary) => summary,
            None => {
                conservative = FunctionSummary::conservative(arguments.len());
                &conservative
            }
        };

        if let Some(requirement) = summary.call_site {
            self.require(requirement, &op, RequirementTarget::ControlFlow, span, cf);
        }
        let mut result_targets = vec![cf];
        if summary.result_may_be_non_uniform {
            result_targets.push(self.source(NonUniformCause::Result(op.clone()), span));
        }

        let values: Vec<NodeId> = arguments
            .iter()
            .map(|&argument| self.value_at(argument, cf))
            .collect();
        // For `function` address space pointer arguments: the slot, whether
        // the pointer covers the whole slot, and the contents before the call.
        let pointees: Vec<Option<(usize, bool, NodeId)>> = arguments
            .iter()
            .zip(summary.params.iter())
            .map(|(&argument, param)| {
                param.pointer_out.as_ref()?;
                match self.root(argument) {
                    (Root::Slot(slot), full) => Some((slot, full, state.slots[slot])),
                    _ => None,
                }
            })
            .collect();

        for (i, param) in summary.params.iter().enumerate().take(arguments.len()) {
            let index = i as u32;
            if let Some(requirement) = param.required {
                let target = RequirementTarget::Argument(index);
                self.require(requirement, &op, target, span, values[i]);
            }
            if param.result_depends_on_value {
                result_targets.push(values[i]);
            }
            if let Some((_, _, before)) = pointees[i] {
                if param.result_depends_on_contents {
                    result_targets.push(before);
                }
            }
        }

        let mut updates: Vec<(usize, NodeId)> = Vec::new();
        for (i, param) in summary.params.iter().enumerate().take(arguments.len()) {
            let (Some(out), Some((slot, full, before))) = (param.pointer_out.as_ref(), pointees[i])
            else {
                continue;
            };
            let mut targets = Vec::new();
            if out.may_be_non_uniform {
                let cause = NonUniformCause::PointerArgument {
                    function: name.clone(),
                    index: i as u32,
                };
                targets.push(self.source(cause, span));
            }
            if out.depends_on_call_site {
                targets.push(cf);
            }
            for &dependency in out.depends_on.iter() {
                let node = match dependency {
                    ParamDependency::Value(j) => values.get(j as usize).copied(),
                    ParamDependency::Contents(j) => pointees
                        .get(j as usize)
                        .copied()
                        .flatten()
                        .map(|(_, _, before)| before),
                };
                targets.extend(node);
            }
            // The spec has no rule for a pointer to part of a variable, because
            // WGSL requires a whole variable without the
            // `unrestricted_pointer_parameters` extension. Treat it like a
            // partial assignment.
            if !full {
                targets.push(before);
                targets.push(values[i]);
            }
            if targets.is_empty() {
                targets.push(cf);
            }
            let node = self.graph.join(&targets);
            updates.push((slot, node));
        }
        // Pointer arguments may alias, so join updates to the same slot.
        for (k, &(slot, _)) in updates.iter().enumerate() {
            if updates[..k].iter().any(|&(s, _)| s == slot) {
                continue;
            }
            let targets: Vec<NodeId> = updates
                .iter()
                .filter(|&&(s, _)| s == slot)
                .map(|&(_, node)| node)
                .collect();
            state.slots[slot] = self.graph.join(&targets);
        }

        for (i, param) in summary.params.iter().enumerate().take(arguments.len()) {
            if let (Some(requirement), Some((slot, _, _))) = (param.contents_required, pointees[i])
            {
                // The spec requires `Vout(call)` to be uniform. This might not
                // be right: the callee reads the contents from before the call.
                let target = RequirementTarget::ArgumentContents(i as u32);
                self.require(requirement, &op, target, span, state.slots[slot]);
            }
        }

        if let Some(result) = result {
            self.expr_nodes[result.index()] = Some(self.graph.join(&result_targets));
        }
    }
}

/// Push the operands of `expr` whose values affect its result.
fn push_operands(expr: &Expression, out: &mut Vec<Handle<Expression>>) {
    use Expression as E;
    match *expr {
        E::Literal(_)
        | E::Constant(_)
        | E::Override(_)
        | E::ZeroValue(_)
        | E::FunctionArgument(_)
        | E::GlobalVariable(_)
        | E::LocalVariable(_)
        | E::CallResult(_)
        | E::AtomicResult { .. }
        | E::WorkGroupUniformLoadResult { .. }
        | E::RayQueryProceedResult
        | E::SubgroupBallotResult
        | E::SubgroupOperationResult { .. } => {}
        E::Compose { ref components, .. } => out.extend(components.iter().copied()),
        E::Access { base, index } => out.extend([base, index]),
        E::AccessIndex { base, .. } => out.push(base),
        E::Splat { value, .. } => out.push(value),
        E::Swizzle { vector, .. } => out.push(vector),
        E::Load { pointer } => out.push(pointer),
        E::ImageSample {
            image,
            sampler,
            gather: _,
            coordinate,
            array_index,
            offset,
            ref level,
            depth_ref,
            clamp_to_edge: _,
        } => {
            out.extend([image, sampler, coordinate]);
            out.extend(array_index);
            out.extend(offset);
            out.extend(depth_ref);
            match *level {
                crate::SampleLevel::Auto | crate::SampleLevel::Zero => {}
                crate::SampleLevel::Exact(e) | crate::SampleLevel::Bias(e) => out.push(e),
                crate::SampleLevel::Gradient { x, y } => out.extend([x, y]),
            }
        }
        E::ImageLoad {
            image,
            coordinate,
            array_index,
            sample,
            level,
        } => {
            out.extend([image, coordinate]);
            out.extend(array_index);
            out.extend(sample);
            out.extend(level);
        }
        E::ImageQuery { image, ref query } => {
            out.push(image);
            if let crate::ImageQuery::Size { level: Some(level) } = *query {
                out.push(level);
            }
        }
        E::Unary { expr, .. } => out.push(expr),
        E::Binary { left, right, .. } => out.extend([left, right]),
        E::Select {
            condition,
            accept,
            reject,
        } => out.extend([condition, accept, reject]),
        E::Derivative { expr, .. } => out.push(expr),
        E::Relational { argument, .. } => out.push(argument),
        E::Math {
            arg,
            arg1,
            arg2,
            arg3,
            ..
        } => {
            out.push(arg);
            out.extend(arg1);
            out.extend(arg2);
            out.extend(arg3);
        }
        E::As { expr, .. } | E::Bitcast { expr, .. } => out.push(expr),
        E::ArrayLength(pointer) => out.push(pointer),
        E::RayQueryVertexPositions { query, .. } | E::RayQueryGetIntersection { query, .. } => {
            out.push(query)
        }
        E::HitObjectQuery { hit_object, .. } => out.push(hit_object),
        E::CooperativeLoad { ref data, .. } => out.extend([data.pointer, data.stride]),
        E::CooperativeMultiplyAdd { a, b, c } => out.extend([a, b, c]),
    }
}
