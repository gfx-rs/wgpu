//! Flow-sensitive provenance for memory access restrictions.

use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
    vec,
    vec::Vec,
};

use core::cell::Cell;

use super::{FunctionError, FunctionInfo};
use crate::{Expression as E, Handle, Statement as S, StorageAccess as A, TypeInner as T};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Place {
    root: (Vec<usize>, usize),
    path: Vec<Option<u32>>,
    widened: bool,
}

impl Place {
    fn overlaps(&self, other: &Self) -> bool {
        self.root == other.root
            && self
                .path
                .iter()
                .zip(&other.path)
                .all(|(a, b)| a.is_none() || b.is_none() || a == b)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Value {
    immutable: bool,
    denied: A,
    places: Vec<Place>,
    fields: BTreeMap<u32, Value>,
    unknown_element: Option<Box<Value>>,
}

impl Value {
    fn join(&mut self, other: &Self) {
        self.immutable |= other.immutable;
        self.denied |= other.denied;
        for place in &other.places {
            if !self.places.contains(place) {
                self.places.push(place.clone());
            }
        }
        self.places.sort();
        if let Some(ref other) = other.unknown_element {
            self.unknown_element
                .get_or_insert_with(Default::default)
                .join(other);
        }
        for (&index, value) in &other.fields {
            self.fields.entry(index).or_default().join(value);
        }
    }

    fn component(&self, index: Option<u32>) -> Self {
        let mut value = Self {
            immutable: self.immutable,
            denied: self.denied,
            places: self.places.clone(),
            fields: BTreeMap::new(),
            unknown_element: None,
        };
        if let Some(ref summary) = self.unknown_element {
            value.join(summary);
        }
        if let Some(index) = index {
            if let Some(field) = self.fields.get(&index) {
                value.join(field);
            }
        } else {
            for field in self.fields.values() {
                value.join(field);
            }
        }
        value
    }

    fn read(&self, path: &[Option<u32>]) -> Self {
        match path.split_first() {
            Some((&index, rest)) => self.component(index).read(rest),
            None => self.clone(),
        }
    }

    fn store(&mut self, path: &[Option<u32>], value: &Self, strong: bool) {
        match path.split_first() {
            Some((&Some(index), rest)) => {
                self.fields
                    .entry(index)
                    .or_default()
                    .store(rest, value, strong);
            }
            Some((&None, rest)) => {
                // Keep unknown elements separate from the fields of their stored values.
                self.unknown_element
                    .get_or_insert_with(Default::default)
                    .store(rest, value, false);
            }
            None if strong => *self = value.clone(),
            None => self.join(value),
        }
    }

    fn flatten(&self) -> Self {
        let mut result = self.clone();
        result.fields.clear();
        result.unknown_element = None;
        if let Some(ref summary) = self.unknown_element {
            result.join(&summary.flatten());
        }
        for field in self.fields.values() {
            result.join(&field.flatten());
        }
        result
    }
}

type Memory = BTreeMap<(Vec<usize>, usize), Value>;

#[derive(Clone, Default, PartialEq, Eq)]
struct State {
    memory: Memory,
    values: Vec<Value>,
    immutable: BTreeSet<Place>,
    written: BTreeSet<Place>,
}

impl State {
    fn join(&mut self, other: &Self) {
        for (target, source) in [
            (&mut self.immutable, &other.immutable),
            (&mut self.written, &other.written),
        ] {
            target.extend(source.iter().cloned());
        }
        for (place, value) in &other.memory {
            self.memory.entry(place.clone()).or_default().join(value);
        }
        for (value, other) in self.values.iter_mut().zip(&other.values) {
            value.join(other);
        }
    }
}

fn merge(target: &mut Option<State>, source: Option<State>) {
    if let Some(source) = source {
        if let Some(ref mut target) = *target {
            target.join(&source);
        } else {
            *target = Some(source);
        }
    }
}

#[derive(Default)]
struct Flow {
    next: Option<State>,
    breaks: Option<State>,
    continues: Option<State>,
    returns: Option<State>,
    result: Value,
}

impl Flow {
    fn exits(&mut self, other: Self) {
        merge(&mut self.breaks, other.breaks);
        merge(&mut self.continues, other.continues);
        merge(&mut self.returns, other.returns);
        self.result.join(&other.result);
    }
}

struct Frame<'a> {
    module: &'a crate::Module,
    infos: &'a [FunctionInfo],
    fun: &'a crate::Function,
    info: &'a FunctionInfo,
    id: Vec<usize>,
    budget: &'a Cell<usize>,
    restrictions: &'a [A],
    pointer_types: &'a [bool],
}

impl Frame<'_> {
    fn charge(&self) -> Result<(), FunctionError> {
        let remaining = self
            .budget
            .get()
            .checked_sub(1)
            .ok_or(FunctionError::PointerAccessAnalysisLimit)?;
        self.budget.set(remaining);
        Ok(())
    }

    fn ty(&self, handle: Handle<E>) -> &T {
        self.info[handle].ty.inner_with(&self.module.types)
    }

    fn unknown_value(&self, ty: &T, root: (Vec<usize>, usize)) -> Result<Value, FunctionError> {
        self.charge()?;
        if root.0.len() > 128 {
            return Err(FunctionError::PointerAccessAnalysisLimit);
        }
        Ok(match *ty {
            T::Pointer { .. } | T::ValuePointer { .. } => Value {
                places: vec![Place {
                    root,
                    path: Vec::new(),
                    widened: false,
                }],
                ..Value::default()
            },
            T::Struct { ref members, .. } => {
                let mut fields = BTreeMap::new();
                for (index, member) in members.iter().enumerate() {
                    if self.pointer_types[member.ty.index()] {
                        let mut root = root.clone();
                        root.0.push(index + 1);
                        fields.insert(
                            index as u32,
                            self.unknown_value(&self.module.types[member.ty].inner, root)?,
                        );
                    }
                }
                Value {
                    fields,
                    ..Value::default()
                }
            }
            T::Array { base, .. } if self.pointer_types[base.index()] => {
                let mut root = root;
                root.0.push(0);
                Value {
                    unknown_element: Some(Box::new(
                        self.unknown_value(&self.module.types[base].inner, root)?,
                    )),
                    ..Value::default()
                }
            }
            _ => Value::default(),
        })
    }

    fn pointee_restrictions(&self, pointer: Handle<E>) -> A {
        match *self.ty(pointer) {
            T::Pointer { base, .. } => self.restrictions[base.index()],
            _ => A::empty(),
        }
    }

    fn check(&self, state: &State, pointer: Handle<E>, access: A) -> Result<(), FunctionError> {
        let value = &state.values[pointer.index()];
        if access.contains(A::STORE)
            && (value.immutable
                || value.places.iter().any(|place| {
                    state
                        .immutable
                        .iter()
                        .any(|immutable| immutable.overlaps(place))
                }))
        {
            return Err(FunctionError::InvalidStorePointer(pointer));
        }
        if (value.denied | self.pointee_restrictions(pointer)).intersects(access) {
            return Err(FunctionError::InvalidPointerAccess { pointer, access });
        }
        Ok(())
    }

    fn project(&self, state: &State, base: Handle<E>, index: Option<u32>) -> Value {
        let mut value = state.values[base.index()].clone();
        if let T::Pointer { base: ty, .. } = *self.ty(base) {
            if let T::Struct { ref members, .. } = self.module.types[ty].inner {
                if let Some(member) = index.and_then(|index| members.get(index as usize)) {
                    value.denied |= member
                        .access
                        .map_or(A::empty(), |access| (A::LOAD | A::STORE) & !access);
                }
            }
            for place in &mut value.places {
                if !place.widened {
                    // Widen cyclic cast/access paths so loop analysis reaches a fixed point.
                    if place.path.len() >= self.module.types.len() {
                        place.path.clear();
                        place.path.push(None);
                        place.widened = true;
                    } else {
                        place.path.push(index);
                    }
                }
            }
            value
        } else if self.ty(base).pointer_space().is_some() {
            value
        } else {
            value.component(index)
        }
    }

    fn eval(&self, state: &State, handle: Handle<E>) -> Result<Value, FunctionError> {
        self.charge()?;
        let get = |h: Handle<E>| state.values[h.index()].clone();
        let mut value = match self.fun.expressions[handle] {
            E::LocalVariable(local) => Value {
                places: vec![Place {
                    root: (self.id.clone(), local.index()),
                    path: Vec::new(),
                    widened: false,
                }],
                ..Value::default()
            },
            E::GlobalVariable(global) => Value {
                places: vec![Place {
                    root: (Vec::new(), global.index()),
                    path: Vec::new(),
                    widened: false,
                }],
                ..Value::default()
            },
            E::FunctionArgument(_) | E::CallResult(_) => get(handle),
            E::Compose { ref components, .. } => Value {
                fields: components
                    .iter()
                    .enumerate()
                    .map(|(i, &h)| (i as u32, get(h)))
                    .collect(),
                ..Value::default()
            },
            E::AccessIndex { base, index } => self.project(state, base, Some(index)),
            E::Access { base, index } => {
                let index = match self.fun.expressions[index] {
                    E::Literal(crate::Literal::U32(v)) => Some(v),
                    E::Literal(crate::Literal::I32(v)) => u32::try_from(v).ok(),
                    _ => None,
                };
                self.project(state, base, index)
            }
            E::Load { pointer } => {
                self.check(state, pointer, A::LOAD)?;
                let mut result = Value::default();
                for place in &get(pointer).places {
                    if let Some(value) = state.memory.get(&place.root) {
                        result.join(&value.read(&place.path));
                    }
                }
                // The holder's restrictions do not qualify the loaded pointer's target.
                if result == Value::default() {
                    let mut root = self.id.clone();
                    root.push(0);
                    result = self.unknown_value(self.ty(handle), (root, handle.index()))?;
                }
                result
            }
            E::PointerAlignment { pointer, .. }
            | E::AtomicPointer { pointer, .. }
            | E::CoherentPointer { pointer, .. } => get(pointer),
            E::PointerOffset { pointer, .. } => {
                let mut value = get(pointer);
                for place in &mut value.places {
                    place.path.clear();
                    place.path.push(None);
                    place.widened = true;
                }
                value
            }
            E::PointerCast { expr, .. } => {
                let mut value = get(expr);
                value.denied |= self.pointee_restrictions(expr);
                value
            }
            E::Select { accept, reject, .. } => {
                let mut value = get(accept);
                value.join(&get(reject));
                value
            }
            E::Unary { expr, .. } | E::As { expr, .. } => get(expr).flatten(),
            E::Binary { left, right, .. } => {
                let mut value = get(left).flatten();
                value.join(&get(right).flatten());
                for place in &mut value.places {
                    place.path = vec![None];
                    place.widened = true;
                }
                value
            }
            E::Math {
                arg,
                arg1,
                arg2,
                arg3,
                ..
            } => {
                let mut value = get(arg).flatten();
                for h in [arg1, arg2, arg3].into_iter().flatten() {
                    value.join(&get(h).flatten());
                }
                value
            }
            E::Splat { size, value } => Value {
                fields: (0..size as u32).map(|i| (i, get(value))).collect(),
                ..Value::default()
            },
            E::Swizzle {
                size,
                vector,
                pattern,
            } => Value {
                fields: (0..size as u32)
                    .map(|i| (i, get(vector).component(Some(pattern[i as usize] as u32))))
                    .collect(),
                ..Value::default()
            },
            E::MatrixLoad { ref data, .. } | E::CooperativeLoad { ref data, .. } => {
                self.check(state, data.pointer, A::LOAD)?;
                Value::default()
            }
            E::Literal(_)
            | E::Constant(_)
            | E::Override(_)
            | E::ZeroValue(_)
            | E::ImageSample { .. }
            | E::ImageLoad { .. }
            | E::ImageQuery { .. }
            | E::Derivative { .. }
            | E::Relational { .. }
            | E::AtomicResult { .. }
            | E::WorkGroupUniformLoadResult { .. }
            | E::ArrayLength(_)
            | E::RayQueryVertexPositions { .. }
            | E::RayQueryProceedResult
            | E::RayQueryGetIntersection { .. }
            | E::SubgroupBallotResult
            | E::SubgroupOperationResult { .. }
            | E::CooperativeMultiplyAdd { .. } => Value::default(),
        };
        if self.ty(handle).pointer_space().is_some() || matches!(self.ty(handle), T::Scalar(_)) {
            value = value.flatten();
        }
        if self.ty(handle).pointer_space() == Some(crate::AddressSpace::PhysicalStorage)
            && value.places.is_empty()
        {
            let mut root = self.id.clone();
            root.push(0);
            value.places.push(Place {
                root: (root, handle.index()),
                path: Vec::new(),
                widened: false,
            });
        }
        Ok(value)
    }

    fn run(&self, arguments: &[Value], mut state: State) -> Result<Flow, FunctionError> {
        self.charge()?;
        if self.id.len() > 128 {
            return Err(FunctionError::PointerAccessAnalysisLimit);
        }
        state.values = vec![Value::default(); self.fun.expressions.len()];
        for (h, expr) in self.fun.expressions.iter() {
            if let E::FunctionArgument(index) = *expr {
                let mut value = arguments[index as usize].clone();
                value.immutable |= self.fun.arguments[index as usize].immutable_pointee;
                if value.immutable {
                    if value
                        .places
                        .iter()
                        .any(|place| state.written.iter().any(|written| written.overlaps(place)))
                    {
                        return Err(FunctionError::InvalidStorePointer(h));
                    }
                    state.immutable.extend(value.places.iter().cloned());
                }
                state.values[h.index()] = value;
            } else if expr.needs_pre_emit() {
                state.values[h.index()] = self.eval(&state, h)?;
            }
        }
        for (h, _) in self.fun.local_variables.iter() {
            state.memory.remove(&(self.id.clone(), h.index()));
        }
        self.block(&self.fun.body, state)
    }

    fn block(&self, block: &crate::Block, state: State) -> Result<Flow, FunctionError> {
        let mut flow = Flow {
            next: Some(state),
            ..Flow::default()
        };
        for statement in block.iter() {
            self.charge()?;
            let Some(mut state) = flow.next.take() else {
                break;
            };
            match *statement {
                S::Emit(ref range) => {
                    for h in range.clone() {
                        let value = self.eval(&state, h)?;
                        if let E::Load { pointer } = self.fun.expressions[h] {
                            // Repeated loads from a known holder may return the same pointer.
                            for place in state.values[pointer.index()].places.clone() {
                                state.memory.entry(place.root).or_default().store(
                                    &place.path,
                                    &value,
                                    false,
                                );
                            }
                        }
                        state.values[h.index()] = value;
                    }
                }
                S::Store { pointer, value } => {
                    self.check(&state, pointer, A::STORE)?;
                    let places = state.values[pointer.index()].places.clone();
                    state.written.extend(places.iter().cloned());
                    for place in &places {
                        state.memory.entry(place.root.clone()).or_default().store(
                            &place.path,
                            &state.values[value.index()],
                            places.len() == 1,
                        );
                    }
                }
                S::Atomic {
                    pointer,
                    ref fun,
                    value,
                    result,
                } => {
                    self.check(&state, pointer, A::LOAD | A::STORE)?;
                    let places = state.values[pointer.index()].places.clone();
                    let mut previous = Value::default();
                    for place in &places {
                        if let Some(stored) = state.memory.get(&place.root) {
                            previous.join(&stored.read(&place.path));
                        }
                    }
                    let previous = previous.flatten();
                    let mut next = state.values[value.index()].flatten();
                    if !matches!(*fun, crate::AtomicFunction::Exchange { compare: None }) {
                        next.join(&previous);
                    }
                    if !matches!(*fun, crate::AtomicFunction::Exchange { .. }) {
                        for place in &mut next.places {
                            place.path = vec![None];
                            place.widened = true;
                        }
                    }
                    for place in &places {
                        state.memory.entry(place.root.clone()).or_default().store(
                            &place.path,
                            &next,
                            places.len() == 1,
                        );
                    }
                    if let Some(result) = result {
                        state.values[result.index()] =
                            if matches!(*fun, crate::AtomicFunction::Exchange { compare: Some(_) })
                            {
                                Value {
                                    fields: [(0, previous)].into_iter().collect(),
                                    ..Value::default()
                                }
                            } else {
                                previous
                            };
                    }
                    state.written.extend(places);
                }
                S::MatrixStore { ref data, .. } | S::CooperativeStore { ref data, .. } => {
                    self.check(&state, data.pointer, A::STORE)?;
                    state
                        .written
                        .extend(state.values[data.pointer.index()].places.iter().cloned());
                }
                S::Call {
                    function,
                    ref arguments,
                    result,
                } => {
                    let mut id = self.id.clone();
                    id.push(core::ptr::from_ref(statement) as usize);
                    let frame = Frame {
                        fun: &self.module.functions[function],
                        info: &self.infos[function.index()],
                        id,
                        ..*self
                    };
                    let args: Vec<_> = arguments
                        .iter()
                        .map(|h| state.values[h.index()].clone())
                        .collect();
                    let mut called = frame.run(&args, state.clone()).map_err(|source| {
                        FunctionError::InvalidCall {
                            function,
                            error: super::CallError::PointerAccess {
                                source: Box::new(source),
                            },
                        }
                    })?;
                    merge(&mut called.returns, called.next.take());
                    if let Some(returned) = called.returns {
                        state.memory = returned.memory;
                        state.immutable = returned.immutable;
                        state.written = returned.written;
                        if let Some(result) = result {
                            state.values[result.index()] = called.result;
                        }
                    } else {
                        continue;
                    }
                }
                S::Block(ref block) => {
                    let mut nested = self.block(block, state)?;
                    flow.next = nested.next.take();
                    flow.exits(nested);
                    continue;
                }
                S::If {
                    ref accept,
                    ref reject,
                    ..
                } => {
                    for branch in [accept, reject] {
                        let mut nested = self.block(branch, state.clone())?;
                        merge(&mut flow.next, nested.next.take());
                        flow.exits(nested);
                    }
                    continue;
                }
                S::Switch { ref cases, .. } => {
                    let mut fallthrough = None;
                    for case in cases {
                        let mut input = Some(state.clone());
                        merge(&mut input, fallthrough.take());
                        let mut nested = self.block(&case.body, input.unwrap())?;
                        merge(&mut flow.next, nested.breaks.take());
                        if case.fall_through {
                            fallthrough = nested.next.take();
                        } else {
                            merge(&mut flow.next, nested.next.take());
                        }
                        flow.exits(nested);
                    }
                    continue;
                }
                S::Loop {
                    ref body,
                    ref continuing,
                    break_if,
                } => {
                    let mut header = state;
                    loop {
                        let mut body_flow = self.block(body, header.clone())?;
                        merge(&mut flow.next, body_flow.breaks.take());
                        let mut back = body_flow.next.take();
                        merge(&mut back, body_flow.continues.take());
                        flow.exits(body_flow);
                        let Some(back) = back else { break };
                        let mut tail = self.block(continuing, back)?;
                        merge(&mut flow.next, tail.breaks.take());
                        let back = tail.next.take();
                        if break_if.is_some() {
                            merge(&mut flow.next, back.clone());
                        }
                        flow.exits(tail);
                        let Some(back) = back else { break };
                        let old = header.clone();
                        header.join(&back);
                        if old == header {
                            break;
                        }
                    }
                    continue;
                }
                S::Return { value } => {
                    if let Some(value) = value {
                        flow.result.join(&state.values[value.index()]);
                    }
                    merge(&mut flow.returns, Some(state));
                    continue;
                }
                S::Break => {
                    merge(&mut flow.breaks, Some(state));
                    continue;
                }
                S::Continue => {
                    merge(&mut flow.continues, Some(state));
                    continue;
                }
                S::Kill => continue,
                S::WorkGroupUniformLoad { pointer, result } => {
                    self.check(&state, pointer, A::LOAD)?;
                    let mut value = Value::default();
                    for place in &state.values[pointer.index()].places {
                        if let Some(stored) = state.memory.get(&place.root) {
                            value.join(&stored.read(&place.path));
                        }
                    }
                    state.values[result.index()] = value;
                }
                S::SubgroupCollectiveOperation {
                    argument, result, ..
                }
                | S::SubgroupGather {
                    argument, result, ..
                } => {
                    state.values[result.index()] = state.values[argument.index()].clone();
                }
                S::ControlBarrier(_)
                | S::MemoryBarrier(_)
                | S::ImageStore { .. }
                | S::ImageAtomic { .. }
                | S::RayQuery { .. }
                | S::SubgroupBallot { .. }
                | S::RayPipelineFunction(..)
                | S::DebugPrintf { .. } => {}
            }
            flow.next = Some(state);
        }
        Ok(flow)
    }
}

pub(super) fn validate(
    fun: &crate::Function,
    module: &crate::Module,
    info: &FunctionInfo,
    infos: &[FunctionInfo],
) -> Result<(), FunctionError> {
    let has_restrictions = module.types.iter().any(|(_, ty)| {
        matches!(&ty.inner,
        T::Struct { members, .. } if members.iter().any(|member| member.access.is_some()))
    });
    let has_immutable_pointee =
        |fun: &crate::Function| fun.arguments.iter().any(|arg| arg.immutable_pointee);
    let has_immutable = has_immutable_pointee(fun)
        || module
            .functions
            .iter()
            .any(|(_, fun)| has_immutable_pointee(fun));
    if !has_restrictions && !has_immutable {
        return Ok(());
    }
    // Type handles are topologically ordered after structural validation.
    let mut restrictions = Vec::with_capacity(module.types.len());
    let mut pointer_types = Vec::with_capacity(module.types.len());
    for (_, ty) in module.types.iter() {
        let (access, pointer) = match ty.inner {
            T::Struct { ref members, .. } => {
                members
                    .iter()
                    .fold((A::empty(), false), |(access, pointer), member| {
                        (
                            access
                                | restrictions[member.ty.index()]
                                | member
                                    .access
                                    .map_or(A::empty(), |a| (A::LOAD | A::STORE) & !a),
                            pointer || pointer_types[member.ty.index()],
                        )
                    })
            }
            T::Array { base, .. } => (restrictions[base.index()], pointer_types[base.index()]),
            T::Pointer { .. } | T::ValuePointer { .. } => (A::empty(), true),
            _ => (A::empty(), false),
        };
        restrictions.push(access);
        pointer_types.push(pointer);
    }
    let budget = Cell::new(100_000);
    let frame = Frame {
        budget: &budget,
        restrictions: &restrictions,
        pointer_types: &pointer_types,
        module,
        infos,
        fun,
        info,
        id: vec![1],
    };
    let arguments = fun
        .arguments
        .iter()
        .enumerate()
        .map(|(index, argument)| {
            frame.unknown_value(&module.types[argument.ty].inner, (vec![2], index))
        })
        .collect::<Result<Vec<_>, _>>()?;
    frame.run(&arguments, State::default())?;
    Ok(())
}
