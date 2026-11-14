//! Decides whether a function's arithmetic control helpers need `[noinline]`.
//!
//! HLSL `precise` applies to every operation that contributes to the marked
//! value. A controlled operation is written as a `precise` result inside a
//! helper function. If DXC inlines that helper, the operations that compute
//! the helper's operands become `precise` as well. `[noinline]` prevents that,
//! but the helper then stays a real call with its operands passed through
//! memory, which some drivers execute far more slowly.
//!
//! The boundary is only needed when it protects something. An operation whose
//! result does not depend on whether it is `precise` (a sign change, a
//! bit cast, an integer operation, a rounding function) can be marked freely,
//! and so can an operation that carries a control of its own. This module
//! follows everything a `precise` result could reach in a function. If all of
//! it is of that kind, the helpers are written without `[noinline]`.
//!
//! # What DXC marks
//!
//! DXC inlines functions and promotes local variables to values before it
//! propagates `precise`, and it propagates over whole values before it splits
//! vectors into scalars. The walk in [`controls_are_isolated`] follows these
//! rules, which `naga/tests/naga/hlsl_inout.rs` exercises one by one:
//!
//! - A vector operation is one value. Reading one component of its result
//!   reads the whole operation and every component of each operand.
//! - A component read sees through constructors, swizzles and stores to
//!   single components: `vec4(a, b, c, d).y` reads `b` only. This holds only
//!   as long as nothing but constructors, swizzles and loads lies between the
//!   read and the component. A swizzle that is the operand of a vector
//!   operation reads its whole source vector.
//! - A load of a local scalar or vector that is only indexed with constants
//!   reads the stores that reach it, not later ones. A load of the whole
//!   vector that is used as a vector reads the last store to the whole
//!   variable and every store to a single component since then, including
//!   stores that a later one has overwritten.
//! - A load of any other variable reads every store in the function that may
//!   overlap it, before or after the load. If the variable is indexed
//!   dynamically anywhere, that is every store to the variable.
//! - A condition is read by everything it controls, and a called function's
//!   arguments by everything that reads them. Neither is modeled: only
//!   straight-line functions are accepted.
//!
//! The analysis answers "not isolated" for anything it does not model.

use alloc::vec::Vec;

use crate::{
    back::FunctionType, proc::any_statement, valid, AddressSpace, Arena, Block, Expression,
    FastHashMap, FastHashSet, Function, GlobalVariable, Handle, Literal, LocalVariable,
    MathFunction, Module, RelationalFunction, ScalarKind, Statement, TypeInner,
};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Root {
    Local(Handle<LocalVariable>),
    Global(Handle<GlobalVariable>),
}

/// The component of a vector that a pointer expression selects.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lane {
    /// The pointer refers to the whole value.
    All,
    One(u32),
    /// A dynamic index, or a path into a matrix.
    Unknown,
}

/// The part of a variable that a pointer expression refers to.
struct Place {
    root: Root,
    /// Array indices and struct member indices below the root. `None` is an
    /// index that is not constant. The path ends at the first vector or
    /// matrix; `lane` describes the rest.
    path: Vec<Option<u32>>,
    lane: Lane,
    /// The index expressions that are not constant.
    dynamic_indices: Vec<Handle<Expression>>,
}

impl Place {
    fn may_overlap(&self, other: &Self) -> bool {
        self.root == other.root
            && self
                .path
                .iter()
                .zip(&other.path)
                .all(|(left, right)| left.is_none() || right.is_none() || left == right)
    }
}

struct Store {
    place: Place,
    value: Handle<Expression>,
}

/// The value of an [`Expression::Access`] index that is a literal or a
/// constant initialized with one.
fn constant_index(
    module: &Module,
    expressions: &Arena<Expression>,
    index: Handle<Expression>,
) -> Option<u32> {
    let literal = match expressions[index] {
        Expression::Literal(literal) => literal,
        Expression::Constant(constant) => {
            match module.global_expressions[module.constants[constant].init] {
                Expression::Literal(literal) => literal,
                _ => return None,
            }
        }
        _ => return None,
    };
    match literal {
        Literal::U32(value) => Some(value),
        Literal::I32(value) => u32::try_from(value).ok(),
        _ => None,
    }
}

/// Resolves a pointer expression that starts at a local or global variable.
fn place(module: &Module, function: &Function, mut pointer: Handle<Expression>) -> Option<Place> {
    let expressions = &function.expressions;
    let mut path = Vec::new();
    let mut dynamic_indices = Vec::new();
    let (root, mut ty) = loop {
        match expressions[pointer] {
            Expression::AccessIndex { base, index } => {
                path.push(Some(index));
                pointer = base;
            }
            Expression::Access { base, index } => {
                let constant = constant_index(module, expressions, index);
                if constant.is_none() {
                    dynamic_indices.push(index);
                }
                path.push(constant);
                pointer = base;
            }
            Expression::LocalVariable(handle) => {
                break (Root::Local(handle), function.local_variables[handle].ty);
            }
            Expression::GlobalVariable(handle) => {
                break (Root::Global(handle), module.global_variables[handle].ty);
            }
            _ => return None,
        }
    };
    path.reverse();

    let mut depth = 0;
    let mut lane = Lane::All;
    for &index in &path {
        ty = match module.types[ty].inner {
            TypeInner::Array { base, .. } | TypeInner::BindingArray { base, .. } => base,
            TypeInner::Struct { ref members, .. } => match index {
                Some(index) => members.get(index as usize)?.ty,
                None => return None,
            },
            TypeInner::Vector { .. } => {
                lane = index.map_or(Lane::Unknown, Lane::One);
                break;
            }
            _ => {
                lane = Lane::Unknown;
                break;
            }
        };
        depth += 1;
    }
    path.truncate(depth);

    Some(Place {
        root,
        path,
        lane,
        dynamic_indices,
    })
}

/// Whether every store to `global` outside `function` writes a value that no
/// operation computed: a literal, a constant, a zero value, or an entry point
/// argument.
///
/// All functions of a module are written to one HLSL source, where a
/// `precise` result that reads a `static` variable also applies to whatever
/// any of them stores there.
fn stores_elsewhere_are_plain(
    module: &Module,
    function: &Function,
    global: Handle<GlobalVariable>,
) -> bool {
    fn block_is_plain(
        module: &Module,
        function: &Function,
        is_entry_point: bool,
        block: &Block,
        global: Handle<GlobalVariable>,
    ) -> bool {
        let refers_to_global = |pointer| {
            place(module, function, pointer).map(|place| place.root == Root::Global(global))
        };
        block.iter().all(|statement| match *statement {
            Statement::Store { pointer, value } => match refers_to_global(pointer) {
                Some(true) => match function.expressions[value] {
                    Expression::Literal(_) | Expression::Constant(_) | Expression::ZeroValue(_) => {
                        true
                    }
                    Expression::FunctionArgument(_) => is_entry_point,
                    _ => false,
                },
                Some(false) => true,
                // A store through a pointer argument could alias the global.
                None => false,
            },
            Statement::Atomic { pointer, .. } => refers_to_global(pointer) == Some(false),
            Statement::Block(ref block) => {
                block_is_plain(module, function, is_entry_point, block, global)
            }
            Statement::If {
                ref accept,
                ref reject,
                ..
            } => {
                block_is_plain(module, function, is_entry_point, accept, global)
                    && block_is_plain(module, function, is_entry_point, reject, global)
            }
            Statement::Switch { ref cases, .. } => cases
                .iter()
                .all(|case| block_is_plain(module, function, is_entry_point, &case.body, global)),
            Statement::Loop {
                ref body,
                ref continuing,
                ..
            } => {
                block_is_plain(module, function, is_entry_point, body, global)
                    && block_is_plain(module, function, is_entry_point, continuing, global)
            }
            _ => true,
        })
    }

    let functions = module
        .functions
        .iter()
        .map(|(_, function)| (function, false));
    let entry_points = module
        .entry_points
        .iter()
        .map(|entry_point| (&entry_point.function, true));
    functions
        .chain(entry_points)
        .filter(|&(other, _)| !core::ptr::eq(other, function))
        .all(|(other, is_entry_point)| {
            block_is_plain(module, other, is_entry_point, &other.body, global)
        })
}

/// Whether `block` consists of `Emit`, `Store`, `Block` and `Return`
/// statements and, if `calls` is set, of calls to functions with such bodies.
fn is_straight_line(module: &Module, block: &Block, calls: bool) -> bool {
    block.iter().all(|statement| match *statement {
        Statement::Emit(_) | Statement::Store { .. } | Statement::Return { .. } => true,
        Statement::Block(ref block) => is_straight_line(module, block, calls),
        Statement::Call { function, .. } => {
            calls && is_straight_line(module, &module.functions[function].body, false)
        }
        _ => false,
    })
}

/// Whether every call of `callee` is made by an entry point in which no
/// statement is conditional.
///
/// DXC inlines `callee` into its callers. A `precise` result then also
/// applies to the conditions that decide whether the call site is reached.
fn call_sites_are_unconditional(module: &Module, callee: Handle<Function>) -> bool {
    let calls = |function: &Function| {
        any_statement(
            &function.body,
            &mut |statement| matches!(*statement, Statement::Call { function, .. } if function == callee),
        )
    };
    module
        .functions
        .iter()
        .all(|(_, function)| !calls(function))
        && module.entry_points.iter().all(|entry_point| {
            !calls(&entry_point.function)
                || is_straight_line(module, &entry_point.function.body, true)
        })
}

/// What a `precise` result reads of a value: a set of vector components, or
/// [`WHOLE`].
///
/// A scalar is component 0. Components are read separately only through
/// expressions that DXC resolves before it propagates `precise`.
type Demand = u8;

/// The value is read as a whole: as the operand of an operation.
const WHOLE: Demand = 0x80;

/// The stores that reach a load of a promoted local variable.
#[derive(Clone, Copy)]
struct Reaching {
    local: Handle<LocalVariable>,
    /// The initializer or the last value stored to the whole variable.
    base: Option<Handle<Expression>>,
    /// The stores to single components since then, as a range of the
    /// variable's entry in [`Flow::lane_stores`].
    since: usize,
    until: usize,
}

/// The stores of a straight-line function body.
#[derive(Default)]
struct Flow {
    /// For local scalars and vectors that are only indexed with constants:
    /// every store to a single component, in program order.
    lane_stores: FastHashMap<Handle<LocalVariable>, Vec<(u32, Handle<Expression>)>>,
    /// For the same variables, the state at the current statement:
    /// the whole value and the start of its component stores.
    current: FastHashMap<Handle<LocalVariable>, (Option<Handle<Expression>>, usize)>,
    /// The state at each load of such a variable.
    reaching: FastHashMap<Handle<Expression>, Reaching>,
    /// The stores to all other variables.
    stores: Vec<Store>,
}

impl Flow {
    /// Records the loads and stores of `block` in program order.
    ///
    /// Returns `false` for a body that contains control flow, a call, or any
    /// other statement with an effect, and for a store that [`place`] does
    /// not resolve.
    fn collect(&mut self, module: &Module, function: &Function, block: &Block) -> bool {
        block.iter().all(|statement| match *statement {
            Statement::Emit(ref range) => {
                for handle in range.clone() {
                    let Expression::Load { pointer } = function.expressions[handle] else {
                        continue;
                    };
                    let Some(Root::Local(local)) =
                        place(module, function, pointer).map(|place| place.root)
                    else {
                        continue;
                    };
                    if let Some(&(base, since)) = self.current.get(&local) {
                        let until = self.lane_stores[&local].len();
                        self.reaching.insert(
                            handle,
                            Reaching {
                                local,
                                base,
                                since,
                                until,
                            },
                        );
                    }
                }
                true
            }
            Statement::Return { .. } => true,
            Statement::Store { pointer, value } => {
                let Some(place) = place(module, function, pointer) else {
                    return false;
                };
                let promoted = match place.root {
                    Root::Local(local) => self.current.get_mut(&local).map(|state| (local, state)),
                    Root::Global(_) => None,
                };
                match (promoted, place.lane) {
                    (Some((local, state)), Lane::All) => {
                        *state = (Some(value), self.lane_stores[&local].len());
                    }
                    (Some((local, _)), Lane::One(lane)) => {
                        self.lane_stores
                            .get_mut(&local)
                            .unwrap()
                            .push((lane, value));
                    }
                    (Some(_), Lane::Unknown) => return false,
                    (None, _) => self.stores.push(Store { place, value }),
                }
                true
            }
            Statement::Block(ref block) => self.collect(module, function, block),
            _ => false,
        })
    }
}

/// Whether the arithmetic control helpers of `function` can be written
/// without `[noinline]`.
///
/// This is the case when an HLSL `precise` result on each controlled
/// operation, applied to everything that contributes to it, only reaches
/// operations that are controlled themselves or whose result does not depend
/// on `precise`.
///
/// The walk starts at every controlled expression and follows operands the
/// way the [module documentation](self) describes.
///
/// The answer is `false` when the walk reaches
///
/// - floating-point arithmetic or a floating-point comparison that carries no
///   control;
/// - a math function that reads or produces a floating-point value and
///   carries no control, other than `abs`, `floor`, `ceil`, `trunc` and
///   `round`;
/// - a conversion from a floating-point value to an integer, which is
///   written with a clamp, or a bit cast that changes the component layout;
/// - `isnan` or `isinf`;
/// - an argument of a function that is not an entry point, a call result, a
///   texture sample, a derivative, or any other expression not listed in the
///   body of this function;
/// - a private or workgroup variable that another function stores a computed
///   value to.
///
/// It is also `false` when the function body contains anything but `Emit`,
/// `Store`, `Block` and `Return` statements, and when a function that is not
/// an entry point has a call site that
/// [`call_sites_are_unconditional`] does not accept.
pub(super) fn controls_are_isolated(
    module: &Module,
    ty: FunctionType,
    function: &Function,
    info: &valid::FunctionInfo,
) -> bool {
    let is_entry_point = match ty {
        FunctionType::EntryPoint(_) => true,
        FunctionType::Function(handle) => {
            if !call_sites_are_unconditional(module, handle) {
                return false;
            }
            false
        }
    };
    let expressions = &function.expressions;
    let inner = |handle: Handle<Expression>| info[handle].ty.inner_with(&module.types);
    let is_float =
        |handle: Handle<Expression>| inner(handle).scalar_kind() == Some(ScalarKind::Float);
    // Every component of a scalar or vector; `WHOLE` for anything else.
    let all_lanes = |handle: Handle<Expression>| match *inner(handle) {
        TypeInner::Scalar(_) => 1,
        TypeInner::Vector { size, .. } => (1 << size as u8) - 1,
        _ => WHOLE,
    };

    // Variables that are indexed dynamically anywhere in the function.
    let mut dynamic_roots = FastHashSet::default();
    for (handle, expression) in expressions.iter() {
        if let Expression::Access { index, .. } = *expression {
            if constant_index(module, expressions, index).is_none() {
                match place(module, function, handle) {
                    Some(place) => dynamic_roots.insert(place.root),
                    // Not a pointer to a variable: a dynamically indexed value.
                    None => false,
                };
            }
        }
    }

    // DXC promotes these to values, so that a load reads its reaching stores.
    let mut flow = Flow::default();
    for (local, variable) in function.local_variables.iter() {
        let promoted = matches!(
            module.types[variable.ty].inner,
            TypeInner::Scalar(_) | TypeInner::Vector { .. }
        ) && !dynamic_roots.contains(&Root::Local(local));
        if promoted {
            flow.lane_stores.insert(local, Vec::new());
            flow.current.insert(local, (variable.init, 0));
        }
    }
    if !flow.collect(module, function, &function.body) {
        return false;
    }

    let mut pending: Vec<(Handle<Expression>, Demand)> = function
        .expression_controls
        .keys()
        .map(|&handle| (handle, all_lanes(handle)))
        .collect();
    let mut visited: FastHashMap<Handle<Expression>, Demand> = FastHashMap::default();
    while let Some((handle, demand)) = pending.pop() {
        // Reduce the demand to what the type has.
        let demand = match all_lanes(handle) {
            WHOLE => WHOLE,
            1 => 1,
            _ if demand & WHOLE != 0 => WHOLE,
            lanes => demand & lanes,
        };
        let seen = visited.entry(handle).or_insert(0);
        let demand = if *seen & WHOLE != 0 {
            0
        } else if demand == WHOLE {
            WHOLE
        } else {
            demand & !*seen
        };
        if demand == 0 {
            continue;
        }
        *seen |= demand;

        // An operand of an operation: the operation reads it as one value.
        let operand = |handle: Handle<Expression>| (handle, WHOLE);
        let controlled = function.expression_controls.contains_key(&handle);
        if controlled {
            // A controlled operation is written one component at a time, with
            // each helper operand read from a component of the operand value.
            let operands = match expressions[handle] {
                Expression::Binary { left, right, .. } => [Some(left), Some(right), None, None],
                Expression::Math {
                    arg,
                    arg1,
                    arg2,
                    arg3,
                    ..
                } => [Some(arg), arg1, arg2, arg3],
                _ => return false,
            };
            let operands = operands.into_iter().flatten();
            pending.extend(operands.map(|operand| (operand, all_lanes(operand))));
            continue;
        }
        match expressions[handle] {
            Expression::Literal(_)
            | Expression::Constant(_)
            | Expression::Override(_)
            | Expression::ZeroValue(_)
            | Expression::GlobalVariable(_)
            | Expression::LocalVariable(_) => {}
            // In a caller, an argument is an arbitrary expression.
            Expression::FunctionArgument(_) => {
                if !is_entry_point {
                    return false;
                }
            }
            Expression::Compose { ty, ref components } => {
                let is_vector = matches!(module.types[ty].inner, TypeInner::Vector { .. });
                if is_vector && demand != WHOLE {
                    let mut first = 0;
                    for &component in components {
                        let width = match *inner(component) {
                            TypeInner::Vector { size, .. } => size as u8,
                            _ => 1,
                        };
                        let part = (demand >> first) & ((1 << width) - 1);
                        if part != 0 {
                            pending.push((component, part));
                        }
                        first += width;
                    }
                } else {
                    pending.extend(components.iter().map(|&component| operand(component)));
                }
            }
            Expression::Splat { value, .. } => pending.push(operand(value)),
            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => {
                if demand == WHOLE {
                    pending.push(operand(vector));
                } else {
                    let lanes = pattern[..size as usize]
                        .iter()
                        .enumerate()
                        .filter(|&(lane, _)| demand & (1 << lane) != 0)
                        .fold(0, |lanes, (_, &component)| lanes | (1 << component as u8));
                    pending.push((vector, lanes));
                }
            }
            Expression::AccessIndex { base, index } => match *inner(base) {
                TypeInner::Vector { .. } => pending.push((base, 1 << index)),
                _ => pending.push(operand(base)),
            },
            Expression::Access { base, index } => {
                match (constant_index(module, expressions, index), inner(base)) {
                    (Some(lane), &TypeInner::Vector { .. }) if lane < 4 => {
                        pending.push((base, 1 << lane));
                    }
                    _ => pending.extend([operand(base), operand(index)]),
                }
            }
            Expression::Load { pointer } => {
                let Some(place) = place(module, function, pointer) else {
                    return false;
                };
                if let Some(reaching) = flow.reaching.get(&handle) {
                    let lane_stores =
                        &flow.lane_stores[&reaching.local][reaching.since..reaching.until];
                    let mut read_lane = |lane: u32| match lane_stores
                        .iter()
                        .rev()
                        .find(|&&(stored, _)| stored == lane)
                    {
                        Some(&(_, value)) => pending.push(operand(value)),
                        None => pending.extend(reaching.base.map(|base| (base, 1 << lane))),
                    };
                    match place.lane {
                        Lane::One(lane) => read_lane(lane),
                        Lane::All if demand != WHOLE => {
                            (0..4)
                                .filter(|lane| demand & (1 << lane) != 0)
                                .for_each(read_lane);
                        }
                        // The loaded value is the chain of every component
                        // store since the last store to the whole variable.
                        Lane::All => {
                            pending.extend(lane_stores.iter().map(|&(_, value)| operand(value)));
                            pending.extend(reaching.base.map(operand));
                        }
                        Lane::Unknown => return false,
                    }
                    continue;
                }
                match place.root {
                    Root::Local(local) => {
                        // A promoted variable whose load no `Emit` covers.
                        if flow.current.contains_key(&local) {
                            return false;
                        }
                        pending.extend(function.local_variables[local].init.map(operand));
                    }
                    Root::Global(global) => {
                        let shared = matches!(
                            module.global_variables[global].space,
                            AddressSpace::Private | AddressSpace::WorkGroup
                        );
                        if shared && !stores_elsewhere_are_plain(module, function, global) {
                            return false;
                        }
                    }
                }
                let dynamic = dynamic_roots.contains(&place.root);
                for store in flow.stores.iter().filter(|store| {
                    store.place.root == place.root && (dynamic || store.place.may_overlap(&place))
                }) {
                    pending.push(operand(store.value));
                    pending.extend(
                        store
                            .place
                            .dynamic_indices
                            .iter()
                            .map(|&index| operand(index)),
                    );
                }
                pending.extend(place.dynamic_indices.into_iter().map(operand));
            }
            Expression::Unary { expr, .. } => pending.push(operand(expr)),
            Expression::As { expr, target } => {
                // A conversion to an integer is written as a function that
                // clamps the value first, one to `bool` as a comparison.
                if is_float(expr) && target.kind != ScalarKind::Float {
                    return false;
                }
                pending.push(operand(expr));
            }
            Expression::Bitcast { expr, .. } => {
                // Only the casts that are written as `asfloat`, `asint` or
                // `asuint` of a value with the same components.
                let same_layout = match (inner(expr), inner(handle)) {
                    (&TypeInner::Scalar(from), &TypeInner::Scalar(to)) => from.width == to.width,
                    (
                        &TypeInner::Vector {
                            size: from_size,
                            scalar: from,
                        },
                        &TypeInner::Vector {
                            size: to_size,
                            scalar: to,
                        },
                    ) => from_size == to_size && from.width == to.width,
                    _ => false,
                };
                if !same_layout || inner(handle).scalar_width() != Some(4) {
                    return false;
                }
                pending.push(operand(expr));
            }
            Expression::Binary { left, right, .. } => {
                if is_float(left) || is_float(right) {
                    return false;
                }
                pending.extend([operand(left), operand(right)]);
            }
            Expression::Select {
                condition,
                accept,
                reject,
            } => pending.extend([operand(condition), operand(accept), operand(reject)]),
            Expression::Relational { fun, argument } => match fun {
                RelationalFunction::All | RelationalFunction::Any => {
                    pending.push(operand(argument))
                }
                // Without `precise`, the compiler may assume that there are no
                // NaNs or infinities.
                RelationalFunction::IsNan | RelationalFunction::IsInf => return false,
            },
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                use MathFunction as Mf;
                let exact = matches!(fun, Mf::Abs | Mf::Floor | Mf::Ceil | Mf::Trunc | Mf::Round);
                let integer = !is_float(arg) && !is_float(handle);
                if !exact && !integer {
                    return false;
                }
                let arguments = [Some(arg), arg1, arg2, arg3];
                pending.extend(arguments.into_iter().flatten().map(operand));
            }
            Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => {
                let operands = [Some(image), Some(coordinate), array_index, sample, level];
                pending.extend(operands.into_iter().flatten().map(operand));
            }
            Expression::ImageQuery { image, query } => {
                pending.push(operand(image));
                if let crate::ImageQuery::Size { level: Some(level) } = query {
                    pending.push(operand(level));
                }
            }
            Expression::ArrayLength(pointer) => pending.push(operand(pointer)),
            _ => return false,
        }
    }
    true
}
