//! Rewriting the expressions that use decomposed globals.

use alloc::{format, vec, vec::Vec};
use core::mem;

use super::{is_affected_matrix, Layouts, MemberMap, UniformMatrixLayoutError};
use crate::back::passes::emit::Emitting;
use crate::{
    arena::{Arena, Range},
    compact::expressions::for_each_operand,
    non_max_u32::NonMaxU32,
    ArraySize, BinaryOperator, Block, Expression, FastHashMap, FastHashSet, Function,
    FunctionArgument, FunctionResult, GlobalVariable, Handle, Literal, LocalVariable, Module,
    Scalar, Span, Statement, Type, TypeInner, UniqueArena, VectorSize,
};

use UniformMatrixLayoutError::UnsupportedUse;

/// Converters for arrays with more elements than this use a loop.
const UNROLL_LIMIT: u32 = 16;

const U32: Type = Type {
    name: None,
    inner: TypeInner::Scalar(Scalar::U32),
};

type Places = FastHashMap<Handle<Expression>, Place>;

/// Rewrite the uses of the `globals`, whose original types are given, in
/// every function of `module`.
pub(super) fn rewrite(
    module: &mut Module,
    layouts: &Layouts,
    globals: &FastHashMap<Handle<GlobalVariable>, Handle<Type>>,
) -> Result<(), UniformMatrixLayoutError> {
    let mut rewriter = Rewriter {
        types: &module.types,
        layouts,
        globals,
        converters: FastHashMap::default(),
        new_functions: Vec::new(),
        function_count: module.functions.len(),
        add_u32: false,
    };
    for (_, function) in module.functions.iter_mut() {
        rewriter.function(function)?;
    }
    for entry_point in module.entry_points.iter_mut() {
        rewriter.function(&mut entry_point.function)?;
    }

    let Rewriter {
        new_functions: converters,
        add_u32,
        ..
    } = rewriter;
    if add_u32 {
        module.types.insert(U32, Span::UNDEFINED);
    }
    // The converters' handles follow the existing functions, as the calls to
    // them assume.
    for (function, span) in converters {
        module.functions.append(function, span);
    }
    Ok(())
}

/// What a pointer expression rooted at a decomposed global refers to.
#[derive(Clone, Copy)]
enum Place {
    /// A pointer to the decomposed counterpart of this type.
    Pointer(Handle<Type>),
    /// A matrix whose columns are the members of the struct `base` points to,
    /// starting at `first`.
    SplitMatrix {
        base: Handle<Expression>,
        first: u32,
        columns: VectorSize,
        matrix: Handle<Type>,
    },
    /// An access into a dynamically indexed column of a split matrix. It is
    /// rewritten as an access into the loaded matrix.
    Value,
}

/// A value that has yet to be given a handle.
enum Pending {
    Expression(Expression),
    /// The result of calling a conversion function with this argument.
    Call(Handle<Function>, Handle<Expression>),
}

struct Rewriter<'a> {
    types: &'a UniqueArena<Type>,
    layouts: &'a Layouts,
    /// The original type of each decomposed global.
    globals: &'a FastHashMap<Handle<GlobalVariable>, Handle<Type>>,
    /// The conversion function for each decomposed array type.
    converters: FastHashMap<Handle<Type>, Handle<Function>>,
    /// The conversion functions and their spans, numbered as if appended
    /// after the module's `function_count` functions.
    new_functions: Vec<(Function, Span)>,
    function_count: usize,
    /// Whether [`U32`] must be added to the module's types.
    add_u32: bool,
}

impl Rewriter<'_> {
    /// Rewrite `function`'s uses of decomposed globals. Return whether it
    /// changed.
    fn function(&mut self, function: &mut Function) -> Result<bool, UniformMatrixLayoutError> {
        let (places, loads) = self.places(&mut function.expressions)?;
        if places.is_empty() {
            return Ok(false);
        }
        function
            .named_expressions
            .retain(|handle, _| matches!(places.get(handle), None | Some(&Place::Pointer(_))));
        let mut body = mem::take(&mut function.body);
        let result = self.block(&mut body, &mut function.expressions, &places, &loads);
        function.body = body;
        result.map(|()| true)
    }

    /// Find the place of each pointer expression rooted at a decomposed
    /// global, remapping struct member indices in place, and the loads from
    /// these places.
    fn places(
        &self,
        expressions: &mut Arena<Expression>,
    ) -> Result<(Places, FastHashSet<Handle<Expression>>), UniformMatrixLayoutError> {
        let mut places = FastHashMap::default();
        let mut loads = FastHashSet::default();
        for (handle, expr) in expressions.iter_mut() {
            let place = match *expr {
                Expression::GlobalVariable(global) => self
                    .globals
                    .get(&global)
                    .and_then(|&ty| self.pointer_place(handle, ty)),
                Expression::AccessIndex {
                    base,
                    ref mut index,
                } => match places.get(&base).copied() {
                    None => None,
                    Some(Place::Pointer(ty)) => match self.types[ty].inner {
                        TypeInner::Struct { ref members, .. } => {
                            let member =
                                members.get(*index as usize).ok_or(UnsupportedUse(handle))?;
                            match self.layouts.map[&ty].members[*index as usize] {
                                MemberMap::Same(new) => {
                                    *index = new;
                                    self.pointer_place(handle, member.ty)
                                }
                                MemberMap::Split { first, columns } => Some(Place::SplitMatrix {
                                    base,
                                    first,
                                    columns,
                                    matrix: member.ty,
                                }),
                            }
                        }
                        TypeInner::Array { base, .. } | TypeInner::BindingArray { base, .. } => {
                            self.pointer_place(handle, base)
                        }
                        _ => return Err(UnsupportedUse(handle)),
                    },
                    Some(Place::SplitMatrix { base, first, .. }) => {
                        let index = first + *index;
                        *expr = Expression::AccessIndex { base, index };
                        None
                    }
                    Some(Place::Value) => Some(Place::Value),
                },
                Expression::Access { base, .. } => match places.get(&base).copied() {
                    None => None,
                    Some(Place::Pointer(ty)) => match self.types[ty].inner {
                        TypeInner::Array { base, .. } | TypeInner::BindingArray { base, .. } => {
                            self.pointer_place(handle, base)
                        }
                        _ => return Err(UnsupportedUse(handle)),
                    },
                    Some(Place::SplitMatrix { .. } | Place::Value) => Some(Place::Value),
                },
                Expression::Load { pointer } => {
                    if places.contains_key(&pointer) {
                        loads.insert(handle);
                    }
                    None
                }
                ref other => {
                    let mut uses_place = false;
                    for_each_operand(other, |operand| {
                        uses_place |= places.contains_key(&operand);
                    });
                    if uses_place {
                        return Err(UnsupportedUse(handle));
                    }
                    None
                }
            };
            if let Some(place) = place {
                places.insert(handle, place);
            }
        }
        Ok((places, loads))
    }

    /// The place of `pointer`, a pointer to a value of type `ty`, if `ty` is
    /// affected.
    fn pointer_place(&self, pointer: Handle<Expression>, ty: Handle<Type>) -> Option<Place> {
        match self.types[ty].inner {
            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } if is_affected_matrix(rows, scalar) => Some(Place::SplitMatrix {
                base: pointer,
                first: 0,
                columns,
                matrix: ty,
            }),
            _ => self
                .layouts
                .map
                .contains_key(&ty)
                .then_some(Place::Pointer(ty)),
        }
    }

    fn block(
        &mut self,
        block: &mut Block,
        expressions: &mut Arena<Expression>,
        places: &Places,
        loads: &FastHashSet<Handle<Expression>>,
    ) -> Result<(), UniformMatrixLayoutError> {
        for (mut statement, span) in mem::take(block).span_into_iter() {
            let mut place_operand = None;
            for_each_direct_operand(&statement, |operand| {
                if places.contains_key(&operand) {
                    place_operand.get_or_insert(operand);
                }
            });
            if let Some(operand) = place_operand {
                return Err(UnsupportedUse(operand));
            }
            match statement {
                Statement::Emit(ref range) => {
                    if range.clone().any(|handle| loads.contains(&handle)) {
                        let range = range.clone();
                        self.split_emit(block, expressions, places, loads, range, span)?;
                        continue;
                    }
                }
                Statement::Block(ref mut inner) => {
                    self.block(inner, expressions, places, loads)?;
                }
                Statement::If {
                    ref mut accept,
                    ref mut reject,
                    ..
                } => {
                    self.block(accept, expressions, places, loads)?;
                    self.block(reject, expressions, places, loads)?;
                }
                Statement::Switch { ref mut cases, .. } => {
                    for case in cases {
                        self.block(&mut case.body, expressions, places, loads)?;
                    }
                }
                Statement::Loop {
                    ref mut body,
                    ref mut continuing,
                    ..
                } => {
                    self.block(body, expressions, places, loads)?;
                    self.block(continuing, expressions, places, loads)?;
                }
                _ => {}
            }
            block.push(statement, span);
        }
        Ok(())
    }

    /// Push `range` to `block`, rewriting the loads in it. Each load's new
    /// expressions get an `Emit` of their own, just before the load.
    fn split_emit(
        &mut self,
        block: &mut Block,
        expressions: &mut Arena<Expression>,
        places: &Places,
        loads: &FastHashSet<Handle<Expression>>,
        range: Range<Expression>,
        span: Span,
    ) -> Result<(), UniformMatrixLayoutError> {
        let mut start = range.index_range().start;
        for load in range.clone().filter(|handle| loads.contains(handle)) {
            let at = load.index() as u32;
            push_emit(block, expressions, start..at, span);
            let mut out = Emitting::new(expressions, block, expressions.get_span(load));
            let value = self.load(&mut out, places, load)?;
            start = if out.replace(load, value) { at } else { at + 1 };
        }
        push_emit(block, expressions, start..range.index_range().end, span);
        Ok(())
    }

    fn load(
        &mut self,
        out: &mut Emitting,
        places: &Places,
        load: Handle<Expression>,
    ) -> Result<Pending, UniformMatrixLayoutError> {
        let Expression::Load { pointer } = out.expressions[load] else {
            return Err(UnsupportedUse(load));
        };
        match places[&pointer] {
            Place::Pointer(ty) => {
                let raw = out.expr(Expression::Load { pointer });
                self.convert(out, raw, ty, load)
            }
            Place::SplitMatrix {
                base,
                first,
                columns,
                matrix,
            } => {
                let columns = out.load_columns(base, first, columns);
                Ok(Pending::Expression(Expression::Compose {
                    ty: matrix,
                    components: columns,
                }))
            }
            Place::Value => {
                let mut chain = Vec::new();
                let mut cursor = pointer;
                while let Some(&Place::Value) = places.get(&cursor) {
                    chain.push(cursor);
                    cursor = access_base(&out.expressions[cursor]);
                }
                let Some(&Place::SplitMatrix {
                    base,
                    first,
                    columns,
                    matrix,
                }) = places.get(&cursor)
                else {
                    return Err(UnsupportedUse(cursor));
                };
                let columns = out.load_columns(base, first, columns);
                let mut value = out.expr(Expression::Compose {
                    ty: matrix,
                    components: columns,
                });
                let (&last, rest) = chain.split_first().unwrap();
                for &access in rest.iter().rev() {
                    value = out.expr(with_base(&out.expressions[access], value));
                }
                Ok(Pending::Expression(with_base(
                    &out.expressions[last],
                    value,
                )))
            }
        }
    }

    /// Convert `value`, of the decomposed counterpart of `ty`, to `ty`.
    ///
    /// `site` is the load being rewritten, for errors.
    fn convert(
        &mut self,
        out: &mut Emitting,
        value: Handle<Expression>,
        ty: Handle<Type>,
        site: Handle<Expression>,
    ) -> Result<Pending, UniformMatrixLayoutError> {
        let (types, layouts) = (self.types, self.layouts);
        let components = match types[ty].inner {
            TypeInner::Matrix { columns, .. } => out.columns(value, 0, columns),
            TypeInner::Struct { ref members, .. } => {
                let mut components = Vec::with_capacity(members.len());
                for (member, &map) in members.iter().zip(&layouts.map[&ty].members) {
                    components.push(match map {
                        MemberMap::Same(index) => {
                            let member_value =
                                out.expr(Expression::AccessIndex { base: value, index });
                            if layouts.map.contains_key(&member.ty) {
                                let pending = self.convert(out, member_value, member.ty, site)?;
                                out.value(pending)
                            } else {
                                member_value
                            }
                        }
                        MemberMap::Split { first, columns } => {
                            let columns = out.columns(value, first, columns);
                            out.expr(Expression::Compose {
                                ty: member.ty,
                                components: columns,
                            })
                        }
                    });
                }
                components
            }
            TypeInner::Array { .. } => {
                let converter = self.converter(ty, site, out.span)?;
                return Ok(Pending::Call(converter, value));
            }
            _ => return Err(UnsupportedUse(site)),
        };
        Ok(Pending::Expression(Expression::Compose { ty, components }))
    }

    /// The function that converts the decomposed counterpart of the array
    /// type `ty` to `ty`.
    ///
    /// `site` is the load being rewritten, for errors, and `span` its span.
    fn converter(
        &mut self,
        ty: Handle<Type>,
        site: Handle<Expression>,
        span: Span,
    ) -> Result<Handle<Function>, UniformMatrixLayoutError> {
        if let Some(&function) = self.converters.get(&ty) {
            return Ok(function);
        }
        let TypeInner::Array {
            base,
            size: ArraySize::Constant(size),
            ..
        } = self.types[ty].inner
        else {
            return Err(UnsupportedUse(site));
        };

        let decomposed = self.layouts.map[&ty].ty;
        let mut function = Function {
            arguments: vec![FunctionArgument {
                name: Some("v".into()),
                ty: decomposed,
                binding: None,
            }],
            result: Some(FunctionResult { ty, binding: None }),
            ..Function::default()
        };
        let argument = function
            .expressions
            .append(Expression::FunctionArgument(0), span);
        if size.get() <= UNROLL_LIMIT {
            let mut out = Emitting::new(&mut function.expressions, &mut function.body, span);
            let mut components = Vec::with_capacity(size.get() as usize);
            for index in 0..size.get() {
                let element = out.expr(Expression::AccessIndex {
                    base: argument,
                    index,
                });
                let pending = self.convert(&mut out, element, base, site)?;
                components.push(out.value(pending));
            }
            let value = out.expr(Expression::Compose { ty, components });
            out.push(Statement::Return { value: Some(value) });
        } else {
            self.converter_loop(&mut function, argument, ty, base, size.get(), site, span)?;
        }

        let index = self.new_functions.len();
        function.name = Some(format!("convert_std140_{index}"));
        let handle = nth_handle(self.function_count + index);
        self.new_functions.push((function, span));
        self.converters.insert(ty, handle);
        Ok(handle)
    }

    /// The `u32` type. If the module has none, its handle once [`U32`] is
    /// added after rewriting.
    fn u32_type(&mut self) -> Handle<Type> {
        self.types.get(&U32).unwrap_or_else(|| {
            self.add_u32 = true;
            nth_handle(self.types.len())
        })
    }

    /// Fill in the body of `function`, a converter for the array type `ty` of
    /// `size` elements of type `base`, as a loop over the elements.
    #[expect(clippy::too_many_arguments)]
    fn converter_loop(
        &mut self,
        function: &mut Function,
        argument: Handle<Expression>,
        ty: Handle<Type>,
        base: Handle<Type>,
        size: u32,
        site: Handle<Expression>,
        span: Span,
    ) -> Result<(), UniformMatrixLayoutError> {
        let u32_ty = self.u32_type();
        let local = |function: &mut Function, name: &str, ty, init| {
            let variable = function.local_variables.append(
                LocalVariable {
                    name: Some(name.into()),
                    ty,
                    init,
                },
                span,
            );
            function
                .expressions
                .append(Expression::LocalVariable(variable), span)
        };
        let literal = |function: &mut Function, value| {
            function
                .expressions
                .append(Expression::Literal(Literal::U32(value)), span)
        };
        let source = local(function, "s", self.layouts.map[&ty].ty, None);
        let result = local(function, "r", ty, None);
        let zero = literal(function, 0);
        let counter = local(function, "i", u32_ty, Some(zero));
        let len = literal(function, size);
        let one = literal(function, 1);

        function.body.push(
            Statement::Store {
                pointer: source,
                value: argument,
            },
            span,
        );

        let mut body = Block::new();
        let mut out = Emitting::new(&mut function.expressions, &mut body, span);
        let index = out.expr(Expression::Load { pointer: counter });
        let condition = out.expr(Expression::Binary {
            op: BinaryOperator::Less,
            left: index,
            right: len,
        });
        out.push(Statement::If {
            condition,
            accept: Block::new(),
            reject: Block::from_vec(vec![Statement::Break]),
        });
        let pointer = out.expr(Expression::Access {
            base: source,
            index,
        });
        let element = out.expr(Expression::Load { pointer });
        let pending = self.convert(&mut out, element, base, site)?;
        let value = out.value(pending);
        let pointer = out.expr(Expression::Access {
            base: result,
            index,
        });
        out.push(Statement::Store { pointer, value });

        let mut continuing = Block::new();
        let mut out = Emitting::new(&mut function.expressions, &mut continuing, span);
        let index = out.expr(Expression::Load { pointer: counter });
        let value = out.expr(Expression::Binary {
            op: BinaryOperator::Add,
            left: index,
            right: one,
        });
        out.push(Statement::Store {
            pointer: counter,
            value,
        });

        function.body.push(
            Statement::Loop {
                body,
                continuing,
                break_if: None,
            },
            span,
        );
        let mut out = Emitting::new(&mut function.expressions, &mut function.body, span);
        let value = out.expr(Expression::Load { pointer: result });
        out.push(Statement::Return { value: Some(value) });
        Ok(())
    }
}

/// Pass-specific helpers.
impl Emitting<'_> {
    fn call(
        &mut self,
        function: Handle<Function>,
        argument: Handle<Expression>,
        result: Handle<Expression>,
    ) {
        self.block.push(
            Statement::Call {
                function,
                arguments: vec![argument],
                result: Some(result),
            },
            self.span,
        );
    }

    fn value(&mut self, pending: Pending) -> Handle<Expression> {
        match pending {
            Pending::Expression(expr) => self.expr(expr),
            Pending::Call(function, argument) => {
                self.flush();
                let result = self.expr(Expression::CallResult(function));
                self.call(function, argument, result);
                self.start = self.expressions.len();
                result
            }
        }
    }

    /// Make the existing expression `handle` evaluate to `pending`. Return
    /// whether `handle` must still be emitted.
    fn replace(&mut self, handle: Handle<Expression>, pending: Pending) -> bool {
        self.flush();
        match pending {
            Pending::Expression(expr) => {
                self.expressions[handle] = expr;
                true
            }
            Pending::Call(function, argument) => {
                self.expressions[handle] = Expression::CallResult(function);
                self.call(function, argument, handle);
                false
            }
        }
    }

    /// The members of `base`, a struct value, that hold a split matrix's
    /// columns.
    fn columns(
        &mut self,
        base: Handle<Expression>,
        first: u32,
        columns: VectorSize,
    ) -> Vec<Handle<Expression>> {
        (first..first + columns as u32)
            .map(|index| self.expr(Expression::AccessIndex { base, index }))
            .collect()
    }

    /// Load the columns of a split matrix in the struct `base` points to.
    fn load_columns(
        &mut self,
        base: Handle<Expression>,
        first: u32,
        columns: VectorSize,
    ) -> Vec<Handle<Expression>> {
        (first..first + columns as u32)
            .map(|index| {
                let pointer = self.expr(Expression::AccessIndex { base, index });
                self.expr(Expression::Load { pointer })
            })
            .collect()
    }
}

fn push_emit(
    block: &mut Block,
    expressions: &Arena<Expression>,
    range: core::ops::Range<u32>,
    span: Span,
) {
    if !range.is_empty() {
        block.push(
            Statement::Emit(Range::from_index_range(range, expressions)),
            span,
        );
    }
}

/// Call `f` on the expressions `statement` uses directly. These include every
/// pointer a statement can take, but not the operands nested in its function
/// payload, such as a ray query's.
fn for_each_direct_operand(statement: &Statement, mut f: impl FnMut(Handle<Expression>)) {
    match *statement {
        Statement::If { condition, .. } => f(condition),
        Statement::Switch { selector, .. } => f(selector),
        Statement::Loop { break_if, .. } => break_if.into_iter().for_each(f),
        Statement::Return { value } => value.into_iter().for_each(f),
        Statement::Store { pointer, value } => {
            f(pointer);
            f(value);
        }
        Statement::ImageStore {
            image,
            coordinate,
            array_index,
            value,
        }
        | Statement::ImageAtomic {
            image,
            coordinate,
            array_index,
            value,
            ..
        } => {
            f(image);
            f(coordinate);
            array_index.into_iter().for_each(&mut f);
            f(value);
        }
        Statement::Atomic { pointer, value, .. } => {
            f(pointer);
            f(value);
        }
        Statement::WorkGroupUniformLoad { pointer, .. } => f(pointer),
        Statement::Call { ref arguments, .. } | Statement::DebugPrintf { ref arguments, .. } => {
            arguments.iter().copied().for_each(f)
        }
        Statement::RayQuery { query, .. } => f(query),
        Statement::HitObject { hit_object, .. } => f(hit_object),
        Statement::SubgroupBallot { predicate, .. } => predicate.into_iter().for_each(f),
        Statement::SubgroupGather { argument, .. }
        | Statement::SubgroupCollectiveOperation { argument, .. }
        | Statement::SubgroupBallotFindBit { argument, .. } => f(argument),
        Statement::CooperativeStore { target, .. } => f(target),
        Statement::Emit(_)
        | Statement::Block(_)
        | Statement::Break
        | Statement::Continue
        | Statement::Kill
        | Statement::ControlBarrier(_)
        | Statement::MemoryBarrier(_)
        | Statement::RayPipelineFunction(_) => {}
    }
}

fn access_base(expr: &Expression) -> Handle<Expression> {
    match *expr {
        Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => base,
        _ => unreachable!("only accesses have the `Value` place"),
    }
}

/// `expr`, an access, with its base replaced by `base`.
fn with_base(expr: &Expression, base: Handle<Expression>) -> Expression {
    match *expr {
        Expression::Access { index, .. } => Expression::Access { base, index },
        Expression::AccessIndex { index, .. } => Expression::AccessIndex { base, index },
        _ => unreachable!("only accesses have the `Value` place"),
    }
}

const fn nth_handle<T>(index: usize) -> Handle<T> {
    Handle::new(NonMaxU32::new(index as u32).unwrap())
}

#[cfg(all(test, feature = "wgsl-in"))]
mod tests {
    use alloc::borrow::Cow;

    use super::super::{run, PassError};
    use super::*;
    use crate::{
        valid::{Capabilities, ModuleInfo, ValidationFlags, Validator},
        AddressSpace,
    };

    fn parse(source: &str) -> Module {
        crate::front::wgsl::parse_str(source).unwrap()
    }

    fn validator() -> Validator {
        Validator::new(ValidationFlags::all(), Capabilities::all())
    }

    /// Run the pass on `source`, then compact and validate it as the backend
    /// pass runner does. Also return whether the pass changed the module.
    fn try_prepare(source: &str) -> Result<(Module, bool), PassError> {
        let module = parse(source);
        validator()
            .validate(&module)
            .unwrap_or_else(|e| panic!("invalid input: {}", e.emit_to_string(source)));
        let mut prepared = Cow::Borrowed(&module);
        let changed = run(&mut prepared)?;
        let (prepared, _) = super::super::super::finish(
            prepared,
            changed,
            false,
            &crate::compact::Dropped::default(),
        )?;
        Ok((prepared.into_owned(), changed))
    }

    fn prepare(source: &str) -> (Module, ModuleInfo) {
        let (module, _) = try_prepare(source).unwrap();
        let info = validator()
            .validate_resolved_overrides(&module)
            .unwrap_or_else(|e| panic!("invalid output: {e:?}\n{module:#?}"));
        (module, info)
    }

    fn global_type_name<'a>(module: &'a Module, name: &str) -> Option<&'a str> {
        let (_, global) = module
            .global_variables
            .iter()
            .find(|&(_, g)| g.name.as_deref() == Some(name))
            .unwrap();
        module.types[global.ty].name.as_deref()
    }

    fn named(function: &Function, name: &str) -> Handle<Expression> {
        *function
            .named_expressions
            .iter()
            .find(|&(_, n)| n == name)
            .unwrap()
            .0
    }

    fn has_pointer_to_affected_matrix(module: &Module, info: &ModuleInfo) -> bool {
        let function = &module.entry_points[0].function;
        let info = info.get_entry_point(0);
        function.expressions.iter().any(|(handle, _)| {
            match *info[handle].ty.inner_with(&module.types) {
                TypeInner::Pointer { base, .. } => matches!(
                    module.types[base].inner,
                    TypeInner::Matrix { rows, scalar, .. } if is_affected_matrix(rows, scalar)
                ),
                _ => false,
            }
        })
    }

    fn calls(block: &[Statement], out: &mut Vec<Handle<Function>>) {
        for statement in block {
            match *statement {
                Statement::Call { function, .. } => out.push(function),
                Statement::Block(ref block) => calls(block, out),
                Statement::If {
                    ref accept,
                    ref reject,
                    ..
                } => {
                    calls(accept, out);
                    calls(reject, out);
                }
                Statement::Switch { ref cases, .. } => {
                    for case in cases {
                        calls(&case.body, out);
                    }
                }
                Statement::Loop {
                    ref body,
                    ref continuing,
                    ..
                } => {
                    calls(body, out);
                    calls(continuing, out);
                }
                _ => {}
            }
        }
    }

    fn calls_of(function: &Function) -> Vec<Handle<Function>> {
        let mut out = Vec::new();
        calls(&function.body, &mut out);
        out
    }

    fn is_converter(function: &Function) -> bool {
        function
            .name
            .as_deref()
            .is_some_and(|name| name.starts_with("convert"))
    }

    #[test]
    fn load_split_member() {
        let (module, info) = prepare(
            "
            struct S { m: mat3x2<f32> }
            @group(0) @binding(0) var<uniform> u: S;
            @compute @workgroup_size(1) fn main() { let x = u.m; }
            ",
        );
        assert_eq!(global_type_name(&module, "u"), Some("S_std140"));
        assert!(!has_pointer_to_affected_matrix(&module, &info));
        let function = &module.entry_points[0].function;
        assert!(matches!(
            *info.get_entry_point(0)[named(function, "x")]
                .ty
                .inner_with(&module.types),
            TypeInner::Matrix {
                columns: VectorSize::Tri,
                rows: VectorSize::Bi,
                ..
            }
        ));
    }

    #[test]
    fn load_whole_struct() {
        let (module, info) = prepare(
            "
            struct S { a: f32, m: mat3x2<f32> }
            @group(0) @binding(0) var<uniform> u: S;
            @compute @workgroup_size(1) fn main() { let s = u; }
            ",
        );
        let (s, _) = module
            .types
            .iter()
            .find(|&(_, ty)| ty.name.as_deref() == Some("S"))
            .unwrap();
        let function = &module.entry_points[0].function;
        assert_eq!(
            info.get_entry_point(0)[named(function, "s")].ty.handle(),
            Some(s)
        );
    }

    #[test]
    fn constant_column() {
        let (module, _) = prepare(
            "
            struct S { m: mat3x2<f32> }
            @group(0) @binding(0) var<uniform> u: S;
            @compute @workgroup_size(1) fn main() { let c = u.m[1]; }
            ",
        );
        let function = &module.entry_points[0].function;
        assert!(function.expressions.iter().any(|(_, expr)| match *expr {
            Expression::AccessIndex { base, index: 1 } => {
                matches!(function.expressions[base], Expression::GlobalVariable(_))
            }
            _ => false,
        }));
        assert!(!function.expressions.iter().any(|(_, expr)| match *expr {
            Expression::Compose { ty, .. } =>
                matches!(module.types[ty].inner, TypeInner::Matrix { .. }),
            _ => false,
        }));
    }

    #[test]
    fn dynamic_column_then_component() {
        let (module, info) = prepare(
            "
            struct Inner { m: mat2x2<f32> }
            struct S { m: mat3x2<f32>, @align(16) s: Inner }
            @group(0) @binding(0) var<uniform> u: S;
            @compute @workgroup_size(1) fn main() {
                var i = 1;
                let y = u.m[i].y;
                let z = u.s.m[i][i];
                let w = u.m[i] + u.m[1];
                let p = &u.m;
                let q = (*p)[i];
                if i > 0 {
                    let k = u.s.m;
                }
            }
            ",
        );
        assert!(!has_pointer_to_affected_matrix(&module, &info));
    }

    #[test]
    fn top_level_matrix() {
        let (module, info) = prepare(
            "
            @group(0) @binding(0) var<uniform> m: mat2x2<f32>;
            @compute @workgroup_size(1) fn main() {
                var i = 1;
                let x = m;
                let c = m[1];
                let d = m[i].x;
            }
            ",
        );
        assert_eq!(global_type_name(&module, "m"), Some("mat2x2f32_std140"));
        assert!(!has_pointer_to_affected_matrix(&module, &info));
    }

    #[test]
    fn array_whole_load() {
        let (module, _) = prepare(
            "
            @group(0) @binding(0) var<uniform> a: array<mat2x2<f32>, 4>;
            @compute @workgroup_size(1) fn main() {
                let x = a;
                let e = a[1];
                let c = a[2][1];
            }
            ",
        );
        let (first, converter) = module.functions.iter().next().unwrap();
        assert!(is_converter(converter));
        assert_eq!(module.functions.len(), 1);
        assert_eq!(calls_of(&module.entry_points[0].function), [first]);
    }

    #[test]
    fn helper_function_loads_array() {
        let (module, _) = prepare(
            "
            @group(0) @binding(0) var<uniform> a: array<mat2x2<f32>, 4>;
            fn fetch() -> array<mat2x2<f32>, 4> { return a; }
            @compute @workgroup_size(1) fn main() { let x = fetch(); }
            ",
        );
        let functions: Vec<_> = module.functions.iter().collect();
        assert_eq!(functions.len(), 2);
        assert!(is_converter(functions[0].1));
        assert_eq!(functions[1].1.name.as_deref(), Some("fetch"));
        assert_eq!(calls_of(functions[1].1), [functions[0].0]);
        assert_eq!(calls_of(&module.entry_points[0].function), [functions[1].0]);
    }

    #[test]
    fn nested_array_whole_load() {
        let (module, _) = prepare(
            "
            struct T { m: mat2x2<f32>, arr: array<mat2x2<f32>, 2> }
            @group(0) @binding(0) var<uniform> a: array<array<mat2x2<f32>, 2>, 3>;
            @group(0) @binding(1) var<uniform> t: array<T, 2>;
            @compute @workgroup_size(1) fn main() {
                let x = a;
                let y = t;
                let z = t[1];
                let w = a[1];
            }
            ",
        );
        assert_eq!(module.functions.len(), 3);
        assert!(module.functions.iter().all(|(_, f)| is_converter(f)));
    }

    #[test]
    fn shared_struct_across_spaces() {
        let (module, _) = prepare(
            "
            struct S { m: mat2x2<f32> }
            @group(0) @binding(0) var<uniform> u: S;
            @group(0) @binding(1) var<storage, read_write> o: S;
            @compute @workgroup_size(1) fn main() {
                o = u;
                var l: S = u;
                o.m = l.m;
            }
            ",
        );
        assert_eq!(global_type_name(&module, "u"), Some("S_std140"));
        assert_eq!(global_type_name(&module, "o"), Some("S"));
    }

    #[test]
    fn binding_array_of_uniform_struct() {
        let (module, info) = prepare(
            "
            enable wgpu_binding_array;
            struct S { m: mat2x2<f32> }
            @group(0) @binding(0) var<uniform> b: binding_array<S, 2>;
            @compute @workgroup_size(1) fn main() {
                let x = b[0].m;
                let y = b[1];
            }
            ",
        );
        assert!(!has_pointer_to_affected_matrix(&module, &info));
    }

    #[test]
    fn f16_matrices() {
        let (module, info) = prepare(
            "
            enable f16;
            struct S { m: mat2x3<f16>, n: mat4x4<f16> }
            @group(0) @binding(0) var<uniform> u: S;
            @compute @workgroup_size(1) fn main() {
                var i = 1;
                let x = u.m;
                let y = u.n[i];
                let z = u;
            }
            ",
        );
        assert!(!has_pointer_to_affected_matrix(&module, &info));
    }

    #[test]
    fn untouched_without_affected() {
        let source = "
            struct S { m: mat3x3<f32> }
            @group(0) @binding(0) var<uniform> u: S;
            @group(0) @binding(1) var<uniform> v: mat3x3<f32>;
            fn fetch() -> mat3x3<f32> { return v; }
            @compute @workgroup_size(1) fn main() { let x = u.m; let y = fetch(); }
            ";
        let before = parse(source);
        let (after, changed) = try_prepare(source).unwrap();
        assert!(!changed);
        assert_eq!(after.types.len(), before.types.len());
        assert_eq!(after.functions.len(), before.functions.len());
        for ((_, a), (_, b)) in after
            .global_variables
            .iter()
            .zip(before.global_variables.iter())
        {
            assert_eq!(a.ty, b.ty);
        }
    }

    /// The member index that `handle`, a column or a load of one, reads.
    fn member_index(function: &Function, handle: Handle<Expression>) -> u32 {
        match function.expressions[handle] {
            Expression::Load { pointer } => member_index(function, pointer),
            Expression::AccessIndex { index, .. } => index,
            ref other => panic!("not a member access: {other:?}"),
        }
    }

    fn compose_members(function: &Function, handle: Handle<Expression>) -> Vec<u32> {
        let Expression::Compose { ref components, .. } = function.expressions[handle] else {
            panic!("not a compose: {:?}", function.expressions[handle]);
        };
        components
            .iter()
            .map(|&component| member_index(function, component))
            .collect()
    }

    #[test]
    fn split_member_indices() {
        let (module, _) = prepare(
            "
            struct S { m: mat2x2<f32>, n: mat2x2<f32> }
            @group(0) @binding(0) var<uniform> u: S;
            @compute @workgroup_size(1) fn main() {
                var i = 1;
                let whole = u.n;
                let column = u.n[1];
                let dynamic = u.n[i];
                let s = u;
            }
            ",
        );
        let function = &module.entry_points[0].function;

        assert_eq!(compose_members(function, named(function, "whole")), [2, 3]);

        let Expression::Load { pointer } = function.expressions[named(function, "column")] else {
            panic!("constant column is not a load");
        };
        let Expression::AccessIndex { base, index } = function.expressions[pointer] else {
            panic!("constant column is not a member access");
        };
        assert!(matches!(
            function.expressions[base],
            Expression::GlobalVariable(_)
        ));
        assert_eq!(index, 3);

        let Expression::Access { base, .. } = function.expressions[named(function, "dynamic")]
        else {
            panic!("dynamic column is not an access");
        };
        assert_eq!(compose_members(function, base), [2, 3]);

        let Expression::Compose { ref components, .. } = function.expressions[named(function, "s")]
        else {
            panic!("whole struct is not a compose");
        };
        let members: Vec<_> = components
            .iter()
            .map(|&matrix| compose_members(function, matrix))
            .collect();
        assert_eq!(members, [[0, 1], [2, 3]]);
    }

    fn has_loop(block: &[Statement]) -> bool {
        block.iter().any(|statement| match *statement {
            Statement::Loop { .. } => true,
            Statement::Block(ref block) => has_loop(block),
            _ => false,
        })
    }

    #[test]
    fn small_array_converter_is_unrolled() {
        let (module, _) = prepare(
            "
            @group(0) @binding(0) var<uniform> a: array<mat2x2<f32>, 16>;
            @compute @workgroup_size(1) fn main() { let x = a; }
            ",
        );
        let (_, converter) = module.functions.iter().next().unwrap();
        assert!(is_converter(converter));
        assert!(!has_loop(&converter.body));
        assert!(converter.local_variables.is_empty());
    }

    #[test]
    fn large_array_converter_uses_loop() {
        let (module, _) = prepare(
            "
            struct T { m: mat2x2<f32>, arr: array<mat2x2<f32>, 64> }
            @group(0) @binding(0) var<uniform> a: array<mat2x2<f32>, 64>;
            @group(0) @binding(1) var<uniform> b: array<array<mat2x2<f32>, 32>, 32>;
            @group(0) @binding(2) var<uniform> t: array<T, 2>;
            @compute @workgroup_size(1) fn main() {
                let x = a;
                let y = b;
                let z = t;
            }
            ",
        );
        let converters: Vec<_> = module.functions.iter().collect();
        assert_eq!(converters.len(), 4);
        for &(_, converter) in &converters {
            assert!(is_converter(converter));
        }
        let looped = converters
            .iter()
            .filter(|&&(_, converter)| has_loop(&converter.body))
            .count();
        assert_eq!(looped, 3);
        assert!(module
            .types
            .iter()
            .any(|(_, ty)| ty.inner == TypeInner::Scalar(Scalar::U32)));
    }

    #[test]
    fn huge_array_converter_is_small() {
        let (module, _) = prepare(
            "
            @group(0) @binding(0) var<uniform> a: array<mat2x2<f32>, 65536>;
            @compute @workgroup_size(1) fn main() { let x = a; }
            ",
        );
        let (_, converter) = module.functions.iter().next().unwrap();
        assert!(has_loop(&converter.body));
        assert!(converter.expressions.len() < 32);
    }

    #[test]
    fn loads_in_loops_and_switches() {
        let (module, info) = prepare(
            "
            struct S { m: mat2x2<f32> }
            @group(0) @binding(0) var<uniform> a: array<mat2x2<f32>, 4>;
            @group(0) @binding(1) var<uniform> u: S;
            @compute @workgroup_size(1) fn main() {
                var i = 0;
                var acc = vec2<f32>();
                loop {
                    acc += u.m * acc;
                    let c = u.m[i];
                    continuing {
                        let x = a;
                        acc += (x[i] + u.m + a[i]) * c;
                        i += 1;
                        break if u.m[i].x > 0.0;
                    }
                }
                switch i {
                    case 0: { acc += (a[1] + u.m)[0]; }
                    default: { let y = a; acc += y[0] * u.m[i]; }
                }
            }
            ",
        );
        assert!(!has_pointer_to_affected_matrix(&module, &info));
        let (converter, _) = module.functions.iter().next().unwrap();
        assert_eq!(
            calls_of(&module.entry_points[0].function),
            [converter, converter]
        );
    }

    #[test]
    fn entry_points_share_helper() {
        let module = parse(
            "
            @group(0) @binding(0) var<uniform> a: array<mat2x2<f32>, 4>;
            fn fetch() -> array<mat2x2<f32>, 4> { return a; }
            @compute @workgroup_size(1) fn first() { let x = fetch(); }
            @compute @workgroup_size(1) fn second() { let y = fetch()[1]; }
            ",
        );
        validator().validate(&module).unwrap();
        let (module, _) = super::super::super::run(
            Cow::Borrowed(&module),
            &super::super::super::Passes {
                compact: false,
                uniform_matrix_layout: true,
            },
        )
        .unwrap();
        let functions: Vec<_> = module.functions.iter().collect();
        assert_eq!(functions.len(), 2);
        assert!(is_converter(functions[0].1));
        assert_eq!(functions[1].1.name.as_deref(), Some("fetch"));
        assert_eq!(calls_of(functions[1].1), [functions[0].0]);
        assert_eq!(module.entry_points.len(), 2);
        for entry_point in &module.entry_points {
            assert_eq!(calls_of(&entry_point.function), [functions[1].0]);
        }
    }

    #[test]
    fn function_doc_comments_follow_functions() {
        let source = "
            @group(0) @binding(0) var<uniform> a: array<mat2x2<f32>, 4>;
            /// Fetches the array.
            fn fetch() -> array<mat2x2<f32>, 4> { return a; }
            /// Returns one.
            fn one() -> f32 { return 1.0; }
            @compute @workgroup_size(1) fn main() { let x = fetch(); let y = one(); }
            ";
        let module = crate::front::wgsl::Frontend::new_with_options(crate::front::wgsl::Options {
            parse_doc_comments: true,
            ..crate::front::wgsl::Options::new()
        })
        .parse(source)
        .unwrap();
        let comments =
            |module: &Module| -> Vec<(Option<alloc::string::String>, Vec<alloc::string::String>)> {
                let mut out: Vec<_> = module
                    .doc_comments
                    .as_ref()
                    .unwrap()
                    .functions
                    .iter()
                    .map(|(&handle, comments)| {
                        (module.functions[handle].name.clone(), comments.clone())
                    })
                    .collect();
                out.sort();
                out
            };
        let before = comments(&module);
        assert_eq!(before.len(), 2);
        let mut prepared = Cow::Borrowed(&module);
        run(&mut prepared).unwrap();
        let mut module = prepared.into_owned();
        crate::compact::try_compact(
            &mut module,
            crate::compact::KeepUnused::Yes,
            &crate::compact::Dropped::default(),
        )
        .unwrap();
        assert_eq!(module.functions.len(), 3);
        assert_eq!(comments(&module), before);
    }

    #[test]
    fn store_to_split_matrix_is_an_error() {
        let mut module = parse(
            "
            struct S { m: mat2x2<f32> }
            var<private> g: S;
            @compute @workgroup_size(1) fn main() { g.m = mat2x2<f32>(); }
            ",
        );
        for (_, global) in module.global_variables.iter_mut() {
            global.space = AddressSpace::Uniform;
        }
        let result = run(&mut Cow::Borrowed(&module));
        assert!(matches!(
            result,
            Err(PassError::UniformMatrixLayout(UnsupportedUse(_)))
        ));
    }

    #[test]
    fn pointer_argument_is_an_error() {
        let mut module = parse(
            "
            struct S { m: mat2x2<f32> }
            var<private> g: S;
            fn f(p: ptr<private, S>) {}
            @compute @workgroup_size(1) fn main() { f(&g); }
            ",
        );
        for (_, global) in module.global_variables.iter_mut() {
            global.space = AddressSpace::Uniform;
        }
        let result = run(&mut Cow::Borrowed(&module));
        assert!(matches!(
            result,
            Err(PassError::UniformMatrixLayout(UnsupportedUse(_)))
        ));
    }
}
