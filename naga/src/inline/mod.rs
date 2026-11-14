//! Replacing function calls with the body of the called function.
//!
//! WGSL's `unrestricted_pointer_parameters` language extension permits pointer
//! parameters in the `storage`, `uniform` and `workgroup` address spaces, and
//! pointer arguments that refer to part of a variable, like `&a[i]`. Not every
//! backend can express these, so [`inline`] replaces such calls with the called
//! function's body.

mod adjust;

use alloc::{borrow::Cow, boxed::Box, string::String, vec, vec::Vec};

use crate::{
    arena::{Arena, Handle, Range},
    compact::{compact, KeepUnused},
    non_max_u32::NonMaxU32,
    valid::{Capabilities, ModuleInfo, ValidationError, ValidationFlags, Validator},
    AddressSpace, Block, Expression, Function, Literal, LocalVariable, Module, Scalar, Span,
    Statement, SwitchCase, SwitchValue, Type, TypeInner, WithSpan,
};

/// Which calls [`inline`] replaces with the body of the called function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum InlineStrategy {
    /// Inline nothing.
    Never,
    /// Inline only the calls that the given backend cannot express.
    OnlyNeeded(InlineTarget),
    /// Inline every call.
    All,
}

/// The backend that [`InlineStrategy::OnlyNeeded`] inlines calls for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum InlineTarget {
    /// Logical SPIR-V: inline calls to functions with `storage`, `uniform` or
    /// `workgroup` pointer parameters, and calls with pointer arguments that
    /// refer to part of a variable.
    Spirv,
    /// Inline calls to functions with `storage`, `uniform` or `workgroup`
    /// pointer parameters.
    Hlsl,
    /// Inline calls to functions with `storage`, `uniform` or `workgroup`
    /// pointer parameters.
    Glsl,
    /// Inline calls to functions with pointer parameters to runtime-sized
    /// types, whose length the callee could not determine.
    Msl,
}

/// The error returned by [`process`] if the inlined module fails validation.
pub type InlineError = Box<WithSpan<ValidationError>>;

/// Inline calls in `module` according to `strategy`, then compact and
/// revalidate the result.
///
/// If nothing needs to be inlined, return `module` and `module_info`
/// unchanged.
pub fn process<'a>(
    module: &'a Module,
    module_info: &'a ModuleInfo,
    strategy: InlineStrategy,
) -> Result<(Cow<'a, Module>, Cow<'a, ModuleInfo>), InlineError> {
    if !needs_inline(module, strategy) {
        return Ok((Cow::Borrowed(module), Cow::Borrowed(module_info)));
    }
    let mut module = module.clone();
    inline(&mut module, strategy);
    compact(&mut module, KeepUnused::No);
    let module_info =
        Validator::new(ValidationFlags::all(), Capabilities::all()).validate(&module)?;
    Ok((Cow::Owned(module), Cow::Owned(module_info)))
}

/// Return `true` if [`process`] would change `module`: some call must be
/// inlined, or some function must be inlined at every call site and should be
/// removed even if it is never called.
pub fn needs_inline(module: &Module, strategy: InlineStrategy) -> bool {
    let Some(checker) = Checker::new(module, strategy) else {
        return false;
    };
    find_needs_inline_impl(module, &checker).is_some()
}

/// Return the name of a function that `target` cannot express unless
/// [`process`] has inlined it, or that contains a call `target` cannot
/// express.
pub(crate) fn find_needs_inline(module: &Module, target: InlineTarget) -> Option<String> {
    let checker = Checker::new(module, InlineStrategy::OnlyNeeded(target))?;
    find_needs_inline_impl(module, &checker)
}

fn find_needs_inline_impl(module: &Module, checker: &Checker) -> Option<String> {
    let name = |name: &Option<String>| name.clone().unwrap_or_default();
    if let Some((_, function)) = module
        .functions
        .iter()
        .find(|&(handle, _)| checker.needs_inline[handle.index()])
    {
        return Some(name(&function.name));
    }
    module
        .functions
        .iter()
        .map(|(_, function)| function)
        .chain(module.entry_points.iter().map(|ep| &ep.function))
        .find(|function| checker.block_needs_inline(module, function, &function.body))
        .map(|function| name(&function.name))
}

/// Inline calls in every function and entry point of `module`, according to
/// `strategy`. Return `true` if anything was inlined.
///
/// The result may contain forward references between expressions, so it must
/// be passed through [`compact`] before it is validated. Functions that were
/// inlined at every call site are removed by [`compact`] with
/// [`KeepUnused::No`].
///
/// [`compact`]: crate::compact::compact
/// [`KeepUnused::No`]: crate::compact::KeepUnused::No
pub fn inline(module: &mut Module, strategy: InlineStrategy) -> bool {
    let Some(checker) = Checker::new(module, strategy) else {
        return false;
    };
    let mut inliner = Inliner {
        prepared: (0..module.functions.len()).map(|_| None).collect(),
        checker,
        module,
        inlined: false,
    };

    // A function may only call functions that precede it, so every callee is
    // processed before its callers.
    for index in 0..inliner.module.functions.len() {
        let handle = Handle::new(NonMaxU32::new(index as u32).unwrap());
        let mut function = core::mem::take(&mut inliner.module.functions[handle]);
        inliner.inline_calls(&mut function);
        inliner.module.functions[handle] = function;
    }
    for index in 0..inliner.module.entry_points.len() {
        let mut function = core::mem::take(&mut inliner.module.entry_points[index].function);
        inliner.inline_calls(&mut function);
        inliner.module.entry_points[index].function = function;
    }

    inliner.inlined
}

const fn is_unrestricted_pointer_space(space: AddressSpace) -> bool {
    matches!(
        space,
        AddressSpace::Uniform | AddressSpace::Storage { .. } | AddressSpace::WorkGroup
    )
}

/// Return `true` if every call to `function` must be inlined for `target`.
fn function_needs_inline(module: &Module, function: &Function, target: InlineTarget) -> bool {
    function.arguments.iter().any(|argument| {
        let TypeInner::Pointer { base, space } = module.types[argument.ty].inner else {
            return false;
        };
        match target {
            InlineTarget::Spirv | InlineTarget::Hlsl | InlineTarget::Glsl => {
                is_unrestricted_pointer_space(space)
            }
            InlineTarget::Msl => module.types[base].inner.is_dynamically_sized(&module.types),
        }
    })
}

/// A copy of a function, rewritten so that its body can be copied in place of
/// a call to it.
struct Prepared {
    /// The rewritten function.
    ///
    /// It stores its return value in a local variable instead of returning
    /// it. If it returned anywhere but at the end, its body is wrapped in a
    /// [`Statement::Switch`] with a single case, and its `Return` statements
    /// have become `Break`s out of it.
    function: Function,

    /// The [`Expression::LocalVariable`] for the return value, if the function
    /// has one.
    return_pointer: Option<Handle<Expression>>,
}

struct Inliner<'a> {
    module: &'a mut Module,
    checker: Checker,
    prepared: Vec<Option<Prepared>>,
    inlined: bool,
}

/// Decides which calls must be inlined.
struct Checker {
    /// For each function in `module`, whether every call to it must be
    /// inlined.
    needs_inline: Vec<bool>,
    inline_sub_object_arguments: bool,
}

impl Checker {
    fn new(module: &Module, strategy: InlineStrategy) -> Option<Self> {
        let (needs_inline, inline_sub_object_arguments) = match strategy {
            InlineStrategy::Never => return None,
            InlineStrategy::OnlyNeeded(target) => (
                module
                    .functions
                    .iter()
                    .map(|(_, function)| function_needs_inline(module, function, target))
                    .collect(),
                target == InlineTarget::Spirv,
            ),
            InlineStrategy::All => (vec![true; module.functions.len()], false),
        };
        Some(Self {
            needs_inline,
            inline_sub_object_arguments,
        })
    }

    fn call_needs_inline(
        &self,
        module: &Module,
        caller: &Function,
        callee: Handle<Function>,
        arguments: &[Handle<Expression>],
    ) -> bool {
        if self.needs_inline[callee.index()] {
            return true;
        }
        if !self.inline_sub_object_arguments {
            return false;
        }
        let parameters = &module.functions[callee].arguments;
        parameters
            .iter()
            .zip(arguments)
            .any(|(parameter, &argument)| {
                module.types[parameter.ty].inner.pointer_space().is_some()
                    && !matches!(
                        caller.expressions[argument],
                        Expression::LocalVariable(_)
                            | Expression::GlobalVariable(_)
                            | Expression::FunctionArgument(_)
                    )
            })
    }

    fn block_needs_inline(&self, module: &Module, function: &Function, block: &Block) -> bool {
        block.iter().any(|statement| match *statement {
            Statement::Call {
                function: callee,
                ref arguments,
                ..
            } => self.call_needs_inline(module, function, callee, arguments),
            Statement::Block(ref block) => self.block_needs_inline(module, function, block),
            Statement::If {
                ref accept,
                ref reject,
                ..
            } => {
                self.block_needs_inline(module, function, accept)
                    || self.block_needs_inline(module, function, reject)
            }
            Statement::Switch { ref cases, .. } => cases
                .iter()
                .any(|case| self.block_needs_inline(module, function, &case.body)),
            Statement::Loop {
                ref body,
                ref continuing,
                ..
            } => {
                self.block_needs_inline(module, function, body)
                    || self.block_needs_inline(module, function, continuing)
            }
            _ => false,
        })
    }
}

impl Inliner<'_> {
    fn inline_calls(&mut self, function: &mut Function) {
        let body = core::mem::take(&mut function.body);
        function.body = self.inline_block(function, body);
    }

    fn inline_block(&mut self, function: &mut Function, block: Block) -> Block {
        let mut new_block = Block::with_capacity(block.len());
        for (statement, span) in block.span_into_iter() {
            let statement = match statement {
                Statement::Call {
                    function: callee,
                    ref arguments,
                    result,
                } if self
                    .checker
                    .call_needs_inline(self.module, function, callee, arguments) =>
                {
                    self.inlined = true;
                    self.prepare(callee);
                    let prepared = self.prepared[callee.index()].as_ref().unwrap();
                    let inlined = splice(prepared, function, arguments, result, span);
                    // Pointer arguments passed through to calls in the callee
                    // may now refer to part of a variable.
                    new_block.extend_block(self.inline_block(function, inlined));
                    continue;
                }
                Statement::Block(block) => Statement::Block(self.inline_block(function, block)),
                Statement::If {
                    condition,
                    accept,
                    reject,
                } => Statement::If {
                    condition,
                    accept: self.inline_block(function, accept),
                    reject: self.inline_block(function, reject),
                },
                Statement::Switch { selector, cases } => Statement::Switch {
                    selector,
                    cases: cases
                        .into_iter()
                        .map(|case| SwitchCase {
                            body: self.inline_block(function, case.body),
                            ..case
                        })
                        .collect(),
                },
                Statement::Loop {
                    body,
                    continuing,
                    break_if,
                } => Statement::Loop {
                    body: self.inline_block(function, body),
                    continuing: self.inline_block(function, continuing),
                    break_if,
                },
                statement => statement,
            };
            new_block.push(statement, span);
        }
        new_block
    }

    fn prepare(&mut self, callee: Handle<Function>) {
        if self.prepared[callee.index()].is_some() {
            return;
        }
        let function = self.module.functions[callee].clone();
        let prepared = prepare(&mut self.module.types, function);
        self.prepared[callee.index()] = Some(prepared);
    }
}

fn prepare(types: &mut crate::UniqueArena<Type>, mut function: Function) -> Prepared {
    let mut body = Block::new();

    // The inlined body may run more than once per call of the caller, for
    // example in a loop, so local variables must be initialized explicitly.
    for (local, variable, &span) in function.local_variables.iter_mut_span() {
        let pointer = function
            .expressions
            .append(Expression::LocalVariable(local), span);
        let value = match variable.init.take() {
            Some(init) => copy_init(&mut function.expressions, init, &mut body),
            None => function
                .expressions
                .append(Expression::ZeroValue(variable.ty), span),
        };
        body.push(Statement::Store { pointer, value }, span);
    }

    let return_pointer = function.result.as_ref().map(|result| {
        let local = function.local_variables.append(
            LocalVariable {
                name: None,
                ty: result.ty,
                init: None,
            },
            Span::UNDEFINED,
        );
        function
            .expressions
            .append(Expression::LocalVariable(local), Span::UNDEFINED)
    });

    let done = has_nested_return(&function.body, false).then(|| {
        let ty = types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::BOOL),
            },
            Span::UNDEFINED,
        );
        let local = function.local_variables.append(
            LocalVariable {
                name: None,
                ty,
                init: None,
            },
            Span::UNDEFINED,
        );
        let pointer = function
            .expressions
            .append(Expression::LocalVariable(local), Span::UNDEFINED);
        let value = function
            .expressions
            .append(Expression::Literal(Literal::Bool(false)), Span::UNDEFINED);
        body.push(Statement::Store { pointer, value }, Span::UNDEFINED);
        let true_value = function
            .expressions
            .append(Expression::Literal(Literal::Bool(true)), Span::UNDEFINED);
        (pointer, true_value)
    });

    let mut old_body = core::mem::take(&mut function.body);
    let trailing_return = match old_body.last() {
        Some(&Statement::Return { value }) => Some(value),
        _ => None,
    };
    let body_len = old_body.len() - usize::from(trailing_return.is_some());
    if has_return(&old_body[..body_len]) {
        // Only `Break` can leave the middle of a block, so wrap the body in a
        // `Switch` with a single case and turn `Return`s into `Break`s.
        let mut rewriter = ReturnRewriter {
            expressions: &mut function.expressions,
            return_pointer,
            done,
        };
        let (old_body, _) = rewriter.rewrite(old_body, false);
        body.extend_block(old_body);
        if let Some(&Statement::Break) = body.last() {
            body.cull(body.len() - 1..);
        }
        let selector = function
            .expressions
            .append(Expression::Literal(Literal::I32(0)), Span::UNDEFINED);
        function.body.push(
            Statement::Switch {
                selector,
                cases: vec![SwitchCase {
                    value: SwitchValue::Default,
                    body,
                    fall_through: false,
                }],
            },
            Span::UNDEFINED,
        );
    } else {
        if let Some(value) = trailing_return {
            old_body.cull(body_len..);
            if let (Some(pointer), Some(value)) = (return_pointer, value) {
                old_body.push(Statement::Store { pointer, value }, Span::UNDEFINED);
            }
        }
        body.extend_block(old_body);
        function.body = body;
    }
    function.named_expressions.clear();

    Prepared {
        function,
        return_pointer,
    }
}

fn has_return(block: &[Statement]) -> bool {
    block.iter().any(|statement| match *statement {
        Statement::Return { .. } => true,
        Statement::Block(ref block) => has_return(block),
        Statement::If {
            ref accept,
            ref reject,
            ..
        } => has_return(accept) || has_return(reject),
        Statement::Switch { ref cases, .. } => cases.iter().any(|case| has_return(&case.body)),
        Statement::Loop { ref body, .. } => has_return(body),
        _ => false,
    })
}

/// Return `true` if `block` has a `Return` inside a `Loop` or `Switch`, from
/// which a `Break` would not leave the function.
fn has_nested_return(block: &[Statement], nested: bool) -> bool {
    block.iter().any(|statement| match *statement {
        Statement::Return { .. } => nested,
        Statement::Block(ref block) => has_nested_return(block, nested),
        Statement::If {
            ref accept,
            ref reject,
            ..
        } => has_nested_return(accept, nested) || has_nested_return(reject, nested),
        Statement::Switch { ref cases, .. } => {
            cases.iter().any(|case| has_nested_return(&case.body, true))
        }
        Statement::Loop { ref body, .. } => has_nested_return(body, true),
        _ => false,
    })
}

/// Copy the expressions computing a local variable's initializer, and emit
/// them at the end of `block`.
fn copy_init(
    expressions: &mut Arena<Expression>,
    init: Handle<Expression>,
    block: &mut Block,
) -> Handle<Expression> {
    fn collect(
        expressions: &Arena<Expression>,
        handle: Handle<Expression>,
        order: &mut Vec<Handle<Expression>>,
    ) {
        if expressions[handle].needs_pre_emit() || order.contains(&handle) {
            return;
        }
        let mut operands = Vec::new();
        adjust::adjust_expression(
            &mut expressions[handle].clone(),
            &mut |&mut operand| operands.push(operand),
            &mut |_| {},
        );
        for operand in operands {
            collect(expressions, operand, order);
        }
        order.push(handle);
    }

    if expressions[init].needs_pre_emit() {
        return init;
    }
    let mut order = Vec::new();
    collect(expressions, init, &mut order);

    let start = expressions.len();
    for &handle in &order {
        let mut expr = expressions[handle].clone();
        adjust::adjust_expression(
            &mut expr,
            &mut |operand| {
                if let Some(index) = order.iter().position(|&h| h == *operand) {
                    *operand = Handle::new(NonMaxU32::new((start + index) as u32).unwrap());
                }
            },
            &mut |_| {},
        );
        let span = expressions.get_span(handle);
        expressions.append(expr, span);
    }
    block.push(
        Statement::Emit(expressions.range_from(start)),
        Span::UNDEFINED,
    );
    Handle::new(NonMaxU32::new((start + order.len() - 1) as u32).unwrap())
}

struct ReturnRewriter<'a> {
    expressions: &'a mut Arena<Expression>,
    return_pointer: Option<Handle<Expression>>,
    /// The pointer to the local variable recording that the function has
    /// returned, and the `true` literal to store in it.
    done: Option<(Handle<Expression>, Handle<Expression>)>,
}

impl ReturnRewriter<'_> {
    /// Rewrite the `Return` statements in `block` to `Break` out of the
    /// `Switch` wrapping the function body. If `nested`, `block` is inside a `Loop` or
    /// `Switch` of the function.
    ///
    /// Return the new block, and whether it contains a nested `Return`.
    fn rewrite(&mut self, block: Block, nested: bool) -> (Block, bool) {
        let mut new_block = Block::with_capacity(block.len());
        let mut has_return = false;
        for (statement, span) in block.span_into_iter() {
            match statement {
                Statement::Return { value } => {
                    if let (Some(pointer), Some(value)) = (self.return_pointer, value) {
                        new_block.push(Statement::Store { pointer, value }, span);
                    }
                    if nested {
                        let (pointer, value) = self.done.unwrap();
                        new_block.push(Statement::Store { pointer, value }, span);
                        has_return = true;
                    }
                    new_block.push(Statement::Break, span);
                }
                Statement::Block(block) => {
                    let (block, ret) = self.rewrite(block, nested);
                    has_return |= ret;
                    new_block.push(Statement::Block(block), span);
                }
                Statement::If {
                    condition,
                    accept,
                    reject,
                } => {
                    let (accept, ret_accept) = self.rewrite(accept, nested);
                    let (reject, ret_reject) = self.rewrite(reject, nested);
                    has_return |= ret_accept | ret_reject;
                    new_block.push(
                        Statement::If {
                            condition,
                            accept,
                            reject,
                        },
                        span,
                    );
                }
                Statement::Switch { selector, cases } => {
                    let mut ret = false;
                    let cases = cases
                        .into_iter()
                        .map(|case| {
                            let (body, ret_case) = self.rewrite(case.body, true);
                            ret |= ret_case;
                            SwitchCase { body, ..case }
                        })
                        .collect();
                    new_block.push(Statement::Switch { selector, cases }, span);
                    if ret {
                        self.break_if_done(&mut new_block, span);
                        has_return = true;
                    }
                }
                // `Return` may not appear in `continuing`.
                Statement::Loop {
                    body,
                    continuing,
                    break_if,
                } => {
                    let (body, ret) = self.rewrite(body, true);
                    new_block.push(
                        Statement::Loop {
                            body,
                            continuing,
                            break_if,
                        },
                        span,
                    );
                    if ret {
                        self.break_if_done(&mut new_block, span);
                        has_return = true;
                    }
                }
                statement => new_block.push(statement, span),
            }
        }
        (new_block, has_return)
    }

    /// Push statements to `block` that `Break` if the function has returned.
    fn break_if_done(&mut self, block: &mut Block, span: Span) {
        let (pointer, _) = self.done.unwrap();
        let start = self.expressions.len();
        let condition = self.expressions.append(Expression::Load { pointer }, span);
        block.push(Statement::Emit(self.expressions.range_from(start)), span);
        block.push(
            Statement::If {
                condition,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            span,
        );
    }
}

/// Copy `prepared` into `caller`, in place of a call with `arguments` and
/// `result`.
fn splice(
    prepared: &Prepared,
    caller: &mut Function,
    arguments: &[Handle<Expression>],
    result: Option<Handle<Expression>>,
    span: Span,
) -> Block {
    let callee = &prepared.function;

    let local_offset = caller.local_variables.len();
    for (local, variable) in callee.local_variables.iter() {
        caller
            .local_variables
            .append(variable.clone(), callee.local_variables.get_span(local));
    }

    // `FunctionArgument` expressions become the call's arguments. The callee's
    // expressions may contain forward references, so compute the whole map
    // before copying.
    let mut next = caller.expressions.len();
    let map: Vec<Handle<Expression>> = callee
        .expressions
        .iter()
        .map(|(_, expr)| match *expr {
            Expression::FunctionArgument(index) => arguments[index as usize],
            _ => {
                next += 1;
                Handle::new(NonMaxU32::new(next as u32 - 1).unwrap())
            }
        })
        .collect();
    let mut adjust = |handle: &mut Handle<Expression>| *handle = map[handle.index()];
    let mut adjust_local = |local: &mut Handle<LocalVariable>| {
        *local = Handle::new(NonMaxU32::new((local.index() + local_offset) as u32).unwrap());
    };
    for (handle, expr) in callee.expressions.iter() {
        if let Expression::FunctionArgument(_) = *expr {
            continue;
        }
        let mut expr = expr.clone();
        adjust::adjust_expression(&mut expr, &mut adjust, &mut adjust_local);
        caller
            .expressions
            .append(expr, callee.expressions.get_span(handle));
    }

    let mut block = callee.body.clone();
    adjust::adjust_block(&mut block, &mut adjust);
    for (_, statement_span) in block.span_iter_mut() {
        if let Some(statement_span) = statement_span.filter(|s| **s == Span::UNDEFINED) {
            *statement_span = span;
        }
    }

    if let (Some(result), Some(pointer)) = (result, prepared.return_pointer) {
        caller.expressions[result] = Expression::Load {
            pointer: map[pointer.index()],
        };
        block.push(
            Statement::Emit(Range::new_from_bounds(result, result)),
            span,
        );
    }
    block
}
