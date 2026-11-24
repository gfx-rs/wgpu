//! Bound every loop, so that compilers can't assume a loop terminates.
//!
//! This pass replaces the loop bounding that the SPIR-V, HLSL and MSL backends
//! each used to write themselves. It runs before the passes that generate their
//! own loops, such as workgroup zero initialization, so those loops, which are
//! finite by construction, aren't bounded.
//!
//! The counter is a local variable. SPIR-V could keep it in an `OpPhi`
//! instead (<https://github.com/gfx-rs/wgpu/issues/7116>), but the IR can't
//! express that.
//!
//! # Why
//!
//! In Metal Shading Language, an infinite loop has undefined behavior.
//! (This rule is inherited from C++14.) This means that, if the MSL
//! compiler determines that a given loop will never exit, it may assume
//! that it is never reached. It may thus assume that any conditions
//! sufficient to cause the loop to be reached must be false. Like many
//! optimizing compilers, MSL uses this kind of analysis to establish limits
//! on the range of values variables involved in those conditions might
//! hold.
//!
//! For example, suppose the MSL compiler sees the code:
//!
//! ```ignore
//! if (i >= 10) {
//!     while (true) { }
//! }
//! ```
//!
//! It will recognize that the `while` loop will never terminate, conclude
//! that it must be unreachable, and thus infer that, if this code is
//! reached, then `i < 10` at that point.
//!
//! Now suppose that, at some point where `i` has the same value as above,
//! the compiler sees the code:
//!
//! ```ignore
//! if (i < 10) {
//!     a[i] = 1;
//! }
//! ```
//!
//! Because the compiler is confident that `i < 10`, it will make the
//! assignment to `a[i]` unconditional, rewriting this code as, simply:
//!
//! ```ignore
//! a[i] = 1;
//! ```
//!
//! If that `if` condition was injected by Naga to implement a bounds check,
//! the MSL compiler's optimizations could allow out-of-bounds array
//! accesses to occur.
//!
//! Naga cannot feasibly anticipate whether the MSL compiler will determine
//! that a loop is infinite, so an attacker could craft a Naga module
//! containing an infinite loop protected by conditions that cause the Metal
//! compiler to remove bounds checks that Naga injected elsewhere in the
//! function.
//!
//! This rewrite could occur even if the conditional assignment appears
//! *before* the `while` loop, as long as `i < 10` by the time the loop is
//! reached. This would allow the attacker to save the results of
//! unauthorized reads somewhere accessible before entering the infinite
//! loop. But even worse, the MSL compiler has been observed to simply
//! delete the infinite loop entirely, so that even code dominated by the
//! loop becomes reachable. This would make the attack even more flexible,
//! since shaders that would appear to never terminate would actually exit
//! nicely, after having stolen data from elsewhere in the GPU address
//! space.
//!
//! To avoid UB, Naga must persuade the MSL compiler that no loop Naga
//! generates is infinite. One approach would be to add inline assembly to
//! each loop that is annotated as potentially branching out of the loop,
//! but which in fact generates no instructions. Unfortunately, inline
//! assembly is not handled correctly by some Metal device drivers.
//!
//! A previously used approach was to add the following code to the bottom
//! of every loop:
//!
//! ```ignore
//! if (volatile bool unpredictable = false; unpredictable)
//!     break;
//! ```
//!
//! Although the `if` condition will always be false in any real execution,
//! the `volatile` qualifier prevents the compiler from assuming this. Thus,
//! it must assume that the `break` might be reached, and hence that the
//! loop is not unbounded. This prevents the range analysis impact described
//! above. Unfortunately this prevented the compiler from making important,
//! and safe, optimizations such as loop unrolling and was observed to
//! significantly hurt performance.
//!
//! Our current approach declares a counter before every loop and
//! decrements it every iteration, breaking after 2^64 iterations. In WGSL terms:
//!
//! ```ignore
//! var loop_bound = vec2(4294967295u);
//! loop {
//!   {
//!     let value = loop_bound;
//!     loop_bound = value - vec2(select(0u, 1u, value.y == 0u), 1u);
//!     if all(value == vec2(0u)) { break; }
//!   }
//!   // ...the loop's own body...
//! }
//! ```
//!
//! The counter's initializer sets it for a loop that isn't inside another
//! loop. A loop inside another loop also stores the initial value before it
//! starts, since it may start more than once.
//!
//! The check is a block of its own, and it stores the decremented value before
//! it breaks. Backends that write the loaded value to a temporary then declare
//! it in that block, not in the loop's body. With the temporary in the loop's
//! body, or the store after the `break`, the Metal compiler adds selects and
//! phis to the loop.
//!
//! The check is its own block, so backend temporaries for it end before the
//! loop's own code. Otherwise, C-like compilers route the counter's break, the
//! loop's own exits and the backedge through one shared scope cleanup, and the
//! optimizer merges them into select chains.
//!
//! The counter counts down from `u32::MAX` rather than up from zero, which avoids
//! a hang on certain Intel drivers. See
//! <https://github.com/gfx-rs/wgpu/issues/7319>.
//!
//! This convinces the compiler that the loop is finite and therefore may
//! execute, whilst at the same time allowing optimizations such as loop
//! unrolling. Furthermore the 64-bit counter is large enough it seems
//! implausible that it would affect the execution of any shader.
//!
//! This approach is also used by Chromium WebGPU's Dawn shader compiler:
//! <https://dawn.googlesource.com/dawn/+/d9e2d1f718678ebee0728b999830576c410cce0a/src/tint/lang/core/ir/transform/prevent_infinite_loops.cc>

use alloc::{borrow::Cow, string::String, vec};

use super::emit::Emitting;
use crate::{
    BinaryOperator, Block, Expression, Function, Handle, Literal, LocalVariable, Module,
    RelationalFunction, Scalar, Span, Statement, Type, TypeInner, VectorSize,
};

/// Bound every loop in the functions and entry points of `module`, and return
/// whether there were any.
///
/// Leave `module` untouched if it has no loops.
pub(crate) fn run(module: &mut Cow<'_, Module>) -> bool {
    let has_loops = module
        .functions
        .iter()
        .map(|(_, function)| function)
        .chain(module.entry_points.iter().map(|ep| &ep.function))
        .any(|function| contains_loop(&function.body));
    if !has_loops {
        return false;
    }

    let module = module.to_mut();
    let counter_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Vector {
                size: VectorSize::Bi,
                scalar: Scalar::U32,
            },
        },
        Span::UNDEFINED,
    );
    let functions = module
        .functions
        .iter_mut()
        .map(|(_, function)| function)
        .chain(module.entry_points.iter_mut().map(|ep| &mut ep.function));
    for function in functions {
        if !contains_loop(&function.body) {
            continue;
        }
        let max = function
            .expressions
            .append(Expression::Literal(Literal::U32(u32::MAX)), Span::UNDEFINED);
        let init = function.expressions.append(
            Expression::Splat {
                size: VectorSize::Bi,
                value: max,
            },
            Span::UNDEFINED,
        );
        let counter = Counter {
            ty: counter_ty,
            init,
        };
        let mut body = core::mem::take(&mut function.body);
        bound_loops(&mut body, function, &counter, false);
        function.body = body;
    }
    true
}

/// The type of a function's loop counters, and their initializer.
struct Counter {
    ty: Handle<Type>,
    init: Handle<Expression>,
}

fn contains_loop(block: &Block) -> bool {
    block.iter().any(|statement| match *statement {
        Statement::Loop { .. } => true,
        Statement::Block(ref block) => contains_loop(block),
        Statement::If {
            ref accept,
            ref reject,
            ..
        } => contains_loop(accept) || contains_loop(reject),
        Statement::Switch { ref cases, .. } => cases.iter().any(|case| contains_loop(&case.body)),
        _ => false,
    })
}

/// Bound the loops in `block`, a block of `function`, and in its nested
/// blocks. `in_loop` is whether `block` is inside a loop.
fn bound_loops(block: &mut Block, function: &mut Function, counter: &Counter, in_loop: bool) {
    if !contains_loop(block) {
        return;
    }
    for (mut statement, span) in core::mem::take(block).span_into_iter() {
        match statement {
            Statement::Block(ref mut inner) => bound_loops(inner, function, counter, in_loop),
            Statement::If {
                ref mut accept,
                ref mut reject,
                ..
            } => {
                bound_loops(accept, function, counter, in_loop);
                bound_loops(reject, function, counter, in_loop);
            }
            Statement::Switch { ref mut cases, .. } => {
                for case in cases.iter_mut() {
                    bound_loops(&mut case.body, function, counter, in_loop);
                }
            }
            Statement::Loop {
                ref mut body,
                ref mut continuing,
                ..
            } => {
                // A loop that is known to terminate wouldn't need a counter.
                // This is where such loops could be skipped.
                // Outer loops get their counters first, so they are named
                // first.
                let local = function.local_variables.append(
                    LocalVariable {
                        name: Some(String::from("loop_bound")),
                        ty: counter.ty,
                        init: Some(counter.init),
                    },
                    span,
                );
                bound_loops(body, function, counter, true);
                bound_loops(continuing, function, counter, true);
                if in_loop {
                    reset(block, function, local, span);
                }
                bound_loop(body, function, local, counter.ty, span);
            }
            _ => {}
        }
        block.push(statement, span);
    }
}

/// Store the initial value in `counter`, at the end of `block`.
fn reset(block: &mut Block, function: &mut Function, counter: Handle<LocalVariable>, span: Span) {
    let mut out = Emitting::new(&mut function.expressions, block, span);
    let pointer = out.pre_emit(Expression::LocalVariable(counter));
    let max = out.pre_emit(Expression::Literal(Literal::U32(u32::MAX)));
    let value = out.expr(Expression::Splat {
        size: VectorSize::Bi,
        value: max,
    });
    out.push(Statement::Store { pointer, value });
}

/// Make `counter` bound the loop whose body is `body`: decrement it at the
/// start of every iteration, and break out of the loop if it was zero.
fn bound_loop(
    body: &mut Block,
    function: &mut Function,
    counter: Handle<LocalVariable>,
    counter_ty: Handle<Type>,
    span: Span,
) {
    let mut check = Block::new();
    let mut out = Emitting::new(&mut function.expressions, &mut check, span);
    let pointer = out.pre_emit(Expression::LocalVariable(counter));
    let zero = out.pre_emit(Expression::Literal(Literal::U32(0)));
    let one = out.pre_emit(Expression::Literal(Literal::U32(1)));
    let value = out.expr(Expression::Load { pointer });
    // Decrement the low word, and borrow from the high word when the low word
    // is zero.
    let low = out.expr(Expression::AccessIndex {
        base: value,
        index: 1,
    });
    let low_is_zero = out.expr(Expression::Binary {
        op: BinaryOperator::Equal,
        left: low,
        right: zero,
    });
    let borrow = out.expr(Expression::Select {
        condition: low_is_zero,
        accept: one,
        reject: zero,
    });
    let decrement = out.expr(Expression::Compose {
        ty: counter_ty,
        components: vec![borrow, one],
    });
    let next = out.expr(Expression::Binary {
        op: BinaryOperator::Subtract,
        left: value,
        right: decrement,
    });
    out.push(Statement::Store {
        pointer,
        value: next,
    });
    let zeros = out.expr(Expression::Splat {
        size: VectorSize::Bi,
        value: zero,
    });
    let at_zero = out.expr(Expression::Binary {
        op: BinaryOperator::Equal,
        left: value,
        right: zeros,
    });
    let done = out.expr(Expression::Relational {
        fun: RelationalFunction::All,
        argument: at_zero,
    });
    out.push(Statement::If {
        condition: done,
        accept: Block::from_vec(vec![Statement::Break]),
        reject: Block::new(),
    });

    let mut prologue = Block::new();
    prologue.push(Statement::Block(check), span);
    prologue.append(body);
    *body = prologue;
}

#[cfg(all(test, feature = "wgsl-in"))]
mod tests {
    use alloc::borrow::Cow;

    use crate::valid::{Capabilities, ValidationFlags, Validator};
    use crate::{Block, Expression, Function, Module, Statement};

    /// Run the pass on `source`, then compact and validate the result as the
    /// pass runner does. Return `None` if the pass left the module unchanged.
    fn bound(source: &str) -> Option<Module> {
        let module = crate::front::wgsl::parse_str(source).unwrap();
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap();
        let mut prepared = Cow::Borrowed(&module);
        let changed = super::run(&mut prepared);
        let (prepared, _) = super::super::finish(
            prepared,
            changed,
            false,
            &crate::compact::Dropped::default(),
        )
        .unwrap();
        changed.then(|| prepared.into_owned())
    }

    fn counters(function: &Function) -> usize {
        function
            .local_variables
            .iter()
            .filter(|&(_, local)| local.name.as_deref() == Some("loop_bound"))
            .count()
    }

    /// Whether `body` starts with the counter check: a block that decrements
    /// the counter, then ends with an `If` whose accept block is a lone
    /// `Break`.
    fn starts_with_check(body: &Block) -> bool {
        let Some(&Statement::Block(ref check)) = body.iter().next() else {
            return false;
        };
        matches!(
            check[..],
            [
                Statement::Emit(_),
                Statement::Store { .. },
                Statement::Emit(_),
                Statement::If { ref accept, ref reject, .. },
            ] if matches!(accept[..], [Statement::Break]) && reject.is_empty()
        )
    }

    /// For each loop in `block`, in order, whether the statement before it
    /// stores to a counter.
    fn resets(function: &Function, block: &Block, out: &mut alloc::vec::Vec<bool>) {
        let mut previous = None;
        for statement in block.iter() {
            match *statement {
                Statement::Loop {
                    ref body,
                    ref continuing,
                    ..
                } => {
                    let reset = match previous {
                        Some(&Statement::Store { pointer, .. }) => matches!(
                            function.expressions[pointer],
                            Expression::LocalVariable(local)
                                if function.local_variables[local].name.as_deref()
                                    == Some("loop_bound")
                        ),
                        _ => false,
                    };
                    out.push(reset);
                    resets(function, body, out);
                    resets(function, continuing, out);
                }
                Statement::Block(ref inner) => resets(function, inner, out),
                Statement::If {
                    ref accept,
                    ref reject,
                    ..
                } => {
                    resets(function, accept, out);
                    resets(function, reject, out);
                }
                Statement::Switch { ref cases, .. } => {
                    for case in cases {
                        resets(function, &case.body, out);
                    }
                }
                _ => {}
            }
            previous = Some(statement);
        }
    }

    fn loop_bodies(block: &Block, out: &mut alloc::vec::Vec<Block>) {
        for statement in block.iter() {
            match *statement {
                Statement::Loop {
                    ref body,
                    ref continuing,
                    ..
                } => {
                    out.push(body.clone());
                    loop_bodies(body, out);
                    loop_bodies(continuing, out);
                }
                Statement::Block(ref inner) => loop_bodies(inner, out),
                Statement::If {
                    ref accept,
                    ref reject,
                    ..
                } => {
                    loop_bodies(accept, out);
                    loop_bodies(reject, out);
                }
                Statement::Switch { ref cases, .. } => {
                    for case in cases {
                        loop_bodies(&case.body, out);
                    }
                }
                _ => {}
            }
        }
    }

    #[test]
    fn module_without_loops_is_not_copied() {
        assert!(bound("@compute @workgroup_size(1) fn main() {}").is_none());
    }

    #[test]
    fn every_loop_gets_its_own_counter() {
        let module = bound(
            "
            fn helper(n: i32) -> i32 {
                var total = 0;
                for (var i = 0; i < n; i++) {
                    total += i;
                }
                return total;
            }

            @compute @workgroup_size(1)
            fn main() {
                var x = 0;
                for (var i = 0; i < 4; i++) {
                    while x < 10 {
                        x += helper(i);
                    }
                }
                loop {
                    x -= 1;
                    continuing {
                        break if x < 0;
                    }
                }
            }
            ",
        )
        .unwrap();

        let helper = &module.functions.iter().next().unwrap().1;
        assert_eq!(counters(helper), 1);
        let main = &module.entry_points[0].function;
        assert_eq!(counters(main), 3);

        let mut bodies = alloc::vec::Vec::new();
        loop_bodies(&helper.body, &mut bodies);
        loop_bodies(&main.body, &mut bodies);
        assert_eq!(bodies.len(), 4);
        assert!(bodies.iter().all(starts_with_check));
    }

    /// Every counter starts at `u32::MAX`, from its initializer. Only a loop
    /// inside another loop, which may start more than once, stores that
    /// value again before it starts.
    #[test]
    fn only_nested_loops_reset_their_counters() {
        let module = bound(
            "
            @compute @workgroup_size(1)
            fn main() {
                var x = 0;
                for (var i = 0; i < 4; i++) {
                    if x > 2 {
                        while x < 10 {
                            x += i;
                        }
                    }
                    loop {
                        x -= 1;
                        continuing {
                            break if x < 0;
                        }
                    }
                }
                while x < 20 {
                    x += 3;
                }
            }
            ",
        )
        .unwrap();

        let main = &module.entry_points[0].function;
        assert_eq!(counters(main), 4);
        for (_, local) in main.local_variables.iter() {
            if local.name.as_deref() != Some("loop_bound") {
                continue;
            }
            let init = local.init.unwrap();
            let Expression::Splat { value, .. } = main.expressions[init] else {
                panic!("counter initializer is not a splat");
            };
            assert!(matches!(
                main.expressions[value],
                Expression::Literal(crate::Literal::U32(u32::MAX))
            ));
        }

        let mut out = alloc::vec::Vec::new();
        resets(main, &main.body, &mut out);
        assert_eq!(out, [false, true, true, false]);
    }

    /// The counter's `break` must not make a barrier later in the loop body
    /// look non-uniform to the validator.
    #[test]
    fn barrier_in_bounded_loop_stays_uniform() {
        bound(
            "
            var<workgroup> w: u32;

            @compute @workgroup_size(64)
            fn main() {
                for (var i = 0u; i < 4u; i++) {
                    workgroupBarrier();
                    w += i;
                }
            }
            ",
        )
        .unwrap();
    }
}
