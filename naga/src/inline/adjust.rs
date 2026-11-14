//! Rewriting the handles in expressions and statements copied from one
//! function into another. The matches mirror those in [`crate::compact`].

use crate::{Block, Expression, Handle, LocalVariable, Range, Statement};
use alloc::{vec, vec::Vec};

/// Apply `adjust` to every expression handle in `expr`, and `adjust_local` to
/// its local variable handle, if any.
pub fn adjust_expression(
    expr: &mut Expression,
    adjust: &mut dyn FnMut(&mut Handle<Expression>),
    adjust_local: &mut dyn FnMut(&mut Handle<LocalVariable>),
) {
    fn adjust_option(
        handle: &mut Option<Handle<Expression>>,
        adjust: &mut dyn FnMut(&mut Handle<Expression>),
    ) {
        if let Some(ref mut handle) = *handle {
            adjust(handle);
        }
    }
    use crate::Expression as Ex;
    match *expr {
        // Expressions that do not contain handles that need to be adjusted.
        Ex::Literal(_)
        | Ex::SubgroupBallotResult
        | Ex::RayQueryProceedResult
        | Ex::Constant(_)
        | Ex::Override(_)
        | Ex::ZeroValue(_)
        | Ex::GlobalVariable(_)
        | Ex::CallResult(_)
        | Ex::AtomicResult { .. }
        | Ex::WorkGroupUniformLoadResult { .. }
        | Ex::SubgroupOperationResult { .. }
        | Ex::FunctionArgument(_) => {}

        Ex::LocalVariable(ref mut local) => adjust_local(local),

        // Expressions that contain handles that need to be adjusted.
        Ex::Compose {
            ref mut components, ..
        } => {
            for component in components {
                adjust(component);
            }
        }
        Ex::Access {
            ref mut base,
            ref mut index,
        } => {
            adjust(base);
            adjust(index);
        }
        Ex::AccessIndex {
            ref mut base,
            index: _,
        } => adjust(base),
        Ex::Splat {
            size: _,
            ref mut value,
        } => adjust(value),
        Ex::Swizzle {
            size: _,
            ref mut vector,
            pattern: _,
        } => adjust(vector),
        Ex::Load { ref mut pointer } => adjust(pointer),
        Ex::ImageSample {
            ref mut image,
            ref mut sampler,
            gather: _,
            ref mut coordinate,
            ref mut array_index,
            ref mut offset,
            ref mut level,
            ref mut depth_ref,
            clamp_to_edge: _,
        } => {
            adjust(image);
            adjust(sampler);
            adjust(coordinate);
            adjust_option(array_index, adjust);
            adjust_option(offset, adjust);
            use crate::SampleLevel as Sl;
            match *level {
                Sl::Auto | Sl::Zero => {}
                Sl::Exact(ref mut expr) => adjust(expr),
                Sl::Bias(ref mut expr) => adjust(expr),
                Sl::Gradient {
                    ref mut x,
                    ref mut y,
                } => {
                    adjust(x);
                    adjust(y);
                }
            }
            adjust_option(depth_ref, adjust);
        }
        Ex::ImageLoad {
            ref mut image,
            ref mut coordinate,
            ref mut array_index,
            ref mut sample,
            ref mut level,
        } => {
            adjust(image);
            adjust(coordinate);
            adjust_option(array_index, adjust);
            adjust_option(sample, adjust);
            adjust_option(level, adjust);
        }
        Ex::ImageQuery {
            ref mut image,
            ref mut query,
        } => {
            adjust(image);
            use crate::ImageQuery as Iq;

            match *query {
                Iq::Size { ref mut level } => adjust_option(level, adjust),
                Iq::NumLevels | Iq::NumLayers | Iq::NumSamples => {}
            }
        }
        Ex::Unary {
            op: _,
            ref mut expr,
        } => adjust(expr),
        Ex::Binary {
            op: _,
            ref mut left,
            ref mut right,
        } => {
            adjust(left);
            adjust(right);
        }
        Ex::Select {
            ref mut condition,
            ref mut accept,
            ref mut reject,
        } => {
            adjust(condition);
            adjust(accept);
            adjust(reject);
        }
        Ex::Derivative {
            axis: _,
            ctrl: _,
            ref mut expr,
        } => adjust(expr),
        Ex::Relational {
            fun: _,
            ref mut argument,
        } => adjust(argument),
        Ex::Math {
            fun: _,
            ref mut arg,
            ref mut arg1,
            ref mut arg2,
            ref mut arg3,
        } => {
            adjust(arg);
            adjust_option(arg1, adjust);
            adjust_option(arg2, adjust);
            adjust_option(arg3, adjust);
        }
        Ex::As {
            ref mut expr,
            target: _,
        } => adjust(expr),
        Ex::Bitcast {
            ref mut expr,
            ty: _,
        } => adjust(expr),
        Ex::ArrayLength(ref mut expr) => adjust(expr),
        Ex::RayQueryGetIntersection {
            ref mut query,
            committed: _,
        } => adjust(query),
        Ex::RayQueryVertexPositions {
            ref mut query,
            committed: _,
        } => adjust(query),
        Ex::CooperativeLoad { ref mut data, .. } => {
            adjust(&mut data.pointer);
            adjust(&mut data.stride);
        }
        Ex::CooperativeMultiplyAdd {
            ref mut a,
            ref mut b,
            ref mut c,
        } => {
            adjust(a);
            adjust(b);
            adjust(c);
        }
    }
}

/// Apply `adjust` to every expression handle in `block`, including the bounds
/// of `Emit` ranges.
pub fn adjust_block(block: &mut Block, adjust: &mut dyn FnMut(&mut Handle<Expression>)) {
    let statements: &mut [Statement] = block;
    let mut worklist: Vec<&mut [Statement]> = vec![statements];

    while let Some(last) = worklist.pop() {
        for st in last {
            use crate::Statement as St;
            match *st {
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
                    use crate::AtomicFunction as Af;
                    match *fun {
                        Af::Exchange {
                            compare: Some(ref mut expr),
                        } => {
                            adjust(expr);
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
                    adjust(value);
                    if let Some(ref mut result) = *result {
                        adjust(result);
                    }
                }
                St::ImageAtomic {
                    ref mut image,
                    ref mut coordinate,
                    ref mut array_index,
                    fun: _,
                    ref mut value,
                } => {
                    adjust(image);
                    adjust(coordinate);
                    if let Some(ref mut array_index) = *array_index {
                        adjust(array_index);
                    }
                    adjust(value);
                }
                St::WorkGroupUniformLoad {
                    ref mut pointer,
                    ref mut result,
                } => {
                    adjust(pointer);
                    adjust(result);
                }
                St::Call {
                    ref mut arguments,
                    ref mut result,
                    ..
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
                    use crate::RayQueryFunction as Qf;
                    match *fun {
                        Qf::Initialize {
                            ref mut acceleration_structure,
                            ref mut descriptor,
                        } => {
                            adjust(acceleration_structure);
                            adjust(descriptor);
                        }
                        Qf::Proceed { ref mut result } => {
                            adjust(result);
                        }
                        Qf::GenerateIntersection { ref mut hit_t } => {
                            adjust(hit_t);
                        }
                        Qf::ConfirmIntersection => {}
                        Qf::Terminate => {}
                        Qf::Begin => {}
                    }
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
                        | crate::GatherMode::ShuffleXor(ref mut index)
                        | crate::GatherMode::QuadBroadcast(ref mut index) => adjust(index),
                        crate::GatherMode::QuadSwap(_) => {}
                    }
                    adjust(argument);
                    adjust(result);
                }
                St::CooperativeStore {
                    ref mut target,
                    ref mut data,
                } => {
                    adjust(target);
                    adjust(&mut data.pointer);
                    adjust(&mut data.stride);
                }
                St::RayPipelineFunction(ref mut func) => match *func {
                    crate::RayPipelineFunction::TraceRay {
                        ref mut acceleration_structure,
                        ref mut descriptor,
                        ref mut payload,
                    } => {
                        adjust(acceleration_structure);
                        adjust(descriptor);
                        adjust(payload);
                    }
                },
                St::Emit(ref mut range) => {
                    if let Some((mut first, mut last)) = range.first_and_last() {
                        adjust(&mut first);
                        adjust(&mut last);
                        *range = Range::new_from_bounds(first, last);
                    }
                }
                St::DebugPrintf {
                    format: _,
                    ref mut arguments,
                } => {
                    for expr in arguments {
                        adjust(expr);
                    }
                }
                St::Break
                | St::Continue
                | St::Kill
                | St::ControlBarrier(_)
                | St::MemoryBarrier(_)
                | St::Return { value: None } => {}
            }
        }
    }
}
