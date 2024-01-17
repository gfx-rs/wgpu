# hello_workgroups

Now you finally know what that silly little `@workgroup_size(1)` means!

This example is an extremely bare-bones and arguably somewhat unreasonable demonstration of what workgroup sizes mean in an attempt to explain workgroups in general.

The example starts with two arrays of numbers. One where `a[i] = i` and the other where `b[i] = 2i`. Both are bound to the shader. The program dispatches a workgroup for each index, each workgroup representing both elements at that index in both arrays. Each invocation in each workgroup works on its respective array and adds 1 to the element there.

## To Run

```
cargo run --bin wgpu-examples hello_workgroups
```

## What are Workgroups?

### TLDR / Key Takeaways

- Workgroups fit in a 3d grid of workgroups executed in a single dispatch.
- All invocations in a workgroup are guaranteed to execute concurrently.
- Workgroups carry no other guarantees for concurrency outside of those individual workgroups, meaning...
  - No two workgroups can be guaranteed to be executed in parallel.
  - No two workgroups can be guaranteed NOT to be executed in parallel.
  - No set of workgroups can be guaranteed to execute in any predictable or reliable order in relation to each other.
- Ths size of a workgroup is defined with the `@workgroup_size` attribute on a compute shader main function.
- The location of an invocation within its workgroup grid can be got with `@builtin(local_invocation_id)`.
- The location of an invocation within the entire compute shader grid can be gotten with `@builtin(global_invocation_id)`.
- The location of an invocation's workgroup within the dispatch grid can be gotten with `@builtin(workgroup_id)`.
- Workgroups share memory within the `workgroup` address space. Workgroup memory is similar to private memory but it is shared within a workgroup. Invocations within a workgroup will see the same memory but invocations across workgroups will be accessing different memory.

### Introduction

When you call `ComputePass::dispatch_workgroups`, the function dispatches multiple workgroups in a 3d grid defined by the `x`, `y`, and `z` parameters you pass to the function. For example, `dispatch_workgroups(5, 2, 1)` would create a dispatch grid like
||||||
|---|---|---|---|---|
| W | W | W | W | W |
| W | W | W | W | W |

Where each W is a workgroup. If you want your shader to consider what workgroup within the dispatch the current invocation is in, add a function argument with type `vec3<u32>` and with the attribute `@builtin(workgroup_id)`.

Note here that in this example, the term "dispatch grid" is used throughout to mean the grid of workgroups within the dispatch but is not a proper term within WGSL. Other terms to know though that are proper are "workgroup grid" which refers to the invocations in a single _workgroup_ and "compute shader grid" which refers to the grid of _all_ the invocations in the _entire dispatch_.

### Within the Workgroup

Although with hello-compute and repeated-compute, we used a workgroup size of `(1)`, or rather, (1, 1, 1), and then each workgroup called from `dispatch_workgroups` made _an_ invocation, this isn't always the case. Each workgroup represents its own little grid of individual invocations tied together. This could be just one or practically any number in a 3d grid of invocations. The grid size of each workgroup and thus the number of invocations called per workgroup is determined by the `@workgroup_size` attribute you've seen in other compute shaders. To get the current invocation's location within a workgroup, add a `vec3<u32>` argument to the main function with the attribute `@builtin(local_invocation_id)`. We'll look at the compute shader grid of a dispatch of size (2, 2, 1) with workgroup sizes of (2, 2, 1) as well. Let `w` be the `workgroup_id` and `i` be the `local_invocation_id`.

||||| 
|------------------------|------------------------|------------------------|------------------------|
| w(0, 0, 0), i(0, 0, 0) | w(0, 0, 0), i(1, 0, 0) | w(1, 0, 0), i(0, 0, 0) | w(1, 0, 0), i(1, 0, 0) |
| w(0, 0, 0), i(0, 1, 0) | w(0, 0, 0), i(1, 1, 0) | w(1, 0, 0), i(0, 1, 0) | w(1, 0, 0), i(1, 1, 0) |
| w(0, 1, 0), i(0, 0, 0) | w(0, 1, 0), i(1, 0, 0) | w(1, 1, 0), i(0, 0, 0) | w(1, 1, 0), i(1, 0, 0) |
| w(0, 1, 0), i(0, 1, 0) | w(0, 1, 0), i(1, 1, 0) | w(1, 1, 0), i(0, 1, 0) | w(1, 1, 0), i(1, 1, 0) |

### Execution of Workgroups

As stated before, workgroups are groups of invocations. The invocations within a workgroup are always guaranteed to execute in parallel. That said, the guarantees basically stop there. You cannot get any guarantee as to when any given workgroup will execute, including in relation to other workgroups. You can't guarantee that any two workgroups will execute together nor can you guarantee that they will _not_ execute together. Of the workgroups that don't execute together, you additionally cannot guarantee that they will execute in any particular order. When your function runs in an invocation, you know that it will be working together with its workgroup buddies and that's basically it.

See [the WGSL spec on compute shader execution](https://www.w3.org/TR/2023/WD-WGSL-20230629/#compute-shader-workgroups) for more details.

### Workgroups and their Invocations in a Global Scope

As mentioned above, invocations exist both within the context of a workgroup grid as well as a compute shader grid which is a grid, divided into workgroup sections, of invocations that represents the whole of the dispatch. Similar to how `@builtin(local_invocation_id)` gets you the place of the invocation within the workgroup grid, `@builtin(global_invocation_id)` gets you the place of the invocation within the entire compute shader grid. Slight trivia: you might imagine that this is computed from `local_invocation_id` and `workgroup_id` but it's actually the opposite. Everything operates on the compute shader grid, the workgroups are imagined sectors within the compute shader grid, and `local_invocation_id` and `workgroup_id` are calculated based on global id and known workgroup size. Yes, we live in a matrix... of compute shader invocations. This isn't super useful information but it can help fit things into a larger picture.

## Barriers and Workgroups

Arguably, workgroups are at their most useful when being used alongside barriers. Since barriers are already explained more thoroughly in the hello-synchronization example, this section will be short. Despite affecting different memory address spaces, all synchronization functions affect invocations on a workgroup level, synchronizing the workgroup. See [hello-synchronization/README.md](../hello-synchronization/README.md) for more.

## Links to Technical Resources

For a rather long explainer, this README may still leave the more technically minded person with questions. The specifications for both WebGPU and WGSL ("WebGPU Shading Language") are long and it's rather unintuitive that by far the vast majority of specification on how workgroups and compute shaders more generally work, is all in the WGSL spec. Below are some links into the specifications at a couple interesting points:

- [Here](https://www.w3.org/TR/WGSL/#compute-shader-workgroups) is the main section on workgroups and outlines important terminology in technical terms. It is recommended that everyone looking for something in this section of this README start by reading this.
- [Here](https://www.w3.org/TR/webgpu/#computing-operations) is a section on compute shaders from a WebGPU perspective (instead of WGSL). It's still a stub but hopefully it will grow in the future.
- Don't forget your [`@builtin()`'s](https://www.w3.org/TR/WGSL/#builtin-inputs-outputs)!