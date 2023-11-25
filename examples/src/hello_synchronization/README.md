# hello_synchronization

This example is 
1. A small demonstration of the importance of synchronization.
2. How basic synchronization you can understand from the CPU is preformed on the GPU.

## To Run

```
cargo run --bin wgpu-examples hello_synchronization
```

## A Primer on WGSL Synchronization Functions

The official documentation is a little scattered and sparse. The meat of the subject is found [here](https://www.w3.org/TR/2023/WD-WGSL-20230629/#sync-builtin-functions) but there's also a bit on control barriers [here](https://www.w3.org/TR/2023/WD-WGSL-20230629/#control-barrier). The most important part comes from that first link though, where the spec says "the affected memory and atomic operations program-ordered before the synchronization function must be visible to all other threads in the workgroup before any affected memory or atomic operation program-ordered after the synchronization function is executed by a member of the workgroup." And at the second, we also get "a control barrier is executed by all invocations in the same workgroup as if it were executed concurrently."

That's rather vague (and it is by design) so let's break it down and make a comparison that should make that sentence come a bit more into focus. [Barriers in Rust](https://doc.rust-lang.org/std/sync/struct.Barrier.html#) fit both bills rather nicely. Firstly, Rust barriers are executed as if they were executed concurrently because they are - at least as long as you define the execution by when it finishes, when [`Barrier::wait`](https://doc.rust-lang.org/std/sync/struct.Barrier.html#method.wait) finally unblocks the thread and execution continues concurrently from there. Rust barriers also fit the first bill; because all affected threads must execute `Barrier::wait` in order for execution to continue, we can guarantee that _all (synchronous)_ operations ordered before the wait call are executed before any operations ordered after the wait call begin execution. Applying this to WGSL barriers, we can think of a barrier in WGSL as a checkpoint all invocations within each workgroup must reach before the entire workgroup continues with the program together.

There are two key differences though and one is that although Rust barriers don't enforce that atomic operations called before the barrier are visible after the barrier, WGSL barriers do. This is incredibly useful and important though and is demonstrated in this example.

Another is that WGSL's synchronous functions only affect memory and atomic operations in a certain address space. This applies to the whole 'all atomic operations called before the function are visible after the function' thing. There are currently three different synchronization functions:
- `storageBarrier` which works in the storage address space and is a simple barrier.
- `workgroupBarrier` which works in the workgroup address space and is a simple barrier.
- `workgroupUniformLoad` which also works in the workgroup address space and is more than just a barrier.
Read up on all three [here](https://www.w3.org/TR/2023/WD-WGSL-20230629/#sync-builtin-functions).