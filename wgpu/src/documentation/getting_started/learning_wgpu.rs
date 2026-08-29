/*!
# Learning wgpu: Key Concepts and Resources

If you are new to `wgpu` and graphics programming, this page collects
external learning resources and a primer on the core concepts of the API.

## External resources

Learning materials:

- [Learn Wgpu tutorial](https://sotrh.github.io/learn-wgpu/)
- [Draw You a Triangle for Great Good](https://github.com/dwbrite/wgpu-rendering-project/wiki/Draw-You-a-Triangle-for-Great-Good)
- Chinese version of [学习 wgpu](https://jinleili.github.io/learn-wgpu-zh/)
- [GPU Architecture](https://docs.google.com/presentation/d/1qi2j-SZuzew7Rrf5VKEPDZAQQEitV40k9fKvwJNyicM/edit),
  a presentation by Kangz on how GPUs work under the hood.

Reference material:

- The `wgpu` examples; see
  [Running the Examples](crate::documentation::getting_started::running_the_examples).
- The [WebGPU specification](https://www.w3.org/TR/webgpu/), which `wgpu`
  follows closely.

Runtimes:

- The [Deno](https://deno.land/) JS/TS runtime.

## Important concepts

### Device and Queue

- [`Device`]
  - All resource creation.
  - All of its methods take `&self` (no `&mut`!).
- [`Queue`]
  - All GPU work submission.

_Typically_ both are created once on startup, one of each. A [`Device`] is
created with a set of [`Features`] and [`Limits`], which are enforced
independently of the underlying hardware's actual capabilities (e.g. texture
size and formats, number of bound resources, native extensions, …).

### Relationship of [`RenderPipeline`]s and resource binding

A slightly simplified overview of what you need to set up before rendering a
frame.

*/
#![doc = include_str!("../images/render-pipeline-and-resource-binding.svg")]
/*!

- _Round boxes:_ temporary descriptor structs.
- _Cornered boxes:_ resources created from a [`Device`].
- _Bold:_ what you deal with on a per-frame basis.

Each [`BindGroupLayoutEntry`] has a (mostly) corresponding
[`BindingResource`]. Compute pipelines follow a similar pattern.

### Drawing a "frame"

A simplified overview of what you need to do to draw a frame.

*/
#![doc = include_str!("../images/life-of-a-frame.svg")]
/*!

- _Full lines:_ needed for creation.
- _Dashed lines:_ provided as "information".

[`CommandEncoder::finish`] consumes a [`CommandEncoder`] and produces a
[`CommandBuffer`]; [`Queue::submit`] consumes [`CommandBuffer`]s. The
[`TextureView`] on a [`RenderPassColorAttachment`] can come from a
[`Surface`] — a special target for the final output.

*/

use crate::{
    BindGroupLayoutEntry, BindingResource, CommandBuffer, CommandEncoder, Device, Features, Limits,
    Queue, RenderPassColorAttachment, RenderPipeline, Surface, TextureView,
};
