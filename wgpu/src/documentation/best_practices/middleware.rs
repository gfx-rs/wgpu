/*!
# Encapsulating Graphics Work: the Middleware Pattern

Whether you are designing a library to be used by others or modularizing
your own code, it is important to be able to encapsulate graphics code in a
way that offers the calling code as much flexibility as possible.

The following pattern is called the "middleware pattern" and works well both
in separate libraries and in regular modules.

## Middleware libraries

Middleware is a piece of software that fits unobtrusively into an existing
application, giving it some extra functionality. In the case of `wgpu`,
middleware are libraries that use the `wgpu` context the user provides to do
their work. If a library creates the `wgpu` adapter, device, etc. for you,
it isn't middleware — it would more likely be called a framework.

## API design

This does not have to be the extent of the API; you may have more (or
different) arguments or more functions, but this is the gist of the
interactions with `wgpu`.

```ignore
impl MiddlewareRenderer {
    /// Create all unchanging resources here.
    pub fn new(device: &Device, format: &TextureFormat, ..) -> Self;

    /// Prepare for rendering this frame; create all resources that will be
    /// used during the next render that do not already exist.
    pub fn prepare(&mut self, ..);

    /// Render using a caller-provided render pass.
    pub fn render(&self, render_pass: &mut RenderPass<'_>);
}
```

The goal of this API is to use as few render passes and submissions as
possible.

- On GPUs that use _tiled rendering_, there is significant cost to ending a
  render pass. Therefore, the middleware should accept an existing
  [`RenderPass`] (presuming it is rendering to a surface/texture provided by
  the user).

- [`Queue::submit`] is expensive for `wgpu` to execute. Therefore, if the
  middleware generates a [`CommandBuffer`] when preparing, it should hand
  that buffer back to the caller to become part of a larger submission,
  instead of submitting it alone.

## Functions

### New

```ignore
fn new(device: &Device, format: &TextureFormat, ..) -> Self;
```

This is where you create your renderer and set up all the static resources.
Things like pipelines, buffers, or textures should be created and uploaded
here. When the middleware needs to know the parameters of what it is
rendering to, favor accepting a [`TextureFormat`], `width`, and `height`
over a [`SurfaceConfiguration`], as the user may not be rendering to the
surface but to another texture.

### Prepare

```ignore
fn prepare(&mut self, ..);
```

Ideally there should be a minimal amount of resources created per frame, but
that is often hard to avoid. `prepare()` should create those resources and
do any other computation required to be ready to render.

### Render

```ignore
fn render(&self, render_pass: &mut RenderPass<'_>);
```

This is where the magic happens! Using the resources created during `new()`
and `prepare()`, render everything using the provided render pass.

The split between `prepare()` and `render()` is not critical but improves
flexibility. By avoiding borrowing the middleware object exclusively
(`&mut`) during `render()`, the user has more options for organizing their
code; for example, they might want to perform command encoding — including
the middleware's `render()` — in parallel into multiple command buffers, and
this may be easier if the middleware (and therefore whatever owns it) does
not have to be exclusively borrowed. It also means that the order of calls
to `prepare()` of different middleware can be independent of the order of
drawing.

## Multiple render targets

If your piece of middleware has to render to multiple targets, it is pretty
unavoidable to have multiple render passes. As much as possible, this
pattern should be used as a guideline for the design of your API, but it
doesn't work for every possible piece of middleware out there.
*/

use crate::{CommandBuffer, Queue, RenderPass, SurfaceConfiguration, TextureFormat};
