/*!
# Multiview

`wgpu` supports multiview rendering when [`Features::MULTIVIEW`] is enabled. Multiview lets a single draw call
render to several layers of an array texture, with shaders reading `@builtin(view_index)` to tell the views apart.
Rendering to layers other than the first n requires [`Features::SELECTIVE_MULTIVIEW`], and multiview in mesh
pipelines requires [`Features::EXPERIMENTAL_MESH_SHADER_MULTIVIEW`].

***This is not a core part of WebGPU, but is a wgpu-specific extension.***

## Feature overview

Multiview lets each draw call draw the same contents to multiple layers of a single texture. Your pipeline
must declare a multiview mask, which is a bitmask of which slices of an array texture will be rendered to.

In practice, normally you want to perform separate object culling for each view, unless the contents of
the views are going to be very similar. For this reason, outside of VR, multiview sees limited use.

## wgpu API

wgpu's API includes 3 new features, the latter two of which are explained more below.
* [`Features::MULTIVIEW`]
* [`Features::SELECTIVE_MULTIVIEW`]
* [`Features::EXPERIMENTAL_MESH_SHADER_MULTIVIEW`]

It also includes the additional limits
* [`Limits::max_multiview_view_count`]
* [`Limits::max_mesh_multiview_view_count`]

These are fairly self explanatory.

Render & mesh pipeline descriptors, renderpass descriptors and render bundle descriptors include new `multiview_mask` fields.
These are used to determine which layers of an array texture are rendered to. They must match between a renderpass
and all of its used render pipelines.

WGSL also has a new `@builtin(view_index)` `u32` built-in. It doesn't currently have an enable extension. It
is the index of the layer in the array texture as a whole, not just the index within the layers in the bitmask.

## Use cases

### VR

VR is the primary use case for multiview. You have two displays which will display largely the same contents, so
you draw the same contents to each.

### Foveated rendering in VR

DX12's limit for texture layers with multiview is 4, which may seem odd. Foveated rendering is a usecase for this exact number of layers.

In principle VR requires rendering at extremely consistent framerates to multiple high-resolution displays.
However, some developers realized that they could make use of information about where the wearer is looking with their eyes.
You can render those areas with higher resolution, and the user won't notice the background being slightly worse quality.

Quad-view rendering is a simple (but effective) implementation of this principle. Each display gets rendered to 2 layers.
The first layer is the focused region, maybe half of the screen in each dimension (though this can differ) centered around the wearer's focus.
The second layer renders everything else, maybe with a stencil buffer to exclude the area rendered in higher detail.
Then they are both copied to a higher resolution final texture, with the background upsampled.

Quad-view rendering is effective for performance and is tunable, although it requires some information from the headset. Apple headsets
consider knowing the user's focus to actually be a violation of privacy, so for foveated rendering they essentially require that you
let them handle some of the details.

### Cubemap rendering

In this case you may render each object to each of 6 cube faces. Note that this is discouraged, since most objects won't
be visible on more than 1 face, and this doesn't work with per-face culling. Because it is so rarely used and so ineffective,
DX12 actually doesn't expose a multiview layer limit which a driver could claim to be above 4. So this is unusable on DX12
and some metal devices.

## Note on layered rendering

Layered rendering is very similar, except that the vertex shader determines which layer of a texture array to render to.
It is therefore strictly more powerful than multiview, but this power is unnecessary for most uses. wgpu currently only
supports multiview, though layered rendering support may be added in the future.

## Selective multiview

Selective multiview lets you render to a bitmask of layers which is not just the first n layers of an array texture.
For example, if you have a cubemap texture, you could group objects into draw calls based on which cube faces they render
to, and then make draw calls for each of those groups without duplicating the effort to cube faces which won't see the object.

In practice, pipelines have to be compiled with a specific multiview mask, so this would require many duplicates of each pipeline.
It is not expected to see much use.

## Note on instances

Some drivers polyfill multiview by using multiple instances per vertex shader, and then using layered rendering.

This means that using multiview reduces your maximum instance index, though wgpu doesn't currently expose this as a limit.

Vulkan only exposes a limit for the maximum instance index per device without taking into account the actual number of used multiview layers,
so if a device supports multiview with 32 layers, it may only support (2^32-1) / 32 for the maximum instance index of any multiview draw call.

In most of these cases, on the vulkan backend, wgpu will catch this before passing it to the driver and panic. There are exceptions for e.g. indirect
draws, which don't have this validation at all.

This is tracked as an issue at [#8333](https://github.com/gfx-rs/wgpu/issues/8333).

This also means that on these drivers, multiview is incompatible with mesh shaders. For this reason, using mesh shaders
with multiview requires the [`Features::EXPERIMENTAL_MESH_SHADER_MULTIVIEW`] feature.

Some other drivers, including llvmpipe, simply haven't added support for multiview with mesh shaders, despite not using this method of polyfill.

 */

use crate::{Features, Limits};
