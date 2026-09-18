/*!
# Texture Color Formats and sRGB Conversions

Tips and common misconceptions about color formats. For surface output,
wide-gamut, and high-dynamic-range color, see
[HDR and Wide-Gamut Color Spaces](crate::documentation::color::hdr_surfaces).

_(The following is adapted from conversations on the `wgpu` users Matrix
channel, saved here as a useful resource for future `wgpu` developers.)_

Digital color can be an extremely confusing topic, and the resources in
common circulation aren't always great. Modern low-level graphics APIs (such
as those under the hood of `wgpu`) give us powerful tools to manipulate color
data, though this comes at the cost of having to understand several low-level
principles in how digital color works.

The best place to start is to have an understanding of the problems and
history of gamma correction. This page won't go deeply into the topic, but
some useful links can be found here:

- [What every coder should know about gamma](https://blog.johnnovak.net/2016/09/21/what-every-coder-should-know-about-gamma/)
- [Hitchhiker's Guide to Digital Color](https://hg2dc.com/)
- [Learn OpenGL — Gamma Correction](https://learnopengl.com/Advanced-Lighting/Gamma-Correction)

## What do `TextureFormat`s do?

There are many different texture formats you can specify for the format of
your color data. This can be image data you upload from the CPU, shaded
geometry rendered to intermediate textures (what we used to call frame
buffers in OpenGL), compute shader targets, or geometry shaded to the window
surface itself.

We won't go into all the different formats here, but it's useful to talk
about two of the most commonly used formats and some misconceptions about
what they do (and don't) do.

## What are Unorm and UnormSrgb?

Some of the most common formats you will see used for texture data, render
targets, etc. are [`Rgba8Unorm`](TextureFormat::Rgba8Unorm) and
[`Rgba8UnormSrgb`](TextureFormat::Rgba8UnormSrgb). So what are they, and how
do they relate to the issues of gamma correction, perceptually-linear "sRGB"
space, and physically-accurate "linear" space (as described in detail in the
links above)?

First of all, it will be useful to clarify some terminology, as the words
"linear" and "sRGB" can be overloaded and lead to confusion:

- **sRGB space** is a set of color primaries — this defines what red, green,
  and blue are.

In this sense (and perhaps counter-intuitively), texture formats ending in
"Unorm" and "UnormSrgb" are BOTH in the sRGB color space, and encode their
data in the same way without losing precision (as might be implied by some
descriptions of "linear" vs. "sRGB"); more on this later.

Here is some useful terminology to help avoid "linear" and "sRGB" confusion:

- What we traditionally called "linear space" is more accurately called
  "scene-referred color".
- What we traditionally called "sRGB space" is more accurately called
  "monitor-referred color".
- The conversion from scene-referred color to monitor-referred color is done
  by the sRGB OETF (opto-electrical transfer function \[light → bits\]).
- The conversion from monitor-referred color to scene-referred color is done
  by the sRGB EOTF (electro-optical transfer function \[bits → light\]).

Let's walk our way up the color pipeline. The OS assumes that the bits you
are sending to the screen are monitor-referred and in the sRGB space (unless
otherwise specified). It will interpret the bits this way no matter what.

- If you write/shade to a texture which is in a **Unorm** format, it will
  take the output of your shader and do a float → int conversion, and that's
  it (Unorm floats are written as ints under the hood, though you don't need
  to worry about this).

  This means that if you're using this **Unorm** texture as a surface, *the
  floats you are writing from your shader need to be **monitor-referred*** (i.e.
  numbers that the OS can safely assume are already in sRGB space).

- If you write/shade to a texture which is in a **UnormSrgb** format, it will
  take the output of your shader and then apply the sRGB OETF to it, then
  afterwards do the float → int conversion to store the data.

  This means that if you're using this **UnormSrgb** texture as a surface,
  *the floats you are writing from your shader need to be **scene-referred***,
  as the GPU will do the conversion to monitor-referred as part of the write!

The inverse happens when you read from a texture:

- a **Unorm** texture just does int → float on read.
- a **UnormSrgb** texture applies the sRGB EOTF to do monitor → scene and
  then does the int → float read.

## Common misconceptions

**Q: So Unorm textures are in "linear" space and UnormSrgb textures are in
"sRGB" space?**

**A:** No. Despite the naming, you are responsible for whether the outputs of
your shader represent "scene"- or "monitor"-referred values. The "Srgb" tag on
the texture format just lets the shader know that you want it to do a
conversion from sRGB to linear (or, more specifically, "monitor" to "scene")
when you sample it, or the inverse when you write to it.

## Tips

- Think of texture formats as a combination of data resolution and data
  conversion functions. For example:
  [`Rgba8UnormSrgb`](TextureFormat::Rgba8UnormSrgb) = `Rgba8` data +
  `UnormSrgb` conversion step.
- Let the GPU do the conversion work for you. Historically in WebGL, for
  example, if you wanted to do physically-accurate lighting you would need to
  convert from linear to sRGB manually (often with an approximation like
  `pow(col, 2.2)`). By being aware of the textures you are sampling and
  writing to, you should be able to avoid manual conversions.
*/

use crate::TextureFormat;
