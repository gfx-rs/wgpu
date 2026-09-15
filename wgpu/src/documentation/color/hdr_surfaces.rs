/*!
# HDR Surfaces

A surface can present in different color spaces to get HDR or wide-gamut
output onto the screen. This is configured through
[`SurfaceConfiguration::color_space`], and the rest of this section is a
concept primer on how it works.

### HDR output in a nutshell

By default a surface is standard dynamic range (SDR) with the sRGB gamut: the
value `1.0` is the brightest white, anything above it clips, and only colors
within sRGB are expressible. Other [`SurfaceColorSpace`]s opt into a **wider
gamut** (for example [`DisplayP3`](SurfaceColorSpace::DisplayP3), still SDR
but with more saturated colors than sRGB), **high dynamic range** (values
above `1.0` drive brighter-than-white output), or both (for example
[`Bt2100Pq`](SurfaceColorSpace::Bt2100Pq), aka HDR10), on platforms that
support it.

Three ideas carry most of the weight:

* **Reference white and headroom.** Brightness is measured in *nits*
  (cd/m²). On HDR-capable monitors, SDR reference white (plain white, `(1.0, 1.0, 1.0)` in the
  extended color spaces) sits *below* the display's peak output on purpose, so
  highlights have room above it. That gap is the display's *headroom*.
* **The transfer function is a round-trip.** Your shader applies an
  encoding transfer function (the OETF) to turn the light it computed into a
  stored signal, and the display applies the inverse decoding transfer
  function (the EOTF) to turn it back into light. Choosing a color space
  chooses which transfer functions both ends use.
* **Who applies the encoding transfer function.** wgpu applies it for you
  **only** when you render to an `*Srgb` texture view format, where the GPU
  runs the sRGB OETF when a value is stored to a texture. For every other
  color space (linear extended sRGB, encoded extended sRGB or P3, PQ, HLG)
  **the values your shader writes to the surface texture must already be
  encoded by you**, along with any gamut conversion; in a typical renderer
  this happens in a final tone-mapping or post-processing pass. wgpu hands
  the signal to the compositor unchanged; getting this wrong produces a
  wrong image with no error.

wgpu does **not** tonemap or gamut-map for you. It gives you the surface
and, through [`DisplayHdrInfo`], the display's advisory capabilities;
*choosing* and *applying* a tone curve is your application's job.

### The practical path

1. **Query capabilities.** Call [`Surface::get_capabilities`]. To use HDR or
   wide-gamut output, read [`SurfaceCapabilities::format_capabilities`] (each
   format and the [`SurfaceColorSpaces`] it supports), **not**
   [`SurfaceCapabilities::formats`]: the latter lists only formats usable
   with [`Auto`](SurfaceColorSpace::Auto), which never selects HDR.
   [`SurfaceCapabilities::color_spaces`] is a convenience lookup for one
   format.
2. **Optionally query the display.** Call [`Surface::display_hdr_info`] for the
   current [`DisplayHdrInfo`] (peak and SDR-white nits, EDR headroom,
   primaries, and a coarse dynamic-range/gamut bucket). Use it to pick a
   tone-map target ([`DisplayHdrInfo::tone_map_headroom`]); *whether* HDR is
   worthwhile is the capability question from step 1, not this live value.
   Every field is advisory and optional; `None` means "cannot tell here",
   never "SDR".
3. **Choose a format and color space.** Intersect what you want with what
   step 1 advertises, in your own preference order (for example HDR10, then
   linear extended sRGB, then encoded extended sRGB, then SDR
   [`Srgb`](SurfaceColorSpace::Srgb)). Keep an SDR fallback for when nothing
   HDR is advertised, such as when OS HDR is off.
4. **Configure the surface.** Set [`SurfaceConfiguration::color_space`] and
   `format`. [`Auto`](SurfaceColorSpace::Auto) (the default) reproduces
   wgpu's historical behavior and never picks HDR; any other value must be in
   that format's advertised set or configuration fails validation.
5. **Encode what you write to the surface texture.** For an `*Srgb` format,
   output linear and the hardware encodes for you. Otherwise the values your
   shader writes to the surface texture must already carry the encoding the
   chosen color space expects (sRGB, extended sRGB, PQ, or HLG) **and** any
   gamut conversion (for example outputting in the BT.2020 gamut for HDR10);
   in a typical renderer you do this in a final tone-mapping or
   post-processing pass. See the table below.
6. **Present** as usual. If OS HDR is toggled mid-run, you'll see it on the
   next [`Surface::display_hdr_info`] poll; re-query [`Surface::get_capabilities`]
   and re-run the steps above.

The standalone [HDR surface example] implements every step, including the
encoding transfer function for each color space.

### What to output from your fragment shader

What the values your shader writes to the surface texture must contain for
each color space, and whether wgpu applies the transfer function for you:

| Color space | Typical format | You write | wgpu encodes? |
| ----------- | -------------- | --------- | ------------- |
| `Srgb`, `*Srgb` format | `{Rgba,Bgra}8UnormSrgb` | linear | **yes** (hardware sRGB OETF on store) |
| `Srgb`, non-srgb format | `{Rgba,Bgra}8Unorm` | sRGB-encoded | no; apply the sRGB OETF yourself or use `*Srgb` instead |
| `ExtendedSrgbLinear` (scRGB) | `Rgba16Float` | linear, `1.0` = SDR white | no, but no encoding is necessary |
| `ExtendedSrgb` | `Rgba16Float` | extended sRGB-encoded | no; apply the extended sRGB OETF yourself |
| `DisplayP3` | `Bgra8Unorm` | sRGB-encoded, P3 primaries | no; apply the sRGB OETF (after gamut-mapping to P3) |
| `ExtendedDisplayP3` | `Rgba16Float` | extended sRGB-encoded, P3 primaries | no; apply the extended sRGB OETF (after gamut-mapping to P3) |
| `Bt2100Pq` (HDR10) | `Rgb10a2Unorm` | PQ-encoded, BT.2020 primaries | no; apply the PQ OETF (after gamut-mapping to BT.2020) |
| `Bt2100Hlg` | `Rgb10a2Unorm` | HLG-encoded, BT.2020 primaries | no; apply the HLG OETF (after gamut-mapping to BT.2020) |

In short, wgpu applies the transfer function for you only when you render to
an `*Srgb` format. In every other case the values your shader writes to the
surface texture must already carry both the transfer function and any gamut
conversion. The [HDR surface example] implements every encoder in WGSL.

### Glossary

* **Chromaticity** --- a color's hue and saturation independent of its
  brightness, given as an `(x, y)` coordinate on the CIE 1931 diagram.
* **Primaries / gamut** --- the chromaticities of the red, green, and blue a
  color space addresses, and so the range of colors it can express. [BT.709]
  is the sRGB gamut, [Display P3] is wider, and [BT.2020] is wider still.
* **White point** --- the chromaticity of `R = G = B` (what "white" looks
  like). Every color space here uses [D65], standard daylight.
* **Transfer function (OETF / EOTF)** --- how stored values map to light. The
  *OETF* is the encoding transfer function your application applies; the
  *EOTF* is the inverse decoding transfer function the display applies.
* **SDR / HDR** --- standard dynamic range clips at `1.0` (reference white);
  high dynamic range lets values above `1.0` drive brighter-than-white
  output.
* **Nits and reference white** --- a *nit* (cd/m²) is a unit of brightness;
  *reference white* (also called *paper white*, especially on Windows) is the
  brightness of SDR plain white, set below the panel's peak so highlights have
  room above it.
* **Headroom (EDR)** --- how much brighter than current SDR white the display
  can go right now, as a multiplier (`1.0` means none). Dynamic; see
  [`DisplayHdrInfo::tone_map_headroom`].
* **PQ / HLG** --- the two HDR transfer functions: [PQ] (SMPTE ST 2084,
  HDR10) encodes absolute luminance, [HLG] (BT.2100) encodes relative
  luminance.
* **scRGB / extended-range sRGB** --- sRGB extended past 0.0..=1.0 for
  HDR: [scRGB] is *linear*
  ([`ExtendedSrgbLinear`](SurfaceColorSpace::ExtendedSrgbLinear)), while
  [`ExtendedSrgb`](SurfaceColorSpace::ExtendedSrgb) is the same range but
  sRGB-*encoded* (gamma), the web's HDR path.

[HDR surface example]: https://github.com/gfx-rs/wgpu/tree/v29/examples/standalone/03_hdr_surface
[BT.709]: https://www.itu.int/rec/R-REC-BT.709
[BT.2020]: https://www.itu.int/rec/R-REC-BT.2020
[Display P3]: https://en.wikipedia.org/wiki/DCI-P3#Display_P3
[D65]: https://en.wikipedia.org/wiki/Standard_illuminant#D65_values
[PQ]: https://en.wikipedia.org/wiki/Perceptual_quantizer
[HLG]: https://www.itu.int/rec/R-REC-BT.2100
[scRGB]: https://en.wikipedia.org/wiki/ScRGB
*/

use crate::{
    DisplayHdrInfo, Surface, SurfaceCapabilities, SurfaceColorSpace, SurfaceColorSpaces,
    SurfaceConfiguration,
};
