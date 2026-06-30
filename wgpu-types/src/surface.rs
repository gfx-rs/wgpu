//! Surface presentation configuration: present modes, alpha compositing, and
//! color-space types (HDR and wide-gamut output).
//!
//! This module is re-exported flatly from `wgpu-types`; the user-facing color
//! space and HDR primer lives in the `wgpu` crate's top-level docs.

use alloc::{vec, vec::Vec};

use crate::{link_to_wgpu_docs, link_to_wgpu_item, TextureFormat, TextureUsages};

#[cfg(any(feature = "serde", test))]
use serde::{Deserialize, Serialize};

/// Timing and queueing with which frames are actually displayed to the user.
///
/// Use this as part of a [`SurfaceConfiguration`] to control the behavior of
/// [`SurfaceTexture::present()`].
///
/// Some modes are only supported by some backends.
/// You can use one of the `Auto*` modes, [`Fifo`](Self::Fifo),
/// or choose one of the supported modes from [`SurfaceCapabilities::present_modes`].
///
#[doc = link_to_wgpu_docs!(["presented"]: "struct.SurfaceTexture.html#method.present")]
#[doc = link_to_wgpu_docs!(["`SurfaceTexture::present()`"]: "struct.SurfaceTexture.html#method.present")]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PresentMode {
    /// Chooses the first supported mode out of:
    ///
    /// 1. [`FifoRelaxed`](Self::FifoRelaxed)
    /// 2. [`Fifo`](Self::Fifo)
    ///
    /// Because of the fallback behavior, this is supported everywhere.
    AutoVsync = 0,

    /// Chooses the first supported mode out of:
    ///
    /// 1. [`Immediate`](Self::Immediate)
    /// 2. [`Mailbox`](Self::Mailbox)
    /// 3. [`Fifo`](Self::Fifo)
    ///
    /// Because of the fallback behavior, this is supported everywhere.
    AutoNoVsync = 1,

    /// Presentation frames are kept in a First-In-First-Out queue approximately 3 frames
    /// long. Every vertical blanking period, the presentation engine will pop a frame
    /// off the queue to display. If there is no frame to display, it will present the same
    /// frame again until the next vblank.
    ///
    /// When a present command is executed on the GPU, the presented image is added on the queue.
    ///
    /// Calls to [`Surface::get_current_texture()`] will block until there is a spot in the queue.
    ///
    /// * **Tearing:** No tearing will be observed.
    /// * **Supported on**: All platforms.
    /// * **Also known as**: "Vsync On"
    ///
    /// This is the [default](Self::default) value for `PresentMode`.
    /// If you don't know what mode to choose, choose this mode.
    ///
    #[doc = link_to_wgpu_docs!(["`Surface::get_current_texture()`"]: "struct.Surface.html#method.get_current_texture")]
    #[default]
    Fifo = 2,

    /// Presentation frames are kept in a First-In-First-Out queue approximately 3 frames
    /// long. Every vertical blanking period, the presentation engine will pop a frame
    /// off the queue to display. If there is no frame to display, it will present the
    /// same frame until there is a frame in the queue. The moment there is a frame in the
    /// queue, it will immediately pop the frame off the queue.
    ///
    /// When a present command is executed on the GPU, the presented image is added on the queue.
    ///
    /// Calls to [`Surface::get_current_texture()`] will block until there is a spot in the queue.
    ///
    /// * **Tearing**:
    ///   Tearing will be observed if frames last more than one vblank as the front buffer.
    /// * **Supported on**: AMD on Vulkan.
    /// * **Also known as**: "Adaptive Vsync"
    ///
    #[doc = link_to_wgpu_docs!(["`Surface::get_current_texture()`"]: "struct.Surface.html#method.get_current_texture")]
    FifoRelaxed = 3,

    /// Presentation frames are not queued at all. The moment a present command
    /// is executed on the GPU, the presented image is swapped onto the front buffer
    /// immediately.
    ///
    /// * **Tearing**: Tearing can be observed.
    /// * **Supported on**: Most platforms except older DX12 and Wayland.
    /// * **Also known as**: "Vsync Off"
    Immediate = 4,

    /// Presentation frames are kept in a single-frame queue. Every vertical blanking period,
    /// the presentation engine will pop a frame from the queue. If there is no frame to display,
    /// it will present the same frame again until the next vblank.
    ///
    /// When a present command is executed on the GPU, the frame will be put into the queue.
    /// If there was already a frame in the queue, the new frame will _replace_ the old frame
    /// on the queue.
    ///
    /// * **Tearing**: No tearing will be observed.
    /// * **Supported on**: DX12 on Windows 10, NVidia on Vulkan and Wayland on Vulkan.
    /// * **Also known as**: "Fast Vsync"
    Mailbox = 5,
}

/// Specifies how the alpha channel of the textures should be handled during
/// compositing.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum CompositeAlphaMode {
    /// Chooses either `Opaque` or `Inherit` automatically, depending on the
    /// `alpha_mode` that the current surface can support.
    #[default]
    Auto = 0,
    /// The alpha channel, if it exists, of the textures is ignored in the
    /// compositing process. Instead, the textures is treated as if it has a
    /// constant alpha of 1.0.
    Opaque = 1,
    /// The alpha channel, if it exists, of the textures is respected in the
    /// compositing process. The non-alpha channels of the textures are
    /// expected to already be multiplied by the alpha channel by the
    /// application.
    PreMultiplied = 2,
    /// The alpha channel, if it exists, of the textures is respected in the
    /// compositing process. The non-alpha channels of the textures are not
    /// expected to already be multiplied by the alpha channel by the
    /// application; instead, the compositor will multiply the non-alpha
    /// channels of the texture by the alpha channel during compositing.
    PostMultiplied = 3,
    /// The alpha channel, if it exists, of the textures is unknown for processing
    /// during compositing. Instead, the application is responsible for setting
    /// the composite alpha blending mode using native WSI command. If not set,
    /// then a platform-specific default will be used.
    Inherit = 4,
}

/// The color space in which the presentation engine interprets the values
/// written to a surface texture.
///
/// A color space defines the *primaries*, *white point*, and *transfer
/// function* of the output signal (see the Terminology section below),
/// following the same convention as [CSS predefined color spaces] and
/// [`VkColorSpaceKHR`].
/// It does **not** change the texel format of the surface; it changes how the
/// compositor / display pipeline interprets those texels.
///
/// Support is queried via [`SurfaceCapabilities`], which reports a set of
/// [`SurfaceColorSpaces`] for every supported texture format. Selecting a
/// color space other than [`Srgb`](Self::Srgb) is how an application opts a
/// surface into high-dynamic-range (HDR) or wide-color-gamut output on
/// platforms that support it.
///
/// New to HDR? The `wgpu` crate's top-level docs include a [color space and HDR
/// primer] covering the concepts and the steps to get HDR output on screen.
///
/// # Terminology
///
/// Each variant is described by four properties:
///
/// * **Primaries** (the *gamut*): the chromaticities of the red, green, and
///   blue that color values address, and so the range of colors that can be
///   expressed. [BT.709] (the sRGB / HDTV primaries) is the standard-gamut set;
///   [Display P3] and [BT.2020] are progressively wider.
/// * **White point**: the chromaticity produced by equal red, green, and blue.
///   Every color space here uses [D65], the standard daylight white.
/// * **Transfer function** (the *OETF*): how stored values map to light, such
///   as the [sRGB] transfer function, a linear transfer, or an HDR transfer
///   function like [PQ] or [HLG]. Your shader applies this encoding transfer
///   function; the display applies the inverse (the *EOTF*). Except for writes
///   to an `*Srgb` texture format (where the hardware applies the sRGB encoding
///   for you), wgpu does **not** encode for you: the values your shader writes
///   to the surface texture must already be in whatever encoding the chosen
///   color space expects (linear for a linear transfer). The [HDR surface
///   example] shows the encoder each variant expects.
/// * **Dynamic range**: standard dynamic range (SDR), where `1.0` is reference
///   (SDR) white and values outside 0.0..=1.0 are clamped, or high dynamic
///   range (HDR), where `(1.0, 1.0, 1.0)` is SDR reference white and values
///   above `1.0` drive brighter-than-SDR output on HDR displays.
///
#[doc = include_str!("color_gamuts.svg")]
///
/// *The primaries of each color space form a triangle on the CIE 1931
/// chromaticity diagram; colors inside it are expressible, colors outside are
/// not. [`Srgb`](Self::Srgb) uses the [BT.709] gamut;
/// [`DisplayP3`](Self::DisplayP3) and [`Bt2100Pq`](Self::Bt2100Pq)'s [BT.2020]
/// are progressively wider. All share the [D65] white point.*
///
#[doc = include_str!("sdr_hdr_range.svg")]
///
/// *`0.0` is black and `1.0` is SDR reference white. [`Srgb`](Self::Srgb) and
/// [`DisplayP3`](Self::DisplayP3) clamp above `1.0`; the extended-range and HDR
/// color spaces drive values above `1.0` as brighter-than-SDR output, up to the
/// display's headroom (query it via [`DisplayHdrInfo::tone_map_headroom`]).*
///
/// # Extended-range variants: linear vs encoded
///
/// The extended-range color spaces come in two forms that share a range but
/// differ in transfer: [`ExtendedSrgbLinear`](Self::ExtendedSrgbLinear) carries
/// a **linear** signal, while [`ExtendedSrgb`](Self::ExtendedSrgb) and
/// [`ExtendedDisplayP3`](Self::ExtendedDisplayP3) carry the **sRGB-encoded**
/// (gamma) signal: the sRGB transfer function is applied as usual and then
/// continued to values above `1.0` (brighter than SDR white) and below `0.0`
/// (colors outside the base gamut). Pick by whether the values your shader
/// writes to the surface texture are linear or encoded; confusing the two is the
/// most common HDR setup mistake.
///
/// # Web (WebGPU) backend
///
/// Browsers do not expose these named color spaces directly. A WebGPU canvas is
/// configured with a [`colorSpace`] (only `"srgb"` or `"display-p3"`) plus a
/// separate [`toneMapping`] mode (`"standard"` or `"extended"`), so on the web
/// wgpu offers only the combinations that pair can produce: [`Srgb`](Self::Srgb)
/// and [`DisplayP3`](Self::DisplayP3) with standard tone mapping, plus their
/// extended-range HDR forms [`ExtendedSrgb`](Self::ExtendedSrgb) and
/// [`ExtendedDisplayP3`](Self::ExtendedDisplayP3) with extended tone mapping.
/// There is no linear-transfer canvas color space, so
/// [`ExtendedSrgbLinear`](Self::ExtendedSrgbLinear) (scRGB) is native-only, and
/// [`Bt2100Pq`](Self::Bt2100Pq) and [`Bt2100Hlg`](Self::Bt2100Hlg) are
/// unavailable (browsers expose no PQ or HLG canvas signaling).
///
/// [CSS predefined color spaces]: https://www.w3.org/TR/css-color-4/#predefined
/// [`VkColorSpaceKHR`]: https://registry.khronos.org/vulkan/specs/latest/man/html/VkColorSpaceKHR.html
///
/// [BT.709]: https://www.itu.int/rec/R-REC-BT.709
/// [BT.2020]: https://www.itu.int/rec/R-REC-BT.2020
/// [Display P3]: https://en.wikipedia.org/wiki/DCI-P3#Display_P3
/// [D65]: https://en.wikipedia.org/wiki/Standard_illuminant#D65_values
/// [sRGB]: https://registry.color.org/rgb-registry/srgb
/// [PQ]: https://en.wikipedia.org/wiki/Perceptual_quantizer
/// [HLG]: https://www.itu.int/rec/R-REC-BT.2100
/// [HDR surface example]: https://github.com/gfx-rs/wgpu/tree/trunk/examples/standalone/03_hdr_surface
///
/// [`colorSpace`]: https://www.w3.org/TR/webgpu/#dom-gpucanvasconfiguration-colorspace
/// [`toneMapping`]: https://www.w3.org/TR/webgpu/#gpucanvastonemappingmode
#[doc = link_to_wgpu_docs!(["color space and HDR primer"]: "index.html#surface-color-spaces-and-hdr-output")]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SurfaceColorSpace {
    /// Let the backend choose a color space, reproducing wgpu's historical
    /// behavior:
    ///
    /// * [`ExtendedSrgbLinear`](Self::ExtendedSrgbLinear) if the format is
    ///   [`TextureFormat::Rgba16Float`] and the surface supports it for that
    ///   format,
    /// * otherwise [`Srgb`](Self::Srgb), if the surface supports it for the
    ///   format.
    ///
    /// Apart from the linear [`ExtendedSrgbLinear`](Self::ExtendedSrgbLinear)
    /// above (which fp16 surfaces have historically used and which needs no
    /// extra encoding), `Auto` never resolves to a wide-gamut or HDR color
    /// space, since those would change how the application must encode its
    /// output. If a format is only available in such color spaces (which some
    /// drivers report when the OS is in HDR mode), configuring it with `Auto`
    /// fails validation; such formats are listed in
    /// [`SurfaceCapabilities::format_capabilities`] but excluded from
    /// [`SurfaceCapabilities::formats`].
    ///
    /// On the browser WebGPU backend, `Auto` always keeps the canvas
    /// defaults (sRGB with standard tone mapping), even for
    /// [`TextureFormat::Rgba16Float`]; request
    /// [`ExtendedSrgb`](Self::ExtendedSrgb) explicitly for HDR canvas output
    /// ([`ExtendedSrgbLinear`](Self::ExtendedSrgbLinear) is native-only).
    #[default]
    Auto = 0,

    /// The sRGB color space: BT.709 primaries, D65 white point, sRGB transfer
    /// function, standard dynamic range.
    ///
    /// Values outside of 0.0..=1.0 (after format encoding) are clamped by
    /// the display pipeline.
    ///
    /// This is what every backend produces today for non-`Rgba16Float`
    /// formats and is supported everywhere.
    ///
    /// Note that the transfer function is applied by the *format*, not this
    /// color space choice: an `*Srgb` format applies sRGB encoding on write,
    /// while writes to a non-`*Srgb` format are interpreted as already
    /// sRGB-encoded.
    Srgb = 1,

    /// Extended linear sRGB, also known as [scRGB] (the **linear** encoding of
    /// IEC 61966-2-2): BT.709 primaries, D65 white point, **linear** transfer
    /// function, extended dynamic range. Typically used with
    /// [`TextureFormat::Rgba16Float`].
    ///
    /// The linear counterpart to the sRGB-encoded
    /// [`ExtendedSrgb`](Self::ExtendedSrgb) (IEC 61966-2-2 defines both); pick
    /// this one if the values your shader writes to the surface texture are
    /// **linear**.
    ///
    /// This corresponds to Vulkan's `VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT`,
    /// Metal's extended dynamic range (EDR), and DXGI's
    /// `DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709`.
    ///
    /// * **Supported on**: native only (Vulkan, Metal, DX12). Not available on
    ///   the browser WebGPU backend, which cannot express a linear-transfer
    ///   canvas color space; use [`ExtendedSrgb`](Self::ExtendedSrgb) for HDR
    ///   canvas output on the web.
    /// * **Also known as**: scRGB.
    ///
    /// [scRGB]: https://en.wikipedia.org/wiki/ScRGB
    ExtendedSrgbLinear = 2,

    /// The [Display P3] color space: P3 primaries, D65 white point, sRGB
    /// transfer function, standard dynamic range.
    ///
    /// A wide-gamut SDR color space covering roughly 25% more area than sRGB in
    /// the CIE chromaticity diagram. It uses the wide P3 primaries of theatrical
    /// DCI-P3, but with the D65 white point and the sRGB transfer function (not
    /// DCI's white point and 2.6 gamma).
    ///
    /// Like [`Srgb`](Self::Srgb), this is standard dynamic range (values outside
    /// 0.0..=1.0 are clamped). For wide-gamut HDR that keeps the P3 primaries
    /// but extends the range, use [`ExtendedDisplayP3`](Self::ExtendedDisplayP3).
    ///
    /// * **Supported on**: Vulkan (where the driver exposes it), Metal, and the
    ///   browser WebGPU backend (canvas color space `"display-p3"`). Not
    ///   reported on DX12.
    ///
    /// [Display P3]: https://en.wikipedia.org/wiki/DCI-P3#Display_P3
    DisplayP3 = 3,

    /// BT.2100 perceptual quantization (HDR10): BT.2020/2100 primaries, D65 white
    /// point, SMPTE ST 2084 perceptual quantizer ([PQ]) transfer function, high
    /// dynamic range.
    ///
    /// Texel values are interpreted as a PQ-encoded signal whose encoded range,
    /// `0.0..=1.0`, maps to absolute luminance from 0 to 10,000 nits. The values
    /// your shader writes to the surface texture must already be in the BT.2020
    /// gamut and PQ-encoded into that `0.0..=1.0` range; the [HDR surface example]
    /// shows how. The format is non-sRGB, typically
    /// [`TextureFormat::Rgb10a2Unorm`].
    ///
    /// Commonly known as **HDR10** — though that term additionally implies static
    /// ST 2086 / MaxCLL mastering metadata, which wgpu does not set; this
    /// configures only the PQ color space.
    ///
    /// * **Supported on**: Vulkan (where the driver exposes it), DX12 (on
    ///   `Rgb10a2Unorm`), and Metal. Unavailable on the browser WebGPU backend
    ///   (no PQ canvas signaling).
    ///
    /// [PQ]: https://en.wikipedia.org/wiki/Perceptual_quantizer
    /// [HDR surface example]: https://github.com/gfx-rs/wgpu/tree/trunk/examples/standalone/03_hdr_surface
    Bt2100Pq = 4,

    /// BT.2100 hybrid log-gamma: BT.2020/2100 primaries, D65 white point, [HLG]
    /// (ARIB STD-B67) transfer function, high dynamic range.
    ///
    /// A relative-luminance HDR signal, primarily used for broadcast content. The
    /// values your shader writes to the surface texture must already be in the
    /// BT.2020 gamut and HLG-encoded into `0.0..=1.0`; the [HDR surface example]
    /// shows how. The format is non-sRGB, typically
    /// [`TextureFormat::Rgb10a2Unorm`].
    ///
    /// Unlike [`Bt2100Pq`](Self::Bt2100Pq)'s PQ, the HLG signal is *relative*:
    /// `1.0` maps to the display's nominal peak luminance rather than a fixed
    /// absolute level. BT.2100 defines its reference OOTF at a nominal peak of
    /// 1000 cd/m² (system gamma 1.2); the [HDR surface example] normalizes its
    /// absolute-nit test pattern onto that 1000-nit nominal peak.
    ///
    /// * **Supported on**: Vulkan (where the driver exposes it) and Metal.
    ///   Unavailable on DX12 and the browser WebGPU backend (no HLG canvas
    ///   signaling).
    ///
    /// [HLG]: https://www.itu.int/rec/R-REC-BT.2100
    /// [HDR surface example]: https://github.com/gfx-rs/wgpu/tree/trunk/examples/standalone/03_hdr_surface
    Bt2100Hlg = 5,

    /// Extended-range sRGB (encoded): BT.709 primaries, D65 white point, the
    /// **nonlinear sRGB transfer function extended beyond 0.0..=1.0**,
    /// extended dynamic range.
    ///
    /// The sRGB-encoded sibling of
    /// [`ExtendedSrgbLinear`](Self::ExtendedSrgbLinear): the signal is
    /// sRGB-*encoded* (gamma), not linear. Typically used with
    /// [`TextureFormat::Rgba16Float`].
    ///
    /// If the values your shader writes to the surface texture are **linear**,
    /// you want [`ExtendedSrgbLinear`](Self::ExtendedSrgbLinear) (scRGB) instead;
    /// confusing the two is the most common HDR setup mistake. See the [HDR
    /// surface example] for the encoder.
    ///
    /// This is the "encoded extended range" sRGB used by browser WebGPU (canvas
    /// color space `"srgb"` with `"extended"` tone mapping). It corresponds to
    /// Vulkan's `VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT` and Metal's
    /// `kCGColorSpaceExtendedSRGB`.
    ///
    /// * **Supported on**: Vulkan (where the driver exposes it), Metal, and the
    ///   browser WebGPU backend. Not available on DX12, which has no
    ///   encoded-extended-sRGB swapchain color space.
    ///
    /// [HDR surface example]: https://github.com/gfx-rs/wgpu/tree/trunk/examples/standalone/03_hdr_surface
    ExtendedSrgb = 6,

    /// Extended-range Display-P3 (encoded): P3 primaries, D65 white point, the
    /// **nonlinear sRGB transfer function extended beyond 0.0..=1.0**,
    /// extended dynamic range.
    ///
    /// The wide-gamut (P3) analogue of [`ExtendedSrgb`](Self::ExtendedSrgb), and
    /// the HDR counterpart to the SDR-only [`DisplayP3`](Self::DisplayP3): it
    /// keeps the P3 primaries but extends the encoded range for HDR. Like
    /// [`ExtendedSrgb`](Self::ExtendedSrgb) the signal is sRGB-*encoded* (gamma),
    /// not linear. Typically used with [`TextureFormat::Rgba16Float`].
    ///
    /// * **Supported on**: Metal and the browser WebGPU backend (canvas color
    ///   space `"display-p3"` with `"extended"` tone mapping; Metal's
    ///   `kCGColorSpaceExtendedDisplayP3`). Not available on Vulkan or DX12,
    ///   neither of which has an encoded-extended-Display-P3 swapchain color
    ///   space.
    ExtendedDisplayP3 = 7,
}

impl SurfaceColorSpace {
    /// Returns the [`SurfaceColorSpaces`] flag set holding just this color space,
    /// or `None` for [`Auto`](Self::Auto) (which maps to no specific flag).
    #[must_use]
    pub const fn to_color_spaces(self) -> Option<SurfaceColorSpaces> {
        match self {
            Self::Auto => None,
            Self::Srgb => Some(SurfaceColorSpaces::SRGB),
            Self::ExtendedSrgbLinear => Some(SurfaceColorSpaces::EXTENDED_SRGB_LINEAR),
            Self::DisplayP3 => Some(SurfaceColorSpaces::DISPLAY_P3),
            Self::Bt2100Pq => Some(SurfaceColorSpaces::BT2100_PQ),
            Self::Bt2100Hlg => Some(SurfaceColorSpaces::BT2100_HLG),
            Self::ExtendedSrgb => Some(SurfaceColorSpaces::EXTENDED_SRGB),
            Self::ExtendedDisplayP3 => Some(SurfaceColorSpaces::EXTENDED_DISPLAY_P3),
        }
    }

    /// Whether this is a high-dynamic-range color space: one that drives values
    /// above SDR white (`1.0`) as brighter-than-white output.
    ///
    /// `true` for the extended-range and PQ/HLG spaces; `false` for the SDR ones
    /// ([`Srgb`](Self::Srgb) and the wide-gamut-but-SDR
    /// [`DisplayP3`](Self::DisplayP3)). [`Auto`](Self::Auto) is `false`: it defers
    /// to the backend and is the SDR-safe default, so check the resolved color
    /// space if you need certainty.
    ///
    /// Use this to branch after picking a color space from
    /// [`SurfaceCapabilities`]: an HDR result is the one whose highlights you
    /// scale by [`DisplayHdrInfo::tone_map_headroom`].
    #[must_use]
    pub const fn is_hdr(self) -> bool {
        match self {
            Self::ExtendedSrgbLinear
            | Self::ExtendedSrgb
            | Self::ExtendedDisplayP3
            | Self::Bt2100Pq
            | Self::Bt2100Hlg => true,
            Self::Auto | Self::Srgb | Self::DisplayP3 => false,
        }
    }
}

bitflags::bitflags! {
    /// A set of [`SurfaceColorSpace`]s supported by a surface for a particular
    /// texture format.
    ///
    /// Reported per format in [`SurfaceCapabilities::formats`] via
    /// [`SurfaceFormatCapabilities`].
    #[repr(transparent)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct SurfaceColorSpaces: u32 {
        /// [`SurfaceColorSpace::Srgb`] is supported.
        const SRGB = 1 << 0;
        /// [`SurfaceColorSpace::ExtendedSrgbLinear`] is supported.
        const EXTENDED_SRGB_LINEAR = 1 << 1;
        /// [`SurfaceColorSpace::DisplayP3`] is supported.
        const DISPLAY_P3 = 1 << 2;
        /// [`SurfaceColorSpace::Bt2100Pq`] is supported.
        const BT2100_PQ = 1 << 3;
        /// [`SurfaceColorSpace::Bt2100Hlg`] is supported.
        const BT2100_HLG = 1 << 4;
        /// [`SurfaceColorSpace::ExtendedSrgb`] is supported.
        const EXTENDED_SRGB = 1 << 5;
        /// [`SurfaceColorSpace::ExtendedDisplayP3`] is supported.
        const EXTENDED_DISPLAY_P3 = 1 << 6;
    }
}

/// A texture format supported by a surface, together with the color spaces
/// in which the surface can present it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SurfaceFormatCapabilities {
    /// The texture format.
    pub format: TextureFormat,
    /// The set of color spaces the surface supports for this format.
    ///
    /// This reports which color spaces the surface can be *configured* with; it
    /// does not reflect whether the display is currently in HDR mode. For the
    /// display's live HDR state, see [`DisplayHdrInfo`].
    ///
    /// Guaranteed to be non-empty.
    pub color_spaces: SurfaceColorSpaces,
}

/// Defines the capabilities of a given surface and adapter.
#[derive(Debug)]
pub struct SurfaceCapabilities {
    /// List of supported formats to use with the given adapter. The first format in the vector is preferred.
    ///
    /// Only contains formats that can be configured with the default
    /// [`SurfaceColorSpace::Auto`]; formats available exclusively in
    /// explicit-opt-in (wide-gamut / HDR) color spaces appear only in
    /// [`format_capabilities`](Self::format_capabilities).
    ///
    /// Returns an empty vector if the surface is incompatible with the adapter.
    pub formats: Vec<TextureFormat>,
    /// List of supported formats together with the color spaces supported for
    /// each format, in the same preference order as
    /// [`formats`](Self::formats), of which it is a superset.
    ///
    /// Returns an empty vector if the surface is incompatible with the adapter.
    pub format_capabilities: Vec<SurfaceFormatCapabilities>,
    /// List of supported presentation modes to use with the given adapter.
    ///
    /// Returns an empty vector if the surface is incompatible with the adapter.
    pub present_modes: Vec<PresentMode>,
    /// List of supported alpha modes to use with the given adapter.
    ///
    /// Will return at least one element, [`CompositeAlphaMode::Opaque`] or [`CompositeAlphaMode::Inherit`].
    pub alpha_modes: Vec<CompositeAlphaMode>,
    /// Bitflag of supported texture usages for the surface to use with the given adapter.
    ///
    /// The usage [`TextureUsages::RENDER_ATTACHMENT`] is guaranteed.
    pub usages: TextureUsages,
}

impl SurfaceCapabilities {
    /// Returns the set of color spaces supported for the given format, or an
    /// empty set if the format is not supported.
    ///
    /// This is a convenience lookup over
    /// [`format_capabilities`](Self::format_capabilities): an empty result
    /// means `format` is absent from that list.
    #[must_use]
    pub fn color_spaces(&self, format: TextureFormat) -> SurfaceColorSpaces {
        self.format_capabilities
            .iter()
            .filter(|fc| fc.format == format)
            .fold(SurfaceColorSpaces::empty(), |acc, fc| acc | fc.color_spaces)
    }
}

impl Default for SurfaceCapabilities {
    fn default() -> Self {
        Self {
            formats: Vec::new(),
            format_capabilities: Vec::new(),
            present_modes: Vec::new(),
            alpha_modes: vec![CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT,
        }
    }
}

/// HDR and luminance characteristics of the display backing a [`Surface`], as
/// reported by the platform at query time.
///
/// This describes the display; it does not configure it. Set the output color
/// space through [`SurfaceConfiguration::color_space`]; wgpu does not write HDR
/// metadata (`vkSetHdrMetadataEXT` / DXGI `SetHDRMetaData`).
///
/// Use it for tone mapping, not to decide whether to enable HDR - that is a
/// capability question for [`SurfaceCapabilities`], and holds even when the panel
/// has no headroom right now. The live highlight multiplier is
/// [`tone_map_headroom`](Self::tone_map_headroom).
///
/// The values change as the display does, so re-query after the surface moves or
/// resizes or the display configuration changes.
///
/// Every field is [`Option`] and no platform reports them all; `None` means
/// unknown, never zero and never SDR (Windows reports nits, macOS only a headroom
/// multiplier). The numbers are advisory hints, not contracts: OS/EDID figures run
/// optimistic and report the panel's claim, not what survives the compositor.
///
#[doc = link_to_wgpu_item!(struct Surface)]
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DisplayHdrInfo {
    /// Absolute-nit luminance levels. `Some` only on platforms that report
    /// absolute nits (Windows, via DXGI). `None` on Apple EDR, the web, Vulkan
    /// on non-Windows, and GLES.
    pub luminance: Option<DisplayLuminance>,

    /// Relative EDR-headroom multipliers. `Some` only on Apple. `None`
    /// elsewhere.
    pub headroom: Option<DisplayHeadroom>,

    /// Chromaticity of the display's primaries and white point (CIE 1931 xy).
    /// `Some` only on Windows (via DXGI). `None` on Apple (which exposes no
    /// primaries), the web (boolean-only), Vulkan on non-Windows, and GLES.
    /// Advisory (often EDID-sourced).
    pub chromaticity: Option<DisplayChromaticity>,

    /// Coarse, boolean dynamic-range + gamut bucket. The only luminance-adjacent
    /// data the web exposes (CSS `dynamic-range` / `color-gamut`), and a useful
    /// cross-check elsewhere. `None` only when nothing at all is known.
    pub coarse: Option<DisplayCoarseRange>,

    /// Output signal bit depth, e.g. `8` / `10` / `12` (DXGI `BitsPerColor`).
    /// Advisory and often unreliable (may report `8` on a 10-bit panel). `None`
    /// if unreported.
    pub bits_per_color: Option<u8>,
}

/// Absolute luminance levels in nits (cd/m²). Populated only on Windows (via
/// DXGI); `None` on every other platform.
///
/// Advisory: OS/EDID figures run optimistic. A `0.0` from the OS stays
/// `Some(0.0)`; absence is `None`. These are achromatic (luminance = CIE Y), not a
/// per-color ceiling: a display can't reach [`max_nits`](Self::max_nits) at a
/// saturated chromaticity. Pair them with [`DisplayChromaticity`] for gamut mapping.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DisplayLuminance {
    /// Peak luminance of a small patch, nits. DXGI `MaxLuminance`.
    pub max_nits: Option<f32>,
    /// Sustained full-white-frame luminance, nits: the ceiling for a fully-lit
    /// frame, which power/thermal limits can hold below the small-patch peak
    /// [`max_nits`](Self::max_nits). May equal `max_nits` if the OS reports no
    /// distinct limit. Prefer it over `max_nits` for large bright regions; don't
    /// derive it from `max_nits`. DXGI `MaxFullFrameLuminance`.
    pub max_full_frame_nits: Option<f32>,
    /// Minimum (black) luminance, nits. DXGI `MinLuminance`.
    pub min_nits: Option<f32>,
    /// Luminance the OS maps SDR reference white to, nits; moves with the
    /// brightness slider. Converts between absolute nits and relative EDR headroom
    /// (`max_nits / sdr_white_nits`). Read via the `DISPLAYCONFIG_SDR_WHITE_LEVEL`
    /// query, separate from the other nits, so `None` only if that query fails.
    pub sdr_white_nits: Option<f32>,
}

/// Relative EDR headroom (Apple): unitless multipliers over current SDR white,
/// where `1.0` means no headroom. Moves with brightness, ambient light, battery,
/// and which display the window is on. Apple exposes no absolute-nit equivalent,
/// so this is separate from [`DisplayLuminance`] and can't be converted to nits.
///
/// Populated only on macOS; `None` on iOS, tvOS, and visionOS.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DisplayHeadroom {
    /// Headroom available *right now* (`maximumExtendedDynamicRangeColorComponentValue`
    /// / iOS `UIScreen.currentEDRHeadroom`). `1.0` means no headroom at this
    /// instant, even on an HDR-capable panel.
    pub current: Option<f32>,
    /// Headroom the display could reach under ideal conditions
    /// (`maximumPotentialExtendedDynamicRangeColorComponentValue` /
    /// `UIScreen.potentialEDRHeadroom`).
    pub potential: Option<f32>,
    /// Headroom for reference-white content
    /// (`maximumReferenceExtendedDynamicRangeColorComponentValue`). `None` if
    /// unreported.
    pub reference: Option<f32>,
}

/// CIE 1931 xy chromaticity of a display's primaries and white point. Each
/// coordinate is `[x, y]`; a coordinate the platform omits is `None`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DisplayChromaticity {
    /// xy of the red primary.
    pub red: Option<[f32; 2]>,
    /// xy of the green primary.
    pub green: Option<[f32; 2]>,
    /// xy of the blue primary.
    pub blue: Option<[f32; 2]>,
    /// xy of the white point.
    pub white: Option<[f32; 2]>,
}

/// Coarse, boolean dynamic-range and gamut signal.
///
/// This is the only luminance-adjacent data the web exposes, and a useful
/// cross-check on other platforms.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DisplayCoarseRange {
    /// CSS `@media (dynamic-range: high)`: the display *can* present HDR-range
    /// content. Best-effort and platform-defined — "capable", not "an HDR mode is
    /// active". It feeds [`tone_map_headroom`](DisplayHdrInfo::tone_map_headroom):
    /// `Some(false)` marks a definitively-SDR display, collapsing the headroom to
    /// `1.0`.
    pub high_dynamic_range: Option<bool>,
    /// Best gamut bucket the display covers (CSS `color-gamut`).
    pub gamut: Option<DisplayGamut>,
}

/// Coarse gamut classification, mirroring CSS `color-gamut`.
///
/// These variants are **not** ordered by containment; do not rely on their
/// declaration order to compare gamut sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum DisplayGamut {
    /// Approximately sRGB / Rec.709.
    Srgb,
    /// Approximately Display-P3.
    DisplayP3,
    /// Approximately Rec.2020.
    Rec2020,
}

impl DisplayHdrInfo {
    /// Best-effort tone-map headroom: the linear multiplier of SDR white the
    /// display can drive before clipping. This is the single value most
    /// tone-mappers want, with the subjective parts left to the application.
    ///
    /// Resolution order, first match wins:
    /// 1. Apple EDR: [`DisplayHeadroom::current`] (already a multiplier).
    /// 2. `Some(1.0)` when [`DisplayCoarseRange::high_dynamic_range`] is
    ///    `Some(false)` — a definitively-SDR display (Windows and the web both set
    ///    this flag for an SDR output). Its panel peak may sit above its SDR white,
    ///    but that ratio isn't headroom you can drive, so it isn't reported as
    ///    such.
    /// 3. Absolute nits: `max_nits / sdr_white_nits`, when both are known and
    ///    `sdr_white_nits > 0.0`.
    /// 4. Otherwise `None`: the available figures don't pin a multiplier (e.g.
    ///    `max_nits` known but `sdr_white_nits` unknown).
    ///
    /// Use `unwrap_or(1.0)` on the result for the SDR fallback. Never returns a
    /// non-finite value.
    #[must_use]
    pub fn tone_map_headroom(&self) -> Option<f32> {
        // Apple EDR reports a multiplier directly.
        if let Some(h) = self
            .headroom
            .and_then(|h| h.current)
            .filter(|h| h.is_finite())
        {
            return Some(h);
        }
        // A definitively-SDR display still reports a physical peak against a
        // default SDR white; checked before the nit ratio so that unusable ratio
        // can't surface as phantom headroom.
        if self.coarse.and_then(|c| c.high_dynamic_range) == Some(false) {
            return Some(1.0);
        }
        // Otherwise derive the multiplier from absolute nits, when both the peak
        // and the SDR white level are known.
        if let Some((max, sdr)) = self
            .luminance
            .and_then(|l| l.max_nits.zip(l.sdr_white_nits))
            .filter(|&(max, sdr)| sdr > 0.0 && max.is_finite() && sdr.is_finite())
        {
            return Some(max / sdr);
        }
        None
    }
}

#[cfg(test)]
mod display_hdr_info_tests {
    use super::*;

    #[test]
    fn default_is_unknown() {
        // Nothing known, so no headroom is derived — it never guesses SDR vs HDR.
        assert_eq!(DisplayHdrInfo::default().tone_map_headroom(), None);
    }

    #[test]
    fn apple_headroom_is_used_directly() {
        // Apple reports a live multiplier; it's returned as-is.
        let info = DisplayHdrInfo {
            headroom: Some(DisplayHeadroom {
                current: Some(3.0),
                potential: Some(5.0),
                reference: None,
            }),
            ..Default::default()
        };
        assert_eq!(info.tone_map_headroom(), Some(3.0));
    }

    #[test]
    fn apple_uses_current_not_potential() {
        // A capable panel with no headroom right now (current 1.0, potential 16.0
        // — e.g. macOS at full brightness). The live value wins; the potential
        // ceiling is never tone-mapped against.
        let info = DisplayHdrInfo {
            headroom: Some(DisplayHeadroom {
                current: Some(1.0),
                potential: Some(16.0),
                reference: None,
            }),
            ..Default::default()
        };
        assert_eq!(info.tone_map_headroom(), Some(1.0));
    }

    #[test]
    fn windows_nits_derive_headroom_only_with_sdr_white() {
        // Both nits present and sdr_white > 0, so it returns the ratio.
        let info = DisplayHdrInfo {
            luminance: Some(DisplayLuminance {
                max_nits: Some(800.0),
                sdr_white_nits: Some(200.0),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(info.tone_map_headroom(), Some(4.0));

        // max_nits known but sdr_white unknown, so it won't guess across frames.
        let info = DisplayHdrInfo {
            luminance: Some(DisplayLuminance {
                max_nits: Some(800.0),
                sdr_white_nits: None,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(info.tone_map_headroom(), None);
    }

    #[test]
    fn sdr_display_collapses_to_unity() {
        // A definitively-SDR display (`dynamic-range: standard`) has no usable
        // headroom, even with no luminance figures at all.
        let info = DisplayHdrInfo {
            coarse: Some(DisplayCoarseRange {
                high_dynamic_range: Some(false),
                gamut: Some(DisplayGamut::Srgb),
            }),
            ..Default::default()
        };
        assert_eq!(info.tone_map_headroom(), Some(1.0));
    }

    #[test]
    fn sdr_display_overrides_panel_nits() {
        // An SDR-mode output still reports its EDID peak (270 nits) against a
        // default 80-nit SDR white. That 270/80 ratio is unusable, so the SDR flag
        // wins and the headroom collapses to 1.0 rather than 3.375.
        let info = DisplayHdrInfo {
            luminance: Some(DisplayLuminance {
                max_nits: Some(270.0),
                sdr_white_nits: Some(80.0),
                ..Default::default()
            }),
            coarse: Some(DisplayCoarseRange {
                high_dynamic_range: Some(false),
                gamut: Some(DisplayGamut::DisplayP3),
            }),
            ..Default::default()
        };
        assert_eq!(info.tone_map_headroom(), Some(1.0));
    }

    #[test]
    fn coarse_hdr_capable_alone_derives_nothing() {
        // `dynamic-range: high` (the web's only signal) means the display is
        // HDR-capable, not that headroom is available — and it carries no
        // luminance to derive one from, so the headroom stays unknown.
        let info = DisplayHdrInfo {
            coarse: Some(DisplayCoarseRange {
                high_dynamic_range: Some(true),
                gamut: Some(DisplayGamut::Rec2020),
            }),
            ..Default::default()
        };
        assert_eq!(info.tone_map_headroom(), None);
    }

    #[test]
    fn non_finite_current_falls_through_to_nits() {
        // A non-finite EDR read is skipped, not leaked; the nit ratio answers.
        let info = DisplayHdrInfo {
            headroom: Some(DisplayHeadroom {
                current: Some(f32::INFINITY),
                ..Default::default()
            }),
            luminance: Some(DisplayLuminance {
                max_nits: Some(1000.0),
                sdr_white_nits: Some(100.0),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(info.tone_map_headroom(), Some(10.0));
    }
}

/// Configures a [`Surface`] for presentation.
///
#[doc = link_to_wgpu_item!(struct Surface)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SurfaceConfiguration<V> {
    /// The usage of the swap chain. The only usage guaranteed to be supported is [`TextureUsages::RENDER_ATTACHMENT`].
    pub usage: TextureUsages,
    /// The texture format of the swap chain. The only formats that are guaranteed are
    /// [`TextureFormat::Bgra8Unorm`] and [`TextureFormat::Bgra8UnormSrgb`].
    pub format: TextureFormat,
    /// The color space in which the presentation engine interprets the values
    /// written to the swap chain.
    ///
    /// The supported color spaces for each format are listed in
    /// [`SurfaceCapabilities::format_capabilities`].
    /// [`SurfaceColorSpace::Auto`] (the default) is supported for every
    /// format in [`SurfaceCapabilities::formats`]; any other value must be
    /// present in the format's
    /// [`color_spaces`](SurfaceFormatCapabilities::color_spaces) set.
    pub color_space: SurfaceColorSpace,
    /// Width of the swap chain. Must be the same size as the surface, and nonzero.
    ///
    /// If this is not the same size as the underlying surface (e.g. if it is
    /// set once, and the window is later resized), the behaviour is defined
    /// but platform-specific, and may change in the future (currently macOS
    /// scales the surface, other platforms may do something else).
    pub width: u32,
    /// Height of the swap chain. Must be the same size as the surface, and nonzero.
    ///
    /// If this is not the same size as the underlying surface (e.g. if it is
    /// set once, and the window is later resized), the behaviour is defined
    /// but platform-specific, and may change in the future (currently macOS
    /// scales the surface, other platforms may do something else).
    pub height: u32,
    /// Presentation mode of the swap chain. Fifo is the only mode guaranteed to be supported.
    /// `FifoRelaxed`, `Immediate`, and `Mailbox` will crash if unsupported, while `AutoVsync` and
    /// `AutoNoVsync` will gracefully do a designed sets of fallbacks if their primary modes are
    /// unsupported.
    pub present_mode: PresentMode,
    /// Desired maximum number of monitor refreshes between a [`Surface::get_current_texture`] call and the
    /// texture being presented to the screen. This is sometimes called "Frames in Flight".
    ///
    /// Defaults to `2` when created via [`Surface::get_default_config`] as this is a reasonable default.
    ///
    /// This is ultimately a hint to the backend implementation and will always be clamped
    /// to the supported range.
    ///
    /// Typical values are `1` to `3`, but higher values are valid, though likely to be clamped.
    /// * Choose `1` to minimize latency above all else. This only gives a single monitor refresh for all of
    ///   the CPU and GPU work to complete. ⚠️ As a result of these short swapchains, the CPU and GPU
    ///   cannot run in parallel, prioritizing latency over throughput. For applications like GUIs doing
    ///   a small amount of GPU work each frame that need low latency, this is a reasonable choice.
    /// * Choose `2` for a balance between latency and throughput. The CPU and GPU both can each use
    ///   a full monitor refresh to do their computations. This is a reasonable default for most applications.
    /// * Choose `3` or higher to maximize throughput, sacrificing latency when the CPU and GPU
    ///   are using less than a full monitor refresh each. For applications that use CPU-side pipelining
    ///   of frames this may be a reasonable choice. ⚠️ On 60hz displays the latency can be very noticeable.
    ///
    /// This maps to the backend in the following ways:
    /// - Vulkan: Number of frames in the swapchain is `desired_maximum_frame_latency + 1`,
    ///   clamped to the supported range.
    /// - DX12: Calls [`IDXGISwapChain2::SetMaximumFrameLatency(desired_maximum_frame_latency)`][SMFL].
    /// - Metal: Sets the `maximumDrawableCount` of the underlying `CAMetalLayer` to
    ///   `desired_maximum_frame_latency + 1`, clamped to the supported range.
    /// - OpenGL: Ignored
    ///
    /// It also has various subtle interactions with various present modes and APIs.
    /// - DX12 + Mailbox: Limits framerate to `desired_maximum_frame_latency * Monitor Hz` fps.
    /// - Vulkan/Metal + Mailbox: If this is set to `2`, limits framerate to `2 * Monitor Hz` fps. `3` or higher is unlimited.
    ///
    #[doc = link_to_wgpu_docs!(["`Surface::get_current_texture`"]: "struct.Surface.html#method.get_current_texture")]
    #[doc = link_to_wgpu_docs!(["`Surface::get_default_config`"]: "struct.Surface.html#method.get_default_config")]
    /// [SMFL]: https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_3/nf-dxgi1_3-idxgiswapchain2-setmaximumframelatency
    pub desired_maximum_frame_latency: u32,
    /// Specifies how the alpha channel of the textures should be handled during compositing.
    pub alpha_mode: CompositeAlphaMode,
    /// Specifies what view formats will be allowed when calling `Texture::create_view` on the texture returned by `Surface::get_current_texture`.
    ///
    /// View formats of the same format as the texture are always allowed.
    ///
    /// Note: currently, only the srgb-ness is allowed to change. (ex: `Rgba8Unorm` texture + `Rgba8UnormSrgb` view)
    pub view_formats: V,
}

impl<V: Clone> SurfaceConfiguration<V> {
    /// Map `view_formats` of the texture descriptor into another.
    pub fn map_view_formats<'a, M>(
        &'a self,
        fun: impl FnOnce(&'a V) -> M,
    ) -> SurfaceConfiguration<M> {
        SurfaceConfiguration {
            usage: self.usage,
            format: self.format,
            color_space: self.color_space,
            width: self.width,
            height: self.height,
            present_mode: self.present_mode,
            desired_maximum_frame_latency: self.desired_maximum_frame_latency,
            alpha_mode: self.alpha_mode,
            view_formats: fun(&self.view_formats),
        }
    }
}

/// Status of the received surface image.
#[repr(C)]
#[derive(Debug)]
pub enum SurfaceStatus {
    /// No issues.
    Good,
    /// The swap chain is operational, but it does no longer perfectly
    /// match the surface. A re-configuration is needed.
    Suboptimal,
    /// Unable to get the next frame, timed out.
    ///
    /// Try reconfiguring your surface.
    Timeout,
    /// The window is occluded (e.g. minimized or behind another window).
    ///
    /// Try again once the window is no longer occluded.
    Occluded,
    /// The surface under the swap chain has changed.
    ///
    /// Try reconfiguring your surface.
    Outdated,
    /// The surface under the swap chain is lost.
    Lost,
    /// `Surface::get_current_texture` has hit a validation error which was caught
    /// by a error scope.
    Validation,
}

/// Nanosecond timestamp used by the presentation engine.
///
/// The specific clock depends on the window system integration (WSI) API used.
///
/// <table>
/// <tr>
///     <td>WSI</td>
///     <td>Clock</td>
/// </tr>
/// <tr>
///     <td>IDXGISwapchain</td>
///     <td><a href="https://docs.microsoft.com/en-us/windows/win32/api/profileapi/nf-profileapi-queryperformancecounter">QueryPerformanceCounter</a></td>
/// </tr>
/// <tr>
///     <td>IPresentationManager</td>
///     <td><a href="https://docs.microsoft.com/en-us/windows/win32/api/realtimeapiset/nf-realtimeapiset-queryinterrupttimeprecise">QueryInterruptTimePrecise</a></td>
/// </tr>
/// <tr>
///     <td>CAMetalLayer</td>
///     <td><a href="https://developer.apple.com/documentation/kernel/1462446-mach_absolute_time">mach_absolute_time</a></td>
/// </tr>
/// <tr>
///     <td>VK_GOOGLE_display_timing</td>
///     <td><a href="https://linux.die.net/man/3/clock_gettime">clock_gettime(CLOCK_MONOTONIC)</a></td>
/// </tr>
/// </table>
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PresentationTimestamp(
    /// Timestamp in nanoseconds.
    pub u128,
);

impl PresentationTimestamp {
    /// A timestamp that is invalid due to the platform not having a timestamp system.
    pub const INVALID_TIMESTAMP: Self = Self(u128::MAX);

    /// Returns true if this timestamp is the invalid timestamp.
    #[must_use]
    pub fn is_invalid(self) -> bool {
        self == Self::INVALID_TIMESTAMP
    }
}
