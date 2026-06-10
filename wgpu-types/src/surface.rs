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
/// A color space defines the color primaries, white point, and transfer
/// function of the output signal, following the same convention as
/// [CSS `predefined color spaces`] and [`VkColorSpaceKHR`]. It does **not**
/// change the texel format of the surface; it changes how the compositor /
/// display pipeline interprets those texels.
///
/// Support is queried via [`SurfaceCapabilities`], which reports a set of
/// [`SurfaceColorSpaces`] for every supported texture format. Selecting a
/// color space other than [`Srgb`](Self::Srgb) is how an application opts a
/// surface into high-dynamic-range (HDR) or wide-color-gamut output on
/// platforms that support it.
///
/// [CSS `predefined color spaces`]: https://www.w3.org/TR/css-color-4/#predefined
/// [`VkColorSpaceKHR`]: https://registry.khronos.org/vulkan/specs/latest/man/html/VkColorSpaceKHR.html
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
    /// `Auto` never resolves to a wide-gamut or HDR color space, since those
    /// change how the application must encode its output. If a format is
    /// only available in such color spaces (which some drivers report when
    /// the OS is in HDR mode), configuring it with `Auto` fails validation;
    /// such formats are listed in
    /// [`SurfaceCapabilities::format_capabilities`] but excluded from
    /// [`SurfaceCapabilities::formats`].
    ///
    /// On the browser WebGPU backend, `Auto` always keeps the canvas
    /// defaults (sRGB with standard tone mapping), even for
    /// [`TextureFormat::Rgba16Float`]; request
    /// [`ExtendedSrgbLinear`](Self::ExtendedSrgbLinear) explicitly for HDR
    /// canvas output.
    #[default]
    Auto = 0,

    /// The sRGB color space: BT.709 primaries, D65 white point, sRGB transfer
    /// function, standard dynamic range.
    ///
    /// Values outside of `[0.0, 1.0]` (after format encoding) are clamped by
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

    /// Extended linear sRGB, also known as scRGB (IEC 61966-2-2): BT.709
    /// primaries, D65 white point, **linear** transfer function, extended
    /// dynamic range.
    ///
    /// `(1.0, 1.0, 1.0)` is SDR reference white; values above `1.0` produce
    /// brighter-than-SDR output on HDR displays and colors outside the BT.709
    /// gamut may be represented with values outside of `[0.0, 1.0]`.
    /// Typically used with [`TextureFormat::Rgba16Float`].
    ///
    /// This corresponds to Vulkan's `VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT`,
    /// Metal's extended dynamic range (EDR), and DXGI's
    /// `DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709`.
    ExtendedSrgbLinear = 2,

    /// The Display-P3 color space: P3 primaries, D65 white point, sRGB
    /// transfer function, standard dynamic range.
    ///
    /// A wide-gamut SDR color space; the same encoded values cover roughly
    /// 25% more chromaticities than sRGB.
    DisplayP3 = 3,

    /// HDR10: BT.2020 primaries, D65 white point, SMPTE ST 2084 perceptual
    /// quantizer (PQ) transfer function, high dynamic range.
    ///
    /// Texel values are interpreted as a PQ-encoded signal where the encoded
    /// range maps to absolute luminance from 0 to 10,000 nits. The
    /// application is responsible for applying the PQ encoding (and the
    /// BT.709 → BT.2020 gamut conversion) in its final render pass; the
    /// surface format itself is non-sRGB, typically
    /// [`TextureFormat::Rgb10a2Unorm`].
    Hdr10 = 4,

    /// BT.2100 hybrid log-gamma: BT.2020 primaries, D65 white point, HLG
    /// (ARIB STD-B67) transfer function, high dynamic range.
    ///
    /// A relative-luminance HDR signal, primarily used for broadcast content.
    /// The application is responsible for applying the HLG OETF in its final
    /// render pass.
    Hlg = 5,
}

impl SurfaceColorSpace {
    /// Returns the corresponding capability flag, or `None` for [`Auto`](Self::Auto).
    #[must_use]
    pub fn to_flag(self) -> Option<SurfaceColorSpaces> {
        match self {
            Self::Auto => None,
            Self::Srgb => Some(SurfaceColorSpaces::SRGB),
            Self::ExtendedSrgbLinear => Some(SurfaceColorSpaces::EXTENDED_SRGB_LINEAR),
            Self::DisplayP3 => Some(SurfaceColorSpaces::DISPLAY_P3),
            Self::Hdr10 => Some(SurfaceColorSpaces::HDR10),
            Self::Hlg => Some(SurfaceColorSpaces::HLG),
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
        /// [`SurfaceColorSpace::Hdr10`] is supported.
        const HDR10 = 1 << 3;
        /// [`SurfaceColorSpace::Hlg`] is supported.
        const HLG = 1 << 4;
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
