//! Backend-agnostic DXGI flip-model swapchain helpers, shared by the DX12 backend and the
//! Windows Vulkan backend's DXGI interop swapchain.

use wgpu_sync::Mutex;
use windows::Win32::{Foundation, Graphics::Dxgi, System::Threading, UI::WindowsAndMessaging};

use crate::auxil::dxgi::{dcomp, factory::DxgiFactory};

/// What a [`SwapChain`](Dxgi::IDXGISwapChain3) presents to.
///
/// The Vulkan backend only builds [`Self::WndHandle`] and [`Self::VisualFromWndHandle`]; the
/// remaining variants come from DX12's external-surface entry points and only exist when the
/// DX12 backend is built.
pub(crate) enum SurfaceTarget {
    /// Borrowed, lifetime externally managed
    WndHandle(Foundation::HWND),
    /// `handle` is borrowed, lifetime externally managed
    VisualFromWndHandle {
        handle: Foundation::HWND,
        dcomp_state: Mutex<dcomp::DCompState>,
    },
    #[cfg(dx12)]
    Visual(windows::Win32::Graphics::DirectComposition::IDCompositionVisual),
    /// Borrowed, lifetime externally managed
    #[cfg(dx12)]
    SurfaceHandle(Foundation::HANDLE),
    #[cfg(dx12)]
    SwapChainPanel(crate::auxil::dxgi::types::ISwapChainPanelNative),
}

/// Whether `factory` reports support for `DXGI_PRESENT_ALLOW_TEARING`.
pub(crate) fn supports_allow_tearing(factory: &DxgiFactory) -> bool {
    let Some(factory5) = factory.as_factory5() else {
        return false;
    };
    let mut allow_tearing = Foundation::FALSE;
    let hr = unsafe {
        factory5.CheckFeatureSupport(
            Dxgi::DXGI_FEATURE_PRESENT_ALLOW_TEARING,
            <*mut _>::cast(&mut allow_tearing),
            size_of_val(&allow_tearing) as u32,
        )
    };
    match hr {
        Err(err) => {
            log::warn!("Unable to check for tearing support: {err}");
            false
        }
        Ok(()) => true,
    }
}

/// The swapchain creation flags.
///
/// `ALLOW_TEARING` must be baked in at creation (it cannot be changed by `ResizeBuffers`) so it
/// is always set when supported, regardless of present mode. `FRAME_LATENCY_WAITABLE_OBJECT` is
/// set only when the caller will actually use the waitable object — pairing the flag with the
/// waitable keeps the two consistent.
pub(crate) fn swap_chain_flags(
    supports_allow_tearing: bool,
    frame_latency_waitable: bool,
) -> Dxgi::DXGI_SWAP_CHAIN_FLAG {
    let mut flags = Dxgi::DXGI_SWAP_CHAIN_FLAG::default();
    if supports_allow_tearing {
        flags |= Dxgi::DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
    }
    if frame_latency_waitable {
        flags |= Dxgi::DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
    }
    flags
}

/// Builds the flip-model `DXGI_SWAP_CHAIN_DESC1`.
pub(crate) fn swap_chain_descriptor(
    format: Dxgi::Common::DXGI_FORMAT,
    size: wgt::Extent3d,
    buffer_count: u32,
    alpha_mode: Dxgi::Common::DXGI_ALPHA_MODE,
    flags: Dxgi::DXGI_SWAP_CHAIN_FLAG,
) -> Dxgi::DXGI_SWAP_CHAIN_DESC1 {
    Dxgi::DXGI_SWAP_CHAIN_DESC1 {
        AlphaMode: alpha_mode,
        Width: size.width,
        Height: size.height,
        Format: format,
        Stereo: false.into(),
        SampleDesc: Dxgi::Common::DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: Dxgi::DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: buffer_count,
        Scaling: Dxgi::DXGI_SCALING_STRETCH,
        SwapEffect: Dxgi::DXGI_SWAP_EFFECT_FLIP_DISCARD,
        Flags: flags.0 as u32,
    }
}

/// Maps a [`wgt::PresentMode`] to the `(sync_interval, flags)` pair for `IDXGISwapChain::Present`.
pub(crate) fn present_flags(present_mode: wgt::PresentMode) -> (u32, Dxgi::DXGI_PRESENT) {
    match present_mode {
        // We only allow immediate if ALLOW_TEARING is valid.
        wgt::PresentMode::Immediate => (0, Dxgi::DXGI_PRESENT_ALLOW_TEARING),
        wgt::PresentMode::Mailbox => (0, Dxgi::DXGI_PRESENT::default()),
        wgt::PresentMode::Fifo => (1, Dxgi::DXGI_PRESENT::default()),
        m => unreachable!("Cannot make surface with present mode {m:?}"),
    }
}

/// Waits on the swapchain's frame-latency waitable object for CPU-side frame pacing.
///
/// Returns `Ok(false)` if the wait timed out and `Ok(true)` otherwise (including when there is
/// no waitable object).
pub(crate) fn wait_for_waitable(
    waitable: Option<Foundation::HANDLE>,
    timeout: Option<core::time::Duration>,
) -> Result<bool, crate::SurfaceError> {
    let Some(waitable) = waitable else {
        return Ok(true);
    };
    let timeout_ms = match timeout {
        Some(duration) => duration.as_millis() as u32,
        None => Threading::INFINITE,
    };
    match unsafe { Threading::WaitForSingleObject(waitable, timeout_ms) } {
        Foundation::WAIT_ABANDONED | Foundation::WAIT_FAILED => Err(crate::SurfaceError::Lost),
        Foundation::WAIT_OBJECT_0 => Ok(true),
        Foundation::WAIT_TIMEOUT => Ok(false),
        other => {
            log::error!("Unexpected wait status: 0x{other:x?}");
            Err(crate::SurfaceError::Lost)
        }
    }
}

/// Maps a [`wgt::SurfaceColorSpace`] to a `DXGI_COLOR_SPACE_TYPE` for `SetColorSpace1`.
pub(crate) fn map_surface_color_space(
    color_space: wgt::SurfaceColorSpace,
) -> Dxgi::Common::DXGI_COLOR_SPACE_TYPE {
    use wgt::SurfaceColorSpace as Scs;
    match color_space {
        Scs::Srgb => Dxgi::Common::DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
        Scs::ExtendedSrgbLinear => Dxgi::Common::DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709,
        Scs::Bt2100Pq => Dxgi::Common::DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
        Scs::Auto
        | Scs::DisplayP3
        | Scs::Bt2100Hlg
        | Scs::ExtendedSrgb
        | Scs::ExtendedDisplayP3 => {
            unreachable!("`{color_space:?}` is never reported in the DXGI surface capabilities")
        }
    }
}

/// The presentation capabilities of a DXGI flip-model surface; identical for both backends, since
/// they drive the same kind of swapchain.
pub(crate) fn surface_capabilities(
    target: &SurfaceTarget,
    supports_allow_tearing: bool,
) -> crate::SurfaceCapabilities {
    let mut present_modes = vec![wgt::PresentMode::Mailbox, wgt::PresentMode::Fifo];
    if supports_allow_tearing {
        present_modes.push(wgt::PresentMode::Immediate);
    }

    // `Surface::configure` applies the requested color space with
    // `IDXGISwapChain3::SetColorSpace1`. fp16 buffers keep DXGI's scRGB interpretation
    // (`DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709`) and `Rgb10a2Unorm` additionally supports
    // BT.2100 PQ (HDR10).
    //
    // These color spaces are advertised unconditionally, not gated on whether the output is
    // currently in HDR mode: Windows always composites in scRGB and tone-maps PQ down to an SDR
    // output, so the color space is configurable regardless, and `CheckColorSpaceSupport`
    // returning false does not mean it won't present. Whether HDR is actually *visible* is a
    // separate, live question (the display-HDR query, #9739), not a configuration gate. Display-P3
    // and HLG are never reported: DXGI has no RGB HLG swapchain color space, and P3 isn't a DXGI
    // swapchain color space.
    let formats = [
        wgt::TextureFormat::Bgra8UnormSrgb,
        wgt::TextureFormat::Bgra8Unorm,
        wgt::TextureFormat::Rgba8UnormSrgb,
        wgt::TextureFormat::Rgba8Unorm,
        wgt::TextureFormat::Rgb10a2Unorm,
        wgt::TextureFormat::Rgba16Float,
    ]
    .map(|format| wgt::SurfaceFormatCapabilities {
        format,
        color_spaces: match format {
            wgt::TextureFormat::Rgba16Float => wgt::SurfaceColorSpaces::EXTENDED_SRGB_LINEAR,
            wgt::TextureFormat::Rgb10a2Unorm => {
                wgt::SurfaceColorSpaces::SRGB | wgt::SurfaceColorSpaces::BT2100_PQ
            }
            _ => wgt::SurfaceColorSpaces::SRGB,
        },
    })
    .to_vec();

    let composite_alpha_modes = match target {
        SurfaceTarget::WndHandle(_) => vec![wgt::CompositeAlphaMode::Opaque],
        _ => vec![
            wgt::CompositeAlphaMode::Auto,
            wgt::CompositeAlphaMode::Inherit,
            wgt::CompositeAlphaMode::Opaque,
            wgt::CompositeAlphaMode::PostMultiplied,
            wgt::CompositeAlphaMode::PreMultiplied,
        ],
    };

    crate::SurfaceCapabilities {
        formats,
        // See https://learn.microsoft.com/en-us/windows/win32/api/dxgi/nf-dxgi-idxgidevice1-setmaximumframelatency
        maximum_frame_latency: 1..=16,
        current_extent: surface_extent(target),
        usage: wgt::TextureUses::COLOR_TARGET
            | wgt::TextureUses::COPY_SRC
            | wgt::TextureUses::COPY_DST,
        present_modes,
        composite_alpha_modes,
    }
}

/// The current client-area extent of `target`'s backing window, or `None` for targets without one
/// (the DX12 external-surface variants) or when the query fails.
fn surface_extent(target: &SurfaceTarget) -> Option<wgt::Extent3d> {
    let handle = match target {
        SurfaceTarget::WndHandle(handle) | SurfaceTarget::VisualFromWndHandle { handle, .. } => {
            *handle
        }
        #[allow(unreachable_patterns)]
        _ => return None,
    };
    let mut rect = Default::default();
    if unsafe { WindowsAndMessaging::GetClientRect(handle, &mut rect) }.is_ok() {
        Some(wgt::Extent3d {
            width: (rect.right - rect.left) as u32,
            height: (rect.bottom - rect.top) as u32,
            depth_or_array_layers: 1,
        })
    } else {
        log::warn!("Unable to get the window client rect");
        None
    }
}
