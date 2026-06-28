//! Backend-agnostic DXGI flip-model swapchain helpers, shared by the DX12 backend and the
//! Windows Vulkan backend's DXGI interop swapchain.

use wgpu_sync::Mutex;
use windows::Win32::{Foundation, Graphics::Dxgi, System::Threading};

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
