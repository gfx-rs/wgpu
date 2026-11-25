use wgpu_sync::Mutex;
use windows::Win32::Foundation;

use crate::auxil::dxgi::dcomp;

/// What a [`SwapChain`](windows::Win32::Graphics::Dxgi::IDXGISwapChain3) presents to.
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

// SAFETY: `HWND` and `HANDLE` are identifiers that any thread can use, and any thread can call
// DirectComposition. `ISwapChainPanelNative` is not free-threaded. The DX12 backend has always
// treated it as `Send` and `Sync`, and the caller must use such a surface on the panel's UI thread.
unsafe impl Send for SurfaceTarget {}
unsafe impl Sync for SurfaceTarget {}
