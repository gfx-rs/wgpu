use wgpu_sync::Mutex;
use windows::Win32::{Foundation, Graphics::DirectComposition};

use crate::auxil::dxgi::{dcomp, types::ISwapChainPanelNative};

pub(crate) enum SurfaceTarget {
    /// Borrowed, lifetime externally managed
    WndHandle(Foundation::HWND),
    /// `handle` is borrowed, lifetime externally managed
    VisualFromWndHandle {
        handle: Foundation::HWND,
        dcomp_state: Mutex<dcomp::DCompState>,
    },
    Visual(DirectComposition::IDCompositionVisual),
    /// Borrowed, lifetime externally managed
    SurfaceHandle(Foundation::HANDLE),
    SwapChainPanel(ISwapChainPanelNative),
}
