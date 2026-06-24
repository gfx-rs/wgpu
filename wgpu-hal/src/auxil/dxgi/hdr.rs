//! Mapping a DXGI output description into the backend-agnostic
//! `DisplayHdrInfo`.
//!
//! This lives in `auxil/dxgi` so the DX12 and Vulkan-on-Windows backends share
//! one mapping and the same monitor reports identical numbers under either
//! backend. The per-backend monitor *walk* differs (DX12 enumerates its own
//! `IDXGIAdapter`'s outputs; the Vulkan path creates its own DXGI factory), but
//! both feed the resulting [`DXGI_OUTPUT_DESC1`] through this one function.
//!
//! [`DXGI_OUTPUT_DESC1`]: windows::Win32::Graphics::Dxgi::DXGI_OUTPUT_DESC1

use windows::{
    core::Interface as _,
    Win32::{
        Foundation::HWND,
        Graphics::{Dxgi, Gdi},
    },
};

/// Finds the [`DXGI_OUTPUT_DESC1`] for the monitor that currently backs
/// `wnd_handle`, walking *every* adapter's outputs via a fresh DXGI factory.
///
/// DXGI is an OS service that needs no D3D device, so this is reachable from both
/// the DX12 and Vulkan backends, which call it identically — the same monitor
/// reports the same descriptor under either backend. Enumerating *all* adapters
/// (rather than only the rendering adapter's) means the monitor is found even on
/// hybrid-GPU systems where the window sits on a display wired to a different
/// adapter.
///
/// Returns `None` if the output cannot be identified or queried: a headless or
/// composition window, a pre-Win10-1703 system without `IDXGIOutput6`, or a COM
/// failure. Never panics.
///
/// [`DXGI_OUTPUT_DESC1`]: Dxgi::DXGI_OUTPUT_DESC1
pub fn output_desc1_for_window(wnd_handle: HWND) -> Option<Dxgi::DXGI_OUTPUT_DESC1> {
    // SAFETY: `MonitorFromWindow` is sound for any `HWND`; an invalid one yields
    // a null `HMONITOR`, checked below.
    let hmonitor = unsafe { Gdi::MonitorFromWindow(wnd_handle, Gdi::MONITOR_DEFAULTTONEAREST) };
    if hmonitor.is_invalid() {
        log::warn!("MonitorFromWindow failed; cannot identify the window's output");
        return None;
    }
    // SAFETY: `CreateDXGIFactory1` takes no caller pointers; the `windows`
    // binding fills the interface out-pointer itself.
    let factory: Dxgi::IDXGIFactory1 = match unsafe { Dxgi::CreateDXGIFactory1() } {
        Ok(factory) => factory,
        Err(e) => {
            log::warn!("CreateDXGIFactory1 failed: {e}");
            return None;
        }
    };
    for adapter_index in 0.. {
        // SAFETY: `factory` is live (created above); `EnumAdapters1` takes only
        // an index.
        let adapter = match unsafe { factory.EnumAdapters1(adapter_index) } {
            Ok(adapter) => adapter,
            // End of the adapter list: the monitor matched none of them.
            Err(e) if e.code() == Dxgi::DXGI_ERROR_NOT_FOUND => break,
            Err(e) => {
                log::warn!("IDXGIFactory1::EnumAdapters1 failed: {e}");
                break;
            }
        };
        for output_index in 0.. {
            // SAFETY: `adapter` is live (from above); `EnumOutputs` takes only
            // an index.
            let output = match unsafe { adapter.EnumOutputs(output_index) } {
                Ok(output) => output,
                // End of this adapter's outputs: move on to the next adapter.
                Err(e) if e.code() == Dxgi::DXGI_ERROR_NOT_FOUND => break,
                Err(e) => {
                    log::warn!("IDXGIAdapter1::EnumOutputs failed: {e}");
                    break;
                }
            };
            // SAFETY: `output` is live (from above); `GetDesc` returns a
            // binding-owned `DXGI_OUTPUT_DESC` by value.
            let desc = match unsafe { output.GetDesc() } {
                Ok(desc) => desc,
                Err(e) => {
                    log::warn!("IDXGIOutput::GetDesc failed: {e}");
                    continue;
                }
            };
            if desc.Monitor != hmonitor {
                continue;
            }
            // The window's monitor matched, so a failure past this point is a
            // real anomaly. `IDXGIOutput6` (DXGI 1.6 / Win10 1703+) is required
            // for `GetDesc1`.
            let output6 = match output.cast::<Dxgi::IDXGIOutput6>() {
                Ok(output6) => output6,
                Err(e) => {
                    log::warn!("Casting to IDXGIOutput6 failed: {e}");
                    return None;
                }
            };
            // SAFETY: `output6` is live (from the cast above); `GetDesc1`
            // returns a binding-owned `DXGI_OUTPUT_DESC1` by value.
            return match unsafe { output6.GetDesc1() } {
                Ok(desc1) => Some(desc1),
                Err(e) => {
                    log::warn!("IDXGIOutput6::GetDesc1 failed: {e}");
                    None
                }
            };
        }
    }
    log::warn!("No DXGI output matches the window's monitor");
    None
}
