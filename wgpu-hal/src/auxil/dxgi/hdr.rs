//! Reading a window's [`wgt::DisplayHdrInfo`] through DXGI.
//!
//! Lives in `auxil/dxgi` so the DX12 and Vulkan-on-Windows backends share it:
//! both query through a [`DxgiHdrSource`], so the same monitor reports identical
//! numbers under either backend.

use parking_lot::Mutex;
use windows::{
    core::Interface as _,
    Win32::{
        Foundation::HWND,
        Graphics::{Dxgi, Gdi},
    },
};

/// The primary colors of a color space, in CIE 1931 xy:
/// `[[red_x, red_y], [green_x, green_y], [blue_x, blue_y]]`.
type Primaries = [[f32; 2]; 3];

/// Reference primaries (CIE 1931 xy: red, green, blue) for the three coarse
/// gamut buckets, used to classify a display's reported primaries.
const REC709: Primaries = [[0.640, 0.330], [0.300, 0.600], [0.150, 0.060]];
const DISPLAY_P3: Primaries = [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]];
const REC2020: Primaries = [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]];

/// Classifies reported primaries into the closest coarse [`wgt::DisplayGamut`]
/// bucket (least-squares over the R/G/B chromaticities). Advisory only.
///
/// Returns `None` if every primary is zeroed (the EDID reported nothing usable),
/// so a bogus all-zero descriptor is not misclassified as sRGB.
fn classify_gamut(red: [f32; 2], green: [f32; 2], blue: [f32; 2]) -> Option<wgt::DisplayGamut> {
    let measured = [red, green, blue];
    if measured.iter().all(|p| p[0] == 0.0 && p[1] == 0.0) {
        return None;
    }
    let dist = |reference: &Primaries| -> f32 {
        measured
            .iter()
            .zip(reference)
            .map(|(m, r)| {
                let dx = m[0] - r[0];
                let dy = m[1] - r[1];
                dx * dx + dy * dy
            })
            .sum()
    };
    [
        (wgt::DisplayGamut::Srgb, dist(&REC709)),
        (wgt::DisplayGamut::DisplayP3, dist(&DISPLAY_P3)),
        (wgt::DisplayGamut::Rec2020, dist(&REC2020)),
    ]
    .into_iter()
    .min_by(|a, b| a.1.total_cmp(&b.1))
    .map(|(gamut, _)| gamut)
}

/// Maps an [`IDXGIOutput6::GetDesc1`] result and the SDR white level (from
/// [`sdr_white_nits_for_monitor`]) into the backend-agnostic
/// [`wgt::DisplayHdrInfo`].
///
/// Pure: it interprets `desc1` without any system call. Numeric values are
/// advisory (EDID-sourced); see [`wgt::DisplayLuminance`].
///
/// [`IDXGIOutput6::GetDesc1`]: windows::Win32::Graphics::Dxgi::IDXGIOutput6::GetDesc1
fn display_hdr_info_from_desc1(
    desc1: &Dxgi::DXGI_OUTPUT_DESC1,
    sdr_white_nits: Option<f32>,
) -> wgt::DisplayHdrInfo {
    // The output is in an HDR color space if its `ColorSpace` is one of the two
    // Windows uses for HDR swap-chains: HDR10 / PQ (`G2084_NONE_P2020`) or
    // scRGB / linear-extended-sRGB (`G10_NONE_P709`). Checking both ensures scRGB
    // HDR is not misreported as SDR.
    let high_dynamic_range = desc1.ColorSpace
        == Dxgi::Common::DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020
        || desc1.ColorSpace == Dxgi::Common::DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709;

    let luminance = wgt::DisplayLuminance {
        max_nits: Some(desc1.MaxLuminance),
        max_full_frame_nits: Some(desc1.MaxFullFrameLuminance),
        min_nits: Some(desc1.MinLuminance),
        sdr_white_nits,
    };

    let chromaticity = wgt::DisplayChromaticity {
        red: Some(desc1.RedPrimary),
        green: Some(desc1.GreenPrimary),
        blue: Some(desc1.BluePrimary),
        white: Some(desc1.WhitePoint),
    };

    let coarse = wgt::DisplayCoarseRange {
        high_dynamic_range: Some(high_dynamic_range),
        gamut: classify_gamut(desc1.RedPrimary, desc1.GreenPrimary, desc1.BluePrimary),
    };

    // `headroom` is `None`: DXGI exposes absolute luminance (captured in
    // `luminance`) rather than a relative headroom ratio. `bits_per_color` drops
    // values that don't fit `u8`.
    wgt::DisplayHdrInfo {
        luminance: Some(luminance),
        headroom: None,
        chromaticity: Some(chromaticity),
        coarse: Some(coarse),
        bits_per_color: u8::try_from(desc1.BitsPerColor).ok(),
    }
}

/// A window's display HDR info, read through DXGI.
///
/// Bundles the window's `HWND` with the DXGI factory used to query it. The factory
/// is built once and reused, rebuilt only when DXGI reports it stale; the monitor
/// lookup and adapter/output walk still run every call, since the window can move
/// between displays and the SDR white level changes at runtime.
pub struct DxgiHdrSource {
    hwnd: HWND,
    factory: Mutex<Option<Dxgi::IDXGIFactory1>>,
}

// SAFETY: `HWND` and `IDXGIFactory1` are `!Send`/`!Sync` in windows-rs. The `HWND`
// is only ever read to run a DXGI query (fine from any thread), and the factory is
// free-threaded for the enumeration done here with the `Mutex` serializing access.
// This mirrors the manual `Send`/`Sync` on `dx12::Surface`.
unsafe impl Send for DxgiHdrSource {}
unsafe impl Sync for DxgiHdrSource {}

impl DxgiHdrSource {
    /// Creates a source for the window identified by `hwnd`. The `HWND` is
    /// borrowed; using the source after that window is destroyed is undefined
    /// behavior.
    pub fn new(hwnd: HWND) -> Self {
        Self {
            hwnd,
            factory: Mutex::new(None),
        }
    }

    /// Reads the window's [`wgt::DisplayHdrInfo`], or `None` if its monitor can't
    /// be identified or queried (a headless or composition window, a
    /// pre-Win10-1703 system without `IDXGIOutput6`, or a COM failure). Never
    /// panics.
    pub fn display_hdr_info(&self) -> Option<wgt::DisplayHdrInfo> {
        let desc1 = self.output_desc1()?;
        let sdr_white_nits = sdr_white_nits_for_monitor(desc1.Monitor);
        Some(display_hdr_info_from_desc1(&desc1, sdr_white_nits))
    }

    /// The [`DXGI_OUTPUT_DESC1`] for the window's monitor, reusing the cached
    /// factory unless DXGI reports it stale.
    ///
    /// [`DXGI_OUTPUT_DESC1`]: Dxgi::DXGI_OUTPUT_DESC1
    fn output_desc1(&self) -> Option<Dxgi::DXGI_OUTPUT_DESC1> {
        let mut factory = self.factory.lock();

        // A factory only enumerates the adapter/output topology it saw at
        // creation, so rebuild it once it reports `IsCurrent() == false` (a
        // monitor or GPU was plugged or unplugged).
        // SAFETY: a cached factory is live; `IsCurrent` takes no caller pointers.
        let need_new = factory
            .as_ref()
            .is_none_or(|f| !unsafe { f.IsCurrent() }.as_bool());
        if need_new {
            // SAFETY: `CreateDXGIFactory1` takes no caller pointers; the `windows`
            // binding fills the interface out-pointer itself.
            match unsafe { Dxgi::CreateDXGIFactory1() } {
                Ok(new) => *factory = Some(new),
                Err(e) => {
                    log::warn!("CreateDXGIFactory1 failed: {e}");
                    return None;
                }
            }
        }
        output_desc1_from_factory(factory.as_ref()?, self.hwnd)
    }
}

/// The [`DXGI_OUTPUT_DESC1`] for the monitor backing `wnd_handle`, found by
/// walking every adapter's outputs on `factory` — not just the rendering
/// adapter's, so the monitor is found on hybrid-GPU systems where the window sits
/// on a display wired to a different adapter. `None` if none match.
///
/// [`DXGI_OUTPUT_DESC1`]: Dxgi::DXGI_OUTPUT_DESC1
fn output_desc1_from_factory(
    factory: &Dxgi::IDXGIFactory1,
    wnd_handle: HWND,
) -> Option<Dxgi::DXGI_OUTPUT_DESC1> {
    // SAFETY: `MonitorFromWindow` is sound for any `HWND`; an invalid one yields
    // a null `HMONITOR`, checked below.
    let hmonitor = unsafe { Gdi::MonitorFromWindow(wnd_handle, Gdi::MONITOR_DEFAULTTONEAREST) };
    if hmonitor.is_invalid() {
        log::warn!("MonitorFromWindow failed; cannot identify the window's output");
        return None;
    }
    for adapter_index in 0.. {
        // SAFETY: `factory` is live; `EnumAdapters1` takes only an index.
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

/// Best-effort SDR white level in nits for `hmonitor`, via the Windows
/// DisplayConfig API. Tracks the brightness slider, so it changes at runtime.
///
/// This is the only [`wgt::DisplayLuminance`] field `GetDesc1` doesn't carry; it
/// bridges absolute nits and relative headroom. It lives in
/// `DISPLAYCONFIG_SDR_WHITE_LEVEL`, outside DXGI, so it needs its own query.
///
/// Returns `None` on any failure (no match, query error, or a `0` reading). Never
/// panics. Pass [`Dxgi::DXGI_OUTPUT_DESC1::Monitor`] so the value matches the rest
/// of the `DisplayHdrInfo`.
fn sdr_white_nits_for_monitor(hmonitor: Gdi::HMONITOR) -> Option<f32> {
    use windows::Win32::{
        Devices::Display::{
            DisplayConfigGetDeviceInfo, GetDisplayConfigBufferSizes, QueryDisplayConfig,
            DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL,
            DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME, DISPLAYCONFIG_MODE_INFO,
            DISPLAYCONFIG_PATH_INFO, DISPLAYCONFIG_SDR_WHITE_LEVEL,
            DISPLAYCONFIG_SOURCE_DEVICE_NAME, QDC_ONLY_ACTIVE_PATHS,
        },
        Foundation::ERROR_SUCCESS,
        Graphics::Gdi::{GetMonitorInfoW, MONITORINFO, MONITORINFOEXW},
    };

    // 1. HMONITOR -> GDI device name (e.g. `\\.\DISPLAY1`).
    let mut monitor_info = MONITORINFOEXW::default();
    monitor_info.monitorInfo.cbSize = size_of::<MONITORINFOEXW>() as u32;
    // `GetMonitorInfoW` takes a `*mut MONITORINFO`; the `EXW` variant is
    // layout-compatible (it starts with `MONITORINFO`) and reports `cbSize`.
    let lpmi = core::ptr::from_mut(&mut monitor_info).cast::<MONITORINFO>();
    // SAFETY: `lpmi` points at a live `MONITORINFOEXW` with `cbSize` set, which
    // bounds the write; an invalid `hmonitor` just fails (handled below).
    let got_info = unsafe { GetMonitorInfoW(hmonitor, lpmi) };
    if !got_info.as_bool() {
        return None;
    }
    let gdi_device_name = monitor_info.szDevice;

    // 2. Enumerate the active DisplayConfig paths.
    let mut path_count = 0u32;
    let mut mode_count = 0u32;
    // SAFETY: both out-params are live `u32`s; the call only writes the required
    // buffer sizes into them.
    if unsafe {
        GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &mut path_count, &mut mode_count)
    } != ERROR_SUCCESS
    {
        return None;
    }
    let mut paths = alloc::vec![DISPLAYCONFIG_PATH_INFO::default(); path_count as usize];
    let mut modes = alloc::vec![DISPLAYCONFIG_MODE_INFO::default(); mode_count as usize];
    // SAFETY: `paths`/`modes` are sized to the counts reported above and passed
    // with those counts, bounding the writes.
    if unsafe {
        QueryDisplayConfig(
            QDC_ONLY_ACTIVE_PATHS,
            &mut path_count,
            paths.as_mut_ptr(),
            &mut mode_count,
            modes.as_mut_ptr(),
            None,
        )
    } != ERROR_SUCCESS
    {
        return None;
    }

    // 3. Match the path whose source is this monitor, then read its target's
    //    SDR white level. `DisplayConfigGetDeviceInfo` returns a WIN32 code as
    //    `i32` (`0` == `ERROR_SUCCESS`).
    for path in paths.iter().take(path_count as usize) {
        let mut source = DISPLAYCONFIG_SOURCE_DEVICE_NAME::default();
        source.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
        source.header.size = size_of::<DISPLAYCONFIG_SOURCE_DEVICE_NAME>() as u32;
        source.header.adapterId = path.sourceInfo.adapterId;
        source.header.id = path.sourceInfo.id;
        // SAFETY: `source` is live with `header.type`/`header.size` set, which
        // bounds the call's access to that struct.
        if unsafe { DisplayConfigGetDeviceInfo(&mut source.header) } != ERROR_SUCCESS.0 as i32 {
            continue;
        }
        if source.viewGdiDeviceName != gdi_device_name {
            continue;
        }

        let mut white = DISPLAYCONFIG_SDR_WHITE_LEVEL::default();
        white.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL;
        white.header.size = size_of::<DISPLAYCONFIG_SDR_WHITE_LEVEL>() as u32;
        white.header.adapterId = path.targetInfo.adapterId;
        white.header.id = path.targetInfo.id;
        // SAFETY: `white` is live with `header.type`/`header.size` set, which
        // bounds the call's access to that struct.
        if unsafe { DisplayConfigGetDeviceInfo(&mut white.header) } != ERROR_SUCCESS.0 as i32 {
            return None;
        }
        // `SDRWhiteLevel` encodes nits as `(nits / 80) * 1000`; `0` means the OS
        // reported nothing usable.
        return (white.SDRWhiteLevel > 0).then(|| white.SDRWhiteLevel as f32 * 80.0 / 1000.0);
    }
    None
}
