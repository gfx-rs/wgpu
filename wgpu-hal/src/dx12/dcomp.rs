use windows::Win32::{
    Foundation::HWND,
    Graphics::{Direct3D11, Direct3D11on12, DirectComposition, Dxgi},
};
use windows_core::Interface as _;

#[derive(Default)]
pub struct DCompState {
    inner: Option<InnerState>,
}

impl DCompState {
    pub fn get_or_init(
        &mut self,
        hwnd: &HWND,
        device: &super::Device,
    ) -> Result<&mut InnerState, crate::SurfaceError> {
        if self.inner.is_none() {
            self.inner = Some(unsafe { InnerState::init(hwnd, device) }?);
        }
        Ok(self.inner.as_mut().unwrap())
    }
}

pub struct InnerState {
    pub visual: DirectComposition::IDCompositionVisual,
    pub device: DirectComposition::IDCompositionDevice,
    // Must be kept alive but is otherwise unused after initialization.
    pub _target: DirectComposition::IDCompositionTarget,
}

impl InnerState {
    /// Creates a DirectComposition device and a target for the given window handle.
    /// From a Direct3D 12 device, it creates a Direct3D 11 device and then a DirectComposition device.
    pub unsafe fn init(hwnd: &HWND, device: &super::Device) -> Result<Self, crate::SurfaceError> {
        let mut d3d11_device = None;
        {
            profiling::scope!("Direct3D11on12::D3D11On12CreateDevice");
            unsafe {
                Direct3D11on12::D3D11On12CreateDevice(
                    &device.raw,
                    Direct3D11::D3D11_CREATE_DEVICE_BGRA_SUPPORT.0,
                    None,
                    None,
                    0,
                    Some(&mut d3d11_device),
                    None,
                    None,
                )
            }
            .map_err(|err| {
                log::error!("Direct3D11on12::D3D11On12CreateDevice failed: {err}");
                crate::SurfaceError::Other("Direct3D11on12::D3D11On12CreateDevice")
            })?;
        }
        let d3d11_device = d3d11_device.unwrap();

        let dxgi_device = {
            profiling::scope!("IDXGIDevice::QueryInterface");
            d3d11_device.cast::<Dxgi::IDXGIDevice>().map_err(|err| {
                log::error!("IDXGIDevice::QueryInterface failed: {err}");
                crate::SurfaceError::Other("IDXGIDevice::QueryInterface")
            })?
        };

        let dcomp_device = {
            profiling::scope!("DirectComposition::DCompositionCreateDevice");
            unsafe {
                DirectComposition::DCompositionCreateDevice::<
                    _,
                    DirectComposition::IDCompositionDevice,
                >(&dxgi_device)
            }
            .map_err(|err| {
                log::error!("DirectComposition::DCompositionCreateDevice failed: {err}");
                crate::SurfaceError::Other("DirectComposition::DCompositionCreateDevice")
            })?
        };

        let target = {
            profiling::scope!("IDCompositionDevice::CreateTargetForHwnd");
            unsafe { dcomp_device.CreateTargetForHwnd(*hwnd, false) }.map_err(|err| {
                log::error!("IDCompositionDevice::CreateTargetForHwnd failed: {err}");
                crate::SurfaceError::Other("IDCompositionDevice::CreateTargetForHwnd")
            })?
        };

        let visual = {
            profiling::scope!("IDCompositionDevice::CreateVisual");
            unsafe { dcomp_device.CreateVisual() }.map_err(|err| {
                log::error!("IDCompositionDevice::CreateVisual failed: {err}");
                crate::SurfaceError::Other("IDCompositionDevice::CreateVisual")
            })?
        };

        {
            profiling::scope!("IDCompositionTarget::SetRoot");
            unsafe { target.SetRoot(&visual) }.map_err(|err| {
                log::error!("IDCompositionTarget::SetRoot failed: {err}");
                crate::SurfaceError::Other("IDCompositionTarget::SetRoot")
            })?;
        }

        Ok(InnerState {
            visual,
            device: dcomp_device,
            _target: target,
        })
    }
}
