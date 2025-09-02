use windows::Win32::{Foundation::HWND, Graphics::DirectComposition};

#[derive(Default)]
pub struct DCompState {
    inner: Option<InnerState>,
}

impl DCompState {
    /// This will create a DirectComposition device and a target for the window handle if not already initialized.
    /// If the device is already initialized, it will return the existing state.
    pub unsafe fn get_or_init(
        &mut self,
        hwnd: &HWND,
    ) -> Result<&mut InnerState, crate::SurfaceError> {
        if self.inner.is_none() {
            self.inner = Some(unsafe { InnerState::init(hwnd) }?);
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
    pub unsafe fn init(hwnd: &HWND) -> Result<Self, crate::SurfaceError> {
        profiling::scope!("DCompState::init");
        let dcomp_device: DirectComposition::IDCompositionDevice = {
            unsafe { DirectComposition::DCompositionCreateDevice2(None) }.map_err(|err| {
                log::error!("DirectComposition::DCompositionCreateDevice failed: {err}");
                crate::SurfaceError::Other("DirectComposition::DCompositionCreateDevice")
            })?
        };

        let target = unsafe { dcomp_device.CreateTargetForHwnd(*hwnd, false) }.map_err(|err| {
            log::error!("IDCompositionDevice::CreateTargetForHwnd failed: {err}");
            crate::SurfaceError::Other("IDCompositionDevice::CreateTargetForHwnd")
        })?;

        let visual = unsafe { dcomp_device.CreateVisual() }.map_err(|err| {
            log::error!("IDCompositionDevice::CreateVisual failed: {err}");
            crate::SurfaceError::Other("IDCompositionDevice::CreateVisual")
        })?;

        unsafe { target.SetRoot(&visual) }.map_err(|err| {
            log::error!("IDCompositionTarget::SetRoot failed: {err}");
            crate::SurfaceError::Other("IDCompositionTarget::SetRoot")
        })?;

        Ok(InnerState {
            visual,
            device: dcomp_device,
            _target: target,
        })
    }
}
