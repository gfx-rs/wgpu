use std::borrow::Cow;

use winapi::shared::winerror;

pub(crate) trait HResult<O> {
    fn into_result(self) -> Result<O, Cow<'static, str>>;
    fn into_device_result(self, description: &str) -> Result<O, crate::DeviceError>;
}
impl HResult<()> for i32 {
    fn into_result(self) -> Result<(), Cow<'static, str>> {
        if self >= 0 {
            return Ok(());
        }
        let description = match self {
            winerror::E_UNEXPECTED => "unexpected",
            winerror::E_NOTIMPL => "not implemented",
            winerror::E_OUTOFMEMORY => "out of memory",
            winerror::E_INVALIDARG => "invalid argument",
            _ => return Err(Cow::Owned(format!("0x{:X}", self as u32))),
        };
        Err(Cow::Borrowed(description))
    }
    fn into_device_result(self, description: &str) -> Result<(), crate::DeviceError> {
        #![allow(unreachable_code)]

        self.into_result().map_err(|err| {
            log::error!("{} failed: {}", description, err);

            match self {
                winerror::E_OUTOFMEMORY => {
                    #[cfg(feature = "oom_panic")]
                    panic!("{description} failed: Out of memory");
                }
                winerror::DXGI_ERROR_DEVICE_RESET | winerror::DXGI_ERROR_DEVICE_REMOVED => {
                    #[cfg(feature = "device_lost_panic")]
                    panic!("{description} failed: Device lost ({err})");
                }
                _ => {
                    #[cfg(feature = "internal_error_panic")]
                    panic!("{description} failed: {err}");
                }
            }

            if self == winerror::E_OUTOFMEMORY {
                crate::DeviceError::OutOfMemory
            } else {
                crate::DeviceError::Lost
            }
        })
    }
}

impl<T> HResult<T> for (T, i32) {
    fn into_result(self) -> Result<T, Cow<'static, str>> {
        self.1.into_result().map(|()| self.0)
    }
    fn into_device_result(self, description: &str) -> Result<T, crate::DeviceError> {
        self.1.into_device_result(description).map(|()| self.0)
    }
}
