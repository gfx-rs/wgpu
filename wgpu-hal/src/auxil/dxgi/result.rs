use std::borrow::Cow;

use windows::Win32::{Foundation, Graphics::Dxgi};

pub(crate) trait HResult<O> {
    fn into_result(self) -> Result<O, Cow<'static, str>>;
    fn into_device_result(self, description: &str) -> Result<O, crate::DeviceError>;
}
impl<T> HResult<T> for windows::core::Result<T> {
    fn into_result(self) -> Result<T, Cow<'static, str>> {
        // TODO: use windows-rs built-in error formatting?
        let description = match self {
            Ok(t) => return Ok(t),
            Err(e) if e.code() == Foundation::E_UNEXPECTED => "unexpected",
            Err(e) if e.code() == Foundation::E_NOTIMPL => "not implemented",
            Err(e) if e.code() == Foundation::E_OUTOFMEMORY => "out of memory",
            Err(e) if e.code() == Foundation::E_INVALIDARG => "invalid argument",
            Err(e) => return Err(Cow::Owned(format!("{e:?}"))),
        };
        Err(Cow::Borrowed(description))
    }
    fn into_device_result(self, description: &str) -> Result<T, crate::DeviceError> {
        #![allow(unreachable_code)]

        let err_code = if let Err(err) = &self {
            Some(err.code())
        } else {
            None
        };
        self.into_result().map_err(|err| {
            log::error!("{} failed: {}", description, err);

            let Some(err_code) = err_code else {
                unreachable!()
            };

            match err_code {
                Foundation::E_OUTOFMEMORY => {
                    #[cfg(feature = "oom_panic")]
                    panic!("{description} failed: Out of memory");
                    return crate::DeviceError::OutOfMemory;
                }
                Dxgi::DXGI_ERROR_DEVICE_RESET | Dxgi::DXGI_ERROR_DEVICE_REMOVED => {
                    #[cfg(feature = "device_lost_panic")]
                    panic!("{description} failed: Device lost ({err})");
                }
                _ => {
                    #[cfg(feature = "internal_error_panic")]
                    panic!("{description} failed: {err}");
                }
            }

            crate::DeviceError::Lost
        })
    }
}
