#[derive(Debug)]
pub struct DynLib {
    inner: libloading::Library,
}

impl DynLib {
    pub unsafe fn new<P>(filename: P) -> Result<Self, libloading::Error>
    where
        P: AsRef<std::ffi::OsStr>,
    {
        unsafe { libloading::Library::new(filename) }.map(|inner| Self { inner })
    }

    pub unsafe fn get<T>(
        &self,
        symbol: &[u8],
    ) -> Result<libloading::Symbol<'_, T>, crate::DeviceError> {
        unsafe { self.inner.get(symbol) }.map_err(|e| match e {
            libloading::Error::GetProcAddress { .. } | libloading::Error::GetProcAddressUnknown => {
                crate::DeviceError::Unexpected
            }
            libloading::Error::IncompatibleSize
            | libloading::Error::CreateCString { .. }
            | libloading::Error::CreateCStringWithTrailing { .. } => crate::hal_internal_error(e),
            _ => crate::DeviceError::Unexpected, // could be unreachable!() but we prefer to be more robust
        })
    }
}
