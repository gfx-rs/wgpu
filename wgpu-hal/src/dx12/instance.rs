use super::HResult as _;
use std::{borrow::Cow, slice, sync::Arc};
use winapi::{
    shared::{dxgi, dxgi1_2, dxgi1_6, winerror},
    um::{errhandlingapi, winnt},
    vc::excpt,
    Interface,
};

const MESSAGE_PREFIXES: &[(&str, log::Level)] = &[
    ("CORRUPTION", log::Level::Error),
    ("ERROR", log::Level::Error),
    ("WARNING", log::Level::Warn),
    ("INFO", log::Level::Info),
    ("MESSAGE", log::Level::Debug),
];

unsafe extern "system" fn output_debug_string_handler(
    exception_info: *mut winnt::EXCEPTION_POINTERS,
) -> i32 {
    // See https://stackoverflow.com/a/41480827
    let record = &*(*exception_info).ExceptionRecord;
    if record.NumberParameters != 2 {
        return excpt::EXCEPTION_CONTINUE_SEARCH;
    }
    let message = match record.ExceptionCode {
        winnt::DBG_PRINTEXCEPTION_C => String::from_utf8_lossy(slice::from_raw_parts(
            record.ExceptionInformation[1] as *const u8,
            record.ExceptionInformation[0],
        )),
        winnt::DBG_PRINTEXCEPTION_WIDE_C => {
            Cow::Owned(String::from_utf16_lossy(slice::from_raw_parts(
                record.ExceptionInformation[1] as *const u16,
                record.ExceptionInformation[0],
            )))
        }
        _ => return excpt::EXCEPTION_CONTINUE_SEARCH,
    };

    let message = match message.strip_prefix("D3D12 ") {
        Some(msg) => msg
            .trim_end_matches("\n\0")
            .trim_end_matches("[ STATE_CREATION WARNING #0: UNKNOWN]"),
        None => return excpt::EXCEPTION_CONTINUE_SEARCH,
    };

    let (message, level) = match MESSAGE_PREFIXES
        .iter()
        .find(|&&(prefix, _)| message.starts_with(prefix))
    {
        Some(&(prefix, level)) => (&message[prefix.len() + 2..], level),
        None => (message, log::Level::Debug),
    };

    if level == log::Level::Warn && message.contains("#82") {
        // This is are useless spammy warnings (#820, #821):
        // "The application did not pass any clear value to resource creation"
        return excpt::EXCEPTION_CONTINUE_SEARCH;
    }

    log::log!(level, "{}", message);

    if cfg!(debug_assertions) && level == log::Level::Error {
        // Panicking behind FFI is UB, so we just exit.
        std::process::exit(1);
    }

    excpt::EXCEPTION_CONTINUE_EXECUTION
}

impl Drop for super::Instance {
    fn drop(&mut self) {
        unsafe {
            self.factory.destroy();
            errhandlingapi::RemoveVectoredExceptionHandler(output_debug_string_handler as *mut _);
        }
    }
}

impl crate::Instance<super::Api> for super::Instance {
    unsafe fn init(desc: &crate::InstanceDescriptor) -> Result<Self, crate::InstanceError> {
        let lib_main = native::D3D12Lib::new().map_err(|_| crate::InstanceError)?;

        let lib_dxgi = native::DxgiLib::new().map_err(|_| crate::InstanceError)?;
        let mut factory_flags = native::FactoryCreationFlags::empty();

        if desc.flags.contains(crate::InstanceFlags::VALIDATION) {
            // Enable debug layer
            match lib_main.get_debug_interface() {
                Ok(pair) => match pair.into_result() {
                    Ok(debug_controller) => {
                        debug_controller.enable_layer();
                        debug_controller.Release();
                    }
                    Err(err) => {
                        log::warn!("Unable to enable D3D12 debug interface: {}", err);
                    }
                },
                Err(err) => {
                    log::warn!("Debug interface function for D3D12 not found: {:?}", err);
                }
            }

            // The `DXGI_CREATE_FACTORY_DEBUG` flag is only allowed to be passed to
            // `CreateDXGIFactory2` if the debug interface is actually available. So
            // we check for whether it exists first.
            match lib_dxgi.get_debug_interface1() {
                Ok(pair) => match pair.into_result() {
                    Ok(debug_controller) => {
                        debug_controller.destroy();
                        factory_flags |= native::FactoryCreationFlags::DEBUG;
                    }
                    Err(err) => {
                        log::warn!("Unable to enable DXGI debug interface: {}", err);
                    }
                },
                Err(err) => {
                    log::warn!("Debug interface function for DXGI not found: {:?}", err);
                }
            }

            // Intercept `OutputDebugString` calls
            errhandlingapi::AddVectoredExceptionHandler(0, Some(output_debug_string_handler));
        }

        // Create DXGI factory
        let factory = match lib_dxgi.create_factory2(factory_flags) {
            Ok(pair) => match pair.into_result() {
                Ok(factory) => factory,
                Err(err) => {
                    log::warn!("Failed to create DXGI factory: {}", err);
                    return Err(crate::InstanceError);
                }
            },
            Err(err) => {
                log::warn!("Factory creation function for DXGI not found: {:?}", err);
                return Err(crate::InstanceError);
            }
        };

        Ok(Self {
            factory,
            library: Arc::new(lib_main),
            _lib_dxgi: lib_dxgi,
            flags: desc.flags,
        })
    }

    unsafe fn create_surface(
        &self,
        has_handle: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Result<super::Surface, crate::InstanceError> {
        match has_handle.raw_window_handle() {
            raw_window_handle::RawWindowHandle::Windows(handle) => Ok(super::Surface {
                factory: self.factory,
                wnd_handle: handle.hwnd as *mut _,
                swap_chain: None,
            }),
            _ => Err(crate::InstanceError),
        }
    }
    unsafe fn destroy_surface(&self, _surface: super::Surface) {
        // just drop
    }

    unsafe fn enumerate_adapters(&self) -> Vec<crate::ExposedAdapter<super::Api>> {
        // Try to use high performance order by default (returns None on Windows < 1803)
        let factory6 = match self.factory.cast::<dxgi1_6::IDXGIFactory6>().into_result() {
            Ok(f6) => {
                // It's okay to decrement the refcount here because we
                // have another reference to the factory already owned by `self`.
                f6.destroy();
                Some(f6)
            }
            Err(err) => {
                log::info!("Failed to cast DXGI to 1.6: {}", err);
                None
            }
        };

        // Enumerate adapters
        let mut adapters = Vec::new();
        for cur_index in 0.. {
            let raw = match factory6 {
                Some(factory) => {
                    profiling::scope!("IDXGIFactory6::EnumAdapterByGpuPreference");
                    let mut adapter2 = native::WeakPtr::<dxgi1_2::IDXGIAdapter2>::null();
                    let hr = factory.EnumAdapterByGpuPreference(
                        cur_index,
                        dxgi1_6::DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                        &dxgi1_2::IDXGIAdapter2::uuidof(),
                        adapter2.mut_void(),
                    );

                    if hr == winerror::DXGI_ERROR_NOT_FOUND {
                        break;
                    }
                    if let Err(err) = hr.into_result() {
                        log::error!("Failed enumerating adapters: {}", err);
                        break;
                    }

                    adapter2
                }
                None => {
                    profiling::scope!("IDXGIFactory1::EnumAdapters1");
                    let mut adapter1 = native::WeakPtr::<dxgi::IDXGIAdapter1>::null();
                    let hr = self
                        .factory
                        .EnumAdapters1(cur_index, adapter1.mut_void() as *mut *mut _);

                    if hr == winerror::DXGI_ERROR_NOT_FOUND {
                        break;
                    }
                    if let Err(err) = hr.into_result() {
                        log::error!("Failed enumerating adapters: {}", err);
                        break;
                    }

                    match adapter1.cast::<dxgi1_2::IDXGIAdapter2>().into_result() {
                        Ok(adapter2) => {
                            adapter1.destroy();
                            adapter2
                        }
                        Err(err) => {
                            log::error!("Failed casting to Adapter2: {}", err);
                            break;
                        }
                    }
                }
            };

            adapters.extend(super::Adapter::expose(raw, &self.library, self.flags));
        }
        adapters
    }
}
