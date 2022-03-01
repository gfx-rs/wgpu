use winapi::{
    shared::{dxgi, dxgi1_2, dxgi1_4, dxgi1_5, dxgi1_6, winerror},
    Interface,
};

use super::result::HResult as _;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DxgiFactoryType {
    Factory1,
    Factory2,
    Factory4,
    Factory6,
}

#[derive(Copy, Clone)]
pub enum DxgiFactory {
    Factory1(native::Factory1),
    Factory2(native::Factory2),
    Factory4(native::Factory4),
    Factory6(native::Factory6),
}

impl DxgiFactory {
    pub fn destroy(&self) {
        match *self {
            DxgiFactory::Factory1(f) => unsafe { f.destroy() },
            DxgiFactory::Factory2(f) => unsafe { f.destroy() },
            DxgiFactory::Factory4(f) => unsafe { f.destroy() },
            DxgiFactory::Factory6(f) => unsafe { f.destroy() },
        }
    }

    #[track_caller]
    pub fn as_factory6(self) -> Option<native::Factory6> {
        match self {
            DxgiFactory::Factory1(_) => None,
            DxgiFactory::Factory2(_) => None,
            DxgiFactory::Factory4(_) => None,
            DxgiFactory::Factory6(f) => Some(f),
        }
    }

    #[track_caller]
    pub fn as_factory1(self) -> native::Factory1 {
        match self {
            DxgiFactory::Factory1(f) => f,
            DxgiFactory::Factory2(f) => unsafe {
                native::Factory1::from_raw(f.as_mut_ptr() as *mut dxgi::IDXGIFactory1)
            },
            DxgiFactory::Factory4(f) => unsafe {
                native::Factory1::from_raw(f.as_mut_ptr() as *mut dxgi::IDXGIFactory1)
            },
            DxgiFactory::Factory6(f) => unsafe {
                native::Factory1::from_raw(f.as_mut_ptr() as *mut dxgi::IDXGIFactory1)
            },
        }
    }

    #[track_caller]
    pub fn as_factory2(self) -> Option<native::Factory2> {
        match self {
            DxgiFactory::Factory1(_) => None,
            DxgiFactory::Factory2(f) => Some(f),
            DxgiFactory::Factory4(f) => unsafe {
                Some(native::Factory2::from_raw(
                    f.as_mut_ptr() as *mut dxgi1_2::IDXGIFactory2
                ))
            },
            DxgiFactory::Factory6(f) => unsafe {
                Some(native::Factory2::from_raw(
                    f.as_mut_ptr() as *mut dxgi1_2::IDXGIFactory2
                ))
            },
        }
    }

    #[track_caller]
    pub fn as_factory5(self) -> Option<native::WeakPtr<dxgi1_5::IDXGIFactory5>> {
        match self {
            DxgiFactory::Factory1(_) => None,
            DxgiFactory::Factory2(_) => None,
            DxgiFactory::Factory4(_) => None,
            DxgiFactory::Factory6(f) => unsafe {
                Some(native::WeakPtr::from_raw(
                    f.as_mut_ptr() as *mut dxgi1_5::IDXGIFactory5
                ))
            },
        }
    }

    pub fn enumerate_adapters(&self) -> Vec<native::WeakPtr<dxgi1_2::IDXGIAdapter2>> {
        let factory6 = self.as_factory6();

        let mut adapters = Vec::with_capacity(8);

        for cur_index in 0.. {
            if let Some(factory) = factory6 {
                profiling::scope!("IDXGIFactory6::EnumAdapterByGpuPreference");
                let mut adapter2 = native::WeakPtr::<dxgi1_2::IDXGIAdapter2>::null();
                let hr = unsafe {
                    factory.EnumAdapterByGpuPreference(
                        cur_index,
                        dxgi1_6::DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                        &dxgi1_2::IDXGIAdapter2::uuidof(),
                        adapter2.mut_void(),
                    )
                };

                if hr == winerror::DXGI_ERROR_NOT_FOUND {
                    break;
                }
                if let Err(err) = hr.into_result() {
                    log::error!("Failed enumerating adapters: {}", err);
                    break;
                }

                adapters.push(adapter2);
                continue;
            }

            profiling::scope!("IDXGIFactory1::EnumAdapters1");
            let mut adapter1 = native::WeakPtr::<dxgi::IDXGIAdapter1>::null();
            let hr = unsafe {
                self.as_factory1()
                    .EnumAdapters1(cur_index, adapter1.mut_void() as *mut *mut _)
            };

            if hr == winerror::DXGI_ERROR_NOT_FOUND {
                break;
            }
            if let Err(err) = hr.into_result() {
                log::error!("Failed enumerating adapters: {}", err);
                break;
            }

            match unsafe { adapter1.cast::<dxgi1_2::IDXGIAdapter2>() }.into_result() {
                Ok(adapter2) => {
                    unsafe { adapter1.destroy() };
                    adapters.push(adapter2);
                }
                Err(err) => {
                    log::error!("Failed casting Adapter1 to Adapter2: {}", err);
                    break;
                }
            }
        }

        adapters
    }
}

/// Tries to create a IDXGIFactory6, then a IDXGIFactory4, then a IDXGIFactory2, then a IDXGIFactory1,
/// returning the one that succeeds, or if the required_factory_type fails to be
/// created.
pub fn create_factory(
    required_factory_type: DxgiFactoryType,
    instance_flags: crate::InstanceFlags,
) -> Result<(native::DxgiLib, DxgiFactory), crate::InstanceError> {
    let lib_dxgi = native::DxgiLib::new().map_err(|_| crate::InstanceError)?;

    let mut factory_flags = native::FactoryCreationFlags::empty();

    if instance_flags.contains(crate::InstanceFlags::VALIDATION) {
        // The `DXGI_CREATE_FACTORY_DEBUG` flag is only allowed to be passed to
        // `CreateDXGIFactory2` if the debug interface is actually available. So
        // we check for whether it exists first.
        match lib_dxgi.get_debug_interface1() {
            Ok(pair) => match pair.into_result() {
                Ok(debug_controller) => {
                    unsafe { debug_controller.destroy() };
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
        super::exception::register_exception_handler();
    }

    // Try to create IDXGIFactory4
    let factory4 = match lib_dxgi.create_factory2(factory_flags) {
        Ok(pair) => match pair.into_result() {
            Ok(factory) => Some(factory),
            // We hard error here as we _should have_ been able to make a factory4 but couldn't.
            Err(err) => {
                log::error!("Failed to create IDXGIFactory4: {}", err);
                return Err(crate::InstanceError);
            }
        },
        // If we require factory4, hard error.
        Err(err) if required_factory_type == DxgiFactoryType::Factory4 => {
            log::error!("IDXGIFactory1 creation function not found: {:?}", err);
            return Err(crate::InstanceError);
        }
        // If we don't print it to info as all win7 will hit this case.
        Err(err) => {
            log::info!("IDXGIFactory1 creation function not found: {:?}", err);
            None
        }
    };

    if let Some(factory4) = factory4 {
        //  Try to cast the IDXGIFactory4 into IDXGIFactory6
        let factory6 = unsafe { factory4.cast::<dxgi1_6::IDXGIFactory6>().into_result() };
        match factory6 {
            Ok(factory6) => {
                unsafe {
                    factory4.destroy();
                }
                return Ok((lib_dxgi, DxgiFactory::Factory6(factory6)));
            }
            // If we require factory6, hard error.
            Err(err) if required_factory_type == DxgiFactoryType::Factory6 => {
                log::warn!("Failed to cast IDXGIFactory4 to IDXGIFactory6: {:?}", err);
                return Err(crate::InstanceError);
            }
            // If we don't print it to info.
            Err(err) => {
                log::info!("Failed to cast IDXGIFactory4 to IDXGIFactory6: {:?}", err);
                return Ok((lib_dxgi, DxgiFactory::Factory4(factory4)));
            }
        }
    }

    // Try to create IDXGIFactory1
    let factory1 = match lib_dxgi.create_factory1() {
        Ok(pair) => match pair.into_result() {
            Ok(factory) => factory,
            Err(err) => {
                log::error!("Failed to create IDXGIFactory1: {}", err);
                return Err(crate::InstanceError);
            }
        },
        // We always require at least factory1, so hard error
        Err(err) => {
            log::error!("IDXGIFactory1 creation function not found: {:?}", err);
            return Err(crate::InstanceError);
        }
    };

    // Try to cast the IDXGIFactory1 into IDXGIFactory2
    let factory2 = unsafe { factory1.cast::<dxgi1_2::IDXGIFactory2>().into_result() };
    match factory2 {
        Ok(factory2) => {
            unsafe {
                factory1.destroy();
            }
            return Ok((lib_dxgi, DxgiFactory::Factory2(factory2)));
        }
        // If we require factory2, hard error.
        Err(err) if required_factory_type == DxgiFactoryType::Factory2 => {
            log::warn!("Failed to cast IDXGIFactory1 to IDXGIFactory2: {:?}", err);
            return Err(crate::InstanceError);
        }
        // If we don't print it to info.
        Err(err) => {
            log::info!("Failed to cast IDXGIFactory1 to IDXGIFactory2: {:?}", err);
        }
    }

    // We tried to create 4 and 2, but only succeeded with 1.
    Ok((lib_dxgi, DxgiFactory::Factory1(factory1)))
}
