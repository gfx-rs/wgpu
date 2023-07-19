use winapi::{
    shared::{dxgi, dxgi1_2, dxgi1_4, dxgi1_6, winerror},
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

pub fn enumerate_adapters(factory: d3d12::DxgiFactory) -> Vec<d3d12::DxgiAdapter> {
    let mut adapters = Vec::with_capacity(8);

    for cur_index in 0.. {
        if let Some(factory6) = factory.as_factory6() {
            profiling::scope!("IDXGIFactory6::EnumAdapterByGpuPreference");
            // We're already at dxgi1.6, we can grab IDXGIAdapater4 directly
            let mut adapter4 = d3d12::ComPtr::<dxgi1_6::IDXGIAdapter4>::null();
            let hr = unsafe {
                factory6.EnumAdapterByGpuPreference(
                    cur_index,
                    dxgi1_6::DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                    &dxgi1_6::IDXGIAdapter4::uuidof(),
                    adapter4.mut_void(),
                )
            };

            if hr == winerror::DXGI_ERROR_NOT_FOUND {
                break;
            }
            if let Err(err) = hr.into_result() {
                log::error!("Failed enumerating adapters: {}", err);
                break;
            }

            adapters.push(d3d12::DxgiAdapter::Adapter4(adapter4));
            continue;
        }

        profiling::scope!("IDXGIFactory1::EnumAdapters1");
        let mut adapter1 = d3d12::ComPtr::<dxgi::IDXGIAdapter1>::null();
        let hr = unsafe { factory.EnumAdapters1(cur_index, adapter1.mut_self()) };

        if hr == winerror::DXGI_ERROR_NOT_FOUND {
            break;
        }
        if let Err(err) = hr.into_result() {
            log::error!("Failed enumerating adapters: {}", err);
            break;
        }

        // Do the most aggressive casts first, skipping Adpater4 as we definitely don't have dxgi1_6.

        // Adapter1 -> Adapter3
        unsafe {
            match adapter1.cast::<dxgi1_4::IDXGIAdapter3>().into_result() {
                Ok(adapter3) => {
                    adapters.push(d3d12::DxgiAdapter::Adapter3(adapter3));
                    continue;
                }
                Err(err) => {
                    log::info!("Failed casting Adapter1 to Adapter3: {}", err);
                }
            }
        }

        // Adapter1 -> Adapter2
        unsafe {
            match adapter1.cast::<dxgi1_2::IDXGIAdapter2>().into_result() {
                Ok(adapter2) => {
                    adapters.push(d3d12::DxgiAdapter::Adapter2(adapter2));
                    continue;
                }
                Err(err) => {
                    log::info!("Failed casting Adapter1 to Adapter2: {}", err);
                }
            }
        }

        adapters.push(d3d12::DxgiAdapter::Adapter1(adapter1));
    }

    adapters
}

/// Tries to create a IDXGIFactory6, then a IDXGIFactory4, then a IDXGIFactory2, then a IDXGIFactory1,
/// returning the one that succeeds, or if the required_factory_type fails to be
/// created.
pub fn create_factory(
    required_factory_type: DxgiFactoryType,
    instance_flags: crate::InstanceFlags,
) -> Result<(d3d12::DxgiLib, d3d12::DxgiFactory), crate::InstanceError> {
    let lib_dxgi = d3d12::DxgiLib::new().map_err(|_| crate::InstanceError)?;

    let mut factory_flags = d3d12::FactoryCreationFlags::empty();

    if instance_flags.contains(crate::InstanceFlags::VALIDATION) {
        // The `DXGI_CREATE_FACTORY_DEBUG` flag is only allowed to be passed to
        // `CreateDXGIFactory2` if the debug interface is actually available. So
        // we check for whether it exists first.
        match lib_dxgi.get_debug_interface1() {
            Ok(pair) => match pair.into_result() {
                Ok(_debug_controller) => {
                    factory_flags |= d3d12::FactoryCreationFlags::DEBUG;
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
                return Ok((lib_dxgi, d3d12::DxgiFactory::Factory6(factory6)));
            }
            // If we require factory6, hard error.
            Err(err) if required_factory_type == DxgiFactoryType::Factory6 => {
                log::warn!("Failed to cast IDXGIFactory4 to IDXGIFactory6: {:?}", err);
                return Err(crate::InstanceError);
            }
            // If we don't print it to info.
            Err(err) => {
                log::info!("Failed to cast IDXGIFactory4 to IDXGIFactory6: {:?}", err);
                return Ok((lib_dxgi, d3d12::DxgiFactory::Factory4(factory4)));
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
            return Ok((lib_dxgi, d3d12::DxgiFactory::Factory2(factory2)));
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
    Ok((lib_dxgi, d3d12::DxgiFactory::Factory1(factory1)))
}
