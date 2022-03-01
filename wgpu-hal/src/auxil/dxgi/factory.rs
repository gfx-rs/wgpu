use winapi::shared::dxgi1_2;

use super::result::HResult as _;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DxgiFactoryType {
    Factory1,
    Factory2,
    Factory4,
}

pub enum DxgiFactory {
    Factory1(native::Factory1),
    Factory2(native::Factory2),
    Factory4(native::Factory4),
}

impl DxgiFactory {
    #[track_caller]
    pub fn unwrap_factory4(self) -> native::Factory4 {
        match self {
            DxgiFactory::Factory1(_) => panic!("unwrapping a factory4, got a factory1"),
            DxgiFactory::Factory2(_) => panic!("unwrapping a factory4, got a factory2"),
            DxgiFactory::Factory4(f) => f,
        }
    }
}

/// Tries to create a IDXGIFactory4, then a IDXGIFactory2, then a IDXGIFactory1,
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
    match lib_dxgi.create_factory2(factory_flags) {
        Ok(pair) => match pair.into_result() {
            Ok(factory) => return Ok((lib_dxgi, DxgiFactory::Factory4(factory))),
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
        }
    };

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
        Ok(factory2) => return Ok((lib_dxgi, DxgiFactory::Factory2(factory2))),
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
