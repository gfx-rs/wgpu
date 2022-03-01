use super::SurfaceTarget;
use crate::auxil::{self, dxgi::result::HResult as _};
use std::sync::Arc;
use winapi::{
    shared::{dxgi, dxgi1_2, dxgi1_6, winerror},
    Interface,
};

impl Drop for super::Instance {
    fn drop(&mut self) {
        unsafe {
            self.factory.destroy();
            crate::auxil::dxgi::exception::unregister_exception_handler();
        }
    }
}

impl crate::Instance<super::Api> for super::Instance {
    unsafe fn init(desc: &crate::InstanceDescriptor) -> Result<Self, crate::InstanceError> {
        let lib_main = native::D3D12Lib::new().map_err(|_| crate::InstanceError)?;

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
        }

        // Create DXGIFactory4
        let (lib_dxgi, factory) = auxil::dxgi::factory::create_factory(
            auxil::dxgi::factory::DxgiFactoryType::Factory4,
            desc.flags,
        )?;

        Ok(Self {
            // The call to create_factory will only succeed if we get a factory4, so this is safe.
            factory: factory.unwrap_factory4(),
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
            raw_window_handle::RawWindowHandle::Win32(handle) => Ok(super::Surface {
                factory: self.factory,
                target: SurfaceTarget::WndHandle(handle.hwnd as *mut _),
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
