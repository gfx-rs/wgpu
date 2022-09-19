use crate::auxil;

impl crate::Instance<super::Api> for super::Instance {
    unsafe fn init(desc: &crate::InstanceDescriptor) -> Result<Self, crate::InstanceError> {
        let enable_dx11 = match std::env::var("WGPU_UNSTABLE_DX11_BACKEND") {
            Ok(string) => string == "1" || string == "true",
            Err(_) => false,
        };

        if !enable_dx11 {
            return Err(crate::InstanceError);
        }

        let lib_d3d11 = super::library::D3D11Lib::new().ok_or(crate::InstanceError)?;

        let (lib_dxgi, factory) = auxil::dxgi::factory::create_factory(
            auxil::dxgi::factory::DxgiFactoryType::Factory1,
            desc.flags,
        )?;

        Ok(super::Instance {
            lib_d3d11,
            lib_dxgi,
            factory,
        })
    }

    unsafe fn create_surface(
        &self,
        display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
    ) -> Result<super::Surface, crate::InstanceError> {
        todo!()
    }

    unsafe fn destroy_surface(&self, surface: super::Surface) {
        todo!()
    }

    unsafe fn enumerate_adapters(&self) -> Vec<crate::ExposedAdapter<super::Api>> {
        let adapters = auxil::dxgi::factory::enumerate_adapters(self.factory);

        adapters
            .into_iter()
            .filter_map(|adapter| super::Adapter::expose(&self.lib_d3d11, adapter))
            .collect()
    }
}
