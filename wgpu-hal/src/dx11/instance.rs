use crate::auxil;

impl crate::Instance<super::Api> for super::Instance {
    unsafe fn init(desc: &crate::InstanceDescriptor) -> Result<Self, crate::InstanceError> {
        let (lib_dxgi, factory) = auxil::dxgi::factory::create_factory(
            auxil::dxgi::factory::DxgiFactoryType::Factory1,
            desc.flags,
        )?;

        Ok(super::Instance { lib_dxgi, factory })
    }

    unsafe fn create_surface(
        &self,
        rwh: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Result<super::Surface, crate::InstanceError> {
        todo!()
    }

    unsafe fn destroy_surface(&self, surface: super::Surface) {
        todo!()
    }

    unsafe fn enumerate_adapters(&self) -> Vec<crate::ExposedAdapter<super::Api>> {
        todo!()
    }
}
