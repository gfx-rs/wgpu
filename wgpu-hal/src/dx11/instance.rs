impl crate::Instance<super::Api> for super::Instance {
    unsafe fn init(desc: &crate::InstanceDescriptor) -> Result<Self, crate::InstanceError> {
        todo!()
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
