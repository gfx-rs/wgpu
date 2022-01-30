impl crate::Adapter<super::Api> for super::Adapter {
    unsafe fn open(
        &self,
        features: wgt::Features,
        limits: &wgt::Limits,
    ) -> Result<crate::OpenDevice<super::Api>, crate::DeviceError> {
        todo!()
    }

    unsafe fn texture_format_capabilities(
        &self,
        format: wgt::TextureFormat,
    ) -> crate::TextureFormatCapabilities {
        todo!()
    }

    unsafe fn surface_capabilities(
        &self,
        surface: &super::Surface,
    ) -> Option<crate::SurfaceCapabilities> {
        todo!()
    }
}
