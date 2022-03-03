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

impl super::Adapter {
    pub(super) fn expose(
        instance: &super::library::D3D11Lib,
        adapter: native::DxgiAdapter,
    ) -> Option<crate::ExposedAdapter<super::Api>> {
        let (device, feature_level) = instance.create_device(adapter)?;

        todo!()
    }
}
