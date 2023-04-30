
impl crate::Texture<super::Api> for super::Texture {
    unsafe fn get_size(&self) -> wgt::Extent3d {
        wgt::Extent3d{
            width: 0,
            height: 0,
            depth_or_array_layers: 0,
        }
    }
}
