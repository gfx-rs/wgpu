use windows::Win32::Graphics::Direct3D12::{
    ID3D12Device, ID3D12GraphicsCommandList, ID3D12Resource, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT,
};

use crate::dx12::PrivateCapabilities;

pub struct TextureCopyHandler {}

impl TextureCopyHandler {
    pub fn new(device: &ID3D12Device) -> Self {
        Self {}
    }

    pub fn encode_copy(
        &self,
        list: &ID3D12GraphicsCommandList,
        src: &super::Buffer,
        dst: &super::Texture,
        copy: crate::BufferTextureCopy,
    ) -> ID3D12Resource {
        todo!()
    }
}

enum CopyType {
    /// Copy can be expressed natively using a D3D12 copy command.
    Native,
    /// Copy needs to be done layer by layer, because the rows per image is not
    /// the "natural" rows per image.
    LayerByLayer,
    /// Offset is not properly aligned, so the entire upload needs to be
    /// copied in bulk to a temporary buffer first.
    AlignmentOnly,
}

impl CopyType {
    fn from_copy(c: &crate::BufferTextureCopy) -> Self {
        let natural_rows_per_image = c.size.depth;
        let rows_per_image = c
            .buffer_layout
            .rows_per_image
            .unwrap_or(natural_rows_per_image);

        if rows_per_image != natural_rows_per_image {
            return CopyType::LayerByLayer;
        }

        if c.buffer_layout.offset % D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT as u64 != 0 {
            return CopyType::AlignmentOnly;
        }

        CopyType::Native
    }
}
