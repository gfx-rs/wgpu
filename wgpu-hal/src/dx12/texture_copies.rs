use core::mem;

use alloc::{string::String, sync::Arc};

use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

use crate::{
    auxil::dxgi::result::HResult,
    dx12::{suballocation::AllocationWrapper, PrivateCapabilities},
    Device,
};

const DEFAULT_SIZE: u64 = 1 << 18; // 256 KiB

pub struct TextureCopyHandler {
    temporary_buffer: TemporaryBuffer,
}

impl TextureCopyHandler {
    pub fn new(device: &super::Device) -> Result<Self, crate::DeviceError> {
        Ok(Self {
            temporary_buffer: TemporaryBuffer::new(device)?,
        })
    }

    pub unsafe fn encode_copy(
        &self,
        caps: &PrivateCapabilities,
        list: &ID3D12GraphicsCommandList,
        src: &super::Buffer,
        dst: &super::Texture,
        copy: &crate::BufferTextureCopy,
    ) -> Result<Arc<super::Buffer>, crate::DeviceError> {
        let copy_type = CopyType::from_copy(caps, copy);

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
    fn from_copy(caps: &PrivateCapabilities, c: &crate::BufferTextureCopy) -> Self {
        let natural_rows_per_image = c.size.depth;
        let rows_per_image = c
            .buffer_layout
            .rows_per_image
            .unwrap_or(natural_rows_per_image);

        if rows_per_image != natural_rows_per_image {
            return CopyType::LayerByLayer;
        }

        // If unrestricted pitch is supported, we can use the native copy
        // even if the offset is not aligned.
        if caps.unrestricted_buffer_texture_copy_pitch {
            return CopyType::Native;
        }

        if c.buffer_layout.offset % D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT as u64 != 0 {
            return CopyType::AlignmentOnly;
        }

        CopyType::Native
    }
}

pub struct TemporaryBuffer {
    buffer: Arc<super::Buffer>,
}

impl TemporaryBuffer {
    pub fn new(device: &super::Device) -> Result<Self, crate::DeviceError> {
        let size = DEFAULT_SIZE;

        let label = label(size);
        let desc = buffer_desc(&label, size);

        let buffer = Arc::new(unsafe { device.create_buffer(&desc)? });

        Ok(Self { buffer })
    }

    pub fn get_resource(&mut self, device: &super::Device, size: u64) -> Arc<super::Buffer> {
        if size > self.buffer.size {
            let label = label(size);
            let desc = buffer_desc(&label, size);

            // Recreate the buffer with the new size
            let new_buffer = Arc::new(unsafe { device.create_buffer(&desc).unwrap() });

            // Update the buffer
            let old_buffer = mem::replace(&mut self.buffer, new_buffer);

            // Release the old buffer if we are the last reference to it
            if let Some(buffer) = Arc::into_inner(old_buffer) {
                unsafe { device.destroy_buffer(buffer) };
            }
        }

        Arc::clone(&self.buffer)
    }
}

fn label(size: u64) -> String {
    format!("wgpu-hal texture copy temporary buffer ({} bytes)", size)
}

fn buffer_desc<'label>(label: &'label str, size: u64) -> crate::BufferDescriptor<'label> {
    crate::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgt::BufferUses::COPY_SRC | wgt::BufferUses::COPY_DST,
        memory_flags: crate::MemoryFlags::empty(),
    }
}
