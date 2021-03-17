use std::convert::TryFrom;

/// Describes a [Buffer](crate::Buffer) when allocating.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferInitDescriptor<'a> {
    /// Debug label of a buffer. This will show up in graphics debuggers for easy identification.
    pub label: Option<&'a str>,
    /// Contents of a buffer on creation.
    pub contents: &'a [u8],
    /// Usages of a buffer. If the buffer is used in any way that isn't specified here, the operation
    /// will panic.
    pub usage: crate::BufferUsage,
}

/// Utility methods not meant to be in the main API.
pub trait DeviceExt {
    /// Creates a [Buffer](crate::Buffer) with data to initialize it.
    fn create_buffer_init(&self, desc: &BufferInitDescriptor) -> crate::Buffer;

    /// Upload an entire texture and its mipmaps from a source buffer.
    ///
    /// Expects all mipmaps to be tightly packed in the data buffer.
    ///
    /// If the texture is a 2DArray texture, uploads each layer in order, expecting
    /// each layer and its mips to be tightly packed.
    ///
    /// Example:
    /// Layer0Mip0 Layer0Mip1 Layer0Mip2 ... Layer1Mip0 Layer1Mip1 Layer1Mip2 ...
    fn create_texture_with_data(
        &self,
        queue: &crate::Queue,
        desc: &crate::TextureDescriptor,
        data: &[u8],
    ) -> crate::Texture;
}

impl DeviceExt for crate::Device {
    fn create_buffer_init(&self, descriptor: &BufferInitDescriptor<'_>) -> crate::Buffer {
        let unpadded_size = descriptor.contents.len() as crate::BufferAddress;
        // Valid vulkan usage is
        // 1. buffer size must be a multiple of COPY_BUFFER_ALIGNMENT.
        // 2. buffer size must be greater than 0.
        // Therefore we round the value up to the nearest multiple, and ensure it's at least COPY_BUFFER_ALIGNMENT.
        let align_mask = crate::COPY_BUFFER_ALIGNMENT - 1;
        let padded_size = ((unpadded_size + align_mask) & !align_mask).max(crate::COPY_BUFFER_ALIGNMENT);

        let wgt_descriptor = crate::BufferDescriptor {
            label: descriptor.label,
            size: padded_size,
            usage: descriptor.usage,
            mapped_at_creation: true,
        };

        let mut map_context = crate::MapContext::new(padded_size);

        map_context.initial_range = 0..padded_size;

        let buffer = self.create_buffer(&wgt_descriptor);
        {
            let mut slice = buffer.slice(..).get_mapped_range_mut();
            slice[0..unpadded_size as usize].copy_from_slice(descriptor.contents);

            for i in unpadded_size..padded_size {
                slice[i as usize] = 0;
            }
        }
        buffer.unmap();
        buffer
    }

    fn create_texture_with_data(
        &self,
        queue: &crate::Queue,
        desc: &crate::TextureDescriptor,
        data: &[u8],
    ) -> crate::Texture {
        let texture = self.create_texture(desc);

        let format_info = desc.format.describe();

        let (layer_iterations, mip_extent) = if desc.dimension == crate::TextureDimension::D3 {
            (1, desc.size)
        } else {
            (
                desc.size.depth_or_array_layers,
                crate::Extent3d {
                    depth_or_array_layers: 1,
                    ..desc.size
                },
            )
        };

        let mip_level_count =
            u8::try_from(desc.mip_level_count).expect("mip level count overflows a u8");

        let mut binary_offset = 0;
        for layer in 0..layer_iterations {
            for mip in 0..mip_level_count {
                let mip_size = mip_extent.at_mip_level(mip).unwrap();

                // When uploading mips of compressed textures and the mip is supposed to be
                // a size that isn't a multiple of the block size, the mip needs to be uploaded
                // as it's "physical size" which is the size rounded up to the nearest block size.
                let mip_physical = mip_size.physical_size(desc.format);

                // All these calculations are performed on the physical size as that's the
                // data that exists in the buffer.
                let width_blocks = mip_physical.width / format_info.block_dimensions.0 as u32;
                let height_blocks = mip_physical.height / format_info.block_dimensions.1 as u32;

                let bytes_per_row = width_blocks * format_info.block_size as u32;
                let data_size = bytes_per_row * height_blocks * mip_extent.depth_or_array_layers;

                let end_offset = binary_offset + data_size as usize;

                queue.write_texture(
                    crate::TextureCopyView {
                        texture: &texture,
                        mip_level: mip as u32,
                        origin: crate::Origin3d {
                            x: 0,
                            y: 0,
                            z: layer,
                        },
                    },
                    &data[binary_offset..end_offset],
                    crate::TextureDataLayout {
                        offset: 0,
                        bytes_per_row,
                        rows_per_image: mip_physical.height,
                    },
                    mip_physical,
                );

                binary_offset = end_offset;
            }
        }

        texture
    }
}
