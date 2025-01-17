use std::sync::Arc;

use wgpu_test::{gpu_test, FailureCase, GpuTestConfiguration, TestParameters, TestingContext};

#[gpu_test]
static TEST_SINGLE_WRITE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(|ctx| async move { run_test(ctx, false).await });

#[gpu_test]
static TEST_SCATTER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            // See https://github.com/gfx-rs/wgpu/issues/6827
            .expect_fail(FailureCase::backend_adapter(
                wgpu::Backends::METAL,
                "Apple M", // M1,M2 etc
            ))
            .expect_fail(FailureCase::backend_adapter(
                wgpu::Backends::METAL,
                "Apple Paravirtual device", // CI on M1
            ))
            .expect_fail(
                // Unfortunately this depends on if `D3D12_FEATURE_DATA_D3D12_OPTIONS13.UnrestrictedBufferTextureCopyPitchSupported`
                // is true, which we have no way to encode. This reproduces in CI though, so not too worried about it.
                FailureCase::backend(wgpu::Backends::DX12)
                    .flaky()
                    .validation_error(
                        "D3D12_PLACED_SUBRESOURCE_FOOTPRINT::Offset must be a multiple of 512",
                    )
                    .panic("GraphicsCommandList::close failed: The parameter is incorrect"),
            ),
    )
    .run_async(|ctx| async move { run_test(ctx, true).await });

async fn run_test(ctx: TestingContext, use_many_writes: bool) {
    let device = ctx.device;
    let queue = ctx.queue;

    let size = wgpu::Extent3d {
        width: 4,
        height: 4,
        depth_or_array_layers: 4,
    };
    let texture = {
        device.create_texture(&wgpu::TextureDescriptor {
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Uint,
            view_formats: &[],
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            label: None,
        })
    };

    if use_many_writes {
        many_writes(&texture, &device, &queue);
    } else {
        single_write(&texture, &queue);
    }

    let light_texels: Vec<[u8; 4]> = {
        let tc = TextureCopyParameters::from_texture(&texture);
        let temp_buffer = tc.copy_texture_to_new_buffer(&device, &queue, &texture);

        let result_cell =
            Arc::new(std::sync::OnceLock::<Result<(), wgpu::BufferAsyncError>>::new());
        temp_buffer.slice(..).map_async(wgpu::MapMode::Read, {
            let result_cell = result_cell.clone();
            move |result| result_cell.set(result).unwrap()
        });
        device.poll(wgpu::Maintain::Wait);
        result_cell
            .get()
            .as_ref()
            .expect("cell not set")
            .as_ref()
            .expect("mapping failed");

        tc.copy_mapped_to_vec(1, &temp_buffer)
    };

    let mut wrong_texels = Vec::new();
    for (zyx_index, cube) in texel_iter(&texture).enumerate() {
        #[allow(clippy::cast_possible_wrap)]
        let expected = texel_for_cube(cube);
        let actual = light_texels[zyx_index];
        if expected != actual {
            println!("{:?}", (cube, expected, actual));
            wrong_texels.push((cube, expected, actual));
        }
    }

    let volume = size.width * size.height * size.depth_or_array_layers;
    assert!(
        wrong_texels.is_empty(),
        "out of {volume}, {len} were wrong",
        len = wrong_texels.len(),
    );
}

// -------------------------------------------------------------------------------------------------

type Texel = [u8; COMPONENTS];

pub fn texel_for_cube(point: [u32; 3]) -> [u8; 4] {
    [
        10 + point[0] as u8,
        10 + point[1] as u8,
        10 + point[2] as u8,
        10,
    ]
}

const COMPONENTS: usize = 4;

fn texel_iter(texture: &wgpu::Texture) -> impl Iterator<Item = [u32; 3]> {
    itertools::iproduct!(
        0..texture.depth_or_array_layers(),
        0..texture.height(),
        0..texture.width()
    )
    .map(|(z, y, x)| [x, y, z])
}

fn compute_data(texture: &wgpu::Texture) -> Vec<Texel> {
    let mut data = Vec::new();
    for point in texel_iter(texture) {
        data.push(texel_for_cube(point));
    }
    data
}

pub fn single_write(texture: &wgpu::Texture, queue: &wgpu::Queue) {
    let data = compute_data(texture);

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data.as_flattened(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(texture.width() * COMPONENTS as u32),
            rows_per_image: Some(texture.height()),
        },
        texture.size(),
    )
}

pub fn many_writes(texture: &wgpu::Texture, device: &wgpu::Device, queue: &wgpu::Queue) {
    let data: Vec<Texel> = compute_data(texture);

    let copy_buffer_2 = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::try_from(data.len() * COMPONENTS).unwrap(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    for (index, cube) in texel_iter(texture).enumerate() {
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &copy_buffer_2,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: (index * COMPONENTS) as u64,
                    bytes_per_row: None,
                    rows_per_image: None,
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: cube[0],
                    y: cube[1],
                    z: cube[2],
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    queue.write_buffer(&copy_buffer_2, 0, data.as_flattened());
    queue.submit([encoder.finish()]);
}

// -------------------------------------------------------------------------------------------------

/// Elements of GPU-to-CPU copying
#[derive(Clone, Copy, Debug)]
struct TextureCopyParameters {
    pub size: wgpu::Extent3d,
    pub byte_size_of_texel: u32,
}
impl TextureCopyParameters {
    pub fn from_texture(texture: &wgpu::Texture) -> Self {
        let format = texture.format();
        assert_eq!(
            format.block_dimensions(),
            (1, 1),
            "compressed texture format {format:?} not supported",
        );

        Self {
            size: texture.size(),
            byte_size_of_texel: format
                .block_copy_size(None)
                .expect("non-color texture format {format:} not supported"),
        }
    }

    pub fn dense_bytes_per_row(&self) -> u32 {
        self.size.width * self.byte_size_of_texel
    }

    pub fn padded_bytes_per_row(&self) -> u32 {
        self.dense_bytes_per_row()
            .div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
    }

    #[track_caller]
    pub fn copy_texture_to_new_buffer(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> wgpu::Buffer {
        let padded_bytes_per_row = self.padded_bytes_per_row();

        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU-to-CPU image copy buffer"),
            size: u64::from(padded_bytes_per_row)
                * u64::from(self.size.height)
                * u64::from(self.size.depth_or_array_layers),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.copy_texture_to_buffer(
                texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &temp_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bytes_per_row),
                        rows_per_image: Some(self.size.height),
                    },
                },
                texture.size(),
            );
            queue.submit(Some(encoder.finish()));
        }

        temp_buffer
    }

    /// Given a mapped buffer, make a [`Vec<C>`] of it.
    ///
    /// `size_of::<C>() * components` must be equal to the byte size of a texel.
    pub fn copy_mapped_to_vec<C>(&self, components: usize, buffer: &wgpu::Buffer) -> Vec<C>
    where
        C: bytemuck::AnyBitPattern,
    {
        assert_eq!(
            u32::try_from(components * size_of::<C>()).ok(),
            Some(self.byte_size_of_texel),
            "Texture format does not match requested format",
        );

        // Copy the mapped buffer data into a Rust vector, removing row padding if present
        // by copying it one row at a time.
        let mut texel_vector: Vec<C> = Vec::new();
        {
            let mapped: &[u8] = &buffer.slice(..).get_mapped_range();
            for row in 0..self.row_count() {
                let byte_start_of_row = (self.padded_bytes_per_row()) as usize * row;
                texel_vector.extend(bytemuck::cast_slice::<u8, C>(
                    &mapped[byte_start_of_row..][..self.dense_bytes_per_row() as usize],
                ));
            }
        }

        texel_vector
    }

    fn row_count(&self) -> usize {
        self.size.height as usize * self.size.depth_or_array_layers as usize
    }
}
