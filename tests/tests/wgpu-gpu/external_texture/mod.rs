use approx::assert_abs_diff_eq;
use wgpu_test::{
    gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        EXTERNAL_TEXTURE_DIMENSIONS,
        EXTERNAL_TEXTURE_LOAD,
        EXTERNAL_TEXTURE_LOAD_YUV,
        EXTERNAL_TEXTURE_LOAD_TRANSFORM,
        EXTERNAL_TEXTURE_LOAD_INVALID_ADDRESS,
        EXTERNAL_TEXTURE_SAMPLE,
        EXTERNAL_TEXTURE_SAMPLE_YUV,
        EXTERNAL_TEXTURE_SAMPLE_TRANSFORM,
    ]);
}

// Input texture data
const RED_U8: [u8; 4] = [0xFF, 0x00, 0x00, 0xFF];
const GREEN_U8: [u8; 4] = [0x00, 0xFF, 0x00, 0xFF];
const BLUE_U8: [u8; 4] = [0x00, 0x00, 0xFF, 0xFF];
const YELLOW_U8: [u8; 4] = [0xFF, 0xFF, 0x00, 0xFF];
const BLACK_U8: [u8; 4] = [0x00, 0x00, 0x00, 0xFF];

// Data for a 2x2 or 4x1 texture with red, green, blue, and yellow pixels
const RGBA_TEXTURE_DATA: [[u8; 4]; 4] = [RED_U8, GREEN_U8, BLUE_U8, YELLOW_U8];

// Data for a 4x4 texture with red, green, blue, and yellow pixels in the
// centre, surrounded by a black border.
#[rustfmt::skip]
const RGBA_TEXTURE_DATA_WITH_BORDER: [[u8; 4]; 16] = [
    BLACK_U8, BLACK_U8, BLACK_U8,  BLACK_U8,
    BLACK_U8, RED_U8,   GREEN_U8,  BLACK_U8,
    BLACK_U8, BLUE_U8,  YELLOW_U8, BLACK_U8,
    BLACK_U8, BLACK_U8, BLACK_U8,  BLACK_U8,
];

// Red, green, blue, and yellow in BT601 limited range YUV. Values obtained by
// extracting raw frames from the WebGPU CTS' "four-colors" video [1], and
// inspecting the contents.
// [1] https://github.com/gpuweb/cts/blob/44dac855ba07b23c49d2cbc2c9d87bf8e6f38c47/src/resources/four-colors-vp8-bt601.webm
const RED_Y_U8: u8 = 0x51;
const RED_U_U8: u8 = 0x5A;
const RED_V_U8: u8 = 0xF0;
const GREEN_Y_U8: u8 = 0x91;
const GREEN_U_U8: u8 = 0x35;
const GREEN_V_U8: u8 = 0x22;
const BLUE_Y_U8: u8 = 0x29;
const BLUE_U_U8: u8 = 0xF0;
const BLUE_V_U8: u8 = 0x6E;
const YELLOW_Y_U8: u8 = 0xD2;
const YELLOW_U_U8: u8 = 0x10;
const YELLOW_V_U8: u8 = 0x92;

// Data for a 4x4 texture with 4:2:0 chroma subsampling. The top-left quadrant
// is red, top-right green, bottom-left blue, and bottom-right yellow.
#[rustfmt::skip]
const Y_TEXTURE_DATA: [u8; 16] = [
    RED_Y_U8,  RED_Y_U8,  GREEN_Y_U8,  GREEN_Y_U8,
    RED_Y_U8,  RED_Y_U8,  GREEN_Y_U8,  GREEN_Y_U8,
    BLUE_Y_U8, BLUE_Y_U8, YELLOW_Y_U8, YELLOW_Y_U8,
    BLUE_Y_U8, BLUE_Y_U8, YELLOW_Y_U8, YELLOW_Y_U8,
];
const U_TEXTURE_DATA: [u8; 4] = [RED_U_U8, GREEN_U_U8, BLUE_U_U8, YELLOW_U_U8];
const V_TEXTURE_DATA: [u8; 4] = [RED_V_U8, GREEN_V_U8, BLUE_V_U8, YELLOW_V_U8];

// Expected results after texture load/sample.
const RED_F32: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const GREEN_F32: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
const BLUE_F32: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
const YELLOW_F32: [f32; 4] = [1.0, 1.0, 0.0, 1.0];
const OPAQUE_BLACK_F32: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const TRANSPARENT_BLACK_F32: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

// Expected results after texture load/sample in sRGB color space. Values
// taken from the WebGPU CTS:
// https://github.com/gpuweb/cts/blob/44dac855ba07b23c49d2cbc2c9d87bf8e6f38c47/src/webgpu/web_platform/util.ts#L36-L43
const RED_SRGB_F32: [f32; 4] = [0.9729456, 0.14179438, -0.020958992, 1.0];
const GREEN_SRGB_F32: [f32; 4] = [0.24823427, 0.98481035, -0.056470133, 1.0];
const BLUE_SRGB_F32: [f32; 4] = [0.10159736, 0.13545112, 1.0026299, 1.0];
const YELLOW_SRGB_F32: [f32; 4] = [0.99547076, 0.9927421, -0.07742912, 1.0];

#[rustfmt::skip]
const IDENTITY_YUV_CONVERSION_MATRIX: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];
#[rustfmt::skip]
const BT601_YUV_CONVERSION_MATRIX: [f32; 16] = [
    1.1643835,  1.1643834,   1.1643835,  0.0,
    0.0,        -0.39176223, 2.017232,   0.0,
    1.5960265,  -0.81296754, 0.0,        0.0,
    -0.8742022, 0.5316678,   -1.0856307, 1.0
];

const SRGB_TRANSFER_FUNCTION: wgpu::ExternalTextureTransferFunction =
    wgpu::ExternalTextureTransferFunction {
        a: 1.055,
        b: 0.003130805,
        g: 2.4,
        k: 12.92,
    };

const IDENTITY_GAMUT_CONVERSION_MATRIX: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
#[rustfmt::skip]
const BT601_TO_SRGB_GAMUT_CONVERSION_MATRIX: [f32; 9] = [
    0.93954253,  0.017772198, -0.0016215984,
    0.050181333, 0.96579295,  -0.0043697506,
    0.010276437, 0.016434949, 1.0059911,
];

const IDENTITY_SAMPLE_TRANSFORM: [f32; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
const IDENTITY_LOAD_TRANSFORM: [f32; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
// Flips a 2x2 texture horizontally
const HORIZONTAL_FLIP_2X2_SAMPLE_TRANSFORM: [f32; 6] = [-1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
const HORIZONTAL_FLIP_2X2_LOAD_TRANSFORM: [f32; 6] = [-1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
// Flips a 2x2 texture vertically
const VERTICAL_FLIP_2X2_SAMPLE_TRANSFORM: [f32; 6] = [1.0, 0.0, 0.0, -1.0, 0.0, 1.0];
const VERTICAL_FLIP_2X2_LOAD_TRANSFORM: [f32; 6] = [1.0, 0.0, 0.0, -1.0, 0.0, 1.0];
// Rotates a 4x1 texture 90 degrees
const ROTATE_90_4X1_SAMPLE_TRANSFORM: [f32; 6] = [0.0, -1.0, 1.0, 0.0, 0.0, 1.0];
const ROTATE_90_4X1_LOAD_TRANSFORM: [f32; 6] = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
// Rotates a 4x1 texture 180 degrees
const ROTATE_180_4X1_SAMPLE_TRANSFORM: [f32; 6] = [-1.0, 0.0, 0.0, -1.0, 1.0, 1.0];
const ROTATE_180_4X1_LOAD_TRANSFORM: [f32; 6] = [-1.0, 0.0, 0.0, 0.0, 3.0, 0.0];
// Rotates a 4xx1 texture 270 degrees
const ROTATE_270_4X1_SAMPLE_TRANSFORM: [f32; 6] = [0.0, 1.0, -1.0, 0.0, 1.0, 0.0];
const ROTATE_270_4X1_LOAD_TRANSFORM: [f32; 6] = [0.0, 0.0, -1.0, 0.0, 3.0, 0.0];
// Crops the middle 2x2 pixels from a 4x4 texture
const CROP_4X4_SAMPLE_TRANSFORM: [f32; 6] = [0.5, 0.0, 0.0, 0.5, 0.25, 0.25];
const CROP_4X4_LOAD_TRANSFORM: [f32; 6] = [0.5, 0.0, 0.0, 0.5, 1.0, 1.0];

/// Helper function to create a 2D texture and a view, optionally writing the
/// provided data to the texture, and returning the view.
fn create_texture_and_view(
    ctx: &TestingContext,
    size: wgpu::Extent3d,
    format: wgpu::TextureFormat,
    data: Option<&[u8]>,
) -> wgpu::TextureView {
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    if let Some(data) = data {
        ctx.queue.write_texture(
            texture.as_image_copy(),
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.width * format.components() as u32),
                rows_per_image: None,
            },
            size,
        );
    }
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Helper function to perform textureDimensions() and return the result.
fn get_dimensions(ctx: &TestingContext, texture_resource: wgpu::BindingResource) -> [u32; 2] {
    let module = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("dimensions.wgsl"));
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: std::mem::size_of::<[u32; 2]>() as _,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let download_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: texture_resource,
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &download_buffer, 0, output_buffer.size());
    ctx.queue.submit(Some(encoder.finish()));
    let buffer_slice = download_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    ctx.device.poll(wgpu::PollType::Wait).unwrap();

    let data = buffer_slice.get_mapped_range();
    let size: &[u32] = bytemuck::cast_slice(&data);
    size.try_into().unwrap()
}

/// Helper function to perform `textureLoad()` for the specified coordinates and return
/// the loaded values.
fn get_loads(
    ctx: &TestingContext,
    coords: &[[u32; 2]],
    texture_resource: wgpu::BindingResource,
) -> Vec<[f32; 4]> {
    let module = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("load.wgsl"));
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let coords_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: std::mem::size_of_val(coords) as _,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue
        .write_buffer(&coords_buffer, 0, bytemuck::cast_slice(coords));

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (coords.len() * std::mem::size_of::<[f32; 4]>()) as _,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let download_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: texture_resource,
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: coords_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(coords.len() as _, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &download_buffer, 0, output_buffer.size());
    ctx.queue.submit(Some(encoder.finish()));
    let buffer_slice = download_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    ctx.device.poll(wgpu::PollType::Wait).unwrap();

    let data = buffer_slice.get_mapped_range();
    let values: &[[f32; 4]] = bytemuck::cast_slice(&data);
    values.to_vec()
}

/// Helper function to perform `textureSampleBaseClampToEdge()` for the specified
/// coordinates and return the sampled values.
fn get_samples(
    ctx: &TestingContext,
    coords: &[[f32; 2]],
    texture_resource: wgpu::BindingResource,
) -> Vec<[f32; 4]> {
    let module = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("sample.wgsl"));
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let coords_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: std::mem::size_of_val(coords) as _,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue
        .write_buffer(&coords_buffer, 0, bytemuck::cast_slice(coords));

    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (coords.len() * std::mem::size_of::<[f32; 4]>()) as _,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let download_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output_buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let sampler = ctx
        .device
        .create_sampler(&wgpu::SamplerDescriptor::default());

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: texture_resource,
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: coords_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(coords.len() as _, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &download_buffer, 0, output_buffer.size());
    ctx.queue.submit(Some(encoder.finish()));
    let buffer_slice = download_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    ctx.device.poll(wgpu::PollType::Wait).unwrap();

    let data = buffer_slice.get_mapped_range();
    let values: &[[f32; 4]] = bytemuck::cast_slice(&data);
    values.to_vec()
}

/// Tests that `textureDimensions()` returns the correct value for both external textures
/// and texture views bound to an external texture binding.
#[gpu_test]
static EXTERNAL_TEXTURE_DIMENSIONS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::EXTERNAL_TEXTURE),
    )
    .run_async(|ctx| async move {
        const TEXTURE_WIDTH: u32 = 128;
        const TEXTURE_HEIGHT: u32 = 64;
        let view = create_texture_and_view(
            &ctx,
            wgpu::Extent3d {
                width: TEXTURE_WIDTH,
                height: TEXTURE_HEIGHT,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::Rgba8Unorm,
            None,
        );
        let dims = get_dimensions(&ctx, wgpu::BindingResource::TextureView(&view));
        assert_eq!(dims, [TEXTURE_WIDTH, TEXTURE_HEIGHT]);

        const EXTERNAL_TEXTURE_WIDTH: u32 = 32;
        const EXTERNAL_TEXTURE_HEIGHT: u32 = 16;
        let external_texture = ctx.device.create_external_texture(
            &wgpu::ExternalTextureDescriptor {
                label: None,
                width: EXTERNAL_TEXTURE_WIDTH,
                height: EXTERNAL_TEXTURE_HEIGHT,
                format: wgpu::ExternalTextureFormat::Rgba,
                yuv_conversion_matrix: IDENTITY_YUV_CONVERSION_MATRIX,
                gamut_conversion_matrix: IDENTITY_GAMUT_CONVERSION_MATRIX,
                src_transfer_function: Default::default(),
                dst_transfer_function: Default::default(),
                sample_transform: IDENTITY_SAMPLE_TRANSFORM,
                load_transform: IDENTITY_LOAD_TRANSFORM,
            },
            &[&view],
        );
        let dims = get_dimensions(
            &ctx,
            wgpu::BindingResource::ExternalTexture(&external_texture),
        );
        // This should return the dimensions provided in the ExternalTextureDescriptor,
        // rather than the dimensions of the underlying texture.
        assert_eq!(dims, [EXTERNAL_TEXTURE_WIDTH, EXTERNAL_TEXTURE_HEIGHT])
    });

/// Tests that `textureLoad()` returns the correct values for both RGBA format external
/// textures and texture views bound to an external texture binding.
#[gpu_test]
static EXTERNAL_TEXTURE_LOAD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::EXTERNAL_TEXTURE),
    )
    .run_async(|ctx| async move {
        let view = create_texture_and_view(
            &ctx,
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::Rgba8Unorm,
            Some(RGBA_TEXTURE_DATA.as_flattened()),
        );
        let loads = get_loads(
            &ctx,
            &[[0, 0], [1, 0], [0, 1], [1, 1]],
            wgpu::BindingResource::TextureView(&view),
        );
        assert_eq!(&loads, &[RED_F32, GREEN_F32, BLUE_F32, YELLOW_F32]);

        let external_texture = ctx.device.create_external_texture(
            &wgpu::ExternalTextureDescriptor {
                label: None,
                width: 2,
                height: 2,
                format: wgpu::ExternalTextureFormat::Rgba,
                yuv_conversion_matrix: IDENTITY_YUV_CONVERSION_MATRIX,
                gamut_conversion_matrix: IDENTITY_GAMUT_CONVERSION_MATRIX,
                src_transfer_function: Default::default(),
                dst_transfer_function: Default::default(),
                sample_transform: IDENTITY_SAMPLE_TRANSFORM,
                load_transform: IDENTITY_LOAD_TRANSFORM,
            },
            &[&view],
        );

        let loads = get_loads(
            &ctx,
            &[[0, 0], [1, 0], [0, 1], [1, 1]],
            wgpu::BindingResource::ExternalTexture(&external_texture),
        );
        assert_eq!(&loads, &[RED_F32, GREEN_F32, BLUE_F32, YELLOW_F32]);
    });

/// Tests that `textureLoad()` returns the correct values for YUV format external
/// textures.
#[gpu_test]
static EXTERNAL_TEXTURE_LOAD_YUV: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::EXTERNAL_TEXTURE),
    )
    .run_async(|ctx| async move {
        let y_view = create_texture_and_view(
            &ctx,
            wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::R8Unorm,
            Some(&Y_TEXTURE_DATA),
        );
        let u_view = create_texture_and_view(
            &ctx,
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::R8Unorm,
            Some(&U_TEXTURE_DATA),
        );
        let v_view = create_texture_and_view(
            &ctx,
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::R8Unorm,
            Some(&V_TEXTURE_DATA),
        );

        let external_texture = ctx.device.create_external_texture(
            &wgpu::ExternalTextureDescriptor {
                label: None,
                width: 4,
                height: 4,
                format: wgpu::ExternalTextureFormat::Yu12,
                yuv_conversion_matrix: BT601_YUV_CONVERSION_MATRIX,
                gamut_conversion_matrix: BT601_TO_SRGB_GAMUT_CONVERSION_MATRIX,
                src_transfer_function: SRGB_TRANSFER_FUNCTION,
                dst_transfer_function: SRGB_TRANSFER_FUNCTION,
                sample_transform: IDENTITY_SAMPLE_TRANSFORM,
                load_transform: IDENTITY_LOAD_TRANSFORM,
            },
            &[&y_view, &u_view, &v_view],
        );
        let loads = get_loads(
            &ctx,
            &[[1, 1], [2, 1], [1, 2], [2, 2]],
            wgpu::BindingResource::ExternalTexture(&external_texture),
        );

        // `assert_abs_diff_eq!()` works for slices but not arrays, hence the
        // following tomfoolery.
        let loads = loads.iter().map(|arr| arr.as_slice()).collect::<Vec<_>>();
        let expected = [RED_SRGB_F32, GREEN_SRGB_F32, BLUE_SRGB_F32, YELLOW_SRGB_F32];
        let expected = expected.each_ref().map(|arr| arr.as_slice());
        // We expect slight inaccuracies due to floating point maths.
        assert_abs_diff_eq!(loads.as_slice(), expected.as_slice(), epsilon = 0.01);
    });

/// Tests that `textureLoad()` returns the correct values for external textures with
/// various load transforms.
#[gpu_test]
static EXTERNAL_TEXTURE_LOAD_TRANSFORM: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::EXTERNAL_TEXTURE),
    )
    .run_async(|ctx| async move {
        let view_2x2 = create_texture_and_view(
            &ctx,
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::Rgba8Unorm,
            Some(RGBA_TEXTURE_DATA.as_flattened()),
        );

        let flip_h_external_texture = ctx.device.create_external_texture(
            &wgpu::ExternalTextureDescriptor {
                label: None,
                width: 2,
                height: 2,
                format: wgpu::ExternalTextureFormat::Rgba,
                yuv_conversion_matrix: IDENTITY_YUV_CONVERSION_MATRIX,
                gamut_conversion_matrix: IDENTITY_GAMUT_CONVERSION_MATRIX,
                src_transfer_function: Default::default(),
                dst_transfer_function: Default::default(),
                sample_transform: HORIZONTAL_FLIP_2X2_SAMPLE_TRANSFORM,
                load_transform: HORIZONTAL_FLIP_2X2_LOAD_TRANSFORM,
            },
            &[&view_2x2],
        );
        let loads = get_loads(
            &ctx,
            &[[0, 0], [1, 0], [0, 1], [1, 1]],
            wgpu::BindingResource::ExternalTexture(&flip_h_external_texture),
        );
        assert_eq!(&loads, &[GREEN_F32, RED_F32, YELLOW_F32, BLUE_F32]);

        let flip_v_external_texture = ctx.device.create_external_texture(
            &wgpu::ExternalTextureDescriptor {
                label: None,
                width: 2,
                height: 2,
                format: wgpu::ExternalTextureFormat::Rgba,
                yuv_conversion_matrix: IDENTITY_YUV_CONVERSION_MATRIX,
                gamut_conversion_matrix: IDENTITY_GAMUT_CONVERSION_MATRIX,
                src_transfer_function: Default::default(),
                dst_transfer_function: Default::default(),
                sample_transform: VERTICAL_FLIP_2X2_SAMPLE_TRANSFORM,
                load_transform: VERTICAL_FLIP_2X2_LOAD_TRANSFORM,
            },
            &[&view_2x2],
        );
        let loads = get_loads(
            &ctx,
            &[[0, 0], [1, 0], [0, 1], [1, 1]],
            wgpu::BindingResource::ExternalTexture(&flip_v_external_texture),
        );
        assert_eq!(&loads, &[BLUE_F32, YELLOW_F32, RED_F32, GREEN_F32]);

        // Use a non-square texture for the rotation cases as it's more
        // interesting
        let view_4x1 = create_texture_and_view(
            &ctx,
            wgpu::Extent3d {
                width: 4,
                height: 1,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::Rgba8Unorm,
            Some(RGBA_TEXTURE_DATA.as_flattened()),
        );

        let rotate_90_external_texture = ctx.device.create_external_texture(
            &wgpu::ExternalTextureDescriptor {
                label: None,
                width: 1,
                height: 4,
                format: wgpu::ExternalTextureFormat::Rgba,
                yuv_conversion_matrix: IDENTITY_YUV_CONVERSION_MATRIX,
                gamut_conversion_matrix: IDENTITY_GAMUT_CONVERSION_MATRIX,
                src_transfer_function: Default::default(),
                dst_transfer_function: Default::default(),
                sample_transform: ROTATE_90_4X1_SAMPLE_TRANSFORM,
                load_transform: ROTATE_90_4X1_LOAD_TRANSFORM,
            },
            &[&view_4x1],
        );
        let loads = get_loads(
            &ctx,
            &[[0, 0], [0, 1], [0, 2], [0, 3]],
            wgpu::BindingResource::ExternalTexture(&rotate_90_external_texture),
        );
        assert_eq!(&loads, &[RED_F32, GREEN_F32, BLUE_F32, YELLOW_F32]);

        let rotate_180_external_texture = ctx.device.create_external_texture(
            &wgpu::ExternalTextureDescriptor {
                label: None,
                width: 4,
                height: 1,
                format: wgpu::ExternalTextureFormat::Rgba,
                yuv_conversion_matrix: IDENTITY_YUV_CONVERSION_MATRIX,
                gamut_conversion_matrix: IDENTITY_GAMUT_CONVERSION_MATRIX,
                src_transfer_function: Default::default(),
                dst_transfer_function: Default::default(),
                sample_transform: ROTATE_180_4X1_SAMPLE_TRANSFORM,
                load_transform: ROTATE_180_4X1_LOAD_TRANSFORM,
            },
            &[&view_4x1],
        );
        let loads = get_loads(
            &ctx,
            &[[0, 0], [1, 0], [2, 0], [3, 0]],
            wgpu::BindingResource::ExternalTexture(&rotate_180_external_texture),
        );
        assert_eq!(&loads, &[YELLOW_F32, BLUE_F32, GREEN_F32, RED_F32]);

        let rotate_270_external_texture = ctx.device.create_external_texture(
            &wgpu::ExternalTextureDescriptor {
                label: None,
                width: 1,
                height: 4,
                format: wgpu::ExternalTextureFormat::Rgba,
                yuv_conversion_matrix: IDENTITY_YUV_CONVERSION_MATRIX,
                gamut_conversion_matrix: IDENTITY_GAMUT_CONVERSION_MATRIX,
                src_transfer_function: Default::default(),
                dst_transfer_function: Default::default(),
                sample_transform: ROTATE_270_4X1_SAMPLE_TRANSFORM,
                load_transform: ROTATE_270_4X1_LOAD_TRANSFORM,
            },
            &[&view_4x1],
        );
        let loads = get_loads(
            &ctx,
            &[[0, 0], [0, 1], [0, 2], [0, 3]],
            wgpu::BindingResource::ExternalTexture(&rotate_270_external_texture),
        );
        assert_eq!(&loads, &[YELLOW_F32, BLUE_F32, GREEN_F32, RED_F32]);

        let view_4x4 = create_texture_and_view(
            &ctx,
            wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::Rgba8Unorm,
            Some(RGBA_TEXTURE_DATA_WITH_BORDER.as_flattened()),
        );
        let crop_tex = ctx.device.create_external_texture(
            &wgpu::ExternalTextureDescriptor {
                label: None,
                width: 2,
                height: 2,
                format: wgpu::ExternalTextureFormat::Rgba,
                yuv_conversion_matrix: IDENTITY_YUV_CONVERSION_MATRIX,
                gamut_conversion_matrix: IDENTITY_GAMUT_CONVERSION_MATRIX,
                src_transfer_function: Default::default(),
                dst_transfer_function: Default::default(),
                sample_transform: CROP_4X4_SAMPLE_TRANSFORM,
                load_transform: CROP_4X4_LOAD_TRANSFORM,
            },
            &[&view_4x4],
        );
        let loads = get_loads(
            &ctx,
            &[[0, 0], [1, 0], [0, 1], [1, 1]],
            wgpu::BindingResource::ExternalTexture(&crop_tex),
        );
        assert_eq!(&loads, &[RED_F32, GREEN_F32, BLUE_F32, YELLOW_F32]);
    });

/// Tests that `textureLoad()` for an invalid address returns an allowed value.
#[gpu_test]
static EXTERNAL_TEXTURE_LOAD_INVALID_ADDRESS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::EXTERNAL_TEXTURE),
    )
    .run_async(|ctx| async move {
        // Create a 2x2 texture, but specify the width and height of the
        // external texture to be 1x1. (0, 1), (1, 0), and (1, 1) will
        // therefore be invalid addresses, despite falling within the bounds of
        // the underlying texture.
        let view = create_texture_and_view(
            &ctx,
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::Rgba8Unorm,
            Some(RGBA_TEXTURE_DATA.as_flattened()),
        );

        let external_texture = ctx.device.create_external_texture(
            &wgpu::ExternalTextureDescriptor {
                label: None,
                width: 1,
                height: 1,
                format: wgpu::ExternalTextureFormat::Rgba,
                yuv_conversion_matrix: IDENTITY_YUV_CONVERSION_MATRIX,
                gamut_conversion_matrix: IDENTITY_GAMUT_CONVERSION_MATRIX,
                src_transfer_function: Default::default(),
                dst_transfer_function: Default::default(),
                sample_transform: IDENTITY_SAMPLE_TRANSFORM,
                load_transform: IDENTITY_LOAD_TRANSFORM,
            },
            &[&view],
        );

        let loads = get_loads(
            &ctx,
            &[[0, 0], [1, 0], [0, 1], [1, 1]],
            wgpu::BindingResource::ExternalTexture(&external_texture),
        );
        for load in &loads {
            // From https://www.w3.org/TR/WGSL/#textureload:
            // If the logical texel address is invalid, the built-in function returns one of:
            //   * The data for some texel within bounds of the texture
            //   * A vector (0,0,0,0) or (0,0,0,1) of the appropriate type for non-depth textures
            //   * 0.0 for depth textures
            // We therefore expect the loaded values to be red, opaque black,
            // or transparent black. They must not be green, blue, or yellow.
            assert!([RED_F32, OPAQUE_BLACK_F32, TRANSPARENT_BLACK_F32].contains(load));
        }
    });

/// Tests that `textureSampleBaseClampToEdge()` returns the correct values for both RGBA
/// format external textures and texture views bound to an external texture binding.
#[gpu_test]
static EXTERNAL_TEXTURE_SAMPLE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::EXTERNAL_TEXTURE),
    )
    .run_async(|ctx| async move {
        let view = create_texture_and_view(
            &ctx,
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::Rgba8Unorm,
            Some(RGBA_TEXTURE_DATA.as_flattened()),
        );
        let samples = get_samples(
            &ctx,
            &[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]],
            wgpu::BindingResource::TextureView(&view),
        );
        assert_eq!(&samples, &[RED_F32, GREEN_F32, BLUE_F32, YELLOW_F32]);

        let external_texture = ctx.device.create_external_texture(
            &wgpu::ExternalTextureDescriptor {
                label: None,
                width: 2,
                height: 2,
                format: wgpu::ExternalTextureFormat::Rgba,
                yuv_conversion_matrix: IDENTITY_YUV_CONVERSION_MATRIX,
                gamut_conversion_matrix: IDENTITY_GAMUT_CONVERSION_MATRIX,
                src_transfer_function: Default::default(),
                dst_transfer_function: Default::default(),
                sample_transform: IDENTITY_SAMPLE_TRANSFORM,
                load_transform: IDENTITY_LOAD_TRANSFORM,
            },
            &[&view],
        );

        let samples = get_samples(
            &ctx,
            &[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]],
            wgpu::BindingResource::ExternalTexture(&external_texture),
        );
        assert_eq!(&samples, &[RED_F32, GREEN_F32, BLUE_F32, YELLOW_F32]);
    });

/// Tests that `textureSampleBaseClampToEdge()` returns the correct values for YUV
/// format external textures.
#[gpu_test]
static EXTERNAL_TEXTURE_SAMPLE_YUV: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::EXTERNAL_TEXTURE),
    )
    .run_async(|ctx| async move {
        let y_view = create_texture_and_view(
            &ctx,
            wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::R8Unorm,
            Some(&Y_TEXTURE_DATA),
        );
        let u_view = create_texture_and_view(
            &ctx,
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::R8Unorm,
            Some(&U_TEXTURE_DATA),
        );
        let v_view = create_texture_and_view(
            &ctx,
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::R8Unorm,
            Some(&V_TEXTURE_DATA),
        );

        let external_texture = ctx.device.create_external_texture(
            &wgpu::ExternalTextureDescriptor {
                label: None,
                width: 4,
                height: 4,
                format: wgpu::ExternalTextureFormat::Yu12,
                yuv_conversion_matrix: BT601_YUV_CONVERSION_MATRIX,
                gamut_conversion_matrix: BT601_TO_SRGB_GAMUT_CONVERSION_MATRIX,
                src_transfer_function: SRGB_TRANSFER_FUNCTION,
                dst_transfer_function: SRGB_TRANSFER_FUNCTION,
                sample_transform: IDENTITY_SAMPLE_TRANSFORM,
                load_transform: IDENTITY_LOAD_TRANSFORM,
            },
            &[&y_view, &u_view, &v_view],
        );
        let samples = get_samples(
            &ctx,
            &[
                [0.375, 0.375],
                [0.625, 0.375],
                [0.375, 0.625],
                [0.625, 0.625],
            ],
            wgpu::BindingResource::ExternalTexture(&external_texture),
        );

        // `assert_abs_diff_eq!()` works for slices but not arrays, hence the
        // following tomfoolery.
        let samples = samples.iter().map(|arr| arr.as_slice()).collect::<Vec<_>>();
        let expected = [RED_SRGB_F32, GREEN_SRGB_F32, BLUE_SRGB_F32, YELLOW_SRGB_F32];
        let expected = expected.each_ref().map(|arr| arr.as_slice());
        // We expect slight inaccuracies due to floating point maths.
        assert_abs_diff_eq!(samples.as_slice(), expected.as_slice(), epsilon = 0.01);
    });

/// Tests that `textureSampleBaseClampToEdge()` returns the correct values for external
/// textures with various sample transforms.
#[gpu_test]
static EXTERNAL_TEXTURE_SAMPLE_TRANSFORM: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .test_features_limits()
            .features(wgpu::Features::EXTERNAL_TEXTURE),
    )
    .run_async(|ctx| async move {
        let view_2x2 = create_texture_and_view(
            &ctx,
            wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::Rgba8Unorm,
            Some(RGBA_TEXTURE_DATA.as_flattened()),
        );

        let flip_h_external_texture = ctx.device.create_external_texture(
            &wgpu::ExternalTextureDescriptor {
                label: None,
                width: 2,
                height: 2,
                format: wgpu::ExternalTextureFormat::Rgba,
                yuv_conversion_matrix: IDENTITY_YUV_CONVERSION_MATRIX,
                gamut_conversion_matrix: IDENTITY_GAMUT_CONVERSION_MATRIX,
                src_transfer_function: Default::default(),
                dst_transfer_function: Default::default(),
                sample_transform: HORIZONTAL_FLIP_2X2_SAMPLE_TRANSFORM,
                load_transform: HORIZONTAL_FLIP_2X2_LOAD_TRANSFORM,
            },
            &[&view_2x2],
        );
        let samples = get_samples(
            &ctx,
            &[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]],
            wgpu::BindingResource::ExternalTexture(&flip_h_external_texture),
        );
        assert_eq!(&samples, &[GREEN_F32, RED_F32, YELLOW_F32, BLUE_F32]);

        let flip_v_external_texture = ctx.device.create_external_texture(
            &wgpu::ExternalTextureDescriptor {
                label: None,
                width: 2,
                height: 2,
                format: wgpu::ExternalTextureFormat::Rgba,
                yuv_conversion_matrix: IDENTITY_YUV_CONVERSION_MATRIX,
                gamut_conversion_matrix: IDENTITY_GAMUT_CONVERSION_MATRIX,
                src_transfer_function: Default::default(),
                dst_transfer_function: Default::default(),
                sample_transform: VERTICAL_FLIP_2X2_SAMPLE_TRANSFORM,
                load_transform: VERTICAL_FLIP_2X2_LOAD_TRANSFORM,
            },
            &[&view_2x2],
        );
        let samples = get_samples(
            &ctx,
            &[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]],
            wgpu::BindingResource::ExternalTexture(&flip_v_external_texture),
        );
        assert_eq!(&samples, &[BLUE_F32, YELLOW_F32, RED_F32, GREEN_F32]);

        // Use a non-square texture for the rotation cases as it's more
        // interesting
        let view_4x1 = create_texture_and_view(
            &ctx,
            wgpu::Extent3d {
                width: 4,
                height: 1,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::Rgba8Unorm,
            Some(RGBA_TEXTURE_DATA.as_flattened()),
        );

        let rotate_90_external_texture = ctx.device.create_external_texture(
            &wgpu::ExternalTextureDescriptor {
                label: None,
                width: 1,
                height: 4,
                format: wgpu::ExternalTextureFormat::Rgba,
                yuv_conversion_matrix: IDENTITY_YUV_CONVERSION_MATRIX,
                gamut_conversion_matrix: IDENTITY_GAMUT_CONVERSION_MATRIX,
                src_transfer_function: Default::default(),
                dst_transfer_function: Default::default(),
                sample_transform: ROTATE_90_4X1_SAMPLE_TRANSFORM,
                load_transform: ROTATE_90_4X1_LOAD_TRANSFORM,
            },
            &[&view_4x1],
        );
        let samples = get_samples(
            &ctx,
            &[[0.5, 0.125], [0.5, 0.375], [0.5, 0.625], [0.5, 0.875]],
            wgpu::BindingResource::ExternalTexture(&rotate_90_external_texture),
        );
        assert_eq!(&samples, &[RED_F32, GREEN_F32, BLUE_F32, YELLOW_F32]);

        let rotate_180_external_texture = ctx.device.create_external_texture(
            &wgpu::ExternalTextureDescriptor {
                label: None,
                width: 4,
                height: 1,
                format: wgpu::ExternalTextureFormat::Rgba,
                yuv_conversion_matrix: IDENTITY_YUV_CONVERSION_MATRIX,
                gamut_conversion_matrix: IDENTITY_GAMUT_CONVERSION_MATRIX,
                src_transfer_function: Default::default(),
                dst_transfer_function: Default::default(),
                sample_transform: ROTATE_180_4X1_SAMPLE_TRANSFORM,
                load_transform: ROTATE_180_4X1_LOAD_TRANSFORM,
            },
            &[&view_4x1],
        );
        let samples = get_samples(
            &ctx,
            &[[0.125, 0.5], [0.375, 0.5], [0.625, 0.5], [0.875, 0.5]],
            wgpu::BindingResource::ExternalTexture(&rotate_180_external_texture),
        );
        assert_eq!(&samples, &[YELLOW_F32, BLUE_F32, GREEN_F32, RED_F32]);

        let rotate_270_external_texture = ctx.device.create_external_texture(
            &wgpu::ExternalTextureDescriptor {
                label: None,
                width: 1,
                height: 4,
                format: wgpu::ExternalTextureFormat::Rgba,
                yuv_conversion_matrix: IDENTITY_YUV_CONVERSION_MATRIX,
                gamut_conversion_matrix: IDENTITY_GAMUT_CONVERSION_MATRIX,
                src_transfer_function: Default::default(),
                dst_transfer_function: Default::default(),
                sample_transform: ROTATE_270_4X1_SAMPLE_TRANSFORM,
                load_transform: ROTATE_270_4X1_LOAD_TRANSFORM,
            },
            &[&view_4x1],
        );
        let samples = get_samples(
            &ctx,
            &[[0.5, 0.125], [0.5, 0.375], [0.5, 0.625], [0.5, 0.875]],
            wgpu::BindingResource::ExternalTexture(&rotate_270_external_texture),
        );
        assert_eq!(&samples, &[YELLOW_F32, BLUE_F32, GREEN_F32, RED_F32]);

        let view_4x4 = create_texture_and_view(
            &ctx,
            wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            wgpu::TextureFormat::Rgba8Unorm,
            Some(RGBA_TEXTURE_DATA_WITH_BORDER.as_flattened()),
        );
        let crop_tex = ctx.device.create_external_texture(
            &wgpu::ExternalTextureDescriptor {
                label: None,
                width: 2,
                height: 2,
                format: wgpu::ExternalTextureFormat::Rgba,
                yuv_conversion_matrix: IDENTITY_YUV_CONVERSION_MATRIX,
                gamut_conversion_matrix: IDENTITY_GAMUT_CONVERSION_MATRIX,
                src_transfer_function: Default::default(),
                dst_transfer_function: Default::default(),
                sample_transform: CROP_4X4_SAMPLE_TRANSFORM,
                load_transform: CROP_4X4_LOAD_TRANSFORM,
            },
            &[&view_4x4],
        );
        // Deliberately sample from the edges of the external texture rather
        // than the texel centres. This tests that clamping to a half-texel
        // from the edge works as expected even when the external texture is
        // cropped from a larger texture. If this weren't working, the black
        // border would affect the sampled values.
        let samples = get_samples(
            &ctx,
            &[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            wgpu::BindingResource::ExternalTexture(&crop_tex),
        );
        assert_eq!(&samples, &[RED_F32, GREEN_F32, BLUE_F32, YELLOW_F32]);
    });
