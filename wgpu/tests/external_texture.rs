#![cfg(all(target_arch = "wasm32", not(features = "emscripten")))]

use std::num::NonZeroU32;

use crate::common::{fail_if, initialize_test, TestParameters};
use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
async fn image_bitmap_import() {
    let image_encoded = include_bytes!("3x3_colors.png");

    // Create an array-of-arrays for Blob's constructor
    let array = js_sys::Array::new();
    array.push(&js_sys::Uint8Array::from(&image_encoded[..]));

    // We're passing an array of Uint8Arrays
    let blob = web_sys::Blob::new_with_u8_array_sequence(&array).unwrap();

    // Parse the image from the blob
    let image_bitmap_promise = web_sys::window()
        .unwrap()
        .create_image_bitmap_with_blob(&blob)
        .unwrap();

    // Wait for the parsing to be done
    let image_bitmap: web_sys::ImageBitmap =
        wasm_bindgen_futures::JsFuture::from(image_bitmap_promise)
            .await
            .unwrap()
            .dyn_into()
            .unwrap();

    // Sanity checks
    assert_eq!(image_bitmap.width(), 3);
    assert_eq!(image_bitmap.height(), 3);

    // Decode it cpu side
    let raw_image = image::load_from_memory_with_format(image_encoded, image::ImageFormat::Png)
        .unwrap()
        .into_rgba8();

    // Set of test cases to test with image import
    #[derive(Debug)]
    enum TestCase {
        // Import the image as normal
        Normal,
        // Set both the input offset and output offset to 1 in x, so the first column is omitted.
        TrimLeft,
        // Set the size to 2 in x, so the last column is omitted
        TrimRight,
        // Set only the output offset to 1, so the second column gets the first column's data.
        SlideRight,
        // Try to copy from out of bounds of the source image
        SourceOutOfBounds,
        // Try to copy from out of bounds of the destination image
        DestOutOfBounds,
        // Try to copy more than one slice from the source
        MultiSliceCopy,
        // Copy into the second slice of a 2D array texture,
        SecondSliceCopy,
    }
    let cases = [
        TestCase::Normal,
        TestCase::TrimLeft,
        TestCase::TrimRight,
        TestCase::SlideRight,
        TestCase::SourceOutOfBounds,
        TestCase::DestOutOfBounds,
        TestCase::MultiSliceCopy,
        TestCase::SecondSliceCopy,
    ];

    initialize_test(TestParameters::default(), |ctx| {
        for case in cases {
            // Copy the data, so we can modify it for tests
            let mut raw_image = raw_image.clone();
            // The origin used for the external copy on the source side.
            let mut src_origin = wgpu::Origin2d::ZERO;
            // The origin used for the external copy on the destination side.
            let mut dest_origin = wgpu::Origin3d::ZERO;
            // The layer the external image's data should end up in.
            let mut dest_data_layer = 0;
            // Size of the external copy
            let mut copy_size = wgpu::Extent3d {
                width: 3,
                height: 3,
                depth_or_array_layers: 1,
            };
            // Width of the destination texture
            let mut dest_width = 3;
            // Layer count of the destination texture
            let mut dest_layers = 1;

            // If the test is suppoed to be valid call to copyExternal.
            let mut valid = true;
            // If the result is incorrect
            let mut correct = true;
            match case {
                TestCase::Normal => {}
                TestCase::TrimLeft => {
                    valid = ctx
                        .adapter_downlevel_capabilities
                        .flags
                        .contains(wgt::DownlevelFlags::UNRESTRICTED_EXTERNAL_TEXTURE_COPIES);
                    src_origin.x = 1;
                    dest_origin.x = 1;
                    copy_size.width = 2;
                    for y in 0..3 {
                        raw_image[(0, y)].0 = [0; 4];
                    }
                }
                TestCase::TrimRight => {
                    copy_size.width = 2;
                    for y in 0..3 {
                        raw_image[(2, y)].0 = [0; 4];
                    }
                }
                TestCase::SlideRight => {
                    dest_origin.x = 1;
                    copy_size.width = 2;
                    for x in (1..3).rev() {
                        for y in 0..3 {
                            raw_image[(x, y)].0 = raw_image[(x - 1, y)].0;
                        }
                    }
                    for y in 0..3 {
                        raw_image[(0, y)].0 = [0; 4];
                    }
                }
                TestCase::SourceOutOfBounds => {
                    valid = false;
                    // It's now in bounds for the destination
                    dest_width = 4;
                    copy_size.width = 4;
                }
                TestCase::DestOutOfBounds => {
                    valid = false;
                    // It's now out bounds for the destination
                    dest_width = 2;
                }
                TestCase::MultiSliceCopy => {
                    valid = false;
                    copy_size.depth_or_array_layers = 2;
                    dest_layers = 2;
                }
                TestCase::SecondSliceCopy => {
                    correct = false; // TODO: what?
                    dest_origin.z = 1;
                    dest_data_layer = 1;
                    dest_layers = 2;
                }
            }

            let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("import dest"),
                size: wgpu::Extent3d {
                    width: dest_width,
                    height: 3,
                    depth_or_array_layers: dest_layers,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
            });

            fail_if(&ctx.device, !valid, || {
                ctx.queue.copy_external_image_to_texture(
                    &wgpu::ImageCopyExternalImage {
                        source: wgpu::ExternalImageSource::ImageBitmap(image_bitmap.clone()),
                        origin: src_origin,
                        flip_y: false,
                    },
                    wgpu::ImageCopyTextureTagged {
                        texture: &texture,
                        mip_level: 0,
                        origin: dest_origin,
                        aspect: wgpu::TextureAspect::All,
                        color_space: wgpu::PredefinedColorSpace::Srgb,
                        premultiplied_alpha: false,
                    },
                    copy_size,
                );
            });

            let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback buffer"),
                size: 4 * 64 * 3,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: dest_data_layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &readback_buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(NonZeroU32::new(256).unwrap()),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width: dest_width,
                    height: 3,
                    depth_or_array_layers: 1,
                },
            );

            ctx.queue.submit(Some(encoder.finish()));
            readback_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, |_| ());
            ctx.device.poll(wgpu::Maintain::Wait);

            let buffer = readback_buffer.slice(..).get_mapped_range();

            // 64 because of 256 byte alignment / 4.
            let gpu_image = image::RgbaImage::from_vec(64, 3, buffer.to_vec()).unwrap();
            let gpu_image_cropped = image::imageops::crop_imm(&gpu_image, 0, 0, 3, 3).to_image();

            if valid && correct {
                assert_eq!(raw_image, gpu_image_cropped, "Failed on test case {case:?}");
            } else {
                assert_ne!(raw_image, gpu_image_cropped, "Failed on test case {case:?}");
            }
        }
    })
}
