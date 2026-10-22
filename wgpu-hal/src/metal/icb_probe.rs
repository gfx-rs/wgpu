use core::{ptr::NonNull, slice};

use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLArgumentEncoder, MTLBlitCommandEncoder, MTLBuffer, MTLClearColor, MTLCommandBuffer,
    MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLFunction, MTLLanguageVersion, MTLLibrary, MTLLoadAction, MTLPixelFormat,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderPipelineDescriptor,
    MTLResourceOptions, MTLResourceUsage, MTLSize, MTLStorageMode, MTLStoreAction,
    MTLTextureDescriptor, MTLTextureUsage,
};

const PROBE_SHADER: &str = include_str!("./shaders/icb_probe.metal");

/// Whether `device` executes a GPU-generated render ICB correctly: one draw,
/// generated, executed and read back exactly as the multi-draw lowering does
/// it, with the same ICB descriptor and with buffers inherited at the slots
/// wgpu binds them to. `optimize` mirrors
/// `PrivateCapabilities::indirect_command_buffers_optimize`.
pub(super) fn supports_render_icb(
    device: &ProtocolObject<dyn MTLDevice>,
    msl_version: MTLLanguageVersion,
    optimize: bool,
) -> bool {
    let library = match super::device::compile_msl_library(device, msl_version, false, PROBE_SHADER)
    {
        Ok(library) => library,
        Err(error) => {
            log::debug!("Metal render ICB probe shader compilation failed: {error}");
            return false;
        }
    };
    let Some(vertex) = library.newFunctionWithName(&NSString::from_str("probe_vertex")) else {
        return false;
    };
    let Some(fragment) = library.newFunctionWithName(&NSString::from_str("probe_fragment")) else {
        return false;
    };
    let Some(generate) = library.newFunctionWithName(&NSString::from_str("probe_generate")) else {
        return false;
    };
    let generation_pipeline = match device.newComputePipelineStateWithFunction_error(&generate) {
        Ok(pipeline) => pipeline,
        Err(error) => {
            log::debug!("Metal render ICB probe generation pipeline creation failed: {error}");
            return false;
        }
    };

    let pipeline_descriptor = MTLRenderPipelineDescriptor::new();
    pipeline_descriptor.setVertexFunction(Some(&vertex));
    pipeline_descriptor.setFragmentFunction(Some(&fragment));
    pipeline_descriptor.setSupportIndirectCommandBuffers(true);
    unsafe {
        pipeline_descriptor
            .colorAttachments()
            .objectAtIndexedSubscript(0)
            .setPixelFormat(MTLPixelFormat::RGBA8Unorm);
    }
    let pipeline = match device.newRenderPipelineStateWithDescriptor_error(&pipeline_descriptor) {
        Ok(pipeline) => pipeline,
        Err(error) => {
            log::debug!("Metal render ICB probe pipeline creation failed: {error}");
            return false;
        }
    };

    let texture_descriptor = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            MTLPixelFormat::RGBA8Unorm,
            1,
            1,
            false,
        )
    };
    texture_descriptor.setStorageMode(MTLStorageMode::Private);
    texture_descriptor.setUsage(MTLTextureUsage::RenderTarget);
    let Some(texture) = device.newTextureWithDescriptor(&texture_descriptor) else {
        return false;
    };
    let Some(readback) =
        device.newBufferWithLength_options(256, MTLResourceOptions::StorageModeShared)
    else {
        return false;
    };
    // One triangle covering the target, and the color it is drawn in.
    let positions: [f32; 6] = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
    let color: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    let Some(vertex_buffer) = (unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::from(&positions).cast(),
            size_of_val(&positions),
            MTLResourceOptions::StorageModeShared,
        )
    }) else {
        return false;
    };
    let Some(color_buffer) = (unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::from(&color).cast(),
            size_of_val(&color),
            MTLResourceOptions::StorageModeShared,
        )
    }) else {
        return false;
    };

    let Some(icb) = (unsafe {
        device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
            &super::icb::icb_descriptor(super::icb::ICB_KIND_DRAW),
            1,
            MTLResourceOptions::StorageModePrivate,
        )
    }) else {
        return false;
    };

    let argument_encoder = unsafe { generate.newArgumentEncoderWithBufferIndex(0) };
    let Some(argument_buffer) = device.newBufferWithLength_options(
        argument_encoder.encodedLength(),
        MTLResourceOptions::StorageModeShared,
    ) else {
        return false;
    };
    unsafe {
        argument_encoder.setArgumentBuffer_offset(Some(&argument_buffer), 0);
        argument_encoder.setIndirectCommandBuffer_atIndex(Some(&icb), 0);
    }

    let pass = MTLRenderPassDescriptor::renderPassDescriptor();
    let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
    attachment.setTexture(Some(&texture));
    attachment.setLoadAction(MTLLoadAction::Clear);
    attachment.setStoreAction(MTLStoreAction::Store);
    attachment.setClearColor(MTLClearColor {
        red: 0.0,
        green: 0.0,
        blue: 0.0,
        alpha: 0.0,
    });

    let Some(queue) = device.newCommandQueue() else {
        return false;
    };

    let Some(generation_buffer) = queue.commandBuffer() else {
        return false;
    };
    let Some(compute) = generation_buffer.computeCommandEncoder() else {
        return false;
    };
    compute.setComputePipelineState(&generation_pipeline);
    unsafe {
        compute.setBuffer_offset_atIndex(Some(&argument_buffer), 0, 0);
        compute.useResource_usage(ProtocolObject::from_ref(&*icb), MTLResourceUsage::Write);
    }
    // Uniform threadgroups, like the generation kernels themselves:
    // `dispatchThreads:` (non-uniform threadgroup sizes) is an Apple4+ API,
    // and on Apple3 Metal raises an exception for it rather than returning
    // an error, which would abort the process instead of failing the probe.
    compute.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    compute.endEncoding();
    if optimize {
        let Some(blit) = generation_buffer.blitCommandEncoder() else {
            return false;
        };
        unsafe {
            blit.optimizeIndirectCommandBuffer_withRange(
                &icb,
                NSRange {
                    location: 0,
                    length: 1,
                },
            );
        }
        blit.endEncoding();
    }
    generation_buffer.commit();
    generation_buffer.waitUntilCompleted();
    if generation_buffer.status() != MTLCommandBufferStatus::Completed {
        log::debug!(
            "Metal render ICB generation probe failed: {:?}",
            generation_buffer.error()
        );
        return false;
    }

    let Some(command_buffer) = queue.commandBuffer() else {
        return false;
    };
    let Some(render) = command_buffer.renderCommandEncoderWithDescriptor(&pass) else {
        return false;
    };
    render.setRenderPipelineState(&pipeline);
    unsafe {
        render.setVertexBuffer_offset_atIndex(Some(&vertex_buffer), 0, 30);
        render.setFragmentBuffer_offset_atIndex(Some(&color_buffer), 0, 0);
    }
    #[expect(deprecated)]
    render.useResource_usage(ProtocolObject::from_ref(&*icb), MTLResourceUsage::Read);
    unsafe {
        render.executeCommandsInBuffer_withRange(
            &icb,
            NSRange {
                location: 0,
                length: 1,
            },
        );
    }
    render.endEncoding();

    let Some(blit) = command_buffer.blitCommandEncoder() else {
        return false;
    };
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            &texture,
            0,
            0,
            objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize { width: 1, height: 1, depth: 1 },
            &readback,
            0,
            256,
            256,
        );
    }
    blit.endEncoding();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    if command_buffer.status() != MTLCommandBufferStatus::Completed {
        log::debug!(
            "Metal render ICB probe command failed: {:?}",
            command_buffer.error()
        );
        return false;
    }

    let pixel = unsafe { slice::from_raw_parts(readback.contents().as_ptr().cast::<u8>(), 4) };
    pixel == [255, 0, 0, 255]
}
