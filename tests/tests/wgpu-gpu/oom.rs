use wgpu::{
    AccelerationStructureFlags, AccelerationStructureGeometryFlags,
    AccelerationStructureUpdateMode, Backends, BlasGeometrySizeDescriptors,
    BlasTriangleGeometrySizeDescriptor, BufferDescriptor, BufferUsages, CreateBlasDescriptor,
    CreateTlasDescriptor, Error, ErrorFilter, Extent3d, Features, QuerySetDescriptor, QueryType,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, VertexFormat,
};
use wgpu_test::{gpu_test, FailureCase, GpuTestConfiguration, TestParameters};

// Tests in this file must all end with "OOM_TEST" so that nextest doesn't run any other tests while it runs one of the OOM tests.
// This is done so that other tests that create resources will not fail with OOM errors due to the OOM tests running in parallel.

/// Backends for which OOM detection is implemented
const OOM_DETECTION_IMPL: Backends = Backends::DX12.union(Backends::VULKAN);

/// Backends for which query set OOM detection is implemented
const QUERY_SET_OOM_DETECTION_IMPL: Backends = Backends::DX12;

// All tests skip llvmpipe.
// Even though llvmpipe supports VK_EXT_memory_budget it's happy to continue creating resources until
// the process crashes with SIGABRT "memory allocation of X bytes failed" or the test times out.

/// Nr of resources tests will try to create before failing.
const LOOP_BOUND: u32 = 1_000_000;

#[gpu_test]
static TEXTURE_OOM_TEST: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .skip(FailureCase::backend(!OOM_DETECTION_IMPL))
            // see comment at the top of the file
            .skip(FailureCase::backend_adapter(Backends::VULKAN, "llvmpipe")),
    )
    .run_async(|ctx| async move {
        let mut textures = Vec::new();
        for _ in 0..LOOP_BOUND {
            ctx.device.push_error_scope(ErrorFilter::OutOfMemory);
            let texture = ctx.device.create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: 2048,
                    height: 2048,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            if let Some(err) = ctx.device.pop_error_scope().await {
                match err {
                    Error::OutOfMemory { .. } => {
                        return;
                    }
                    _ => unreachable!(),
                }
            }
            textures.push(texture);
        }
        panic!("Failed to OOM after {LOOP_BOUND} iterations.");
    });

#[gpu_test]
static BUFFER_OOM_TEST: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .skip(FailureCase::backend(!OOM_DETECTION_IMPL))
            // see comment at the top of the file
            .skip(FailureCase::backend_adapter(Backends::VULKAN, "llvmpipe")),
    )
    .run_async(|ctx| async move {
        let mut buffers = Vec::new();
        for _ in 0..LOOP_BOUND {
            ctx.device.push_error_scope(ErrorFilter::OutOfMemory);
            let buffer = ctx.device.create_buffer(&BufferDescriptor {
                label: None,
                size: 256 * 1024 * 1024,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            if let Some(err) = ctx.device.pop_error_scope().await {
                match err {
                    Error::OutOfMemory { .. } => {
                        return;
                    }
                    _ => unreachable!(),
                }
            }
            buffers.push(buffer);
        }
        panic!("Failed to OOM after {LOOP_BOUND} iterations.");
    });

#[gpu_test]
static MAPPING_BUFFER_OOM_TEST: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .skip(FailureCase::backend(!OOM_DETECTION_IMPL))
            // see comment at the top of the file
            .skip(FailureCase::backend_adapter(Backends::VULKAN, "llvmpipe")),
    )
    .run_async(|ctx| async move {
        let mut buffers = Vec::new();
        for _ in 0..LOOP_BOUND {
            ctx.device.push_error_scope(ErrorFilter::OutOfMemory);
            let buffer = ctx.device.create_buffer(&BufferDescriptor {
                label: None,
                size: 256 * 1024 * 1024,
                usage: BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
                mapped_at_creation: false,
            });
            if let Some(err) = ctx.device.pop_error_scope().await {
                match err {
                    Error::OutOfMemory { .. } => {
                        return;
                    }
                    _ => unreachable!(),
                }
            }
            buffers.push(buffer);
        }
        panic!("Failed to OOM after {LOOP_BOUND} iterations.");
    });

#[gpu_test]
static QUERY_SET_OOM_TEST: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            // Vulkan: https://github.com/gfx-rs/wgpu/issues/7817
            .skip(FailureCase::backend(!QUERY_SET_OOM_DETECTION_IMPL))
            // see comment at the top of the file
            .skip(FailureCase::backend_adapter(Backends::VULKAN, "llvmpipe")),
    )
    .run_async(|ctx| async move {
        let mut query_sets = Vec::new();
        for _ in 0..LOOP_BOUND {
            ctx.device.push_error_scope(ErrorFilter::OutOfMemory);
            let query_set = ctx.device.create_query_set(&QuerySetDescriptor {
                label: None,
                ty: QueryType::Occlusion,
                count: 4096,
            });
            if let Some(err) = ctx.device.pop_error_scope().await {
                match err {
                    Error::OutOfMemory { .. } => {
                        return;
                    }
                    _ => unreachable!(),
                }
            }
            query_sets.push(query_set);
        }
        panic!("Failed to OOM after {LOOP_BOUND} iterations.");
    });

#[gpu_test]
static BLAS_OOM_TEST: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::EXPERIMENTAL_RAY_QUERY)
            .skip(FailureCase::backend(!OOM_DETECTION_IMPL))
            // see comment at the top of the file
            .skip(FailureCase::backend_adapter(Backends::VULKAN, "llvmpipe")),
    )
    .run_async(|ctx| async move {
        let mut blases = Vec::new();
        for _ in 0..LOOP_BOUND {
            ctx.device.push_error_scope(ErrorFilter::OutOfMemory);
            let blas = ctx.device.create_blas(
                &CreateBlasDescriptor {
                    label: None,
                    flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: AccelerationStructureUpdateMode::Build,
                },
                BlasGeometrySizeDescriptors::Triangles {
                    descriptors: vec![BlasTriangleGeometrySizeDescriptor {
                        vertex_format: VertexFormat::Float32x3,
                        vertex_count: 1024 * 1024,
                        index_format: None,
                        index_count: None,
                        flags: AccelerationStructureGeometryFlags::OPAQUE,
                    }],
                },
            );
            if let Some(err) = ctx.device.pop_error_scope().await {
                match err {
                    Error::OutOfMemory { .. } => {
                        return;
                    }
                    _ => unreachable!(),
                }
            }
            blases.push(blas);
        }
        panic!("Failed to OOM after {LOOP_BOUND} iterations.");
    });

#[gpu_test]
static TLAS_OOM_TEST: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::EXPERIMENTAL_RAY_QUERY)
            .skip(FailureCase::backend(!OOM_DETECTION_IMPL))
            // see comment at the top of the file
            .skip(FailureCase::backend_adapter(Backends::VULKAN, "llvmpipe")),
    )
    .run_async(|ctx| async move {
        let mut tlases = Vec::new();
        for _ in 0..LOOP_BOUND {
            ctx.device.push_error_scope(ErrorFilter::OutOfMemory);
            let tlas = ctx.device.create_tlas(&CreateTlasDescriptor {
                label: None,
                max_instances: 1024 * 1024,
                flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: AccelerationStructureUpdateMode::Build,
            });
            if let Some(err) = ctx.device.pop_error_scope().await {
                match err {
                    Error::OutOfMemory { .. } => {
                        return;
                    }
                    _ => unreachable!(),
                }
            }
            tlases.push(tlas);
        }
        panic!("Failed to OOM after {LOOP_BOUND} iterations.");
    });
