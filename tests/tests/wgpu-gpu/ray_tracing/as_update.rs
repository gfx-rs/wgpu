use std::{mem::size_of, sync::mpsc};

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::AccelerationStructureBuildMode::{Build, Update};
use wgpu::*;
use wgpu_test::{
    apply, fail, gpu_test, valid, GpuTestConfiguration, GpuTestInitializer, TestParameters,
    TestingContext,
};

use crate::ray_tracing::acceleration_structure_limits;

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        BLAS_UPDATE_MOVES_HITS,
        INDEXED_BLAS_UPDATE_MOVES_HITS,
        TRANSFORMED_BLAS_UPDATE_MOVES_HITS,
        AABB_BLAS_UPDATE_MOVES_HITS,
        TLAS_UPDATE_MOVES_HITS,
        EMPTY_TLAS_UPDATE,
        ZERO_GEOMETRY_BLAS_UPDATE,
        BLAS_UPDATE_WITH_SAME_INPUTS,
        BLAS_UPDATE_WITHOUT_ALLOW_UPDATE,
        BLAS_UPDATE_WHEN_NEVER_BUILT,
        BLAS_UPDATE_AFTER_VERTEX_COUNT_CHANGE,
        BLAS_UPDATE_AFTER_GEOMETRY_COUNT_CHANGE,
        BLAS_UPDATE_AFTER_STRIDE_CHANGE,
        INDEXED_BLAS_UPDATE_WITH_SAME_INPUTS,
        INDEXED_BLAS_UPDATE_AFTER_INDEX_COUNT_CHANGE,
        AABB_BLAS_UPDATE_WITH_SAME_INPUTS,
        AABB_BLAS_UPDATE_AFTER_PRIMITIVE_COUNT_CHANGE,
        AABB_BLAS_UPDATE_AFTER_STRIDE_CHANGE,
        TLAS_UPDATE_WITH_SAME_INSTANCE_COUNT,
        TLAS_UPDATE_WITHOUT_ALLOW_UPDATE,
        TLAS_UPDATE_WHEN_NEVER_BUILT,
        TLAS_UPDATE_AFTER_INSTANCE_COUNT_CHANGE,
        UPDATE_BEFORE_BUILD_IN_SAME_COMMAND_BUFFER,
        UPDATE_SUBMISSION_ORDER,
        UPDATE_AFTER_FAILED_SUBMISSION,
        UPDATE_AFTER_MARK_BUILT,
        BLAS_UPDATE_AFTER_COMPACTION,
        BLAS_COMPACTION_AFTER_UPDATE,
        DUPLICATE_BLAS_BUILD,
        DUPLICATE_TLAS_BUILD,
    ]);
}

const UPDATABLE: AccelerationStructureFlags = AccelerationStructureFlags::ALLOW_UPDATE;
const NOT_UPDATABLE: AccelerationStructureFlags = AccelerationStructureFlags::empty();

const INVALID_UPDATE: &str = "is updated, but it was not built";
const MISSING_ALLOW_UPDATE: &str = "was not created with AccelerationStructureFlags::ALLOW_UPDATE";

const NUM_RAYS: usize = 2;
const MISS: Option<f32> = None;
const VERTEX_STRIDE: BufferAddress = size_of::<[f32; 3]>() as BufferAddress;
const AABB_STRIDE: BufferAddress = size_of::<Aabb>() as BufferAddress;

fn update_test_parameters() -> TestParameters {
    TestParameters::default()
        .test_features_limits()
        .limits(acceleration_structure_limits())
        .features(Features::EXPERIMENTAL_RAY_QUERY)
}

/// A triangle in the plane `z`, hit by the ray cast from `(x, 0, 0)`.
fn triangle(x: f32, z: f32) -> [[f32; 3]; 3] {
    [[x - 1.0, -1.0, z], [x + 1.0, -1.0, z], [x, 1.0, z]]
}

fn translation(x: f32, z: f32) -> [f32; 12] {
    [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, z]
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Aabb {
    min: [f32; 3],
    max: [f32; 3],
}

/// A box starting at the plane `z`, hit by the ray cast from `(x, 0, 0)`.
fn aabb(x: f32, z: f32) -> Aabb {
    Aabb {
        min: [x - 1.0, -1.0, z],
        max: [x + 1.0, 1.0, z + 1.0],
    }
}

/// Submits `command_buffer`, which updates an acceleration structure, and checks that the
/// submission is valid exactly when `valid_update` is set.
fn submit_update(ctx: &TestingContext, command_buffer: CommandBuffer, valid_update: bool) {
    if valid_update {
        valid(&ctx.device, || ctx.queue.submit([command_buffer]));
    } else {
        fail(
            &ctx.device,
            || ctx.queue.submit([command_buffer]),
            Some(INVALID_UPDATE),
        );
    }
}

/// Casts the rays and reads back their committed intersections.
struct Tracer {
    pipeline: ComputePipeline,
    out: Buffer,
    readback: Buffer,
}

impl Tracer {
    fn new(ctx: &TestingContext, entry_point: &str) -> Self {
        let shader = ctx
            .device
            .create_shader_module(include_wgsl!("as_update.wgsl"));
        let pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });

        let out_size = (NUM_RAYS * size_of::<[f32; 2]>()) as BufferAddress;
        let out = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("out"),
            size: out_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("readback"),
            size: out_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            out,
            readback,
        }
    }

    /// Returns the `t` of the hit of each ray, or `None` for a miss.
    fn trace(&self, ctx: &TestingContext, tlas: &Tlas) -> [Option<f32>; NUM_RAYS] {
        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::AccelerationStructure(tlas),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.out.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(NUM_RAYS as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.out, 0, &self.readback, 0, self.out.size());
        ctx.queue.submit([encoder.finish()]);

        let (send, recv) = mpsc::channel();
        self.readback
            .map_async(MapMode::Read, .., move |res| send.send(res).unwrap());
        ctx.device.poll(PollType::wait_indefinitely()).unwrap();
        recv.recv().unwrap().unwrap();

        let results = {
            let range = self.readback.get_mapped_range(..).unwrap();
            let hits: &[[f32; 2]] = bytemuck::cast_slice(&range);
            core::array::from_fn(|i| {
                let [kind, t] = hits[i];
                (kind != 0.0).then_some(t)
            })
        };
        self.readback.unmap();
        results
    }
}

/// Which optional inputs the triangle geometries of a [`Scene`] use.
#[derive(Clone, Copy, Default)]
struct Layout {
    indexed: bool,
    transformed: bool,
}

const INDEXED: Layout = Layout {
    indexed: true,
    transformed: false,
};

/// The inputs of one triangle BLAS build.
#[derive(Clone, Copy)]
struct BlasInputs {
    /// The vertex count of each geometry.
    vertex_counts: &'static [u32],
    /// The index count of each geometry of an indexed BLAS, or the vertex counts if `None`.
    index_counts: Option<&'static [u32]>,
    vertex_stride: BufferAddress,
}

impl BlasInputs {
    const fn vertices(vertex_counts: &'static [u32]) -> Self {
        Self {
            vertex_counts,
            index_counts: None,
            vertex_stride: VERTEX_STRIDE,
        }
    }
}

/// A triangle BLAS in a TLAS.
struct Scene {
    vertices: Buffer,
    /// `0, 1, 2, ...`, used if the layout is indexed.
    indices: Buffer,
    /// One 3x4 matrix, used if the layout is transformed.
    transform: Buffer,
    layout: Layout,
    /// One geometry per element, each covering all vertices.
    sizes: Vec<BlasTriangleGeometrySizeDescriptor>,
    blas: Blas,
    tlas: Tlas,
    tracer: Tracer,
}

impl Scene {
    fn new(
        ctx: &TestingContext,
        vertices: &[[f32; 3]],
        geometry_count: usize,
        blas_flags: AccelerationStructureFlags,
        tlas_flags: AccelerationStructureFlags,
    ) -> Self {
        Self::with_layout(
            ctx,
            vertices,
            geometry_count,
            Layout::default(),
            blas_flags,
            tlas_flags,
        )
    }

    fn with_layout(
        ctx: &TestingContext,
        vertices: &[[f32; 3]],
        geometry_count: usize,
        layout: Layout,
        mut blas_flags: AccelerationStructureFlags,
        tlas_flags: AccelerationStructureFlags,
    ) -> Self {
        let vertex_count = vertices.len() as u32;
        let vertices_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vertices"),
            contents: bytemuck::cast_slice(vertices),
            usage: BufferUsages::BLAS_INPUT | BufferUsages::COPY_DST,
        });
        let indices = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("indices"),
            contents: bytemuck::cast_slice(&(0..vertex_count).collect::<Vec<u32>>()),
            usage: BufferUsages::BLAS_INPUT,
        });
        let transform = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("transform"),
            contents: bytemuck::cast_slice(&translation(0.0, 0.0)),
            usage: BufferUsages::BLAS_INPUT | BufferUsages::COPY_DST,
        });

        if layout.transformed {
            blas_flags |= AccelerationStructureFlags::USE_TRANSFORM;
        }
        let size = BlasTriangleGeometrySizeDescriptor {
            vertex_format: VertexFormat::Float32x3,
            vertex_count,
            index_format: layout.indexed.then_some(IndexFormat::Uint32),
            index_count: layout.indexed.then_some(vertex_count),
            flags: AccelerationStructureGeometryFlags::OPAQUE,
        };
        let sizes = vec![size; geometry_count];

        let blas = ctx.device.create_blas(
            &CreateBlasDescriptor {
                label: Some("BLAS"),
                flags: blas_flags,
            },
            BlasGeometrySizeDescriptors::Triangles {
                descriptors: sizes.clone(),
            },
        );

        let mut tlas = ctx.device.create_tlas(&CreateTlasDescriptor {
            label: Some("TLAS"),
            max_instances: 2,
            flags: tlas_flags,
        });
        tlas[0] = Some(TlasInstance::from_blas_untransformed(&blas));

        Self {
            vertices: vertices_buffer,
            indices,
            transform,
            layout,
            sizes,
            blas,
            tlas,
            tracer: Tracer::new(ctx, "trace"),
        }
    }

    fn geometry<'a>(
        &'a self,
        size: &'a BlasTriangleGeometrySizeDescriptor,
        vertex_stride: BufferAddress,
    ) -> BlasTriangleGeometry<'a> {
        BlasTriangleGeometry {
            size,
            vertex_buffer: &self.vertices,
            first_vertex: 0,
            vertex_stride,
            index_buffer: self.layout.indexed.then_some(&self.indices),
            first_index: self.layout.indexed.then_some(0),
            transform_buffer: self.layout.transformed.then_some(&self.transform),
            transform_buffer_offset: self.layout.transformed.then_some(0),
        }
    }

    /// Encodes a build of the BLAS with `blas_inputs` if set, and of the TLAS if `tlas` is set.
    fn encode(
        &self,
        ctx: &TestingContext,
        blas: Option<(BlasInputs, AccelerationStructureBuildMode)>,
        tlas: Option<AccelerationStructureBuildMode>,
    ) -> CommandBuffer {
        let sizes = blas.map(|(inputs, _)| {
            inputs
                .vertex_counts
                .iter()
                .enumerate()
                .map(|(i, &vertex_count)| BlasTriangleGeometrySizeDescriptor {
                    vertex_count,
                    index_count: self
                        .layout
                        .indexed
                        .then(|| inputs.index_counts.unwrap_or(inputs.vertex_counts)[i]),
                    ..self.sizes[i].clone()
                })
                .collect::<Vec<_>>()
        });
        let blas_entry = blas
            .zip(sizes.as_ref())
            .map(|((inputs, mode), sizes)| BlasBuildEntry {
                blas: &self.blas,
                geometry: BlasGeometries::TriangleGeometries(
                    sizes
                        .iter()
                        .map(|size| self.geometry(size, inputs.vertex_stride))
                        .collect(),
                ),
                mode,
            });
        let tlas_entry = tlas.map(|mode| TlasBuildEntry {
            tlas: &self.tlas,
            mode,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.build_acceleration_structures(blas_entry.as_ref(), tlas_entry.as_ref());
        encoder.finish()
    }

    fn encode_blas(
        &self,
        ctx: &TestingContext,
        inputs: BlasInputs,
        mode: AccelerationStructureBuildMode,
    ) -> CommandBuffer {
        self.encode(ctx, Some((inputs, mode)), None)
    }

    fn encode_tlas(
        &self,
        ctx: &TestingContext,
        mode: AccelerationStructureBuildMode,
    ) -> CommandBuffer {
        self.encode(ctx, None, Some(mode))
    }

    /// Encodes a build of the BLAS with all vertices in every geometry, and of the TLAS.
    fn encode_full(
        &self,
        ctx: &TestingContext,
        blas_mode: AccelerationStructureBuildMode,
        tlas_mode: AccelerationStructureBuildMode,
    ) -> CommandBuffer {
        let entry = BlasBuildEntry {
            blas: &self.blas,
            geometry: BlasGeometries::TriangleGeometries(
                self.sizes
                    .iter()
                    .map(|size| self.geometry(size, VERTEX_STRIDE))
                    .collect(),
            ),
            mode: blas_mode,
        };
        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.build_acceleration_structures(
            [&entry],
            [&TlasBuildEntry {
                tlas: &self.tlas,
                mode: tlas_mode,
            }],
        );
        encoder.finish()
    }

    /// Keeps the first `count` instances of the TLAS and removes the others.
    fn set_instance_count(&mut self, count: usize) {
        for index in 0..2 {
            self.tlas[index] =
                (index < count).then(|| TlasInstance::from_blas_untransformed(&self.blas));
        }
    }

    fn trace(&self, ctx: &TestingContext) -> [Option<f32>; NUM_RAYS] {
        self.tracer.trace(ctx, &self.tlas)
    }
}

/// An updatable AABB BLAS with one geometry, in a TLAS.
struct AabbScene {
    aabbs: Buffer,
    size: BlasAABBGeometrySizeDescriptor,
    blas: Blas,
    tlas: Tlas,
    tracer: Tracer,
}

impl AabbScene {
    fn new(ctx: &TestingContext, aabbs: &[Aabb]) -> Self {
        let aabbs_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("aabbs"),
            contents: bytemuck::cast_slice(aabbs),
            usage: BufferUsages::BLAS_INPUT | BufferUsages::COPY_DST,
        });
        let size = BlasAABBGeometrySizeDescriptor {
            primitive_count: aabbs.len() as u32,
            flags: AccelerationStructureGeometryFlags::empty(),
        };
        let blas = ctx.device.create_blas(
            &CreateBlasDescriptor {
                label: Some("AABB BLAS"),
                flags: UPDATABLE,
            },
            BlasGeometrySizeDescriptors::AABBs {
                descriptors: vec![size.clone()],
            },
        );
        let mut tlas = ctx.device.create_tlas(&CreateTlasDescriptor {
            label: Some("TLAS"),
            max_instances: 1,
            flags: NOT_UPDATABLE,
        });
        tlas[0] = Some(TlasInstance::from_blas_untransformed(&blas));

        Self {
            aabbs: aabbs_buffer,
            size,
            blas,
            tlas,
            tracer: Tracer::new(ctx, "trace_aabbs"),
        }
    }

    /// Encodes a build of the BLAS with `primitive_count` boxes, and a full build of the TLAS if
    /// `tlas` is set.
    fn encode(
        &self,
        ctx: &TestingContext,
        primitive_count: u32,
        stride: BufferAddress,
        mode: AccelerationStructureBuildMode,
        tlas: bool,
    ) -> CommandBuffer {
        let size = BlasAABBGeometrySizeDescriptor {
            primitive_count,
            ..self.size.clone()
        };
        let tlas_entry = tlas.then_some(TlasBuildEntry {
            tlas: &self.tlas,
            mode: Build,
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.build_acceleration_structures(
            [&BlasBuildEntry {
                blas: &self.blas,
                geometry: BlasGeometries::AabbGeometries(vec![BlasAabbGeometry {
                    size: &size,
                    stride,
                    aabb_buffer: &self.aabbs,
                    primitive_offset: 0,
                }]),
                mode,
            }],
            tlas_entry.as_ref(),
        );
        encoder.finish()
    }
}

/// Checks that an update with inputs `second` after a submitted build with inputs `first` is
/// valid exactly when `valid_update` is set.
fn check_blas_update(
    ctx: TestingContext,
    layout: Layout,
    first: Option<BlasInputs>,
    second: BlasInputs,
    valid_update: bool,
) {
    let vertices = [triangle(0.0, 1.0), triangle(10.0, 1.0)].concat();
    let scene = Scene::with_layout(&ctx, &vertices, 2, layout, UPDATABLE, NOT_UPDATABLE);

    if let Some(first) = first {
        ctx.queue.submit([scene.encode_blas(&ctx, first, Build)]);
    }
    submit_update(&ctx, scene.encode_blas(&ctx, second, Update), valid_update);
}

#[apply(gpu_test!)]
static BLAS_UPDATE_WITH_SAME_INPUTS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(|ctx| {
        let inputs = BlasInputs::vertices(&[3, 6]);
        check_blas_update(ctx, Layout::default(), Some(inputs), inputs, true)
    });

#[apply(gpu_test!)]
static BLAS_UPDATE_WHEN_NEVER_BUILT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(|ctx| {
        let inputs = BlasInputs::vertices(&[3, 6]);
        check_blas_update(ctx, Layout::default(), None, inputs, false)
    });

#[apply(gpu_test!)]
static BLAS_UPDATE_AFTER_VERTEX_COUNT_CHANGE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(|ctx| {
        let first = BlasInputs::vertices(&[3, 6]);
        let second = BlasInputs::vertices(&[6, 6]);
        check_blas_update(ctx, Layout::default(), Some(first), second, false)
    });

#[apply(gpu_test!)]
static BLAS_UPDATE_AFTER_GEOMETRY_COUNT_CHANGE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(|ctx| {
        let first = BlasInputs::vertices(&[3, 3]);
        let second = BlasInputs::vertices(&[3]);
        check_blas_update(ctx, Layout::default(), Some(first), second, false)
    });

#[apply(gpu_test!)]
static BLAS_UPDATE_AFTER_STRIDE_CHANGE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(|ctx| {
        let first = BlasInputs::vertices(&[3, 3]);
        let second = BlasInputs {
            vertex_stride: 2 * VERTEX_STRIDE,
            ..first
        };
        check_blas_update(ctx, Layout::default(), Some(first), second, false)
    });

#[apply(gpu_test!)]
static INDEXED_BLAS_UPDATE_WITH_SAME_INPUTS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(|ctx| {
        let inputs = BlasInputs::vertices(&[6, 6]);
        check_blas_update(ctx, INDEXED, Some(inputs), inputs, true)
    });

#[apply(gpu_test!)]
static INDEXED_BLAS_UPDATE_AFTER_INDEX_COUNT_CHANGE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(update_test_parameters())
        .run_sync(|ctx| {
            let first = BlasInputs::vertices(&[6, 6]);
            let second = BlasInputs {
                index_counts: Some(&[3, 6]),
                ..first
            };
            check_blas_update(ctx, INDEXED, Some(first), second, false)
        });

#[apply(gpu_test!)]
static BLAS_UPDATE_WITHOUT_ALLOW_UPDATE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(|ctx| {
        let scene = Scene::new(&ctx, &triangle(0.0, 1.0), 1, NOT_UPDATABLE, NOT_UPDATABLE);
        let inputs = BlasInputs::vertices(&[3]);
        ctx.queue.submit([scene.encode_blas(&ctx, inputs, Build)]);
        fail(
            &ctx.device,
            || scene.encode_blas(&ctx, inputs, Update),
            Some(MISSING_ALLOW_UPDATE),
        );
    });

/// Checks that an AABB BLAS update with `second` boxes and `stride` after a submitted build with
/// `first` boxes is valid exactly when `valid_update` is set.
fn check_aabb_blas_update(
    ctx: TestingContext,
    first: u32,
    second: u32,
    stride: BufferAddress,
    valid_update: bool,
) {
    let scene = AabbScene::new(&ctx, &[aabb(0.0, 1.0), aabb(10.0, 1.0)]);
    ctx.queue
        .submit([scene.encode(&ctx, first, AABB_STRIDE, Build, false)]);
    submit_update(
        &ctx,
        scene.encode(&ctx, second, stride, Update, false),
        valid_update,
    );
}

#[apply(gpu_test!)]
static AABB_BLAS_UPDATE_WITH_SAME_INPUTS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(|ctx| check_aabb_blas_update(ctx, 2, 2, AABB_STRIDE, true));

#[apply(gpu_test!)]
static AABB_BLAS_UPDATE_AFTER_PRIMITIVE_COUNT_CHANGE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(update_test_parameters())
        .run_sync(|ctx| check_aabb_blas_update(ctx, 2, 1, AABB_STRIDE, false));

#[apply(gpu_test!)]
static AABB_BLAS_UPDATE_AFTER_STRIDE_CHANGE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(|ctx| check_aabb_blas_update(ctx, 1, 1, 2 * AABB_STRIDE, false));

/// Checks that a TLAS update with `second` instances after a submitted build with `first`
/// instances is valid exactly when `valid_update` is set.
fn check_tlas_update(ctx: TestingContext, first: Option<usize>, second: usize, valid_update: bool) {
    let mut scene = Scene::new(&ctx, &triangle(0.0, 1.0), 1, NOT_UPDATABLE, UPDATABLE);
    ctx.queue
        .submit([scene.encode_blas(&ctx, BlasInputs::vertices(&[3]), Build)]);

    if let Some(first) = first {
        scene.set_instance_count(first);
        ctx.queue.submit([scene.encode_tlas(&ctx, Build)]);
    }
    scene.set_instance_count(second);
    submit_update(&ctx, scene.encode_tlas(&ctx, Update), valid_update);
}

#[apply(gpu_test!)]
static TLAS_UPDATE_WITH_SAME_INSTANCE_COUNT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(|ctx| check_tlas_update(ctx, Some(1), 1, true));

#[apply(gpu_test!)]
static TLAS_UPDATE_WHEN_NEVER_BUILT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(|ctx| check_tlas_update(ctx, None, 1, false));

#[apply(gpu_test!)]
static TLAS_UPDATE_AFTER_INSTANCE_COUNT_CHANGE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(|ctx| check_tlas_update(ctx, Some(1), 2, false));

#[apply(gpu_test!)]
static TLAS_UPDATE_WITHOUT_ALLOW_UPDATE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(|ctx| {
        let scene = Scene::new(&ctx, &triangle(0.0, 1.0), 1, NOT_UPDATABLE, NOT_UPDATABLE);
        ctx.queue.submit([scene.encode_full(&ctx, Build, Build)]);
        fail(
            &ctx.device,
            || scene.encode_tlas(&ctx, Update),
            Some(MISSING_ALLOW_UPDATE),
        );
    });

#[apply(gpu_test!)]
static UPDATE_BEFORE_BUILD_IN_SAME_COMMAND_BUFFER: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(update_test_parameters())
        .run_sync(|ctx| {
            let scene = Scene::new(&ctx, &triangle(0.0, 1.0), 1, UPDATABLE, NOT_UPDATABLE);
            let entry = |mode| BlasBuildEntry {
                blas: &scene.blas,
                geometry: BlasGeometries::TriangleGeometries(vec![
                    scene.geometry(&scene.sizes[0], VERTEX_STRIDE)
                ]),
                mode,
            };
            let mut encoder = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor::default());
            encoder.build_acceleration_structures([&entry(Update)], []);
            encoder.build_acceleration_structures([&entry(Build)], []);
            submit_update(&ctx, encoder.finish(), false);
        });

#[apply(gpu_test!)]
static UPDATE_SUBMISSION_ORDER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(update_submission_order);

/// Whether an update is valid depends on the order of submission, not of encoding.
fn update_submission_order(ctx: TestingContext) {
    let scene = Scene::new(&ctx, &triangle(0.0, 1.0), 1, UPDATABLE, NOT_UPDATABLE);
    let inputs = BlasInputs::vertices(&[3]);

    let build = scene.encode_blas(&ctx, inputs, Build);
    let update = scene.encode_blas(&ctx, inputs, Update);
    valid(&ctx.device, || ctx.queue.submit([build, update]));

    let update = scene.encode_blas(&ctx, inputs, Update);
    let other_build = scene.encode_blas(&ctx, BlasInputs::vertices(&[0]), Build);
    ctx.queue.submit([other_build]);
    submit_update(&ctx, update, false);
}

#[apply(gpu_test!)]
static UPDATE_AFTER_FAILED_SUBMISSION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(update_after_failed_submission);

/// Builds in a submission that fails validation never execute, so they cannot be updated.
fn update_after_failed_submission(ctx: TestingContext) {
    let scene = Scene::new(&ctx, &triangle(0.0, 1.0), 1, UPDATABLE, UPDATABLE);
    let inputs = BlasInputs::vertices(&[3]);

    let invalid = || {
        let buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("destroyed"),
            size: 4,
            usage: BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.clear_buffer(&buffer, 0, None);
        let command_buffer = encoder.finish();
        buffer.destroy();
        command_buffer
    };

    let build = scene.encode_blas(&ctx, inputs, Build);
    fail(&ctx.device, || ctx.queue.submit([build, invalid()]), None);
    submit_update(&ctx, scene.encode_blas(&ctx, inputs, Update), false);

    ctx.queue.submit([scene.encode_blas(&ctx, inputs, Build)]);
    let build = scene.encode_tlas(&ctx, Build);
    fail(&ctx.device, || ctx.queue.submit([build, invalid()]), None);
    submit_update(&ctx, scene.encode_tlas(&ctx, Update), false);
}

#[apply(gpu_test!)]
static UPDATE_AFTER_MARK_BUILT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(update_after_mark_built);

/// wgpu-core does not know the inputs of a build it did not encode, so an acceleration structure
/// marked as built cannot be updated.
fn update_after_mark_built(ctx: TestingContext) {
    let scene = Scene::new(&ctx, &triangle(0.0, 1.0), 1, UPDATABLE, UPDATABLE);
    ctx.queue.submit([scene.encode_full(&ctx, Build, Build)]);

    // Stands in for a build through `as_hal` with the same inputs. The acceleration structures
    // already hold a valid build, so the safety requirements hold.
    let encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    unsafe { encoder.mark_acceleration_structures_built([&scene.blas], [&scene.tlas]) };
    ctx.queue.submit([encoder.finish()]);

    submit_update(
        &ctx,
        scene.encode_blas(&ctx, BlasInputs::vertices(&[3]), Update),
        false,
    );
    submit_update(&ctx, scene.encode_tlas(&ctx, Update), false);
}

/// Prepares `blas` for compaction, then compacts it and returns the compacted copy.
fn compact(ctx: &TestingContext, blas: &Blas) -> Blas {
    let (send, recv) = mpsc::channel();
    blas.prepare_compaction_async(move |res| send.send(res).unwrap());
    ctx.device.poll(PollType::wait_indefinitely()).unwrap();
    recv.recv().unwrap().unwrap();
    assert!(blas.ready_for_compaction());
    let compacted = ctx.queue.compact_blas(blas);
    ctx.queue.submit([]);
    compacted
}

const COMPACTABLE_UPDATABLE: AccelerationStructureFlags =
    AccelerationStructureFlags::ALLOW_UPDATE.union(AccelerationStructureFlags::ALLOW_COMPACTION);

#[apply(gpu_test!)]
static BLAS_UPDATE_AFTER_COMPACTION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(blas_update_after_compaction);

/// Compaction copies the BLAS, so the source can still be updated, and the update does not
/// change the copy. The copy can be neither built nor updated.
fn blas_update_after_compaction(ctx: TestingContext) {
    let mut scene = Scene::new(
        &ctx,
        &triangle(0.0, 1.0),
        1,
        COMPACTABLE_UPDATABLE,
        NOT_UPDATABLE,
    );
    ctx.queue.submit([scene.encode_full(&ctx, Build, Build)]);
    let compacted = compact(&ctx, &scene.blas);

    ctx.queue.write_buffer(
        &scene.vertices,
        0,
        bytemuck::cast_slice(&triangle(10.0, 2.0)),
    );
    submit_update(&ctx, scene.encode_full(&ctx, Update, Build), true);
    assert_eq!(scene.trace(&ctx), [MISS, Some(2.0)]);

    scene.tlas[0] = Some(TlasInstance::from_blas_untransformed(&compacted));
    ctx.queue.submit([scene.encode_tlas(&ctx, Build)]);
    assert_eq!(scene.trace(&ctx), [Some(1.0), MISS]);

    for mode in [Build, Update] {
        let entry = BlasBuildEntry {
            blas: &compacted,
            geometry: BlasGeometries::TriangleGeometries(vec![
                scene.geometry(&scene.sizes[0], VERTEX_STRIDE)
            ]),
            mode,
        };
        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.build_acceleration_structures([&entry], []);
        fail(&ctx.device, || encoder.finish(), Some("compacted"));
    }
}

#[apply(gpu_test!)]
static BLAS_COMPACTION_AFTER_UPDATE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(blas_compaction_after_update);

/// An update cancels a prepared compaction like a full build does, and compacting after the
/// update copies the updated BLAS.
fn blas_compaction_after_update(ctx: TestingContext) {
    let mut scene = Scene::new(
        &ctx,
        &triangle(0.0, 1.0),
        1,
        COMPACTABLE_UPDATABLE,
        NOT_UPDATABLE,
    );
    ctx.queue.submit([scene.encode_full(&ctx, Build, Build)]);

    let (send, recv) = mpsc::channel();
    scene
        .blas
        .prepare_compaction_async(move |res| send.send(res).unwrap());
    ctx.device.poll(PollType::wait_indefinitely()).unwrap();
    recv.recv().unwrap().unwrap();
    assert!(scene.blas.ready_for_compaction());

    ctx.queue.write_buffer(
        &scene.vertices,
        0,
        bytemuck::cast_slice(&triangle(10.0, 2.0)),
    );
    submit_update(&ctx, scene.encode_full(&ctx, Update, Build), true);
    assert!(!scene.blas.ready_for_compaction());

    let compacted = compact(&ctx, &scene.blas);
    scene.tlas[0] = Some(TlasInstance::from_blas_untransformed(&compacted));
    ctx.queue.submit([scene.encode_tlas(&ctx, Build)]);
    assert_eq!(scene.trace(&ctx), [MISS, Some(2.0)]);
}

#[apply(gpu_test!)]
static BLAS_UPDATE_MOVES_HITS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(|ctx| blas_update_moves_hits(ctx, Layout::default()));

#[apply(gpu_test!)]
static INDEXED_BLAS_UPDATE_MOVES_HITS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(|ctx| blas_update_moves_hits(ctx, INDEXED));

fn blas_update_moves_hits(ctx: TestingContext, layout: Layout) {
    let scene = Scene::with_layout(
        &ctx,
        &triangle(0.0, 1.0),
        1,
        layout,
        UPDATABLE,
        NOT_UPDATABLE,
    );

    ctx.queue.submit([scene.encode_full(&ctx, Build, Build)]);
    assert_eq!(scene.trace(&ctx), [Some(1.0), MISS]);

    // Update of a build in an earlier submission.
    ctx.queue.write_buffer(
        &scene.vertices,
        0,
        bytemuck::cast_slice(&triangle(10.0, 2.0)),
    );
    submit_update(&ctx, scene.encode_full(&ctx, Update, Build), true);
    assert_eq!(scene.trace(&ctx), [MISS, Some(2.0)]);

    // Update of a build earlier in the same command buffer.
    let staging = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("staging"),
        contents: bytemuck::cast_slice(&[triangle(0.0, 3.0), triangle(10.0, 4.0)]),
        usage: BufferUsages::COPY_SRC,
    });
    let triangle_size = size_of::<[[f32; 3]; 3]>() as BufferAddress;
    let entry = |mode| BlasBuildEntry {
        blas: &scene.blas,
        geometry: BlasGeometries::TriangleGeometries(vec![
            scene.geometry(&scene.sizes[0], VERTEX_STRIDE)
        ]),
        mode,
    };
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(&staging, 0, &scene.vertices, 0, triangle_size);
    encoder.build_acceleration_structures([&entry(Build)], []);
    encoder.copy_buffer_to_buffer(&staging, triangle_size, &scene.vertices, 0, triangle_size);
    encoder.build_acceleration_structures(
        [&entry(Update)],
        [&TlasBuildEntry {
            tlas: &scene.tlas,
            mode: Build,
        }],
    );
    submit_update(&ctx, encoder.finish(), true);
    assert_eq!(scene.trace(&ctx), [MISS, Some(4.0)]);

    // A full build of an updatable BLAS, e.g. to restore trace performance after many updates.
    ctx.queue.write_buffer(
        &scene.vertices,
        0,
        bytemuck::cast_slice(&triangle(0.0, 5.0)),
    );
    ctx.queue.submit([scene.encode_full(&ctx, Build, Build)]);
    assert_eq!(scene.trace(&ctx), [Some(5.0), MISS]);
    submit_update(&ctx, scene.encode_full(&ctx, Update, Build), true);
    assert_eq!(scene.trace(&ctx), [Some(5.0), MISS]);
}

#[apply(gpu_test!)]
static TRANSFORMED_BLAS_UPDATE_MOVES_HITS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(transformed_blas_update_moves_hits);

fn transformed_blas_update_moves_hits(ctx: TestingContext) {
    let layout = Layout {
        indexed: false,
        transformed: true,
    };
    let scene = Scene::with_layout(
        &ctx,
        &triangle(0.0, 1.0),
        1,
        layout,
        UPDATABLE,
        NOT_UPDATABLE,
    );

    ctx.queue.submit([scene.encode_full(&ctx, Build, Build)]);
    assert_eq!(scene.trace(&ctx), [Some(1.0), MISS]);

    ctx.queue.write_buffer(
        &scene.transform,
        0,
        bytemuck::cast_slice(&translation(10.0, 1.0)),
    );
    submit_update(&ctx, scene.encode_full(&ctx, Update, Build), true);
    assert_eq!(scene.trace(&ctx), [MISS, Some(2.0)]);
}

#[apply(gpu_test!)]
static AABB_BLAS_UPDATE_MOVES_HITS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(aabb_blas_update_moves_hits);

fn aabb_blas_update_moves_hits(ctx: TestingContext) {
    let scene = AabbScene::new(&ctx, &[aabb(0.0, 1.0)]);

    ctx.queue
        .submit([scene.encode(&ctx, 1, AABB_STRIDE, Build, true)]);
    assert_eq!(scene.tracer.trace(&ctx, &scene.tlas), [Some(1.0), MISS]);

    ctx.queue
        .write_buffer(&scene.aabbs, 0, bytemuck::bytes_of(&aabb(10.0, 1.0)));
    submit_update(&ctx, scene.encode(&ctx, 1, AABB_STRIDE, Update, true), true);
    assert_eq!(scene.tracer.trace(&ctx, &scene.tlas), [MISS, Some(1.0)]);
}

#[apply(gpu_test!)]
static TLAS_UPDATE_MOVES_HITS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(tlas_update_moves_hits);

fn tlas_update_moves_hits(ctx: TestingContext) {
    let mut scene = Scene::new(&ctx, &triangle(0.0, 1.0), 1, NOT_UPDATABLE, UPDATABLE);

    ctx.queue.submit([scene.encode_full(&ctx, Build, Build)]);
    assert_eq!(scene.trace(&ctx), [Some(1.0), MISS]);

    for (x, z, expected) in [
        (10.0, 1.0, [MISS, Some(2.0)]),
        (0.0, 2.0, [Some(3.0), MISS]),
    ] {
        scene.tlas[0].as_mut().unwrap().transform = translation(x, z);
        submit_update(&ctx, scene.encode_tlas(&ctx, Update), true);
        assert_eq!(scene.trace(&ctx), expected);
    }

    // Update of a build earlier in the same command buffer.
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    scene.tlas[0].as_mut().unwrap().transform = translation(10.0, 3.0);
    encoder.build_acceleration_structures(
        [],
        [&TlasBuildEntry {
            tlas: &scene.tlas,
            mode: Build,
        }],
    );
    scene.tlas[0].as_mut().unwrap().transform = translation(0.0, 4.0);
    encoder.build_acceleration_structures(
        [],
        [&TlasBuildEntry {
            tlas: &scene.tlas,
            mode: Update,
        }],
    );
    submit_update(&ctx, encoder.finish(), true);
    assert_eq!(scene.trace(&ctx), [Some(5.0), MISS]);
}

#[apply(gpu_test!)]
static EMPTY_TLAS_UPDATE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(empty_tlas_update);

fn empty_tlas_update(ctx: TestingContext) {
    let mut scene = Scene::new(&ctx, &triangle(0.0, 1.0), 1, NOT_UPDATABLE, UPDATABLE);
    scene.set_instance_count(0);

    ctx.queue.submit([scene.encode_full(&ctx, Build, Build)]);
    submit_update(&ctx, scene.encode_tlas(&ctx, Update), true);
    assert_eq!(scene.trace(&ctx), [MISS, MISS]);
}

#[apply(gpu_test!)]
static ZERO_GEOMETRY_BLAS_UPDATE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(zero_geometry_blas_update);

fn zero_geometry_blas_update(ctx: TestingContext) {
    let scene = Scene::new(&ctx, &triangle(0.0, 1.0), 1, UPDATABLE, NOT_UPDATABLE);
    let no_geometry = BlasInputs::vertices(&[]);

    ctx.queue
        .submit([scene.encode(&ctx, Some((no_geometry, Build)), Some(Build))]);
    submit_update(
        &ctx,
        scene.encode(&ctx, Some((no_geometry, Update)), Some(Build)),
        true,
    );
    assert_eq!(scene.trace(&ctx), [MISS, MISS]);
}

#[apply(gpu_test!)]
static DUPLICATE_BLAS_BUILD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(duplicate_blas_build);

fn duplicate_blas_build(ctx: TestingContext) {
    let scene = Scene::new(&ctx, &triangle(0.0, 1.0), 1, UPDATABLE, NOT_UPDATABLE);
    let entry = BlasBuildEntry {
        blas: &scene.blas,
        geometry: BlasGeometries::TriangleGeometries(vec![
            scene.geometry(&scene.sizes[0], VERTEX_STRIDE)
        ]),
        mode: Build,
    };

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.build_acceleration_structures([&entry, &entry], []);
    fail(
        &ctx.device,
        || encoder.finish(),
        Some("more than once in a single build"),
    );
}

#[apply(gpu_test!)]
static DUPLICATE_TLAS_BUILD: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(update_test_parameters())
    .run_sync(duplicate_tlas_build);

fn duplicate_tlas_build(ctx: TestingContext) {
    let scene = Scene::new(&ctx, &triangle(0.0, 1.0), 1, NOT_UPDATABLE, UPDATABLE);
    ctx.queue
        .submit([scene.encode_blas(&ctx, BlasInputs::vertices(&[3]), Build)]);

    let entry = TlasBuildEntry {
        tlas: &scene.tlas,
        mode: Build,
    };
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.build_acceleration_structures([], [&entry, &entry]);
    fail(
        &ctx.device,
        || encoder.finish(),
        Some("more than once in a single build"),
    );
}
