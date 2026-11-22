use wgpu::util::DeviceExt;
use wgpu::{BufferDescriptor, BufferUsages, DownlevelFlags, Limits, MapMode, PollType};
use wgpu_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        UNRESTRICTED_POINTER_PARAMETERS,
        UNRESTRICTED_POINTER_PARAMETERS_IMMEDIATE,
    ]);
}

const SHADER: &str = r#"
    requires unrestricted_pointer_parameters;

    struct Params {
        values: array<vec4<u32>, 2>,
    }

    @group(0) @binding(0) var<storage, read_write> out: array<u32>;
    @group(0) @binding(1) var<uniform> params: Params;
    var<workgroup> wg: array<u32, 4>;
    var<private> pv: array<u32, 4>;

    fn store(p: ptr<storage, u32, read_write>, v: u32) {
        *p = v;
    }

    fn load_add(p: ptr<storage, u32, read_write>, v: u32) -> u32 {
        let old = *p;
        *p = old + v;
        return old;
    }

    fn find(p: ptr<workgroup, array<u32, 4>>, needle: u32) -> i32 {
        for (var i = 0; i < 4; i++) {
            if p[i] == needle {
                return i;
            }
        }
        return -1;
    }

    fn classify(p: ptr<storage, u32, read_write>) -> u32 {
        switch *p {
            case 0u: {
                return 10u;
            }
            case 1u: {
                *p = 2u;
            }
            default: {
                return 30u;
            }
        }
        return 20u;
    }

    fn bump(p: ptr<function, u32>) {
        var count = 1u;
        *p += count;
        count += 1u;
    }

    fn bump_twice(p: ptr<function, u32>) {
        bump(p);
        bump(p);
    }

    fn set_private(p: ptr<private, u32>) {
        *p = 7u;
    }

    fn early(p: ptr<storage, u32, read_write>, c: bool) {
        if c {
            return;
        }
        *p = 5u;
    }

    fn read_uniform(p: ptr<uniform, vec4<u32>>) -> u32 {
        return (*p).y;
    }

    fn len(p: ptr<storage, array<u32>, read_write>) -> u32 {
        return arrayLength(p);
    }

    @compute @workgroup_size(1)
    fn main() {
        for (var i = 0u; i < 4u; i++) {
            wg[i] = i * 10u;
        }
        store(&out[0], 11u);
        out[1] = 100u;
        out[2] = load_add(&out[1], 5u) + load_add(&out[1], 7u);
        out[3] = u32(find(&wg, 20u));
        out[4] = u32(find(&wg, 25u) + 10);
        out[5] = 1u;
        out[6] = classify(&out[5]);
        out[7] = 0u;
        out[8] = classify(&out[7]);
        var local = array<u32, 2>(0u, 100u);
        for (var i = 0u; i < 2u; i++) {
            bump_twice(&local[i]);
        }
        out[9] = local[0];
        out[10] = local[1];
        set_private(&pv[2]);
        out[11] = pv[2];
        out[12] = 0u;
        early(&out[12], true);
        out[13] = 0u;
        early(&out[13], false);
        out[14] = read_uniform(&params.values[1]);
        out[15] = len(&out);
    }
"#;

const EXPECTED: [u32; 16] = [11, 112, 205, 2, 9, 2, 20, 0, 10, 2, 102, 7, 0, 5, 42, 16];

#[apply(gpu_test!)]
static UNRESTRICTED_POINTER_PARAMETERS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let out = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("out"),
            size: size_of_val(&EXPECTED) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params: [u32; 8] = [0, 0, 0, 0, 0, 42, 0, 0];
        let params = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::cast_slice(&params),
                usage: BufferUsages::UNIFORM,
            });
        let mapping = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("mapping"),
            size: out.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
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
        encoder.copy_buffer_to_buffer(&out, 0, &mapping, 0, out.size());
        ctx.queue.submit(Some(encoder.finish()));

        mapping.slice(..).map_async(MapMode::Read, |_| ());
        ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();
        let mapped = mapping.slice(..).get_mapped_range().unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&mapped), EXPECTED);
    });

const IMMEDIATE_SHADER: &str = r#"
    requires unrestricted_pointer_parameters;

    struct Pc {
        a: u32,
        b: vec2<u32>,
    }

    var<immediate> pc: Pc;
    @group(0) @binding(0) var<storage, read_write> out: array<u32>;

    fn read(p: ptr<immediate, u32>) -> u32 {
        return *p;
    }

    fn read_vec(p: ptr<immediate, vec2<u32>>) -> u32 {
        return (*p).y;
    }

    fn read_whole(p: ptr<immediate, Pc>) -> u32 {
        return (*p).a + (*p).b.x;
    }

    @compute @workgroup_size(1)
    fn main() {
        out[0] = read(&pc.a);
        out[1] = read_vec(&pc.b);
        out[2] = read_whole(&pc);
    }
"#;

#[apply(gpu_test!)]
static UNRESTRICTED_POINTER_PARAMETERS_IMMEDIATE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
                .features(wgpu::Features::IMMEDIATES)
                .limits(Limits {
                    max_immediate_size: 16,
                    ..Limits::downlevel_defaults()
                }),
        )
        .run_async(|ctx| async move {
            let module = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(IMMEDIATE_SHADER.into()),
                });
            let bgl = ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
            let layout = ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&bgl)],
                    immediate_size: 16,
                });
            let pipeline = ctx
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            let out = ctx.device.create_buffer(&BufferDescriptor {
                label: Some("out"),
                size: 16,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mapping = ctx.device.create_buffer(&BufferDescriptor {
                label: Some("mapping"),
                size: out.size(),
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out.as_entire_binding(),
                }],
            });

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                let immediates: [u32; 4] = [3, 0, 40, 500];
                pass.set_immediates(0, bytemuck::cast_slice(&immediates));
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&out, 0, &mapping, 0, out.size());
            ctx.queue.submit(Some(encoder.finish()));

            mapping.slice(..).map_async(MapMode::Read, |_| ());
            ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();
            let mapped = mapping.slice(..).get_mapped_range().unwrap();
            assert_eq!(bytemuck::cast_slice::<u8, u32>(&mapped), [3, 500, 43, 0]);
        });
