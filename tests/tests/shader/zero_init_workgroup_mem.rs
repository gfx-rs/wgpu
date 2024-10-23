use std::num::NonZeroU64;

use wgpu::{
    include_wgsl, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBinding, BufferBindingType,
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, DownlevelFlags, Limits, Maintain, MapMode, PipelineLayoutDescriptor,
    ShaderStages,
};

use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters};

#[gpu_test]
static ZERO_INIT_WORKGROUP_MEMORY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| async move {
        let bgl = ctx
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let output_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("output buffer"),
            size: BUFFER_SIZE,
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let mapping_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("mapping buffer"),
            size: BUFFER_SIZE,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bg = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &output_buffer,
                    offset: 0,
                    size: Some(NonZeroU64::new(BUFFER_BINDING_SIZE as u64).unwrap()),
                }),
            }],
        });

        let pll = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let sm = ctx
            .device
            .create_shader_module(include_wgsl!("zero_init_workgroup_mem.wgsl"));

        let pipeline_read = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("pipeline read"),
                layout: Some(&pll),
                module: &sm,
                entry_point: Some("read"),
                compilation_options: Default::default(),
                cache: None,
            });

        let pipeline_write = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("pipeline write"),
                layout: None,
                module: &sm,
                entry_point: Some("write"),
                compilation_options: Default::default(),
                cache: None,
            });

        // -- Initializing data --

        let output_pre_init_data = vec![1; OUTPUT_ARRAY_SIZE as usize];
        ctx.queue.write_buffer(
            &output_buffer,
            0,
            bytemuck::cast_slice(&output_pre_init_data),
        );

        // -- Run test --

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor::default());

        cpass.set_pipeline(&pipeline_write);
        for _ in 0..NR_OF_DISPATCHES {
            cpass.dispatch_workgroups(DISPATCH_SIZE.0, DISPATCH_SIZE.1, DISPATCH_SIZE.2);
        }

        cpass.set_pipeline(&pipeline_read);
        for i in 0..NR_OF_DISPATCHES {
            cpass.set_bind_group(0, &bg, &[i * BUFFER_BINDING_SIZE]);
            cpass.dispatch_workgroups(DISPATCH_SIZE.0, DISPATCH_SIZE.1, DISPATCH_SIZE.2);
        }
        drop(cpass);

        // -- Pulldown data --

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &mapping_buffer, 0, BUFFER_SIZE);

        ctx.queue.submit(Some(encoder.finish()));

        mapping_buffer.slice(..).map_async(MapMode::Read, |_| ());
        ctx.async_poll(Maintain::wait()).await.panic_on_timeout();

        let mapped = mapping_buffer.slice(..).get_mapped_range();

        let typed: &[u32] = bytemuck::cast_slice(&mapped);

        // -- Check results --

        let num_disptaches_failed = typed.iter().filter(|&&res| res != 0).count();
        let ratio = (num_disptaches_failed as f32 / OUTPUT_ARRAY_SIZE as f32) * 100.;

        assert!(
            num_disptaches_failed == 0,
            "Zero-initialization of workgroup memory failed ({ratio:.0}% of disptaches failed)."
        );

        drop(mapped);
        mapping_buffer.unmap();
    });

const DISPATCH_SIZE: (u32, u32, u32) = (64, 64, 64);
const TOTAL_WORK_GROUPS: u32 = DISPATCH_SIZE.0 * DISPATCH_SIZE.1 * DISPATCH_SIZE.2;

/// nr of bytes we use in the shader
const SHADER_WORKGROUP_MEMORY: u32 = 512 * 4 + 4;
// assume we have this much workgroup memory (2GB)
const MAX_DEVICE_WORKGROUP_MEMORY: u32 = i32::MAX as u32;
const NR_OF_DISPATCHES: u32 =
    MAX_DEVICE_WORKGROUP_MEMORY / (SHADER_WORKGROUP_MEMORY * TOTAL_WORK_GROUPS) + 1; // TODO: use div_ceil once stabilized

const OUTPUT_ARRAY_SIZE: u32 = TOTAL_WORK_GROUPS * NR_OF_DISPATCHES;
const BUFFER_SIZE: u64 = OUTPUT_ARRAY_SIZE as u64 * 4;
const BUFFER_BINDING_SIZE: u32 = TOTAL_WORK_GROUPS * 4;
