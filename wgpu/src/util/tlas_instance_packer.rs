#![cfg(feature = "wgsl")]

use crate::util::{BufferInitDescriptor, DeviceExt};
use crate::{
    include_wgsl, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferUsages, CommandEncoder,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderStages,
};

/// The buffers for one [`TlasInstancePacker::pack`] dispatch.
///
/// Every buffer is indexed per instance (`0..count`) except `blas_addresses`, which is indexed by
/// the values in `blas_indices`.
#[derive(Debug)]
pub struct TlasInstancePackParams<'a> {
    /// `count * 12` `f32`s: one row-major `[f32; 12]` (object-to-world) transform per instance,
    /// laid out contiguously. Needs [`BufferUsages::STORAGE`].
    pub transforms: &'a Buffer,
    /// `count` `u32`s: each instance's index into `blas_addresses`. Needs
    /// [`BufferUsages::STORAGE`].
    pub blas_indices: &'a Buffer,
    /// One `u64` BLAS device address per unique BLAS, each from
    /// [`Blas::handle`](crate::Blas::handle). Needs [`BufferUsages::STORAGE`].
    pub blas_addresses: &'a Buffer,
    /// `count` `u32`s: per-instance custom data (low 24 bits used), surfaced to shaders as
    /// `RayIntersection::instance_custom_data`. Needs [`BufferUsages::STORAGE`].
    pub custom_data: &'a Buffer,
    /// Output: `count` [`RawTlasInstance`](wgt::RawTlasInstance) (64 bytes each). Needs
    /// [`BufferUsages::STORAGE`] (written here) `|` [`BufferUsages::TLAS_INPUT`] (read by the
    /// subsequent [`build_tlas_from_instances_buffer`](crate::CommandEncoder::build_tlas_from_instances_buffer)).
    pub instances: &'a Buffer,
    /// Number of instances to pack.
    pub count: u32,
}

/// A reusable compute pipeline that packs per-instance transforms and BLAS references into the
/// native [`RawTlasInstance`](wgt::RawTlasInstance) records consumed by
/// [`CommandEncoder::build_tlas_from_instances_buffer`](crate::CommandEncoder::build_tlas_from_instances_buffer),
/// so callers don't have to write the packing shader themselves.
///
/// It writes each instance's supplied `custom_data` (low 24 bits) together with a fully-visible
/// (`0xFF`) mask, and leaves `sbt_offset_and_flags` zero.
///
/// Create it once and reuse it across frames.
#[derive(Debug)]
pub struct TlasInstancePacker {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

impl TlasInstancePacker {
    /// Create the packer and its compute pipeline.
    pub fn new(device: &Device) -> Self {
        let read_storage = |binding| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("wgpu::util::TlasInstancePacker::bind_group_layout"),
            entries: &[
                read_storage(0),
                read_storage(1),
                read_storage(2),
                read_storage(3),
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("wgpu::util::TlasInstancePacker::pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(include_wgsl!("tlas_instance_packer.wgsl"));
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("wgpu::util::TlasInstancePacker::pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("pack"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Record a dispatch onto `encoder` that fills `params.instances` with packed instance
    /// records. Run this before
    /// [`build_tlas_from_instances_buffer`](crate::CommandEncoder::build_tlas_from_instances_buffer)
    /// on the same instance buffer.
    pub fn pack(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        params: &TlasInstancePackParams<'_>,
    ) {
        let count_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("wgpu::util::TlasInstancePacker::count"),
            contents: bytemuck::cast_slice(&[params.count, 0, 0, 0]),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("wgpu::util::TlasInstancePacker::bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params.transforms.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: params.blas_indices.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: params.blas_addresses.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: params.custom_data.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: params.instances.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: count_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("wgpu::util::TlasInstancePacker::pack"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);

        let workgroups = params.count.div_ceil(64);
        let max_dim = device.limits().max_compute_workgroups_per_dimension.max(1);
        let (grid_x, grid_y) = if workgroups <= max_dim {
            (workgroups, 1)
        } else {
            (max_dim, workgroups.div_ceil(max_dim))
        };
        pass.dispatch_workgroups(grid_x, grid_y, 1);
    }
}
