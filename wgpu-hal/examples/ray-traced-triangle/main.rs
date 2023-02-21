extern crate wgpu_hal as hal;

use hal::{
    Adapter as _, CommandEncoder as _, Device as _, Instance as _, Queue as _, Surface as _,
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

use glam::{Mat4, Vec3};
use std::{
    borrow::{Borrow, Cow},
    iter, mem,
    mem::{align_of, size_of},
    ptr::{self, copy_nonoverlapping},
    time::Instant,
};

const COMMAND_BUFFER_PER_CONTEXT: usize = 100;
const DESIRED_FRAMES: u32 = 3;

fn pack_24_8(low_24: u32, high_8: u8) -> u32 {
    (low_24 & 0x00ff_ffff) | (u32::from(high_8) << 24)
}

#[derive(Debug)]
#[repr(C)]
struct Instance {
    transform: [f32; 12],
    instance_custom_index_and_mask: u32,
    instance_shader_binding_table_record_offset_and_flags: u32,
    acceleration_structure_reference: u64,
}

fn transpose_matrix_for_acceleration_structure_instance(matrix: Mat4) -> [f32; 12] {
    let row_0 = matrix.row(0);
    let row_1 = matrix.row(1);
    let row_2 = matrix.row(2);
    [
        row_0.x, row_0.y, row_0.z, row_0.w, row_1.x, row_1.y, row_1.z, row_1.w, row_2.x, row_2.y,
        row_2.z, row_2.w,
    ]
}

struct ExecutionContext<A: hal::Api> {
    encoder: A::CommandEncoder,
    fence: A::Fence,
    fence_value: hal::FenceValue,
    used_views: Vec<A::TextureView>,
    used_cmd_bufs: Vec<A::CommandBuffer>,
    frames_recorded: usize,
}

impl<A: hal::Api> ExecutionContext<A> {
    unsafe fn wait_and_clear(&mut self, device: &A::Device) {
        device.wait(&self.fence, self.fence_value, !0).unwrap();
        self.encoder.reset_all(self.used_cmd_bufs.drain(..));
        for view in self.used_views.drain(..) {
            device.destroy_texture_view(view);
        }
        self.frames_recorded = 0;
    }
}

#[allow(dead_code)]
struct Example<A: hal::Api> {
    instance: A::Instance,
    adapter: A::Adapter,
    surface: A::Surface,
    surface_format: wgt::TextureFormat,
    device: A::Device,
    queue: A::Queue,

    contexts: Vec<ExecutionContext<A>>,
    context_index: usize,
    extent: [u32; 2],
    start: Instant,
    pipeline: A::ComputePipeline,
    bind_group: A::BindGroup,
    bgl: A::BindGroupLayout,
    shader_module: A::ShaderModule,
    texture_view: A::TextureView,
    uniform_buffer: A::Buffer,
    pipeline_layout: A::PipelineLayout,
    vertices_buffer: A::Buffer,
    indices_buffer: A::Buffer,
    texture: A::Texture,
    instances: [Instance; 1],
    instances_buffer: A::Buffer,
    blas: A::AccelerationStructure,
    tlas: A::AccelerationStructure,
    scratch_buffer: A::Buffer,
    time: f32,
}

impl<A: hal::Api> Example<A> {
    fn init(window: &winit::window::Window) -> Result<Self, hal::InstanceError> {
        let instance_desc = hal::InstanceDescriptor {
            name: "example",
            flags: if cfg!(debug_assertions) {
                hal::InstanceFlags::all()
            } else {
                hal::InstanceFlags::empty()
            },
            dx12_shader_compiler: wgt::Dx12Compiler::Fxc,
        };
        let instance = unsafe { A::Instance::init(&instance_desc)? };
        let mut surface = unsafe {
            instance
                .create_surface(window.raw_display_handle(), window.raw_window_handle())
                .unwrap()
        };

        let (adapter, features) = unsafe {
            let mut adapters = instance.enumerate_adapters();
            if adapters.is_empty() {
                return Err(hal::InstanceError);
            }
            let exposed = adapters.swap_remove(0);
            dbg!(exposed.features);
            (exposed.adapter, exposed.features)
        };
        let surface_caps =
            unsafe { adapter.surface_capabilities(&surface) }.ok_or(hal::InstanceError)?;
        log::info!("Surface caps: {:#?}", surface_caps);

        let hal::OpenDevice { device, mut queue } =
            unsafe { adapter.open(features, &wgt::Limits::default()).unwrap() };

        let window_size: (u32, u32) = window.inner_size().into();
        dbg!(&surface_caps.formats);
        let surface_format = if surface_caps
            .formats
            .contains(&wgt::TextureFormat::Rgba8Snorm)
        {
            wgt::TextureFormat::Rgba8Unorm
        } else {
            *surface_caps.formats.first().unwrap()
        };
        let surface_config = hal::SurfaceConfiguration {
            swap_chain_size: DESIRED_FRAMES
                .max(*surface_caps.swap_chain_sizes.start())
                .min(*surface_caps.swap_chain_sizes.end()),
            present_mode: wgt::PresentMode::Fifo,
            composite_alpha_mode: wgt::CompositeAlphaMode::Opaque,
            format: surface_format,
            extent: wgt::Extent3d {
                width: window_size.0,
                height: window_size.1,
                depth_or_array_layers: 1,
            },
            usage: hal::TextureUses::COLOR_TARGET | hal::TextureUses::COPY_DST,
            view_formats: vec![surface_format],
        };
        unsafe {
            surface.configure(&device, &surface_config).unwrap();
        };

        #[allow(dead_code)]
        struct Uniforms {
            view_inverse: glam::Mat4,
            proj_inverse: glam::Mat4,
        }

        let bgl_desc = hal::BindGroupLayoutDescriptor {
            label: None,
            flags: hal::BindGroupLayoutFlags::empty(),
            entries: &[
                wgt::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgt::ShaderStages::COMPUTE,
                    ty: wgt::BindingType::Buffer {
                        ty: wgt::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgt::BufferSize::new(mem::size_of::<Uniforms>() as _),
                    },
                    count: None,
                },
                wgt::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgt::ShaderStages::COMPUTE,
                    ty: wgt::BindingType::StorageTexture {
                        access: wgt::StorageTextureAccess::WriteOnly,
                        format: wgt::TextureFormat::Rgba8Unorm,
                        view_dimension: wgt::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgt::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgt::ShaderStages::COMPUTE,
                    ty: wgt::BindingType::AccelerationStructure,
                    count: None,
                },
            ],
        };

        let bgl = unsafe { device.create_bind_group_layout(&bgl_desc).unwrap() };

        pub fn make_spirv_raw(data: &[u8]) -> Cow<[u32]> {
            const MAGIC_NUMBER: u32 = 0x0723_0203;
            assert_eq!(
                data.len() % size_of::<u32>(),
                0,
                "data size is not a multiple of 4"
            );

            //If the data happens to be aligned, directly use the byte array,
            // otherwise copy the byte array in an owned vector and use that instead.
            let words = if data.as_ptr().align_offset(align_of::<u32>()) == 0 {
                let (pre, words, post) = unsafe { data.align_to::<u32>() };
                debug_assert!(pre.is_empty());
                debug_assert!(post.is_empty());
                Cow::from(words)
            } else {
                let mut words = vec![0u32; data.len() / size_of::<u32>()];
                unsafe {
                    copy_nonoverlapping(data.as_ptr(), words.as_mut_ptr() as *mut u8, data.len());
                }
                Cow::from(words)
            };

            assert_eq!(
                words[0], MAGIC_NUMBER,
                "wrong magic word {:x}. Make sure you are using a binary SPIRV file.",
                words[0]
            );

            words
        }

        let shader_module = unsafe {
            device
                .create_shader_module(
                    &hal::ShaderModuleDescriptor {
                        label: None,
                        runtime_checks: false,
                    },
                    hal::ShaderInput::SpirV(&make_spirv_raw(include_bytes!("shader.comp.spv"))),
                )
                .unwrap()
        };

        let pipeline_layout_desc = hal::PipelineLayoutDescriptor {
            label: None,
            flags: hal::PipelineLayoutFlags::empty(),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        };
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_desc)
                .unwrap()
        };

        let pipeline = unsafe {
            device.create_compute_pipeline(&hal::ComputePipelineDescriptor {
                label: Some("pipeline"),
                layout: &pipeline_layout,
                stage: hal::ProgrammableStage {
                    module: &shader_module,
                    entry_point: "main",
                },
            })
        }
        .unwrap();

        let vertices: [f32; 9] = [1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0, 0.0];

        let vertices_size_in_bytes = vertices.len() * 4;

        let indices: [u32; 3] = [0, 1, 2];

        let indices_size_in_bytes = indices.len() * 4;

        let transform_matrix = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        let vertices_buffer = unsafe {
            let vertices_buffer = device
                .create_buffer(&hal::BufferDescriptor {
                    label: Some("vertices buffer"),
                    size: vertices_size_in_bytes as u64,
                    usage: hal::BufferUses::MAP_WRITE
                        | hal::BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                    memory_flags: hal::MemoryFlags::TRANSIENT | hal::MemoryFlags::PREFER_COHERENT,
                })
                .unwrap();

            let mapping = device
                .map_buffer(&vertices_buffer, 0..vertices_size_in_bytes as u64)
                .unwrap();
            ptr::copy_nonoverlapping(
                vertices.as_ptr() as *const u8,
                mapping.ptr.as_ptr(),
                vertices_size_in_bytes,
            );
            device.unmap_buffer(&vertices_buffer).unwrap();
            assert!(mapping.is_coherent);

            vertices_buffer
        };

        let indices_buffer = unsafe {
            let indices_buffer = device
                .create_buffer(&hal::BufferDescriptor {
                    label: Some("indices buffer"),
                    size: indices_size_in_bytes as u64,
                    usage: hal::BufferUses::MAP_WRITE
                        | hal::BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                    memory_flags: hal::MemoryFlags::TRANSIENT | hal::MemoryFlags::PREFER_COHERENT,
                })
                .unwrap();

            let mapping = device
                .map_buffer(&indices_buffer, 0..indices_size_in_bytes as u64)
                .unwrap();
            ptr::copy_nonoverlapping(
                indices.as_ptr() as *const u8,
                mapping.ptr.as_ptr(),
                indices_size_in_bytes,
            );
            device.unmap_buffer(&indices_buffer).unwrap();
            assert!(mapping.is_coherent);

            indices_buffer
        };

        let blas_sizes = unsafe {
            device.get_acceleration_structure_build_sizes(
                &hal::GetAccelerationStructureBuildSizesDescriptor {
                    geometry_info: hal::AccelerationStructureGeometryInfo::Triangles {
                        vertex_format: wgt::VertexFormat::Float32x3,
                        max_vertex: 3,
                        index_format: Some(wgt::IndexFormat::Uint32),
                    },
                    format: hal::AccelerationStructureFormat::BottomLevel,
                    mode: hal::AccelerationStructureBuildMode::Build,
                    flags: hal::AccelerationStructureBuildFlags::PREFER_FAST_TRACE,
                    primitive_count: 1,
                },
            )
        };

        let tlas_flags = hal::AccelerationStructureBuildFlags::PREFER_FAST_TRACE
            | hal::AccelerationStructureBuildFlags::ALLOW_UPDATE;

        let tlas_sizes = unsafe {
            device.get_acceleration_structure_build_sizes(
                &hal::GetAccelerationStructureBuildSizesDescriptor {
                    geometry_info: hal::AccelerationStructureGeometryInfo::Instances,
                    format: hal::AccelerationStructureFormat::TopLevel,
                    mode: hal::AccelerationStructureBuildMode::Build,
                    flags: tlas_flags,
                    primitive_count: 1,
                },
            )
        };

        let blas = unsafe {
            device.create_acceleration_structure(&hal::AccelerationStructureDescriptor {
                label: Some("blas"),
                size: blas_sizes.acceleration_structure_size,
                format: hal::AccelerationStructureFormat::BottomLevel,
            })
        }
        .unwrap();

        let tlas = unsafe {
            device.create_acceleration_structure(&hal::AccelerationStructureDescriptor {
                label: Some("tlas"),
                size: tlas_sizes.acceleration_structure_size,
                format: hal::AccelerationStructureFormat::TopLevel,
            })
        }
        .unwrap();

        let uniforms = {
            let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 2.5), Vec3::ZERO, Vec3::Y);
            let proj = Mat4::perspective_rh(59.0_f32.to_radians(), 1.0, 0.001, 1000.0);

            Uniforms {
                view_inverse: view.inverse(),
                proj_inverse: proj.inverse(),
            }
        };

        let uniforms_size = std::mem::size_of::<Uniforms>();

        let uniform_buffer = unsafe {
            let uniform_buffer = device
                .create_buffer(&hal::BufferDescriptor {
                    label: Some("uniform buffer"),
                    size: uniforms_size as u64,
                    usage: hal::BufferUses::MAP_WRITE | hal::BufferUses::UNIFORM,
                    memory_flags: hal::MemoryFlags::PREFER_COHERENT,
                })
                .unwrap();

            let mapping = device
                .map_buffer(&uniform_buffer, 0..uniforms_size as u64)
                .unwrap();
            ptr::copy_nonoverlapping(
                &uniforms as *const Uniforms as *const u8,
                mapping.ptr.as_ptr(),
                uniforms_size,
            );
            device.unmap_buffer(&uniform_buffer).unwrap();
            assert!(mapping.is_coherent);
            uniform_buffer
        };

        let texture_desc = hal::TextureDescriptor {
            label: None,
            size: wgt::Extent3d {
                width: 512,
                height: 512,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgt::TextureDimension::D2,
            format: wgt::TextureFormat::Rgba8Unorm,
            usage: hal::TextureUses::STORAGE_READ_WRITE | hal::TextureUses::COPY_SRC,
            memory_flags: hal::MemoryFlags::empty(),
            view_formats: vec![wgt::TextureFormat::Rgba8Unorm],
        };
        let texture = unsafe { device.create_texture(&texture_desc).unwrap() };

        let view_desc = hal::TextureViewDescriptor {
            label: None,
            format: texture_desc.format,
            dimension: wgt::TextureViewDimension::D2,
            usage: hal::TextureUses::STORAGE_READ_WRITE | hal::TextureUses::COPY_SRC,
            range: wgt::ImageSubresourceRange::default(),
        };
        let texture_view = unsafe { device.create_texture_view(&texture, &view_desc).unwrap() };

        let bind_group = {
            let buffer_binding = hal::BufferBinding {
                buffer: &uniform_buffer,
                offset: 0,
                size: None,
            };
            let texture_binding = hal::TextureBinding {
                view: &texture_view,
                usage: hal::TextureUses::STORAGE_READ_WRITE,
            };
            let group_desc = hal::BindGroupDescriptor {
                label: Some("bind group"),
                layout: &bgl,
                buffers: &[buffer_binding],
                samplers: &[],
                textures: &[texture_binding],
                acceleration_structures: &[&tlas],
                entries: &[
                    hal::BindGroupEntry {
                        binding: 0,
                        resource_index: 0,
                        count: 1,
                    },
                    hal::BindGroupEntry {
                        binding: 1,
                        resource_index: 0,
                        count: 1,
                    },
                    hal::BindGroupEntry {
                        binding: 2,
                        resource_index: 0,
                        count: 1,
                    },
                ],
            };
            unsafe { device.create_bind_group(&group_desc).unwrap() }
        };

        let scratch_buffer = unsafe {
            device
                .create_buffer(&hal::BufferDescriptor {
                    label: Some("scratch buffer"),
                    size: blas_sizes
                        .build_scratch_size
                        .max(tlas_sizes.build_scratch_size),
                    usage: hal::BufferUses::ACCELERATION_STRUCTURE_SCRATCH,
                    memory_flags: hal::MemoryFlags::empty(),
                })
                .unwrap()
        };

        let instances = [
            Instance {
                transform: transform_matrix,
                instance_custom_index_and_mask: pack_24_8(0, 0xff),
                instance_shader_binding_table_record_offset_and_flags: pack_24_8(0, 0),
                acceleration_structure_reference: unsafe {
                    device.get_acceleration_structure_device_address(&blas)
                },
            },
            /*Instance {
                transform: transpose_matrix_for_acceleration_structure_instance(
                    Mat4::from_rotation_y(1.0),
                ),
                instance_custom_index_and_mask: pack_24_8(0, 0xff),
                instance_shader_binding_table_record_offset_and_flags: pack_24_8(0, 0),
                acceleration_structure_reference: unsafe {
                    device.get_acceleration_structure_device_address(&blas)
                },
            },
            Instance {
                transform: transpose_matrix_for_acceleration_structure_instance(
                    Mat4::from_rotation_y(-1.0),
                ),
                instance_custom_index_and_mask: pack_24_8(0, 0xff),
                instance_shader_binding_table_record_offset_and_flags: pack_24_8(0, 0),
                acceleration_structure_reference: unsafe {
                    device.get_acceleration_structure_device_address(&blas)
                },
            },*/
        ];

        let instances_buffer_size = instances.len() * std::mem::size_of::<Instance>();

        let instances_buffer = unsafe {
            let instances_buffer = device
                .create_buffer(&hal::BufferDescriptor {
                    label: Some("instances_buffer"),
                    size: instances_buffer_size as u64,
                    usage: hal::BufferUses::MAP_WRITE
                        | hal::BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                    memory_flags: hal::MemoryFlags::TRANSIENT | hal::MemoryFlags::PREFER_COHERENT,
                })
                .unwrap();

            let mapping = device
                .map_buffer(&instances_buffer, 0..instances_buffer_size as u64)
                .unwrap();
            ptr::copy_nonoverlapping(
                instances.as_ptr() as *const u8,
                mapping.ptr.as_ptr(),
                instances_buffer_size,
            );
            device.unmap_buffer(&instances_buffer).unwrap();
            assert!(mapping.is_coherent);

            instances_buffer
        };

        let cmd_encoder_desc = hal::CommandEncoderDescriptor {
            label: None,
            queue: &queue,
        };
        let mut cmd_encoder = unsafe { device.create_command_encoder(&cmd_encoder_desc).unwrap() };

        unsafe { cmd_encoder.begin_encoding(Some("init")).unwrap() };

        unsafe {
            cmd_encoder.build_acceleration_structures(&hal::BuildAccelerationStructureDescriptor {
                geometry: &hal::AccelerationStructureGeometry::Triangles {
                    vertex_buffer: &vertices_buffer,
                    vertex_format: wgt::VertexFormat::Float32x3,
                    max_vertex: vertices.len() as u32,
                    vertex_stride: 3 * 4,
                    indices: Some(hal::AccelerationStructureGeometryIndices {
                        buffer: &indices_buffer,
                        format: wgt::IndexFormat::Uint32,
                    }),
                },
                format: hal::AccelerationStructureFormat::BottomLevel,
                mode: hal::AccelerationStructureBuildMode::Build,
                flags: hal::AccelerationStructureBuildFlags::PREFER_FAST_TRACE,
                primitive_count: indices.len() as u32 / 3,
                primitive_offset: 0,
                destination_acceleration_structure: &blas,
                scratch_buffer: &scratch_buffer,
            });

            let as_barrier = hal::BufferBarrier {
                buffer: &scratch_buffer,
                usage: hal::BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT
                    ..hal::BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT,
            };
            cmd_encoder.transition_buffers(iter::once(as_barrier));

            cmd_encoder.build_acceleration_structures(&hal::BuildAccelerationStructureDescriptor {
                geometry: &hal::AccelerationStructureGeometry::Instances {
                    buffer: &instances_buffer,
                },
                format: hal::AccelerationStructureFormat::TopLevel,
                mode: hal::AccelerationStructureBuildMode::Build,
                flags: tlas_flags,
                primitive_count: instances.len() as u32,
                primitive_offset: 0,
                destination_acceleration_structure: &tlas,
                scratch_buffer: &scratch_buffer,
            });

            let texture_barrier = hal::TextureBarrier {
                texture: &texture,
                range: wgt::ImageSubresourceRange::default(),
                usage: hal::TextureUses::UNINITIALIZED..hal::TextureUses::STORAGE_READ_WRITE,
            };

            cmd_encoder.transition_textures(iter::once(texture_barrier));
        }

        let init_fence_value = 1;
        let fence = unsafe {
            let mut fence = device.create_fence().unwrap();
            let init_cmd = cmd_encoder.end_encoding().unwrap();
            queue
                .submit(&[&init_cmd], Some((&mut fence, init_fence_value)))
                .unwrap();
            device.wait(&fence, init_fence_value, !0).unwrap();
            cmd_encoder.reset_all(iter::once(init_cmd));
            fence
        };

        Ok(Self {
            instance,
            adapter,
            surface,
            surface_format: surface_config.format,
            device,
            queue,
            pipeline,
            contexts: vec![ExecutionContext {
                encoder: cmd_encoder,
                fence,
                fence_value: init_fence_value + 1,
                used_views: Vec::new(),
                used_cmd_bufs: Vec::new(),
                frames_recorded: 0,
            }],
            context_index: 0,
            extent: [window_size.0, window_size.1],
            start: Instant::now(),
            pipeline_layout,
            bind_group,
            texture,
            instances,
            instances_buffer,
            blas,
            tlas,
            scratch_buffer,
            time: 0.0,
            indices_buffer,
            vertices_buffer,
            uniform_buffer,
            texture_view,
            bgl,
            shader_module,
        })
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {}

    fn render(&mut self) {
        let ctx = &mut self.contexts[self.context_index];

        let surface_tex = unsafe { self.surface.acquire_texture(None).unwrap().unwrap().texture };

        let target_barrier0 = hal::TextureBarrier {
            texture: surface_tex.borrow(),
            range: wgt::ImageSubresourceRange::default(),
            usage: hal::TextureUses::UNINITIALIZED..hal::TextureUses::COPY_DST,
        };

        let instances_buffer_size = self.instances.len() * std::mem::size_of::<Instance>();

        let tlas_flags = hal::AccelerationStructureBuildFlags::PREFER_FAST_TRACE
            | hal::AccelerationStructureBuildFlags::ALLOW_UPDATE;

        self.time += 1.0 / 60.0;

        self.instances[0] = Instance {
            transform: transpose_matrix_for_acceleration_structure_instance(Mat4::from_rotation_y(
                self.time,
            )),
            instance_custom_index_and_mask: pack_24_8(0, 0xff),
            instance_shader_binding_table_record_offset_and_flags: pack_24_8(0, 0),
            acceleration_structure_reference: unsafe {
                self.device
                    .get_acceleration_structure_device_address(&self.blas)
            },
        };

        unsafe {
            let mapping = self
                .device
                .map_buffer(&self.instances_buffer, 0..instances_buffer_size as u64)
                .unwrap();
            ptr::copy_nonoverlapping(
                self.instances.as_ptr() as *const u8,
                mapping.ptr.as_ptr(),
                instances_buffer_size,
            );
            self.device.unmap_buffer(&self.instances_buffer).unwrap();
            assert!(mapping.is_coherent);
        }

        unsafe {
            ctx.encoder.begin_encoding(Some("frame")).unwrap();

            ctx.encoder
                .build_acceleration_structures(&hal::BuildAccelerationStructureDescriptor {
                    geometry: &hal::AccelerationStructureGeometry::Instances {
                        buffer: &self.instances_buffer,
                    },
                    format: hal::AccelerationStructureFormat::TopLevel,
                    mode: hal::AccelerationStructureBuildMode::Update,
                    flags: tlas_flags,
                    primitive_count: self.instances.len() as u32,
                    primitive_offset: 0,
                    destination_acceleration_structure: &self.tlas,
                    scratch_buffer: &self.scratch_buffer,
                });

            let as_barrier = hal::BufferBarrier {
                buffer: &self.scratch_buffer,
                usage: hal::BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT
                    ..hal::BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT,
            };
            ctx.encoder.transition_buffers(iter::once(as_barrier));

            ctx.encoder.transition_textures(iter::once(target_barrier0));
        }

        let surface_view_desc = hal::TextureViewDescriptor {
            label: None,
            format: self.surface_format,
            dimension: wgt::TextureViewDimension::D2,
            usage: hal::TextureUses::COPY_DST,
            range: wgt::ImageSubresourceRange::default(),
        };
        let surface_tex_view = unsafe {
            self.device
                .create_texture_view(surface_tex.borrow(), &surface_view_desc)
                .unwrap()
        };
        unsafe {
            ctx.encoder
                .begin_compute_pass(&hal::ComputePassDescriptor { label: None });
            ctx.encoder.set_compute_pipeline(&self.pipeline);
            ctx.encoder
                .set_bind_group(&self.pipeline_layout, 0, &self.bind_group, &[]);
            ctx.encoder.dispatch([512 / 8, 512 / 8, 1]);
        }

        ctx.frames_recorded += 1;
        let do_fence = ctx.frames_recorded > COMMAND_BUFFER_PER_CONTEXT;

        let target_barrier1 = hal::TextureBarrier {
            texture: surface_tex.borrow(),
            range: wgt::ImageSubresourceRange::default(),
            usage: hal::TextureUses::COPY_DST..hal::TextureUses::PRESENT,
        };
        let target_barrier2 = hal::TextureBarrier {
            texture: &self.texture,
            range: wgt::ImageSubresourceRange::default(),
            usage: hal::TextureUses::STORAGE_READ_WRITE..hal::TextureUses::COPY_SRC,
        };
        let target_barrier3 = hal::TextureBarrier {
            texture: &self.texture,
            range: wgt::ImageSubresourceRange::default(),
            usage: hal::TextureUses::COPY_SRC..hal::TextureUses::STORAGE_READ_WRITE,
        };
        unsafe {
            ctx.encoder.end_compute_pass();
            ctx.encoder.transition_textures(iter::once(target_barrier2));
            ctx.encoder.copy_texture_to_texture(
                &self.texture,
                hal::TextureUses::COPY_SRC,
                &surface_tex.borrow(),
                std::iter::once(hal::TextureCopy {
                    src_base: hal::TextureCopyBase {
                        mip_level: 0,
                        array_layer: 0,
                        origin: wgt::Origin3d::ZERO,
                        aspect: hal::FormatAspects::COLOR,
                    },
                    dst_base: hal::TextureCopyBase {
                        mip_level: 0,
                        array_layer: 0,
                        origin: wgt::Origin3d::ZERO,
                        aspect: hal::FormatAspects::COLOR,
                    },
                    size: hal::CopyExtent {
                        width: 512,
                        height: 512,
                        depth: 1,
                    },
                }),
            );
            ctx.encoder.transition_textures(iter::once(target_barrier1));
            ctx.encoder.transition_textures(iter::once(target_barrier3));
        }

        unsafe {
            let cmd_buf = ctx.encoder.end_encoding().unwrap();
            let fence_param = if do_fence {
                Some((&mut ctx.fence, ctx.fence_value))
            } else {
                None
            };
            self.queue.submit(&[&cmd_buf], fence_param).unwrap();
            self.queue.present(&mut self.surface, surface_tex).unwrap();
            ctx.used_cmd_bufs.push(cmd_buf);
            ctx.used_views.push(surface_tex_view);
        };

        if do_fence {
            log::info!("Context switch from {}", self.context_index);
            let old_fence_value = ctx.fence_value;
            if self.contexts.len() == 1 {
                let hal_desc = hal::CommandEncoderDescriptor {
                    label: None,
                    queue: &self.queue,
                };
                self.contexts.push(unsafe {
                    ExecutionContext {
                        encoder: self.device.create_command_encoder(&hal_desc).unwrap(),
                        fence: self.device.create_fence().unwrap(),
                        fence_value: 0,
                        used_views: Vec::new(),
                        used_cmd_bufs: Vec::new(),
                        frames_recorded: 0,
                    }
                });
            }
            self.context_index = (self.context_index + 1) % self.contexts.len();
            let next = &mut self.contexts[self.context_index];
            unsafe {
                next.wait_and_clear(&self.device);
            }
            next.fence_value = old_fence_value + 1;
        }
    }

    fn exit(mut self) {
        unsafe {
            {
                let ctx = &mut self.contexts[self.context_index];
                self.queue
                    .submit(&[], Some((&mut ctx.fence, ctx.fence_value)))
                    .unwrap();
            }

            for mut ctx in self.contexts {
                ctx.wait_and_clear(&self.device);
                self.device.destroy_command_encoder(ctx.encoder);
                self.device.destroy_fence(ctx.fence);
            }

            self.device.destroy_bind_group(self.bind_group);
            self.device.destroy_buffer(self.scratch_buffer);
            self.device.destroy_buffer(self.instances_buffer);
            self.device.destroy_buffer(self.indices_buffer);
            self.device.destroy_buffer(self.vertices_buffer);
            self.device.destroy_buffer(self.uniform_buffer);
            self.device.destroy_acceleration_structure(self.tlas);
            self.device.destroy_acceleration_structure(self.blas);
            self.device.destroy_texture_view(self.texture_view);
            self.device.destroy_texture(self.texture);
            self.device.destroy_compute_pipeline(self.pipeline);
            self.device.destroy_pipeline_layout(self.pipeline_layout);
            self.device.destroy_bind_group_layout(self.bgl);
            self.device.destroy_shader_module(self.shader_module);

            self.surface.unconfigure(&self.device);
            self.device.exit(self.queue);
            self.instance.destroy_surface(self.surface);
            drop(self.adapter);
        }
    }
}

#[cfg(all(feature = "metal"))]
type Api = hal::api::Metal;
#[cfg(all(feature = "vulkan", not(feature = "metal")))]
type Api = hal::api::Vulkan;
#[cfg(all(feature = "gles", not(feature = "metal"), not(feature = "vulkan")))]
type Api = hal::api::Gles;
#[cfg(all(
    feature = "dx12",
    not(feature = "metal"),
    not(feature = "vulkan"),
    not(feature = "gles")
))]
type Api = hal::api::Dx12;
#[cfg(not(any(
    feature = "metal",
    feature = "vulkan",
    feature = "gles",
    feature = "dx12"
)))]
type Api = hal::api::Empty;

fn main() {
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("hal-bunnymark")
        .with_inner_size(winit::dpi::PhysicalSize {
            width: 512,
            height: 512,
        })
        .build(&event_loop)
        .unwrap();

    let example_result = Example::<Api>::init(&window);
    let mut example = Some(example_result.expect("Selected backend is not supported"));

    event_loop.run(move |event, _, control_flow| {
        let _ = &window; // force ownership by the closure
        *control_flow = winit::event_loop::ControlFlow::Poll;
        match event {
            winit::event::Event::RedrawEventsCleared => {
                window.request_redraw();
            }
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::KeyboardInput {
                    input:
                        winit::event::KeyboardInput {
                            virtual_keycode: Some(winit::event::VirtualKeyCode::Escape),
                            state: winit::event::ElementState::Pressed,
                            ..
                        },
                    ..
                }
                | winit::event::WindowEvent::CloseRequested => {
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }
                _ => {
                    example.as_mut().unwrap().update(event);
                }
            },
            winit::event::Event::RedrawRequested(_) => {
                let ex = example.as_mut().unwrap();

                ex.render();
            }
            winit::event::Event::LoopDestroyed => {
                example.take().unwrap().exit();
            }
            _ => {}
        }
    });
}
