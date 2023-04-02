//! This example shows basic usage of wgpu-hal by rendering
//! a ton of moving sprites, each with a separate texture and draw call.
extern crate wgpu_hal as hal;

use hal::{
    Adapter as _, CommandEncoder as _, Device as _, Instance as _, Queue as _, Surface as _,
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

use std::{
    borrow::{Borrow, Cow},
    iter, mem, ptr,
    time::Instant,
};

const MAX_BUNNIES: usize = 1 << 20;
const BUNNY_SIZE: f32 = 0.15 * 256.0;
const GRAVITY: f32 = -9.8 * 100.0;
const MAX_VELOCITY: f32 = 750.0;
const COMMAND_BUFFER_PER_CONTEXT: usize = 100;
const DESIRED_FRAMES: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy)]
struct Globals {
    mvp: [[f32; 4]; 4],
    size: [f32; 2],
    pad: [f32; 2],
}

#[repr(C, align(256))]
#[derive(Clone, Copy)]
struct Locals {
    position: [f32; 2],
    velocity: [f32; 2],
    color: u32,
    _pad: u32,
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
    global_group: A::BindGroup,
    local_group: A::BindGroup,
    global_group_layout: A::BindGroupLayout,
    local_group_layout: A::BindGroupLayout,
    pipeline_layout: A::PipelineLayout,
    shader: A::ShaderModule,
    pipeline: A::RenderPipeline,
    bunnies: Vec<Locals>,
    local_buffer: A::Buffer,
    local_alignment: u32,
    global_buffer: A::Buffer,
    sampler: A::Sampler,
    texture: A::Texture,
    texture_view: A::TextureView,
    contexts: Vec<ExecutionContext<A>>,
    context_index: usize,
    extent: [u32; 2],
    start: Instant,
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
            // Can't rely on having DXC available, so use FXC instead
            dx12_shader_compiler: wgt::Dx12Compiler::Fxc,
        };
        let instance = unsafe { A::Instance::init(&instance_desc)? };
        let mut surface = unsafe {
            instance
                .create_surface(window.raw_display_handle(), window.raw_window_handle())
                .unwrap()
        };

        let (adapter, capabilities) = unsafe {
            let mut adapters = instance.enumerate_adapters();
            if adapters.is_empty() {
                return Err(hal::InstanceError);
            }
            let exposed = adapters.swap_remove(0);
            (exposed.adapter, exposed.capabilities)
        };
        let surface_caps =
            unsafe { adapter.surface_capabilities(&surface) }.ok_or(hal::InstanceError)?;
        log::info!("Surface caps: {:#?}", surface_caps);

        let hal::OpenDevice { device, mut queue } = unsafe {
            adapter
                .open(wgt::Features::empty(), &wgt::Limits::default())
                .unwrap()
        };

        let window_size: (u32, u32) = window.inner_size().into();
        let surface_config = hal::SurfaceConfiguration {
            swap_chain_size: DESIRED_FRAMES.clamp(
                *surface_caps.swap_chain_sizes.start(),
                *surface_caps.swap_chain_sizes.end(),
            ),
            present_mode: wgt::PresentMode::Fifo,
            composite_alpha_mode: wgt::CompositeAlphaMode::Opaque,
            format: wgt::TextureFormat::Bgra8UnormSrgb,
            extent: wgt::Extent3d {
                width: window_size.0,
                height: window_size.1,
                depth_or_array_layers: 1,
            },
            usage: hal::TextureUses::COLOR_TARGET,
            view_formats: vec![],
        };
        unsafe {
            surface.configure(&device, &surface_config).unwrap();
        };

        let naga_shader = {
            let shader_file = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("examples")
                .join("halmark")
                .join("shader.wgsl");
            let source = std::fs::read_to_string(shader_file).unwrap();
            let module = naga::front::wgsl::Frontend::new().parse(&source).unwrap();
            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap();
            hal::NagaShader {
                module: Cow::Owned(module),
                info,
            }
        };
        let shader_desc = hal::ShaderModuleDescriptor {
            label: None,
            runtime_checks: false,
        };
        let shader = unsafe {
            device
                .create_shader_module(&shader_desc, hal::ShaderInput::Naga(naga_shader))
                .unwrap()
        };

        let global_bgl_desc = hal::BindGroupLayoutDescriptor {
            label: None,
            flags: hal::BindGroupLayoutFlags::empty(),
            entries: &[
                wgt::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgt::ShaderStages::VERTEX,
                    ty: wgt::BindingType::Buffer {
                        ty: wgt::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgt::BufferSize::new(mem::size_of::<Globals>() as _),
                    },
                    count: None,
                },
                wgt::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgt::ShaderStages::FRAGMENT,
                    ty: wgt::BindingType::Texture {
                        sample_type: wgt::TextureSampleType::Float { filterable: true },
                        view_dimension: wgt::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgt::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgt::ShaderStages::FRAGMENT,
                    ty: wgt::BindingType::Sampler(wgt::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        };

        let global_group_layout =
            unsafe { device.create_bind_group_layout(&global_bgl_desc).unwrap() };

        let local_bgl_desc = hal::BindGroupLayoutDescriptor {
            label: None,
            flags: hal::BindGroupLayoutFlags::empty(),
            entries: &[wgt::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgt::ShaderStages::VERTEX,
                ty: wgt::BindingType::Buffer {
                    ty: wgt::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgt::BufferSize::new(mem::size_of::<Locals>() as _),
                },
                count: None,
            }],
        };
        let local_group_layout =
            unsafe { device.create_bind_group_layout(&local_bgl_desc).unwrap() };

        let pipeline_layout_desc = hal::PipelineLayoutDescriptor {
            label: None,
            flags: hal::PipelineLayoutFlags::empty(),
            bind_group_layouts: &[&global_group_layout, &local_group_layout],
            push_constant_ranges: &[],
        };
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_desc)
                .unwrap()
        };

        let pipeline_desc = hal::RenderPipelineDescriptor {
            label: None,
            layout: &pipeline_layout,
            vertex_stage: hal::ProgrammableStage {
                module: &shader,
                entry_point: "vs_main",
            },
            vertex_buffers: &[],
            fragment_stage: Some(hal::ProgrammableStage {
                module: &shader,
                entry_point: "fs_main",
            }),
            primitive: wgt::PrimitiveState {
                topology: wgt::PrimitiveTopology::TriangleStrip,
                ..wgt::PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: wgt::MultisampleState::default(),
            color_targets: &[Some(wgt::ColorTargetState {
                format: surface_config.format,
                blend: Some(wgt::BlendState::ALPHA_BLENDING),
                write_mask: wgt::ColorWrites::default(),
            })],
            multiview: None,
        };
        let pipeline = unsafe { device.create_render_pipeline(&pipeline_desc).unwrap() };

        let texture_data = vec![0xFFu8; 4];

        let staging_buffer_desc = hal::BufferDescriptor {
            label: Some("stage"),
            size: texture_data.len() as wgt::BufferAddress,
            usage: hal::BufferUses::MAP_WRITE | hal::BufferUses::COPY_SRC,
            memory_flags: hal::MemoryFlags::TRANSIENT | hal::MemoryFlags::PREFER_COHERENT,
        };
        let staging_buffer = unsafe { device.create_buffer(&staging_buffer_desc).unwrap() };
        unsafe {
            let mapping = device
                .map_buffer(&staging_buffer, 0..staging_buffer_desc.size)
                .unwrap();
            ptr::copy_nonoverlapping(
                texture_data.as_ptr(),
                mapping.ptr.as_ptr(),
                texture_data.len(),
            );
            device.unmap_buffer(&staging_buffer).unwrap();
            assert!(mapping.is_coherent);
        }

        let texture_desc = hal::TextureDescriptor {
            label: None,
            size: wgt::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgt::TextureDimension::D2,
            format: wgt::TextureFormat::Rgba8UnormSrgb,
            usage: hal::TextureUses::COPY_DST | hal::TextureUses::RESOURCE,
            memory_flags: hal::MemoryFlags::empty(),
            view_formats: vec![],
        };
        let texture = unsafe { device.create_texture(&texture_desc).unwrap() };

        let cmd_encoder_desc = hal::CommandEncoderDescriptor {
            label: None,
            queue: &queue,
        };
        let mut cmd_encoder = unsafe { device.create_command_encoder(&cmd_encoder_desc).unwrap() };
        unsafe { cmd_encoder.begin_encoding(Some("init")).unwrap() };
        {
            let buffer_barrier = hal::BufferBarrier {
                buffer: &staging_buffer,
                usage: hal::BufferUses::empty()..hal::BufferUses::COPY_SRC,
            };
            let texture_barrier1 = hal::TextureBarrier {
                texture: &texture,
                range: wgt::ImageSubresourceRange::default(),
                usage: hal::TextureUses::UNINITIALIZED..hal::TextureUses::COPY_DST,
            };
            let texture_barrier2 = hal::TextureBarrier {
                texture: &texture,
                range: wgt::ImageSubresourceRange::default(),
                usage: hal::TextureUses::COPY_DST..hal::TextureUses::RESOURCE,
            };
            let copy = hal::BufferTextureCopy {
                buffer_layout: wgt::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: None,
                },
                texture_base: hal::TextureCopyBase {
                    origin: wgt::Origin3d::ZERO,
                    mip_level: 0,
                    array_layer: 0,
                    aspect: hal::FormatAspects::COLOR,
                },
                size: hal::CopyExtent {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            };
            unsafe {
                cmd_encoder.transition_buffers(iter::once(buffer_barrier));
                cmd_encoder.transition_textures(iter::once(texture_barrier1));
                cmd_encoder.copy_buffer_to_texture(&staging_buffer, &texture, iter::once(copy));
                cmd_encoder.transition_textures(iter::once(texture_barrier2));
            }
        }

        let sampler_desc = hal::SamplerDescriptor {
            label: None,
            address_modes: [wgt::AddressMode::ClampToEdge; 3],
            mag_filter: wgt::FilterMode::Linear,
            min_filter: wgt::FilterMode::Nearest,
            mipmap_filter: wgt::FilterMode::Nearest,
            lod_clamp: 0.0..32.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        };
        let sampler = unsafe { device.create_sampler(&sampler_desc).unwrap() };

        let globals = Globals {
            // cgmath::ortho() projection
            mvp: [
                [2.0 / window_size.0 as f32, 0.0, 0.0, 0.0],
                [0.0, 2.0 / window_size.1 as f32, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [-1.0, -1.0, 0.0, 1.0],
            ],
            size: [BUNNY_SIZE; 2],
            pad: [0.0; 2],
        };

        let global_buffer_desc = hal::BufferDescriptor {
            label: Some("global"),
            size: mem::size_of::<Globals>() as wgt::BufferAddress,
            usage: hal::BufferUses::MAP_WRITE | hal::BufferUses::UNIFORM,
            memory_flags: hal::MemoryFlags::PREFER_COHERENT,
        };
        let global_buffer = unsafe {
            let buffer = device.create_buffer(&global_buffer_desc).unwrap();
            let mapping = device
                .map_buffer(&buffer, 0..global_buffer_desc.size)
                .unwrap();
            ptr::copy_nonoverlapping(
                &globals as *const Globals as *const u8,
                mapping.ptr.as_ptr(),
                mem::size_of::<Globals>(),
            );
            device.unmap_buffer(&buffer).unwrap();
            assert!(mapping.is_coherent);
            buffer
        };

        let local_alignment = wgt::math::align_to(
            mem::size_of::<Locals>() as u32,
            capabilities.limits.min_uniform_buffer_offset_alignment,
        );
        let local_buffer_desc = hal::BufferDescriptor {
            label: Some("local"),
            size: (MAX_BUNNIES as wgt::BufferAddress) * (local_alignment as wgt::BufferAddress),
            usage: hal::BufferUses::MAP_WRITE | hal::BufferUses::UNIFORM,
            memory_flags: hal::MemoryFlags::PREFER_COHERENT,
        };
        let local_buffer = unsafe { device.create_buffer(&local_buffer_desc).unwrap() };

        let view_desc = hal::TextureViewDescriptor {
            label: None,
            format: texture_desc.format,
            dimension: wgt::TextureViewDimension::D2,
            usage: hal::TextureUses::RESOURCE,
            range: wgt::ImageSubresourceRange::default(),
        };
        let texture_view = unsafe { device.create_texture_view(&texture, &view_desc).unwrap() };

        let global_group = {
            let global_buffer_binding = hal::BufferBinding {
                buffer: &global_buffer,
                offset: 0,
                size: None,
            };
            let texture_binding = hal::TextureBinding {
                view: &texture_view,
                usage: hal::TextureUses::RESOURCE,
            };
            let global_group_desc = hal::BindGroupDescriptor {
                label: Some("global"),
                layout: &global_group_layout,
                buffers: &[global_buffer_binding],
                samplers: &[&sampler],
                textures: &[texture_binding],
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
            unsafe { device.create_bind_group(&global_group_desc).unwrap() }
        };

        let local_group = {
            let local_buffer_binding = hal::BufferBinding {
                buffer: &local_buffer,
                offset: 0,
                size: wgt::BufferSize::new(mem::size_of::<Locals>() as _),
            };
            let local_group_desc = hal::BindGroupDescriptor {
                label: Some("local"),
                layout: &local_group_layout,
                buffers: &[local_buffer_binding],
                samplers: &[],
                textures: &[],
                entries: &[hal::BindGroupEntry {
                    binding: 0,
                    resource_index: 0,
                    count: 1,
                }],
            };
            unsafe { device.create_bind_group(&local_group_desc).unwrap() }
        };

        let init_fence_value = 1;
        let fence = unsafe {
            let mut fence = device.create_fence().unwrap();
            let init_cmd = cmd_encoder.end_encoding().unwrap();
            queue
                .submit(&[&init_cmd], Some((&mut fence, init_fence_value)))
                .unwrap();
            device.wait(&fence, init_fence_value, !0).unwrap();
            device.destroy_buffer(staging_buffer);
            cmd_encoder.reset_all(iter::once(init_cmd));
            fence
        };

        Ok(Example {
            instance,
            surface,
            surface_format: surface_config.format,
            adapter,
            device,
            queue,
            pipeline_layout,
            shader,
            pipeline,
            global_group,
            local_group,
            global_group_layout,
            local_group_layout,
            bunnies: Vec::new(),
            local_buffer,
            local_alignment,
            global_buffer,
            sampler,
            texture,
            texture_view,
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
        })
    }

    fn is_empty(&self) -> bool {
        self.bunnies.is_empty()
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

            self.device.destroy_bind_group(self.local_group);
            self.device.destroy_bind_group(self.global_group);
            self.device.destroy_buffer(self.local_buffer);
            self.device.destroy_buffer(self.global_buffer);
            self.device.destroy_texture_view(self.texture_view);
            self.device.destroy_texture(self.texture);
            self.device.destroy_sampler(self.sampler);
            self.device.destroy_shader_module(self.shader);
            self.device.destroy_render_pipeline(self.pipeline);
            self.device
                .destroy_bind_group_layout(self.local_group_layout);
            self.device
                .destroy_bind_group_layout(self.global_group_layout);
            self.device.destroy_pipeline_layout(self.pipeline_layout);

            self.surface.unconfigure(&self.device);
            self.device.exit(self.queue);
            self.instance.destroy_surface(self.surface);
            drop(self.adapter);
        }
    }

    fn update(&mut self, event: winit::event::WindowEvent) {
        if let winit::event::WindowEvent::KeyboardInput {
            input:
                winit::event::KeyboardInput {
                    virtual_keycode: Some(winit::event::VirtualKeyCode::Space),
                    state: winit::event::ElementState::Pressed,
                    ..
                },
            ..
        } = event
        {
            let spawn_count = 64 + self.bunnies.len() / 2;
            let elapsed = self.start.elapsed();
            let color = elapsed.as_nanos() as u32;
            println!(
                "Spawning {} bunnies, total at {}",
                spawn_count,
                self.bunnies.len() + spawn_count
            );
            for i in 0..spawn_count {
                let random = ((elapsed.as_nanos() * (i + 1) as u128) & 0xFF) as f32 / 255.0;
                let speed = random * MAX_VELOCITY - (MAX_VELOCITY * 0.5);
                self.bunnies.push(Locals {
                    position: [0.0, 0.5 * (self.extent[1] as f32)],
                    velocity: [speed, 0.0],
                    color,
                    _pad: 0,
                });
            }
        }
    }

    fn render(&mut self) {
        let delta = 0.01;
        for bunny in self.bunnies.iter_mut() {
            bunny.position[0] += bunny.velocity[0] * delta;
            bunny.position[1] += bunny.velocity[1] * delta;
            bunny.velocity[1] += GRAVITY * delta;
            if (bunny.velocity[0] > 0.0
                && bunny.position[0] + 0.5 * BUNNY_SIZE > self.extent[0] as f32)
                || (bunny.velocity[0] < 0.0 && bunny.position[0] - 0.5 * BUNNY_SIZE < 0.0)
            {
                bunny.velocity[0] *= -1.0;
            }
            if bunny.velocity[1] < 0.0 && bunny.position[1] < 0.5 * BUNNY_SIZE {
                bunny.velocity[1] *= -1.0;
            }
        }

        if !self.bunnies.is_empty() {
            let size = self.bunnies.len() * self.local_alignment as usize;
            unsafe {
                let mapping = self
                    .device
                    .map_buffer(&self.local_buffer, 0..size as wgt::BufferAddress)
                    .unwrap();
                ptr::copy_nonoverlapping(
                    self.bunnies.as_ptr() as *const u8,
                    mapping.ptr.as_ptr(),
                    size,
                );
                assert!(mapping.is_coherent);
                self.device.unmap_buffer(&self.local_buffer).unwrap();
            }
        }

        let ctx = &mut self.contexts[self.context_index];

        let surface_tex = unsafe { self.surface.acquire_texture(None).unwrap().unwrap().texture };

        let target_barrier0 = hal::TextureBarrier {
            texture: surface_tex.borrow(),
            range: wgt::ImageSubresourceRange::default(),
            usage: hal::TextureUses::UNINITIALIZED..hal::TextureUses::COLOR_TARGET,
        };
        unsafe {
            ctx.encoder.begin_encoding(Some("frame")).unwrap();
            ctx.encoder.transition_textures(iter::once(target_barrier0));
        }

        let surface_view_desc = hal::TextureViewDescriptor {
            label: None,
            format: self.surface_format,
            dimension: wgt::TextureViewDimension::D2,
            usage: hal::TextureUses::COLOR_TARGET,
            range: wgt::ImageSubresourceRange::default(),
        };
        let surface_tex_view = unsafe {
            self.device
                .create_texture_view(surface_tex.borrow(), &surface_view_desc)
                .unwrap()
        };
        let pass_desc = hal::RenderPassDescriptor {
            label: None,
            extent: wgt::Extent3d {
                width: self.extent[0],
                height: self.extent[1],
                depth_or_array_layers: 1,
            },
            sample_count: 1,
            color_attachments: &[Some(hal::ColorAttachment {
                target: hal::Attachment {
                    view: &surface_tex_view,
                    usage: hal::TextureUses::COLOR_TARGET,
                },
                resolve_target: None,
                ops: hal::AttachmentOps::STORE,
                clear_value: wgt::Color {
                    r: 0.1,
                    g: 0.2,
                    b: 0.3,
                    a: 1.0,
                },
            })],
            depth_stencil_attachment: None,
            multiview: None,
            timestamp_writes: &[],
        };
        unsafe {
            ctx.encoder.begin_render_pass(&pass_desc);
            ctx.encoder.set_render_pipeline(&self.pipeline);
            ctx.encoder
                .set_bind_group(&self.pipeline_layout, 0, &self.global_group, &[]);
        }

        for i in 0..self.bunnies.len() {
            let offset = (i as wgt::DynamicOffset) * (self.local_alignment as wgt::DynamicOffset);
            unsafe {
                ctx.encoder
                    .set_bind_group(&self.pipeline_layout, 1, &self.local_group, &[offset]);
                ctx.encoder.draw(0, 4, 0, 1);
            }
        }

        ctx.frames_recorded += 1;
        let do_fence = ctx.frames_recorded > COMMAND_BUFFER_PER_CONTEXT;

        let target_barrier1 = hal::TextureBarrier {
            texture: surface_tex.borrow(),
            range: wgt::ImageSubresourceRange::default(),
            usage: hal::TextureUses::COLOR_TARGET..hal::TextureUses::PRESENT,
        };
        unsafe {
            ctx.encoder.end_render_pass();
            ctx.encoder.transition_textures(iter::once(target_barrier1));
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
        .build(&event_loop)
        .unwrap();

    let example_result = Example::<Api>::init(&window);
    let mut example = Some(example_result.expect("Selected backend is not supported"));

    let mut last_frame_inst = Instant::now();
    let (mut frame_count, mut accum_time) = (0, 0.0);

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
                {
                    accum_time += last_frame_inst.elapsed().as_secs_f32();
                    last_frame_inst = Instant::now();
                    frame_count += 1;
                    if frame_count == 100 && !ex.is_empty() {
                        println!(
                            "Avg frame time {}ms",
                            accum_time * 1000.0 / frame_count as f32
                        );
                        accum_time = 0.0;
                        frame_count = 0;
                    }
                }
                ex.render();
            }
            winit::event::Event::LoopDestroyed => {
                example.take().unwrap().exit();
            }
            _ => {}
        }
    });
}
