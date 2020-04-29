/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/*! This is a player for WebGPU traces.
 *
 * # Notes
 * - we call device_maintain_ids() before creating any refcounted resource,
 *   which is basically everything except for BGL and shader modules,
 *   so that we don't accidentally try to use the same ID.
!*/

use wgc::device::trace;

use std::{
    ffi::CString,
    fmt::Debug,
    fs::File,
    marker::PhantomData,
    path::{Path, PathBuf},
    ptr,
};

macro_rules! gfx_select {
    ($id:expr => $global:ident.$method:ident( $($param:expr),+ )) => {
        match $id.backend() {
            #[cfg(not(any(target_os = "ios", target_os = "macos")))]
            wgt::Backend::Vulkan => $global.$method::<wgc::backend::Vulkan>( $($param),+ ),
            #[cfg(any(target_os = "ios", target_os = "macos"))]
            wgt::Backend::Metal => $global.$method::<wgc::backend::Metal>( $($param),+ ),
            #[cfg(windows)]
            wgt::Backend::Dx12 => $global.$method::<wgc::backend::Dx12>( $($param),+ ),
            #[cfg(windows)]
            wgt::Backend::Dx11 => $global.$method::<wgc::backend::Dx11>( $($param),+ ),
            _ => unreachable!()
        }
    };
}

struct Label(Option<CString>);
impl Label {
    fn new(text: &str) -> Self {
        Self(if text.is_empty() {
            None
        } else {
            Some(CString::new(text).expect("invalid label"))
        })
    }

    fn as_ptr(&self) -> *const std::os::raw::c_char {
        match self.0 {
            Some(ref c_string) => c_string.as_ptr(),
            None => ptr::null(),
        }
    }
}

struct OwnedProgrammableStage {
    desc: wgc::pipeline::ProgrammableStageDescriptor,
    #[allow(dead_code)]
    entry_point: CString,
}

impl From<trace::ProgrammableStageDescriptor> for OwnedProgrammableStage {
    fn from(stage: trace::ProgrammableStageDescriptor) -> Self {
        let entry_point = CString::new(stage.entry_point.as_str()).unwrap();
        OwnedProgrammableStage {
            desc: wgc::pipeline::ProgrammableStageDescriptor {
                module: stage.module,
                entry_point: entry_point.as_ptr(),
            },
            entry_point,
        }
    }
}

#[derive(Debug)]
struct IdentityPassThrough<I>(PhantomData<I>);

impl<I: Clone + Debug + wgc::id::TypedId> wgc::hub::IdentityHandler<I> for IdentityPassThrough<I> {
    type Input = I;
    fn process(&self, id: I, backend: wgt::Backend) -> I {
        let (index, epoch, _backend) = id.unzip();
        I::zip(index, epoch, backend)
    }
    fn free(&self, _id: I) {}
}

struct IdentityPassThroughFactory;

impl<I: Clone + Debug + wgc::id::TypedId> wgc::hub::IdentityHandlerFactory<I>
    for IdentityPassThroughFactory
{
    type Filter = IdentityPassThrough<I>;
    fn spawn(&self, _min_index: u32) -> Self::Filter {
        IdentityPassThrough(PhantomData)
    }
}
impl wgc::hub::GlobalIdentityHandlerFactory for IdentityPassThroughFactory {}

trait GlobalExt {
    fn encode_commands<B: wgc::hub::GfxBackend>(
        &self,
        encoder: wgc::id::CommandEncoderId,
        commands: Vec<trace::Command>,
    ) -> wgc::id::CommandBufferId;
    /*fn process<B: wgc::hub::GfxBackend>(
        &self,
        device: wgc::id::DeviceId,
        action: trace::Action,
    ) {

    }*/
}

impl GlobalExt for wgc::hub::Global<IdentityPassThroughFactory> {
    fn encode_commands<B: wgc::hub::GfxBackend>(
        &self,
        encoder: wgc::id::CommandEncoderId,
        commands: Vec<trace::Command>,
    ) -> wgc::id::CommandBufferId {
        for command in commands {
            match command {
                trace::Command::CopyBufferToBuffer {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => self.command_encoder_copy_buffer_to_buffer::<B>(
                    encoder, src, src_offset, dst, dst_offset, size,
                ),
                trace::Command::CopyBufferToTexture { src, dst, size } => {
                    self.command_encoder_copy_buffer_to_texture::<B>(encoder, &src, &dst, size)
                }
                trace::Command::CopyTextureToBuffer { src, dst, size } => {
                    self.command_encoder_copy_texture_to_buffer::<B>(encoder, &src, &dst, size)
                }
                trace::Command::CopyTextureToTexture { src, dst, size } => {
                    self.command_encoder_copy_texture_to_texture::<B>(encoder, &src, &dst, size)
                }
                trace::Command::RunComputePass {
                    commands,
                    dynamic_offsets,
                } => unsafe {
                    let mut offsets = &dynamic_offsets[..];
                    let mut pass = wgc::command::RawPass::new_compute(encoder);
                    for com in commands {
                        pass.encode(&com);
                        if let wgc::command::ComputeCommand::SetBindGroup {
                            num_dynamic_offsets,
                            ..
                        } = com
                        {
                            pass.encode_slice(&offsets[..num_dynamic_offsets as usize]);
                            offsets = &offsets[num_dynamic_offsets as usize..];
                        }
                    }
                    let (data, _) = pass.finish_compute();
                    self.command_encoder_run_compute_pass::<B>(encoder, &data);
                },
                trace::Command::RunRenderPass {
                    target_colors,
                    target_depth_stencil,
                    commands,
                    dynamic_offsets,
                } => unsafe {
                    let mut offsets = &dynamic_offsets[..];
                    let mut pass = wgc::command::RawPass::new_render(
                        encoder,
                        &wgc::command::RenderPassDescriptor {
                            color_attachments: target_colors.as_ptr(),
                            color_attachments_length: target_colors.len(),
                            depth_stencil_attachment: target_depth_stencil.as_ref(),
                        },
                    );
                    for com in commands {
                        pass.encode(&com);
                        if let wgc::command::RenderCommand::SetBindGroup {
                            num_dynamic_offsets,
                            ..
                        } = com
                        {
                            pass.encode_slice(&offsets[..num_dynamic_offsets as usize]);
                            offsets = &offsets[num_dynamic_offsets as usize..];
                        }
                    }
                    let (data, _) = pass.finish_render();
                    self.command_encoder_run_render_pass::<B>(encoder, &data);
                },
            }
        }
        self.command_encoder_finish::<B>(encoder, &wgt::CommandBufferDescriptor { todo: 0 })
    }
}

fn main() {
    #[cfg(feature = "winit")]
    use winit::{event_loop::EventLoop, window::WindowBuilder};

    env_logger::init();

    //TODO: setting for the backend bits
    //TODO: setting for the target frame, or controls

    let dir = match std::env::args().nth(1) {
        Some(arg) if Path::new(&arg).is_dir() => PathBuf::from(arg),
        _ => panic!("Provide the dir path as the parameter"),
    };

    log::info!("Loading trace '{:?}'", dir);
    let file = File::open(dir.join(trace::FILE_NAME)).unwrap();
    let mut actions: Vec<trace::Action> = ron::de::from_reader(file).unwrap();
    actions.reverse(); // allows us to pop from the top
    log::info!("Found {} actions", actions.len());

    #[cfg(feature = "winit")]
    let mut event_loop = {
        log::info!("Creating a window");
        EventLoop::new()
    };
    #[cfg(feature = "winit")]
    let window = WindowBuilder::new()
        .with_title("wgpu player")
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

    let global = wgc::hub::Global::new("player", IdentityPassThroughFactory);
    let mut adapter_id_manager = wgc::hub::IdentityManager::default();
    let mut command_buffer_id_manager = wgc::hub::IdentityManager::default();

    #[cfg(feature = "winit")]
    let surface = global.instance_create_surface(
        raw_window_handle::HasRawWindowHandle::raw_window_handle(&window),
        wgc::id::TypedId::zip(0, 1, wgt::Backend::Empty),
    );

    let adapter = global
        .pick_adapter(
            &wgc::instance::RequestAdapterOptions {
                power_preference: wgt::PowerPreference::Default,
                #[cfg(feature = "winit")]
                compatible_surface: Some(surface),
                #[cfg(not(feature = "winit"))]
                compatible_surface: None,
            },
            wgc::instance::AdapterInputs::IdSet(
                &vec![
                    adapter_id_manager.alloc(wgt::Backend::Vulkan),
                    adapter_id_manager.alloc(wgt::Backend::Dx12),
                    adapter_id_manager.alloc(wgt::Backend::Metal),
                ],
                |id| id.backend(),
            ),
        )
        .unwrap();

    log::info!("Initializing the device");
    let device = match actions.pop() {
        Some(trace::Action::Init { limits }) => {
            gfx_select!(adapter => global.adapter_request_device(
                adapter,
                &wgt::DeviceDescriptor {
                    extensions: wgt::Extensions {
                        anisotropic_filtering: false,
                    },
                    limits,
                },
                wgc::id::TypedId::zip(1, 0, wgt::Backend::Empty)
            ))
        }
        _ => panic!("Expected Action::Init"),
    };
    let mut frame_count = 0;

    log::info!("Executing actions");
    for action in actions {
        use wgc::device::trace::Action as A;
        match action {
            A::Init { .. } => panic!("Unexpected Action::Init"),
            A::CreateBuffer { id, desc } => {
                let label = Label::new(&desc.label);
                gfx_select!(device => global.device_maintain_ids(device));
                gfx_select!(device => global.device_create_buffer(device, &desc.map_label(|_| label.as_ptr()), id));
            }
            A::DestroyBuffer(id) => {
                gfx_select!(device => global.buffer_destroy(id));
            }
            A::CreateTexture { id, desc } => {
                let label = Label::new(&desc.label);
                gfx_select!(device => global.device_maintain_ids(device));
                gfx_select!(device => global.device_create_texture(device, &desc.map_label(|_| label.as_ptr()), id));
            }
            A::DestroyTexture(id) => {
                gfx_select!(device => global.texture_destroy(id));
            }
            A::CreateTextureView {
                id,
                parent_id,
                desc,
            } => {
                let label = desc.as_ref().map_or(Label(None), |d| Label::new(&d.label));
                gfx_select!(device => global.device_maintain_ids(device));
                gfx_select!(device => global.texture_create_view(parent_id, desc.map(|d| d.map_label(|_| label.as_ptr())).as_ref(), id));
            }
            A::DestroyTextureView(id) => {
                gfx_select!(device => global.texture_view_destroy(id));
            }
            A::CreateSampler { id, desc } => {
                let label = Label::new(&desc.label);
                gfx_select!(device => global.device_maintain_ids(device));
                gfx_select!(device => global.device_create_sampler(device, &desc.map_label(|_| label.as_ptr()), id));
            }
            A::DestroySampler(id) => {
                gfx_select!(device => global.sampler_destroy(id));
            }
            A::CreateSwapChain { id, desc } => {
                #[cfg(feature = "winit")]
                {
                    log::info!("Initializing the swapchain");
                    assert_eq!(id.to_surface_id(), surface);
                    window.set_inner_size(winit::dpi::PhysicalSize::new(desc.width, desc.height));
                    gfx_select!(device => global.device_create_swap_chain(device, surface, &desc));
                }
                #[cfg(not(feature = "winit"))]
                {
                    let _ = (id, desc);
                    panic!("Enable `winit` feature to work with swapchains");
                }
            }
            A::GetSwapChainTexture { id, parent_id } => {
                gfx_select!(device => global.swap_chain_get_next_texture(parent_id, id)).unwrap();
            }
            A::PresentSwapChain(id) => {
                frame_count += 1;
                log::debug!("Presenting frame {}", frame_count);
                gfx_select!(device => global.swap_chain_present(id));
            }
            A::CreateBindGroupLayout { id, label, entries } => {
                let label = Label::new(&label);
                gfx_select!(device => global.device_create_bind_group_layout(
                    device,
                    &wgc::binding_model::BindGroupLayoutDescriptor {
                        label: label.as_ptr(),
                        entries: entries.as_ptr(),
                        entries_length: entries.len(),
                    },
                    id));
            }
            A::DestroyBindGroupLayout(id) => {
                gfx_select!(device => global.bind_group_layout_destroy(id));
            }
            A::CreatePipelineLayout {
                id,
                bind_group_layouts,
            } => {
                gfx_select!(device => global.device_maintain_ids(device));
                gfx_select!(device => global.device_create_pipeline_layout(
                    device,
                    &wgc::binding_model::PipelineLayoutDescriptor {
                        bind_group_layouts: bind_group_layouts.as_ptr(),
                        bind_group_layouts_length: bind_group_layouts.len(),
                    },
                    id));
            }
            A::DestroyPipelineLayout(id) => {
                gfx_select!(device => global.pipeline_layout_destroy(id));
            }
            A::CreateBindGroup {
                id,
                label,
                layout_id,
                entries,
            } => {
                use wgc::binding_model as bm;
                let label = Label::new(&label);
                let entry_vec = entries
                    .into_iter()
                    .map(|(binding, res)| wgc::binding_model::BindGroupEntry {
                        binding,
                        resource: match res {
                            trace::BindingResource::Buffer { id, offset, size } => {
                                bm::BindingResource::Buffer(bm::BufferBinding {
                                    buffer: id,
                                    offset,
                                    size,
                                })
                            }
                            trace::BindingResource::Sampler(id) => bm::BindingResource::Sampler(id),
                            trace::BindingResource::TextureView(id) => {
                                bm::BindingResource::TextureView(id)
                            }
                        },
                    })
                    .collect::<Vec<_>>();
                gfx_select!(device => global.device_maintain_ids(device));
                gfx_select!(device => global.device_create_bind_group(
                    device,
                    &wgc::binding_model::BindGroupDescriptor {
                        label: label.as_ptr(),
                        layout: layout_id,
                        entries: entry_vec.as_ptr(),
                        entries_length: entry_vec.len(),
                    },
                    id
                ));
            }
            A::DestroyBindGroup(id) => {
                gfx_select!(device => global.bind_group_destroy(id));
            }
            A::CreateShaderModule { id, data } => {
                let spv = wgt::read_spirv(File::open(dir.join(data)).unwrap()).unwrap();
                gfx_select!(device => global.device_create_shader_module(
                    device,
                    &wgc::pipeline::ShaderModuleDescriptor {
                        code: wgc::U32Array {
                            bytes: spv.as_ptr(),
                            length: spv.len(),
                        },
                    },
                    id
                ));
            }
            A::DestroyShaderModule(id) => {
                gfx_select!(device => global.shader_module_destroy(id));
            }
            A::CreateComputePipeline { id, desc } => {
                let cs_stage = OwnedProgrammableStage::from(desc.compute_stage);
                gfx_select!(device => global.device_maintain_ids(device));
                gfx_select!(device => global.device_create_compute_pipeline(
                    device,
                    &wgc::pipeline::ComputePipelineDescriptor {
                        layout: desc.layout,
                        compute_stage: cs_stage.desc,
                    },
                    id));
            }
            A::DestroyComputePipeline(id) => {
                gfx_select!(device => global.compute_pipeline_destroy(id));
            }
            A::CreateRenderPipeline { id, desc } => {
                let vs_stage = OwnedProgrammableStage::from(desc.vertex_stage);
                let fs_stage = desc.fragment_stage.map(OwnedProgrammableStage::from);
                let vertex_buffers = desc
                    .vertex_state
                    .vertex_buffers
                    .iter()
                    .map(|vb| wgc::pipeline::VertexBufferLayoutDescriptor {
                        array_stride: vb.array_stride,
                        step_mode: vb.step_mode,
                        attributes: vb.attributes.as_ptr(),
                        attributes_length: vb.attributes.len(),
                    })
                    .collect::<Vec<_>>();
                gfx_select!(device => global.device_maintain_ids(device));
                gfx_select!(device => global.device_create_render_pipeline(
                    device,
                    &wgc::pipeline::RenderPipelineDescriptor {
                        layout: desc.layout,
                        vertex_stage: vs_stage.desc,
                        fragment_stage: fs_stage.as_ref().map_or(ptr::null(), |s| &s.desc),
                        primitive_topology: desc.primitive_topology,
                        rasterization_state: desc.rasterization_state.as_ref().map_or(ptr::null(), |rs| rs),
                        color_states: desc.color_states.as_ptr(),
                        color_states_length: desc.color_states.len(),
                        depth_stencil_state: desc.depth_stencil_state.as_ref().map_or(ptr::null(), |ds| ds),
                        vertex_state: wgc::pipeline::VertexStateDescriptor {
                            index_format: desc.vertex_state.index_format,
                            vertex_buffers: vertex_buffers.as_ptr(),
                            vertex_buffers_length: vertex_buffers.len(),
                        },
                        sample_count: desc.sample_count,
                        sample_mask: desc.sample_mask,
                        alpha_to_coverage_enabled: desc.alpha_to_coverage_enabled,
                    },
                    id));
            }
            A::DestroyRenderPipeline(id) => {
                gfx_select!(device => global.render_pipeline_destroy(id));
            }
            A::WriteBuffer { id, data, range } => {
                let bin = std::fs::read(dir.join(data)).unwrap();
                let size = (range.end - range.start) as usize;
                gfx_select!(device => global.device_wait_for_buffer(device, id));
                gfx_select!(device => global.device_set_buffer_sub_data(device, id, range.start, &bin[..size]));
            }
            A::Submit(commands) => {
                let encoder = gfx_select!(device => global.device_create_command_encoder(
                    device,
                    &wgt::CommandEncoderDescriptor {
                        label: ptr::null(),
                    },
                    command_buffer_id_manager.alloc(device.backend())
                ));
                let comb = gfx_select!(device => global.encode_commands(encoder, commands));
                gfx_select!(device => global.queue_submit(device, &[comb]));
            }
        }
    }

    log::info!("Done replay");
    #[cfg(feature = "winit")]
    winit::platform::desktop::EventLoopExtDesktop::run_return(
        &mut event_loop,
        move |event, _, control_flow| {
            use winit::{
                event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
                event_loop::ControlFlow,
            };

            *control_flow = match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    }
                    | WindowEvent::CloseRequested => ControlFlow::Exit,
                    _ => ControlFlow::Poll,
                },
                _ => ControlFlow::Poll,
            }
        },
    );
}
