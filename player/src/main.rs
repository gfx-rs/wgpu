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
    collections::HashMap,
    ffi::CString,
    fmt::Debug,
    fs,
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

struct SwapChainData {
    their_view: Option<wgc::id::TextureViewId>,
    our_view: wgc::id::TextureViewId,
    bind_group: wgc::id::BindGroupId,
}

trait GlobalExt {
    fn encode_commands<B: wgc::hub::GfxBackend>(
        &self,
        encoder: wgc::id::CommandEncoderId,
        commands: Vec<trace::Command>,
    ) -> wgc::id::CommandBufferId;
    fn process<B: wgc::hub::GfxBackend>(
        &self,
        device: wgc::id::DeviceId,
        action: trace::Action,
        dir: &PathBuf,
        comb_manager: &mut wgc::hub::IdentityManager,
        present_bgl: wgc::id::BindGroupLayoutId,
        swap_chain_map: &mut HashMap<wgc::id::SwapChainId, SwapChainData>,
    );
    #[cfg(feature = "winit")]
    fn present<B: wgc::hub::GfxBackend>(
        &self,
        device: wgc::id::DeviceId,
        swap_chain: wgc::id::SwapChainId,
        data: &SwapChainData,
        comb_manager: &mut wgc::hub::IdentityManager,
    ) -> bool;
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
                } => self
                    .command_encoder_copy_buffer_to_buffer::<B>(
                        encoder, src, src_offset, dst, dst_offset, size,
                    )
                    .unwrap(),
                trace::Command::CopyBufferToTexture { src, dst, size } => self
                    .command_encoder_copy_buffer_to_texture::<B>(encoder, &src, &dst, &size)
                    .unwrap(),
                trace::Command::CopyTextureToBuffer { src, dst, size } => self
                    .command_encoder_copy_texture_to_buffer::<B>(encoder, &src, &dst, &size)
                    .unwrap(),
                trace::Command::CopyTextureToTexture { src, dst, size } => self
                    .command_encoder_copy_texture_to_texture::<B>(encoder, &src, &dst, &size)
                    .unwrap(),
                trace::Command::RunComputePass { base } => {
                    self.command_encoder_run_compute_pass_impl::<B>(encoder, base.as_ref());
                }
                trace::Command::RunRenderPass {
                    base,
                    target_colors,
                    target_depth_stencil,
                } => {
                    self.command_encoder_run_render_pass_impl::<B>(
                        encoder,
                        base.as_ref(),
                        &target_colors,
                        target_depth_stencil.as_ref(),
                    );
                }
            }
        }
        self.command_encoder_finish::<B>(encoder, &wgt::CommandBufferDescriptor { todo: 0 })
    }

    fn process<B: wgc::hub::GfxBackend>(
        &self,
        device: wgc::id::DeviceId,
        action: trace::Action,
        dir: &PathBuf,
        comb_manager: &mut wgc::hub::IdentityManager,
        present_bgl: wgc::id::BindGroupLayoutId,
        swap_chain_map: &mut HashMap<wgc::id::SwapChainId, SwapChainData>,
    ) {
        use wgc::device::trace::Action as A;
        match action {
            A::Init { .. } => panic!("Unexpected Action::Init: has to be the first action only"),
            A::CreateSwapChain { id, desc } => {
                let texture = self.device_create_texture::<B>(
                    device,
                    &wgt::TextureDescriptor {
                        label: Some("SwapChain"),
                        size: wgt::Extent3d {
                            width: desc.width,
                            height: desc.height,
                            depth: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgt::TextureDimension::D2,
                        format: desc.format,
                        usage: desc.usage,
                    },
                );
                let view = self.texture_create_view::<B>(texture, None);
                let bind_group = self
                    .device_create_bind_group::<B>(
                        device,
                        &wgc::binding_model::BindGroupDescriptor {
                            label: Some("SwapChain"),
                            layout: present_bgl,
                            entries: &[wgc::binding_model::BindGroupEntry {
                                binding: 0,
                                resource: wgc::binding_model::BindingResource::TextureView(view),
                            }],
                        },
                    )
                    .unwrap();
                swap_chain_map.insert(
                    id,
                    SwapChainData {
                        their_view: None,
                        our_view: view,
                        bind_group,
                    },
                );
            }
            A::PresentSwapChain(_) => unreachable!(),
            A::CreateBuffer { id, desc } => {
                let label = Label::new(&desc.label);
                self.device_maintain_ids::<B>(device);
                self.device_create_buffer::<B>(device, &desc.map_label(|_| label.as_ptr()), id);
            }
            A::DestroyBuffer(id) => {
                self.buffer_destroy::<B>(id);
            }
            A::CreateTexture { id, desc } => {
                let label = Label::new(&desc.label);
                self.device_maintain_ids::<B>(device);
                self.device_create_texture::<B>(device, &desc.map_label(|_| label.as_ptr()), id);
            }
            A::DestroyTexture(id) => {
                self.texture_destroy::<B>(id);
            }
            A::CreateTextureView {
                id,
                parent_id,
                desc,
            } => {
                let label = desc.as_ref().map_or(Label(None), |d| Label::new(&d.label));
                self.device_maintain_ids::<B>(device);
                self.texture_create_view::<B>(
                    parent_id,
                    desc.map(|d| d.map_label(|_| label.as_ptr())).as_ref(),
                    id,
                );
            }
            A::DestroyTextureView(id) => {
                self.texture_view_destroy::<B>(id);
            }
            A::CreateSampler { id, desc } => {
                let label = Label::new(&desc.label);
                self.device_maintain_ids::<B>(device);
                self.device_create_sampler::<B>(device, &desc.map_label(|_| label.as_ptr()), id);
            }
            A::DestroySampler(id) => {
                self.sampler_destroy::<B>(id);
            }
            A::GetSwapChainTexture { id, parent_id } => {
                swap_chain_map.get_mut(&parent_id).unwrap().their_view = id;
            }
            A::CreateBindGroupLayout {
                id,
                ref label,
                ref entries,
            } => {
                self.device_create_bind_group_layout::<B>(
                    device,
                    &wgt::BindGroupLayoutDescriptor {
                        label: Some(label),
                        entries,
                    },
                    id,
                )
                .unwrap();
            }
            A::DestroyBindGroupLayout(id) => {
                self.bind_group_layout_destroy::<B>(id);
            }
            A::CreatePipelineLayout {
                id,
                bind_group_layouts,
            } => {
                self.device_maintain_ids::<B>(device);
                self.device_create_pipeline_layout::<B>(
                    device,
                    &wgc::binding_model::PipelineLayoutDescriptor {
                        bind_group_layouts: bind_group_layouts.as_ptr(),
                        bind_group_layouts_length: bind_group_layouts.len(),
                    },
                    id,
                )
                .unwrap();
            }
            A::DestroyPipelineLayout(id) => {
                self.pipeline_layout_destroy::<B>(id);
            }
            A::CreateBindGroup {
                id,
                label,
                layout_id,
                entries,
            } => {
                use wgc::binding_model as bm;
                let entry_vec = entries
                    .iter()
                    .map(|(binding, res)| wgc::binding_model::BindGroupEntry {
                        binding: *binding,
                        resource: match *res {
                            trace::BindingResource::Buffer { id, offset, size } => {
                                bm::BindingResource::Buffer(bm::BufferBinding {
                                    buffer_id: id,
                                    offset,
                                    size,
                                })
                            }
                            trace::BindingResource::Sampler(id) => bm::BindingResource::Sampler(id),
                            trace::BindingResource::TextureView(id) => {
                                bm::BindingResource::TextureView(id)
                            }
                            trace::BindingResource::TextureViewArray(ref binding_array) => {
                                bm::BindingResource::TextureViewArray(binding_array)
                            }
                        },
                    })
                    .collect::<Vec<_>>();
                self.device_maintain_ids::<B>(device);
                self.device_create_bind_group::<B>(
                    device,
                    &wgc::binding_model::BindGroupDescriptor {
                        label: Some(&label),
                        layout: layout_id,
                        entries: &entry_vec,
                    },
                    id,
                )
                .unwrap();
            }
            A::DestroyBindGroup(id) => {
                self.bind_group_destroy::<B>(id);
            }
            A::CreateShaderModule { id, data } => {
                let byte_vec = fs::read(dir.join(data)).unwrap();
                let spv = byte_vec
                    .chunks(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<_>>();
                self.device_create_shader_module::<B>(
                    device,
                    wgc::pipeline::ShaderModuleSource::SpirV(&spv),
                    id,
                );
            }
            A::DestroyShaderModule(id) => {
                self.shader_module_destroy::<B>(id);
            }
            A::CreateComputePipeline { id, desc } => {
                let compute_stage = desc.compute_stage.to_core();
                self.device_maintain_ids::<B>(device);
                self.device_create_compute_pipeline::<B>(
                    device,
                    &wgc::pipeline::ComputePipelineDescriptor {
                        layout: desc.layout,
                        compute_stage,
                    },
                    id,
                )
                .unwrap();
            }
            A::DestroyComputePipeline(id) => {
                self.compute_pipeline_destroy::<B>(id);
            }
            A::CreateRenderPipeline { id, desc } => {
                let vertex_stage = desc.vertex_stage.to_core();
                let fragment_stage = desc.fragment_stage.as_ref().map(|fs| fs.to_core());
                let vertex_buffers = desc
                    .vertex_state
                    .vertex_buffers
                    .iter()
                    .map(|vb| wgt::VertexBufferDescriptor {
                        stride: vb.stride,
                        step_mode: vb.step_mode,
                        attributes: &vb.attributes,
                    })
                    .collect::<Vec<_>>();
                self.device_maintain_ids::<B>(device);
                self.device_create_render_pipeline::<B>(
                    device,
                    &wgc::pipeline::RenderPipelineDescriptor {
                        layout: desc.layout,
                        vertex_stage,
                        fragment_stage,
                        primitive_topology: desc.primitive_topology,
                        rasterization_state: desc.rasterization_state,
                        color_states: &desc.color_states,
                        depth_stencil_state: desc.depth_stencil_state,
                        vertex_state: wgt::VertexStateDescriptor {
                            index_format: desc.vertex_state.index_format,
                            vertex_buffers: &vertex_buffers,
                        },
                        sample_count: desc.sample_count,
                        sample_mask: desc.sample_mask,
                        alpha_to_coverage_enabled: desc.alpha_to_coverage_enabled,
                    },
                    id,
                )
                .unwrap();
            }
            A::DestroyRenderPipeline(id) => {
                self.render_pipeline_destroy::<B>(id);
            }
            A::CreateRenderBundle { id, desc, base } => {
                let label = Label::new(&desc.label);
                let bundle = wgc::command::RenderBundleEncoder::new(
                    &wgt::RenderBundleEncoderDescriptor {
                        label: None,
                        color_formats: &desc.color_formats,
                        depth_stencil_format: desc.depth_stencil_format,
                        sample_count: desc.sample_count,
                    },
                    device,
                    Some(base),
                );
                self.render_bundle_encoder_finish::<B>(
                    bundle,
                    &wgt::RenderBundleDescriptor {
                        label: label.as_ptr(),
                    },
                    id,
                );
            }
            A::DestroyRenderBundle(id) => {
                self.render_bundle_destroy::<B>(id);
            }
            A::WriteBuffer {
                id,
                data,
                range,
                queued,
            } => {
                let bin = std::fs::read(dir.join(data)).unwrap();
                let size = (range.end - range.start) as usize;
                if queued {
                    self.queue_write_buffer::<B>(device, id, range.start, &bin);
                } else {
                    self.device_wait_for_buffer::<B>(device, id);
                    self.device_set_buffer_sub_data::<B>(device, id, range.start, &bin[..size]);
                }
            }
            A::WriteTexture {
                to,
                data,
                layout,
                size,
            } => {
                let bin = std::fs::read(dir.join(data)).unwrap();
                self.queue_write_texture::<B>(device, &to, &bin, &layout, &size);
            }
            A::Submit(_index, commands) => {
                let encoder = self.device_create_command_encoder::<B>(
                    device,
                    &wgt::CommandEncoderDescriptor { label: ptr::null() },
                    comb_manager.alloc(device.backend()),
                );
                let comb = self.encode_commands::<B>(encoder, commands);
                self.queue_submit::<B>(device, &[comb]);
            }
        }
    }

    #[cfg(feature = "winit")]
    fn present<B: wgc::hub::GfxBackend>(
        &self,
        device: wgc::id::DeviceId,
        swap_chain: wgc::id::SwapChainId,
        data: &SwapChainData,
        comb_manager: &mut wgc::hub::IdentityManager,
    ) -> bool {
        let frame = match data.their_view {
            Some(id) => self.swap_chain_get_next_texture::<B>(swap_chain, id),
            None => return false,
        };
        let attachment = match frame.view_id {
            Some(id) => id,
            None => return false,
        };
        let command_encoder = self.device_create_command_encoder::<B>(
            device,
            &wgt::CommandEncoderDescriptor { label: ptr::null() },
            comb_manager.alloc(device.backend()),
        );
        let render_pass = wgc::command::RenderPass::new(
            command_encoder,
            wgc::command::RenderPassDescriptor {
                color_attachments: &[wgc::command::ColorAttachmentDescriptor {
                    attachment,
                    resolve_target: None,
                    channel: wgt::PassChannel {
                        load_op: wgt::LoadOp::Clear,
                        clear_value: wgt::Color::BLACK,
                        store_op: wgt::StoreOp::Store,
                        read_only: false,
                    },
                }],
                depth_stencil_attachment: None,
            },
        );
        self.command_encoder_run_render_pass::<B>(command_encoder, &render_pass);
        let command_buffer = self.command_encoder_finish::<B>(
            command_encoder,
            &wgt::CommandBufferDescriptor { todo: 0 },
        );
        self.queue_submit::<B>(device, &[command_buffer]);
        self.swap_chain_present::<B>(swap_chain);

        true
    }
}

fn main() {
    #[cfg(feature = "winit")]
    use winit::{event_loop::EventLoop, window::WindowBuilder};

    env_logger::init();

    #[cfg(feature = "renderdoc")]
    #[cfg_attr(feature = "winit", allow(unused))]
    let mut rd = renderdoc::RenderDoc::<renderdoc::V110>::new()
        .expect("Failed to connect to RenderDoc: are you running without it?");

    //TODO: setting for the backend bits
    //TODO: setting for the target frame, or controls

    let dir = match std::env::args().nth(1) {
        Some(arg) if Path::new(&arg).is_dir() => PathBuf::from(arg),
        _ => panic!("Provide the dir path as the parameter"),
    };

    log::info!("Loading trace '{:?}'", dir);
    let file = fs::File::open(dir.join(trace::FILE_NAME)).unwrap();
    let mut actions: Vec<trace::Action> = ron::de::from_reader(file).unwrap();
    actions.reverse(); // allows us to pop from the top
    log::info!("Found {} actions", actions.len());

    #[cfg(feature = "winit")]
    let event_loop = {
        log::info!("Creating a window");
        EventLoop::new()
    };
    #[cfg(feature = "winit")]
    let window = WindowBuilder::new()
        .with_title("wgpu player")
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

    let global =
        wgc::hub::Global::new("player", IdentityPassThroughFactory, wgt::BackendBit::all());
    let mut command_buffer_id_manager = wgc::hub::IdentityManager::default();
    let mut swap_chain_map = HashMap::<wgc::id::SwapChainId, SwapChainData>::new();

    #[cfg(feature = "winit")]
    let surface =
        global.instance_create_surface(&window, wgc::id::TypedId::zip(0, 1, wgt::Backend::Empty));

    let device = match actions.pop() {
        Some(trace::Action::Init { desc, backend }) => {
            log::info!("Initializing the device for backend: {:?}", backend);
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
                        &[wgc::id::TypedId::zip(0, 0, backend)],
                        |id| id.backend(),
                    ),
                )
                .expect("Unable to find an adapter for selected backend");

            let info = gfx_select!(adapter => global.adapter_get_info(adapter));
            log::info!("Picked '{}'", info.name);
            gfx_select!(adapter => global.adapter_request_device(
                adapter,
                &desc,
                None,
                wgc::id::TypedId::zip(1, 0, wgt::Backend::Empty)
            ))
            .expect("Failed to request device")
        }
        _ => panic!("Expected Action::Init"),
    };

    let present_bgl = gfx_select!(device => global.device_create_bind_group_layout(device, &wgt::BindGroupLayoutDescriptor {
        label: Some("SwapChain"),
        entries: &[
            wgt::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgt::ShaderStage::FRAGMENT,
                ty: wgt::BindingType::SampledTexture {
                    dimension: wgt::TextureViewDimension::D2,
                    component_type: wgt::TextureComponentType::Float,
                    multisampled: false,
                },
                count: None,
            },
        ],
    }));

    log::info!("Executing actions");
    #[cfg(not(feature = "winit"))]
    {
        #[cfg(feature = "renderdoc")]
        rd.start_frame_capture(ptr::null(), ptr::null());

        while let Some(action) = actions.pop() {
            gfx_select!(device => global.process(device, action, &dir, &mut command_buffer_id_manager, present_bgl, &mut swap_chain_map));
        }

        #[cfg(feature = "renderdoc")]
        rd.end_frame_capture(ptr::null(), ptr::null());
        gfx_select!(device => global.device_poll(device, true));
    }
    #[cfg(feature = "winit")]
    {
        use winit::{
            event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
            event_loop::ControlFlow,
        };

        let swap_chain =
            gfx_select!(device => global.device_create_swap_chain(device, surface, &desc));

        let mut frame_count = 0;
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;
            match event {
                Event::MainEventsCleared => {
                    window.request_redraw();
                }
                Event::RedrawRequested(_) => loop {
                    match actions.pop() {
                        Some(trace::Action::CreateSwapChain { id, desc }) => {
                            log::info!("Initializing the swapchain");
                            assert_eq!(id.to_surface_id(), surface);
                            let texture = gfx_select!(device => global.device_create_texture(device, &wgt::TextureDescriptor {
                                label: Some("SwapChain"),
                                size: wgt::Extent3d { width: desc.width, height: desc.height, depth: 1 },
                                mip_level_count: 1,
                                sample_count: 1,
                                dimension: wgt::TextureDimension::D2,
                                format: desc.format,
                                usage: desc.usage,
                            }));
                            let view = gfx_select!(device => global.texture_create_view(texture, None));
                            let bind_group = gfx_select!(device => global.device_create_bind_group(device, &wgt::BindGroupDescriptor {
                                label: Some("SwapChain"),
                                entries: &[
                                    wgt::BindGroupEntry {
                                        binding: 0,
                                        resource: wgt::BindingResource::TextureView(view),
                                    },
                                ],
                            }));
                            swap_chain_map.insert(id, SwapChainData {
                                their_view: None,
                                our_view: view,
                                bind_group,
                            });
                        }
                        Some(trace::Action::PresentSwapChain(id)) => {
                            frame_count += 1;
                            log::debug!("Presenting frame {}", frame_count);
                            if !gfx_select!(device => global.present(device, swap_chain, &swap_chain_map[id], &mut command_buffer_id_manager)) {
                                log::warn!("\tfailed");
                            }
                            break;
                        }
                        Some(action) => {
                            gfx_select!(device => global.process(device, action, &dir, &mut command_buffer_id_manager, present_bgl, &mut swap_chain_map));
                        }
                        None => break,
                    }
                },
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
                    | WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => {}
                },
                Event::LoopDestroyed => {
                    log::info!("Closing");
                    gfx_select!(device => global.device_poll(device, true));
                }
                _ => {}
            }
        });
    }
}
