use std::future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use wgpu::custom::*;
use wgpu_native::{native, *};

use crate::command::{CCommandEncoder, CRenderBundleEncoder};
use crate::conv;
use crate::resource::*;

// ── CDevice ───────────────────────────────────────────────────────────────────

pub(crate) type ErrorHandler = Arc<Mutex<Option<Arc<dyn wgpu::UncapturedErrorHandler>>>>;

pub struct CDevice {
    pub(crate) ptr: native::WGPUDevice,
    pub(crate) info: wgpu::AdapterInfo,
    pub(crate) error_handler: ErrorHandler,
}

impl std::fmt::Debug for CDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CDevice").field("ptr", &self.ptr).finish()
    }
}

unsafe impl Send for CDevice {}
unsafe impl Sync for CDevice {}

impl Drop for CDevice {
    fn drop(&mut self) {
        unsafe { wgpuDeviceRelease(self.ptr) };
    }
}

impl DeviceInterface for CDevice {
    fn features(&self) -> wgpu::Features {
        let mut supported: native::WGPUSupportedFeatures = unsafe { std::mem::zeroed() };
        unsafe { wgpuDeviceGetFeatures(self.ptr, Some(&mut supported)) };
        let result = conv::map_supported_features(&supported);
        unsafe { wgpuSupportedFeaturesFreeMembers(supported) };
        result
    }

    fn limits(&self) -> wgpu::Limits {
        let mut limits: native::WGPULimits = unsafe { std::mem::zeroed() };
        unsafe { wgpuDeviceGetLimits(self.ptr, Some(&mut limits)) };
        conv::map_limits(&limits)
    }

    fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.info.clone()
    }

    fn create_shader_module(
        &self,
        desc: wgpu::ShaderModuleDescriptor<'_>,
        _shader_bound_checks: wgpu::ShaderRuntimeChecks,
    ) -> DispatchShaderModule {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());

        match &desc.source {
            #[cfg(feature = "wgsl")]
            wgpu::ShaderSource::Wgsl(code) => {
                let code_sv = conv::str_to_string_view(code.as_ref());
                let mut wgsl_chain = native::WGPUShaderSourceWGSL {
                    chain: native::WGPUChainedStruct {
                        next: std::ptr::null_mut(),
                        sType: native::WGPUSType_ShaderSourceWGSL,
                    },
                    code: code_sv,
                };
                let c_desc = native::WGPUShaderModuleDescriptor {
                    nextInChain: std::ptr::from_mut::<native::WGPUChainedStruct>(
                        &mut wgsl_chain.chain,
                    ),
                    label: label_sv,
                };
                let ptr = unsafe { wgpuDeviceCreateShaderModule(self.ptr, Some(&c_desc)) };
                DispatchShaderModule::custom(CShaderModule { ptr })
            }
            #[cfg(feature = "spirv")]
            wgpu::ShaderSource::SpirV(words) => {
                let c_desc = native::WGPUShaderModuleDescriptorSpirV {
                    label: label_sv,
                    sourceSize: words.len() as u32,
                    source: words.as_ptr(),
                };
                let ptr = unsafe { wgpuDeviceCreateShaderModuleSpirV(self.ptr, Some(&c_desc)) };
                DispatchShaderModule::custom(CShaderModule { ptr })
            }
            _ => unimplemented!("wgpu-native does not support this shader source type"),
        }
    }

    unsafe fn create_shader_module_passthrough(
        &self,
        desc: &wgpu::ShaderModuleDescriptorPassthrough<'_>,
    ) -> DispatchShaderModule {
        // Try WGSL first, then SpirV.
        let label_sv = conv::opt_str_to_string_view(desc.label);
        if let Some(wgsl) = &desc.wgsl {
            let code_sv = conv::str_to_string_view(wgsl.as_ref());
            let mut wgsl_chain = native::WGPUShaderSourceWGSL {
                chain: native::WGPUChainedStruct {
                    next: std::ptr::null_mut(),
                    sType: native::WGPUSType_ShaderSourceWGSL,
                },
                code: code_sv,
            };
            let c_desc = native::WGPUShaderModuleDescriptor {
                nextInChain: std::ptr::from_mut::<native::WGPUChainedStruct>(&mut wgsl_chain.chain),
                label: label_sv,
            };
            let ptr = unsafe { wgpuDeviceCreateShaderModule(self.ptr, Some(&c_desc)) };
            return DispatchShaderModule::custom(CShaderModule { ptr });
        }
        if let Some(spirv) = &desc.spirv {
            let c_desc = native::WGPUShaderModuleDescriptorSpirV {
                label: label_sv,
                sourceSize: spirv.len() as u32,
                source: spirv.as_ptr(),
            };
            let ptr = unsafe { wgpuDeviceCreateShaderModuleSpirV(self.ptr, Some(&c_desc)) };
            return DispatchShaderModule::custom(CShaderModule { ptr });
        }
        unimplemented!("wgpu-native: no supported shader format in passthrough descriptor")
    }

    fn create_bind_group_layout(
        &self,
        desc: &wgpu::BindGroupLayoutDescriptor<'_>,
    ) -> DispatchBindGroupLayout {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());

        let entries: Vec<native::WGPUBindGroupLayoutEntry> = desc
            .entries
            .iter()
            .map(|e| {
                let mut entry: native::WGPUBindGroupLayoutEntry = unsafe { std::mem::zeroed() };
                entry.binding = e.binding;
                entry.visibility = conv::shader_stages_to_native(e.visibility);
                entry.bindingArraySize = e.count.map(|n| n.get()).unwrap_or(0);
                match e.ty {
                    wgpu::BindingType::Buffer {
                        ty,
                        has_dynamic_offset,
                        min_binding_size,
                    } => {
                        entry.buffer = native::WGPUBufferBindingLayout {
                            nextInChain: std::ptr::null_mut(),
                            type_: conv::buffer_binding_type_to_native(ty),
                            hasDynamicOffset: has_dynamic_offset as u32,
                            minBindingSize: min_binding_size.map(|s| s.get()).unwrap_or(0),
                        };
                    }
                    wgpu::BindingType::Sampler(ty) => {
                        entry.sampler = native::WGPUSamplerBindingLayout {
                            nextInChain: std::ptr::null_mut(),
                            type_: conv::sampler_binding_type_to_native(ty),
                        };
                    }
                    wgpu::BindingType::Texture {
                        sample_type,
                        view_dimension,
                        multisampled,
                    } => {
                        entry.texture = native::WGPUTextureBindingLayout {
                            nextInChain: std::ptr::null_mut(),
                            sampleType: conv::texture_sample_type_to_native(sample_type),
                            viewDimension: conv::texture_view_dimension_to_native(view_dimension),
                            multisampled: multisampled as u32,
                        };
                    }
                    wgpu::BindingType::StorageTexture {
                        access,
                        format,
                        view_dimension,
                    } => {
                        entry.storageTexture = native::WGPUStorageTextureBindingLayout {
                            nextInChain: std::ptr::null_mut(),
                            access: conv::storage_texture_access_to_native(access),
                            format: conv::texture_format_to_native(format),
                            viewDimension: conv::texture_view_dimension_to_native(view_dimension),
                        };
                    }
                    // AccelerationStructure and ExternalTexture not supported by wgpu-native.
                    _ => unimplemented!("wgpu-native does not support this binding type"),
                }
                entry
            })
            .collect();

        let c_desc = native::WGPUBindGroupLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: label_sv,
            entryCount: entries.len(),
            entries: if entries.is_empty() {
                std::ptr::null()
            } else {
                entries.as_ptr()
            },
        };
        let ptr = unsafe { wgpuDeviceCreateBindGroupLayout(self.ptr, Some(&c_desc)) };
        DispatchBindGroupLayout::custom(CBindGroupLayout { ptr })
    }

    fn create_bind_group(&self, desc: &wgpu::BindGroupDescriptor<'_>) -> DispatchBindGroup {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());
        let layout_ptr = desc.layout.as_custom::<CBindGroupLayout>().unwrap().ptr;

        let entries: Vec<native::WGPUBindGroupEntry> = desc
            .entries
            .iter()
            .map(|e| {
                let mut entry: native::WGPUBindGroupEntry = unsafe { std::mem::zeroed() };
                entry.binding = e.binding;
                entry.size = u64::MAX;
                match &e.resource {
                    wgpu::BindingResource::Buffer(bb) => {
                        entry.buffer = bb.buffer.as_custom::<CBuffer>().unwrap().ptr;
                        entry.offset = bb.offset;
                        entry.size = bb.size.map(|s| s.get()).unwrap_or(u64::MAX);
                    }
                    wgpu::BindingResource::Sampler(s) => {
                        entry.sampler = s.as_custom::<CSampler>().unwrap().ptr;
                    }
                    wgpu::BindingResource::TextureView(tv) => {
                        entry.textureView = tv.as_custom::<CTextureView>().unwrap().ptr;
                    }
                    // BufferArray/SamplerArray/TextureViewArray/AccelerationStructure/ExternalTexture
                    // are not supported by the standard wgpu-native bind group API.
                    _ => unimplemented!("wgpu-native does not support this binding resource type"),
                }
                entry
            })
            .collect();

        let c_desc = native::WGPUBindGroupDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: label_sv,
            layout: layout_ptr,
            entryCount: entries.len(),
            entries: if entries.is_empty() {
                std::ptr::null()
            } else {
                entries.as_ptr()
            },
        };
        let ptr = unsafe { wgpuDeviceCreateBindGroup(self.ptr, Some(&c_desc)) };
        DispatchBindGroup::custom(CBindGroup { ptr })
    }

    fn create_pipeline_layout(
        &self,
        desc: &wgpu::PipelineLayoutDescriptor<'_>,
    ) -> DispatchPipelineLayout {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());
        let layouts: Vec<native::WGPUBindGroupLayout> = desc
            .bind_group_layouts
            .iter()
            .map(|bgl| {
                bgl.as_ref()
                    .map(|l| l.as_custom::<CBindGroupLayout>().unwrap().ptr)
                    .unwrap_or(std::ptr::null_mut())
            })
            .collect();
        let c_desc = native::WGPUPipelineLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: label_sv,
            bindGroupLayoutCount: layouts.len(),
            bindGroupLayouts: if layouts.is_empty() {
                std::ptr::null()
            } else {
                layouts.as_ptr()
            },
            immediateSize: desc.immediate_size,
        };
        let ptr = unsafe { wgpuDeviceCreatePipelineLayout(self.ptr, Some(&c_desc)) };
        DispatchPipelineLayout::custom(CPipelineLayout { ptr })
    }

    #[allow(unused_assignments)] // else-branch initializes storage vecs required by definite-assignment
    fn create_render_pipeline(
        &self,
        desc: &wgpu::RenderPipelineDescriptor<'_>,
    ) -> DispatchRenderPipeline {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());

        let layout_ptr = desc
            .layout
            .map(|l| l.as_custom::<CPipelineLayout>().unwrap().ptr)
            .unwrap_or(std::ptr::null_mut());

        // Vertex state.
        let v_module = desc.vertex.module.as_custom::<CShaderModule>().unwrap().ptr;
        let v_ep_owned = desc.vertex.entry_point.map(|s| s.to_owned());
        let v_ep_sv = v_ep_owned
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());

        let v_constants: Vec<native::WGPUConstantEntry> = desc
            .vertex
            .compilation_options
            .constants
            .iter()
            .map(|(k, v)| native::WGPUConstantEntry {
                nextInChain: std::ptr::null_mut(),
                key: conv::str_to_string_view(k),
                value: *v,
            })
            .collect();

        // Vertex buffers — WGPUVertexBufferLayout with optional holes.
        let v_attribs_per_buf: Vec<Vec<native::WGPUVertexAttribute>> = desc
            .vertex
            .buffers
            .iter()
            .map(|opt_buf| {
                if let Some(buf) = opt_buf {
                    buf.attributes
                        .iter()
                        .map(|a| native::WGPUVertexAttribute {
                            nextInChain: std::ptr::null_mut(),
                            format: conv::vertex_format_to_native(a.format),
                            offset: a.offset,
                            shaderLocation: a.shader_location,
                        })
                        .collect()
                } else {
                    vec![]
                }
            })
            .collect();

        let v_buffers: Vec<native::WGPUVertexBufferLayout> = desc
            .vertex
            .buffers
            .iter()
            .zip(v_attribs_per_buf.iter())
            .map(|(opt_buf, attribs)| {
                if let Some(buf) = opt_buf {
                    native::WGPUVertexBufferLayout {
                        nextInChain: std::ptr::null_mut(),
                        stepMode: conv::vertex_step_mode_to_native(buf.step_mode),
                        arrayStride: buf.array_stride,
                        attributeCount: attribs.len(),
                        attributes: if attribs.is_empty() {
                            std::ptr::null()
                        } else {
                            attribs.as_ptr()
                        },
                    }
                } else {
                    // Hole: stepMode=Undefined + empty attributes.
                    native::WGPUVertexBufferLayout {
                        nextInChain: std::ptr::null_mut(),
                        stepMode: native::WGPUVertexStepMode_Undefined,
                        arrayStride: 0,
                        attributeCount: 0,
                        attributes: std::ptr::null(),
                    }
                }
            })
            .collect();

        let c_vertex = native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: v_module,
            entryPoint: v_ep_sv,
            constantCount: v_constants.len(),
            constants: if v_constants.is_empty() {
                std::ptr::null()
            } else {
                v_constants.as_ptr()
            },
            bufferCount: v_buffers.len(),
            buffers: if v_buffers.is_empty() {
                std::ptr::null()
            } else {
                v_buffers.as_ptr()
            },
        };

        // Primitive state.
        let prim = &desc.primitive;
        let c_primitive = native::WGPUPrimitiveState {
            nextInChain: std::ptr::null_mut(),
            topology: conv::primitive_topology_to_native(prim.topology),
            stripIndexFormat: prim
                .strip_index_format
                .map(conv::index_format_to_native)
                .unwrap_or(native::WGPUIndexFormat_Undefined),
            frontFace: conv::front_face_to_native(prim.front_face),
            cullMode: conv::cull_mode_to_native(prim.cull_mode),
            unclippedDepth: prim.unclipped_depth as u32,
        };

        // Depth stencil.
        let ds_state: Option<native::WGPUDepthStencilState> =
            desc.depth_stencil
                .as_ref()
                .map(|ds| native::WGPUDepthStencilState {
                    nextInChain: std::ptr::null_mut(),
                    format: conv::texture_format_to_native(ds.format),
                    depthWriteEnabled: ds
                        .depth_write_enabled
                        .map(conv::bool_to_optional_bool)
                        .unwrap_or(native::WGPUOptionalBool_Undefined),
                    depthCompare: ds
                        .depth_compare
                        .map(conv::compare_function_to_native)
                        .unwrap_or(native::WGPUCompareFunction_Undefined),
                    stencilFront: stencil_face_to_native(ds.stencil.front),
                    stencilBack: stencil_face_to_native(ds.stencil.back),
                    stencilReadMask: ds.stencil.read_mask,
                    stencilWriteMask: ds.stencil.write_mask,
                    depthBias: ds.bias.constant,
                    depthBiasSlopeScale: ds.bias.slope_scale,
                    depthBiasClamp: ds.bias.clamp,
                });
        let ds_ptr: *const native::WGPUDepthStencilState = ds_state
            .as_ref()
            .map(std::ptr::from_ref)
            .unwrap_or(std::ptr::null());

        // Multisample.
        let ms = &desc.multisample;
        let c_multisample = native::WGPUMultisampleState {
            nextInChain: std::ptr::null_mut(),
            count: ms.count,
            mask: ms.mask as u32,
            alphaToCoverageEnabled: ms.alpha_to_coverage_enabled as u32,
        };

        // Fragment state — storage kept alive until after wgpuDeviceCreateRenderPipeline.
        // WGPUFragmentState holds raw pointers into these vecs and owned strings.
        let frag_ep_owned: Option<String>;
        let frag_constants: Vec<native::WGPUConstantEntry>;
        let frag_blend_states: Vec<Option<native::WGPUBlendState>>;
        let frag_targets_raw: Vec<native::WGPUColorTargetState>;
        let fragment_state: Option<native::WGPUFragmentState>;

        if let Some(frag) = &desc.fragment {
            let frag_module = frag.module.as_custom::<CShaderModule>().unwrap().ptr;
            frag_ep_owned = frag.entry_point.map(|s| s.to_owned());
            let frag_ep_sv = frag_ep_owned
                .as_deref()
                .map(conv::str_to_string_view)
                .unwrap_or(conv::null_string_view());
            frag_constants = frag
                .compilation_options
                .constants
                .iter()
                .map(|(k, v)| native::WGPUConstantEntry {
                    nextInChain: std::ptr::null_mut(),
                    key: conv::str_to_string_view(k),
                    value: *v,
                })
                .collect();
            frag_blend_states = frag
                .targets
                .iter()
                .map(|opt_t| {
                    opt_t.as_ref().and_then(|t| t.blend.as_ref()).map(|blend| {
                        native::WGPUBlendState {
                            color: blend_component_to_native(blend.color),
                            alpha: blend_component_to_native(blend.alpha),
                        }
                    })
                })
                .collect();
            frag_targets_raw = frag
                .targets
                .iter()
                .zip(frag_blend_states.iter())
                .map(|(opt_t, opt_blend)| {
                    if let Some(t) = opt_t {
                        native::WGPUColorTargetState {
                            nextInChain: std::ptr::null_mut(),
                            format: conv::texture_format_to_native(t.format),
                            blend: opt_blend
                                .as_ref()
                                .map(std::ptr::from_ref)
                                .unwrap_or(std::ptr::null()),
                            writeMask: conv::color_writes_to_native(t.write_mask),
                        }
                    } else {
                        native::WGPUColorTargetState {
                            nextInChain: std::ptr::null_mut(),
                            format: native::WGPUTextureFormat_Undefined,
                            blend: std::ptr::null(),
                            writeMask: native::WGPUColorWriteMask_None,
                        }
                    }
                })
                .collect();
            fragment_state = Some(native::WGPUFragmentState {
                nextInChain: std::ptr::null_mut(),
                module: frag_module,
                entryPoint: frag_ep_sv,
                constantCount: frag_constants.len(),
                constants: if frag_constants.is_empty() {
                    std::ptr::null()
                } else {
                    frag_constants.as_ptr()
                },
                targetCount: frag_targets_raw.len(),
                targets: if frag_targets_raw.is_empty() {
                    std::ptr::null()
                } else {
                    frag_targets_raw.as_ptr()
                },
            });
        } else {
            frag_ep_owned = None;
            frag_constants = vec![];
            frag_blend_states = vec![];
            frag_targets_raw = vec![];
            fragment_state = None;
        }

        let mut cache_extras = desc
            .cache
            .and_then(|c| c.as_custom::<CPipelineCache>())
            .map(|c| native::WGPURenderPipelineDescriptorExtras {
                chain: native::WGPUChainedStruct {
                    next: std::ptr::null_mut(),
                    sType: native::WGPUSType_RenderPipelineDescriptorExtras,
                },
                cache: c.ptr,
                multiviewMask: 0,
                zeroInitializeWorkgroupMemory: false as _,
            });
        let c_desc = native::WGPURenderPipelineDescriptor {
            nextInChain: cache_extras
                .as_mut()
                .map(|e| std::ptr::from_mut::<native::WGPUChainedStruct>(&mut e.chain))
                .unwrap_or(std::ptr::null_mut()),
            label: label_sv,
            layout: layout_ptr,
            vertex: c_vertex,
            primitive: c_primitive,
            depthStencil: ds_ptr,
            multisample: c_multisample,
            fragment: fragment_state
                .as_ref()
                .map(std::ptr::from_ref)
                .unwrap_or(std::ptr::null()),
        };

        let ptr = unsafe { wgpuDeviceCreateRenderPipeline(self.ptr, Some(&c_desc)) };
        DispatchRenderPipeline::custom(CRenderPipeline { ptr })
    }

    #[allow(unused_assignments)]
    fn create_mesh_pipeline(
        &self,
        desc: &wgpu::MeshPipelineDescriptor<'_>,
    ) -> DispatchRenderPipeline {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());

        let layout_ptr = desc
            .layout
            .map(|l| l.as_custom::<CPipelineLayout>().unwrap().ptr)
            .unwrap_or(std::ptr::null_mut());

        // Task stage (optional).
        let task_ep_owned: Option<String>;
        let task_constants: Vec<native::WGPUConstantEntry>;
        let task_state: Option<native::WGPUTaskState>;

        if let Some(task) = &desc.task {
            let task_module = task.module.as_custom::<CShaderModule>().unwrap().ptr;
            task_ep_owned = task.entry_point.map(|s| s.to_owned());
            let task_ep_sv = task_ep_owned
                .as_deref()
                .map(conv::str_to_string_view)
                .unwrap_or(conv::null_string_view());
            task_constants = task
                .compilation_options
                .constants
                .iter()
                .map(|(k, v)| native::WGPUConstantEntry {
                    nextInChain: std::ptr::null_mut(),
                    key: conv::str_to_string_view(k),
                    value: *v,
                })
                .collect();
            task_state = Some(native::WGPUTaskState {
                nextInChain: std::ptr::null_mut(),
                module: task_module,
                entryPoint: task_ep_sv,
                constantCount: task_constants.len(),
                constants: if task_constants.is_empty() {
                    std::ptr::null()
                } else {
                    task_constants.as_ptr()
                },
            });
        } else {
            task_ep_owned = None;
            task_constants = vec![];
            task_state = None;
        }

        // Mesh stage (required).
        let mesh_module = desc.mesh.module.as_custom::<CShaderModule>().unwrap().ptr;
        let mesh_ep_owned = desc.mesh.entry_point.map(|s| s.to_owned());
        let mesh_ep_sv = mesh_ep_owned
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());
        let mesh_constants: Vec<native::WGPUConstantEntry> = desc
            .mesh
            .compilation_options
            .constants
            .iter()
            .map(|(k, v)| native::WGPUConstantEntry {
                nextInChain: std::ptr::null_mut(),
                key: conv::str_to_string_view(k),
                value: *v,
            })
            .collect();
        let c_mesh = native::WGPUMeshState {
            nextInChain: std::ptr::null_mut(),
            module: mesh_module,
            entryPoint: mesh_ep_sv,
            constantCount: mesh_constants.len(),
            constants: if mesh_constants.is_empty() {
                std::ptr::null()
            } else {
                mesh_constants.as_ptr()
            },
        };

        // Primitive state.
        let prim = &desc.primitive;
        let c_primitive = native::WGPUPrimitiveState {
            nextInChain: std::ptr::null_mut(),
            topology: conv::primitive_topology_to_native(prim.topology),
            stripIndexFormat: prim
                .strip_index_format
                .map(conv::index_format_to_native)
                .unwrap_or(native::WGPUIndexFormat_Undefined),
            frontFace: conv::front_face_to_native(prim.front_face),
            cullMode: conv::cull_mode_to_native(prim.cull_mode),
            unclippedDepth: prim.unclipped_depth as u32,
        };

        // Depth stencil.
        let ds_state: Option<native::WGPUDepthStencilState> =
            desc.depth_stencil
                .as_ref()
                .map(|ds| native::WGPUDepthStencilState {
                    nextInChain: std::ptr::null_mut(),
                    format: conv::texture_format_to_native(ds.format),
                    depthWriteEnabled: ds
                        .depth_write_enabled
                        .map(conv::bool_to_optional_bool)
                        .unwrap_or(native::WGPUOptionalBool_Undefined),
                    depthCompare: ds
                        .depth_compare
                        .map(conv::compare_function_to_native)
                        .unwrap_or(native::WGPUCompareFunction_Undefined),
                    stencilFront: stencil_face_to_native(ds.stencil.front),
                    stencilBack: stencil_face_to_native(ds.stencil.back),
                    stencilReadMask: ds.stencil.read_mask,
                    stencilWriteMask: ds.stencil.write_mask,
                    depthBias: ds.bias.constant,
                    depthBiasSlopeScale: ds.bias.slope_scale,
                    depthBiasClamp: ds.bias.clamp,
                });
        let ds_ptr: *const native::WGPUDepthStencilState = ds_state
            .as_ref()
            .map(std::ptr::from_ref)
            .unwrap_or(std::ptr::null());

        // Multisample.
        let ms = &desc.multisample;
        let c_multisample = native::WGPUMultisampleState {
            nextInChain: std::ptr::null_mut(),
            count: ms.count,
            mask: ms.mask as u32,
            alphaToCoverageEnabled: ms.alpha_to_coverage_enabled as u32,
        };

        // Fragment state.
        #[allow(unused_assignments)]
        let frag_ep_owned: Option<String>;
        let frag_constants: Vec<native::WGPUConstantEntry>;
        let frag_blend_states: Vec<Option<native::WGPUBlendState>>;
        let frag_targets_raw: Vec<native::WGPUColorTargetState>;
        let fragment_state: Option<native::WGPUFragmentState>;

        if let Some(frag) = &desc.fragment {
            let frag_module = frag.module.as_custom::<CShaderModule>().unwrap().ptr;
            frag_ep_owned = frag.entry_point.map(|s| s.to_owned());
            let frag_ep_sv = frag_ep_owned
                .as_deref()
                .map(conv::str_to_string_view)
                .unwrap_or(conv::null_string_view());
            frag_constants = frag
                .compilation_options
                .constants
                .iter()
                .map(|(k, v)| native::WGPUConstantEntry {
                    nextInChain: std::ptr::null_mut(),
                    key: conv::str_to_string_view(k),
                    value: *v,
                })
                .collect();
            frag_blend_states = frag
                .targets
                .iter()
                .map(|opt_t| {
                    opt_t.as_ref().and_then(|t| t.blend.as_ref()).map(|blend| {
                        native::WGPUBlendState {
                            color: blend_component_to_native(blend.color),
                            alpha: blend_component_to_native(blend.alpha),
                        }
                    })
                })
                .collect();
            frag_targets_raw = frag
                .targets
                .iter()
                .zip(frag_blend_states.iter())
                .map(|(opt_t, opt_blend)| {
                    if let Some(t) = opt_t {
                        native::WGPUColorTargetState {
                            nextInChain: std::ptr::null_mut(),
                            format: conv::texture_format_to_native(t.format),
                            blend: opt_blend
                                .as_ref()
                                .map(std::ptr::from_ref)
                                .unwrap_or(std::ptr::null()),
                            writeMask: conv::color_writes_to_native(t.write_mask),
                        }
                    } else {
                        native::WGPUColorTargetState {
                            nextInChain: std::ptr::null_mut(),
                            format: native::WGPUTextureFormat_Undefined,
                            blend: std::ptr::null(),
                            writeMask: native::WGPUColorWriteMask_None,
                        }
                    }
                })
                .collect();
            fragment_state = Some(native::WGPUFragmentState {
                nextInChain: std::ptr::null_mut(),
                module: frag_module,
                entryPoint: frag_ep_sv,
                constantCount: frag_constants.len(),
                constants: if frag_constants.is_empty() {
                    std::ptr::null()
                } else {
                    frag_constants.as_ptr()
                },
                targetCount: frag_targets_raw.len(),
                targets: if frag_targets_raw.is_empty() {
                    std::ptr::null()
                } else {
                    frag_targets_raw.as_ptr()
                },
            });
        } else {
            frag_ep_owned = None;
            frag_constants = vec![];
            frag_blend_states = vec![];
            frag_targets_raw = vec![];
            fragment_state = None;
        }

        // Pipeline cache extras (optional).
        let mut cache_extras = desc
            .cache
            .and_then(|c| c.as_custom::<CPipelineCache>())
            .map(|c| native::WGPUMeshPipelineDescriptorExtras {
                chain: native::WGPUChainedStruct {
                    next: std::ptr::null_mut(),
                    sType: native::WGPUSType_MeshPipelineDescriptorExtras,
                },
                cache: c.ptr,
                multiviewMask: 0,
                zeroInitializeWorkgroupMemory: false as _,
            });

        let c_desc = native::WGPUMeshPipelineDescriptor {
            nextInChain: cache_extras
                .as_mut()
                .map(|e| std::ptr::from_mut::<native::WGPUChainedStruct>(&mut e.chain))
                .unwrap_or(std::ptr::null_mut()),
            label: label_sv,
            layout: layout_ptr,
            task: task_state
                .as_ref()
                .map(std::ptr::from_ref)
                .unwrap_or(std::ptr::null()),
            mesh: c_mesh,
            primitive: c_primitive,
            depthStencil: ds_ptr,
            multisample: c_multisample,
            fragment: fragment_state
                .as_ref()
                .map(std::ptr::from_ref)
                .unwrap_or(std::ptr::null()),
        };

        let ptr = unsafe { wgpuDeviceCreateMeshPipeline(self.ptr, Some(&c_desc)) };
        DispatchRenderPipeline::custom(CRenderPipeline { ptr })
    }

    fn create_compute_pipeline(
        &self,
        desc: &wgpu::ComputePipelineDescriptor<'_>,
    ) -> DispatchComputePipeline {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());
        let layout_ptr = desc
            .layout
            .map(|l| l.as_custom::<CPipelineLayout>().unwrap().ptr)
            .unwrap_or(std::ptr::null_mut());
        let module_ptr = desc.module.as_custom::<CShaderModule>().unwrap().ptr;
        let ep_owned = desc.entry_point.map(|s| s.to_owned());
        let ep_sv = ep_owned
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());
        let constants: Vec<native::WGPUConstantEntry> = desc
            .compilation_options
            .constants
            .iter()
            .map(|(k, v)| native::WGPUConstantEntry {
                nextInChain: std::ptr::null_mut(),
                key: conv::str_to_string_view(k),
                value: *v,
            })
            .collect();
        let mut cache_extras = desc
            .cache
            .and_then(|c| c.as_custom::<CPipelineCache>())
            .map(|c| native::WGPUComputePipelineDescriptorExtras {
                chain: native::WGPUChainedStruct {
                    next: std::ptr::null_mut(),
                    sType: native::WGPUSType_ComputePipelineDescriptorExtras,
                },
                cache: c.ptr,
                zeroInitializeWorkgroupMemory: false as _,
            });
        let c_desc = native::WGPUComputePipelineDescriptor {
            nextInChain: cache_extras
                .as_mut()
                .map(|e| std::ptr::from_mut::<native::WGPUChainedStruct>(&mut e.chain))
                .unwrap_or(std::ptr::null_mut()),
            label: label_sv,
            layout: layout_ptr,
            compute: native::WGPUComputeState {
                nextInChain: std::ptr::null_mut(),
                module: module_ptr,
                entryPoint: ep_sv,
                constantCount: constants.len(),
                constants: if constants.is_empty() {
                    std::ptr::null()
                } else {
                    constants.as_ptr()
                },
            },
        };
        let ptr = unsafe { wgpuDeviceCreateComputePipeline(self.ptr, Some(&c_desc)) };
        DispatchComputePipeline::custom(CComputePipeline { ptr })
    }

    unsafe fn create_pipeline_cache(
        &self,
        desc: &wgpu::PipelineCacheDescriptor<'_>,
    ) -> DispatchPipelineCache {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());
        let c_desc = native::WGPUPipelineCacheDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: label_sv,
            dataSize: desc.data.map(|d| d.len()).unwrap_or(0),
            data: desc.data.map(|d| d.as_ptr()).unwrap_or(std::ptr::null()),
            fallback: desc.fallback as u32,
        };
        let ptr = unsafe { wgpuDeviceCreatePipelineCache(self.ptr, Some(&c_desc)) };
        DispatchPipelineCache::custom(CPipelineCache { ptr })
    }

    fn create_buffer(&self, desc: &wgpu::BufferDescriptor<'_>) -> DispatchBuffer {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());
        let c_desc = native::WGPUBufferDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: label_sv,
            usage: conv::buffer_usage_to_native(desc.usage),
            size: desc.size,
            mappedAtCreation: desc.mapped_at_creation as u32,
        };
        let ptr = unsafe { wgpuDeviceCreateBuffer(self.ptr, Some(&c_desc)) };
        DispatchBuffer::custom(CBuffer { ptr })
    }

    fn create_texture(&self, desc: &wgpu::TextureDescriptor<'_>) -> DispatchTexture {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());
        let size = conv::extent3d_to_native(desc.size);
        let view_formats: Vec<native::WGPUTextureFormat> = desc
            .view_formats
            .iter()
            .map(|&f| conv::texture_format_to_native(f))
            .collect();
        let c_desc = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: label_sv,
            usage: conv::texture_usage_to_native(desc.usage),
            dimension: conv::texture_dimension_to_native(desc.dimension),
            size,
            format: conv::texture_format_to_native(desc.format),
            mipLevelCount: desc.mip_level_count,
            sampleCount: desc.sample_count,
            viewFormatCount: view_formats.len(),
            viewFormats: if view_formats.is_empty() {
                std::ptr::null()
            } else {
                view_formats.as_ptr()
            },
        };
        let ptr = unsafe { wgpuDeviceCreateTexture(self.ptr, Some(&c_desc)) };
        DispatchTexture::custom(CTexture { ptr })
    }

    fn create_external_texture(
        &self,
        _desc: &wgpu::ExternalTextureDescriptor<'_>,
        _planes: &[&wgpu::TextureView],
    ) -> DispatchExternalTexture {
        // wgpu-native has no external texture support.
        unimplemented!("wgpu-native does not support external textures")
    }

    fn create_blas(
        &self,
        _desc: &wgpu::CreateBlasDescriptor<'_>,
        _sizes: wgpu::BlasGeometrySizeDescriptors,
    ) -> (Option<u64>, DispatchBlas) {
        // wgpu-native has no ray tracing support.
        unimplemented!("wgpu-native does not support acceleration structures")
    }

    fn create_tlas(&self, _desc: &wgpu::CreateTlasDescriptor<'_>) -> DispatchTlas {
        // wgpu-native has no ray tracing support.
        unimplemented!("wgpu-native does not support acceleration structures")
    }

    fn create_sampler(&self, desc: &wgpu::SamplerDescriptor<'_>) -> DispatchSampler {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());
        let c_desc = native::WGPUSamplerDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: label_sv,
            addressModeU: conv::address_mode_to_native(desc.address_mode_u),
            addressModeV: conv::address_mode_to_native(desc.address_mode_v),
            addressModeW: conv::address_mode_to_native(desc.address_mode_w),
            magFilter: conv::filter_mode_to_native(desc.mag_filter),
            minFilter: conv::filter_mode_to_native(desc.min_filter),
            mipmapFilter: conv::mipmap_filter_to_native(desc.mipmap_filter),
            lodMinClamp: desc.lod_min_clamp,
            lodMaxClamp: desc.lod_max_clamp,
            compare: desc
                .compare
                .map(conv::compare_function_to_native)
                .unwrap_or(native::WGPUCompareFunction_Undefined),
            maxAnisotropy: desc.anisotropy_clamp,
        };
        let ptr = unsafe { wgpuDeviceCreateSampler(self.ptr, Some(&c_desc)) };
        DispatchSampler::custom(CSampler { ptr })
    }

    fn create_query_set(&self, desc: &wgpu::QuerySetDescriptor<'_>) -> DispatchQuerySet {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());
        let c_desc = native::WGPUQuerySetDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: label_sv,
            type_: conv::query_type_to_native(desc.ty),
            count: desc.count,
        };
        let ptr = unsafe { wgpuDeviceCreateQuerySet(self.ptr, Some(&c_desc)) };
        DispatchQuerySet::custom(CQuerySet { ptr })
    }

    fn create_command_encoder(
        &self,
        desc: &wgpu::CommandEncoderDescriptor<'_>,
    ) -> DispatchCommandEncoder {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());
        let c_desc = native::WGPUCommandEncoderDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: label_sv,
        };
        let ptr = unsafe { wgpuDeviceCreateCommandEncoder(self.ptr, Some(&c_desc)) };
        DispatchCommandEncoder::custom(CCommandEncoder { ptr })
    }

    fn create_render_bundle_encoder(
        &self,
        desc: &wgpu::RenderBundleEncoderDescriptor<'_>,
    ) -> DispatchRenderBundleEncoder {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());
        let color_formats: Vec<native::WGPUTextureFormat> = desc
            .color_formats
            .iter()
            .map(|opt_f| {
                opt_f
                    .map(conv::texture_format_to_native)
                    .unwrap_or(native::WGPUTextureFormat_Undefined)
            })
            .collect();
        let c_desc = native::WGPURenderBundleEncoderDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: label_sv,
            colorFormatCount: color_formats.len(),
            colorFormats: if color_formats.is_empty() {
                std::ptr::null()
            } else {
                color_formats.as_ptr()
            },
            depthStencilFormat: desc
                .depth_stencil
                .map(|ds| conv::texture_format_to_native(ds.format))
                .unwrap_or(native::WGPUTextureFormat_Undefined),
            sampleCount: desc.sample_count,
            depthReadOnly: desc
                .depth_stencil
                .map(|ds| ds.depth_read_only as u32)
                .unwrap_or(0),
            stencilReadOnly: desc
                .depth_stencil
                .map(|ds| ds.stencil_read_only as u32)
                .unwrap_or(0),
        };
        let ptr = unsafe { wgpuDeviceCreateRenderBundleEncoder(self.ptr, Some(&c_desc)) };
        DispatchRenderBundleEncoder::custom(CRenderBundleEncoder { ptr })
    }

    fn set_device_lost_callback(&self, _device_lost_callback: BoxDeviceLostCallback) {
        // wgpu-native sets the device lost callback at creation time; ignore post-creation sets.
    }

    fn on_uncaptured_error(&self, handler: std::sync::Arc<dyn wgpu::UncapturedErrorHandler>) {
        *self.error_handler.lock().unwrap() = Some(handler);
    }

    fn push_error_scope(&self, filter: wgpu::ErrorFilter) -> u32 {
        unsafe { wgpuDevicePushErrorScope(self.ptr, conv::error_filter_to_native(filter)) };
        0
    }

    fn pop_error_scope(&self, _index: u32) -> Pin<Box<dyn PopErrorScopeFuture>> {
        struct Out {
            error: Option<Option<wgpu::Error>>,
        }

        unsafe extern "C" fn cb(
            status: native::WGPUPopErrorScopeStatus,
            type_: native::WGPUErrorType,
            message: native::WGPUStringView,
            userdata1: *mut std::ffi::c_void,
            _userdata2: *mut std::ffi::c_void,
        ) {
            let out = &mut *(userdata1 as *mut Out);
            if status != native::WGPUPopErrorScopeStatus_Success {
                out.error = Some(None);
                return;
            }
            out.error = Some(match type_ {
                native::WGPUErrorType_NoError => None,
                native::WGPUErrorType_Validation => {
                    let msg = unsafe { conv::string_view_to_string(message) };
                    Some(wgpu::Error::Validation {
                        source: Box::new(std::io::Error::other(msg.clone())),
                        description: msg,
                    })
                }
                native::WGPUErrorType_OutOfMemory => {
                    let msg = unsafe { conv::string_view_to_string(message) };
                    Some(wgpu::Error::OutOfMemory {
                        source: Box::new(std::io::Error::other(msg)),
                    })
                }
                _ => {
                    let msg = unsafe { conv::string_view_to_string(message) };
                    Some(wgpu::Error::Internal {
                        source: Box::new(std::io::Error::other(msg.clone())),
                        description: msg,
                    })
                }
            });
        }

        let mut out = Out { error: None };
        let callback_info = native::WGPUPopErrorScopeCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowSpontaneous,
            callback: Some(cb),
            userdata1: std::ptr::addr_of_mut!(out).cast(),
            userdata2: std::ptr::null_mut(),
        };
        unsafe { wgpuDevicePopErrorScope(self.ptr, callback_info) };
        Box::pin(future::ready(out.error.unwrap_or(None)))
    }

    unsafe fn start_graphics_debugger_capture(&self) {
        unsafe { wgpuDeviceStartGraphicsDebuggerCapture(self.ptr) };
    }

    unsafe fn stop_graphics_debugger_capture(&self) {
        unsafe { wgpuDeviceStopGraphicsDebuggerCapture(self.ptr) };
    }

    fn poll(
        &self,
        poll_type: wgpu::wgt::PollType<u64>,
    ) -> Result<wgpu::PollStatus, wgpu::PollError> {
        let (wait, submission_index) = match poll_type {
            wgpu::wgt::PollType::Poll => (false, None),
            wgpu::wgt::PollType::Wait {
                submission_index, ..
            } => (true, submission_index),
        };
        let result = unsafe { wgpuDevicePoll(self.ptr, wait, submission_index.as_ref()) };
        if result {
            Ok(wgpu::PollStatus::QueueEmpty)
        } else {
            Ok(wgpu::PollStatus::Poll)
        }
    }

    fn get_internal_counters(&self) -> wgpu::InternalCounters {
        // wgpu-native has no internal counters query.
        unimplemented!("wgpu-native does not expose internal counters")
    }

    fn generate_allocator_report(&self) -> Option<wgpu::AllocatorReport> {
        None
    }

    fn destroy(&self) {
        unsafe { wgpuDeviceDestroy(self.ptr) };
    }
}

// ── Helper conversion functions ───────────────────────────────────────────────

fn stencil_face_to_native(sf: wgpu::StencilFaceState) -> native::WGPUStencilFaceState {
    native::WGPUStencilFaceState {
        compare: conv::compare_function_to_native(sf.compare),
        failOp: conv::stencil_op_to_native(sf.fail_op),
        depthFailOp: conv::stencil_op_to_native(sf.depth_fail_op),
        passOp: conv::stencil_op_to_native(sf.pass_op),
    }
}

fn blend_component_to_native(bc: wgpu::BlendComponent) -> native::WGPUBlendComponent {
    native::WGPUBlendComponent {
        operation: conv::blend_op_to_native(bc.operation),
        srcFactor: conv::blend_factor_to_native(bc.src_factor),
        dstFactor: conv::blend_factor_to_native(bc.dst_factor),
    }
}

// ── CQueue ────────────────────────────────────────────────────────────────────

pub struct CQueue {
    pub(crate) ptr: native::WGPUQueue,
}

impl std::fmt::Debug for CQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CQueue").field("ptr", &self.ptr).finish()
    }
}

unsafe impl Send for CQueue {}
unsafe impl Sync for CQueue {}

impl Drop for CQueue {
    fn drop(&mut self) {
        unsafe { wgpuQueueRelease(self.ptr) };
    }
}

impl QueueInterface for CQueue {
    fn write_buffer(&self, buffer: &DispatchBuffer, offset: wgpu::BufferAddress, data: &[u8]) {
        let buf_ptr = buffer.as_custom::<CBuffer>().unwrap().ptr;
        unsafe {
            wgpuQueueWriteBuffer(self.ptr, buf_ptr, offset, data.as_ptr().cast(), data.len())
        };
    }

    fn create_staging_buffer(&self, _size: wgpu::BufferSize) -> Option<DispatchQueueWriteBuffer> {
        None
    }

    fn validate_write_buffer(
        &self,
        _buffer: &DispatchBuffer,
        _offset: wgpu::BufferAddress,
        _size: wgpu::BufferSize,
    ) -> Option<()> {
        None
    }

    fn write_staging_buffer(
        &self,
        _buffer: &DispatchBuffer,
        _offset: wgpu::BufferAddress,
        _staging_buffer: &DispatchQueueWriteBuffer,
    ) {
        // wgpu-native has no staging buffer API.
        unimplemented!("wgpu-native does not expose staging buffers")
    }

    fn write_texture(
        &self,
        texture: wgpu::TexelCopyTextureInfo<'_>,
        data: &[u8],
        data_layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        let tex_ptr = texture.texture.as_custom::<CTexture>().unwrap().ptr;
        let c_dst = conv::image_copy_texture_to_native(&texture, tex_ptr);
        let c_layout = native::WGPUTexelCopyBufferLayout {
            offset: data_layout.offset,
            bytesPerRow: data_layout.bytes_per_row.unwrap_or(0),
            rowsPerImage: data_layout.rows_per_image.unwrap_or(0),
        };
        let c_size = conv::extent3d_to_native(size);
        unsafe {
            wgpuQueueWriteTexture(
                self.ptr,
                Some(&c_dst),
                data.as_ptr().cast(),
                data.len(),
                Some(&c_layout),
                Some(&c_size),
            )
        };
    }

    fn submit(&self, command_buffers: &mut dyn Iterator<Item = DispatchCommandBuffer>) -> u64 {
        // Collect first so DispatchCommandBuffers stay alive across wgpuQueueSubmitForIndex.
        // Consuming them inside map() would call wgpuCommandBufferRelease before submit,
        // leaving dangling raw pointers.
        let dispatches: Vec<DispatchCommandBuffer> = command_buffers.collect();
        let ptrs: Vec<native::WGPUCommandBuffer> = dispatches
            .iter()
            .map(|cb| cb.as_custom::<CCommandBuffer>().unwrap().ptr)
            .collect();
        unsafe { wgpuQueueSubmitForIndex(self.ptr, ptrs.len(), ptrs.as_ptr()) }
    }

    fn get_timestamp_period(&self) -> f32 {
        unsafe { wgpuQueueGetTimestampPeriod(self.ptr) }
    }

    fn on_submitted_work_done(&self, callback: BoxSubmittedWorkDoneCallback) {
        struct Out {
            callback: Option<BoxSubmittedWorkDoneCallback>,
        }

        unsafe extern "C" fn cb(
            _status: native::WGPUQueueWorkDoneStatus,
            _message: native::WGPUStringView,
            userdata1: *mut std::ffi::c_void,
            _userdata2: *mut std::ffi::c_void,
        ) {
            let out = unsafe { Box::from_raw(userdata1 as *mut Out) };
            if let Some(cb) = out.callback {
                cb();
            }
        }

        let out = Box::new(Out {
            callback: Some(callback),
        });
        let callback_info = native::WGPUQueueWorkDoneCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowSpontaneous,
            callback: Some(cb),
            userdata1: Box::into_raw(out).cast(),
            userdata2: std::ptr::null_mut(),
        };
        unsafe { wgpuQueueOnSubmittedWorkDone(self.ptr, callback_info) };
    }

    fn compact_blas(&self, _blas: &DispatchBlas) -> (Option<u64>, DispatchBlas) {
        // wgpu-native has no ray tracing support.
        unimplemented!("wgpu-native does not support acceleration structures")
    }

    fn present(&self, detail: &DispatchSurfaceOutputDetail) {
        let surface_ptr = detail
            .as_custom::<crate::surface::CSurfaceOutputDetail>()
            .unwrap()
            .surface_ptr;
        unsafe { wgpuSurfacePresent(surface_ptr) };
    }
}
