use std::future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use wgpu::custom::*;
use wgpu_native::{native, *};

use crate::command::{CCommandEncoder, CRenderBundleEncoder};
use crate::conv;
use crate::resource::*;

// ── CDevice ───────────────────────────────────────────────────────────────────

pub(crate) type ErrorHandler = Arc<Mutex<Option<Arc<dyn wgpu::UncapturedErrorHandler>>>>;
pub(crate) type DeviceLostHandler = Mutex<Option<BoxDeviceLostCallback>>;

pub struct CDevice {
    pub(crate) ptr: native::WGPUDevice,
    pub(crate) info: wgpu::AdapterInfo,
    pub(crate) error_handler: Box<ErrorHandler>,
    pub(crate) device_lost_handler: Box<DeviceLostHandler>,
    /// Tracks the number of active error scopes. Shared with the device's CQueue.
    /// Used to decide whether cross-device submit should panic (no scope) or let
    /// wgpu-native route the error to the active scope instead.
    pub(crate) error_scope_depth: Arc<AtomicU32>,
    /// Set to true by CQueue::drop. Used to detect use-after-queue-drop.
    pub(crate) queue_dropped: Arc<AtomicBool>,
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
        // wgpu-native's WGPUDeviceImpl::drop calls device_poll which can panic via
        // handle_error_fatal if the device is in an error state. Catch that here so it
        // doesn't abort during Drop.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            unsafe { wgpuDeviceRelease(self.ptr) };
        }));
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
        let mut native_limits: native::WGPUNativeLimits = unsafe { std::mem::zeroed() };
        native_limits.chain = native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_NativeLimits,
        };
        let mut limits: native::WGPULimits = unsafe { std::mem::zeroed() };
        limits.nextInChain = std::ptr::from_mut::<native::WGPUChainedStruct>(&mut native_limits.chain);
        unsafe { wgpuDeviceGetLimits(self.ptr, Some(&mut limits)) };
        conv::map_limits(&limits, Some(&native_limits))
    }

    fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.info.clone()
    }

    fn create_shader_module(
        &self,
        desc: wgpu::ShaderModuleDescriptor<'_>,
        shader_bound_checks: wgpu::ShaderRuntimeChecks,
    ) -> DispatchShaderModule {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());

        let mut extras = native::WGPUShaderModuleDescriptorExtras {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_ShaderModuleDescriptorExtras,
            },
            boundsChecks: shader_bound_checks.bounds_checks as _,
            forceLoopBounding: shader_bound_checks.force_loop_bounding as _,
            rayQueryInitializationTracking: shader_bound_checks
                .ray_query_initialization_tracking as _,
            taskShaderDispatchTracking: shader_bound_checks.task_shader_dispatch_tracking as _,
            meshShaderPrimitiveIndicesClamp: shader_bound_checks
                .mesh_shader_primitive_indices_clamp as _,
        };

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
                extras.chain.next =
                    std::ptr::from_mut::<native::WGPUChainedStruct>(&mut wgsl_chain.chain);
                let c_desc = native::WGPUShaderModuleDescriptor {
                    nextInChain: std::ptr::from_mut::<native::WGPUChainedStruct>(&mut extras.chain),
                    label: label_sv,
                };
                let ptr = unsafe { wgpuDeviceCreateShaderModule(self.ptr, Some(&c_desc)) };
                DispatchShaderModule::custom(CShaderModule { ptr })
            }
            #[cfg(feature = "spirv")]
            wgpu::ShaderSource::SpirV(words) => {
                let c_desc = native::WGPUShaderModuleDescriptorPassthrough {
                    label: label_sv,
                    entryPointCount: 0,
                    entryPoints: std::ptr::null(),
                    spirvSize: words.len() as u32,
                    spirv: words.as_ptr(),
                    dxilSize: 0,
                    dxil: std::ptr::null(),
                    hlsl: conv::null_string_view(),
                    metallibSize: 0,
                    metallib: std::ptr::null(),
                    msl: conv::null_string_view(),
                };
                let ptr = unsafe { wgpuDeviceCreateShaderModulePassthrough(self.ptr, Some(&c_desc)) };
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
        // All non-WGSL formats go through WGPUShaderModuleDescriptorPassthrough.
        // Fill in every available format; wgpu-native picks the right one for the platform.
        if desc.spirv.is_none()
            && desc.dxil.is_none()
            && desc.hlsl.is_none()
            && desc.metallib.is_none()
            && desc.msl.is_none()
        {
            unimplemented!(
                "wgpu-native: passthrough descriptor has no supported shader format (GLSL not supported)"
            );
        }
        let native_eps: Vec<native::WGPUPassthroughShaderEntryPoint> = desc
            .entry_points
            .iter()
            .map(|ep| native::WGPUPassthroughShaderEntryPoint {
                name: conv::str_to_string_view(&ep.name),
                workgroupSizeX: ep.workgroup_size.0,
                workgroupSizeY: ep.workgroup_size.1,
                workgroupSizeZ: ep.workgroup_size.2,
            })
            .collect();
        let c_desc = native::WGPUShaderModuleDescriptorPassthrough {
            label: label_sv,
            entryPointCount: native_eps.len(),
            entryPoints: native_eps.as_ptr(),
            spirvSize: desc.spirv.as_ref().map(|s| s.len() as u32).unwrap_or(0),
            spirv: desc.spirv.as_ref().map(|s| s.as_ptr()).unwrap_or(std::ptr::null()),
            dxilSize: desc.dxil.as_ref().map(|d| d.len()).unwrap_or(0),
            dxil: desc.dxil.as_ref().map(|d| d.as_ptr()).unwrap_or(std::ptr::null()),
            hlsl: desc
                .hlsl
                .as_deref()
                .map(conv::str_to_string_view)
                .unwrap_or_else(conv::null_string_view),
            metallibSize: desc.metallib.as_ref().map(|m| m.len()).unwrap_or(0),
            metallib: desc
                .metallib
                .as_ref()
                .map(|m| m.as_ptr())
                .unwrap_or(std::ptr::null()),
            msl: desc
                .msl
                .as_deref()
                .map(conv::str_to_string_view)
                .unwrap_or_else(conv::null_string_view),
        };
        let ptr = unsafe { wgpuDeviceCreateShaderModulePassthrough(self.ptr, Some(&c_desc)) };
        DispatchShaderModule::custom(CShaderModule { ptr })
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

        // AccelerationStructure entries require a chained WGPUAccelerationStructureBindingLayout.
        // We Box each chain struct for a stable heap address, then set nextInChain after the
        // entries Vec is finalized (so Vec reallocation can't invalidate the entry addresses).
        let mut entries: Vec<native::WGPUBindGroupLayoutEntry> =
            Vec::with_capacity(desc.entries.len());
        // (entry_index, Box<chain>) — Box gives stable address even if this Vec reallocates.
        let mut as_chains: Vec<(usize, Box<native::WGPUAccelerationStructureBindingLayout>)> =
            Vec::new();

        for e in desc.entries.iter() {
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
                wgpu::BindingType::AccelerationStructure { vertex_return } => {
                    // entry.nextInChain is set below after the entries Vec is finalized.
                    as_chains.push((
                        entries.len(),
                        Box::new(native::WGPUAccelerationStructureBindingLayout {
                            chain: native::WGPUChainedStruct {
                                next: std::ptr::null_mut(),
                                sType: native::WGPUSType_AccelerationStructureBindingLayout,
                            },
                            vertexReturn: vertex_return as u32,
                        }),
                    ));
                }
                // Unknown binding types: leave the entry zeroed so wgpu-native treats it
                // as an unrecognized entry and generates a validation error. This is safe
                // only for binding types that don't trigger the "invalid entry" panic in
                // wgpu-native's map_bind_group_layout_entry (which fires when none of the
                // standard types match AND no as_layout chain is present).
                // NOTE: any truly unknown variant here will still SIGABRT via the same
                // panic path — they must be added above before shipping.
                _ => {}
            }
            entries.push(entry);
        }

        // Wire AccelerationStructure chain pointers now that entries is finalized.
        // Box<T> guarantees the inner T doesn't move, so the raw pointer stays valid
        // for the duration of the wgpuDeviceCreateBindGroupLayout call below.
        for (idx, chain) in &as_chains {
            entries[*idx].nextInChain =
                chain.as_ref() as *const native::WGPUAccelerationStructureBindingLayout
                    as *mut native::WGPUChainedStruct;
        }

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
        DispatchBindGroupLayout::custom(CBindGroupLayout { ptr, device_ptr: self.ptr })
    }

    fn create_bind_group(&self, desc: &wgpu::BindGroupDescriptor<'_>) -> DispatchBindGroup {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());
        let layout = desc.layout.as_custom::<CBindGroupLayout>().unwrap();
        if !layout.device_ptr.is_null() && layout.device_ptr != self.ptr {
            panic!("bind group layout was created from a different device");
        }
        let layout_ptr = layout.ptr;

        // Boxed extras storage for entries that need WGPUBindGroupEntryExtras.
        // Box gives stable heap addresses even after Vec reallocation.
        struct ExtrasStorage {
            extras: native::WGPUBindGroupEntryExtras,
            _buffers: Vec<native::WGPUBuffer>,
            _samplers: Vec<native::WGPUSampler>,
            _texture_views: Vec<native::WGPUTextureView>,
        }
        let mut extras_by_entry: Vec<(usize, Box<ExtrasStorage>)> = Vec::new();

        let mut entries: Vec<native::WGPUBindGroupEntry> =
            Vec::with_capacity(desc.entries.len());
        for (idx, e) in desc.entries.iter().enumerate() {
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
                wgpu::BindingResource::AccelerationStructure(tlas) => {
                    let tlas_ptr = tlas.as_custom::<CTlas>().unwrap().ptr;
                    extras_by_entry.push((
                        idx,
                        Box::new(ExtrasStorage {
                            extras: native::WGPUBindGroupEntryExtras {
                                chain: native::WGPUChainedStruct {
                                    next: std::ptr::null_mut(),
                                    sType: native::WGPUSType_BindGroupEntryExtras,
                                },
                                buffers: std::ptr::null(),
                                bufferCount: 0,
                                samplers: std::ptr::null(),
                                samplerCount: 0,
                                textureViews: std::ptr::null(),
                                textureViewCount: 0,
                                tlas: tlas_ptr,
                            },
                            _buffers: Vec::new(),
                            _samplers: Vec::new(),
                            _texture_views: Vec::new(),
                        }),
                    ));
                }
                wgpu::BindingResource::BufferArray(arr) => {
                    let bufs: Vec<native::WGPUBuffer> = arr
                        .iter()
                        .map(|bb| bb.buffer.as_custom::<CBuffer>().unwrap().ptr)
                        .collect();
                    let buf_ptr = if bufs.is_empty() { std::ptr::null() } else { bufs.as_ptr() };
                    let buf_len = bufs.len();
                    extras_by_entry.push((
                        idx,
                        Box::new(ExtrasStorage {
                            extras: native::WGPUBindGroupEntryExtras {
                                chain: native::WGPUChainedStruct {
                                    next: std::ptr::null_mut(),
                                    sType: native::WGPUSType_BindGroupEntryExtras,
                                },
                                buffers: buf_ptr,
                                bufferCount: buf_len,
                                samplers: std::ptr::null(),
                                samplerCount: 0,
                                textureViews: std::ptr::null(),
                                textureViewCount: 0,
                                tlas: std::ptr::null_mut(),
                            },
                            _buffers: bufs,
                            _samplers: Vec::new(),
                            _texture_views: Vec::new(),
                        }),
                    ));
                }
                wgpu::BindingResource::SamplerArray(arr) => {
                    let samps: Vec<native::WGPUSampler> = arr
                        .iter()
                        .map(|s| s.as_custom::<CSampler>().unwrap().ptr)
                        .collect();
                    let samp_ptr = if samps.is_empty() { std::ptr::null() } else { samps.as_ptr() };
                    let samp_len = samps.len();
                    extras_by_entry.push((
                        idx,
                        Box::new(ExtrasStorage {
                            extras: native::WGPUBindGroupEntryExtras {
                                chain: native::WGPUChainedStruct {
                                    next: std::ptr::null_mut(),
                                    sType: native::WGPUSType_BindGroupEntryExtras,
                                },
                                buffers: std::ptr::null(),
                                bufferCount: 0,
                                samplers: samp_ptr,
                                samplerCount: samp_len,
                                textureViews: std::ptr::null(),
                                textureViewCount: 0,
                                tlas: std::ptr::null_mut(),
                            },
                            _buffers: Vec::new(),
                            _samplers: samps,
                            _texture_views: Vec::new(),
                        }),
                    ));
                }
                wgpu::BindingResource::TextureViewArray(arr) => {
                    let tvs: Vec<native::WGPUTextureView> = arr
                        .iter()
                        .map(|tv| tv.as_custom::<CTextureView>().unwrap().ptr)
                        .collect();
                    let tv_ptr = if tvs.is_empty() { std::ptr::null() } else { tvs.as_ptr() };
                    let tv_len = tvs.len();
                    extras_by_entry.push((
                        idx,
                        Box::new(ExtrasStorage {
                            extras: native::WGPUBindGroupEntryExtras {
                                chain: native::WGPUChainedStruct {
                                    next: std::ptr::null_mut(),
                                    sType: native::WGPUSType_BindGroupEntryExtras,
                                },
                                buffers: std::ptr::null(),
                                bufferCount: 0,
                                samplers: std::ptr::null(),
                                samplerCount: 0,
                                textureViews: tv_ptr,
                                textureViewCount: tv_len,
                                tlas: std::ptr::null_mut(),
                            },
                            _buffers: Vec::new(),
                            _samplers: Vec::new(),
                            _texture_views: tvs,
                        }),
                    ));
                }
                // AccelerationStructureArray and ExternalTexture are not supported.
                _ => unimplemented!("wgpu-native does not support this binding resource type"),
            }
            entries.push(entry);
        }

        // Wire chain pointers now that entries Vec is finalized (no more reallocation).
        // Box<ExtrasStorage> guarantees the extras struct and its backing Vecs don't move,
        // so the raw pointers into _buffers/_samplers/_texture_views remain valid.
        for (idx, storage) in &extras_by_entry {
            entries[*idx].nextInChain =
                &storage.extras.chain as *const native::WGPUChainedStruct as *mut _;
        }

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

        let render_cache_ptr = desc
            .cache
            .and_then(|c| c.as_custom::<CPipelineCache>())
            .map(|c| c.ptr)
            .unwrap_or(std::ptr::null_mut());
        let mut render_extras = native::WGPURenderPipelineDescriptorExtras {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_RenderPipelineDescriptorExtras,
            },
            cache: render_cache_ptr,
            multiviewMask: desc.multiview_mask.map_or(0, |v| v.get()),
            zeroInitializeWorkgroupMemory: desc
                .vertex
                .compilation_options
                .zero_initialize_workgroup_memory as _,
        };
        let c_desc = native::WGPURenderPipelineDescriptor {
            nextInChain: std::ptr::from_mut::<native::WGPUChainedStruct>(&mut render_extras.chain),
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

        let mesh_cache_ptr = desc
            .cache
            .and_then(|c| c.as_custom::<CPipelineCache>())
            .map(|c| c.ptr)
            .unwrap_or(std::ptr::null_mut());
        let mut mesh_extras = native::WGPUMeshPipelineDescriptorExtras {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_MeshPipelineDescriptorExtras,
            },
            cache: mesh_cache_ptr,
            multiviewMask: desc.multiview.map_or(0, |v| v.get()),
            zeroInitializeWorkgroupMemory: desc
                .mesh
                .compilation_options
                .zero_initialize_workgroup_memory as _,
        };

        let c_desc = native::WGPUMeshPipelineDescriptor {
            nextInChain: std::ptr::from_mut::<native::WGPUChainedStruct>(&mut mesh_extras.chain),
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
        let cache_ptr = desc
            .cache
            .and_then(|c| c.as_custom::<CPipelineCache>())
            .map(|c| c.ptr)
            .unwrap_or(std::ptr::null_mut());
        let mut extras = native::WGPUComputePipelineDescriptorExtras {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_ComputePipelineDescriptorExtras,
            },
            cache: cache_ptr,
            zeroInitializeWorkgroupMemory: desc
                .compilation_options
                .zero_initialize_workgroup_memory as _,
        };
        let c_desc = native::WGPUComputePipelineDescriptor {
            nextInChain: std::ptr::from_mut::<native::WGPUChainedStruct>(&mut extras.chain),
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
        // Any bits not in KNOWN_BUFFER_USAGE_BITS cannot be represented in the C API.
        // Pass usage=0 so wgpu-core generates a validation error (empty usage is always
        // invalid) captured by any active error scope — matching expected wgpu semantics.
        let native_usage = if (desc.usage.bits() & !conv::KNOWN_BUFFER_USAGE_BITS.bits()) == 0 {
            conv::buffer_usage_to_native(desc.usage)
        } else {
            0
        };
        let c_desc = native::WGPUBufferDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: label_sv,
            usage: native_usage,
            size: desc.size,
            mappedAtCreation: desc.mapped_at_creation as u32,
        };
        let ptr = unsafe { wgpuDeviceCreateBuffer(self.ptr, Some(&c_desc)) };
        let buf = if desc.mapped_at_creation {
            CBuffer::new_mapped_at_creation(ptr)
        } else {
            CBuffer::new(ptr)
        };
        DispatchBuffer::custom(buf)
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
        desc: &wgpu::CreateBlasDescriptor<'_>,
        sizes: wgpu::BlasGeometrySizeDescriptors,
    ) -> (Option<u64>, DispatchBlas) {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());
        let c_desc = native::WGPUBlasDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: label_sv,
            flags: conv::acceleration_structure_flags_to_native(desc.flags),
            updateMode: conv::acceleration_structure_update_mode_to_native(desc.update_mode),
        };
        let ptr = match sizes {
            wgpu::BlasGeometrySizeDescriptors::Triangles { ref descriptors } => {
                let c_tris: Vec<native::WGPUBlasTriangleGeometrySizeDescriptor> = descriptors
                    .iter()
                    .map(|d| native::WGPUBlasTriangleGeometrySizeDescriptor {
                        vertexFormat: conv::vertex_format_to_native(d.vertex_format),
                        vertexCount: d.vertex_count,
                        indexFormat: d
                            .index_format
                            .map(conv::index_format_to_native)
                            .unwrap_or(native::WGPUIndexFormat_Undefined),
                        indexCount: d.index_count.unwrap_or(0),
                        flags: conv::acceleration_structure_geometry_flags_to_native(d.flags),
                    })
                    .collect();
                let c_sizes = native::WGPUBlasSizeDescriptors {
                    kind: native::WGPUBlasGeometryKind_Triangles,
                    triangleDescriptors: if c_tris.is_empty() {
                        std::ptr::null()
                    } else {
                        c_tris.as_ptr()
                    },
                    triangleDescriptorCount: c_tris.len(),
                    aabbDescriptors: std::ptr::null(),
                    aabbDescriptorCount: 0,
                };
                unsafe { wgpuDeviceCreateBlas(self.ptr, Some(&c_desc), c_sizes) }
            }
            wgpu::BlasGeometrySizeDescriptors::AABBs { ref descriptors } => {
                let c_aabbs: Vec<native::WGPUBlasAABBGeometrySizeDescriptor> = descriptors
                    .iter()
                    .map(|d| native::WGPUBlasAABBGeometrySizeDescriptor {
                        primitiveCount: d.primitive_count,
                        flags: conv::acceleration_structure_geometry_flags_to_native(d.flags),
                    })
                    .collect();
                let c_sizes = native::WGPUBlasSizeDescriptors {
                    kind: native::WGPUBlasGeometryKind_AABBs,
                    triangleDescriptors: std::ptr::null(),
                    triangleDescriptorCount: 0,
                    aabbDescriptors: if c_aabbs.is_empty() {
                        std::ptr::null()
                    } else {
                        c_aabbs.as_ptr()
                    },
                    aabbDescriptorCount: c_aabbs.len(),
                };
                unsafe { wgpuDeviceCreateBlas(self.ptr, Some(&c_desc), c_sizes) }
            }
        };
        let raw_handle = unsafe { wgpuBlasGetHandle(ptr) };
        let handle = if raw_handle == 0 { None } else { Some(raw_handle) };
        (handle, DispatchBlas::custom(CBlas { ptr }))
    }

    fn create_tlas(&self, desc: &wgpu::CreateTlasDescriptor<'_>) -> DispatchTlas {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());
        let c_desc = native::WGPUTlasDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: label_sv,
            maxInstances: desc.max_instances,
            flags: conv::acceleration_structure_flags_to_native(desc.flags),
            updateMode: conv::acceleration_structure_update_mode_to_native(desc.update_mode),
        };
        let ptr = unsafe { wgpuDeviceCreateTlas(self.ptr, Some(&c_desc)) };
        DispatchTlas::custom(CTlas { ptr })
    }

    fn create_sampler(&self, desc: &wgpu::SamplerDescriptor<'_>) -> DispatchSampler {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());
        let mut extras = desc.border_color.map(|bc| native::WGPUSamplerDescriptorExtras {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_SamplerDescriptorExtras,
            },
            borderColor: conv::border_color_to_native(bc),
        });
        let c_desc = native::WGPUSamplerDescriptor {
            nextInChain: extras
                .as_mut()
                .map(|e| std::ptr::from_mut::<native::WGPUChainedStruct>(&mut e.chain))
                .unwrap_or(std::ptr::null_mut()),
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

        // PipelineStatistics queries require WGPUQuerySetDescriptorExtras listing the
        // specific statistics to collect.
        let ps_names: Vec<native::WGPUPipelineStatisticName>;
        let mut ps_extras_opt: Option<native::WGPUQuerySetDescriptorExtras> = None;

        if let wgpu::QueryType::PipelineStatistics(flags) = desc.ty {
            ps_names = conv::pipeline_statistics_to_native(flags);
            ps_extras_opt = Some(native::WGPUQuerySetDescriptorExtras {
                chain: native::WGPUChainedStruct {
                    next: std::ptr::null_mut(),
                    sType: native::WGPUSType_QuerySetDescriptorExtras,
                },
                pipelineStatistics: if ps_names.is_empty() {
                    std::ptr::null()
                } else {
                    ps_names.as_ptr()
                },
                pipelineStatisticCount: ps_names.len(),
            });
        } else {
            ps_names = Vec::new();
        }

        let c_desc = native::WGPUQuerySetDescriptor {
            nextInChain: ps_extras_opt
                .as_mut()
                .map(|e| std::ptr::from_mut::<native::WGPUChainedStruct>(&mut e.chain))
                .unwrap_or(std::ptr::null_mut()),
            label: label_sv,
            type_: conv::query_type_to_native(desc.ty),
            count: desc.count,
        };
        let ptr = unsafe { wgpuDeviceCreateQuerySet(self.ptr, Some(&c_desc)) };
        let _ = ps_names; // ensure Vec stays alive until after the call
        DispatchQuerySet::custom(CQuerySet { ptr })
    }

    fn create_command_encoder(
        &self,
        desc: &wgpu::CommandEncoderDescriptor<'_>,
    ) -> DispatchCommandEncoder {
        if self.queue_dropped.load(Ordering::Acquire) {
            panic!("device's queue has been dropped");
        }
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
        DispatchCommandEncoder::custom(CCommandEncoder { ptr, device_ptr: self.ptr })
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

    fn set_device_lost_callback(&self, device_lost_callback: BoxDeviceLostCallback) {
        *self.device_lost_handler.lock().unwrap() = Some(device_lost_callback);
    }

    fn on_uncaptured_error(&self, handler: std::sync::Arc<dyn wgpu::UncapturedErrorHandler>) {
        *self.error_handler.lock().unwrap() = Some(handler);
    }

    fn push_error_scope(&self, filter: wgpu::ErrorFilter) -> u32 {
        self.error_scope_depth.fetch_add(1, Ordering::Relaxed);
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
        self.error_scope_depth.fetch_sub(1, Ordering::Relaxed);
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
        // Re-raise any panic that occurred inside a map callback during polling.
        crate::resume_callback_panic();
        if result {
            Ok(wgpu::PollStatus::QueueEmpty)
        } else {
            Ok(wgpu::PollStatus::Poll)
        }
    }

    fn get_internal_counters(&self) -> wgpu::InternalCounters {
        let c = unsafe { wgpuDeviceGetInternalCounters(self.ptr) };
        let hal = c.hal;
        let make = |v: i64| {
            let c = wgpu::wgt::InternalCounter::new();
            c.set(v as isize);
            c
        };
        wgpu::InternalCounters {
            core: wgpu::wgt::CoreCounters {},
            hal: wgpu::wgt::HalCounters {
                buffers: make(hal.buffers),
                textures: make(hal.textures),
                texture_views: make(hal.textureViews),
                bind_groups: make(hal.bindGroups),
                bind_group_layouts: make(hal.bindGroupLayouts),
                render_pipelines: make(hal.renderPipelines),
                compute_pipelines: make(hal.computePipelines),
                pipeline_layouts: make(hal.pipelineLayouts),
                samplers: make(hal.samplers),
                command_encoders: make(hal.commandEncoders),
                shader_modules: make(hal.shaderModules),
                query_sets: make(hal.querySets),
                fences: make(hal.fences),
                buffer_memory: make(hal.bufferMemory),
                texture_memory: make(hal.textureMemory),
                acceleration_structure_memory: make(hal.accelerationStructureMemory),
                memory_allocations: make(hal.memoryAllocations),
            },
        }
    }

    fn generate_allocator_report(&self) -> Option<wgpu::AllocatorReport> {
        None
    }

    fn destroy(&self) {
        unsafe { wgpuDeviceDestroy(self.ptr) };
        // wgpu-native does not wire WGPUDeviceLostCallbackInfo to wgpu-core's
        // device_lost_closure, so the callback never fires automatically after
        // wgpuDeviceDestroy. Call it directly here, matching the expected semantics.
        if let Some(callback) = self.device_lost_handler.lock().unwrap().take() {
            crate::catch_callback_panic(|| {
                callback(wgpu::DeviceLostReason::Destroyed, String::new())
            });
            crate::resume_callback_panic();
        }
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
    /// Device this queue belongs to. Used to detect cross-device command buffer submission.
    pub(crate) device_ptr: native::WGPUDevice,
    /// Shared with the owning CDevice. Mirrors the active error scope count.
    pub(crate) error_scope_depth: Arc<AtomicU32>,
    /// Shared with the owning CDevice. Set to true when this queue is dropped.
    pub(crate) queue_dropped: Arc<AtomicBool>,
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
        self.queue_dropped.store(true, Ordering::Release);
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

    fn create_staging_buffer(&self, size: wgpu::BufferSize) -> Option<DispatchQueueWriteBuffer> {
        Some(DispatchQueueWriteBuffer::custom(CQueueWriteBuffer {
            data: vec![0u8; size.get() as usize],
        }))
    }

    fn validate_write_buffer(
        &self,
        _buffer: &DispatchBuffer,
        _offset: wgpu::BufferAddress,
        _size: wgpu::BufferSize,
    ) -> Option<()> {
        Some(())
    }

    fn write_staging_buffer(
        &self,
        buffer: &DispatchBuffer,
        offset: wgpu::BufferAddress,
        staging_buffer: &DispatchQueueWriteBuffer,
    ) {
        let buf_ptr = buffer.as_custom::<CBuffer>().unwrap().ptr;
        let wb = staging_buffer.as_custom::<CQueueWriteBuffer>().unwrap();
        unsafe {
            wgpuQueueWriteBuffer(
                self.ptr,
                buf_ptr,
                offset,
                wb.data.as_ptr().cast(),
                wb.data.len(),
            )
        };
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
            bytesPerRow: data_layout.bytes_per_row.unwrap_or(native::WGPU_COPY_STRIDE_UNDEFINED),
            rowsPerImage: data_layout.rows_per_image.unwrap_or(native::WGPU_COPY_STRIDE_UNDEFINED),
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
            .map(|cb| {
                let cb = cb.as_custom::<CCommandBuffer>().unwrap();
                if cb.device_ptr != self.device_ptr
                    && self.error_scope_depth.load(Ordering::Relaxed) == 0
                {
                    // No active error scope: cross-device submit is a fatal error — panic so
                    // the caller's catch_unwind (if any) can observe it, matching wgpu-core's
                    // behavior where such errors are fatal when uncaptured.
                    panic!("Command buffer was created on a different device than the queue it's being submitted to");
                }
                cb.ptr
            })
            .collect();
        // wgpu-native's wgpuQueueSubmitForIndex calls handle_error_fatal (which panics)
        // for fatal validation errors. Catch those panics here so they re-raise cleanly
        // in Rust context instead of aborting due to unwinding through extern "C" frames.
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                wgpuQueueSubmitForIndex(self.ptr, ptrs.len(), ptrs.as_ptr())
            }));
        match result {
            Ok(idx) => idx,
            Err(payload) => std::panic::resume_unwind(payload),
        }
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

    fn compact_blas(&self, blas: &DispatchBlas) -> (Option<u64>, DispatchBlas) {
        let old_ptr = blas.as_custom::<CBlas>().unwrap().ptr;
        let new_ptr = unsafe { wgpuQueueCompactBlas(self.ptr, old_ptr) };
        let raw_handle = unsafe { wgpuBlasGetHandle(new_ptr) };
        let handle = if raw_handle == 0 { None } else { Some(raw_handle) };
        (handle, DispatchBlas::custom(CBlas { ptr: new_ptr }))
    }

    fn present(&self, detail: &DispatchSurfaceOutputDetail) {
        let surface_ptr = detail
            .as_custom::<crate::surface::CSurfaceOutputDetail>()
            .unwrap()
            .surface_ptr;
        unsafe { wgpuSurfacePresent(surface_ptr) };
    }
}
